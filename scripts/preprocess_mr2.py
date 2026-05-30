"""Convert the MR2 dataset into the JSONL format expected by ``src.data``.

MR2 layout (after unzipping ``MR2.zip``)::

    <mr2_root>/queries_dataset_merge/
        dataset_items_{train,val,test}.json   # claim metadata
        {train,val,test}/
            img/<id>.jpg                       # claim (query) image
            img_html_news/<id>/                # direct image-search results
                <eid>.jpg, <eid>.html
                direct_annotation.json
            inverse_search/<id>/               # reverse image-search results
                <eid>.html
                inverse_annotation.json

For each claim we collect the union of evidences from ``direct_annotation.json``
and ``inverse_annotation.json``. Direct evidences carry both an image and a
page; inverse evidences are text-only. We flatten the heterogeneous
``caption`` field (sometimes a dict like ``{"alt_node": "..."}``) into a
single string and concatenate ``title + snippet + caption``.

Each output JSONL line conforms to README §13::

    {claim_id, claim, label, positive_evidence: [...], negative_evidence: []}

Negatives are left empty: contrastive training relies on in-batch negatives.
The full per-split evidence pool is reassembled at evaluation time by
``EvidenceCorpus``.

Usage::

    python scripts/preprocess_mr2.py \\
        --mr2_root /mnt/data/yangjun/data/mr2/queries_dataset_merge \\
        --out_dir data/processed/mr2
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten_caption(cap: Any) -> str:
    if not cap:
        return ""
    if isinstance(cap, str):
        return cap.strip()
    if isinstance(cap, dict):
        parts = []
        for key in ("title_node", "alt_node", "text"):
            v = cap.get(key)
            if v:
                parts.append(str(v).strip())
        if not parts:
            parts = [str(v).strip() for v in cap.values() if v]
        return " ".join(parts).strip()
    if isinstance(cap, list):
        return " ".join(_flatten_caption(c) for c in cap if c).strip()
    return str(cap).strip()


def _safe_abs(base: str, rel: Optional[str], extra_bases: Optional[List[str]] = None) -> Optional[str]:
    """Resolve a relative path (possibly starting with ``./``) under ``base``.

    Tries three lookups in order:
      1. ``base / rel`` as written.
      2. ``base / basename(rel)`` (MR2 sometimes records `./img_html_news/X.jpg`
         even though X.jpg lives directly in the per-sample folder).
      3. for each path in ``extra_bases``: same two lookups.

    Returns the absolute path of the first hit, else ``None``.
    """
    if not rel:
        return None
    rel = rel.strip()
    if not rel:
        return None
    if rel.startswith("./"):
        rel = rel[2:]
    if os.path.isabs(rel):
        return rel if os.path.isfile(rel) else None
    candidates = [
        os.path.join(base, rel),
        os.path.join(base, os.path.basename(rel)),
    ]
    if extra_bases:
        for b in extra_bases:
            candidates.append(os.path.join(b, rel))
            candidates.append(os.path.join(b, os.path.basename(rel)))
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def _build_evidence_text(*chunks: Any) -> str:
    parts: List[str] = []
    seen = set()
    for c in chunks:
        s = _flatten_caption(c) if not isinstance(c, str) else c.strip()
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        parts.append(s)
    return " ".join(parts).strip()


# ---------------------------------------------------------------------------
# Evidence collection
# ---------------------------------------------------------------------------

def _collect_direct_evidences(
    direct_dir_abs: str,
    split_root_abs: str,
    direct_anno: Dict[str, Any],
    claim_id: str,
) -> List[Dict[str, Any]]:
    """Each direct-search result has a page (html) and a paired image.

    ``image_path`` / ``html_path`` in the annotation JSON are written like
    ``./img_html_news/<eid>.jpg`` and resolve relative to the **split root**
    (e.g. ``<mr2_root>/val``), not the per-claim folder.

    Sections (per the MR2 dump):
      - ``images_with_captions``
      - ``images_with_no_captions``
      - ``images_with_caption_matched_tags``
    """
    out: List[Dict[str, Any]] = []
    for section in (
        "images_with_captions",
        "images_with_no_captions",
        "images_with_caption_matched_tags",
    ):
        for idx, ev in enumerate(direct_anno.get(section, []) or []):
            img_abs = _safe_abs(
                direct_dir_abs,
                ev.get("image_path"),
                extra_bases=[split_root_abs],
            )
            text = _build_evidence_text(
                ev.get("page_title"),
                ev.get("snippet"),
                ev.get("caption"),
            )
            if not text and not img_abs:
                continue
            evi_id = f"{claim_id}__d__{section}__{idx}"
            out.append({
                "evidence_id": evi_id,
                "image_path": img_abs or "",
                "text": text,
                "caption": _flatten_caption(ev.get("caption")),
                "title": ev.get("page_title") or "",
                "snippet": ev.get("snippet") or "",
                "ocr": "",
                "source": f"mr2_direct/{section}",
                "url": ev.get("page_link") or "",
            })
    return out


def _collect_inverse_evidences(
    inv_dir_abs: str,
    inv_anno: Dict[str, Any],
    claim_id: str,
) -> List[Dict[str, Any]]:
    """Inverse-search results are text-only pages.

    Useful structured signals:
      - ``best_guess_lbl``   : Google's best guess for the image as a whole
      - ``entities``         : entities Google extracted from the image
      - per-page entries store ``title`` (+ optional ``caption``).
    """
    out: List[Dict[str, Any]] = []

    # entity/best-guess summary — attach as a synthetic evidence so it ends up
    # in the corpus and can be retrieved.
    entities = inv_anno.get("entities") or []
    best_guess = inv_anno.get("best_guess_lbl") or []
    summary = _build_evidence_text(
        "best guess: " + ", ".join(best_guess) if best_guess else "",
        "entities: " + ", ".join(entities) if entities else "",
    )
    if summary:
        out.append({
            "evidence_id": f"{claim_id}__i__summary",
            "image_path": "",
            "text": summary,
            "caption": "",
            "title": "",
            "snippet": summary,
            "ocr": "",
            "source": "mr2_inverse/summary",
            "url": "",
        })

    for section in (
        "all_fully_matched_captions",
        "all_partially_matched_captions",
        "fully_matched_no_text",
        "partially_matched_no_text",
    ):
        for idx, ev in enumerate(inv_anno.get(section, []) or []):
            text = _build_evidence_text(
                ev.get("title"),
                ev.get("caption"),
            )
            if not text:
                continue
            evi_id = f"{claim_id}__i__{section}__{idx}"
            out.append({
                "evidence_id": evi_id,
                "image_path": "",
                "text": text,
                "caption": _flatten_caption(ev.get("caption")),
                "title": ev.get("title") or "",
                "snippet": "",
                "ocr": "",
                "source": f"mr2_inverse/{section}",
                "url": ev.get("page_link") or "",
            })
    return out


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert_split(
    mr2_root: str,
    split: str,
    out_path: str,
    keep_empty_caption: bool,
    min_evidences: int,
    include_query_image: bool,
) -> Tuple[int, int]:
    meta_path = os.path.join(mr2_root, f"dataset_items_{split}.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    n_written = 0
    n_evidences = 0
    skipped_empty_caption = 0
    skipped_no_evidence = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for sid, item in meta.items():
            claim_id = f"{split}_{sid}"
            claim_text = (item.get("caption") or "").strip()
            if not claim_text:
                if not keep_empty_caption:
                    skipped_empty_caption += 1
                    continue
                claim_text = "[no caption]"

            label = item.get("label", -1)

            direct_path = item.get("direct_path") or ""
            inv_path = item.get("inv_path") or ""
            direct_abs = os.path.join(mr2_root, direct_path) if direct_path else None
            inv_abs = os.path.join(mr2_root, inv_path) if inv_path else None
            split_root_abs = os.path.join(mr2_root, split)

            positives: List[Dict[str, Any]] = []
            if direct_abs and os.path.isdir(direct_abs):
                da_file = os.path.join(direct_abs, "direct_annotation.json")
                if os.path.isfile(da_file):
                    with open(da_file, "r", encoding="utf-8") as f:
                        da = json.load(f)
                    positives.extend(
                        _collect_direct_evidences(
                            direct_abs, split_root_abs, da, claim_id,
                        )
                    )
            if inv_abs and os.path.isdir(inv_abs):
                ia_file = os.path.join(inv_abs, "inverse_annotation.json")
                if os.path.isfile(ia_file):
                    with open(ia_file, "r", encoding="utf-8") as f:
                        ia = json.load(f)
                    positives.extend(
                        _collect_inverse_evidences(inv_abs, ia, claim_id)
                    )

            if len(positives) < min_evidences:
                skipped_no_evidence += 1
                continue

            record: Dict[str, Any] = {
                "claim_id": claim_id,
                "claim": claim_text,
                "label": label,
                "positive_evidence": positives,
                "negative_evidence": [],
            }
            if include_query_image:
                query_img_abs = _safe_abs(mr2_root, item.get("image_path"))
                if query_img_abs:
                    record["claim_image_path"] = query_img_abs

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1
            n_evidences += len(positives)

    print(
        f"[{split}] wrote {n_written} claims, {n_evidences} evidences -> {out_path}\n"
        f"         skipped {skipped_empty_caption} (empty caption), "
        f"{skipped_no_evidence} (no evidence)"
    )
    return n_written, n_evidences


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--mr2_root",
        required=True,
        help="path to <mr2>/queries_dataset_merge",
    )
    parser.add_argument(
        "--out_dir",
        default="data/processed/mr2",
        help="directory for output JSONL files",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val", "test"],
    )
    parser.add_argument(
        "--keep_empty_caption", action="store_true",
        help="if set, claims with no caption are kept with placeholder text",
    )
    parser.add_argument(
        "--min_evidences", type=int, default=1,
        help="minimum #evidences per claim to keep",
    )
    parser.add_argument(
        "--include_query_image", action="store_true",
        help="export `claim_image_path` for future multimodal-claim extensions",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for split in args.splits:
        out_path = os.path.join(args.out_dir, f"mr2_{split}.jsonl")
        convert_split(
            mr2_root=args.mr2_root,
            split=split,
            out_path=out_path,
            keep_empty_caption=args.keep_empty_caption,
            min_evidences=args.min_evidences,
            include_query_image=args.include_query_image,
        )


if __name__ == "__main__":
    main()
