"""Convert MOCHEG into the EACER JSONL format.

MOCHEG raw layout (`/mnt/data/public/Mocheg/mocheg/`):
  Corpus3.csv                                 # claim-level document corpus
  {train,val,test}/Corpus2.csv                # per-row evidence rows for each claim
  {train,val,test}/images/                    # claim & evidence images, named "{claim_id}-..."
  {train,val,test}/img_evidence_qrels.csv     # image relevance per claim
  {train,val,test}/text_evidence_qrels_*.csv  # text relevance per claim

Output (per split):
  data/processed/mocheg/mocheg_{train,val,test}.jsonl

Each output line:
  {claim_id, claim, label, positive_evidence: [...], negative_evidence: []}

We follow MR2's schema so the existing dataset/collator/retriever code works
without changes:
  - positive_evidence[i].text       = the curated `Evidence` text
  - positive_evidence[i].image_path = best-effort matched image under
                                      {split}/images/ that starts with the
                                      claim_id and is marked RELEVANT in
                                      img_qrels. Falls back to round-robin
                                      assignment among any claim_id-* images.
                                      Empty string if none.
  - claim_image_path                = first image whose name starts with
                                      the claim_id (so Step-2 claim-image
                                      module has something to use).

Negatives are intentionally left empty — contrastive training uses in-batch
+ optional hard-negative mining (separate script).
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List

import pandas as pd

DEFAULT_ROOT = "/mnt/data/public/Mocheg/mocheg"


def _image_index(split_dir: str) -> Dict[str, List[str]]:
    """Group images by leading numeric claim_id (e.g. "75-proof-...jpg" -> "75")."""
    by_cid: Dict[str, List[str]] = defaultdict(list)
    img_dir = os.path.join(split_dir, "images")
    if not os.path.isdir(img_dir):
        return by_cid
    for fname in os.listdir(img_dir):
        if not fname.endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue
        head = fname.split("-", 1)[0]
        if head.isdigit():
            by_cid[head].append(fname)
    for v in by_cid.values():
        v.sort()
    return by_cid


def _relevant_imgs_per_claim(split_dir: str) -> Dict[str, List[str]]:
    """From img_evidence_qrels.csv, return {claim_id: [relevant image filenames]}.

    The qrels' `DOCUMENT#` column carries the candidate image filename.
    """
    qrels_path = os.path.join(split_dir, "img_evidence_qrels.csv")
    if not os.path.isfile(qrels_path):
        return {}
    df = pd.read_csv(qrels_path)
    df = df[df["RELEVANCY"] == 1]
    out: Dict[str, List[str]] = defaultdict(list)
    for _, row in df.iterrows():
        out[str(row["TOPIC"])].append(str(row["DOCUMENT#"]))
    # Stable order, dedup
    return {k: sorted(set(v)) for k, v in out.items()}


def _normalise_evidence(ev_text: Any, max_chars: int = 4096) -> str:
    if ev_text is None or (isinstance(ev_text, float) and pd.isna(ev_text)):
        return ""
    s = str(ev_text).strip()
    # Strip stray HTML left over in some Snopes evidence cells.
    s = s.replace("<p>", " ").replace("</p>", " ").replace("\n", " ")
    s = " ".join(s.split())
    return s[:max_chars]


def build_split(
    root: str,
    split: str,
    out_path: str,
) -> int:
    split_dir = os.path.join(root, split)
    c2_path = os.path.join(split_dir, "Corpus2.csv")
    if not os.path.isfile(c2_path):
        raise FileNotFoundError(c2_path)

    print(f"[{split}] reading {c2_path}", flush=True)
    df = pd.read_csv(c2_path)

    img_index = _image_index(split_dir)
    rel_imgs = _relevant_imgs_per_claim(split_dir)
    print(f"[{split}] images indexed for {len(img_index)} claims; "
          f"img_qrels (RELEVANCY=1) cover {len(rel_imgs)} claims", flush=True)

    img_abs_root = os.path.join(split_dir, "images")

    # Group rows by claim_id
    grouped: Dict[str, List[Any]] = defaultdict(list)
    claim_meta: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        cid = str(row["claim_id"])
        grouped[cid].append(row)
        if cid not in claim_meta:
            claim_meta[cid] = {
                "claim": str(row.get("Claim", "")).strip(),
                "label": str(row.get("cleaned_truthfulness") or row.get("Truthfulness") or "UNK"),
            }

    n_written = 0
    n_skipped_no_claim = 0
    n_skipped_no_evidence = 0
    n_with_image = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for cid, rows in grouped.items():
            meta = claim_meta[cid]
            claim_text = meta["claim"]
            if not claim_text:
                n_skipped_no_claim += 1
                continue

            relevant_imgs = list(rel_imgs.get(cid, []))
            all_imgs = list(img_index.get(cid, []))
            # Prefer qrels-relevant; fall back to any claim image
            img_queue: List[str] = []
            seen = set()
            for x in (relevant_imgs + all_imgs):
                if x in seen or not os.path.isfile(os.path.join(img_abs_root, x)):
                    continue
                seen.add(x)
                img_queue.append(x)
            claim_img = (
                os.path.join(img_abs_root, img_queue[0]) if img_queue else ""
            )

            positives: List[Dict[str, Any]] = []
            for i, row in enumerate(rows):
                text = _normalise_evidence(row.get("Evidence"))
                if not text:
                    continue
                # Round-robin assign an image to this evidence if any remain
                img_path = ""
                if img_queue:
                    img_path = os.path.join(img_abs_root, img_queue[i % len(img_queue)])
                title = str(row.get("Headline") or "").strip()
                caption = str(row.get("Description") or "").strip()
                positives.append({
                    "evidence_id": f"{cid}_{int(row.get('evidence_id', i))}",
                    "text": text,
                    "title": title,
                    "caption": caption,
                    "ocr": "",
                    "image_path": img_path,
                    "source": "mocheg/curated",
                    "url": str(row.get("Snopes URL") or "").strip(),
                })

            if not positives:
                n_skipped_no_evidence += 1
                continue

            sample = {
                "claim_id": cid,
                "claim": claim_text,
                "label": meta["label"],
                "claim_image_path": claim_img,
                "positive_evidence": positives,
                "negative_evidence": [],
            }
            if claim_img:
                n_with_image += 1
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"[{split}] wrote {n_written} claims to {out_path}", flush=True)
    print(f"[{split}]   with claim_image: {n_with_image} ({100*n_with_image/max(n_written,1):.1f}%)", flush=True)
    print(f"[{split}]   skipped no-claim: {n_skipped_no_claim}, no-evidence-text: {n_skipped_no_evidence}", flush=True)
    return n_written


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mocheg_root", default=DEFAULT_ROOT)
    p.add_argument("--out_dir", default="data/processed/mocheg")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    totals = {}
    for split in args.splits:
        out_path = os.path.join(args.out_dir, f"mocheg_{split}.jsonl")
        totals[split] = build_split(args.mocheg_root, split, out_path)

    print()
    print("=== summary ===")
    for split, n in totals.items():
        print(f"  {split}: {n} claims")


if __name__ == "__main__":
    main()
