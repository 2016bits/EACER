"""Mine BM25 hard negatives for the training JSONL.

Why: the current MR2 training JSONL has `negative_evidence: []` for every
claim, so the contrastive loss only sees in-batch (random-other-claim)
negatives. After Step 1 we observed that ECER cannot rerank BM25's top
candidates because it never saw lexically-similar-but-wrong evidences at
training time. This script fixes that by:

  1. Building BM25 over the full training evidence corpus.
  2. For each claim, scoring the corpus.
  3. Skipping positives (and optional sibling-claim positives if --strict).
  4. Writing the next `--num_negatives` hits into `negative_evidence`.

Output is a new JSONL (`*_hardneg.jsonl`) with the same schema as the input
but with the negative pool populated. The original JSONL is left untouched.

Usage::

    python scripts/mine_bm25_negatives.py \\
        --in_jsonl  data/processed/mr2/mr2_train.jsonl \\
        --out_jsonl data/processed/mr2/mr2_train_hardneg.jsonl \\
        --num_negatives 8 --skip 0

`--skip N` lets you drop the top-N hits (those tend to be near-duplicates
of the positives). Defaults to 0; try 5 if you want harder-but-not-identical
negatives.
"""

import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, logging as hf_logging

hf_logging.set_verbosity_error()

from baseline_bm25 import build_bm25, build_postings, bm25_score_batch, tokenize  # noqa


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _evidence_text(ev: Dict[str, Any]) -> str:
    parts = []
    for key in ("title", "caption", "text", "ocr"):
        v = ev.get(key)
        if v:
            parts.append(str(v))
    return " ".join(parts).strip() or "[no text]"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_jsonl", required=True)
    p.add_argument("--out_jsonl", required=True)
    p.add_argument("--tokenizer", default="xlm-roberta-base")
    p.add_argument("--num_negatives", type=int, default=8,
                   help="hard negatives kept per claim (collator will subsample)")
    p.add_argument("--skip", type=int, default=0,
                   help="drop the top-K BM25 hits before picking negatives "
                        "(useful if top results are near-duplicates of positives)")
    p.add_argument("--k1", type=float, default=1.5)
    p.add_argument("--b", type=float, default=0.75)
    args = p.parse_args()

    items = _load_jsonl(args.in_jsonl)
    print(f"[mine] loaded {len(items)} claims from {args.in_jsonl}")

    # ---- build flat evidence corpus -----------------------------------------
    seen = set()
    corpus: List[Dict[str, Any]] = []
    for it in items:
        for key in ("positive_evidence", "negative_evidence"):
            for ev in it.get(key, []) or []:
                eid = str(ev.get("evidence_id", ""))
                if not eid or eid in seen:
                    continue
                seen.add(eid)
                corpus.append(ev)
    N = len(corpus)
    eid_to_idx = {str(ev["evidence_id"]): i for i, ev in enumerate(corpus)}
    print(f"[mine] corpus = {N} unique evidences")

    # ---- BM25 ----------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    docs_tokens = [tokenize(tokenizer, _evidence_text(ev)) for ev in tqdm(corpus, desc="tokenize corpus")]
    idf, dl, avg_dl, tf, k1, b = build_bm25(docs_tokens, args.k1, args.b)
    postings = build_postings(tf)
    print(f"[mine] bm25 ready: vocab={len(idf)}, avg_dl={avg_dl:.1f}")

    # ---- per-claim mining ----------------------------------------------------
    queries_tokens = [tokenize(tokenizer, it["claim"]) for it in tqdm(items, desc="tokenize queries")]
    sims = bm25_score_batch(queries_tokens, idf, dl, avg_dl, tf, k1, b, postings)
    print(f"[mine] sims shape: {sims.shape}")

    # Vectorised top-(skip + n_pos_max + num_negs) to give us headroom to
    # remove positives without falling short.
    headroom = args.skip + 32 + args.num_negatives
    headroom = min(headroom, N)
    top_idx = np.argpartition(-sims, headroom - 1, axis=1)[:, :headroom]
    # Sort each row's headroom by descending score to make positive-filtering
    # honour BM25 ordering.
    rows = np.arange(sims.shape[0])[:, None]
    order = np.argsort(-sims[rows, top_idx], axis=1)
    top_idx = top_idx[rows, order]

    out_items: List[Dict[str, Any]] = []
    stats = Counter()
    for qi, it in enumerate(tqdm(items, desc="building output")):
        pos_ids = {str(ev["evidence_id"]) for ev in it.get("positive_evidence", []) or [] if ev.get("evidence_id")}
        hits = top_idx[qi]
        negs: List[Dict[str, Any]] = []
        seen_neg = set()
        skipped = 0
        for ev_idx in hits:
            ev_idx = int(ev_idx)
            ev = corpus[ev_idx]
            eid = str(ev.get("evidence_id", ""))
            if eid in pos_ids or eid in seen_neg:
                continue
            if skipped < args.skip:
                skipped += 1
                continue
            seen_neg.add(eid)
            negs.append(ev)
            if len(negs) >= args.num_negatives:
                break
        stats[len(negs)] += 1

        out = dict(it)
        out["negative_evidence"] = negs
        out_items.append(out)

    # ---- write ---------------------------------------------------------------
    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for it in out_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    # ---- stats ---------------------------------------------------------------
    print(f"\n[mine] wrote {args.out_jsonl}  ({len(out_items)} claims)")
    print("[mine] hard-negative count distribution:")
    for n in sorted(stats):
        pct = 100.0 * stats[n] / len(out_items)
        print(f"  {n:>3d} negs : {stats[n]:>6d} claims  ({pct:5.2f}%)")
    avg = sum(n * c for n, c in stats.items()) / max(len(out_items), 1)
    print(f"[mine] avg negatives per claim: {avg:.2f}")


if __name__ == "__main__":
    main()
