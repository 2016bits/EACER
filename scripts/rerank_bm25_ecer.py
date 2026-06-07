"""Two-stage retrieval: BM25 first-stage, ECER reranker.

Standard IR recipe:
    Stage-1: BM25 retrieves top-K candidates per query  (high recall, cheap)
    Stage-2: ECER re-scores ONLY those K candidates     (high precision, expensive)
    All non-candidate evidences get score = -inf so they cannot appear in the
    final ranking.

This is different from RRF fusion in ``hybrid_bm25_ecer.py``: RRF averages
ranks from two independent rankers; rerank uses ECER scores as the final
ranking but constrained to BM25's top-K shortlist.

Why this works on MR2:
- BM25 already nails R@100 = 35% on its own — it's a strong, cheap shortlist
- ECER alone struggles end-to-end (R@10 = 11%) but should be good at
  re-ordering within a small, already-relevant pool
- Final R@k is bounded above by BM25 R@K_pool

We sweep K_pool ∈ {50, 100, 200, 500} so the precision/recall trade-off is
visible: small K = higher precision but lower recall ceiling; large K = the
opposite.

Usage::

    python scripts/rerank_bm25_ecer.py --ckpt outputs/mr2_lambda_20/best.pt --split test
"""

import argparse
import json
import os
import sys
from typing import List

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np
import torch

from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

from src.data.dataset import ClaimQueries
from src.utils.metrics import retrieval_metrics

# Reuse already-tested encoders + BM25 from hybrid script
from hybrid_bm25_ecer import encode_with_ecer, run_bm25, topk_to_ids


def _normalise_per_row(x: np.ndarray) -> np.ndarray:
    """Min-max normalise each row to [0, 1]. Rows with constant value stay 0."""
    lo = x.min(axis=1, keepdims=True)
    hi = x.max(axis=1, keepdims=True)
    rng = np.where(hi > lo, hi - lo, 1.0)
    return (x - lo) / rng


def rerank_within_pool(
    sims_first: np.ndarray,            # (Q, N), BM25 scores
    sims_second: np.ndarray,           # (Q, N), ECER scores
    k_pool: int,
    alpha: float = 0.0,                # 0.0 = pure ECER rerank; >0 mixes BM25 back
) -> np.ndarray:
    """Return (Q, N) scores where only each query's BM25 top-K_pool keep a
    score; everything else is -inf so it cannot enter the final top-k.

    Within the shortlist, the ranking is:
        score = (1 - alpha) * norm(ecer) + alpha * norm(bm25)
    alpha=0 → pure ECER rerank; alpha=1 → keep BM25 ranking inside the pool.
    """
    Q, N = sims_first.shape
    norm_b = _normalise_per_row(sims_first)
    norm_e = _normalise_per_row(sims_second)
    fused = (1.0 - alpha) * norm_e + alpha * norm_b
    out = np.full_like(sims_second, -np.inf, dtype=np.float64)
    topk_idx = np.argpartition(-sims_first, k_pool - 1, axis=1)[:, :k_pool]
    rows = np.arange(Q)[:, None]
    out[rows, topk_idx] = fused[rows, topk_idx]
    return out


def report(name: str, scores: np.ndarray, ids: List[str], golds: List[List[str]], ks=(1, 5, 10, 20, 100)):
    ranked = topk_to_ids(scores, ids, max(ks))
    m = retrieval_metrics(ranked, golds, ks=ks)
    print(f"\n== {name} ==")
    for k in ["Recall@1", "Recall@5", "Recall@10", "Recall@20", "Recall@100", "MRR", "mAP", "NDCG@10"]:
        print(f"  {k:>11s}: {m.get(k, 0.0):.4f}")
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="path to ECER best.pt")
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--jsonl", default=None)
    p.add_argument("--k_pools", type=int, nargs="+", default=[50, 100, 200, 500],
                   help="BM25 shortlist sizes to sweep")
    p.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.3, 0.5, 0.7],
                   help="BM25 mixing weights to sweep (0=pure ECER, 1=pure BM25 within pool)")
    p.add_argument("--out_json", default=None)
    args = p.parse_args()

    jsonl = args.jsonl or f"data/processed/mr2/mr2_{args.split}.jsonl"
    out_json = args.out_json or f"outputs/rerank_{args.split}_results.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Stage-2 encoder: ECER ----------------------------------------------
    print(f"[1/2] encoding with ECER from {args.ckpt}")
    claim_ids_e, claim_mat, evi_ids_e, evi_mat, golds, corpus = encode_with_ecer(args.ckpt, jsonl, device)
    sims_e = claim_mat @ evi_mat.T                                  # (Q, N)

    # --- Stage-1 retriever: BM25 --------------------------------------------
    print("[2/2] running BM25")
    queries = ClaimQueries(jsonl)
    sims_b = run_bm25(corpus, queries, "xlm-roberta-base")          # (Q, N)

    # Sanity checks
    evi_ids = [str(ev.get("evidence_id", "")) for ev in corpus.items]
    assert evi_ids_e == evi_ids
    assert claim_ids_e == [q["claim_id"] for q in queries.items]
    N_corpus = len(evi_ids)
    print(f"\ncorpus size: {N_corpus}, queries: {len(claim_ids_e)}")

    # --- Baselines ----------------------------------------------------------
    metrics = {}
    metrics["bm25_only"] = report("BM25 only", sims_b, evi_ids, golds)
    metrics["ecer_only"] = report(f"ECER only ({os.path.basename(os.path.dirname(args.ckpt))})",
                                  sims_e, evi_ids, golds)

    # --- Rerank @ each (K_pool, alpha) --------------------------------------
    for k_pool in args.k_pools:
        if k_pool > N_corpus:
            print(f"\n[skip] k_pool={k_pool} exceeds corpus size {N_corpus}")
            continue
        for alpha in args.alphas:
            tag = f"rerank_top{k_pool}_alpha{alpha:.1f}"
            sims_r = rerank_within_pool(sims_b, sims_e, k_pool, alpha=alpha)
            metrics[tag] = report(
                f"BM25 top-{k_pool} -> rerank (alpha={alpha:.1f})",
                sims_r, evi_ids, golds,
            )

    # --- Save ----------------------------------------------------------------
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    payload = {
        "split": args.split,
        "ckpt": args.ckpt,
        "corpus_size": int(N_corpus),
        "k_pools": args.k_pools,
        "metrics": metrics,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {out_json}")

    # --- Markdown summary table ---------------------------------------------
    md_lines = [
        "# BM25 → ECER Rerank — comparison table",
        f"checkpoint: `{args.ckpt}`  |  split: `{args.split}`  |  corpus: {N_corpus} evidences",
        "",
        "| System | R@1 | R@5 | R@10 | R@20 | R@100 | MRR | mAP | NDCG@10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    name_map = {
        "bm25_only": "BM25 only",
        "ecer_only": "ECER only",
    }
    rerank_keys = []
    for k_pool in args.k_pools:
        for alpha in args.alphas:
            tag = f"rerank_top{k_pool}_alpha{alpha:.1f}"
            name_map[tag] = f"BM25 top-{k_pool} → rerank α={alpha:.1f}"
            rerank_keys.append(tag)
    for key in ["bm25_only", "ecer_only"] + rerank_keys:
        if key not in metrics:
            continue
        m = metrics[key]
        row = (
            f"| {name_map[key]} | "
            f"{100*m.get('Recall@1', 0):.2f} | "
            f"{100*m.get('Recall@5', 0):.2f} | "
            f"{100*m.get('Recall@10', 0):.2f} | "
            f"{100*m.get('Recall@20', 0):.2f} | "
            f"{100*m.get('Recall@100', 0):.2f} | "
            f"{100*m.get('MRR', 0):.2f} | "
            f"{100*m.get('mAP', 0):.2f} | "
            f"{100*m.get('NDCG@10', 0):.2f} |"
        )
        md_lines.append(row)
    md_path = out_json.replace(".json", ".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
