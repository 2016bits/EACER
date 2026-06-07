"""Hybrid BM25 + ECER retrieval via Reciprocal Rank Fusion.

Standard IR practice: take per-query rankings from a sparse (BM25) and a dense
(ECER) retriever, fuse with RRF:

    score_hybrid(d, q) = sum_r 1 / (k + rank_r(d, q))

where r iterates over rankers and k is a small constant (default 60).
RRF is parameter-free at the query level and surprisingly hard to beat.

This script re-runs BM25 from scratch, encodes the test set with a chosen
ECER checkpoint, and reports retrieval metrics for: BM25-only, ECER-only,
and BM25+ECER (hybrid). Outputs results to ``outputs/hybrid_{split}_results.json``.

Usage::

    python scripts/hybrid_bm25_ecer.py --ckpt outputs/mr2_lambda_20/best.pt --split test
"""

import argparse
import json
import os
import sys
from collections import defaultdict

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPImageProcessor
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

from src.data.collator import EncodeCollator
from src.data.dataset import ClaimQueries, EvidenceCorpus
from src.models import ECERRetriever
from src.utils.metrics import retrieval_metrics

# Reuse BM25 implementation
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from baseline_bm25 import build_bm25, build_postings, bm25_score_batch, tokenize  # noqa


def _move(b, device):
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}


@torch.no_grad()
def encode_with_ecer(ckpt_path: str, jsonl: str, device: torch.device):
    """Returns (claim_ids, claim_mat, evi_ids, evi_mat, gold_lists)."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = state["config"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["text_encoder"])
    image_processor = CLIPImageProcessor.from_pretrained(cfg["model"]["visual_encoder"])
    model = ECERRetriever(cfg).to(device).eval()
    model.load_state_dict(state["model"], strict=False)

    data_cfg = cfg["data"]
    use_claim_image = cfg["model"].get("use_claim_image", False)
    corpus = EvidenceCorpus(jsonl, data_cfg["image_root"], image_size=data_cfg["image_size"])
    queries = ClaimQueries(
        jsonl,
        with_claim_image=use_claim_image,
        image_root=data_cfg["image_root"],
        image_size=data_cfg["image_size"],
    )
    evi_coll = EncodeCollator(tokenizer, image_processor,
                              max_len=data_cfg["max_evidence_len"], text_key="text", with_image=True)
    claim_coll = EncodeCollator(
        tokenizer,
        image_processor=image_processor if use_claim_image else None,
        max_len=data_cfg["max_claim_len"], text_key="claim", with_image=use_claim_image,
    )
    bs = max(cfg["optim"]["batch_size"], 32)
    evi_loader = DataLoader(corpus, batch_size=bs, shuffle=False,
                            num_workers=data_cfg["num_workers"], collate_fn=evi_coll)
    claim_loader = DataLoader(queries, batch_size=bs, shuffle=False,
                              num_workers=data_cfg["num_workers"], collate_fn=claim_coll)

    evi_ids, evi_vecs = [], []
    for b in tqdm(evi_loader, desc="[ecer] encode evidence"):
        bb = _move(b, device)
        v = model.encode_evidence_for_index(
            input_ids=bb["input_ids"], attention_mask=bb["attention_mask"], pixel_values=bb["pixel_values"])
        evi_ids.extend(b["ids"]); evi_vecs.append(v.cpu())
    claim_ids, claim_vecs, gold_lists = [], [], []
    for b in tqdm(claim_loader, desc="[ecer] encode claims"):
        bb = _move(b, device)
        v = model.encode_claim_for_index(
            bb["input_ids"], bb["attention_mask"],
            claim_pixel_values=bb.get("pixel_values"),
        )
        claim_ids.extend(b["ids"]); claim_vecs.append(v.cpu()); gold_lists.extend(b["gold_evidence_ids"])
    return claim_ids, torch.cat(claim_vecs).numpy(), evi_ids, torch.cat(evi_vecs).numpy(), gold_lists, corpus


def run_bm25(corpus, queries, tokenizer_name: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    docs_tokens = [tokenize(tokenizer, ev.get("text") or "") for ev in tqdm(corpus.items, desc="[bm25] tokenize corpus")]
    idf, dl, avg_dl, tf, k1, b = build_bm25(docs_tokens)
    postings = build_postings(tf)
    queries_tokens = [tokenize(tokenizer, q["claim"]) for q in tqdm(queries.items, desc="[bm25] tokenize queries")]
    sims = bm25_score_batch(queries_tokens, idf, dl, avg_dl, tf, k1, b, postings)
    return sims


def ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    """For each row, return rank of each column (1-indexed; argmax -> 1)."""
    # higher score = better rank
    order = np.argsort(-scores, axis=1)
    ranks = np.empty_like(order)
    rows = np.arange(scores.shape[0])[:, None]
    cols = np.broadcast_to(rows, order.shape)
    ranks[cols, order] = np.arange(1, scores.shape[1] + 1)
    return ranks


def rrf_combine(rank_lists, k_rrf: int = 60) -> np.ndarray:
    """Sum 1/(k + rank) across ranker outputs. Higher RRF score = better."""
    out = np.zeros_like(rank_lists[0], dtype=np.float64)
    for r in rank_lists:
        out += 1.0 / (k_rrf + r)
    return out


def topk_to_ids(scores: np.ndarray, ids: list, top_k: int) -> list:
    """Return ranked lists of ids for each row (descending by score)."""
    top = np.argsort(-scores, axis=1)[:, :top_k]
    return [[ids[i] for i in row] for row in top]


def report(name: str, scores: np.ndarray, ids: list, golds: list, ks=(1, 5, 10, 20, 100)):
    ranked = topk_to_ids(scores, ids, max(ks))
    m = retrieval_metrics(ranked, golds, ks=ks)
    print(f"\n== {name} ==")
    for k in ["Recall@1", "Recall@5", "Recall@10", "Recall@20", "Recall@100", "MRR", "mAP", "NDCG@10"]:
        print(f"  {k:>10s}: {m.get(k, 0.0):.4f}")
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="ECER best.pt")
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--jsonl", default=None)
    p.add_argument("--k_rrf", type=int, default=60)
    p.add_argument("--out_json", default=None)
    args = p.parse_args()

    jsonl = args.jsonl or f"data/processed/mr2/mr2_{args.split}.jsonl"
    out_json = args.out_json or f"outputs/hybrid_{args.split}_results.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- ECER side -----------------------------------------------------------
    claim_ids_e, claim_mat, evi_ids_e, evi_mat, golds, corpus = encode_with_ecer(args.ckpt, jsonl, device)
    sims_e = claim_mat @ evi_mat.T                                  # (Q, N)

    # --- BM25 side -----------------------------------------------------------
    # Re-load queries to ensure same order; ClaimQueries is deterministic.
    queries = ClaimQueries(jsonl)
    sims_b = run_bm25(corpus, queries, "xlm-roberta-base")          # (Q, N)

    # Sanity: ECER and BM25 must share the same evidence ordering.
    evi_ids_b = [str(ev.get("evidence_id", "")) for ev in corpus.items]
    assert evi_ids_e == evi_ids_b, "ECER and BM25 evidence index ordering diverge"
    claim_ids_b = [q["claim_id"] for q in queries.items]
    assert claim_ids_e == claim_ids_b, "ECER and BM25 claim ordering diverge"

    # --- RRF fusion ----------------------------------------------------------
    ranks_e = ranks_from_scores(sims_e)
    ranks_b = ranks_from_scores(sims_b)
    sims_h = rrf_combine([ranks_e, ranks_b], k_rrf=args.k_rrf)

    m_b = report("BM25 only", sims_b, evi_ids_b, golds)
    m_e = report(f"ECER only ({os.path.basename(os.path.dirname(args.ckpt))})", sims_e, evi_ids_e, golds)
    m_h = report(f"Hybrid (RRF k={args.k_rrf})", sims_h, evi_ids_b, golds)

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "split": args.split,
            "ckpt": args.ckpt,
            "k_rrf": args.k_rrf,
            "bm25_only": m_b,
            "ecer_only": m_e,
            "hybrid": m_h,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
