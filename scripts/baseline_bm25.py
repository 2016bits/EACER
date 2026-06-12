"""BM25 external baseline on MR2 retrieval.

No training: just BM25 scoring of (claim caption) against (evidence text).
Tokenises with the same XLM-RoBERTa SentencePiece tokenizer the neural model
uses, so vocabulary coverage is identical (no jieba/whitespace bias).

Outputs the same Recall@K / MRR / mAP / NDCG@K metrics as ``evaluate_all.py``
so the numbers drop straight into the comparison table.

Usage::

    python scripts/baseline_bm25.py --split test
    python scripts/baseline_bm25.py --split val      # for sanity vs val-set runs
"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from typing import List

_CJK_RE = re.compile(r"[一-鿿]")


def _chinese_ratio(text: str) -> float:
    if not text:
        return 0.0
    n_cjk = sum(1 for c in text if _CJK_RE.match(c))
    n_letters = sum(1 for c in text if c.isalpha() or _CJK_RE.match(c))
    return n_cjk / max(n_letters, 1)


def _filter_query_indices(queries_items, mode: str):
    if mode == "all":
        return list(range(len(queries_items)))
    keep = []
    for i, q in enumerate(queries_items):
        ratio = _chinese_ratio(q["claim"])
        if mode == "zh" and ratio >= 0.5:
            keep.append(i)
        elif mode == "en" and ratio < 0.1:
            keep.append(i)
    return keep

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

from src.data.dataset import ClaimQueries, EvidenceCorpus
from src.utils.metrics import retrieval_metrics


def tokenize(tokenizer, text: str) -> List[int]:
    return tokenizer(text, add_special_tokens=False, truncation=True, max_length=256)["input_ids"]


def build_bm25(docs_tokens: List[List[int]], k1: float = 1.5, b: float = 0.75):
    """Return (idf, doc_lengths, avg_dl, tf_per_doc) for BM25 scoring."""
    N = len(docs_tokens)
    df = Counter()
    for tokens in docs_tokens:
        for t in set(tokens):
            df[t] += 1
    idf = {t: math.log((N - n + 0.5) / (n + 0.5) + 1.0) for t, n in df.items()}
    doc_lengths = np.array([len(d) for d in docs_tokens], dtype=np.float32)
    avg_dl = float(doc_lengths.mean()) if N > 0 else 1.0
    tf_per_doc = [Counter(d) for d in docs_tokens]
    return idf, doc_lengths, avg_dl, tf_per_doc, k1, b


def bm25_score(query_tokens: List[int], idf, doc_lengths, avg_dl, tf_per_doc, k1, b) -> np.ndarray:
    """Score every document against the query. Returns (N,) np.array."""
    N = len(tf_per_doc)
    scores = np.zeros(N, dtype=np.float32)
    q_terms = set(query_tokens)
    for t in q_terms:
        if t not in idf:
            continue
        idf_t = idf[t]
        for i in range(N):
            f = tf_per_doc[i].get(t, 0)
            if f == 0:
                continue
            dl = doc_lengths[i]
            denom = f + k1 * (1.0 - b + b * dl / avg_dl)
            scores[i] += idf_t * (f * (k1 + 1)) / denom
    return scores


def bm25_score_batch(query_tokens_list, idf, doc_lengths, avg_dl, tf_per_doc, k1, b,
                      term_postings) -> np.ndarray:
    """Faster version using inverted index. Returns (Q, N)."""
    Q = len(query_tokens_list)
    N = len(tf_per_doc)
    out = np.zeros((Q, N), dtype=np.float32)
    # Precompute per-doc normalisation factor
    norm = k1 * (1.0 - b + b * doc_lengths / avg_dl)        # (N,)
    for qi, q_tokens in enumerate(query_tokens_list):
        seen = set()
        for t in q_tokens:
            if t in seen or t not in idf:
                continue
            seen.add(t)
            idf_t = idf[t]
            for doc_i, f in term_postings.get(t, ()):
                denom = f + norm[doc_i]
                out[qi, doc_i] += idf_t * (f * (k1 + 1)) / denom
    return out


def build_postings(tf_per_doc):
    """t -> list of (doc_idx, term_freq)."""
    postings = {}
    for i, tfs in enumerate(tf_per_doc):
        for t, f in tfs.items():
            postings.setdefault(t, []).append((i, f))
    return postings


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", default=None,
                   help="Defaults to data/processed/mr2/mr2_{split}.jsonl")
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--tokenizer", default="xlm-roberta-base")
    p.add_argument("--k1", type=float, default=1.5)
    p.add_argument("--b", type=float, default=0.75)
    p.add_argument("--top_k", type=int, default=100)
    p.add_argument("--lang_filter", default="all", choices=["all", "zh", "en"],
                   help="Subset queries by detected language for analysis.")
    p.add_argument("--out_json", default=None)
    args = p.parse_args()

    jsonl = args.jsonl or f"data/processed/mr2/mr2_{args.split}.jsonl"
    suffix = "" if args.lang_filter == "all" else f"_{args.lang_filter}"
    out_json = args.out_json or f"outputs/bm25_{args.split}{suffix}_results.json"

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"[bm25] tokenizer={args.tokenizer}, split={args.split}, jsonl={jsonl}")

    corpus = EvidenceCorpus(jsonl, image_root="", image_size=224)
    queries = ClaimQueries(jsonl)
    print(f"[bm25] corpus={len(corpus)} evidences, queries={len(queries)}")

    docs_tokens: List[List[int]] = []
    for ev in tqdm(corpus.items, desc="tokenize corpus"):
        text = ev.get("text") or ""
        docs_tokens.append(tokenize(tokenizer, text))
    idf, doc_lengths, avg_dl, tf_per_doc, k1, b = build_bm25(docs_tokens, args.k1, args.b)
    postings = build_postings(tf_per_doc)
    print(f"[bm25] avg_dl={avg_dl:.1f}, vocab={len(idf)}, total tokens={int(doc_lengths.sum())}")

    all_q_tokens: List[List[int]] = []
    all_gold_ids: List[List[str]] = []
    for q in tqdm(queries.items, desc="tokenize queries"):
        all_q_tokens.append(tokenize(tokenizer, q["claim"]))
        all_gold_ids.append(q["gold_evidence_ids"])

    keep_idx = _filter_query_indices(queries.items, args.lang_filter)
    print(f"[bm25] kept {len(keep_idx)} / {len(queries)} queries after lang_filter={args.lang_filter}")
    if not keep_idx:
        print("[bm25] nothing to score for this subset; exiting.")
        return
    queries_tokens = [all_q_tokens[i] for i in keep_idx]
    gold_ids_list = [all_gold_ids[i] for i in keep_idx]

    # Score everything (queries × docs). For ~10k docs × ~1k queries this fits.
    sims = bm25_score_batch(queries_tokens, idf, doc_lengths, avg_dl, tf_per_doc, k1, b, postings)
    print(f"[bm25] score matrix: {sims.shape}")

    # Top-K per query
    top_idx = np.argsort(-sims, axis=1)[:, : args.top_k]
    evi_ids = [str(ev.get("evidence_id", "")) for ev in corpus.items]
    ranked = [[evi_ids[i] for i in row] for row in top_idx]

    ks = [1, 5, 10, 20, 100]
    metrics = retrieval_metrics(ranked, gold_ids_list, ks=ks)

    print(f"\n[bm25] {args.split.upper()}/{args.lang_filter} retrieval metrics:")
    for k in ["Recall@1", "Recall@5", "Recall@10", "Recall@20", "Recall@100",
              "MRR", "mAP", "NDCG@10"]:
        print(f"  {k:>10s}: {metrics.get(k, 0.0):.4f}")

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "tokenizer": args.tokenizer,
            "k1": args.k1, "b": args.b,
            "split": args.split,
            "lang_filter": args.lang_filter,
            "n_queries_kept": len(keep_idx),
            "n_queries_total": len(queries),
            "metrics": metrics,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[bm25] wrote {out_json}")


if __name__ == "__main__":
    main()
