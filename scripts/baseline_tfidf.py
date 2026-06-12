"""TF-IDF external baseline on MR2 retrieval.

Mirrors ``baseline_bm25.py`` so the numbers drop into the same comparison table:
- same XLM-RoBERTa SentencePiece tokenizer (vocabulary aligned to neural model)
- same EvidenceCorpus / ClaimQueries entry points
- same ``retrieval_metrics`` (Recall@K / MRR / mAP / NDCG@K)

Scoring is the classic length-normalised TF-IDF cosine:
    tf_norm(t, d) = (1 + log(tf(t, d))) if tf > 0 else 0
    idf(t)        = log((N + 1) / (df(t) + 1)) + 1   (smoothed, sklearn-style)
    w(t, d)       = tf_norm(t, d) * idf(t)
    score(q, d)   = sum_t w(t, q) * w(t, d) / (||w(*, q)|| * ||w(*, d)||)

Implemented with a hand-rolled inverted index so we never need sklearn (the
project ``bm25`` env has a broken sklearn install).

Usage::

    python scripts/baseline_tfidf.py --split test
    python scripts/baseline_tfidf.py --split val
"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from typing import Dict, List, Tuple

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


def tokenize(tokenizer, text: str, max_length: int = 256) -> List[int]:
    return tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]


def build_tfidf_index(
    docs_tokens: List[List[int]],
) -> Tuple[Dict[int, float], Dict[int, List[Tuple[int, float]]], np.ndarray]:
    """Return (idf, postings[t] = [(doc_i, w_td)], doc_norms[N])."""
    N = len(docs_tokens)
    df: Counter = Counter()
    tf_per_doc: List[Counter] = []
    for tokens in docs_tokens:
        tf = Counter(tokens)
        tf_per_doc.append(tf)
        for t in tf.keys():
            df[t] += 1

    # sklearn-style smoothed idf
    idf: Dict[int, float] = {
        t: math.log((N + 1.0) / (n + 1.0)) + 1.0 for t, n in df.items()
    }

    postings: Dict[int, List[Tuple[int, float]]] = {}
    doc_norms_sq = np.zeros(N, dtype=np.float64)
    for i, tf in enumerate(tf_per_doc):
        for t, f in tf.items():
            if f <= 0:
                continue
            w = (1.0 + math.log(f)) * idf[t]
            postings.setdefault(t, []).append((i, w))
            doc_norms_sq[i] += w * w
    doc_norms = np.sqrt(doc_norms_sq)
    # avoid division-by-zero for empty docs
    doc_norms[doc_norms == 0.0] = 1.0
    return idf, postings, doc_norms


def query_weights(
    query_tokens: List[int], idf: Dict[int, float]
) -> Tuple[Dict[int, float], float]:
    """Return (q_term -> weight, q_norm)."""
    tf = Counter(query_tokens)
    qw: Dict[int, float] = {}
    norm_sq = 0.0
    for t, f in tf.items():
        if t not in idf or f <= 0:
            continue
        w = (1.0 + math.log(f)) * idf[t]
        qw[t] = w
        norm_sq += w * w
    return qw, math.sqrt(norm_sq) or 1.0


def score_batch(
    queries_tokens: List[List[int]],
    idf: Dict[int, float],
    postings: Dict[int, List[Tuple[int, float]]],
    doc_norms: np.ndarray,
) -> np.ndarray:
    """Return (Q, N) cosine TF-IDF similarities."""
    Q = len(queries_tokens)
    N = doc_norms.shape[0]
    out = np.zeros((Q, N), dtype=np.float32)
    for qi, q_tokens in enumerate(queries_tokens):
        qw, q_norm = query_weights(q_tokens, idf)
        if not qw:
            continue
        acc = np.zeros(N, dtype=np.float64)
        for t, w_q in qw.items():
            for doc_i, w_d in postings.get(t, ()):
                acc[doc_i] += w_q * w_d
        out[qi] = (acc / (q_norm * doc_norms)).astype(np.float32)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", default=None,
                   help="Defaults to data/processed/mr2/mr2_{split}.jsonl")
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--tokenizer", default="xlm-roberta-base")
    p.add_argument("--top_k", type=int, default=100)
    p.add_argument("--lang_filter", default="all", choices=["all", "zh", "en"],
                   help="Subset queries by detected language for analysis.")
    p.add_argument("--out_json", default=None)
    args = p.parse_args()

    jsonl = args.jsonl or f"data/processed/mr2/mr2_{args.split}.jsonl"
    suffix = "" if args.lang_filter == "all" else f"_{args.lang_filter}"
    out_json = args.out_json or f"outputs/tfidf_{args.split}{suffix}_results.json"

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"[tfidf] tokenizer={args.tokenizer}, split={args.split}, jsonl={jsonl}")

    corpus = EvidenceCorpus(jsonl, image_root="", image_size=224)
    queries = ClaimQueries(jsonl)
    print(f"[tfidf] corpus={len(corpus)} evidences, queries={len(queries)}")

    docs_tokens: List[List[int]] = []
    for ev in tqdm(corpus.items, desc="tokenize corpus"):
        text = ev.get("text") or ""
        # mirror RetrievalDataset's text concat
        parts = []
        for key in ("title", "caption", "text", "ocr"):
            v = ev.get(key)
            if v:
                parts.append(str(v))
        joined = " ".join(parts).strip() or text
        docs_tokens.append(tokenize(tokenizer, joined))

    idf, postings, doc_norms = build_tfidf_index(docs_tokens)
    print(f"[tfidf] vocab={len(idf)}, avg_doc_len={np.mean([len(d) for d in docs_tokens]):.1f}")

    all_q_tokens: List[List[int]] = []
    all_gold_ids: List[List[str]] = []
    for q in tqdm(queries.items, desc="tokenize queries"):
        all_q_tokens.append(tokenize(tokenizer, q["claim"]))
        all_gold_ids.append(q["gold_evidence_ids"])

    keep_idx = _filter_query_indices(queries.items, args.lang_filter)
    print(f"[tfidf] kept {len(keep_idx)} / {len(queries)} queries after lang_filter={args.lang_filter}")
    if not keep_idx:
        print("[tfidf] nothing to score for this subset; exiting.")
        return
    queries_tokens = [all_q_tokens[i] for i in keep_idx]
    gold_ids_list = [all_gold_ids[i] for i in keep_idx]

    sims = score_batch(queries_tokens, idf, postings, doc_norms)
    print(f"[tfidf] score matrix: {sims.shape}")

    top_idx = np.argsort(-sims, axis=1)[:, : args.top_k]
    evi_ids = [str(ev.get("evidence_id", "")) for ev in corpus.items]
    ranked = [[evi_ids[i] for i in row] for row in top_idx]

    ks = [1, 5, 10, 20, 100]
    metrics = retrieval_metrics(ranked, gold_ids_list, ks=ks)

    print(f"\n[tfidf] {args.split.upper()}/{args.lang_filter} retrieval metrics:")
    for k in ["Recall@1", "Recall@5", "Recall@10", "Recall@20", "Recall@100",
              "MRR", "mAP", "NDCG@10"]:
        print(f"  {k:>10s}: {metrics.get(k, 0.0):.4f}")

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "tokenizer": args.tokenizer,
            "split": args.split,
            "lang_filter": args.lang_filter,
            "n_queries_kept": len(keep_idx),
            "n_queries_total": len(queries),
            "metrics": metrics,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[tfidf] wrote {out_json}")


if __name__ == "__main__":
    main()
