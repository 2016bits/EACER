"""Jieba + BM25 sparse baseline on MR2 retrieval.

Word-level Chinese tokenisation with jieba instead of XLM-R SentencePiece.
Motivation:
  - The XLM-R-tokenised BM25 splits Chinese into character-level subwords,
    losing word boundaries that matter for fact-checking lexical match.
  - Jieba is the standard Chinese word segmenter for retrieval baselines.

Behaviour:
  - Chinese characters are jieba-segmented;
  - non-Chinese ASCII content is lowercased and whitespace-split (so English
    claims still get reasonable lexical match);
  - punctuation / single-character stopwords are filtered.

Also supports ``--lang_filter {all,zh,en}`` to evaluate on the Chinese-only or
English-only claim subset (auto-detected by Chinese-character ratio).

Outputs ``outputs/bm25_jieba_{split}_{lang}_results.json`` and prints the same
metrics as ``baseline_bm25.py``.

Usage::

    python scripts/baseline_bm25_jieba.py --split test
    python scripts/baseline_bm25_jieba.py --split test --lang_filter zh
    python scripts/baseline_bm25_jieba.py --split test --lang_filter en
"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from typing import List, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from tqdm import tqdm

import jieba

jieba.setLogLevel(40)  # silence the "Loading model" notice on first call

from src.data.dataset import ClaimQueries, EvidenceCorpus
from src.utils.metrics import retrieval_metrics


_CJK_RE = re.compile(r"[一-鿿]")
_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_STOP_CHARS = set("的了是和与及或在也都还就才已被把不没我你他她它这那有为之等於于哦呢吧啊呀")


def chinese_ratio(text: str) -> float:
    if not text:
        return 0.0
    n_cjk = sum(1 for c in text if _CJK_RE.match(c))
    n_letters = sum(1 for c in text if c.isalpha() or _CJK_RE.match(c))
    return n_cjk / max(n_letters, 1)


def tokenize_mixed(text: str) -> List[str]:
    """Jieba-cut CJK runs; whitespace + regex tokenise ASCII runs."""
    if not text:
        return []
    tokens: List[str] = []
    # jieba handles mixed text but tends to keep ASCII runs together; we still
    # post-process to lowercase + split ASCII alnum spans so e.g.
    # "iPhone15" -> ["iphone15"] and "Boston Orange" -> ["boston","orange"].
    for piece in jieba.cut(text, cut_all=False):
        piece = piece.strip()
        if not piece:
            continue
        if _CJK_RE.search(piece):
            # CJK token: keep as-is, drop if it's a 1-char stopword
            if len(piece) == 1 and piece in _STOP_CHARS:
                continue
            tokens.append(piece)
        else:
            # ASCII / numeric run: lowercase + re-split on non-alnum
            for sub in _WORD_RE.findall(piece.lower()):
                if len(sub) >= 1:
                    tokens.append(sub)
    return tokens


def build_bm25(docs_tokens, k1: float = 1.5, b: float = 0.75):
    N = len(docs_tokens)
    df: Counter = Counter()
    for tokens in docs_tokens:
        for t in set(tokens):
            df[t] += 1
    idf = {t: math.log((N - n + 0.5) / (n + 0.5) + 1.0) for t, n in df.items()}
    doc_lengths = np.array([len(d) for d in docs_tokens], dtype=np.float32)
    avg_dl = float(doc_lengths.mean()) if N > 0 else 1.0
    tf_per_doc = [Counter(d) for d in docs_tokens]
    return idf, doc_lengths, avg_dl, tf_per_doc, k1, b


def build_postings(tf_per_doc):
    postings = {}
    for i, tfs in enumerate(tf_per_doc):
        for t, f in tfs.items():
            postings.setdefault(t, []).append((i, f))
    return postings


def bm25_score_batch(
    queries_tokens, idf, doc_lengths, avg_dl, tf_per_doc, k1, b, postings
) -> np.ndarray:
    Q = len(queries_tokens)
    N = len(tf_per_doc)
    out = np.zeros((Q, N), dtype=np.float32)
    norm = k1 * (1.0 - b + b * doc_lengths / avg_dl)
    for qi, q_tokens in enumerate(queries_tokens):
        seen = set()
        for t in q_tokens:
            if t in seen or t not in idf:
                continue
            seen.add(t)
            idf_t = idf[t]
            for doc_i, f in postings.get(t, ()):
                denom = f + norm[doc_i]
                out[qi, doc_i] += idf_t * (f * (k1 + 1)) / denom
    return out


def evidence_text(ev: dict) -> str:
    parts = []
    for key in ("title", "caption", "text", "ocr"):
        v = ev.get(key)
        if v:
            parts.append(str(v))
    return " ".join(parts).strip()


def filter_queries(
    queries_items, mode: str
) -> List[int]:
    """Return indices of queries to keep."""
    if mode == "all":
        return list(range(len(queries_items)))
    keep: List[int] = []
    for i, q in enumerate(queries_items):
        ratio = chinese_ratio(q["claim"])
        if mode == "zh" and ratio >= 0.5:
            keep.append(i)
        elif mode == "en" and ratio < 0.1:
            keep.append(i)
    return keep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", default=None,
                   help="Defaults to data/processed/mr2/mr2_{split}.jsonl")
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--k1", type=float, default=1.5)
    p.add_argument("--b", type=float, default=0.75)
    p.add_argument("--top_k", type=int, default=100)
    p.add_argument("--lang_filter", default="all", choices=["all", "zh", "en"],
                   help="Subset the queries by detected language for analysis.")
    p.add_argument("--out_json", default=None)
    args = p.parse_args()

    jsonl = args.jsonl or f"data/processed/mr2/mr2_{args.split}.jsonl"
    out_json = args.out_json or f"outputs/bm25_jieba_{args.split}_{args.lang_filter}_results.json"

    print(f"[bm25-jieba] split={args.split}, lang_filter={args.lang_filter}, jsonl={jsonl}")

    corpus = EvidenceCorpus(jsonl, image_root="", image_size=224)
    queries = ClaimQueries(jsonl)
    print(f"[bm25-jieba] corpus={len(corpus)} evidences, queries={len(queries)}")

    docs_tokens: List[List[str]] = []
    for ev in tqdm(corpus.items, desc="jieba-tokenize corpus"):
        docs_tokens.append(tokenize_mixed(evidence_text(ev)))
    idf, doc_lengths, avg_dl, tf_per_doc, k1, b = build_bm25(docs_tokens, args.k1, args.b)
    postings = build_postings(tf_per_doc)
    print(f"[bm25-jieba] avg_dl={avg_dl:.1f}, vocab={len(idf)}, total tokens={int(doc_lengths.sum())}")

    # tokenise *all* queries first (so the corpus encode cost is paid once)
    all_q_tokens: List[List[str]] = [tokenize_mixed(q["claim"]) for q in queries.items]
    all_gold_ids: List[List[str]] = [q["gold_evidence_ids"] for q in queries.items]

    keep_idx = filter_queries(queries.items, args.lang_filter)
    print(f"[bm25-jieba] kept {len(keep_idx)} / {len(queries)} queries after lang_filter={args.lang_filter}")
    if not keep_idx:
        print("[bm25-jieba] nothing to score for this subset; exiting.")
        return

    queries_tokens = [all_q_tokens[i] for i in keep_idx]
    gold_ids_list = [all_gold_ids[i] for i in keep_idx]

    sims = bm25_score_batch(queries_tokens, idf, doc_lengths, avg_dl, tf_per_doc, k1, b, postings)
    print(f"[bm25-jieba] score matrix: {sims.shape}")

    top_idx = np.argsort(-sims, axis=1)[:, : args.top_k]
    evi_ids = [str(ev.get("evidence_id", "")) for ev in corpus.items]
    ranked = [[evi_ids[i] for i in row] for row in top_idx]

    ks = [1, 5, 10, 20, 100]
    metrics = retrieval_metrics(ranked, gold_ids_list, ks=ks)

    print(f"\n[bm25-jieba] {args.split.upper()}/{args.lang_filter} retrieval metrics:")
    for k in ["Recall@1", "Recall@5", "Recall@10", "Recall@20", "Recall@100",
              "MRR", "mAP", "NDCG@10"]:
        print(f"  {k:>10s}: {metrics.get(k, 0.0):.4f}")

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "tokenizer": "jieba",
            "k1": args.k1, "b": args.b,
            "split": args.split,
            "lang_filter": args.lang_filter,
            "n_queries_kept": len(keep_idx),
            "n_queries_total": len(queries),
            "metrics": metrics,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[bm25-jieba] wrote {out_json}")


if __name__ == "__main__":
    main()
