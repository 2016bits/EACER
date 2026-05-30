import math
from typing import Dict, Iterable, List, Sequence


def _dcg(rels: Sequence[int]) -> float:
    return sum((rel / math.log2(i + 2)) for i, rel in enumerate(rels))


def retrieval_metrics(
    ranked_ids: List[List[str]],
    gold_ids: List[List[str]],
    ks: Iterable[int] = (1, 5, 10, 100),
) -> Dict[str, float]:
    """Compute Recall@K, Precision@K, MRR, NDCG@K, mAP over queries."""
    ks = sorted(set(ks))
    n_queries = len(ranked_ids)
    assert n_queries == len(gold_ids), "queries and golds must align"

    metrics: Dict[str, float] = {f"Recall@{k}": 0.0 for k in ks}
    metrics.update({f"Precision@{k}": 0.0 for k in ks})
    metrics.update({f"NDCG@{k}": 0.0 for k in ks})
    metrics["MRR"] = 0.0
    metrics["mAP"] = 0.0

    for ranks, golds in zip(ranked_ids, gold_ids):
        gold_set = set(golds)
        if not gold_set:
            continue

        rels = [1 if rid in gold_set else 0 for rid in ranks]

        rr = 0.0
        for i, rel in enumerate(rels):
            if rel:
                rr = 1.0 / (i + 1)
                break
        metrics["MRR"] += rr

        ap, hits = 0.0, 0
        for i, rel in enumerate(rels):
            if rel:
                hits += 1
                ap += hits / (i + 1)
        metrics["mAP"] += ap / max(len(gold_set), 1)

        for k in ks:
            topk_rels = rels[:k]
            hit = sum(topk_rels)
            metrics[f"Recall@{k}"] += hit / len(gold_set)
            metrics[f"Precision@{k}"] += hit / k
            ideal = [1] * min(len(gold_set), k)
            idcg = _dcg(ideal) or 1.0
            metrics[f"NDCG@{k}"] += _dcg(topk_rels) / idcg

    for key in metrics:
        metrics[key] /= max(n_queries, 1)
    return metrics
