"""Convert a raw MOCHEG dump into the JSONL format expected by `src.data`.

Expected raw layout (adapt as needed):
  data/raw/mocheg/
    Corpus2.csv                 # gold evidence per claim
    *_claim.csv                 # claim metadata
    images/                     # image files

Output:
  data/processed/mocheg_{train,val,test}.jsonl

Each output line:
  {claim_id, claim, label, positive_evidence: [...], negative_evidence: [...]}

This script is intentionally minimal — fill in dataset-specific parsing for
your exact MOCHEG dump. The defaults assume:
  - claim file columns: claim_id, claim, label, split
  - evidence file columns: claim_id, evidence_id, image_path, text, caption, ocr
"""

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from typing import Dict, List


def _read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build(
    claim_csv: str,
    evidence_csv: str,
    out_dir: str,
    num_negatives: int = 4,
    seed: int = 42,
):
    random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    claims = _read_csv(claim_csv)
    evidences = _read_csv(evidence_csv)

    # group positives by claim_id
    pos_by_claim: Dict[str, List[Dict]] = defaultdict(list)
    for ev in evidences:
        pos_by_claim[str(ev["claim_id"])].append({
            "evidence_id": str(ev.get("evidence_id", f"{ev['claim_id']}_{len(pos_by_claim[ev['claim_id']])}")),
            "image_path": ev.get("image_path", ""),
            "text": ev.get("text", ""),
            "caption": ev.get("caption", ""),
            "ocr": ev.get("ocr", ""),
            "source": ev.get("source", "mocheg"),
        })

    all_evidence = [e for evs in pos_by_claim.values() for e in evs]

    splits: Dict[str, List[Dict]] = defaultdict(list)
    for c in claims:
        cid = str(c["claim_id"])
        positives = pos_by_claim.get(cid, [])
        if not positives:
            continue

        # random negatives from other claims' evidence
        pos_ids = {p["evidence_id"] for p in positives}
        negatives = []
        for _ in range(num_negatives):
            cand = random.choice(all_evidence)
            if cand["evidence_id"] in pos_ids:
                continue
            negatives.append(cand)

        sample = {
            "claim_id": cid,
            "claim": c["claim"],
            "label": c.get("label", "UNK"),
            "positive_evidence": positives,
            "negative_evidence": negatives,
        }
        splits[c.get("split", "train").lower()].append(sample)

    for split, items in splits.items():
        out_path = os.path.join(out_dir, f"mocheg_{split}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"{out_path}: {len(items)} samples")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--claim_csv", required=True)
    parser.add_argument("--evidence_csv", required=True)
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument("--num_negatives", type=int, default=4)
    args = parser.parse_args()
    build(args.claim_csv, args.evidence_csv, args.out_dir, args.num_negatives)


if __name__ == "__main__":
    main()
