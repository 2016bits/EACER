#!/usr/bin/env bash
# After the hardneg training finishes, run all evaluations and dump a
# side-by-side comparison.
#
# Usage:
#   bash scripts/eval_hardneg_after_train.sh
#
# Compares:
#   - mr2_lambda_20   (previous best, no hard negs)  <- baseline
#   - mr2_ecer_hardneg (NEW: trained with BM25 hard negs)
#
# For each: dense-only metrics + BM25 -> rerank sweep (K_pool × alpha).
set -euo pipefail
cd "$(dirname "$0")/.."

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate bm25
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

NEW_CKPT=outputs/mr2_ecer_hardneg/best.pt
OLD_CKPT=outputs/mr2_lambda_20/best.pt

if [[ ! -f "$NEW_CKPT" ]]; then
    echo "FATAL: new checkpoint $NEW_CKPT not found. Training likely failed."
    exit 1
fi

# 1) Dense-only metrics for the new model (test split, full corpus).
echo "=== dense-only eval on test set ==="
python scripts/evaluate_all.py --runs mr2_ecer_hardneg --split test

# 2) Step-1 rerank with new checkpoint.
echo
echo "=== Step-1 BM25 -> ECER rerank with NEW ckpt ==="
python scripts/rerank_bm25_ecer.py \
    --ckpt "$NEW_CKPT" --split test \
    --k_pools 50 100 200 500 \
    --alphas 0.0 0.3 0.5 0.7 \
    --out_json outputs/rerank_test_hardneg.json

# 3) Recap baseline (old ckpt) so the comparison is on one page.
echo
echo "=== Step-1 BM25 -> ECER rerank with OLD ckpt (baseline) ==="
python scripts/rerank_bm25_ecer.py \
    --ckpt "$OLD_CKPT" --split test \
    --k_pools 50 100 200 500 \
    --alphas 0.0 0.3 0.5 0.7 \
    --out_json outputs/rerank_test_baseline.json

echo
echo "DONE. Compare outputs/rerank_test_{baseline,hardneg}.md"
