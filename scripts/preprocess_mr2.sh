#!/usr/bin/env bash
set -euo pipefail

# Convert the MR2 dump under $MR2_ROOT into the project's JSONL format.
# Usage:
#   bash scripts/preprocess_mr2.sh [MR2_ROOT] [OUT_DIR]
MR2_ROOT=${1:-/mnt/data/yangjun/data/mr2/queries_dataset_merge}
OUT_DIR=${2:-data/processed/mr2}

python scripts/preprocess_mr2.py \
    --mr2_root "${MR2_ROOT}" \
    --out_dir  "${OUT_DIR}" \
    --splits train val test \
    --include_query_image
