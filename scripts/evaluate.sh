#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/ecer_mocheg.yaml}
CKPT=${2:-outputs/ecer_mocheg_A/best.pt}
SPLIT=${3:-test}
OUT=${4:-outputs/ecer_mocheg_A/predictions_${SPLIT}.jsonl}
python -m src.evaluate_retrieval --config "$CONFIG" --ckpt "$CKPT" --split "$SPLIT" --out "$OUT"
