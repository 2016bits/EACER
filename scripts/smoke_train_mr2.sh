#!/usr/bin/env bash
# Smoke-train ECER on a tiny MR2 subset (~200 train / 100 val claims).
#
# Goal: verify the *real* pipeline (xlm-roberta-base + CLIP-ViT-B/32) trains
# end-to-end on GPU before launching a full run. What to expect:
#   - first launch downloads CLIP's text tower for the entropy module
#     (~600 MB; cached after first run)
#   - epoch 1 train loss should sit around log(B*(1+n_neg)) ~= log(16) ~= 2.77
#   - by epoch 3 train loss should be visibly lower (typically < 1.5)
#   - val Recall@10 should land well above the ~0.1% random baseline
#     (random for ~900-evidence pool = 10/900 ≈ 1.1%; expect >> that)
#
# Usage (from the project root, with the eacer conda env activated):
#   conda activate eacer
#   bash scripts/smoke_train_mr2.sh
#
# Optional overrides:
#   CONFIG=configs/mr2_tiny.yaml \
#   N_TRAIN=200 N_VAL=100 N_TEST=100 \
#   bash scripts/smoke_train_mr2.sh
# set -euo pipefail

CONFIG=${CONFIG:-configs/mr2_tiny.yaml}
N_TRAIN=${N_TRAIN:-200}
N_VAL=${N_VAL:-100}
N_TEST=${N_TEST:-100}
SRC_DIR=${SRC_DIR:-data/processed/mr2}
DST_DIR=${DST_DIR:-data/processed/mr2_tiny}

if [ ! -f "${SRC_DIR}/mr2_train.jsonl" ]; then
    echo "[smoke] missing ${SRC_DIR}/mr2_train.jsonl — running full preprocess first"
    bash scripts/preprocess_mr2.sh
fi

echo "[smoke] slicing tiny subsets (${N_TRAIN}/${N_VAL}/${N_TEST}) into ${DST_DIR}"
python scripts/make_tiny_subsets.py \
    --src_dir "${SRC_DIR}" \
    --dst_dir "${DST_DIR}" \
    --n_train "${N_TRAIN}" \
    --n_val   "${N_VAL}" \
    --n_test  "${N_TEST}" \
    --force

echo "[smoke] training ${CONFIG}"
python -m src.train --config "${CONFIG}"

echo "[smoke] done — checkpoints under outputs/mr2_tiny/, tensorboard under outputs/mr2_tiny/tb/"
