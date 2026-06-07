#!/usr/bin/env bash
# Launch the second-round experiments in priority order:
#   1. BM25 external baseline           (~5 min, no training)
#   2. Lambda sensitivity sweep         (~2 h, 4 runs)
#   3. ImageNet-1k entropy (K=1000)     (~30 min, 1 run)
#   4. Entropy scheme B (mid-entropy)   (~30 min, 1 run)
#
# Skips any run whose best.pt already exists.
#
# Usage (eacer env active):
#   bash scripts/run_extra_experiments.sh
#
# Customise:
#   STAGES="lambda" bash scripts/run_extra_experiments.sh         # only lambda
#   STAGES="bm25 imagenet" bash scripts/run_extra_experiments.sh
set -uo pipefail

STAGES=${STAGES:-"bm25 lambda imagenet scheme_b"}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export NPROC=${NPROC:-4}
export PATH=/data/yangjun/.conda/envs/eacer/bin:$PATH
export TOKENIZERS_PARALLELISM=false

train_one() {
    local cfg="$1"
    local out
    out=$(grep '^output_dir:' "${cfg}" | awk '{print $2}')
    if [ -f "${out}/best.pt" ]; then
        echo "[extra] SKIP ${cfg} (${out}/best.pt exists)"
        return
    fi
    echo "================================================================"
    echo "[extra] ${cfg} -> ${out}"
    echo "================================================================"
    bash scripts/train_mr2_ddp.sh "${cfg}"
}

for stage in ${STAGES}; do
    case "${stage}" in
        bm25)
            echo "================================================================"
            echo "[extra] running BM25 baseline on test split"
            echo "================================================================"
            python scripts/baseline_bm25.py --split test
            python scripts/baseline_bm25.py --split val
            ;;
        lambda)
            for cfg in configs/mr2_lambda_01.yaml configs/mr2_lambda_03.yaml \
                       configs/mr2_lambda_10.yaml configs/mr2_lambda_20.yaml; do
                train_one "${cfg}"
            done
            ;;
        imagenet)
            if [ ! -f data/imagenet1k_concepts.txt ]; then
                echo "[extra] building ImageNet-1k concept list (needs proxy if HF is blocked)"
                python scripts/build_imagenet1k_concepts.py || {
                    echo "[extra] WARN: concept build failed; entropy will use 60-word fallback"
                }
            fi
            train_one configs/mr2_ecer_K1000.yaml
            ;;
        scheme_b)
            if [ ! -f data/imagenet1k_concepts.txt ]; then
                echo "[extra] WARN: scheme_b config also wants ImageNet concepts; run 'imagenet' first"
            fi
            train_one configs/mr2_ecer_schemeB.yaml
            ;;
        *)
            echo "[extra] unknown stage: ${stage}"
            ;;
    esac
done

echo
echo "[extra] done. Aggregate everything onto test split with:"
echo "    python scripts/evaluate_all.py --split test"
