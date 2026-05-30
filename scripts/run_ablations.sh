#!/usr/bin/env bash
# Run the four ECER ablations sequentially on 4 GPUs.
#
# Baseline (mr2_base.yaml) and Full ECER (mr2_ecer.yaml) are assumed already
# trained; their outputs live in outputs/mr2_base/ and outputs/mr2_ecer_A/.
#
# Each ablation takes ~30-40 min on 4xA6000. Total: ~2-2.5h.
#
# Usage:
#   conda activate eacer
#   bash scripts/run_ablations.sh
#
# Optional: override which configs to run:
#   CONFIGS="configs/mr2_ablation_ciea.yaml" bash scripts/run_ablations.sh
# set -uo pipefail

CONFIGS=${CONFIGS:-"\
configs/mr2_ablation_ciea.yaml \
configs/mr2_ablation_no_S.yaml \
configs/mr2_ablation_no_R.yaml \
configs/mr2_ablation_no_Lcomp.yaml \
"}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export NPROC=${NPROC:-4}
export PATH=/data/yangjun/.conda/envs/eacer/bin:$PATH

for cfg in ${CONFIGS}; do
    out=$(grep '^output_dir:' "${cfg}" | awk '{print $2}')
    if [ -f "${out}/best.pt" ]; then
        echo "[ablation] SKIP ${cfg} — ${out}/best.pt already exists (delete it to re-run)"
        continue
    fi
    echo "================================================================"
    echo "[ablation] running ${cfg} -> ${out}"
    echo "================================================================"
    bash scripts/train_mr2_ddp.sh "${cfg}"
done

echo
echo "[ablation] done — best checkpoints:"
for cfg in ${CONFIGS}; do
    out=$(grep '^output_dir:' "${cfg}" | awk '{print $2}')
    echo "  $(basename ${cfg}): ${out}/best.pt"
done
