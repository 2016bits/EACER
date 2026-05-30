#!/usr/bin/env bash
# Multi-seed runs of (config × seed) for variance estimation.
#
# Default: 3 seeds × {baseline, full ECER} = 6 runs (~3 h on 4xA6000).
# Each run writes to outputs/<orig>_seed{N}/best.pt so existing single-seed
# checkpoints are NOT overwritten. Re-running is idempotent (existing runs
# are skipped).
#
# Usage (eacer env active):
#   bash scripts/run_multi_seed.sh
#
# Customise:
#   CONFIGS="configs/mr2_ecer.yaml configs/mr2_ablation_ciea.yaml" \
#   SEEDS="42 2024 0" \
#   bash scripts/run_multi_seed.sh
#
# After all seeds finish, aggregate with:
#   python scripts/evaluate_all.py --split test
#   (auto-discovers all outputs/*_seed*/best.pt)
set -uo pipefail

CONFIGS=${CONFIGS:-"\
configs/mr2_base.yaml \
configs/mr2_ecer.yaml \
"}
SEEDS=${SEEDS:-"42 2024 0"}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export NPROC=${NPROC:-4}
export PATH=/data/yangjun/.conda/envs/eacer/bin:$PATH
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

for cfg in ${CONFIGS}; do
    out_base=$(grep '^output_dir:' "${cfg}" | awk '{print $2}')
    for seed in ${SEEDS}; do
        out="${out_base}_seed${seed}"
        if [ -f "${out}/best.pt" ]; then
            echo "[multi-seed] SKIP ${cfg} seed=${seed} (${out}/best.pt exists)"
            continue
        fi
        echo "================================================================"
        echo "[multi-seed] ${cfg} seed=${seed} -> ${out}"
        echo "================================================================"
        python -m torch.distributed.run \
            --standalone \
            --nproc_per_node="${NPROC}" \
            --master_port=29501 \
            -m src.train_ddp \
            --config "${cfg}" \
            --seed "${seed}"
    done
done

echo
echo "[multi-seed] done. To aggregate metrics:"
echo "  python scripts/evaluate_all.py --split test"
