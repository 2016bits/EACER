#!/usr/bin/env bash
# 4-GPU DDP training for the ECER MR2 retriever.
#
# Wraps `torchrun` and the rank-aware src/train_ddp.py module. The script
# only sets sensible defaults — point it at any of the MR2 YAMLs:
#
#   bash scripts/train_mr2_ddp.sh configs/mr2_ecer.yaml      # full ECER
#   bash scripts/train_mr2_ddp.sh configs/mr2_base.yaml      # baseline
#   bash scripts/train_mr2_ddp.sh configs/mr2_tiny.yaml      # smoke
#
# Effective contrastive batch on 4 ranks at batch_size=32 (config default):
#   * in-batch negatives per claim: 4 * 32 - 1 = 127
#   * with hard_negatives_per_sample=k, evidence-side pool grows by 4*32*k
set -euo pipefail

CONFIG=${1:-configs/mr2_ecer.yaml}
NPROC=${NPROC:-4}
MASTER_PORT=${MASTER_PORT:-29501}

# HF tokenizer fork warning + verbose 'token > 512' notices: silence both.
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
# Quieter NCCL output (override with NCCL_DEBUG=INFO to debug hangs).
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# Use `python -m torch.distributed.run` (not `torchrun`) so we always pick up
# the currently active python interpreter; the standalone `torchrun` binary
# on this host points at /home/yangjun/.local/bin which is a different env.
python -m torch.distributed.run \
    --standalone \
    --nproc_per_node="${NPROC}" \
    --master_port="${MASTER_PORT}" \
    -m src.train_ddp \
    --config "${CONFIG}"
