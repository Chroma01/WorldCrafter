#!/usr/bin/env bash
set -euo pipefail
# Single node: NPROC_PER_NODE=8 ... --gradient-accumulation-steps 4
# Reference topology: NNODES=4 NPROC_PER_NODE=8 NODE_RANK=0..3 MASTER_ADDR=... (one invocation/node)
cd "$(dirname "$0")/../.."
exec torchrun --nnodes="${NNODES:-1}" --nproc_per_node="${NPROC_PER_NODE:-8}" \
  --node_rank="${NODE_RANK:-0}" --master_addr="${MASTER_ADDR:-127.0.0.1}" \
  --master_port="${MASTER_PORT:-29500}" train.py \
  --config configs/training/base_osp_v2.yaml "$@"
