#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible single-node entrypoint.
# The canonical LR sweep implementation is sweep_wan22_lr_multinode.sh; use
# NNODES=1 here so single-node and multi-node launches share the same code path.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PER_DEVICE_BS="${PER_DEVICE_BS:-64}"

if [ "${NNODES}" != "1" ]; then
  echo "ERROR: sweep_wan22_lr_8gpu.sh is a single-node compatibility wrapper."
  echo "Use scripts/train/sweep_wan22_lr_multinode.sh directly for NNODES=${NNODES}."
  exit 1
fi

if [ -z "${MASTER_ADDR:-}" ]; then
  export MASTER_ADDR="127.0.0.1"
fi

if [ -z "${LRS:-}" ]; then
  export LRS="3e-5 5e-5 7e-5 1e-4 1.5e-4 2e-4 3e-4"
fi

exec bash "${SCRIPT_DIR}/sweep_wan22_lr_multinode.sh"
