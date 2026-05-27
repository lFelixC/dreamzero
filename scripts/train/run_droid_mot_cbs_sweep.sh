#!/bin/bash
set -euo pipefail

# Sequentially run CBS branches from one warmup checkpoint. Override CBS_BATCHES
# to resume from a subset, e.g. CBS_BATCHES="512 1024".

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${WARMUP_CKPT:-}" ]]; then
  echo "ERROR: WARMUP_CKPT is required"
  exit 1
fi

CBS_BATCHES="${CBS_BATCHES:-128 256 512 1024}"
BASE_GLOBAL_BATCH_SIZE="${BASE_GLOBAL_BATCH_SIZE:-128}"
BASE_LEARNING_RATE="${BASE_LEARNING_RATE:-1.5e-5}"
CBS_BASE_STEPS="${CBS_BASE_STEPS:-4096}"
ACTION_LOSS_WEIGHT="${ACTION_LOSS_WEIGHT:-2.0}"
PER_DEVICE_BS="${PER_DEVICE_BS:-32}"
USE_GRADIENT_CHECKPOINTING="${USE_GRADIENT_CHECKPOINTING:-true}"

echo "========== DROID MoT CBS sequential sweep =========="
echo "WARMUP_CKPT=${WARMUP_CKPT}"
echo "CBS_BATCHES=${CBS_BATCHES}"
echo "BASE_GLOBAL_BATCH_SIZE=${BASE_GLOBAL_BATCH_SIZE}"
echo "BASE_LEARNING_RATE=${BASE_LEARNING_RATE}"
echo "CBS_BASE_STEPS=${CBS_BASE_STEPS}"
echo "ACTION_LOSS_WEIGHT=${ACTION_LOSS_WEIGHT}"
echo "PER_DEVICE_BS=${PER_DEVICE_BS}"
echo "USE_GRADIENT_CHECKPOINTING=${USE_GRADIENT_CHECKPOINTING}"
echo "===================================================="

for batch_size in ${CBS_BATCHES}; do
  echo "----- Starting CBS branch: global batch ${batch_size} -----"
  CBS_GLOBAL_BATCH_SIZE="${batch_size}" \
  BASE_GLOBAL_BATCH_SIZE="${BASE_GLOBAL_BATCH_SIZE}" \
  BASE_LEARNING_RATE="${BASE_LEARNING_RATE}" \
  CBS_BASE_STEPS="${CBS_BASE_STEPS}" \
  ACTION_LOSS_WEIGHT="${ACTION_LOSS_WEIGHT}" \
  PER_DEVICE_BS="${PER_DEVICE_BS}" \
  USE_GRADIENT_CHECKPOINTING="${USE_GRADIENT_CHECKPOINTING}" \
  bash "${SCRIPT_DIR}/run_droid_mot_cbs_branch.sh"
  echo "----- Finished CBS branch: global batch ${batch_size} -----"
done
