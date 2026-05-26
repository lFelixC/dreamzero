#!/bin/bash
set -euo pipefail

# Compatibility entry for the old Wan2.2 LoRA smoke script.
# Prefer the canonical joint fseq200 launcher for new full training runs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -n "${DREAMZERO_ROOT:-}" && -d "${DREAMZERO_ROOT}/groot" ]]; then
  :
elif [[ -d "/root/yejink/dreamzero/groot" ]]; then
  export DREAMZERO_ROOT="/root/yejink/dreamzero"
elif [[ -d "/root/dreamzero/groot" ]]; then
  export DREAMZERO_ROOT="/root/dreamzero"
elif [[ -d "${SCRIPT_REPO_ROOT}/groot" ]]; then
  export DREAMZERO_ROOT="${SCRIPT_REPO_ROOT}"
else
  export DREAMZERO_ROOT="${DREAMZERO_ROOT:-/root/yejink/dreamzero}"
fi

if [[ "${DROID_DATA_ROOT:-}" == "./data/droid_lerobot" ]]; then
  export DROID_DATA_ROOT="${DREAMZERO_ROOT}/data/droid_lerobot"
fi

export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${DREAMZERO_ROOT}/checkpoints}"
export DATASET_ROOT="${DATASET_ROOT:-${DROID_DATA_ROOT:-${DREAMZERO_ROOT}/data/droid_lerobot}}"
export OUTPUT_DIR="${OUTPUT_DIR:-${DREAMZERO_ROOT}/checkpoints/dreamzero_droid_wan22_lora}"
export WAN22_CKPT_DIR="${WAN22_CKPT_DIR:-${DREAMZERO_ROOT}/checkpoints/Wan2.2-TI2V-5B}"
export IMAGE_ENCODER_DIR="${IMAGE_ENCODER_DIR:-${DREAMZERO_ROOT}/checkpoints/Wan2.1-I2V-14B-480P}"
export TOKENIZER_DIR="${TOKENIZER_DIR:-${DREAMZERO_ROOT}/checkpoints/umt5-xxl}"
export TRAIN_ARCHITECTURE="${TRAIN_ARCHITECTURE:-lora}"
export SAVE_LORA_ONLY="${SAVE_LORA_ONLY:-true}"
export NUM_GPUS="${NUM_GPUS:-8}"
export PER_DEVICE_BS="${PER_DEVICE_BS:-1}"
export MAX_STEPS="${MAX_STEPS:-100}"
export SAVE_STEPS="${SAVE_STEPS:-1000}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"
export DEEPSPEED_CFG="${DEEPSPEED_CFG:-zero2}"
export DATALOADER_PIN_MEMORY="${DATALOADER_PIN_MEMORY:-false}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-1}"
export MAX_CHUNK_SIZE="${MAX_CHUNK_SIZE:-4}"
export SAVE_STRATEGY="${SAVE_STRATEGY:-no}"
export TARGET_VIDEO_HEIGHT="${TARGET_VIDEO_HEIGHT:-${MODEL_TARGET_HEIGHT:-160}}"
export TARGET_VIDEO_WIDTH="${TARGET_VIDEO_WIDTH:-${MODEL_TARGET_WIDTH:-320}}"
export FRAME_SEQLEN="${FRAME_SEQLEN:-${MODEL_FRAME_SEQLEN:-50}}"

exec "${SCRIPT_DIR}/droid_wan22_joint_fseq200.sh" "$@"
