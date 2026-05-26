#!/bin/bash
set -euo pipefail

# Compatibility entry. Prefer:
#   scripts/train/droid_wan22_joint_fseq200.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DREAMZERO_ROOT="${DREAMZERO_ROOT:-/data/dreamzero}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/data/checkpoints/dreamzero}"
export DATASET_ROOT="${DATASET_ROOT:-/data/datasets/dreamzero}"
export WAN22_CKPT_DIR="${WAN22_CKPT_DIR:-${CHECKPOINT_ROOT}/Wan2.2-TI2V-5B}"
export IMAGE_ENCODER_DIR="${IMAGE_ENCODER_DIR:-${CHECKPOINT_ROOT}/Wan2.1-I2V-14B-480P}"
export TOKENIZER_DIR="${TOKENIZER_DIR:-${CHECKPOINT_ROOT}/umt5-xxl}"
export OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_ROOT}/dreamzero_droid_wan22_full_finetune_320x640_fseq200}"
export PYTHON_BIN="${PYTHON_BIN:-${DREAMZERO_ROOT}/.venv/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export PER_DEVICE_BS="${PER_DEVICE_BS:-1}"
export DEEPSPEED_CFG="${DEEPSPEED_CFG:-zero2_offload}"
export MAX_STEPS="${MAX_STEPS:-50000}"
export SAVE_STEPS="${SAVE_STEPS:-500}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"
export MAX_CHUNK_SIZE="${MAX_CHUNK_SIZE:-2}"
export TARGET_VIDEO_HEIGHT="${TARGET_VIDEO_HEIGHT:-${MODEL_TARGET_HEIGHT:-320}}"
export TARGET_VIDEO_WIDTH="${TARGET_VIDEO_WIDTH:-${MODEL_TARGET_WIDTH:-640}}"
export FRAME_SEQLEN="${FRAME_SEQLEN:-${MODEL_FRAME_SEQLEN:-200}}"
export SWANLAB_SYNC_WANDB="${SWANLAB_SYNC_WANDB:-1}"
export TRAIN_ARCHITECTURE="${TRAIN_ARCHITECTURE:-full}"
export SAVE_LORA_ONLY="${SAVE_LORA_ONLY:-false}"

exec "${SCRIPT_DIR}/droid_wan22_joint_fseq200.sh" "$@"
