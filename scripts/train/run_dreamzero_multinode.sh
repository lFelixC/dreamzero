#!/bin/bash
set -euo pipefail

# Compatibility entry for the old DROID joint multinode launcher.
# Prefer the clearer name:
#   scripts/train/droid_wan22_joint_fseq200.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DREAMZERO_ROOT="${DREAMZERO_ROOT:-/2025133806/dreamzero}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/2025133806/checkpoints/dreamzero}"
export DATASET_ROOT="${DATASET_ROOT:-/2025133806/datasets/dreamzero}"
export PYTHON_BIN="${PYTHON_BIN:-${DREAMZERO_ROOT}/.venv/bin/python}"
export EXPERIMENT_PY="${EXPERIMENT_PY:-${DREAMZERO_ROOT}/groot/vla/experiment/experiment.py}"
export WAN22_CKPT_DIR="${WAN22_CKPT_DIR:-${CHECKPOINT_ROOT}/Wan2.2-TI2V-5B}"
export IMAGE_ENCODER_DIR="${IMAGE_ENCODER_DIR:-${CHECKPOINT_ROOT}/Wan2.1-I2V-14B-480P}"
export TOKENIZER_DIR="${TOKENIZER_DIR:-${CHECKPOINT_ROOT}/umt5-xxl}"
export OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_ROOT}/dreamzero_droid_wan22_5B_full_finetune_320x640_fseq200}"
export TRAIN_ARCHITECTURE="${TRAIN_ARCHITECTURE:-full}"
export PER_DEVICE_BS="${PER_DEVICE_BS:-8}"
export DEEPSPEED_CFG="${DEEPSPEED_CFG:-zero2}"
export MAX_STEPS="${MAX_STEPS:-100000}"
export SAVE_STEPS="${SAVE_STEPS:-250}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-100}"
export MAX_CHUNK_SIZE="${MAX_CHUNK_SIZE:-4}"
export TARGET_VIDEO_HEIGHT="${TARGET_VIDEO_HEIGHT:-${MODEL_TARGET_HEIGHT:-320}}"
export TARGET_VIDEO_WIDTH="${TARGET_VIDEO_WIDTH:-${MODEL_TARGET_WIDTH:-640}}"
export FRAME_SEQLEN="${FRAME_SEQLEN:-${MODEL_FRAME_SEQLEN:-200}}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29400}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export SWANLAB_SYNC_WANDB="${SWANLAB_SYNC_WANDB:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
exec "${SCRIPT_DIR}/droid_wan22_joint_fseq200.sh" "$@"
