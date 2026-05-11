#!/bin/bash
set -euo pipefail

# A800 2-node RoboTwin Wan2.2 joint training with cost-grouped shard scheduling.
# Run this script on every node in the experiment with the same MASTER_ADDR
# and MASTER_PORT, changing only NODE_RANK.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${SCRIPT_DIR}/../robotwin_training_wan22.sh}"

# -----------------------------
# Fixed runtime environment
# -----------------------------
export VIRTUAL_ENV="${VIRTUAL_ENV:-/opt/venvs/dreamzero}"
export PYTHON_BIN="${PYTHON_BIN:-${VIRTUAL_ENV}/bin/python}"
export PATH="${VIRTUAL_ENV}/bin:/usr/local/bin:/root/.local/bin:${PATH:-}"
export DREAMZERO_ROOT="${DREAMZERO_ROOT:-/2023133163/liuf/dreamzero}"
export DATASET_ROOT="${DATASET_ROOT:-/2023133163/datasets/dreamzero}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/2023133163/checkpoints/dreamzero}"
export PYTHONPATH="${DREAMZERO_ROOT}:${PYTHONPATH:-}"

export ARCH="${ARCH:-joint}"
export ROBOTWIN_DATA_ROOT="${ROBOTWIN_DATA_ROOT:-${DATASET_ROOT}/robotwin_unified_dreamzero}"
export OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_ROOT}/dreamzero_robotwin_wan22_joint_cost_group_a800_2node}"
export WAN22_CKPT_DIR="${WAN22_CKPT_DIR:-${CHECKPOINT_ROOT}/Wan2.2-TI2V-5B}"
export IMAGE_ENCODER_DIR="${IMAGE_ENCODER_DIR:-${WAN22_CKPT_DIR}}"
export TOKENIZER_DIR="${TOKENIZER_DIR:-${WAN22_CKPT_DIR}/google/umt5-xxl}"

export WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-dreamzero}"
export PER_DEVICE_BS="${PER_DEVICE_BS:-16}"
export MAX_STEPS="${MAX_STEPS:-60000}"
export SAVE_STEPS="${SAVE_STEPS:-5000}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
export DEEPSPEED_CFG="${DEEPSPEED_CFG:-zero2}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
export DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-1}"
export DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-true}"
export DATALOADER_IN_ORDER="${DATALOADER_IN_ORDER:-false}"
export DATALOADER_WARMUP_BATCHES="${DATALOADER_WARMUP_BATCHES:-20}"
export SHARD_PREFETCH_DELAY_SAMPLES="${SHARD_PREFETCH_DELAY_SAMPLES:-512}"
export TRAIN_STEP_BARRIER="${TRAIN_STEP_BARRIER:-true}"
export TRAIN_STEP_BARRIER_STEPS="${TRAIN_STEP_BARRIER_STEPS:-20}"
export DATASET_SHARD_SAMPLING_RATE="${DATASET_SHARD_SAMPLING_RATE:-0.1}"
export LEARNING_RATE="${LEARNING_RATE:-1e-5}"
export USE_GRADIENT_CHECKPOINTING="${USE_GRADIENT_CHECKPOINTING:-false}"

export NNODES="${NNODES:-2}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29444}"
export GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

export HYDRA_FULL_ERROR=1
export SWANLAB_SYNC_WANDB="${SWANLAB_SYNC_WANDB:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT_NAME}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

# Cost-grouped shard scheduling. The profile must be generated for RoboTwin,
# not reused from DROID or another dataset.
export DREAMZERO_SHARD_SCHEDULE_BALANCE="${DREAMZERO_SHARD_SCHEDULE_BALANCE:-cost_grouped}"
export DREAMZERO_SHARD_COST_PROFILE="${DREAMZERO_SHARD_COST_PROFILE:-${CHECKPOINT_ROOT}/robotwin_shard_profile/shard_profile.jsonl}"
export DREAMZERO_SHARD_COST_KEY="${DREAMZERO_SHARD_COST_KEY:-source_total_bytes}"

# Keep timing on by default so early shard/cache behavior is visible in logs.
export DREAMZERO_SHARD_TIMING="${DREAMZERO_SHARD_TIMING:-1}"
export DREAMZERO_SHARD_TIMING_SAMPLE_INTERVAL="${DREAMZERO_SHARD_TIMING_SAMPLE_INTERVAL:-100}"

if command -v nvcc >/dev/null 2>&1; then
  CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
  export CUDA_HOME
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
elif [[ -d /usr/local/cuda-12.9 ]]; then
  export CUDA_HOME="/usr/local/cuda-12.9"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
else
  echo "WARN: CUDA toolkit not found; continuing without setting CUDA_HOME"
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: Python not found or not executable at ${PYTHON_BIN}"
  exit 1
fi

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "ERROR: RoboTwin train script not found at ${TRAIN_SCRIPT}"
  exit 1
fi

if [[ ! -f "${ROBOTWIN_DATA_ROOT}/meta/modality.json" ]]; then
  echo "ERROR: RoboTwin metadata missing at ${ROBOTWIN_DATA_ROOT}/meta/modality.json"
  exit 1
fi

if [[ ! -f "${DREAMZERO_SHARD_COST_PROFILE}" ]]; then
  echo "ERROR: shard cost profile missing at ${DREAMZERO_SHARD_COST_PROFILE}"
  echo "Generate a RoboTwin profile before enabling cost_grouped scheduling."
  exit 1
fi

IFS=',' read -r -a _GPU_ID_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
LOCAL_NUM_GPUS="${NUM_GPUS:-${#_GPU_ID_ARRAY[@]}}"
if [[ -z "${LOCAL_NUM_GPUS}" ]] || [[ "${LOCAL_NUM_GPUS}" -lt 1 ]]; then
  echo "ERROR: No visible GPU found"
  exit 1
fi
export NUM_GPUS="${LOCAL_NUM_GPUS}"

WORLD_GPUS=$((NNODES * LOCAL_NUM_GPUS))
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((WORLD_GPUS * PER_DEVICE_BS))}"

echo "========== RoboTwin joint cost-group A800 2-node launch =========="
echo "DREAMZERO_ROOT=${DREAMZERO_ROOT}"
echo "TRAIN_SCRIPT=${TRAIN_SCRIPT}"
echo "HOSTNAME=${HOSTNAME:-unknown}"
echo "NNODES=${NNODES}"
echo "NODE_RANK=${NODE_RANK}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NUM_GPUS(local)=${NUM_GPUS}"
echo "WORLD_GPUS(total)=${WORLD_GPUS}"
echo "PER_DEVICE_BS=${PER_DEVICE_BS}"
echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "ROBOTWIN_DATA_ROOT=${ROBOTWIN_DATA_ROOT}"
echo "WAN22_CKPT_DIR=${WAN22_CKPT_DIR}"
echo "DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS}"
echo "DATALOADER_PREFETCH_FACTOR=${DATALOADER_PREFETCH_FACTOR}"
echo "DATALOADER_IN_ORDER=${DATALOADER_IN_ORDER}"
echo "DATALOADER_WARMUP_BATCHES=${DATALOADER_WARMUP_BATCHES}"
echo "SHARD_PREFETCH_DELAY_SAMPLES=${SHARD_PREFETCH_DELAY_SAMPLES}"
echo "TRAIN_STEP_BARRIER=${TRAIN_STEP_BARRIER}"
echo "TRAIN_STEP_BARRIER_STEPS=${TRAIN_STEP_BARRIER_STEPS}"
echo "DREAMZERO_SHARD_SCHEDULE_BALANCE=${DREAMZERO_SHARD_SCHEDULE_BALANCE}"
echo "DREAMZERO_SHARD_COST_PROFILE=${DREAMZERO_SHARD_COST_PROFILE}"
echo "DREAMZERO_SHARD_COST_KEY=${DREAMZERO_SHARD_COST_KEY}"
echo "=================================================================="

exec bash "${TRAIN_SCRIPT}" "$@"
