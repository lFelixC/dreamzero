#!/bin/bash
set -euo pipefail

cd /2025133806/dreamzero

# -----------------------------
# Fixed paths
# -----------------------------
DREAMZERO_ROOT="${DREAMZERO_ROOT:-/2025133806/dreamzero}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/2025133806/checkpoints/dreamzero}"
DATASET_ROOT="${DATASET_ROOT:-/2025133806/datasets/dreamzero}"

WAN22_CKPT_DIR="${CHECKPOINT_ROOT}/Wan2.2-TI2V-5B"
IMAGE_ENCODER_DIR="${CHECKPOINT_ROOT}/Wan2.1-I2V-14B-480P"
TOKENIZER_DIR="${CHECKPOINT_ROOT}/umt5-xxl"
OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_ROOT}/dreamzero_droid_wan22_5B_full_finetune_320x640_fseq200}"

PYTHON_BIN="${PYTHON_BIN:-${DREAMZERO_ROOT}/.venv/bin/python}"
EXPERIMENT_PY="${EXPERIMENT_PY:-${DREAMZERO_ROOT}/groot/vla/experiment/experiment.py}"

# -----------------------------
# User-tunable training knobs
# -----------------------------
PER_DEVICE_BS="${PER_DEVICE_BS:-8}"
DEEPSPEED_CFG="${DEEPSPEED_CFG:-zero2}"
MAX_STEPS="${MAX_STEPS:-100000}"
MODEL_TARGET_HEIGHT="${MODEL_TARGET_HEIGHT:-320}"
MODEL_TARGET_WIDTH="${MODEL_TARGET_WIDTH:-640}"
MODEL_FRAME_SEQLEN="${MODEL_FRAME_SEQLEN:-200}"

# -----------------------------
# Multi-node required envs
# -----------------------------
NNODES="${NNODES:-1}"                         # 1 / 2 / 3 / 4
NODE_RANK="${NODE_RANK:-0}"                   # 0...(NNODES-1)
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"       # current master reachable IP
MASTER_PORT="${MASTER_PORT:-29400}"           # same on all nodes

# Optional
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HYDRA_FULL_ERROR=1
export SWANLAB_SYNC_WANDB="${SWANLAB_SYNC_WANDB:-1}"
export SWANLAB_API_KEY="sOz2TBd9ZvC5K5wQDCJW3"
export WANDB_MODE="${WANDB_MODE:-offline}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

# Export this outside the script instead of hardcoding it
if [ -n "${SWANLAB_API_KEY:-}" ]; then
  export SWANLAB_API_KEY
fi

# -----------------------------
# CUDA toolkit discovery
# -----------------------------
if command -v nvcc >/dev/null 2>&1; then
  CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
  export CUDA_HOME
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
elif [ -d /usr/local/cuda-12.9 ]; then
  export CUDA_HOME="/usr/local/cuda-12.9"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
else
  echo "WARN: CUDA toolkit not found; continue without setting CUDA_HOME"
fi

# -----------------------------
# Basic checks
# -----------------------------
if [ ! -x "${PYTHON_BIN}" ]; then
  echo "ERROR: Python not found at ${PYTHON_BIN}"
  exit 1
fi

if [ ! -f "${EXPERIMENT_PY}" ]; then
  echo "ERROR: experiment.py not found at ${EXPERIMENT_PY}"
  exit 1
fi

# -----------------------------
# GPU count from CUDA_VISIBLE_DEVICES
# -----------------------------
NUM_GPUS="$("${PYTHON_BIN}" - <<'PY'
import os, torch
cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
if cvd:
    print(len([x for x in cvd.split(",") if x.strip() != ""]))
else:
    print(torch.cuda.device_count())
PY
)"

if [ -z "${NUM_GPUS}" ] || [ "${NUM_GPUS}" -lt 1 ]; then
  echo "ERROR: No visible GPU found"
  exit 1
fi

WORLD_GPUS=$((NNODES * NUM_GPUS))
GLOBAL_BATCH_SIZE=$((WORLD_GPUS * PER_DEVICE_BS))

echo "========== launch config =========="
echo "HOSTNAME=${HOSTNAME:-unknown}"
echo "NODE_RANK=${NODE_RANK}"
echo "NNODES=${NNODES}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NUM_GPUS(local)=${NUM_GPUS}"
echo "WORLD_GPUS(total)=${WORLD_GPUS}"
echo "PER_DEVICE_BS=${PER_DEVICE_BS}"
echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
echo "MODEL_TARGET_HEIGHT=${MODEL_TARGET_HEIGHT}"
echo "MODEL_TARGET_WIDTH=${MODEL_TARGET_WIDTH}"
echo "MODEL_FRAME_SEQLEN=${MODEL_FRAME_SEQLEN}"
echo "==================================="

# -----------------------------
# Dataset discovery
# -----------------------------
resolve_dataset_root() {
  local base="$1"
  local candidate

  if [ -f "${base}/meta/modality.json" ]; then
    echo "${base}"
    return 0
  fi

  if [ -f "${base}/droid_lerobot/meta/modality.json" ]; then
    echo "${base}/droid_lerobot"
    return 0
  fi

  candidate="$(find "${base}" -maxdepth 3 -path '*/meta/modality.json' 2>/dev/null | head -n 1 || true)"
  if [ -n "${candidate}" ]; then
    dirname "$(dirname "${candidate}")"
    return 0
  fi

  return 1
}

if ! DROID_DATA_ROOT="$(resolve_dataset_root "${DATASET_ROOT}")"; then
  echo "ERROR: Could not find meta/modality.json under ${DATASET_ROOT}"
  exit 1
fi

echo "Using DROID_DATA_ROOT=${DROID_DATA_ROOT}"

# -----------------------------
# Shared checkpoint prep
# rank0 downloads, others wait
# -----------------------------
prepare_assets() {
  if [ ! -d "${WAN22_CKPT_DIR}" ] || [ -z "$(ls -A "${WAN22_CKPT_DIR}" 2>/dev/null)" ]; then
    echo "Downloading Wan2.2-TI2V-5B ..."
    huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir "${WAN22_CKPT_DIR}"
  fi

  if [ ! -d "${TOKENIZER_DIR}" ] || [ -z "$(ls -A "${TOKENIZER_DIR}" 2>/dev/null)" ]; then
    echo "Downloading umt5-xxl ..."
    huggingface-cli download google/umt5-xxl --local-dir "${TOKENIZER_DIR}"
  fi

  if [ ! -f "${IMAGE_ENCODER_DIR}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" ]; then
    echo "Downloading Wan2.1-I2V-14B-480P ..."
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "${IMAGE_ENCODER_DIR}"
  fi
}

if [ "${NODE_RANK}" = "0" ]; then
  prepare_assets
else
  echo "NODE_RANK=${NODE_RANK} waiting for shared assets ..."
  until [ -f "${WAN22_CKPT_DIR}/Wan2.2_VAE.pth" ] \
    && [ -d "${TOKENIZER_DIR}" ] \
    && [ -f "${IMAGE_ENCODER_DIR}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" ]; do
    sleep 5
  done
fi

mkdir -p "${OUTPUT_DIR}"
cd "${DREAMZERO_ROOT}"

TRAIN_OVERRIDES=(
  "report_to=wandb"
  "data=dreamzero/droid_relative_wan22"
  "wandb_project=dreamzero"
  "train_architecture=full"
  "num_frames=33"
  "action_horizon=24"
  "num_views=3"
  "model=dreamzero/vla"
  "model/dreamzero/action_head=wan_flow_matching_action_tf_wan22"
  "model/dreamzero/transform=dreamzero_cotrain"
  "num_frame_per_block=2"
  "num_action_per_block=24"
  "num_state_per_block=1"
  "seed=42"
  "training_args.learning_rate=1e-5"
  "training_args.deepspeed=groot/vla/configs/deepspeed/${DEEPSPEED_CFG}.json"
  "save_steps=250"
  "training_args.warmup_ratio=0.05"
  "output_dir=${OUTPUT_DIR}"
  "per_device_train_batch_size=${PER_DEVICE_BS}"
  "global_batch_size=${GLOBAL_BATCH_SIZE}"
  "max_steps=${MAX_STEPS}"
  "weight_decay=1e-5"
  "save_total_limit=100"
  "upload_checkpoints=false"
  "bf16=true"
  "tf32=true"
  "eval_bf16=true"
  "dataloader_pin_memory=true"
  "dataloader_num_workers=4"
  "image_resolution_width=320"
  "image_resolution_height=160"
  "frame_seqlen=${MODEL_FRAME_SEQLEN}"
  "action_head_cfg.config.target_video_height=${MODEL_TARGET_HEIGHT}"
  "action_head_cfg.config.target_video_width=${MODEL_TARGET_WIDTH}"
  "save_lora_only=false"
  "max_chunk_size=4"
  "save_strategy=steps"
  "droid_data_root=${DROID_DATA_ROOT}"
  "dit_version=${WAN22_CKPT_DIR}"
  "text_encoder_pretrained_path=${WAN22_CKPT_DIR}/models_t5_umt5-xxl-enc-bf16.pth"
  "image_encoder_pretrained_path=${IMAGE_ENCODER_DIR}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
  "vae_pretrained_path=${WAN22_CKPT_DIR}/Wan2.2_VAE.pth"
  "tokenizer_path=${TOKENIZER_DIR}"
)

exec "${PYTHON_BIN}" -m torch.distributed.run \
  --nnodes="${NNODES}" \
  --nproc-per-node="${NUM_GPUS}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${EXPERIMENT_PY}" \
  "${TRAIN_OVERRIDES[@]}"
