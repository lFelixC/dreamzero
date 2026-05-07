#!/bin/bash
set -euo pipefail

# A800 2-node joint baseline with DROID exterior-view drop enabled.
# Run this script on every node in the experiment with the same MASTER_ADDR
# and MASTER_PORT, changing only NODE_RANK.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${SCRIPT_DIR}/dreamzero_env_common.sh"
dreamzero_a800_source_env

DREAMZERO_ROOT="${DREAMZERO_ROOT:-/data/dreamzero_mot}"
PYTHON_BIN="${PYTHON_BIN:-/data/dreamzero/.venv/bin/python}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/data/checkpoints/dreamzero}"
DATASET_ROOT="${DATASET_ROOT:-/data/datasets/dreamzero/droid_lerobot}"
EXPERIMENT_PY="${EXPERIMENT_PY:-${DREAMZERO_ROOT}/groot/vla/experiment/experiment.py}"

WAN22_CKPT_DIR="${WAN22_CKPT_DIR:-${CHECKPOINT_ROOT}/Wan2.2-TI2V-5B}"
IMAGE_ENCODER_DIR="${IMAGE_ENCODER_DIR:-${WAN22_CKPT_DIR}}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${WAN22_CKPT_DIR}/google/umt5-xxl}"
OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_ROOT}/dreamzero_droid_wan22_joint_drop_a800_2node}"

WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-dreamzero}"
PER_DEVICE_BS="${PER_DEVICE_BS:-16}"
MAX_STEPS="${MAX_STEPS:-30000}"
EVAL_STEPS="${EVAL_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
DEEPSPEED_CFG="${DEEPSPEED_CFG:-zero2}"
NUM_FRAMES="${NUM_FRAMES:-33}"
ACTION_HORIZON="${ACTION_HORIZON:-24}"
MAX_CHUNK_SIZE="${MAX_CHUNK_SIZE:-4}"
NUM_FRAME_PER_BLOCK="${NUM_FRAME_PER_BLOCK:-2}"
NUM_ACTION_PER_BLOCK="${NUM_ACTION_PER_BLOCK:-24}"
NUM_STATE_PER_BLOCK="${NUM_STATE_PER_BLOCK:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-4}"
DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-true}"
USE_GRADIENT_CHECKPOINTING="${USE_GRADIENT_CHECKPOINTING:-false}"
DATASET_SHARD_SAMPLING_RATE="${DATASET_SHARD_SAMPLING_RATE:-0.1}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"

# DROID views are first composed as a 320x640 canvas, then resized back to
# the Wan2.2 5B default 160x320 resolution inside the action head.
MODEL_TARGET_HEIGHT="${MODEL_TARGET_HEIGHT:-160}"
MODEL_TARGET_WIDTH="${MODEL_TARGET_WIDTH:-320}"
MODEL_FRAME_SEQLEN="${MODEL_FRAME_SEQLEN:-50}"

# Enabled by default for this experiment. Override to 1.0 to drop exactly one
# exterior view on every training sample, or to 0.0 to disable.
DROID_RANDOM_DROP_EXTERIOR_VIEW_PROB="${DROID_RANDOM_DROP_EXTERIOR_VIEW_PROB:-0.15}"

NNODES="${NNODES:-2}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29440}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

export CUDA_VISIBLE_DEVICES
export HYDRA_FULL_ERROR=1
export SWANLAB_SYNC_WANDB="${SWANLAB_SYNC_WANDB:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT_NAME}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTHONPATH="${DREAMZERO_ROOT}:${PYTHONPATH:-}"
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"

if [[ -n "${SWANLAB_API_KEY:-}" ]]; then
  export SWANLAB_API_KEY
fi

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

if [[ ! -f "${EXPERIMENT_PY}" ]]; then
  echo "ERROR: experiment.py not found at ${EXPERIMENT_PY}"
  exit 1
fi

IFS=',' read -r -a _GPU_ID_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
LOCAL_NUM_GPUS="${NUM_GPUS:-${#_GPU_ID_ARRAY[@]}}"
if [[ -z "${LOCAL_NUM_GPUS}" ]] || [[ "${LOCAL_NUM_GPUS}" -lt 1 ]]; then
  echo "ERROR: No visible GPU found"
  exit 1
fi

WORLD_GPUS=$((NNODES * LOCAL_NUM_GPUS))
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((WORLD_GPUS * PER_DEVICE_BS))}"

resolve_dataset_root() {
  local base="$1"
  local candidate

  if [[ -f "${base}/meta/modality.json" ]]; then
    echo "${base}"
    return 0
  fi

  if [[ -f "${base}/droid_lerobot/meta/modality.json" ]]; then
    echo "${base}/droid_lerobot"
    return 0
  fi

  candidate="$(find "${base}" -maxdepth 3 -path '*/meta/modality.json' 2>/dev/null | head -n 1 || true)"
  if [[ -n "${candidate}" ]]; then
    dirname "$(dirname "${candidate}")"
    return 0
  fi

  return 1
}

if ! DROID_DATA_ROOT="$(resolve_dataset_root "${DATASET_ROOT}")"; then
  echo "ERROR: Could not find meta/modality.json under ${DATASET_ROOT}"
  exit 1
fi

prepare_assets() {
  if [[ ! -d "${WAN22_CKPT_DIR}" ]] || [[ -z "$(ls -A "${WAN22_CKPT_DIR}" 2>/dev/null)" ]]; then
    echo "Downloading Wan2.2-TI2V-5B ..."
    huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir "${WAN22_CKPT_DIR}"
  fi

  if [[ ! -d "${TOKENIZER_DIR}" ]] || [[ -z "$(ls -A "${TOKENIZER_DIR}" 2>/dev/null)" ]]; then
    echo "Downloading umt5-xxl tokenizer files ..."
    huggingface-cli download google/umt5-xxl \
      --local-dir "${TOKENIZER_DIR}" \
      --include tokenizer.json tokenizer_config.json special_tokens_map.json spiece.model
  fi

  if [[ ! -f "${IMAGE_ENCODER_DIR}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" ]]; then
    local image_encoder_name="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
    local image_encoder_cache_dir="${CHECKPOINT_ROOT}/Wan2.1-I2V-14B-480P"
    local src

    for src in \
      "${image_encoder_cache_dir}/${image_encoder_name}" \
      "${CHECKPOINT_ROOT}/DreamZero-DROID/${image_encoder_name}"; do
      if [[ -f "${src}" ]]; then
        echo "Copying ${image_encoder_name} into ${IMAGE_ENCODER_DIR} ..."
        mkdir -p "${IMAGE_ENCODER_DIR}"
        cp -L "${src}" "${IMAGE_ENCODER_DIR}/${image_encoder_name}"
        return
      fi
    done

    echo "Downloading Wan2.1-I2V-14B-480P image encoder cache ..."
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "${image_encoder_cache_dir}"
    mkdir -p "${IMAGE_ENCODER_DIR}"
    cp -L "${image_encoder_cache_dir}/${image_encoder_name}" "${IMAGE_ENCODER_DIR}/${image_encoder_name}"
  fi
}

if [[ "${PREPARE_ASSETS:-true}" == "true" ]]; then
  if [[ "${NODE_RANK}" == "0" ]]; then
    prepare_assets
  else
    echo "NODE_RANK=${NODE_RANK} waiting for shared assets ..."
    until [[ -f "${WAN22_CKPT_DIR}/Wan2.2_VAE.pth" ]] \
      && [[ -d "${TOKENIZER_DIR}" ]] \
      && [[ -f "${IMAGE_ENCODER_DIR}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" ]]; do
      sleep 5
    done
  fi
fi

mkdir -p "${OUTPUT_DIR}"
cd "${DREAMZERO_ROOT}"

echo "========== joint-drop multi-node launch config =========="
echo "DREAMZERO_ROOT=${DREAMZERO_ROOT}"
echo "SCRIPT_DIR=${SCRIPT_DIR}"
echo "HOSTNAME=${HOSTNAME:-unknown}"
echo "NNODES=${NNODES}"
echo "NODE_RANK=${NODE_RANK}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NUM_GPUS(local)=${LOCAL_NUM_GPUS}"
echo "WORLD_GPUS(total)=${WORLD_GPUS}"
echo "PER_DEVICE_BS=${PER_DEVICE_BS}"
echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "DROID_DATA_ROOT=${DROID_DATA_ROOT}"
echo "WAN22_CKPT_DIR=${WAN22_CKPT_DIR}"
echo "IMAGE_ENCODER_DIR=${IMAGE_ENCODER_DIR}"
echo "TOKENIZER_DIR=${TOKENIZER_DIR}"
echo "MODEL_TARGET_HEIGHT=${MODEL_TARGET_HEIGHT}"
echo "MODEL_TARGET_WIDTH=${MODEL_TARGET_WIDTH}"
echo "MODEL_FRAME_SEQLEN=${MODEL_FRAME_SEQLEN}"
echo "DROID_RANDOM_DROP_EXTERIOR_VIEW_PROB=${DROID_RANDOM_DROP_EXTERIOR_VIEW_PROB}"
echo "USE_GRADIENT_CHECKPOINTING=${USE_GRADIENT_CHECKPOINTING}"
echo "========================================================="

TRAIN_OVERRIDES=(
  "report_to=wandb"
  "data=dreamzero/droid_relative_wan22"
  "wandb_project=${WANDB_PROJECT_NAME}"
  "train_architecture=full"
  "architecture=joint"
  "droid_random_drop_exterior_view_prob=${DROID_RANDOM_DROP_EXTERIOR_VIEW_PROB}"
  "num_frames=${NUM_FRAMES}"
  "action_horizon=${ACTION_HORIZON}"
  "num_views=3"
  "model=dreamzero/vla"
  "model/dreamzero/action_head=wan_flow_matching_action_tf_wan22"
  "action_head_cfg.config.use_gradient_checkpointing=${USE_GRADIENT_CHECKPOINTING}"
  "model/dreamzero/transform=dreamzero_cotrain"
  "num_frame_per_block=${NUM_FRAME_PER_BLOCK}"
  "num_action_per_block=${NUM_ACTION_PER_BLOCK}"
  "num_state_per_block=${NUM_STATE_PER_BLOCK}"
  "data_collator.num_frames=${NUM_FRAMES}"
  "data_collator.max_chunk_size=${MAX_CHUNK_SIZE}"
  "data_collator.num_action_per_block=${NUM_ACTION_PER_BLOCK}"
  "data_collator.num_state_per_block=${NUM_STATE_PER_BLOCK}"
  "seed=42"
  "training_args.learning_rate=${LEARNING_RATE}"
  "training_args.deepspeed=groot/vla/configs/deepspeed/${DEEPSPEED_CFG}.json"
  "output_dir=${OUTPUT_DIR}"
  "per_device_train_batch_size=${PER_DEVICE_BS}"
  "per_device_eval_batch_size=${PER_DEVICE_BS}"
  "global_batch_size=${GLOBAL_BATCH_SIZE}"
  "max_steps=${MAX_STEPS}"
  "save_steps=${SAVE_STEPS}"
  "eval_strategy=no"
  "eval_steps=${EVAL_STEPS}"
  "do_eval=false"
  "weight_decay=1e-5"
  "save_total_limit=5"
  "upload_checkpoints=false"
  "bf16=true"
  "tf32=true"
  "eval_bf16=true"
  "dataloader_pin_memory=true"
  "dataloader_num_workers=${DATALOADER_NUM_WORKERS}"
  "dataloader_persistent_workers=${DATALOADER_PERSISTENT_WORKERS}"
  "image_resolution_width=320"
  "image_resolution_height=160"
  "frame_seqlen=${MODEL_FRAME_SEQLEN}"
  "action_head_cfg.config.target_video_height=${MODEL_TARGET_HEIGHT}"
  "action_head_cfg.config.target_video_width=${MODEL_TARGET_WIDTH}"
  "save_lora_only=false"
  "max_chunk_size=${MAX_CHUNK_SIZE}"
  "save_strategy=steps"
  "droid_data_root=${DROID_DATA_ROOT}"
  "dit_version=${WAN22_CKPT_DIR}"
  "text_encoder_pretrained_path=${WAN22_CKPT_DIR}/models_t5_umt5-xxl-enc-bf16.pth"
  "image_encoder_pretrained_path=${IMAGE_ENCODER_DIR}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
  "vae_pretrained_path=${WAN22_CKPT_DIR}/Wan2.2_VAE.pth"
  "tokenizer_path=${TOKENIZER_DIR}"
  "dataset_shard_sampling_rate=${DATASET_SHARD_SAMPLING_RATE}"
  "+training_args.dataloader_prefetch_factor=${DATALOADER_PREFETCH_FACTOR}"
)

exec "${PYTHON_BIN}" -m torch.distributed.run \
  --nnodes="${NNODES}" \
  --nproc-per-node="${LOCAL_NUM_GPUS}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${EXPERIMENT_PY}" \
  "${TRAIN_OVERRIDES[@]}" \
  "$@"
