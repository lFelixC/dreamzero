#!/bin/bash
set -euo pipefail

# Ego video-only Wan2.2 joint pretraining entrypoint.
# This path intentionally does not load an existing DreamZero robot checkpoint:
# Wan2.2 TI2V/video/text/image/vae assets are used as the base.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ARCH="${ARCH:-joint}"
if [[ "${ARCH}" != "joint" ]]; then
  echo "ERROR: ego video-only pretrain supports ARCH=joint only, got '${ARCH}'"
  exit 1
fi

export VIRTUAL_ENV="${VIRTUAL_ENV:-/data/dreamzero/.venv}"
export PYTHON_BIN="${PYTHON_BIN:-${VIRTUAL_ENV}/bin/python}"
export PATH="${VIRTUAL_ENV}/bin:/usr/local/bin:/root/.local/bin:${PATH:-}"
export DREAMZERO_ROOT="${DREAMZERO_ROOT:-/data/dreamzero_mot}"
export DATASET_ROOT="${DATASET_ROOT:-/data/datasets/dreamzero}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/data/checkpoints/dreamzero}"
export PYTHONPATH="${DREAMZERO_ROOT}:${PYTHONPATH:-}"

EXPERIMENT_PY="${EXPERIMENT_PY:-${DREAMZERO_ROOT}/groot/vla/experiment/experiment.py}"
EGO_DATA_ROOT="${EGO_DATA_ROOT:-${DATASET_ROOT}/basic_pick_place_ego_160x320}"
EGO_DATA_NAME="${EGO_DATA_NAME:-$(basename "${EGO_DATA_ROOT%/}")}"
EGO_DATA_NAME="${EGO_DATA_NAME//[^[:alnum:]_.-]/_}"
WAN22_CKPT_DIR="${WAN22_CKPT_DIR:-${CHECKPOINT_ROOT}/Wan2.2-TI2V-5B}"
IMAGE_ENCODER_DIR="${IMAGE_ENCODER_DIR:-${WAN22_CKPT_DIR}}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${WAN22_CKPT_DIR}/google/umt5-xxl}"
OUTPUT_DIR_WAS_SET="${OUTPUT_DIR+x}"
OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_ROOT}/basic_pick_place_ego_video_only_joint_wan22_10k}"

WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-dreamzero}"
PER_DEVICE_BS="${PER_DEVICE_BS:-64}"
MAX_STEPS="${MAX_STEPS:-10000}"
SAVE_STEPS="${SAVE_STEPS:-5000}"
SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
DEEPSPEED_CFG="${DEEPSPEED_CFG:-zero2}"
NUM_FRAMES="${NUM_FRAMES:-33}"
IMAGE_RESOLUTION_WIDTH="${IMAGE_RESOLUTION_WIDTH:-320}"
IMAGE_RESOLUTION_HEIGHT="${IMAGE_RESOLUTION_HEIGHT:-160}"
TARGET_VIDEO_HEIGHT="${TARGET_VIDEO_HEIGHT:-160}"
TARGET_VIDEO_WIDTH="${TARGET_VIDEO_WIDTH:-320}"
FRAME_SEQLEN="${FRAME_SEQLEN:-50}"
ACTION_HORIZON="${ACTION_HORIZON:-24}"
MAX_CHUNK_SIZE="${MAX_CHUNK_SIZE:-4}"
NUM_FRAME_PER_BLOCK="${NUM_FRAME_PER_BLOCK:-2}"
NUM_ACTION_PER_BLOCK="${NUM_ACTION_PER_BLOCK:-24}"
NUM_STATE_PER_BLOCK="${NUM_STATE_PER_BLOCK:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-4}"
DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-true}"
DATASET_SHARD_SAMPLING_RATE="${DATASET_SHARD_SAMPLING_RATE:-0.1}"
DATASET_SHARD_SAMPLING_STRATEGY="${DATASET_SHARD_SAMPLING_STRATEGY:-random}"
DATASET_SHARD_SAMPLING_BLOCK_SIZE="${DATASET_SHARD_SAMPLING_BLOCK_SIZE:-64}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
USE_GRADIENT_CHECKPOINTING="${USE_GRADIENT_CHECKPOINTING:-true}"
ACTIVATION_CHECKPOINTING_POLICY="${ACTIVATION_CHECKPOINTING_POLICY:-off}"
EPISODE_FILTER_PATH="${EPISODE_FILTER_PATH:-null}"

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29436}"

if [[ "${SMOKE_TEST:-0}" == "1" ]]; then
  NNODES=1
  NODE_RANK=0
  MASTER_ADDR="127.0.0.1"
  if [[ -n "${SMOKE_GPU_IDS:-}" ]]; then
    GPU_IDS="${SMOKE_GPU_IDS}"
  elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    GPU_IDS="${CUDA_VISIBLE_DEVICES}"
  else
    GPU_IDS="${GPU_IDS:-4,5,6,7}"
  fi
  if [[ "${SMOKE_FORCE_SINGLE_GPU:-false}" == "true" && "${GPU_IDS}" == *,* ]]; then
    GPU_IDS="${GPU_IDS%%,*}"
  fi
  PER_DEVICE_BS="${SMOKE_PER_DEVICE_BS:-1}"
  MAX_STEPS="${SMOKE_MAX_STEPS:-2}"
  SAVE_STEPS="${SMOKE_SAVE_STEPS:-1}"
  SAVE_STRATEGY="${SMOKE_SAVE_STRATEGY:-no}"
  SAVE_TOTAL_LIMIT="${SMOKE_SAVE_TOTAL_LIMIT:-5}"
  DATALOADER_NUM_WORKERS="${SMOKE_DATALOADER_NUM_WORKERS:-1}"
  DATALOADER_PREFETCH_FACTOR="${SMOKE_DATALOADER_PREFETCH_FACTOR:-2}"
  DATALOADER_PERSISTENT_WORKERS="${SMOKE_DATALOADER_PERSISTENT_WORKERS:-false}"
  DATASET_SHARD_SAMPLING_RATE="${SMOKE_DATASET_SHARD_SAMPLING_RATE:-1.0}"
  DATASET_SHARD_SAMPLING_STRATEGY="${SMOKE_DATASET_SHARD_SAMPLING_STRATEGY:-${DATASET_SHARD_SAMPLING_STRATEGY}}"
  DATASET_SHARD_SAMPLING_BLOCK_SIZE="${SMOKE_DATASET_SHARD_SAMPLING_BLOCK_SIZE:-${DATASET_SHARD_SAMPLING_BLOCK_SIZE}}"
  if [[ -z "${OUTPUT_DIR_WAS_SET}" ]]; then
    OUTPUT_DIR="${CHECKPOINT_ROOT}/basic_pick_place_ego_video_only_joint_wan22_smoke"
  fi
  if [[ "${EPISODE_FILTER_PATH}" == "null" && -f "${EGO_DATA_ROOT}/meta/smoke_episode_filter.json" ]]; then
    EPISODE_FILTER_PATH="${EGO_DATA_ROOT}/meta/smoke_episode_filter.json"
  fi
  export DREAMZERO_SKIP_FINAL_SAVE="${DREAMZERO_SKIP_FINAL_SAVE:-1}"
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  GPU_IDS="${CUDA_VISIBLE_DEVICES}"
else
  GPU_IDS="${GPU_IDS:-4,5,6,7}"
  export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
fi

export HYDRA_FULL_ERROR=1
export SWANLAB_SYNC_WANDB="${SWANLAB_SYNC_WANDB:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT_NAME}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

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
if [[ ! -f "${EGO_DATA_ROOT}/meta/modality.json" ]]; then
  echo "ERROR: ego metadata missing at ${EGO_DATA_ROOT}/meta/modality.json"
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
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((WORLD_GPUS * PER_DEVICE_BS))}"

download_hf() {
  if command -v hf >/dev/null 2>&1; then
    hf download "$@"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$@"
  else
    echo "ERROR: neither hf nor huggingface-cli is available"
    exit 1
  fi
}

prepare_assets() {
  if [[ ! -d "${WAN22_CKPT_DIR}" ]] || [[ -z "$(ls -A "${WAN22_CKPT_DIR}" 2>/dev/null)" ]]; then
    echo "Downloading Wan2.2-TI2V-5B to ${WAN22_CKPT_DIR} ..."
    download_hf Wan-AI/Wan2.2-TI2V-5B --local-dir "${WAN22_CKPT_DIR}"
  fi

  if [[ ! -d "${TOKENIZER_DIR}" ]] || [[ -z "$(ls -A "${TOKENIZER_DIR}" 2>/dev/null)" ]]; then
    echo "Downloading umt5-xxl tokenizer files to ${TOKENIZER_DIR} ..."
    download_hf google/umt5-xxl \
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
    download_hf Wan-AI/Wan2.1-I2V-14B-480P --local-dir "${image_encoder_cache_dir}"
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

echo "========== Ego video-only Wan2.2 joint launch config =========="
echo "ARCH=${ARCH}"
echo "DREAMZERO_ROOT=${DREAMZERO_ROOT}"
echo "SCRIPT_DIR=${SCRIPT_DIR}"
echo "HOSTNAME=${HOSTNAME:-unknown}"
echo "NNODES=${NNODES}"
echo "NODE_RANK=${NODE_RANK}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "EGO_DATA_ROOT=${EGO_DATA_ROOT}"
echo "EGO_DATA_NAME=${EGO_DATA_NAME}"
echo "WAN22_CKPT_DIR=${WAN22_CKPT_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "NUM_GPUS(local)=${NUM_GPUS}"
echo "WORLD_GPUS(total)=${WORLD_GPUS}"
echo "PER_DEVICE_BS=${PER_DEVICE_BS}"
echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
echo "MAX_STEPS=${MAX_STEPS}"
echo "SAVE_STEPS=${SAVE_STEPS}"
echo "SAVE_STRATEGY=${SAVE_STRATEGY}"
echo "LEARNING_RATE=${LEARNING_RATE}"
echo "DEEPSPEED_CFG=${DEEPSPEED_CFG}"
echo "NUM_FRAMES=${NUM_FRAMES}"
echo "IMAGE_RESOLUTION_HEIGHT=${IMAGE_RESOLUTION_HEIGHT}"
echo "IMAGE_RESOLUTION_WIDTH=${IMAGE_RESOLUTION_WIDTH}"
echo "TARGET_VIDEO_HEIGHT=${TARGET_VIDEO_HEIGHT}"
echo "TARGET_VIDEO_WIDTH=${TARGET_VIDEO_WIDTH}"
echo "FRAME_SEQLEN=${FRAME_SEQLEN}"
echo "ACTION_HORIZON=${ACTION_HORIZON}"
echo "MAX_CHUNK_SIZE=${MAX_CHUNK_SIZE}"
echo "NUM_FRAME_PER_BLOCK=${NUM_FRAME_PER_BLOCK}"
echo "NUM_ACTION_PER_BLOCK=${NUM_ACTION_PER_BLOCK}"
echo "NUM_STATE_PER_BLOCK=${NUM_STATE_PER_BLOCK}"
echo "DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS}"
echo "DATALOADER_PREFETCH_FACTOR=${DATALOADER_PREFETCH_FACTOR}"
echo "DATALOADER_PERSISTENT_WORKERS=${DATALOADER_PERSISTENT_WORKERS}"
echo "DATASET_SHARD_SAMPLING_RATE=${DATASET_SHARD_SAMPLING_RATE}"
echo "DATASET_SHARD_SAMPLING_STRATEGY=${DATASET_SHARD_SAMPLING_STRATEGY}"
echo "DATASET_SHARD_SAMPLING_BLOCK_SIZE=${DATASET_SHARD_SAMPLING_BLOCK_SIZE}"
echo "USE_GRADIENT_CHECKPOINTING=${USE_GRADIENT_CHECKPOINTING}"
echo "ACTIVATION_CHECKPOINTING_POLICY=${ACTIVATION_CHECKPOINTING_POLICY}"
echo "SMOKE_TEST=${SMOKE_TEST:-0}"
echo "DREAMZERO_SKIP_FINAL_SAVE=${DREAMZERO_SKIP_FINAL_SAVE:-<unset>}"
echo "EPISODE_FILTER_PATH=${EPISODE_FILTER_PATH}"
echo "==============================================================="

TRAIN_OVERRIDES=(
  "report_to=wandb"
  "data=dreamzero/basic_pick_place_ego_video_only_wan22"
  "wandb_project=${WANDB_PROJECT_NAME}"
  "train_architecture=full"
  "architecture=${ARCH}"
  "video_only=true"
  "num_frames=${NUM_FRAMES}"
  "action_horizon=${ACTION_HORIZON}"
  "num_views=1"
  "model=dreamzero/vla"
  "model/dreamzero/action_head=wan_flow_matching_action_tf_wan22"
  "action_head_cfg.config.use_gradient_checkpointing=${USE_GRADIENT_CHECKPOINTING}"
  "action_head_cfg.config.activation_checkpointing_policy=${ACTIVATION_CHECKPOINTING_POLICY}"
  "frame_seqlen=${FRAME_SEQLEN}"
  "action_head_cfg.config.target_video_height=${TARGET_VIDEO_HEIGHT}"
  "action_head_cfg.config.target_video_width=${TARGET_VIDEO_WIDTH}"
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
  "do_eval=false"
  "weight_decay=1e-5"
  "save_total_limit=${SAVE_TOTAL_LIMIT}"
  "upload_checkpoints=false"
  "bf16=true"
  "tf32=true"
  "eval_bf16=true"
  "dataloader_pin_memory=true"
  "dataloader_num_workers=${DATALOADER_NUM_WORKERS}"
  "dataloader_persistent_workers=${DATALOADER_PERSISTENT_WORKERS}"
  "image_resolution_width=${IMAGE_RESOLUTION_WIDTH}"
  "image_resolution_height=${IMAGE_RESOLUTION_HEIGHT}"
  "save_lora_only=false"
  "max_chunk_size=${MAX_CHUNK_SIZE}"
  "save_strategy=${SAVE_STRATEGY}"
  "ego_data_root=${EGO_DATA_ROOT}"
  "episode_filter_path=${EPISODE_FILTER_PATH}"
  "dit_version=${WAN22_CKPT_DIR}"
  "text_encoder_pretrained_path=${WAN22_CKPT_DIR}/models_t5_umt5-xxl-enc-bf16.pth"
  "image_encoder_pretrained_path=${IMAGE_ENCODER_DIR}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
  "vae_pretrained_path=${WAN22_CKPT_DIR}/Wan2.2_VAE.pth"
  "tokenizer_path=${TOKENIZER_DIR}"
  "dataset_shard_sampling_rate=${DATASET_SHARD_SAMPLING_RATE}"
  "dataset_shard_sampling_strategy=${DATASET_SHARD_SAMPLING_STRATEGY}"
  "dataset_shard_sampling_block_size=${DATASET_SHARD_SAMPLING_BLOCK_SIZE}"
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
