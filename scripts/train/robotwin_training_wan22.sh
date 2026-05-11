#!/bin/bash
set -euo pipefail

# RoboTwin Wan2.2 training entrypoint.
# Switch architectures with ARCH=joint or ARCH=mot. This script uses its own
# robotwin_* config and never rewrites DROID paths.

ARCH="${ARCH:-joint}"
if [[ "${ARCH}" != "joint" && "${ARCH}" != "mot" ]]; then
  echo "ERROR: ARCH must be joint or mot, got '${ARCH}'"
  exit 1
fi

export VIRTUAL_ENV="${VIRTUAL_ENV:-/data/dreamzero/.venv}"
export PYTHON_BIN="${PYTHON_BIN:-${VIRTUAL_ENV}/bin/python}"
export PATH="${VIRTUAL_ENV}/bin:/usr/local/bin:/root/.local/bin:${PATH:-}"
export DREAMZERO_ROOT="${DREAMZERO_ROOT:-/data/dreamzero_mot}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/data/checkpoints/dreamzero}"
export PYTHONPATH="${DREAMZERO_ROOT}:${PYTHONPATH:-}"

EXPERIMENT_PY="${EXPERIMENT_PY:-${DREAMZERO_ROOT}/groot/vla/experiment/experiment.py}"
ROBOTWIN_DATA_ROOT="${ROBOTWIN_DATA_ROOT:-/data/datasets/dreamzero/robotwin_unified_dreamzero}"
WAN22_CKPT_DIR="${WAN22_CKPT_DIR:-${CHECKPOINT_ROOT}/Wan2.2-TI2V-5B}"
IMAGE_ENCODER_DIR="${IMAGE_ENCODER_DIR:-${WAN22_CKPT_DIR}}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${WAN22_CKPT_DIR}/google/umt5-xxl}"
OUTPUT_DIR_WAS_SET="${OUTPUT_DIR+x}"
OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_ROOT}/dreamzero_robotwin_wan22_${ARCH}}"

WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-dreamzero}"
PER_DEVICE_BS="${PER_DEVICE_BS:-32}"
MAX_STEPS="${MAX_STEPS:-60000}"
SAVE_STEPS="${SAVE_STEPS:-5000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
DEEPSPEED_CFG="${DEEPSPEED_CFG:-zero2}"
NUM_FRAMES="${NUM_FRAMES:-33}"
ACTION_HORIZON="${ACTION_HORIZON:-24}"
MAX_CHUNK_SIZE="${MAX_CHUNK_SIZE:-4}"
NUM_FRAME_PER_BLOCK="${NUM_FRAME_PER_BLOCK:-2}"
NUM_ACTION_PER_BLOCK="${NUM_ACTION_PER_BLOCK:-24}"
NUM_STATE_PER_BLOCK="${NUM_STATE_PER_BLOCK:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-1}"
DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-true}"
DATASET_SHARD_SAMPLING_RATE="${DATASET_SHARD_SAMPLING_RATE:-0.1}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
USE_GRADIENT_CHECKPOINTING="${USE_GRADIENT_CHECKPOINTING:-false}"
EPISODE_FILTER_PATH="${EPISODE_FILTER_PATH:-null}"

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29431}"

if [[ "${SMOKE_TEST:-0}" == "1" ]]; then
  NNODES=1
  NODE_RANK=0
  MASTER_ADDR="127.0.0.1"
  GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0}}"
  if [[ "${GPU_IDS}" == *,* ]]; then
    GPU_IDS="${GPU_IDS%%,*}"
  fi
  PER_DEVICE_BS="${SMOKE_PER_DEVICE_BS:-1}"
  MAX_STEPS="${SMOKE_MAX_STEPS:-2}"
  SAVE_STEPS="${SMOKE_SAVE_STEPS:-1}"
  SAVE_TOTAL_LIMIT="${SMOKE_SAVE_TOTAL_LIMIT:-5}"
  DATALOADER_NUM_WORKERS="${SMOKE_DATALOADER_NUM_WORKERS:-1}"
  DATALOADER_PREFETCH_FACTOR="${SMOKE_DATALOADER_PREFETCH_FACTOR:-2}"
  DATALOADER_PERSISTENT_WORKERS="${SMOKE_DATALOADER_PERSISTENT_WORKERS:-false}"
  DATASET_SHARD_SAMPLING_RATE="${SMOKE_DATASET_SHARD_SAMPLING_RATE:-1.0}"
  if [[ -z "${OUTPUT_DIR_WAS_SET}" ]]; then
    OUTPUT_DIR="${CHECKPOINT_ROOT}/dreamzero_robotwin_wan22_${ARCH}_smoke"
  fi
  if [[ "${EPISODE_FILTER_PATH}" == "null" && -f "${ROBOTWIN_DATA_ROOT}/meta/smoke_episode_filter.json" ]]; then
    EPISODE_FILTER_PATH="${ROBOTWIN_DATA_ROOT}/meta/smoke_episode_filter.json"
  fi
fi

GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
if [[ -n "${GPU_IDS}" ]]; then
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
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: Python not found or not executable at ${PYTHON_BIN}"
  exit 1
fi
if [[ ! -f "${EXPERIMENT_PY}" ]]; then
  echo "ERROR: experiment.py not found at ${EXPERIMENT_PY}"
  exit 1
fi
if [[ ! -f "${ROBOTWIN_DATA_ROOT}/meta/modality.json" ]]; then
  echo "ERROR: RoboTwin metadata missing at ${ROBOTWIN_DATA_ROOT}/meta/modality.json"
  echo "Run scripts/data/convert_robotwin_v3_to_dreamzero.py and scripts/data/convert_lerobot_to_gear.py first."
  exit 1
fi

if [[ -n "${NUM_GPUS:-}" ]]; then
  LOCAL_NUM_GPUS="${NUM_GPUS}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a _GPU_ID_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
  LOCAL_NUM_GPUS="${#_GPU_ID_ARRAY[@]}"
else
  LOCAL_NUM_GPUS="$("${PYTHON_BIN}" - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
fi
if [[ -z "${LOCAL_NUM_GPUS}" ]] || [[ "${LOCAL_NUM_GPUS}" -lt 1 ]]; then
  echo "ERROR: No visible GPU found"
  exit 1
fi

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

if [[ "${ARCH}" == "mot" ]]; then
  ACTION_HEAD_CONFIG="wan_flow_matching_action_tf_wan22_mot"
else
  ACTION_HEAD_CONFIG="wan_flow_matching_action_tf_wan22"
fi

mkdir -p "${OUTPUT_DIR}"
cd "${DREAMZERO_ROOT}"

echo "========== RoboTwin Wan2.2 launch config =========="
echo "ARCH=${ARCH}"
echo "DREAMZERO_ROOT=${DREAMZERO_ROOT}"
echo "ROBOTWIN_DATA_ROOT=${ROBOTWIN_DATA_ROOT}"
echo "WAN22_CKPT_DIR=${WAN22_CKPT_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "NUM_GPUS(local)=${LOCAL_NUM_GPUS}"
echo "NNODES=${NNODES}"
echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
echo "DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS}"
echo "DATALOADER_PREFETCH_FACTOR=${DATALOADER_PREFETCH_FACTOR}"
echo "SMOKE_TEST=${SMOKE_TEST:-0}"
echo "EPISODE_FILTER_PATH=${EPISODE_FILTER_PATH}"
echo "==================================================="

TRAIN_OVERRIDES=(
  "report_to=wandb"
  "data=dreamzero/robotwin_aloha_relative_wan22"
  "wandb_project=${WANDB_PROJECT_NAME}"
  "train_architecture=full"
  "architecture=${ARCH}"
  "num_frames=${NUM_FRAMES}"
  "action_horizon=${ACTION_HORIZON}"
  "num_views=3"
  "model=dreamzero/vla"
  "model/dreamzero/action_head=${ACTION_HEAD_CONFIG}"
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
  "image_resolution_width=320"
  "image_resolution_height=160"
  "save_lora_only=false"
  "max_chunk_size=${MAX_CHUNK_SIZE}"
  "save_strategy=steps"
  "robotwin_data_root=${ROBOTWIN_DATA_ROOT}"
  "episode_filter_path=${EPISODE_FILTER_PATH}"
  "dit_version=${WAN22_CKPT_DIR}"
  "text_encoder_pretrained_path=${WAN22_CKPT_DIR}/models_t5_umt5-xxl-enc-bf16.pth"
  "image_encoder_pretrained_path=${IMAGE_ENCODER_DIR}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
  "vae_pretrained_path=${WAN22_CKPT_DIR}/Wan2.2_VAE.pth"
  "tokenizer_path=${TOKENIZER_DIR}"
  "dataset_shard_sampling_rate=${DATASET_SHARD_SAMPLING_RATE}"
  "+training_args.dataloader_prefetch_factor=${DATALOADER_PREFETCH_FACTOR}"
)

if [[ "${ARCH}" == "mot" ]]; then
  TRAIN_OVERRIDES+=(
    "mot_action_hidden_dim=${MOT_ACTION_HIDDEN_DIM:-1024}"
    "mot_action_ffn_dim=${MOT_ACTION_FFN_DIM:-4096}"
    "mot_action_num_layers=${MOT_ACTION_NUM_LAYERS:-null}"
    "mot_action_num_heads=${MOT_ACTION_NUM_HEADS:-8}"
    "mot_action_video_attention=${MOT_ACTION_VIDEO_ATTENTION:-first_frame}"
    "mot_action_video_ki=${MOT_ACTION_VIDEO_KI:-false}"
    "mot_inference_video_mode=${MOT_INFERENCE_VIDEO_MODE:-auto}"
    "mot_decouple_video_action_noise=${MOT_DECOUPLE_VIDEO_ACTION_NOISE:-false}"
    "mot_video_noise_beta_alpha=${MOT_VIDEO_NOISE_BETA_ALPHA:-3.0}"
    "mot_video_noise_beta_beta=${MOT_VIDEO_NOISE_BETA_BETA:-1.0}"
    "mot_decoupled_inference_video_final_noise=${MOT_DECOUPLED_INFERENCE_VIDEO_FINAL_NOISE:-0.8}"
    "mot_decoupled_inference_video_refresh_steps=${MOT_DECOUPLED_INFERENCE_VIDEO_REFRESH_STEPS:-8}"
  )
fi

exec "${PYTHON_BIN}" -m torch.distributed.run \
  --nnodes="${NNODES}" \
  --nproc-per-node="${LOCAL_NUM_GPUS}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${EXPERIMENT_PY}" \
  "${TRAIN_OVERRIDES[@]}" \
  "$@"
