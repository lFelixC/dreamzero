#!/bin/bash
set -euo pipefail

# A800 multi-node MoT experiment: action expert sees full-video K/V.
# Run this script on every node with the same MASTER_ADDR and MASTER_PORT,
# changing only NODE_RANK.

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

EXPERIMENT_PY="${EXPERIMENT_PY:-${DREAMZERO_ROOT}/groot/vla/experiment/experiment.py}"
WAN22_CKPT_DIR="${WAN22_CKPT_DIR:-${CHECKPOINT_ROOT}/Wan2.2-TI2V-5B}"
IMAGE_ENCODER_DIR="${IMAGE_ENCODER_DIR:-${WAN22_CKPT_DIR}}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${WAN22_CKPT_DIR}/google/umt5-xxl}"
OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_ROOT}/mot_seq200_ki_off}"

# -----------------------------
# User-tunable training knobs
# -----------------------------
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-dreamzero}"
PER_DEVICE_BS="${PER_DEVICE_BS:-16}"
MAX_STEPS="${MAX_STEPS:-30000}"
EVAL_STEPS="${EVAL_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-2000}"
DEEPSPEED_CFG="${DEEPSPEED_CFG:-zero2}"
NUM_FRAMES="${NUM_FRAMES:-33}"
IMAGE_RESOLUTION_WIDTH="${IMAGE_RESOLUTION_WIDTH:-320}"
IMAGE_RESOLUTION_HEIGHT="${IMAGE_RESOLUTION_HEIGHT:-160}"
TARGET_VIDEO_HEIGHT="${TARGET_VIDEO_HEIGHT:-${TARGET_HEIGHT:-${MODEL_TARGET_HEIGHT:-320}}}"
TARGET_VIDEO_WIDTH="${TARGET_VIDEO_WIDTH:-${TARGET_WIDTH:-${MODEL_TARGET_WIDTH:-640}}}"
FRAME_SEQLEN="${FRAME_SEQLEN:-${MODEL_FRAME_SEQLEN:-${FRAMESEQ:-200}}}"
ACTION_HORIZON="${ACTION_HORIZON:-24}"
MAX_CHUNK_SIZE="${MAX_CHUNK_SIZE:-4}"
NUM_FRAME_PER_BLOCK="${NUM_FRAME_PER_BLOCK:-2}"
NUM_ACTION_PER_BLOCK="${NUM_ACTION_PER_BLOCK:-24}"
NUM_STATE_PER_BLOCK="${NUM_STATE_PER_BLOCK:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-4}"
DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-true}"
USE_GRADIENT_CHECKPOINTING="${USE_GRADIENT_CHECKPOINTING:-false}"
MOT_ACTION_VIDEO_ATTENTION="full_video"
# KI means keep the video/action experts independent for action loss.
# false: action loss gradients flow through causal video K/V into the video expert.
# true: action sees causal video K/V, but the K/V is detached for action loss.
MOT_ACTION_VIDEO_KI="${MOT_ACTION_VIDEO_KI:-${MOT_KI:-false}}"
MOT_INFERENCE_VIDEO_MODE="${MOT_INFERENCE_VIDEO_MODE:-auto}"
MOT_DECOUPLE_VIDEO_ACTION_NOISE="${MOT_DECOUPLE_VIDEO_ACTION_NOISE:-false}"
MOT_VIDEO_NOISE_BETA_ALPHA="${MOT_VIDEO_NOISE_BETA_ALPHA:-3.0}"
MOT_VIDEO_NOISE_BETA_BETA="${MOT_VIDEO_NOISE_BETA_BETA:-1.0}"
MOT_DECOUPLED_INFERENCE_VIDEO_FINAL_NOISE="${MOT_DECOUPLED_INFERENCE_VIDEO_FINAL_NOISE:-0.8}"
MOT_DECOUPLED_INFERENCE_VIDEO_REFRESH_STEPS="${MOT_DECOUPLED_INFERENCE_VIDEO_REFRESH_STEPS:-8}"
DROID_RANDOM_DROP_EXTERIOR_VIEW_PROB="${DROID_RANDOM_DROP_EXTERIOR_VIEW_PROB:-0.0}"
DATASET_SHARD_SAMPLING_RATE="${DATASET_SHARD_SAMPLING_RATE:-0.1}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"

# -----------------------------
# Multi-node envs
# -----------------------------
NNODES="${NNODES:-2}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29420}"

# -----------------------------
# Runtime env
# -----------------------------
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

echo "========== full-video MoT multi-node launch config =========="
echo "DREAMZERO_ROOT=${DREAMZERO_ROOT}"
echo "HOSTNAME=${HOSTNAME:-unknown}"
echo "NNODES=${NNODES}"
echo "NODE_RANK=${NODE_RANK}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "NUM_GPUS(local)=${LOCAL_NUM_GPUS}"
echo "WORLD_GPUS(total)=${WORLD_GPUS}"
echo "PER_DEVICE_BS=${PER_DEVICE_BS}"
echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "DROID_DATA_ROOT=${DROID_DATA_ROOT}"
echo "WAN22_CKPT_DIR=${WAN22_CKPT_DIR}"
echo "IMAGE_ENCODER_DIR=${IMAGE_ENCODER_DIR}"
echo "TOKENIZER_DIR=${TOKENIZER_DIR}"
echo "IMAGE_RESOLUTION_HEIGHT=${IMAGE_RESOLUTION_HEIGHT}"
echo "IMAGE_RESOLUTION_WIDTH=${IMAGE_RESOLUTION_WIDTH}"
echo "TARGET_VIDEO_HEIGHT=${TARGET_VIDEO_HEIGHT}"
echo "TARGET_VIDEO_WIDTH=${TARGET_VIDEO_WIDTH}"
echo "FRAME_SEQLEN=${FRAME_SEQLEN}"
echo "MOT_ACTION_VIDEO_ATTENTION=${MOT_ACTION_VIDEO_ATTENTION}"
echo "MOT_ACTION_VIDEO_KI=${MOT_ACTION_VIDEO_KI}"
echo "MOT_INFERENCE_VIDEO_MODE=${MOT_INFERENCE_VIDEO_MODE}"
echo "MOT_DECOUPLE_VIDEO_ACTION_NOISE=${MOT_DECOUPLE_VIDEO_ACTION_NOISE}"
echo "DROID_RANDOM_DROP_EXTERIOR_VIEW_PROB=${DROID_RANDOM_DROP_EXTERIOR_VIEW_PROB}"
echo "USE_GRADIENT_CHECKPOINTING=${USE_GRADIENT_CHECKPOINTING}"
echo "============================================================="

TRAIN_OVERRIDES=(
  "report_to=wandb"
  "data=dreamzero/droid_relative_wan22"
  "wandb_project=${WANDB_PROJECT_NAME}"
  "train_architecture=full"
  "architecture=mot"
  "mot_action_hidden_dim=1024"
  "mot_action_ffn_dim=4096"
  "mot_action_num_layers=null"
  "mot_action_num_heads=8"
  "mot_action_video_attention=${MOT_ACTION_VIDEO_ATTENTION}"
  "mot_action_video_ki=${MOT_ACTION_VIDEO_KI}"
  "mot_inference_video_mode=${MOT_INFERENCE_VIDEO_MODE}"
  "mot_decouple_video_action_noise=${MOT_DECOUPLE_VIDEO_ACTION_NOISE}"
  "mot_video_noise_beta_alpha=${MOT_VIDEO_NOISE_BETA_ALPHA}"
  "mot_video_noise_beta_beta=${MOT_VIDEO_NOISE_BETA_BETA}"
  "mot_decoupled_inference_video_final_noise=${MOT_DECOUPLED_INFERENCE_VIDEO_FINAL_NOISE}"
  "mot_decoupled_inference_video_refresh_steps=${MOT_DECOUPLED_INFERENCE_VIDEO_REFRESH_STEPS}"
  "droid_random_drop_exterior_view_prob=${DROID_RANDOM_DROP_EXTERIOR_VIEW_PROB}"
  "num_frames=${NUM_FRAMES}"
  "action_horizon=${ACTION_HORIZON}"
  "num_views=3"
  "model=dreamzero/vla"
  "model/dreamzero/action_head=wan_flow_matching_action_tf_wan22_mot"
  "action_head_cfg.config.use_gradient_checkpointing=${USE_GRADIENT_CHECKPOINTING}"
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
  "eval_steps=${EVAL_STEPS}"
  "do_eval=false"
  "weight_decay=1e-5"
  "save_total_limit=10"
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
