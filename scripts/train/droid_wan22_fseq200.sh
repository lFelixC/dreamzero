#!/bin/bash
set -euo pipefail

# Canonical DROID Wan2.2 launcher for the 320x640 composite / 200 token setup.
# Use ARCH=joint or ARCH=mot, or call the small wrapper scripts next to this file.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DREAMZERO_ROOT="${DREAMZERO_ROOT:-${REPO_ROOT}}"
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "${DREAMZERO_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${DREAMZERO_ROOT}/.venv/bin/python"
  elif command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3.11)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    PYTHON_BIN="$(command -v python)"
  fi
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: Python not found or not executable: ${PYTHON_BIN}"
  exit 1
fi

if [[ -d /data/checkpoints/dreamzero ]]; then
  DEFAULT_CHECKPOINT_ROOT=/data/checkpoints/dreamzero
else
  DEFAULT_CHECKPOINT_ROOT="${DREAMZERO_ROOT}/checkpoints"
fi

if [[ -d /data/datasets/dreamzero ]]; then
  DEFAULT_DATASET_ROOT=/data/datasets/dreamzero
else
  DEFAULT_DATASET_ROOT="${DREAMZERO_ROOT}/data/droid_lerobot"
fi

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${DEFAULT_CHECKPOINT_ROOT}}"
DATASET_ROOT="${DATASET_ROOT:-${DEFAULT_DATASET_ROOT}}"
EXPERIMENT_PY="${EXPERIMENT_PY:-${DREAMZERO_ROOT}/groot/vla/experiment/experiment.py}"

ARCH="${ARCH:-joint}"
case "${ARCH}" in
  joint)
    ACTION_HEAD_CONFIG="wan_flow_matching_action_tf_wan22"
    DEFAULT_MASTER_PORT=29410
    ;;
  mot)
    ACTION_HEAD_CONFIG="wan_flow_matching_action_tf_wan22_mot"
    DEFAULT_MASTER_PORT=29420
    ;;
  *)
    echo "ERROR: ARCH must be one of: joint, mot. Got: ${ARCH}"
    exit 1
    ;;
esac

WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-dreamzero}"
REPORT_TO="${REPORT_TO:-wandb}"
TRAIN_ARCHITECTURE="${TRAIN_ARCHITECTURE:-full}"
WAN22_CKPT_DIR="${WAN22_CKPT_DIR:-${CHECKPOINT_ROOT}/Wan2.2-TI2V-5B}"
IMAGE_ENCODER_DIR="${IMAGE_ENCODER_DIR:-${WAN22_CKPT_DIR}}"
if [[ -d "${WAN22_CKPT_DIR}/google/umt5-xxl" ]]; then
  DEFAULT_TOKENIZER_DIR="${WAN22_CKPT_DIR}/google/umt5-xxl"
else
  DEFAULT_TOKENIZER_DIR="${CHECKPOINT_ROOT}/umt5-xxl"
fi
TOKENIZER_DIR="${TOKENIZER_DIR:-${DEFAULT_TOKENIZER_DIR}}"
WAN22_DIT_READY_MARKER="${WAN22_DIT_READY_MARKER:-${WAN22_CKPT_DIR}/.dreamzero_dit_ready}"

OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_ROOT}/dreamzero_droid_wan22_${ARCH}_fseq200_${TRAIN_ARCHITECTURE}}"

PER_DEVICE_BS="${PER_DEVICE_BS:-1}"
MAX_STEPS="${MAX_STEPS:-50000}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"
DEEPSPEED_CFG="${DEEPSPEED_CFG:-zero2_offload}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
DYNAMICS_LOSS_WEIGHT="${DYNAMICS_LOSS_WEIGHT:-1.0}"
ACTION_LOSS_WEIGHT="${ACTION_LOSS_WEIGHT:-1.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
EVAL_STRATEGY="${EVAL_STRATEGY:-no}"
EVAL_STEPS="${EVAL_STEPS:-}"
SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"

NUM_FRAMES="${NUM_FRAMES:-33}"
ACTION_HORIZON="${ACTION_HORIZON:-24}"
MAX_CHUNK_SIZE="${MAX_CHUNK_SIZE:-4}"
NUM_FRAME_PER_BLOCK="${NUM_FRAME_PER_BLOCK:-2}"
NUM_ACTION_PER_BLOCK="${NUM_ACTION_PER_BLOCK:-24}"
NUM_STATE_PER_BLOCK="${NUM_STATE_PER_BLOCK:-1}"

# Each camera frame remains 160x320. DreamTransform composes the DROID grid
# into 320x640, then the action head keeps that composite resolution.
IMAGE_RESOLUTION_WIDTH="${IMAGE_RESOLUTION_WIDTH:-320}"
IMAGE_RESOLUTION_HEIGHT="${IMAGE_RESOLUTION_HEIGHT:-160}"
TARGET_VIDEO_HEIGHT="${TARGET_VIDEO_HEIGHT:-${MODEL_TARGET_HEIGHT:-320}}"
TARGET_VIDEO_WIDTH="${TARGET_VIDEO_WIDTH:-${MODEL_TARGET_WIDTH:-640}}"
FRAME_SEQLEN="${FRAME_SEQLEN:-${MODEL_FRAME_SEQLEN:-200}}"

DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
DATALOADER_PIN_MEMORY="${DATALOADER_PIN_MEMORY:-true}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-4}"
DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-true}"
USE_GRADIENT_CHECKPOINTING="${USE_GRADIENT_CHECKPOINTING:-false}"
DATASET_SHARD_SAMPLING_RATE="${DATASET_SHARD_SAMPLING_RATE:-0.1}"
DATASET_SHARD_SAMPLING_STRATEGY="${DATASET_SHARD_SAMPLING_STRATEGY:-random}"
DATASET_SHARD_SAMPLING_BLOCK_SIZE="${DATASET_SHARD_SAMPLING_BLOCK_SIZE:-64}"
DROID_RANDOM_DROP_EXTERIOR_VIEW_PROB="${DROID_RANDOM_DROP_EXTERIOR_VIEW_PROB:-0.0}"

MOT_ACTION_VIDEO_ATTENTION="${MOT_ACTION_VIDEO_ATTENTION:-full_video}"
MOT_ACTION_VIDEO_KI="${MOT_ACTION_VIDEO_KI:-${MOT_KI:-false}}"
MOT_INFERENCE_VIDEO_MODE="${MOT_INFERENCE_VIDEO_MODE:-auto}"
MOT_DECOUPLE_VIDEO_ACTION_NOISE="${MOT_DECOUPLE_VIDEO_ACTION_NOISE:-false}"
MOT_VIDEO_NOISE_BETA_ALPHA="${MOT_VIDEO_NOISE_BETA_ALPHA:-3.0}"
MOT_VIDEO_NOISE_BETA_BETA="${MOT_VIDEO_NOISE_BETA_BETA:-1.0}"
MOT_DECOUPLED_INFERENCE_VIDEO_FINAL_NOISE="${MOT_DECOUPLED_INFERENCE_VIDEO_FINAL_NOISE:-0.8}"
MOT_DECOUPLED_INFERENCE_VIDEO_REFRESH_STEPS="${MOT_DECOUPLED_INFERENCE_VIDEO_REFRESH_STEPS:-8}"

if [[ -z "${SAVE_LORA_ONLY+x}" ]]; then
  if [[ "${TRAIN_ARCHITECTURE}" == "lora" ]]; then
    SAVE_LORA_ONLY=true
  else
    SAVE_LORA_ONLY=false
  fi
fi

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-${DEFAULT_MASTER_PORT}}"

GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-}}"
if [[ -n "${GPU_IDS}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
fi

export HYDRA_FULL_ERROR=1
export SWANLAB_SYNC_WANDB="${SWANLAB_SYNC_WANDB:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT_NAME}"
export PYTHONPATH="${DREAMZERO_ROOT}:${PYTHONPATH:-}"
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
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

quote_hydra_interval() {
  local value="$1"
  case "${value}" in
    no|\"no\"|\'no\')
      printf '"no"'
      ;;
    *)
      printf '%s' "${value}"
      ;;
  esac
}

dit_weights_ready() {
  local index_path="${WAN22_CKPT_DIR}/diffusion_pytorch_model.safetensors.index.json"
  local single_path="${WAN22_CKPT_DIR}/diffusion_pytorch_model.safetensors"
  local shard
  local shards=()

  if [[ -f "${single_path}" ]]; then
    return 0
  fi

  if [[ ! -f "${index_path}" ]]; then
    return 1
  fi

  mapfile -t shards < <("${PYTHON_BIN}" - "${index_path}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    index = json.load(f)

for shard in sorted(set(index.get("weight_map", {}).values())):
    print(shard)
PY
)

  if [[ "${#shards[@]}" -eq 0 ]]; then
    return 1
  fi

  for shard in "${shards[@]}"; do
    [[ -f "${WAN22_CKPT_DIR}/${shard}" ]] || return 1
  done
}

tokenizer_assets_ready() {
  local file

  for file in tokenizer.json tokenizer_config.json special_tokens_map.json spiece.model; do
    [[ -f "${TOKENIZER_DIR}/${file}" ]] || return 1
  done
}

wan22_model_assets_ready() {
  [[ -f "${WAN22_CKPT_DIR}/Wan2.2_VAE.pth" ]] \
    && [[ -f "${WAN22_CKPT_DIR}/models_t5_umt5-xxl-enc-bf16.pth" ]] \
    && dit_weights_ready
}

required_assets_ready() {
  [[ -f "${WAN22_DIT_READY_MARKER}" ]] \
    && wan22_model_assets_ready \
    && tokenizer_assets_ready \
    && [[ -f "${IMAGE_ENCODER_DIR}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" ]]
}

verify_required_assets() {
  local missing=()
  local file
  local index_path="${WAN22_CKPT_DIR}/diffusion_pytorch_model.safetensors.index.json"
  local single_path="${WAN22_CKPT_DIR}/diffusion_pytorch_model.safetensors"
  local shard
  local shards=()

  [[ -f "${WAN22_CKPT_DIR}/Wan2.2_VAE.pth" ]] || missing+=("${WAN22_CKPT_DIR}/Wan2.2_VAE.pth")
  [[ -f "${WAN22_CKPT_DIR}/models_t5_umt5-xxl-enc-bf16.pth" ]] || missing+=("${WAN22_CKPT_DIR}/models_t5_umt5-xxl-enc-bf16.pth")
  [[ -f "${IMAGE_ENCODER_DIR}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" ]] || missing+=("${IMAGE_ENCODER_DIR}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")

  for file in tokenizer.json tokenizer_config.json special_tokens_map.json spiece.model; do
    [[ -f "${TOKENIZER_DIR}/${file}" ]] || missing+=("${TOKENIZER_DIR}/${file}")
  done

  if [[ -f "${single_path}" ]]; then
    :
  elif [[ -f "${index_path}" ]]; then
    mapfile -t shards < <("${PYTHON_BIN}" - "${index_path}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    index = json.load(f)

for shard in sorted(set(index.get("weight_map", {}).values())):
    print(shard)
PY
)
    if [[ "${#shards[@]}" -eq 0 ]]; then
      missing+=("${index_path}: weight_map has no shards")
    fi
    for shard in "${shards[@]}"; do
      [[ -f "${WAN22_CKPT_DIR}/${shard}" ]] || missing+=("${WAN22_CKPT_DIR}/${shard}")
    done
  else
    missing+=("${single_path} or ${index_path}")
  fi

  if [[ "${#missing[@]}" -gt 0 ]]; then
    printf 'ERROR: Missing required asset: %s\n' "${missing[@]}" >&2
    return 1
  fi
}

if [[ -n "${DROID_DATA_ROOT:-}" ]]; then
  if [[ ! -f "${DROID_DATA_ROOT}/meta/modality.json" ]]; then
    echo "ERROR: DROID_DATA_ROOT does not contain meta/modality.json: ${DROID_DATA_ROOT}"
    exit 1
  fi
else
  if ! DROID_DATA_ROOT="$(resolve_dataset_root "${DATASET_ROOT}")"; then
    echo "ERROR: Could not find DROID meta/modality.json under ${DATASET_ROOT}"
    exit 1
  fi
fi

prepare_assets() {
  if ! wan22_model_assets_ready; then
    echo "Downloading Wan2.2-TI2V-5B to ${WAN22_CKPT_DIR} ..."
    huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir "${WAN22_CKPT_DIR}"
  fi

  if ! dit_weights_ready; then
    echo "ERROR: Wan2.2 DiT weights are incomplete under ${WAN22_CKPT_DIR}"
    exit 1
  fi
  mkdir -p "$(dirname "${WAN22_DIT_READY_MARKER}")"
  touch "${WAN22_DIT_READY_MARKER}"

  if ! tokenizer_assets_ready; then
    echo "Downloading google/umt5-xxl tokenizer to ${TOKENIZER_DIR} ..."
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
        mkdir -p "${IMAGE_ENCODER_DIR}"
        cp -L "${src}" "${IMAGE_ENCODER_DIR}/${image_encoder_name}"
        break
      fi
    done

    if [[ ! -f "${IMAGE_ENCODER_DIR}/${image_encoder_name}" ]]; then
      echo "Downloading Wan2.1-I2V-14B-480P CLIP cache to ${image_encoder_cache_dir} ..."
      huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "${image_encoder_cache_dir}"
      mkdir -p "${IMAGE_ENCODER_DIR}"
      cp -L "${image_encoder_cache_dir}/${image_encoder_name}" "${IMAGE_ENCODER_DIR}/${image_encoder_name}"
    fi
  fi

  verify_required_assets
}

if [[ "${PREPARE_ASSETS:-true}" == "true" && "${DRY_RUN:-false}" != "true" ]]; then
  if [[ "${NODE_RANK}" == "0" ]]; then
    prepare_assets
  else
    echo "NODE_RANK=${NODE_RANK} waiting for shared assets ..."
    until required_assets_ready; do
      sleep 5
    done
  fi
fi

mkdir -p "${OUTPUT_DIR}"
cd "${DREAMZERO_ROOT}"

echo "========== DROID Wan2.2 fseq200 launch =========="
echo "ARCH=${ARCH}"
echo "DREAMZERO_ROOT=${DREAMZERO_ROOT}"
echo "DROID_DATA_ROOT=${DROID_DATA_ROOT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "WAN22_CKPT_DIR=${WAN22_CKPT_DIR}"
echo "IMAGE_ENCODER_DIR=${IMAGE_ENCODER_DIR}"
echo "TOKENIZER_DIR=${TOKENIZER_DIR}"
echo "NNODES=${NNODES} NODE_RANK=${NODE_RANK}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "LOCAL_NUM_GPUS=${LOCAL_NUM_GPUS} WORLD_GPUS=${WORLD_GPUS}"
echo "PER_DEVICE_BS=${PER_DEVICE_BS} GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
echo "TRAIN_ARCHITECTURE=${TRAIN_ARCHITECTURE}"
echo "DEEPSPEED_CFG=${DEEPSPEED_CFG}"
echo "DYNAMICS_LOSS_WEIGHT=${DYNAMICS_LOSS_WEIGHT}"
echo "ACTION_LOSS_WEIGHT=${ACTION_LOSS_WEIGHT}"
echo "TARGET_VIDEO_HEIGHT=${TARGET_VIDEO_HEIGHT}"
echo "TARGET_VIDEO_WIDTH=${TARGET_VIDEO_WIDTH}"
echo "FRAME_SEQLEN=${FRAME_SEQLEN}"
echo "MAX_CHUNK_SIZE=${MAX_CHUNK_SIZE}"
if [[ "${ARCH}" == "mot" ]]; then
  echo "MOT_ACTION_VIDEO_ATTENTION=${MOT_ACTION_VIDEO_ATTENTION}"
  echo "MOT_ACTION_VIDEO_KI=${MOT_ACTION_VIDEO_KI}"
  echo "MOT_DECOUPLE_VIDEO_ACTION_NOISE=${MOT_DECOUPLE_VIDEO_ACTION_NOISE}"
fi
echo "================================================="

EVAL_STRATEGY_OVERRIDE="$(quote_hydra_interval "${EVAL_STRATEGY}")"
SAVE_STRATEGY_OVERRIDE="$(quote_hydra_interval "${SAVE_STRATEGY}")"

TRAIN_OVERRIDES=(
  "report_to=${REPORT_TO}"
  "data=dreamzero/droid_relative_wan22"
  "wandb_project=${WANDB_PROJECT_NAME}"
  "train_architecture=${TRAIN_ARCHITECTURE}"
  "architecture=${ARCH}"
  "dynamics_loss_weight=${DYNAMICS_LOSS_WEIGHT}"
  "action_loss_weight=${ACTION_LOSS_WEIGHT}"
  "droid_random_drop_exterior_view_prob=${DROID_RANDOM_DROP_EXTERIOR_VIEW_PROB}"
  "num_frames=${NUM_FRAMES}"
  "action_horizon=${ACTION_HORIZON}"
  "num_views=3"
  "model=dreamzero/vla"
  "model/dreamzero/action_head=${ACTION_HEAD_CONFIG}"
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
  "training_args.warmup_ratio=${WARMUP_RATIO}"
  "output_dir=${OUTPUT_DIR}"
  "per_device_train_batch_size=${PER_DEVICE_BS}"
  "per_device_eval_batch_size=${PER_DEVICE_BS}"
  "global_batch_size=${GLOBAL_BATCH_SIZE}"
  "max_steps=${MAX_STEPS}"
  "save_steps=${SAVE_STEPS}"
  "eval_strategy=${EVAL_STRATEGY_OVERRIDE}"
  "do_eval=false"
  "weight_decay=${WEIGHT_DECAY}"
  "save_total_limit=${SAVE_TOTAL_LIMIT}"
  "upload_checkpoints=false"
  "bf16=true"
  "tf32=true"
  "eval_bf16=true"
  "dataloader_pin_memory=${DATALOADER_PIN_MEMORY}"
  "dataloader_num_workers=${DATALOADER_NUM_WORKERS}"
  "dataloader_persistent_workers=${DATALOADER_PERSISTENT_WORKERS}"
  "image_resolution_width=${IMAGE_RESOLUTION_WIDTH}"
  "image_resolution_height=${IMAGE_RESOLUTION_HEIGHT}"
  "save_lora_only=${SAVE_LORA_ONLY}"
  "max_chunk_size=${MAX_CHUNK_SIZE}"
  "save_strategy=${SAVE_STRATEGY_OVERRIDE}"
  "droid_data_root=${DROID_DATA_ROOT}"
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

if [[ "${ARCH}" == "mot" ]]; then
  TRAIN_OVERRIDES+=(
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
  )
fi

if [[ -n "${EVAL_STEPS}" ]]; then
  TRAIN_OVERRIDES+=("eval_steps=${EVAL_STEPS}")
fi

LAUNCH_CMD=(
  "${PYTHON_BIN}" -m torch.distributed.run
  --nnodes="${NNODES}" \
  --nproc-per-node="${LOCAL_NUM_GPUS}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${EXPERIMENT_PY}"
)

if [[ "${DRY_RUN:-false}" == "true" ]]; then
  printf 'DRY_RUN command:'
  printf ' %q' "${LAUNCH_CMD[@]}" "${TRAIN_OVERRIDES[@]}" "$@"
  printf '\n'
  exit 0
fi

exec "${LAUNCH_CMD[@]}" \
  "${TRAIN_OVERRIDES[@]}" \
  "$@"
