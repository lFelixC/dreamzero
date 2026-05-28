#!/bin/bash
set -euo pipefail

# Run one CBS branch from a warmup checkpoint. This intentionally loads only
# model weights through pretrained_model_path and starts a fresh optimizer and
# constant LR schedule for the short branch.

DREAMZERO_ROOT="${DREAMZERO_ROOT:-/data/dreamzero}"
PYTHON_BIN="${PYTHON_BIN:-${DREAMZERO_ROOT}/.venv/bin/python}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/data/checkpoints/dreamzero}"
DATASET_ROOT="${DATASET_ROOT:-/data/datasets/dreamzero/droid_lerobot}"
EXPERIMENT_PY="${EXPERIMENT_PY:-${DREAMZERO_ROOT}/groot/vla/experiment/experiment.py}"

WAN22_CKPT_DIR="${WAN22_CKPT_DIR:-${CHECKPOINT_ROOT}/Wan2.2-TI2V-5B}"
IMAGE_ENCODER_DIR="${IMAGE_ENCODER_DIR:-${WAN22_CKPT_DIR}}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${WAN22_CKPT_DIR}/google/umt5-xxl}"

if [[ -z "${WARMUP_CKPT:-}" ]]; then
  echo "ERROR: WARMUP_CKPT is required, e.g. /data/checkpoints/dreamzero/cbs_mot_droid/<run>/warmup.../checkpoint-5000"
  exit 1
fi

if [[ ! -d "${WARMUP_CKPT}" ]]; then
  echo "ERROR: WARMUP_CKPT does not exist: ${WARMUP_CKPT}"
  exit 1
fi

CBS_ROOT="${CBS_ROOT:-${CHECKPOINT_ROOT}/cbs_mot_droid}"
if [[ -z "${CBS_RUN_ID:-}" ]]; then
  CBS_RUN_ID="$(basename "$(dirname "$(dirname "${WARMUP_CKPT}")")")"
fi
CBS_BRANCH_ROOT="${CBS_BRANCH_ROOT:-${CBS_ROOT}/${CBS_RUN_ID}/branches}"

BASE_GLOBAL_BATCH_SIZE="${BASE_GLOBAL_BATCH_SIZE:-128}"
BASE_LEARNING_RATE="${BASE_LEARNING_RATE:-1.5e-5}"
CBS_BASE_STEPS="${CBS_BASE_STEPS:-4096}"
CBS_GLOBAL_BATCH_SIZE="${CBS_GLOBAL_BATCH_SIZE:-${GLOBAL_BATCH_SIZE:-256}}"

read -r BRANCH_LEARNING_RATE BRANCH_MAX_STEPS < <("${PYTHON_BIN}" - <<PY
import math
base_bs = float("${BASE_GLOBAL_BATCH_SIZE}")
base_lr = float("${BASE_LEARNING_RATE}")
base_steps = int("${CBS_BASE_STEPS}")
branch_bs = float("${CBS_GLOBAL_BATCH_SIZE}")
lr = base_lr * math.sqrt(branch_bs / base_bs)
steps = int(math.ceil(base_steps * base_bs / branch_bs))
print(f"{lr:.8g} {steps}")
PY
)

OUTPUT_DIR="${OUTPUT_DIR:-${CBS_BRANCH_ROOT}/bs${CBS_GLOBAL_BATCH_SIZE}_lr${BRANCH_LEARNING_RATE}_steps${BRANCH_MAX_STEPS}}"

WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-dreamzero}"
PER_DEVICE_BS="${PER_DEVICE_BS:-32}"
MAX_STEPS="${MAX_STEPS:-${BRANCH_MAX_STEPS}}"
SAVE_STEPS="${SAVE_STEPS:-${MAX_STEPS}}"
DEEPSPEED_CFG="${DEEPSPEED_CFG:-zero2}"
NUM_FRAMES="${NUM_FRAMES:-33}"
ACTION_HORIZON="${ACTION_HORIZON:-24}"
MAX_CHUNK_SIZE="${MAX_CHUNK_SIZE:-4}"
NUM_FRAME_PER_BLOCK="${NUM_FRAME_PER_BLOCK:-2}"
NUM_ACTION_PER_BLOCK="${NUM_ACTION_PER_BLOCK:-24}"
NUM_STATE_PER_BLOCK="${NUM_STATE_PER_BLOCK:-1}"
IMAGE_RESOLUTION_WIDTH="${IMAGE_RESOLUTION_WIDTH:-320}"
IMAGE_RESOLUTION_HEIGHT="${IMAGE_RESOLUTION_HEIGHT:-160}"
TARGET_VIDEO_HEIGHT="${TARGET_VIDEO_HEIGHT:-320}"
TARGET_VIDEO_WIDTH="${TARGET_VIDEO_WIDTH:-640}"
FRAME_SEQLEN="${FRAME_SEQLEN:-200}"
ACTION_LOSS_WEIGHT="${ACTION_LOSS_WEIGHT:-2.0}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-4}"
DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-true}"
USE_GRADIENT_CHECKPOINTING="${USE_GRADIENT_CHECKPOINTING:-true}"
MOT_ACTION_VIDEO_ATTENTION="${MOT_ACTION_VIDEO_ATTENTION:-full_video}"
MOT_ACTION_VIDEO_KI="${MOT_ACTION_VIDEO_KI:-off}"
MOT_INFERENCE_VIDEO_MODE="${MOT_INFERENCE_VIDEO_MODE:-auto}"
MOT_DECOUPLE_VIDEO_ACTION_NOISE="${MOT_DECOUPLE_VIDEO_ACTION_NOISE:-false}"
MOT_VIDEO_NOISE_BETA_ALPHA="${MOT_VIDEO_NOISE_BETA_ALPHA:-3.0}"
MOT_VIDEO_NOISE_BETA_BETA="${MOT_VIDEO_NOISE_BETA_BETA:-1.0}"
MOT_DECOUPLED_INFERENCE_VIDEO_FINAL_NOISE="${MOT_DECOUPLED_INFERENCE_VIDEO_FINAL_NOISE:-0.8}"
MOT_DECOUPLED_INFERENCE_VIDEO_REFRESH_STEPS="${MOT_DECOUPLED_INFERENCE_VIDEO_REFRESH_STEPS:-8}"
DROID_RANDOM_DROP_EXTERIOR_VIEW_PROB="${DROID_RANDOM_DROP_EXTERIOR_VIEW_PROB:-0.0}"
DATASET_SHARD_SAMPLING_RATE="${DATASET_SHARD_SAMPLING_RATE:-0.1}"
DATASET_SHARD_SAMPLING_STRATEGY="${DATASET_SHARD_SAMPLING_STRATEGY:-random}"
DATASET_SHARD_SAMPLING_BLOCK_SIZE="${DATASET_SHARD_SAMPLING_BLOCK_SIZE:-64}"

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29400}"

GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
if [[ -n "${GPU_IDS}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
fi

export HYDRA_FULL_ERROR=1
export SWANLAB_SYNC_WANDB="${SWANLAB_SYNC_WANDB:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT_NAME}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-cbs_mot_droid_${CBS_RUN_ID}}"
export WANDB_NAME="${WANDB_NAME:-cbs_mot_droid_${CBS_RUN_ID}_bs${CBS_GLOBAL_BATCH_SIZE}}"
export WANDB_TAGS="${WANDB_TAGS:-cbs,mot,droid,branch,bs-${CBS_GLOBAL_BATCH_SIZE},lr-${BRANCH_LEARNING_RATE},steps-${MAX_STEPS},action-loss-weight-${ACTION_LOSS_WEIGHT}}"
export WANDB_NOTES="${WANDB_NOTES:-CBS branch from ${WARMUP_CKPT}; AdamW sqrt LR scaling from batch ${BASE_GLOBAL_BATCH_SIZE}, LR ${BASE_LEARNING_RATE}; action loss weight ${ACTION_LOSS_WEIGHT}; constant LR ${BRANCH_LEARNING_RATE}; no second warmup; no LR decay; global batch ${CBS_GLOBAL_BATCH_SIZE}; steps ${MAX_STEPS}.}"
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
GLOBAL_BATCH_SIZE="${CBS_GLOBAL_BATCH_SIZE}"
REQUESTED_PER_DEVICE_BS="${PER_DEVICE_BS}"

if (( GLOBAL_BATCH_SIZE % WORLD_GPUS != 0 )); then
  echo "ERROR: GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} must be divisible by WORLD_GPUS=${WORLD_GPUS}"
  exit 1
fi

MAX_PER_DEVICE_FOR_GLOBAL=$((GLOBAL_BATCH_SIZE / WORLD_GPUS))
if (( PER_DEVICE_BS > MAX_PER_DEVICE_FOR_GLOBAL )); then
  echo "Capping PER_DEVICE_BS from ${PER_DEVICE_BS} to ${MAX_PER_DEVICE_FOR_GLOBAL} for global batch ${GLOBAL_BATCH_SIZE}"
  PER_DEVICE_BS="${MAX_PER_DEVICE_FOR_GLOBAL}"
fi

PER_STEP_BATCH_SIZE=$((PER_DEVICE_BS * WORLD_GPUS))
if (( GLOBAL_BATCH_SIZE % PER_STEP_BATCH_SIZE != 0 )); then
  echo "ERROR: GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} must be divisible by PER_DEVICE_BS * WORLD_GPUS = ${PER_STEP_BATCH_SIZE}"
  exit 1
fi
EXPECTED_GRAD_ACC=$((GLOBAL_BATCH_SIZE / PER_STEP_BATCH_SIZE))

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
cat > "${OUTPUT_DIR}/cbs_metadata.json" <<JSON
{
  "phase": "branch",
  "cbs_run_id": "${CBS_RUN_ID}",
  "warmup_checkpoint": "${WARMUP_CKPT}",
  "base_global_batch_size": ${BASE_GLOBAL_BATCH_SIZE},
  "base_learning_rate": "${BASE_LEARNING_RATE}",
  "base_equivalent_steps": ${CBS_BASE_STEPS},
  "branch_global_batch_size": ${CBS_GLOBAL_BATCH_SIZE},
  "requested_per_device_batch_size": ${REQUESTED_PER_DEVICE_BS},
  "effective_per_device_batch_size": ${PER_DEVICE_BS},
  "expected_gradient_accumulation_steps": ${EXPECTED_GRAD_ACC},
  "branch_learning_rate": "${BRANCH_LEARNING_RATE}",
  "branch_max_steps": ${MAX_STEPS},
  "branch_save_steps": ${SAVE_STEPS},
  "action_loss_weight": ${ACTION_LOSS_WEIGHT},
  "scheduler": "constant",
  "warmup_steps": 0,
  "warmup_ratio": 0.0,
  "lr_scaling": "adamw_sqrt_batch",
  "output_dir": "${OUTPUT_DIR}"
}
JSON

cd "${DREAMZERO_ROOT}"

echo "========== DROID MoT CBS branch =========="
echo "DREAMZERO_ROOT=${DREAMZERO_ROOT}"
echo "WARMUP_CKPT=${WARMUP_CKPT}"
echo "NNODES=${NNODES}"
echo "NODE_RANK=${NODE_RANK}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "WORLD_GPUS=${WORLD_GPUS}"
echo "REQUESTED_PER_DEVICE_BS=${REQUESTED_PER_DEVICE_BS}"
echo "PER_DEVICE_BS=${PER_DEVICE_BS}"
echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
echo "EXPECTED_GRAD_ACC=${EXPECTED_GRAD_ACC}"
echo "BASE_GLOBAL_BATCH_SIZE=${BASE_GLOBAL_BATCH_SIZE}"
echo "BASE_LEARNING_RATE=${BASE_LEARNING_RATE}"
echo "BRANCH_LEARNING_RATE=${BRANCH_LEARNING_RATE}"
echo "CBS_BASE_STEPS=${CBS_BASE_STEPS}"
echo "MAX_STEPS=${MAX_STEPS}"
echo "SAVE_STEPS=${SAVE_STEPS}"
echo "LR scheduler=constant"
echo "Warmup disabled: warmup_ratio=0, warmup_steps=0"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "DROID_DATA_ROOT=${DROID_DATA_ROOT}"
echo "MOT_ACTION_VIDEO_ATTENTION=${MOT_ACTION_VIDEO_ATTENTION}"
echo "ACTION_LOSS_WEIGHT=${ACTION_LOSS_WEIGHT}"
echo "FRAME_SEQLEN=${FRAME_SEQLEN}"
echo "TARGET_VIDEO_HEIGHT=${TARGET_VIDEO_HEIGHT}"
echo "TARGET_VIDEO_WIDTH=${TARGET_VIDEO_WIDTH}"
echo "SwanLab sync via SWANLAB_SYNC_WANDB=${SWANLAB_SYNC_WANDB}"
echo "=========================================="

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
  "action_loss_weight=${ACTION_LOSS_WEIGHT}"
  "action_head_cfg.config.action_loss_weight=${ACTION_LOSS_WEIGHT}"
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
  "learning_rate=${BRANCH_LEARNING_RATE}"
  "training_args.learning_rate=${BRANCH_LEARNING_RATE}"
  "lr_scheduler_type=constant"
  "training_args.lr_scheduler_type=constant"
  "warmup_ratio=0.0"
  "training_args.warmup_ratio=0.0"
  "+training_args.warmup_steps=0"
  "training_args.deepspeed=groot/vla/configs/deepspeed/${DEEPSPEED_CFG}.json"
  "output_dir=${OUTPUT_DIR}"
  "per_device_train_batch_size=${PER_DEVICE_BS}"
  "per_device_eval_batch_size=${PER_DEVICE_BS}"
  "global_batch_size=${GLOBAL_BATCH_SIZE}"
  "max_steps=${MAX_STEPS}"
  "save_steps=${SAVE_STEPS}"
  "logging_steps=10"
  "eval_strategy=no"
  "eval_steps=100"
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
  "dataset_shard_sampling_strategy=${DATASET_SHARD_SAMPLING_STRATEGY}"
  "dataset_shard_sampling_block_size=${DATASET_SHARD_SAMPLING_BLOCK_SIZE}"
  "pretrained_model_path=${WARMUP_CKPT}"
  "+training_args.dataloader_prefetch_factor=${DATALOADER_PREFETCH_FACTOR}"
  "+cbs.phase=branch"
  "+cbs.run_id=${CBS_RUN_ID}"
  "+cbs.warmup_checkpoint=${WARMUP_CKPT}"
  "+cbs.base_global_batch_size=${BASE_GLOBAL_BATCH_SIZE}"
  "+cbs.base_learning_rate=${BASE_LEARNING_RATE}"
  "+cbs.base_equivalent_steps=${CBS_BASE_STEPS}"
  "+cbs.branch_global_batch_size=${GLOBAL_BATCH_SIZE}"
  "+cbs.branch_learning_rate=${BRANCH_LEARNING_RATE}"
  "+cbs.branch_max_steps=${MAX_STEPS}"
  "+cbs.action_loss_weight=${ACTION_LOSS_WEIGHT}"
  "+cbs.scheduler=constant"
  "+cbs.warmup_steps=0"
  "+cbs.lr_scaling=adamw_sqrt_batch"
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
