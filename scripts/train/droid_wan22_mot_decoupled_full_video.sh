#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

DREAMZERO_ROOT="${DREAMZERO_ROOT:-${REPO_ROOT}}"
PYTHON_BIN="${PYTHON_BIN:-${DREAMZERO_ROOT}/.venv/bin/python}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/data/checkpoints/dreamzero}"
DATASET_ROOT="${DATASET_ROOT:-/data/datasets/dreamzero/droid_lerobot}"
OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_ROOT}/dreamzero_droid_wan22_mot_decoupled_full_video}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-dreamzero}"
WAN22_CKPT_DIR="${WAN22_CKPT_DIR:-${CHECKPOINT_ROOT}/Wan2.2-TI2V-5B}"
TEXT_ENCODER_PATH="${TEXT_ENCODER_PATH:-${WAN22_CKPT_DIR}/models_t5_umt5-xxl-enc-bf16.pth}"
IMAGE_ENCODER_PATH="${IMAGE_ENCODER_PATH:-${WAN22_CKPT_DIR}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth}"
VAE_PATH="${VAE_PATH:-${WAN22_CKPT_DIR}/Wan2.2_VAE.pth}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${WAN22_CKPT_DIR}/google/umt5-xxl}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
if [[ -n "${GPU_IDS}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
  IFS=',' read -r -a _GPU_ID_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
  NUM_GPUS="${NUM_GPUS:-${#_GPU_ID_ARRAY[@]}}"
else
  NUM_GPUS="${NUM_GPUS:-1}"
fi

_FIRST_GPU_ID="${GPU_IDS%%,*}"
if [[ ! "${_FIRST_GPU_ID}" =~ ^[0-9]+$ ]]; then
  _FIRST_GPU_ID=0
fi
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((29600 + _FIRST_GPU_ID))}"

PER_DEVICE_BS="${PER_DEVICE_BS:-16}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
MAX_STEPS="${MAX_STEPS:-100000}"
EVAL_STEPS="${EVAL_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-10000}"
DEEPSPEED_CFG="${DEEPSPEED_CFG:-zero2}"
LEARNING_RATE="${LEARNING_RATE:-1.5e-5}"

IMAGE_RESOLUTION_WIDTH="${IMAGE_RESOLUTION_WIDTH:-320}"
IMAGE_RESOLUTION_HEIGHT="${IMAGE_RESOLUTION_HEIGHT:-160}"
MODEL_TARGET_HEIGHT="${MODEL_TARGET_HEIGHT:-320}"
MODEL_TARGET_WIDTH="${MODEL_TARGET_WIDTH:-640}"
MODEL_FRAME_SEQLEN="${MODEL_FRAME_SEQLEN:-200}"

NUM_FRAMES="${NUM_FRAMES:-33}"
ACTION_HORIZON="${ACTION_HORIZON:-24}"
MAX_CHUNK_SIZE="${MAX_CHUNK_SIZE:-4}"
NUM_FRAME_PER_BLOCK="${NUM_FRAME_PER_BLOCK:-2}"
NUM_ACTION_PER_BLOCK="${NUM_ACTION_PER_BLOCK:-24}"
NUM_STATE_PER_BLOCK="${NUM_STATE_PER_BLOCK:-1}"

DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-2}"
DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-true}"
DATALOADER_IN_ORDER="${DATALOADER_IN_ORDER:-false}"
USE_GRADIENT_CHECKPOINTING="${USE_GRADIENT_CHECKPOINTING:-true}"
TRACK_LOSS_GRAD="${TRACK_LOSS_GRAD:-false}"
LOSS_GRAD_LOGGING_STEPS="${LOSS_GRAD_LOGGING_STEPS:-10}"

MOT_ACTION_VIDEO_ATTENTION="${MOT_ACTION_VIDEO_ATTENTION:-full_video}"
MOT_ACTION_VIDEO_KI="${MOT_ACTION_VIDEO_KI:-${MOT_KI:-false}}"
MOT_INFERENCE_VIDEO_MODE="${MOT_INFERENCE_VIDEO_MODE:-auto}"
MOT_DECOUPLE_VIDEO_ACTION_NOISE="${MOT_DECOUPLE_VIDEO_ACTION_NOISE:-true}"
MOT_DECOUPLED_INFERENCE_VIDEO_FINAL_NOISE="${MOT_DECOUPLED_INFERENCE_VIDEO_FINAL_NOISE:-0.8}"
MOT_DECOUPLED_INFERENCE_VIDEO_REFRESH_STEPS="${MOT_DECOUPLED_INFERENCE_VIDEO_REFRESH_STEPS:-1}"
DROID_RANDOM_DROP_EXTERIOR_VIEW_PROB="${DROID_RANDOM_DROP_EXTERIOR_VIEW_PROB:-0.1}"

export HYDRA_FULL_ERROR=1
export SWANLAB_SYNC_WANDB="${SWANLAB_SYNC_WANDB:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT_NAME}"
export PYTHONPATH="${DREAMZERO_ROOT}:${PYTHONPATH:-}"
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"

cd "${DREAMZERO_ROOT}"

echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Using NUM_GPUS=${NUM_GPUS}, PER_DEVICE_BS=${PER_DEVICE_BS}, GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
echo "Using DATASET_ROOT=${DATASET_ROOT}"
echo "Using OUTPUT_DIR=${OUTPUT_DIR}"
echo "Using WAN22_CKPT_DIR=${WAN22_CKPT_DIR}"
echo "Using IMAGE_ENCODER_PATH=${IMAGE_ENCODER_PATH}"
echo "Using TOKENIZER_PATH=${TOKENIZER_PATH}"
echo "Using WANDB_PROJECT=${WANDB_PROJECT}"
echo "Using WANDB_MODE=${WANDB_MODE}"
echo "Using SWANLAB_SYNC_WANDB=${SWANLAB_SYNC_WANDB}"
echo "Using LEARNING_RATE=${LEARNING_RATE}, MAX_STEPS=${MAX_STEPS}"
echo "Using DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS}, DATALOADER_PREFETCH_FACTOR=${DATALOADER_PREFETCH_FACTOR}"
echo "Using DATALOADER_PERSISTENT_WORKERS=${DATALOADER_PERSISTENT_WORKERS}"
echo "Using DATALOADER_IN_ORDER=${DATALOADER_IN_ORDER}"
echo "Using TRACK_LOSS_GRAD=${TRACK_LOSS_GRAD}, LOSS_GRAD_LOGGING_STEPS=${LOSS_GRAD_LOGGING_STEPS}"
echo "Using IMAGE_RESOLUTION=${IMAGE_RESOLUTION_HEIGHT}x${IMAGE_RESOLUTION_WIDTH}"
echo "Using MODEL_TARGET_VIDEO=${MODEL_TARGET_HEIGHT}x${MODEL_TARGET_WIDTH}"
echo "Using MODEL_FRAME_SEQLEN=${MODEL_FRAME_SEQLEN}"
echo "Using ACTION_HEAD_GRADIENT_CHECKPOINTING=${USE_GRADIENT_CHECKPOINTING}"
echo "Using MOT_ACTION_VIDEO_ATTENTION=${MOT_ACTION_VIDEO_ATTENTION}"
echo "Using MOT_ACTION_VIDEO_KI=${MOT_ACTION_VIDEO_KI}"
echo "Using MOT_INFERENCE_VIDEO_MODE=${MOT_INFERENCE_VIDEO_MODE}"
echo "Using MOT_DECOUPLE_VIDEO_ACTION_NOISE=${MOT_DECOUPLE_VIDEO_ACTION_NOISE}"
echo "Using MOT_DECOUPLED_INFERENCE_VIDEO_FINAL_NOISE=${MOT_DECOUPLED_INFERENCE_VIDEO_FINAL_NOISE}"
echo "Using MOT_DECOUPLED_INFERENCE_VIDEO_REFRESH_STEPS=${MOT_DECOUPLED_INFERENCE_VIDEO_REFRESH_STEPS}"
echo "Using DROID_RANDOM_DROP_EXTERIOR_VIEW_PROB=${DROID_RANDOM_DROP_EXTERIOR_VIEW_PROB}"

if [[ "${NUM_GPUS}" -eq 1 ]]; then
  export RANK="${RANK:-0}"
  export WORLD_SIZE="${WORLD_SIZE:-1}"
  export LOCAL_RANK="${LOCAL_RANK:-0}"
  export LOCAL_WORLD_SIZE="${LOCAL_WORLD_SIZE:-1}"
  export NODE_RANK="${NODE_RANK:-0}"
  echo "Using LAUNCHER=python_env_dist MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
  LAUNCH_CMD=("${PYTHON_BIN}")
else
  echo "Using LAUNCHER=torchrun MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
  LAUNCH_CMD=(
    "${PYTHON_BIN}" -m torch.distributed.run
    --nproc_per_node "${NUM_GPUS}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
  )
fi

"${LAUNCH_CMD[@]}" \
  groot/vla/experiment/experiment.py \
  report_to=wandb \
  data=dreamzero/droid_relative_wan22 \
  wandb_project="${WANDB_PROJECT_NAME}" \
  train_architecture=full \
  architecture=mot \
  mot_action_hidden_dim=1024 \
  mot_action_ffn_dim=4096 \
  mot_action_num_layers=null \
  mot_action_num_heads=8 \
  mot_action_video_attention="${MOT_ACTION_VIDEO_ATTENTION}" \
  mot_action_video_ki="${MOT_ACTION_VIDEO_KI}" \
  mot_inference_video_mode="${MOT_INFERENCE_VIDEO_MODE}" \
  mot_decouple_video_action_noise="${MOT_DECOUPLE_VIDEO_ACTION_NOISE}" \
  mot_decoupled_inference_video_final_noise="${MOT_DECOUPLED_INFERENCE_VIDEO_FINAL_NOISE}" \
  mot_decoupled_inference_video_refresh_steps="${MOT_DECOUPLED_INFERENCE_VIDEO_REFRESH_STEPS}" \
  droid_random_drop_exterior_view_prob="${DROID_RANDOM_DROP_EXTERIOR_VIEW_PROB}" \
  num_frames="${NUM_FRAMES}" \
  action_horizon="${ACTION_HORIZON}" \
  num_views=3 \
  model=dreamzero/vla \
  model/dreamzero/action_head=wan_flow_matching_action_tf_wan22_mot \
  action_head_cfg.config.use_gradient_checkpointing="${USE_GRADIENT_CHECKPOINTING}" \
  model/dreamzero/transform=dreamzero_cotrain \
  num_frame_per_block="${NUM_FRAME_PER_BLOCK}" \
  num_action_per_block="${NUM_ACTION_PER_BLOCK}" \
  num_state_per_block="${NUM_STATE_PER_BLOCK}" \
  data_collator.num_frames="${NUM_FRAMES}" \
  data_collator.max_chunk_size="${MAX_CHUNK_SIZE}" \
  data_collator.num_action_per_block="${NUM_ACTION_PER_BLOCK}" \
  data_collator.num_state_per_block="${NUM_STATE_PER_BLOCK}" \
  seed=42 \
  training_args.learning_rate="${LEARNING_RATE}" \
  training_args.deepspeed="groot/vla/configs/deepspeed/${DEEPSPEED_CFG}.json" \
  output_dir="${OUTPUT_DIR}" \
  per_device_train_batch_size="${PER_DEVICE_BS}" \
  per_device_eval_batch_size="${PER_DEVICE_BS}" \
  global_batch_size="${GLOBAL_BATCH_SIZE}" \
  max_steps="${MAX_STEPS}" \
  save_steps="${SAVE_STEPS}" \
  eval_strategy=no \
  eval_steps="${EVAL_STEPS}" \
  do_eval=false \
  weight_decay=1e-5 \
  save_total_limit=5 \
  upload_checkpoints=false \
  bf16=true \
  tf32=true \
  eval_bf16=true \
  trainer.track_loss_grad="${TRACK_LOSS_GRAD}" \
  trainer.loss_grad_logging_steps="${LOSS_GRAD_LOGGING_STEPS}" \
  trainer.dataloader_in_order="${DATALOADER_IN_ORDER}" \
  dataloader_pin_memory=true \
  dataloader_num_workers="${DATALOADER_NUM_WORKERS}" \
  dataloader_persistent_workers="${DATALOADER_PERSISTENT_WORKERS}" \
  image_resolution_width="${IMAGE_RESOLUTION_WIDTH}" \
  image_resolution_height="${IMAGE_RESOLUTION_HEIGHT}" \
  frame_seqlen="${MODEL_FRAME_SEQLEN}" \
  action_head_cfg.config.target_video_height="${MODEL_TARGET_HEIGHT}" \
  action_head_cfg.config.target_video_width="${MODEL_TARGET_WIDTH}" \
  save_lora_only=false \
  max_chunk_size="${MAX_CHUNK_SIZE}" \
  save_strategy=steps \
  droid_data_root="${DATASET_ROOT}" \
  dit_version="${WAN22_CKPT_DIR}" \
  text_encoder_pretrained_path="${TEXT_ENCODER_PATH}" \
  image_encoder_pretrained_path="${IMAGE_ENCODER_PATH}" \
  vae_pretrained_path="${VAE_PATH}" \
  tokenizer_path="${TOKENIZER_PATH}" \
  dataset_shard_sampling_rate=0.1 \
  +training_args.dataloader_prefetch_factor="${DATALOADER_PREFETCH_FACTOR}" \
  "$@"
