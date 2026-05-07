#!/bin/bash
set -euo pipefail

DREAMZERO_ROOT="${DREAMZERO_ROOT:-/data/dreamzero_mot}"
PYTHON_BIN="${PYTHON_BIN:-/data/dreamzero/.venv/bin/python}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/data/checkpoints/dreamzero}"
DATASET_ROOT="${DATASET_ROOT:-/data/datasets/dreamzero/droid_lerobot}"
OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_ROOT}/dreamzero_droid_wan22_mot_mix_att_anchor_full}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-dreamzero}"
WAN22_CKPT_DIR="${WAN22_CKPT_DIR:-${CHECKPOINT_ROOT}/Wan2.2-TI2V-5B}"
TEXT_ENCODER_PATH="${TEXT_ENCODER_PATH:-${WAN22_CKPT_DIR}/models_t5_umt5-xxl-enc-bf16.pth}"
IMAGE_ENCODER_PATH="${IMAGE_ENCODER_PATH:-${WAN22_CKPT_DIR}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth}"
VAE_PATH="${VAE_PATH:-${WAN22_CKPT_DIR}/Wan2.2_VAE.pth}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${WAN22_CKPT_DIR}/google/umt5-xxl}"

GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-4,5,6,7}}"
if [[ -n "${GPU_IDS}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
  IFS=',' read -r -a _GPU_ID_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
  NUM_GPUS="${NUM_GPUS:-${#_GPU_ID_ARRAY[@]}}"
else
  NUM_GPUS="${NUM_GPUS:-1}"
fi
PER_DEVICE_BS="${PER_DEVICE_BS:-96}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((NUM_GPUS * PER_DEVICE_BS))}"
MAX_STEPS="${MAX_STEPS:-30000}"
EVAL_STEPS="${EVAL_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-10000}"
DEEPSPEED_CFG="${DEEPSPEED_CFG:-zero2}"
NUM_FRAMES="${NUM_FRAMES:-33}"
ACTION_HORIZON="${ACTION_HORIZON:-24}"
MAX_CHUNK_SIZE="${MAX_CHUNK_SIZE:-4}"
NUM_FRAME_PER_BLOCK="${NUM_FRAME_PER_BLOCK:-2}"
NUM_ACTION_PER_BLOCK="${NUM_ACTION_PER_BLOCK:-24}"
NUM_STATE_PER_BLOCK="${NUM_STATE_PER_BLOCK:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-4}"
DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-true}"
USE_GRADIENT_CHECKPOINTING="${USE_GRADIENT_CHECKPOINTING:-false}"
MOT_ACTION_VIDEO_ATTENTION="${MOT_ACTION_VIDEO_ATTENTION:-first_frame}"
MOT_ACTION_VIDEO_KI="${MOT_ACTION_VIDEO_KI:-${MOT_KI:-false}}"
MOT_INFERENCE_VIDEO_MODE="${MOT_INFERENCE_VIDEO_MODE:-auto}"
MOT_DECOUPLE_VIDEO_ACTION_NOISE="${MOT_DECOUPLE_VIDEO_ACTION_NOISE:-false}"
MOT_VIDEO_NOISE_BETA_ALPHA="${MOT_VIDEO_NOISE_BETA_ALPHA:-3.0}"
MOT_VIDEO_NOISE_BETA_BETA="${MOT_VIDEO_NOISE_BETA_BETA:-1.0}"
MOT_DECOUPLED_INFERENCE_VIDEO_FINAL_NOISE="${MOT_DECOUPLED_INFERENCE_VIDEO_FINAL_NOISE:-0.8}"
MOT_DECOUPLED_INFERENCE_VIDEO_REFRESH_STEPS="${MOT_DECOUPLED_INFERENCE_VIDEO_REFRESH_STEPS:-8}"
DROID_RANDOM_DROP_EXTERIOR_VIEW_PROB="${DROID_RANDOM_DROP_EXTERIOR_VIEW_PROB:-0.0}"

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
echo "Using WAN22_CKPT_DIR=${WAN22_CKPT_DIR}"
echo "Using IMAGE_ENCODER_PATH=${IMAGE_ENCODER_PATH}"
echo "Using TOKENIZER_PATH=${TOKENIZER_PATH}"
echo "Using WANDB_PROJECT=${WANDB_PROJECT}"
echo "Using DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS}, DATALOADER_PREFETCH_FACTOR=${DATALOADER_PREFETCH_FACTOR}"
echo "Using DATALOADER_PERSISTENT_WORKERS=${DATALOADER_PERSISTENT_WORKERS}"
echo "Using ACTION_HEAD_GRADIENT_CHECKPOINTING=${USE_GRADIENT_CHECKPOINTING}"
echo "Using MOT_ACTION_VIDEO_ATTENTION=${MOT_ACTION_VIDEO_ATTENTION}"
echo "Using MOT_ACTION_VIDEO_KI=${MOT_ACTION_VIDEO_KI}"
echo "Using MOT_INFERENCE_VIDEO_MODE=${MOT_INFERENCE_VIDEO_MODE}"
echo "Using MOT_DECOUPLE_VIDEO_ACTION_NOISE=${MOT_DECOUPLE_VIDEO_ACTION_NOISE}"
echo "Using MOT_VIDEO_NOISE_BETA_ALPHA=${MOT_VIDEO_NOISE_BETA_ALPHA}"
echo "Using MOT_VIDEO_NOISE_BETA_BETA=${MOT_VIDEO_NOISE_BETA_BETA}"
echo "Using MOT_DECOUPLED_INFERENCE_VIDEO_FINAL_NOISE=${MOT_DECOUPLED_INFERENCE_VIDEO_FINAL_NOISE}"
echo "Using MOT_DECOUPLED_INFERENCE_VIDEO_REFRESH_STEPS=${MOT_DECOUPLED_INFERENCE_VIDEO_REFRESH_STEPS}"
echo "Using DROID_RANDOM_DROP_EXTERIOR_VIEW_PROB=${DROID_RANDOM_DROP_EXTERIOR_VIEW_PROB}"

if [[ "${NUM_GPUS}" -eq 1 ]]; then
  _FIRST_GPU_ID="${GPU_IDS%%,*}"
  if [[ ! "${_FIRST_GPU_ID}" =~ ^[0-9]+$ ]]; then
    _FIRST_GPU_ID=0
  fi
  export RANK="${RANK:-0}"
  export WORLD_SIZE="${WORLD_SIZE:-1}"
  export LOCAL_RANK="${LOCAL_RANK:-0}"
  export LOCAL_WORLD_SIZE="${LOCAL_WORLD_SIZE:-1}"
  export NODE_RANK="${NODE_RANK:-0}"
  export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  export MASTER_PORT="${MASTER_PORT:-$((29500 + _FIRST_GPU_ID))}"
  echo "Using LAUNCHER=python_env_dist MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
  LAUNCH_CMD=("${PYTHON_BIN}")
else
  echo "Using LAUNCHER=torchrun"
  LAUNCH_CMD=("${PYTHON_BIN}" -m torch.distributed.run --nproc_per_node "${NUM_GPUS}" --standalone)
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
  mot_video_noise_beta_alpha="${MOT_VIDEO_NOISE_BETA_ALPHA}" \
  mot_video_noise_beta_beta="${MOT_VIDEO_NOISE_BETA_BETA}" \
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
  training_args.learning_rate=2e-5 \
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
  dataloader_pin_memory=true \
  dataloader_num_workers="${DATALOADER_NUM_WORKERS}" \
  dataloader_persistent_workers="${DATALOADER_PERSISTENT_WORKERS}" \
  image_resolution_width=320 \
  image_resolution_height=160 \
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
