#!/usr/bin/env bash
set -euo pipefail

# LR sweep for DROID + Wan2.2 5B full finetune.
#
# This is the canonical entrypoint for both single-node and multi-node sweeps.
# For multi-node runs, run the same command on every node. Keep SWEEP_ID,
# MASTER_ADDR, MASTER_PORT, NNODES, LRS, and training knobs identical on all
# nodes. Only NODE_RANK should differ, with values 0...(NNODES-1).

export PER_DEVICE_BS="${PER_DEVICE_BS:-64}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ -n "${DREAMZERO_ROOT:-}" ] && [ -d "${DREAMZERO_ROOT}/groot" ]; then
  :
elif [ -d "${SCRIPT_REPO_ROOT}/groot" ]; then
  DREAMZERO_ROOT="${SCRIPT_REPO_ROOT}"
else
  echo "ERROR: Could not resolve DREAMZERO_ROOT. Set it to the repo root that contains groot/."
  exit 1
fi
export DREAMZERO_ROOT
cd "${DREAMZERO_ROOT}"

# shellcheck disable=SC1091
source scripts/env/ngc_droid_wan22.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export DEEPSPEED_CFG="${DEEPSPEED_CFG:-zero2}"
export SWEEP_STEPS="${SWEEP_STEPS:-${MAX_STEPS:-1000}}"
export LR_SCHEDULER_STEPS="${LR_SCHEDULER_STEPS:-7000}"
export WARMUP_STEPS="${WARMUP_STEPS:-700}"
export SAVE_STRATEGY="${SAVE_STRATEGY:-no}"
export SAVE_STEPS="${SAVE_STEPS:-1000000}"
export DREAMZERO_SKIP_FINAL_SAVE="${DREAMZERO_SKIP_FINAL_SAVE:-1}"
export LOGGING_STEPS="${LOGGING_STEPS:-1}"
export MAX_CHUNK_SIZE="${MAX_CHUNK_SIZE:-2}"
export REPORT_TO="${REPORT_TO:-wandb}"
export SWANLAB_SYNC_WANDB="${SWANLAB_SYNC_WANDB:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

NNODES="${NNODES:-1}"
resolve_node_rank() {
  local candidate
  for candidate in \
    "${NODE_RANK:-}" \
    "${MACHINE_RANK:-}" \
    "${GROUP_RANK:-}" \
    "${SLURM_NODEID:-}" \
    "${OMPI_COMM_WORLD_NODE_RANK:-}"; do
    if [ -n "${candidate}" ]; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

if NODE_RANK="$(resolve_node_rank)"; then
  :
elif [ "${NNODES}" = "1" ]; then
  NODE_RANK=0
else
  echo "ERROR: Could not resolve NODE_RANK for NNODES=${NNODES}. Set NODE_RANK explicitly or provide MACHINE_RANK, GROUP_RANK, SLURM_NODEID, or OMPI_COMM_WORLD_NODE_RANK."
  exit 1
fi
MASTER_PORT="${MASTER_PORT:-29400}"
if [ -z "${MASTER_ADDR:-}" ]; then
  if [ "${NNODES}" = "1" ]; then
    MASTER_ADDR="127.0.0.1"
  else
    echo "ERROR: MASTER_ADDR must be set to node-0's reachable IP/hostname."
    exit 1
  fi
fi

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#GPU_IDS[@]}"

if ! [[ "${NNODES}" =~ ^[0-9]+$ ]] || [ "${NNODES}" -lt 1 ]; then
  echo "ERROR: NNODES must be a positive integer, got: ${NNODES}"
  exit 1
fi
if ! [[ "${NODE_RANK}" =~ ^[0-9]+$ ]] || [ "${NODE_RANK}" -ge "${NNODES}" ]; then
  echo "ERROR: NODE_RANK must be in [0, NNODES), got NODE_RANK=${NODE_RANK}, NNODES=${NNODES}"
  exit 1
fi
if ! [[ "${MASTER_PORT}" =~ ^[0-9]+$ ]] || [ "${MASTER_PORT}" -lt 1 ]; then
  echo "ERROR: MASTER_PORT must be a positive integer, got: ${MASTER_PORT}"
  exit 1
fi
if ! [[ "${PER_DEVICE_BS}" =~ ^[0-9]+$ ]] || [ "${PER_DEVICE_BS}" -lt 1 ]; then
  echo "ERROR: PER_DEVICE_BS must be a positive integer, got: ${PER_DEVICE_BS}"
  exit 1
fi
if ! [[ "${SWEEP_STEPS}" =~ ^[0-9]+$ ]] || [ "${SWEEP_STEPS}" -lt 1 ]; then
  echo "ERROR: SWEEP_STEPS must be a positive integer for LR sweep, got: ${SWEEP_STEPS}"
  exit 1
fi
if ! [[ "${LR_SCHEDULER_STEPS}" =~ ^[0-9]+$ ]] || [ "${LR_SCHEDULER_STEPS}" -lt 1 ]; then
  echo "ERROR: LR_SCHEDULER_STEPS must be a positive integer, got: ${LR_SCHEDULER_STEPS}"
  exit 1
fi
if ! [[ "${WARMUP_STEPS}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: WARMUP_STEPS must be a non-negative integer, got: ${WARMUP_STEPS}"
  exit 1
fi
if [ "${WARMUP_STEPS}" -gt "${LR_SCHEDULER_STEPS}" ]; then
  echo "ERROR: WARMUP_STEPS must be <= LR_SCHEDULER_STEPS, got WARMUP_STEPS=${WARMUP_STEPS}, LR_SCHEDULER_STEPS=${LR_SCHEDULER_STEPS}"
  exit 1
fi
if [ "${SWEEP_STEPS}" -gt "${LR_SCHEDULER_STEPS}" ]; then
  echo "ERROR: SWEEP_STEPS must be <= LR_SCHEDULER_STEPS, got SWEEP_STEPS=${SWEEP_STEPS}, LR_SCHEDULER_STEPS=${LR_SCHEDULER_STEPS}"
  exit 1
fi
if [ "${NUM_GPUS}" -lt 1 ]; then
  echo "ERROR: CUDA_VISIBLE_DEVICES does not expose any GPU: ${CUDA_VISIBLE_DEVICES}"
  exit 1
fi

WORLD_GPUS=$((NNODES * NUM_GPUS))
GLOBAL_BATCH_SIZE=$((WORLD_GPUS * PER_DEVICE_BS))

LRS="${LRS:-3e-5 5e-5 7e-5 1e-4 1.5e-4 2e-4 3e-4}"
if [ "${NNODES}" != "1" ] && [ -z "${SWEEP_ID:-}" ]; then
  echo "ERROR: Set the same SWEEP_ID on every node, e.g. SWEEP_ID=droid_wan22_4n_bs64_lr_20260504a."
  exit 1
fi
SWEEP_ID="${SWEEP_ID:-droid_wan22_$(date -u +%Y%m%d_%H%M%S)}"
LOG_ROOT="${DREAMZERO_LOG_ROOT:-${DREAMZERO_ROOT}/experiment_logs}"
SWEEP_ROOT="${LOG_ROOT}/lr_sweep"
OUTPUT_ROOT="${DREAMZERO_OUTPUT_ROOT:-/defaultShare/dreamzero_outputs}"
mkdir -p "${SWEEP_ROOT}"

EXPERIMENT_PY="${EXPERIMENT_PY:-${DREAMZERO_ROOT}/groot/vla/experiment/experiment.py}"
DEEPSPEED_CONFIG="${DREAMZERO_ROOT}/groot/vla/configs/deepspeed/${DEEPSPEED_CFG}.json"
MODEL_TARGET_HEIGHT="${MODEL_TARGET_HEIGHT:-320}"
MODEL_TARGET_WIDTH="${MODEL_TARGET_WIDTH:-640}"
MODEL_FRAME_SEQLEN="${MODEL_FRAME_SEQLEN:-200}"

require_file() {
  local path="$1"
  local label="$2"
  if [ ! -f "${path}" ]; then
    echo "ERROR: ${label} not found: ${path}"
    exit 1
  fi
}

require_dir() {
  local path="$1"
  local label="$2"
  if [ ! -d "${path}" ]; then
    echo "ERROR: ${label} not found: ${path}"
    exit 1
  fi
}

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "ERROR: Python is not executable: ${PYTHON_BIN}"
  exit 1
fi
require_file "${EXPERIMENT_PY}" "experiment.py"
require_file "${DEEPSPEED_CONFIG}" "DeepSpeed config"
require_dir "${DROID_DATA_ROOT}" "DROID data root"
require_dir "${WAN22_CKPT_DIR}" "Wan2.2 checkpoint directory"
require_dir "${IMAGE_ENCODER_DIR}" "Wan2.1 image encoder directory"
require_dir "${TOKENIZER_DIR}" "tokenizer directory"
require_file "${WAN22_CKPT_DIR}/models_t5_umt5-xxl-enc-bf16.pth" "Wan2.2 text encoder"
require_file "${WAN22_CKPT_DIR}/Wan2.2_VAE.pth" "Wan2.2 VAE"
require_file "${IMAGE_ENCODER_DIR}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" "Wan2.1 image encoder"

echo "========== LR sweep launch =========="
echo "HOSTNAME=${HOSTNAME:-unknown}"
echo "SWEEP_ID=${SWEEP_ID}"
echo "LRS=${LRS}"
echo "NNODES=${NNODES}"
echo "NODE_RANK=${NODE_RANK}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NUM_GPUS(local)=${NUM_GPUS}"
echo "WORLD_GPUS(total)=${WORLD_GPUS}"
echo "PER_DEVICE_BS=${PER_DEVICE_BS}"
echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
echo "SWEEP_STEPS=${SWEEP_STEPS}"
echo "LR_SCHEDULER_STEPS=${LR_SCHEDULER_STEPS}"
echo "WARMUP_STEPS=${WARMUP_STEPS}"
echo "DEEPSPEED_CFG=${DEEPSPEED_CFG}"
echo "SAVE_STRATEGY=${SAVE_STRATEGY}"
echo "LOG_ROOT=${LOG_ROOT}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "====================================="

sanitize_lr() {
  echo "$1" | sed 's/+//g; s/-/m/g; s/\./p/g'
}

for lr in ${LRS}; do
  lr_name="$(sanitize_lr "${lr}")"
  run_name="lr_${lr_name}_${SWEEP_ID}_bs${PER_DEVICE_BS}_nodes${NNODES}_gpus${WORLD_GPUS}_320x640_sweep${SWEEP_STEPS}_sched${LR_SCHEDULER_STEPS}"
  run_dir="${SWEEP_ROOT}/${run_name}"
  output_dir="${OUTPUT_ROOT}/${run_name}"
  mkdir -p "${run_dir}" "${output_dir}"

  {
    echo "run_name=${run_name}"
    echo "learning_rate=${lr}"
    echo "per_device_bs=${PER_DEVICE_BS}"
    echo "global_batch_size=${GLOBAL_BATCH_SIZE}"
    echo "nnodes=${NNODES}"
    echo "node_rank=${NODE_RANK}"
    echo "num_gpus_per_node=${NUM_GPUS}"
    echo "world_gpus=${WORLD_GPUS}"
    echo "sweep_steps=${SWEEP_STEPS}"
    echo "lr_scheduler_steps=${LR_SCHEDULER_STEPS}"
    echo "warmup_steps=${WARMUP_STEPS}"
    echo "deepspeed=${DEEPSPEED_CFG}"
    echo "output_dir=${output_dir}"
    echo "python_bin=${PYTHON_BIN}"
    echo "master_addr=${MASTER_ADDR}"
    echo "master_port=${MASTER_PORT}"
  } > "${run_dir}/run.env.node${NODE_RANK}"

  echo "Starting LR=${lr} as ${run_name}"

  TRAIN_OVERRIDES=(
    "report_to=${REPORT_TO}"
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
    "training_args.learning_rate=${lr}"
    "training_args.deepspeed=groot/vla/configs/deepspeed/${DEEPSPEED_CFG}.json"
    "save_steps=${SAVE_STEPS}"
    "warmup_ratio=0.0"
    "warmup_steps=${WARMUP_STEPS}"
    "training_args.warmup_ratio=0.0"
    "training_args.warmup_steps=${WARMUP_STEPS}"
    "logging_steps=${LOGGING_STEPS}"
    "output_dir=${output_dir}"
    "per_device_train_batch_size=${PER_DEVICE_BS}"
    "global_batch_size=${GLOBAL_BATCH_SIZE}"
    "max_steps=${LR_SCHEDULER_STEPS}"
    "stop_after_steps=${SWEEP_STEPS}"
    "weight_decay=1e-5"
    "save_total_limit=10"
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
    "max_chunk_size=${MAX_CHUNK_SIZE}"
    "save_strategy=${SAVE_STRATEGY}"
    "droid_data_root=${DROID_DATA_ROOT}"
    "dit_version=${WAN22_CKPT_DIR}"
    "text_encoder_pretrained_path=${WAN22_CKPT_DIR}/models_t5_umt5-xxl-enc-bf16.pth"
    "image_encoder_pretrained_path=${IMAGE_ENCODER_DIR}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
    "vae_pretrained_path=${WAN22_CKPT_DIR}/Wan2.2_VAE.pth"
    "tokenizer_path=${TOKENIZER_DIR}"
  )

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  DREAMZERO_METRICS_DIR="${SWEEP_ROOT}" \
  DREAMZERO_METRICS_RUN_NAME="${run_name}" \
  "${PYTHON_BIN}" -m torch.distributed.run \
    --nnodes="${NNODES}" \
    --nproc-per-node="${NUM_GPUS}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${EXPERIMENT_PY}" \
    "${TRAIN_OVERRIDES[@]}" \
    2>&1 | tee "${run_dir}/stdout.node${NODE_RANK}.log"
done
