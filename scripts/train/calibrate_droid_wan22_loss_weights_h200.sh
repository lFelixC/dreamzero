#!/bin/bash
set -euo pipefail

# Two-stage loss-weight calibration for one 8x H200 node.
#
# Stage 1: run Joint and MoT with equal loss weights and log per-loss grad norms.
# Stage 2: compute action_loss_weight ~= dynamics_grad_norm / action_grad_norm,
#          then run Joint and MoT confirmation jobs with the suggested weights.
#
# This is intentionally not a blind grid sweep. Keep the batch/data settings here
# aligned with the setting you want to use for the actual comparison.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LAUNCHER="${LAUNCHER:-${SCRIPT_DIR}/droid_wan22_fseq200.sh}"
SUGGEST_PY="${SUGGEST_PY:-${SCRIPT_DIR}/suggest_loss_weight_from_grad_norm.py}"

if [[ ! -f "${LAUNCHER}" ]]; then
  echo "ERROR: DROID launcher not found: ${LAUNCHER}"
  exit 1
fi
if [[ ! -f "${SUGGEST_PY}" ]]; then
  echo "ERROR: suggestion helper not found: ${SUGGEST_PY}"
  exit 1
fi

if [[ -d /data/checkpoints/dreamzero ]]; then
  DEFAULT_CHECKPOINT_ROOT=/data/checkpoints/dreamzero
else
  DEFAULT_CHECKPOINT_ROOT="${REPO_ROOT}/checkpoints"
fi

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${DEFAULT_CHECKPOINT_ROOT}}"
SWEEP_NAME="${SWEEP_NAME:-droid_wan22_loss_weight_calib_h200}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${CHECKPOINT_ROOT}/loss_weight_calibration/${SWEEP_NAME}}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/logs}"

GPU_GROUP_JOINT="${GPU_GROUP_JOINT:-0,1,2,3}"
GPU_GROUP_MOT="${GPU_GROUP_MOT:-4,5,6,7}"
MASTER_PORT_JOINT="${MASTER_PORT_JOINT:-29610}"
MASTER_PORT_MOT="${MASTER_PORT_MOT:-29620}"

CALIBRATION_STEPS="${CALIBRATION_STEPS:-3000}"
CONFIRM_STEPS="${CONFIRM_STEPS:-8000}"
SKIP_FIRST_STEPS="${SKIP_FIRST_STEPS:-300}"
GRAD_CONFLICT_LOGGING_STEPS="${GRAD_CONFLICT_LOGGING_STEPS:-20}"
SUGGEST_STEP_MIN="${SUGGEST_STEP_MIN:-${SKIP_FIRST_STEPS}}"
SUGGEST_STEP_MAX="${SUGGEST_STEP_MAX:-$((CALIBRATION_STEPS * 4 / 5))}"

PER_DEVICE_BS="${PER_DEVICE_BS:-8}"
DEEPSPEED_CFG="${DEEPSPEED_CFG:-zero2}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
MAX_CHUNK_SIZE="${MAX_CHUNK_SIZE:-4}"
DATASET_SHARD_SAMPLING_RATE="${DATASET_SHARD_SAMPLING_RATE:-0.05}"
DATASET_SHARD_SAMPLING_STRATEGY="${DATASET_SHARD_SAMPLING_STRATEGY:-random}"
DATASET_SHARD_SAMPLING_BLOCK_SIZE="${DATASET_SHARD_SAMPLING_BLOCK_SIZE:-64}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-4}"
USE_GRADIENT_CHECKPOINTING="${USE_GRADIENT_CHECKPOINTING:-true}"
SAVE_STRATEGY="${SAVE_STRATEGY:-no}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
CONFIRM_ONLY="${CONFIRM_ONLY:-false}"
EXTRA_ARGS=("$@")

count_gpu_group() {
  local group="$1"
  local stripped="${group//,/ }"
  # shellcheck disable=SC2086
  set -- ${stripped}
  echo "$#"
}

JOINT_GROUP_SIZE="$(count_gpu_group "${GPU_GROUP_JOINT}")"
MOT_GROUP_SIZE="$(count_gpu_group "${GPU_GROUP_MOT}")"
if [[ "${JOINT_GROUP_SIZE}" != "${MOT_GROUP_SIZE}" ]]; then
  echo "ERROR: GPU_GROUP_JOINT and GPU_GROUP_MOT must have the same number of GPUs for comparable calibration."
  echo "GPU_GROUP_JOINT=${GPU_GROUP_JOINT} (${JOINT_GROUP_SIZE}), GPU_GROUP_MOT=${GPU_GROUP_MOT} (${MOT_GROUP_SIZE})"
  exit 1
fi
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((PER_DEVICE_BS * JOINT_GROUP_SIZE))}"

export SWANLAB_SYNC_WANDB="${SWANLAB_SYNC_WANDB:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export REPORT_TO="${REPORT_TO:-wandb}"
export WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-dreamzero_loss_weight_calibration}"
export DREAMZERO_TRACK_GRAD_CONFLICT=1
export DREAMZERO_GRAD_CONFLICT_LOGGING_STEPS="${GRAD_CONFLICT_LOGGING_STEPS}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

sanitize() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  echo "${value}"
}

run_one() {
  local arch="$1"
  local gpu_group="$2"
  local port="$3"
  local steps="$4"
  local action_weight="$5"
  local tag="$6"
  local safe_action
  local output_dir
  local log_file

  safe_action="$(sanitize "${action_weight}")"
  output_dir="${OUTPUT_ROOT}/${tag}_${arch}_aw${safe_action}_steps${steps}"
  log_file="${LOG_ROOT}/${tag}_${arch}_aw${safe_action}_steps${steps}.log"

  echo "[calib] launch ${tag}/${arch}: GPUs=${gpu_group}, action_weight=${action_weight}, steps=${steps}"
  (
    unset WANDB_RUN_ID
    unset RUNTIME_ID
    export CUDA_VISIBLE_DEVICES="${gpu_group}"
    export GPU_IDS="${gpu_group}"
    export ARCH="${arch}"
    export OUTPUT_DIR="${output_dir}"
    export CHECKPOINT_ROOT="${CHECKPOINT_ROOT}"
    export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
    export MASTER_PORT="${port}"
    export TRAIN_ARCHITECTURE="${TRAIN_ARCHITECTURE:-full}"
    export DYNAMICS_LOSS_WEIGHT=1.0
    export ACTION_LOSS_WEIGHT="${action_weight}"
    export MAX_STEPS="${steps}"
    export SAVE_STEPS="${steps}"
    export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
    export PER_DEVICE_BS="${PER_DEVICE_BS}"
    export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE}"
    export DEEPSPEED_CFG="${DEEPSPEED_CFG}"
    export LEARNING_RATE="${LEARNING_RATE}"
    export WEIGHT_DECAY="${WEIGHT_DECAY}"
    export WARMUP_RATIO="${WARMUP_RATIO}"
    export MAX_CHUNK_SIZE="${MAX_CHUNK_SIZE}"
    export DATASET_SHARD_SAMPLING_RATE="${DATASET_SHARD_SAMPLING_RATE}"
    export DATASET_SHARD_SAMPLING_STRATEGY="${DATASET_SHARD_SAMPLING_STRATEGY}"
    export DATASET_SHARD_SAMPLING_BLOCK_SIZE="${DATASET_SHARD_SAMPLING_BLOCK_SIZE}"
    export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS}"
    export DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR}"
    export USE_GRADIENT_CHECKPOINTING="${USE_GRADIENT_CHECKPOINTING}"
    export SAVE_STRATEGY="${SAVE_STRATEGY}"
    export PREPARE_ASSETS="${PREPARE_ASSETS:-false}"
    exec bash "${LAUNCHER}" "${EXTRA_ARGS[@]}"
  ) > "${log_file}" 2>&1 &
  LAST_RUN_PID="$!"
}

wait_pair() {
  local left_pid="$1"
  local left_name="$2"
  local right_pid="$3"
  local right_name="$4"
  local failed=0

  if wait "${left_pid}"; then
    echo "[calib] finished: ${left_name}"
  else
    echo "[calib] FAILED: ${left_name}"
    failed=1
  fi
  if wait "${right_pid}"; then
    echo "[calib] finished: ${right_name}"
  else
    echo "[calib] FAILED: ${right_name}"
    failed=1
  fi
  if [[ "${failed}" -ne 0 ]]; then
    exit 1
  fi
}

echo "========== DreamZero loss-weight calibration =========="
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "GPU_GROUP_JOINT=${GPU_GROUP_JOINT}"
echo "GPU_GROUP_MOT=${GPU_GROUP_MOT}"
echo "GPUS_PER_JOB=${JOINT_GROUP_SIZE}"
echo "PER_DEVICE_BS=${PER_DEVICE_BS}"
echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
echo "USE_GRADIENT_CHECKPOINTING=${USE_GRADIENT_CHECKPOINTING}"
echo "MAX_CHUNK_SIZE=${MAX_CHUNK_SIZE}"
echo "WARMUP_RATIO=${WARMUP_RATIO}"
echo "CALIBRATION_STEPS=${CALIBRATION_STEPS}"
echo "CONFIRM_STEPS=${CONFIRM_STEPS}"
echo "SUGGEST_STEP_MIN=${SUGGEST_STEP_MIN}"
echo "SUGGEST_STEP_MAX=${SUGGEST_STEP_MAX}"
echo "GRAD_CONFLICT_LOGGING_STEPS=${GRAD_CONFLICT_LOGGING_STEPS}"
echo "DATASET_SHARD_SAMPLING_RATE=${DATASET_SHARD_SAMPLING_RATE}"
echo "SAVE_STRATEGY=${SAVE_STRATEGY}"
echo "CONFIRM_ONLY=${CONFIRM_ONLY}"
echo "======================================================="

joint_calib_dir="${OUTPUT_ROOT}/calib_joint_aw1p0_steps${CALIBRATION_STEPS}"
mot_calib_dir="${OUTPUT_ROOT}/calib_mot_aw1p0_steps${CALIBRATION_STEPS}"

if [[ "${CONFIRM_ONLY}" != "true" ]]; then
  run_one joint "${GPU_GROUP_JOINT}" "${MASTER_PORT_JOINT}" "${CALIBRATION_STEPS}" 1.0 calib
  joint_calib_pid="${LAST_RUN_PID}"
  run_one mot "${GPU_GROUP_MOT}" "${MASTER_PORT_MOT}" "${CALIBRATION_STEPS}" 1.0 calib
  mot_calib_pid="${LAST_RUN_PID}"
  wait_pair "${joint_calib_pid}" "calib/joint" "${mot_calib_pid}" "calib/mot"
else
  echo "[calib] CONFIRM_ONLY=true; reuse calibration logs under ${OUTPUT_ROOT}"
fi

joint_action_weight="${JOINT_ACTION_LOSS_WEIGHT:-}"
if [[ -z "${joint_action_weight}" ]]; then
  joint_action_weight="$(
    "${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}" "${SUGGEST_PY}" \
      --arch joint \
      --log "${joint_calib_dir}" \
      --skip-first-steps "${SKIP_FIRST_STEPS}" \
      --step-min "${SUGGEST_STEP_MIN}" \
      --step-max "${SUGGEST_STEP_MAX}" \
      --format value
  )"
fi

mot_action_weight="${MOT_ACTION_LOSS_WEIGHT:-}"
if [[ -z "${mot_action_weight}" ]]; then
  mot_action_weight="$(
    "${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}" "${SUGGEST_PY}" \
      --arch mot \
      --log "${mot_calib_dir}" \
      --skip-first-steps "${SKIP_FIRST_STEPS}" \
      --step-min "${SUGGEST_STEP_MIN}" \
      --step-max "${SUGGEST_STEP_MAX}" \
      --format value
  )"
fi

echo "[calib] suggested joint ACTION_LOSS_WEIGHT=${joint_action_weight}"
echo "[calib] suggested mot   ACTION_LOSS_WEIGHT=${mot_action_weight}"

"${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}" "${SUGGEST_PY}" \
  --arch joint \
  --log "${joint_calib_dir}" \
  --skip-first-steps "${SKIP_FIRST_STEPS}" \
  --step-min "${SUGGEST_STEP_MIN}" \
  --step-max "${SUGGEST_STEP_MAX}" \
  > "${OUTPUT_ROOT}/joint_suggestion.json"
"${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}" "${SUGGEST_PY}" \
  --arch mot \
  --log "${mot_calib_dir}" \
  --skip-first-steps "${SKIP_FIRST_STEPS}" \
  --step-min "${SUGGEST_STEP_MIN}" \
  --step-max "${SUGGEST_STEP_MAX}" \
  > "${OUTPUT_ROOT}/mot_suggestion.json"

run_one joint "${GPU_GROUP_JOINT}" "${MASTER_PORT_JOINT}" "${CONFIRM_STEPS}" "${joint_action_weight}" confirm
joint_confirm_pid="${LAST_RUN_PID}"
run_one mot "${GPU_GROUP_MOT}" "${MASTER_PORT_MOT}" "${CONFIRM_STEPS}" "${mot_action_weight}" confirm
mot_confirm_pid="${LAST_RUN_PID}"
wait_pair "${joint_confirm_pid}" "confirm/joint" "${mot_confirm_pid}" "confirm/mot"

echo "[calib] done"
echo "[calib] suggestions:"
cat "${OUTPUT_ROOT}/joint_suggestion.json"
cat "${OUTPUT_ROOT}/mot_suggestion.json"
