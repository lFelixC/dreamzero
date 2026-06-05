#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

export PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/data/checkpoints/dreamzero}"
export OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_ROOT}/dreamzero_droid_wan22_mot_decoupled_full_video}"

resolve_model_path() {
  if [[ -n "${MODEL_PATH:-}" ]]; then
    echo "${MODEL_PATH}"
    return 0
  fi
  if [[ -d "${OUTPUT_DIR}" ]]; then
    local latest
    latest="$(find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)"
    if [[ -n "${latest}" ]]; then
      echo "${latest}"
      return 0
    fi
    if [[ -f "${OUTPUT_DIR}/config.json" ]]; then
      echo "${OUTPUT_DIR}"
      return 0
    fi
  fi
  return 1
}

if ! MODEL_PATH_RESOLVED="$(resolve_model_path)"; then
  echo "ERROR: Set MODEL_PATH or train a checkpoint under OUTPUT_DIR=${OUTPUT_DIR}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
IFS=',' read -r -a CUDA_DEVICE_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#CUDA_DEVICE_ARRAY[@]}}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export MOT_INFERENCE_VIDEO_MODE="decoupled_denoise"
export MOT_DECOUPLED_INFERENCE_VIDEO_FINAL_NOISE="${MOT_DECOUPLED_INFERENCE_VIDEO_FINAL_NOISE:-0.8}"
export MOT_DECOUPLED_INFERENCE_VIDEO_REFRESH_STEPS="1"
export NUM_DIT_STEPS="16"
export DISABLE_TORCH_COMPILE="${DISABLE_TORCH_COMPILE:-true}"
export TORCH_COMPILE_BACKEND="${TORCH_COMPILE_BACKEND:-}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8130}"
INDEX="${INDEX:-9300}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-900}"
HANDSHAKE_TIMEOUT_SECONDS="${HANDSHAKE_TIMEOUT_SECONDS:-0}"

EXTRA_ARGS=()
if [[ -n "${MAX_CHUNK_SIZE_OVERRIDE:-}" ]]; then
  EXTRA_ARGS+=(--max-chunk-size "${MAX_CHUNK_SIZE_OVERRIDE}")
fi

echo "MoT decoupled full-video AR server"
echo "  MODEL_PATH=${MODEL_PATH_RESOLVED}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "  HOST=${HOST}"
echo "  PORT=${PORT}"
echo "  MOT_INFERENCE_VIDEO_MODE=${MOT_INFERENCE_VIDEO_MODE}"
echo "  MOT_DECOUPLED_INFERENCE_VIDEO_REFRESH_STEPS=${MOT_DECOUPLED_INFERENCE_VIDEO_REFRESH_STEPS}"

exec "${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --standalone \
  "${REPO_ROOT}/socket_test_optimized_AR.py" \
  --model-path "${MODEL_PATH_RESOLVED}" \
  --architecture auto \
  --host "${HOST}" \
  --port "${PORT}" \
  --index "${INDEX}" \
  --timeout-seconds "${TIMEOUT_SECONDS}" \
  --handshake-timeout-seconds "${HANDSHAKE_TIMEOUT_SECONDS}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
