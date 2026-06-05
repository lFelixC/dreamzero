#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
REMOTE_HOST="${REMOTE_HOST:-127.0.0.1}"
REMOTE_PORT="${REMOTE_PORT:-8130}"
OPEN_LOOP_HORIZON="${OPEN_LOOP_HORIZON:-24}"
CONTROL_FREQUENCY="${CONTROL_FREQUENCY:-15}"
RIGHT_CAMERA_ID="${RIGHT_CAMERA_ID:-36517165}"
LEFT_CAMERA_ID="${LEFT_CAMERA_ID:-}"
WRIST_CAMERA_ID="${WRIST_CAMERA_ID:-13337231}"
MISSING_LEFT_CAMERA_STRATEGY="${MISSING_LEFT_CAMERA_STRATEGY:-mask_left}"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/droid-client:${REPO_ROOT}/droid-client/scripts:${PYTHONPATH:-}"

echo "MoT decoupled full-video real DROID client"
echo "  REMOTE_HOST=${REMOTE_HOST}"
echo "  REMOTE_PORT=${REMOTE_PORT}"
echo "  OPEN_LOOP_HORIZON=${OPEN_LOOP_HORIZON}"
echo "  CONTROL_FREQUENCY=${CONTROL_FREQUENCY}"
echo "  RIGHT_CAMERA_ID=${RIGHT_CAMERA_ID}"
echo "  LEFT_CAMERA_ID=${LEFT_CAMERA_ID:-<none>}"
echo "  WRIST_CAMERA_ID=${WRIST_CAMERA_ID}"
echo "  MISSING_LEFT_CAMERA_STRATEGY=${MISSING_LEFT_CAMERA_STRATEGY}"
echo "  RTC disabled for decoupled_denoise"

ARGS=(
  --remote-host "${REMOTE_HOST}"
  --remote-port "${REMOTE_PORT}"
  --open-loop-horizon "${OPEN_LOOP_HORIZON}"
  --control-frequency "${CONTROL_FREQUENCY}"
  --right-camera-id "${RIGHT_CAMERA_ID}"
  --wrist-camera-id "${WRIST_CAMERA_ID}"
  --missing-left-camera-strategy "${MISSING_LEFT_CAMERA_STRATEGY}"
)

if [[ -n "${LEFT_CAMERA_ID}" ]]; then
  ARGS+=(--left-camera-id "${LEFT_CAMERA_ID}")
fi

exec "${PYTHON_BIN}" "${REPO_ROOT}/droid-client/scripts/main_dreamzero_test.py" \
  "${ARGS[@]}" \
  "$@"
