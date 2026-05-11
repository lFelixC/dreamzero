#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/dreamzero_mot}"
ROBOTWIN_ENV="${ROBOTWIN_ENV:-/data/envs/robotwin310}"
DREAMZERO_PYTHON="${DREAMZERO_PYTHON:-/data/dreamzero/.venv/bin/python}"
DREAMZERO_TORCHRUN="${DREAMZERO_TORCHRUN:-/data/dreamzero/.venv/bin/torchrun}"
ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-${ROBOTWIN_ENV}/bin/python}"

JOINT_CKPT="${JOINT_CKPT:-/data/checkpoints/dreamzero/dreamzero_robotwin_wan22_joint_smoke/checkpoint-2}"
MOT_CKPT="${MOT_CKPT:-/data/checkpoints/dreamzero/dreamzero_robotwin_wan22_mot_smoke/checkpoint-2}"
CKPT="${CKPT:-both}"

SERVER_CUDA="${SERVER_CUDA:-1}"
CLIENT_CUDA="${CLIENT_CUDA:-2}"
HOST="${HOST:-127.0.0.1}"
SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-1800}"
SERVER_MAX_CHUNK_SIZE="${SERVER_MAX_CHUNK_SIZE:-4}"

ROBOTWIN_TASK="${ROBOTWIN_TASK:-beat_block_hammer}"
EPISODES="${EPISODES:-1}"
EPISODE_LENGTH="${EPISODE_LENGTH:-300}"
MAX_STEPS="${MAX_STEPS:-300}"
OPEN_LOOP_HORIZON="${OPEN_LOOP_HORIZON:-1}"
SAVE_VIDEO="${SAVE_VIDEO:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/checkpoints/dreamzero/robotwin_eval_runs}"

cd "${REPO_ROOT}"

if [[ ! -x "${DREAMZERO_TORCHRUN}" ]]; then
  echo "Missing DreamZero torchrun: ${DREAMZERO_TORCHRUN}" >&2
  exit 1
fi
if [[ ! -x "${ROBOTWIN_PYTHON}" ]]; then
  echo "Missing RoboTwin python: ${ROBOTWIN_PYTHON}" >&2
  exit 1
fi

wait_for_port() {
  local host="$1"
  local port="$2"
  local timeout="$3"
  local server_pid="${4:-}"
  "${DREAMZERO_PYTHON}" - "$host" "$port" "$timeout" "$server_pid" <<'PY'
import os
import socket
import sys
import time

host = sys.argv[1]
port = int(sys.argv[2])
timeout = float(sys.argv[3])
server_pid = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else None
deadline = time.time() + timeout
last_error = None
while time.time() < deadline:
    if server_pid is not None:
        try:
            os.kill(server_pid, 0)
        except OSError:
            raise SystemExit(f"Server process {server_pid} exited before {host}:{port} was ready")
    try:
        with socket.create_connection((host, port), timeout=3):
            sys.exit(0)
    except OSError as exc:
        last_error = exc
        time.sleep(5)
raise SystemExit(f"Timed out waiting for {host}:{port}: {last_error}")
PY
}

require_port_free() {
  local host="$1"
  local port="$2"
  if "${DREAMZERO_PYTHON}" - "$host" "$port" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
try:
    with socket.create_connection((host, port), timeout=2):
        sys.exit(0)
except OSError:
    sys.exit(1)
PY
  then
    echo "Port ${host}:${port} is already accepting connections. Stop the old server or choose PORT=..." >&2
    exit 1
  fi
}

collect_pid_tree() {
  local root_pid="$1"
  local queue=("${root_pid}")
  local all=()
  while ((${#queue[@]})); do
    local pid="${queue[0]}"
    queue=("${queue[@]:1}")
    all+=("${pid}")
    if command -v pgrep >/dev/null 2>&1; then
      local children=()
      mapfile -t children < <(pgrep -P "${pid}" 2>/dev/null || true)
      if ((${#children[@]})); then
        queue+=("${children[@]}")
      fi
    fi
  done
  printf '%s\n' "${all[@]}"
}

terminate_pid_tree() {
  local root_pid="$1"
  local pids=()
  mapfile -t pids < <(collect_pid_tree "${root_pid}")
  local pid pgid
  for pid in "${pids[@]}"; do
    if [[ -z "${pid}" ]] || ! kill -0 "${pid}" 2>/dev/null; then
      continue
    fi
    pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d ' ')"
    if [[ -n "${pgid}" ]]; then
      kill -TERM "-${pgid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
    else
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  sleep 5
  for pid in "${pids[@]}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d ' ')"
      if [[ -n "${pgid}" ]]; then
        kill -KILL "-${pgid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
      else
        kill -KILL "${pid}" 2>/dev/null || true
      fi
    fi
  done
  return 0
}

run_one() {
  local label="$1"
  local ckpt="$2"
  local output_dir="${OUTPUT_ROOT}/${label}"
  local log_dir="${output_dir}/logs"
  local server_log="${log_dir}/server.log"
  local client_log="${log_dir}/client.log"
  local server_pid=""

  if [[ ! -d "${ckpt}" ]]; then
    echo "Missing checkpoint for ${label}: ${ckpt}" >&2
    exit 1
  fi

  mkdir -p "${log_dir}"
  require_port_free "${HOST}" "${PORT}"
  echo "[robotwin] Starting ${label} server from ${ckpt} on GPU ${SERVER_CUDA}, port ${PORT}"
  setsid env CUDA_VISIBLE_DEVICES="${SERVER_CUDA}" \
    "${DREAMZERO_TORCHRUN}" --standalone --nproc_per_node 1 \
      socket_test_optimized_aloha_x5lite_bimanual.py \
      --model_path "${ckpt}" \
      --host "${SERVER_HOST}" \
      --port "${PORT}" \
      --max-chunk-size "${SERVER_MAX_CHUNK_SIZE}" \
      --output-root "${output_dir}/server_outputs" \
      >"${server_log}" 2>&1 &
  server_pid="$!"

  cleanup_server() {
    if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
      echo "[robotwin] Stopping ${label} server"
      terminate_pid_tree "${server_pid}" || true
      wait "${server_pid}" 2>/dev/null || true
    fi
  }
  trap cleanup_server RETURN

  if ! wait_for_port "${HOST}" "${PORT}" "${SERVER_TIMEOUT}" "${server_pid}"; then
    echo "[robotwin] ${label} server failed to become ready; last log lines:" >&2
    tail -80 "${server_log}" >&2 || true
    return 1
  fi
  sleep 5

  local save_video_args=()
  if [[ "${SAVE_VIDEO}" == "1" ]]; then
    save_video_args=(--save-video)
  fi

  echo "[robotwin] Running ${label} RoboTwin rollout on GPU ${CLIENT_CUDA}"
  local client_status=0
  env CUDA_VISIBLE_DEVICES="${CLIENT_CUDA}" \
    PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/RoboTwin:${REPO_ROOT}/third_party/lerobot/src:/data/openpi/packages/openpi-client/src:${PYTHONPATH:-}" \
    "${ROBOTWIN_PYTHON}" example/robotwin/client.py \
      --mode robotwin \
      --remote-host "${HOST}" \
      --remote-port "${PORT}" \
      --episodes "${EPISODES}" \
      --robotwin-task "${ROBOTWIN_TASK}" \
      --episode-length "${EPISODE_LENGTH}" \
      --max-steps "${MAX_STEPS}" \
      --open-loop-horizon "${OPEN_LOOP_HORIZON}" \
      --output-dir "${output_dir}" \
      --checkpoint-label "${label}" \
      --checkpoint-path "${ckpt}" \
      "${save_video_args[@]}" \
      >"${client_log}" 2>&1 || client_status="$?"

  if [[ "${client_status}" != "0" ]]; then
    echo "[robotwin] ${label} client failed with status ${client_status}; last client log lines:" >&2
    tail -120 "${client_log}" >&2 || true
    return "${client_status}"
  fi

  echo "[robotwin] ${label} done. Results: ${output_dir}"
}

case "${CKPT}" in
  both)
    run_one "joint_smoke" "${JOINT_CKPT}"
    run_one "mot_smoke" "${MOT_CKPT}"
    ;;
  joint)
    run_one "joint_smoke" "${JOINT_CKPT}"
    ;;
  mot)
    run_one "mot_smoke" "${MOT_CKPT}"
    ;;
  *)
    run_one "custom" "${CKPT}"
    ;;
esac
