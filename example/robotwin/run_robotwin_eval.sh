#!/usr/bin/env bash
set -euo pipefail

# Unified RoboTwin parallel eval launcher.
#
# Recommended:
#   SERVER_GPU=6,7 CLIENT_GPU=7 TASK=beat_block_hammer NUM_ENVS=8 \
#     bash example/robotwin/run_robotwin_eval.sh
#
# Task selection:
#   TASK=beat_block_hammer
#   TASKS=beat_block_hammer,pick_dual_bottles
#   TASKS=all

REPO_ROOT="${REPO_ROOT:-/data/dreamzero_mot}"
DREAMZERO_TORCHRUN="${DREAMZERO_TORCHRUN:-/data/dreamzero/.venv/bin/torchrun}"
DREAMZERO_PYTHON="${DREAMZERO_PYTHON:-/data/dreamzero/.venv/bin/python}"
ROBOTWIN_ENV="${ROBOTWIN_ENV:-/data/envs/robotwin310}"
ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-${ROBOTWIN_ENV}/bin/python}"

cd "${REPO_ROOT}"

usage() {
  cat >&2 <<'EOF'
Usage:
  SERVER_GPU=6,7 CLIENT_GPU=7 TASK=beat_block_hammer NUM_ENVS=8 \
    bash example/robotwin/run_robotwin_eval.sh

Task selection:
  TASK=beat_block_hammer
  TASKS=beat_block_hammer,pick_dual_bottles
  TASKS=all
  LIST_TASKS=1 TASKS=all bash example/robotwin/run_robotwin_eval.sh

GPU selection:
  SERVER_GPU=6,7  DreamZero websocket server GPUs. SERVER_NPROC defaults to GPU count.
  CLIENT_GPU=7    RoboTwin env worker GPUs. Multiple workers share/round-robin this list.

Common knobs:
  NUM_ENVS=8 EPISODES=8 SAVE_VIDEO=0 OPEN_LOOP_HORIZON=8 PORT=8100

Legacy aliases still accepted:
  SERVER_CUDA -> SERVER_GPU
  ENV_CUDA / ENV_GPU / CLIENT_CUDA -> CLIENT_GPU
EOF
}

if (($# > 0)); then
  echo "Positional MODE arguments are no longer supported; this launcher always runs parallel eval." >&2
  usage
  exit 2
fi

wait_for_port() {
  local host="$1"
  local port="$2"
  local timeout="$3"
  local pid="${4:-}"
  "${DREAMZERO_PYTHON}" - "$host" "$port" "$timeout" "$pid" <<'PY'
import os
import socket
import sys
import time

host = sys.argv[1]
port = int(sys.argv[2])
timeout = float(sys.argv[3])
pid = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else None
deadline = time.time() + timeout
last_error = None
while time.time() < deadline:
    if pid is not None:
        try:
            os.kill(pid, 0)
        except OSError:
            raise SystemExit(f"Server process {pid} exited before {host}:{port} was ready")
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
    echo "Port ${host}:${port} is already accepting connections. Stop it or choose another PORT." >&2
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
  if [[ -z "${root_pid}" ]] || ! kill -0 "${root_pid}" 2>/dev/null; then
    return
  fi
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
    if [[ -z "${pid}" ]] || ! kill -0 "${pid}" 2>/dev/null; then
      continue
    fi
    pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d ' ')"
    if [[ -n "${pgid}" ]]; then
      kill -KILL "-${pgid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
    else
      kill -KILL "${pid}" 2>/dev/null || true
    fi
  done
}

now_seconds() {
  "${DREAMZERO_PYTHON}" - <<'PY'
import time
print(repr(time.time()))
PY
}

csv_count() {
  local raw="$1"
  local IFS=','
  local parts=()
  local count=0
  read -ra parts <<<"${raw}"
  local item
  for item in "${parts[@]}"; do
    item="${item//[[:space:]]/}"
    if [[ -n "${item}" ]]; then
      count=$((count + 1))
    fi
  done
  printf '%s\n' "${count}"
}

resolve_ckpt() {
  local ckpt="$1"
  if [[ -d "${ckpt}/checkpoint-10000" && ! -f "${ckpt}/config.json" ]]; then
    ckpt="${ckpt}/checkpoint-10000"
  fi
  printf '%s\n' "${ckpt}"
}

check_common_bins() {
  local need_server="${1:-1}"
  if [[ "${need_server}" == "1" && ! -x "${DREAMZERO_TORCHRUN}" ]]; then
    echo "Missing DreamZero torchrun: ${DREAMZERO_TORCHRUN}" >&2
    exit 1
  fi
  if [[ ! -x "${DREAMZERO_PYTHON}" ]]; then
    echo "Missing DreamZero python: ${DREAMZERO_PYTHON}" >&2
    exit 1
  fi
  if [[ ! -x "${ROBOTWIN_PYTHON}" ]]; then
    echo "Missing RoboTwin python: ${ROBOTWIN_PYTHON}" >&2
    exit 1
  fi
}

print_resolved_tasks() {
  local raw="$1"
  "${DREAMZERO_PYTHON}" - "${raw}" "${REPO_ROOT}/third_party/RoboTwin/task_config/_eval_step_limit.yml" <<'PY'
import sys
from pathlib import Path

raw = sys.argv[1]
limit_path = Path(sys.argv[2])

def all_tasks() -> list[str]:
    if not limit_path.exists():
        raise SystemExit(f"Missing task limit file: {limit_path}")
    tasks = []
    for line in limit_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        task = line.split(":", 1)[0].strip()
        if task:
            tasks.append(task)
    if not tasks:
        raise SystemExit(f"No tasks found in {limit_path}")
    return tasks

parts = [item.strip() for item in raw.replace(",", " ").split() if item.strip()]
if not parts:
    raise SystemExit("TASK/TASKS resolved to an empty task list")
if len(parts) == 1 and parts[0].lower() == "all":
    parts = all_tasks()
elif any(item.lower() == "all" for item in parts):
    raise SystemExit("Use TASKS=all by itself, or provide explicit task names")

print("\n".join(parts))
PY
}

write_timings() {
  local output_root="$1"
  local run_started_at="$2"
  local server_ready_at="$3"
  local controller_started_at="$4"
  local controller_finished_at="$5"
  "${DREAMZERO_PYTHON}" - "${output_root}" "${run_started_at}" "${server_ready_at:-0}" "${controller_started_at:-0}" "${controller_finished_at:-0}" <<'PY'
import json
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
run_started = float(sys.argv[2] or 0)
server_ready = float(sys.argv[3] or 0)
controller_started = float(sys.argv[4] or 0)
controller_finished = float(sys.argv[5] or 0)
payload = {
    "run_started_at": run_started,
    "server_ready_at": server_ready,
    "controller_started_at": controller_started,
    "controller_finished_at": controller_finished,
    "server_startup_time": max(server_ready - run_started, 0.0) if server_ready else 0.0,
    "controller_wall_time": max(controller_finished - controller_started, 0.0) if controller_finished else 0.0,
    "total_wall_time": max(controller_finished - run_started, 0.0) if controller_finished else 0.0,
}
output_root.mkdir(parents=True, exist_ok=True)
(output_root / "timings.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
}

TASKS_RAW="${TASKS:-${TASK:-beat_block_hammer}}"
if [[ "${LIST_TASKS:-0}" == "1" ]]; then
  print_resolved_tasks "${TASKS_RAW}"
  exit 0
fi

if [[ -z "${NUM_ENVS+x}" || -z "${NUM_ENVS}" ]]; then
  NUM_ENVS="8"
  echo "[eval] NUM_ENVS not set; using default NUM_ENVS=${NUM_ENVS}"
fi

SERVER_GPU_VALUE="${SERVER_GPU:-${SERVER_CUDA:-}}"
CLIENT_GPU_VALUE="${CLIENT_GPU:-${ENV_GPU:-${ENV_CUDA:-${CLIENT_CUDA:-}}}}"

if [[ -z "${SERVER_GPU_VALUE}" || -z "${CLIENT_GPU_VALUE}" ]]; then
  echo "SERVER_GPU and CLIENT_GPU must be set to avoid accidentally occupying the wrong GPUs." >&2
  usage
  exit 2
fi

CKPT_RESOLVED="$(resolve_ckpt "${CKPT:-/data/checkpoints/dreamzero/dreamzero_robotwin}")"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/checkpoints/dreamzero/robotwin_eval_runs/parallel_eval}"
EPISODES="${EPISODES:-8}"
OPEN_LOOP_HORIZON="${OPEN_LOOP_HORIZON:-8}"
MAX_CHUNK_SIZE="${MAX_CHUNK_SIZE:-24}"
EPISODE_LENGTH="${EPISODE_LENGTH:-0}"
MAX_STEPS="${MAX_STEPS:-0}"
RESET_RETRIES="${RESET_RETRIES:-5}"
SAVE_VIDEO="${SAVE_VIDEO:-0}"
DRY_RUN_ACTIONS="${DRY_RUN_ACTIONS:-0}"
PROFILE="${PROFILE:-0}"
CLIENT_IMAGE_RESOLUTION="${CLIENT_IMAGE_RESOLUTION:-none}"
SERVER_NPROC="${SERVER_NPROC:-$(csv_count "${SERVER_GPU_VALUE}")}"
SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8100}"
MASTER_PORT="${MASTER_PORT:-29610}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-1800}"
KEEP_SERVER="${KEEP_SERVER:-0}"

if [[ "${SAVE_VIDEO}" != "0" ]]; then
  echo "Parallel eval requires SAVE_VIDEO=0" >&2
  exit 1
fi
if [[ "${SERVER_NPROC}" -le 0 ]]; then
  echo "SERVER_GPU must contain at least one GPU index" >&2
  exit 1
fi
if [[ "${DRY_RUN_ACTIONS}" != "1" && ! -d "${CKPT_RESOLVED}" ]]; then
  echo "Missing checkpoint: ${CKPT_RESOLVED}" >&2
  exit 1
fi

check_common_bins "$([[ "${DRY_RUN_ACTIONS}" == "1" ]] && echo 0 || echo 1)"

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/server_outputs"

RUN_STARTED_AT="$(now_seconds)"
SERVER_READY_AT=""
CONTROLLER_STARTED_AT=""
CONTROLLER_FINISHED_AT=""
SERVER_PID=""

cleanup_server() {
  if [[ "${KEEP_SERVER}" == "1" ]]; then
    echo "[eval] KEEP_SERVER=1, leaving server pid=${SERVER_PID}"
    return
  fi
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[eval] stopping server pid=${SERVER_PID}"
    terminate_pid_tree "${SERVER_PID}" || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup_server EXIT

start_server() {
  local server_log="${OUTPUT_ROOT}/logs/server_${PORT}.log"
  require_port_free "${HOST}" "${PORT}"
  echo "[server] start port=${PORT} server_gpu=${SERVER_GPU_VALUE} nproc=${SERVER_NPROC} master_port=${MASTER_PORT}"
  setsid env CUDA_VISIBLE_DEVICES="${SERVER_GPU_VALUE}" \
    "${DREAMZERO_TORCHRUN}" \
      --nnodes 1 \
      --nproc_per_node "${SERVER_NPROC}" \
      --master_addr 127.0.0.1 \
      --master_port "${MASTER_PORT}" \
      socket_test_optimized_aloha_x5lite_bimanual.py \
      --model_path "${CKPT_RESOLVED}" \
      --host "${SERVER_HOST}" \
      --port "${PORT}" \
      --max-chunk-size "${MAX_CHUNK_SIZE}" \
      > "${server_log}" 2>&1 &
  SERVER_PID="$!"
  if ! wait_for_port "${HOST}" "${PORT}" "${SERVER_TIMEOUT}" "${SERVER_PID}"; then
    echo "[server] failed to become ready; last log lines:" >&2
    tail -120 "${server_log}" >&2 || true
    exit 1
  fi
  SERVER_READY_AT="$(now_seconds)"
  echo "[server] ready port=${PORT}"
}

run_controller() {
  local dry_args=()
  local profile_args=()
  if [[ "${DRY_RUN_ACTIONS}" == "1" ]]; then
    dry_args=(--dry-run-actions)
  fi
  if [[ "${PROFILE}" == "1" ]]; then
    profile_args=(--profile)
  fi

  echo "[controller] tasks=${TASKS_RAW} episodes=${EPISODES} num_envs=${NUM_ENVS} client_gpu=${CLIENT_GPU_VALUE}"
  CONTROLLER_STARTED_AT="$(now_seconds)"
  PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/RoboTwin:${REPO_ROOT}/third_party/lerobot/src:/data/openpi/packages/openpi-client/src:${PYTHONPATH:-}" \
    "${ROBOTWIN_PYTHON}" example/robotwin/parallel_eval.py \
      --remote-host "${HOST}" \
      --remote-port "${PORT}" \
      --tasks "${TASKS_RAW}" \
      --episodes "${EPISODES}" \
      --num-envs "${NUM_ENVS}" \
      --env-cuda "${CLIENT_GPU_VALUE}" \
      --worker-python "${ROBOTWIN_PYTHON}" \
      --output-dir "${OUTPUT_ROOT}" \
      --episode-length "${EPISODE_LENGTH}" \
      --max-steps "${MAX_STEPS}" \
      --open-loop-horizon "${OPEN_LOOP_HORIZON}" \
      --reset-retries "${RESET_RETRIES}" \
      --checkpoint-label "dreamzero_robotwin_parallel_eval" \
      --checkpoint-path "${CKPT_RESOLVED}" \
      --client-image-resolution "${CLIENT_IMAGE_RESOLUTION}" \
      "${profile_args[@]}" \
      "${dry_args[@]}"
  CONTROLLER_FINISHED_AT="$(now_seconds)"
  write_timings "${OUTPUT_ROOT}" "${RUN_STARTED_AT}" "${SERVER_READY_AT:-0}" "${CONTROLLER_STARTED_AT}" "${CONTROLLER_FINISHED_AT}"
}

echo "[eval] ckpt=${CKPT_RESOLVED}"
echo "[eval] output_root=${OUTPUT_ROOT}"
echo "[eval] server_gpu=${SERVER_GPU_VALUE} server_nproc=${SERVER_NPROC} client_gpu=${CLIENT_GPU_VALUE} port=${PORT}"
echo "[eval] tasks=${TASKS_RAW} episodes=${EPISODES} num_envs=${NUM_ENVS} open_loop_horizon=${OPEN_LOOP_HORIZON}"
echo "[eval] profile=${PROFILE} client_image_resolution=${CLIENT_IMAGE_RESOLUTION}"

if [[ "${DRY_RUN_ACTIONS}" == "1" ]]; then
  echo "[server] DRY_RUN_ACTIONS=1, skipping websocket server"
  SERVER_READY_AT="${RUN_STARTED_AT}"
else
  start_server
fi

run_controller
echo "[eval] done"
cleanup_server
trap - EXIT
