#!/usr/bin/env bash
set -euo pipefail

# Task-level LingBot-style RoboTwin eval launcher.
#
# Each task gets an independent run_robotwin_eval.sh process with its own
# websocket server, port, master port, output directory, and GPU slot.

REPO_ROOT="${REPO_ROOT:-/data/dreamzero_mot}"
cd "${REPO_ROOT}"

usage() {
  cat >&2 <<'EOF'
Usage:
  SERVER_GPU_GROUPS='0;1;2;3' CLIENT_GPU_GROUPS='4;5;6;7' TASKS=all \
    bash example/robotwin/run_robotwin_eval_multigpu.sh

GPU slots:
  SERVER_GPU_GROUPS  Semicolon-separated server GPU groups. Use commas inside a group, e.g. '0,1;2,3'.
  CLIENT_GPU_GROUPS  Semicolon-separated RoboTwin env GPU groups, one per server group.

Aliases:
  SERVER_GPUS='0,1,2,3' is treated as four one-GPU server groups.
  CLIENT_GPUS='4,5,6,7' is treated as four one-GPU client groups.
  For multi-GPU server slots, use SERVER_GPU_GROUPS='0,1;2,3'.

Common knobs are inherited by run_robotwin_eval.sh:
  CKPT EPISODES SEED_START ROBOTWIN_TASK_CONFIG MAX_STEPS SAVE_VIDEO
  START_PORT START_MASTER_PORT OUTPUT_ROOT
EOF
}

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

parse_groups() {
  local raw="$1"
  local split_mode="${3:-auto}"
  local -n out="$2"
  local parts=()
  out=()
  if [[ "${split_mode}" == "semicolon" ]]; then
    local IFS=';'
    read -ra parts <<<"${raw}"
  elif [[ "${split_mode}" == "comma" ]]; then
    local IFS=','
    read -ra parts <<<"${raw}"
  elif [[ "${raw}" == *";"* ]]; then
    local IFS=';'
    read -ra parts <<<"${raw}"
  else
    local IFS=','
    read -ra parts <<<"${raw}"
  fi
  local item
  for item in "${parts[@]}"; do
    item="$(trim "${item}")"
    if [[ -n "${item}" ]]; then
      out+=("${item}")
    fi
  done
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if (($# > 0)); then
  echo "This launcher accepts environment variables only." >&2
  usage
  exit 2
fi

TASKS_RAW="${TASKS:-${TASK:-all}}"
BASE_OUTPUT_ROOT="${OUTPUT_ROOT:-/data/checkpoints/dreamzero/robotwin_eval_runs/lingbot_style_task_parallel}"
START_PORT="${START_PORT:-8200}"
START_MASTER_PORT="${START_MASTER_PORT:-29700}"
CKPT_VALUE="${CKPT:-/data/checkpoints/dreamzero/dreamzero_robotwin/checkpoint-30000}"
EPISODES_VALUE="${EPISODES:-100}"
SEED_START_VALUE="${SEED_START:-10000}"
LOG_ROOT="${LOG_ROOT:-${BASE_OUTPUT_ROOT}/logs}"

SERVER_GROUPS_RAW="${SERVER_GPU_GROUPS:-${SERVER_GPUS:-${SERVER_GPU:-}}}"
CLIENT_GROUPS_RAW="${CLIENT_GPU_GROUPS:-${CLIENT_GPUS:-${CLIENT_GPU:-}}}"
if [[ -z "${SERVER_GROUPS_RAW}" || -z "${CLIENT_GROUPS_RAW}" ]]; then
  echo "SERVER_GPU_GROUPS/SERVER_GPUS and CLIENT_GPU_GROUPS/CLIENT_GPUS are required." >&2
  usage
  exit 2
fi

SERVER_SLOTS=()
CLIENT_SLOTS=()
if [[ -n "${SERVER_GPU_GROUPS:-}" ]]; then
  parse_groups "${SERVER_GROUPS_RAW}" SERVER_SLOTS semicolon
else
  parse_groups "${SERVER_GROUPS_RAW}" SERVER_SLOTS comma
fi
if [[ -n "${CLIENT_GPU_GROUPS:-}" ]]; then
  parse_groups "${CLIENT_GROUPS_RAW}" CLIENT_SLOTS semicolon
else
  parse_groups "${CLIENT_GROUPS_RAW}" CLIENT_SLOTS comma
fi
if ((${#SERVER_SLOTS[@]} == 0 || ${#CLIENT_SLOTS[@]} == 0)); then
  echo "Resolved GPU slot lists must be non-empty." >&2
  exit 2
fi
if ((${#SERVER_SLOTS[@]} != ${#CLIENT_SLOTS[@]})); then
  echo "SERVER and CLIENT slot counts must match: server=${#SERVER_SLOTS[@]} client=${#CLIENT_SLOTS[@]}" >&2
  exit 2
fi

mkdir -p "${LOG_ROOT}" "${BASE_OUTPUT_ROOT}"

mapfile -t TASK_NAMES < <(LIST_TASKS=1 TASKS="${TASKS_RAW}" bash example/robotwin/run_robotwin_eval.sh)
if ((${#TASK_NAMES[@]} == 0)); then
  echo "No tasks resolved from TASKS=${TASKS_RAW}" >&2
  exit 2
fi

slot_count="${#SERVER_SLOTS[@]}"
timestamp="$(date +%Y%m%d_%H%M%S)"
pid_file="${LOG_ROOT}/pids_${timestamp}.txt"
status_file="${LOG_ROOT}/status_${timestamp}.txt"
: >"${pid_file}"
: >"${status_file}"

echo "[multi] tasks=${#TASK_NAMES[@]} slots=${slot_count} output_root=${BASE_OUTPUT_ROOT}"
echo "[multi] ckpt=${CKPT_VALUE} episodes=${EPISODES_VALUE} seed_start=${SEED_START_VALUE}"

task_index=0
while ((task_index < ${#TASK_NAMES[@]})); do
  batch_pids=()
  batch_tasks=()
  for ((slot = 0; slot < slot_count && task_index < ${#TASK_NAMES[@]}; slot++)); do
    task_name="${TASK_NAMES[$task_index]}"
    server_gpu="${SERVER_SLOTS[$slot]}"
    client_gpu="${CLIENT_SLOTS[$slot]}"
    port=$((START_PORT + task_index))
    master_port=$((START_MASTER_PORT + task_index))
    task_output="${BASE_OUTPUT_ROOT}/$(printf '%03d' "${task_index}")_${task_name}"
    log_file="${LOG_ROOT}/${task_index}_${task_name}_${timestamp}.log"
    server_nproc="$(python - "${server_gpu}" <<'PY'
import sys
print(len([item for item in sys.argv[1].split(",") if item.strip()]))
PY
)"

    echo "[multi] launch task=${task_name} slot=${slot} server_gpu=${server_gpu} client_gpu=${client_gpu} port=${port} master_port=${master_port}"
    env \
      TASK="${task_name}" \
      TASKS="" \
      EVAL_MODE=lingbot \
      NUM_ENVS=1 \
      CKPT="${CKPT_VALUE}" \
      EPISODES="${EPISODES_VALUE}" \
      SEED_START="${SEED_START_VALUE}" \
      SERVER_GPU="${server_gpu}" \
      SERVER_NPROC="${server_nproc}" \
      CLIENT_GPU="${client_gpu}" \
      PORT="${port}" \
      MASTER_PORT="${master_port}" \
      OUTPUT_ROOT="${task_output}" \
      bash example/robotwin/run_robotwin_eval.sh >"${log_file}" 2>&1 &
    pid=$!
    echo "${pid} ${task_name} ${log_file}" >>"${pid_file}"
    batch_pids+=("${pid}")
    batch_tasks+=("${task_name}")
    task_index=$((task_index + 1))
  done

  batch_failed=0
  for i in "${!batch_pids[@]}"; do
    pid="${batch_pids[$i]}"
    task_name="${batch_tasks[$i]}"
    if wait "${pid}"; then
      echo "ok ${task_name}" | tee -a "${status_file}"
    else
      echo "failed ${task_name}" | tee -a "${status_file}"
      batch_failed=1
    fi
  done
  if ((batch_failed)); then
    echo "[multi] at least one task failed; see ${LOG_ROOT}" >&2
    exit 1
  fi
done

python - "${BASE_OUTPUT_ROOT}" "${#TASK_NAMES[@]}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = int(sys.argv[2])
rows = []
for report in sorted(root.glob("*/report.json")):
    data = json.loads(report.read_text(encoding="utf-8"))
    task = report.parent.name.split("_", 1)[1] if "_" in report.parent.name else report.parent.name
    rows.append(
        {
            "task": task,
            "episodes": data.get("episodes", 0),
            "successes": data.get("successes", 0),
            "success_rate": data.get("success_rate", 0.0),
            "report": report.as_posix(),
        }
    )
episodes = sum(int(row["episodes"]) for row in rows)
successes = sum(int(row["successes"]) for row in rows)
summary = {
    "mode": "lingbot_style_task_parallel",
    "tasks_expected": expected,
    "tasks_finished": len(rows),
    "episodes": episodes,
    "successes": successes,
    "success_rate": successes / episodes if episodes else 0.0,
    "rows": rows,
}
(root / "task_parallel_report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

echo "[multi] done"
