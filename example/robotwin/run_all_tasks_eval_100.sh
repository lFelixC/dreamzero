#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"

usage() {
    cat <<'EOF'
Usage:
  bash example/robotwin/run_all_tasks_eval_100.sh /path/to/checkpoint

Required:
  arg1                    DreamZero checkpoint path.

Useful optional envs:
  TEST_NUM=100            Episodes per task.
  NUM_GPUS=8              Local GPUs per node. Auto-detected for local mode.
  ROBOTWIN_NODES="ip1 ip2"
                          SSH fan-out nodes. If unset, runs on this node only.
  SSH_USER=root           SSH user for ROBOTWIN_NODES.
  REMOTE_REPO_ROOT=...    Repo path on remote nodes. Defaults to current repo path.
  SLOTS_PER_NODE=8        GPU/client slots per remote node.
  EVAL_RUN_BASE=...       Parent output directory.
  RUN_ID=...              Run name under EVAL_RUN_BASE.
  AUTO_CLEANUP=1          Clean server/client processes on exit, failure, or Ctrl-C.
  PRE_CLEANUP=1           Clean stale server/client processes before starting.
  KEEP_SERVERS=0          Set to 1 to leave policy servers running after eval.
  EPISODE_TIMEOUT_SEC=900 Episode timeout passed to RoboTwin client.
  SAVE_COMPARISON_VIDEO=0 Save comparison videos from client.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

CKPT=${1:-}
if [[ -z "${CKPT}" ]]; then
    usage >&2
    exit 2
fi

DREAMZERO_VENV=${DREAMZERO_VENV:-/opt/venvs/dreamzero}
ROBOTWIN_VENV=${ROBOTWIN_VENV:-/opt/venvs/robotwin310}
START_PORT=${START_PORT:-29556}
MASTER_PORT=${MASTER_PORT:-29661}
TEST_NUM=${TEST_NUM:-100}
SEED=${SEED:-0}
MAX_CHUNK_SIZE=${MAX_CHUNK_SIZE:-24}
SERVER_READY_TIMEOUT_SEC=${SERVER_READY_TIMEOUT_SEC:-1800}
SERVER_LAUNCH_STAGGER_SEC=${SERVER_LAUNCH_STAGGER_SEC:-2}
PROGRESS_INTERVAL_SEC=${PROGRESS_INTERVAL_SEC:-10}
EPISODE_TIMEOUT_SEC=${EPISODE_TIMEOUT_SEC:-900}
SAVE_COMPARISON_VIDEO=${SAVE_COMPARISON_VIDEO:-0}
SAVE_SERVER_VIDEO=${SAVE_SERVER_VIDEO:-0}
AUTO_CLEANUP=${AUTO_CLEANUP:-1}
PRE_CLEANUP=${PRE_CLEANUP:-1}
KEEP_SERVERS=${KEEP_SERVERS:-0}
POLICY_NAME=${POLICY_NAME:-ACT}
TASK_CONFIG=${TASK_CONFIG:-demo_clean}
TRAIN_CONFIG_NAME=${TRAIN_CONFIG_NAME:-0}
MODEL_NAME=${MODEL_NAME:-0}
ACTION_GUIDANCE_SCALE=${ACTION_GUIDANCE_SCALE:-1}
VIDEO_GUIDANCE_SCALE=${VIDEO_GUIDANCE_SCALE:-5}
SSH_USER=${SSH_USER:-root}
REMOTE_REPO_ROOT=${REMOTE_REPO_ROOT:-${REPO_ROOT}}
REMOTE_SCRIPT_PATH=${REMOTE_SCRIPT_PATH:-${REMOTE_REPO_ROOT}/example/robotwin/$(basename "${SCRIPT_PATH}")}

if [[ -d /2023133163/checkpoints/dreamzero || -d /2023133163 ]]; then
    DEFAULT_EVAL_RUN_BASE=/2023133163/checkpoints/dreamzero/robotwin_eval_runs
else
    DEFAULT_EVAL_RUN_BASE="${REPO_ROOT}/outputs/robotwin_eval_runs"
fi
EVAL_RUN_BASE=${EVAL_RUN_BASE:-${DEFAULT_EVAL_RUN_BASE}}
RUN_ID=${RUN_ID:-robotwin_all_tasks_$(date +%Y%m%d_%H%M%S)}
RUN_ROOT=${RUN_ROOT:-${EVAL_RUN_BASE}/${RUN_ID}}
TASK_FILE=${TASK_FILE:-${RUN_ROOT}/tasks.txt}
STATUS_DIR=${STATUS_DIR:-${RUN_ROOT}/task_status}
SUMMARY_DIR=${SUMMARY_DIR:-${RUN_ROOT}/summary}
NODE_LOG_DIR=${NODE_LOG_DIR:-${RUN_ROOT}/node_logs}

mkdir -p "${RUN_ROOT}" "${STATUS_DIR}" "${SUMMARY_DIR}" "${NODE_LOG_DIR}"

SERVER_PIDS=()
WORKER_PIDS=()
COORDINATOR_NODE_PIDS=()
COORDINATOR_NODES=()
MONITOR_PID=""
LOCAL_CLEANED=0
COORDINATOR_CLEANED=0

kill_matching_eval_processes() {
    if [[ "${AUTO_CLEANUP}" != "1" || "${KEEP_SERVERS}" == "1" ]]; then
        return
    fi

    pkill -TERM -f "example.robotwin.eval_polict_client_openpi" >/dev/null 2>&1 || true
    pkill -TERM -f "socket_test_optimized_aloha_x5lite_bimanual.py" >/dev/null 2>&1 || true
    sleep 2
    pkill -KILL -f "example.robotwin.eval_polict_client_openpi" >/dev/null 2>&1 || true
    pkill -KILL -f "socket_test_optimized_aloha_x5lite_bimanual.py" >/dev/null 2>&1 || true
}

pre_cleanup_local_eval_processes() {
    if [[ "${PRE_CLEANUP}" != "1" || "${AUTO_CLEANUP}" != "1" || "${KEEP_SERVERS}" == "1" ]]; then
        return
    fi
    echo "Cleaning stale local RoboTwin eval processes before launch..."
    kill_matching_eval_processes
}

cleanup_local_eval_processes() {
    if [[ "${AUTO_CLEANUP}" != "1" || "${KEEP_SERVERS}" == "1" || "${LOCAL_CLEANED}" == "1" ]]; then
        return
    fi
    LOCAL_CLEANED=1

    echo "Cleaning local RoboTwin eval processes..."
    if (( ${#WORKER_PIDS[@]} > 0 )); then
        kill "${WORKER_PIDS[@]}" >/dev/null 2>&1 || true
    fi
    if (( ${#SERVER_PIDS[@]} > 0 )); then
        kill "${SERVER_PIDS[@]}" >/dev/null 2>&1 || true
    fi
    if (( ${#WORKER_PIDS[@]} > 0 )); then
        wait "${WORKER_PIDS[@]}" >/dev/null 2>&1 || true
    fi
    if (( ${#SERVER_PIDS[@]} > 0 )); then
        wait "${SERVER_PIDS[@]}" >/dev/null 2>&1 || true
    fi
    kill_matching_eval_processes
}

cleanup_remote_eval_processes() {
    if [[ "${AUTO_CLEANUP}" != "1" || "${KEEP_SERVERS}" == "1" ]]; then
        return
    fi
    if (( ${#COORDINATOR_NODES[@]} == 0 )); then
        return
    fi

    echo "Cleaning remote RoboTwin eval processes..."
    for node in "${COORDINATOR_NODES[@]}"; do
        if [[ "${node}" == "local" ]]; then
            kill_matching_eval_processes
            continue
        fi
        ssh -o BatchMode=yes -o ConnectTimeout=10 "${SSH_USER}@${node}" \
            "AUTO_CLEANUP='${AUTO_CLEANUP}' KEEP_SERVERS='${KEEP_SERVERS}' bash -lc '
                if [[ \"\${AUTO_CLEANUP}\" == \"1\" && \"\${KEEP_SERVERS}\" != \"1\" ]]; then
                    pkill -TERM -f example.robotwin.eval_polict_client_openpi >/dev/null 2>&1 || true
                    pkill -TERM -f socket_test_optimized_aloha_x5lite_bimanual.py >/dev/null 2>&1 || true
                    sleep 2
                    pkill -KILL -f example.robotwin.eval_polict_client_openpi >/dev/null 2>&1 || true
                    pkill -KILL -f socket_test_optimized_aloha_x5lite_bimanual.py >/dev/null 2>&1 || true
                fi
            '" >/dev/null 2>&1 || true
    done
}

cleanup_coordinator() {
    if [[ "${AUTO_CLEANUP}" != "1" || "${KEEP_SERVERS}" == "1" || "${COORDINATOR_CLEANED}" == "1" ]]; then
        return
    fi
    COORDINATOR_CLEANED=1

    if [[ -n "${MONITOR_PID}" ]]; then
        kill "${MONITOR_PID}" >/dev/null 2>&1 || true
        wait "${MONITOR_PID}" >/dev/null 2>&1 || true
    fi
    if (( ${#COORDINATOR_NODE_PIDS[@]} > 0 )); then
        kill "${COORDINATOR_NODE_PIDS[@]}" >/dev/null 2>&1 || true
        wait "${COORDINATOR_NODE_PIDS[@]}" >/dev/null 2>&1 || true
    fi
    cleanup_remote_eval_processes
}

detect_gpu_ids() {
    if [[ -n "${CLIENT_GPUS:-}" ]]; then
        tr ',' '\n' <<<"${CLIENT_GPUS}" | awk 'NF'
        return
    fi
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "all" ]]; then
        tr ',' '\n' <<<"${CUDA_VISIBLE_DEVICES}" | awk 'NF'
        return
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        local count
        count=$(nvidia-smi -L 2>/dev/null | wc -l | awk '{print $1}')
        if (( count > 0 )); then
            seq 0 $((count - 1))
            return
        fi
    fi
    echo 0
}

write_task_file() {
    if [[ -n "${TASKS_FILE:-}" ]]; then
        cp "${TASKS_FILE}" "${TASK_FILE}"
        return
    fi

    if [[ -n "${TASKS:-}" && "${TASKS}" != "all" ]]; then
        tr ' ' '\n' <<<"${TASKS}" | awk 'NF' > "${TASK_FILE}"
        return
    fi

    (
        cd "${REPO_ROOT}"
        python - <<'PY'
from pathlib import Path

env_dir = Path("third_party") / "RoboTwin" / "envs"
tasks = [
    path.stem
    for path in sorted(env_dir.glob("*.py"))
    if not path.stem.startswith("_")
]
if not tasks:
    raise SystemExit(f"No RoboTwin tasks found under {env_dir}")
print("\n".join(tasks))
PY
    ) > "${TASK_FILE}"
}

load_tasks() {
    mapfile -t TASK_NAMES < <(awk 'NF && $1 !~ /^#/' "${TASK_FILE}")
    TOTAL_TASKS=${#TASK_NAMES[@]}
    if (( TOTAL_TASKS < 1 )); then
        echo "No tasks found in ${TASK_FILE}" >&2
        exit 1
    fi
}

websocket_is_ready() {
    local host=$1
    local port=$2
    local python_bin="${DREAMZERO_VENV}/bin/python"
    if [[ ! -x "${python_bin}" ]]; then
        python_bin=python
    fi

    timeout 5 "${python_bin}" - "${host}" "${port}" <<'PY' >/dev/null 2>&1
import sys

from websockets.sync.client import connect

host = sys.argv[1]
port = int(sys.argv[2])

with connect(
    f"ws://{host}:{port}",
    compression=None,
    max_size=None,
    open_timeout=2,
    ping_interval=None,
):
    pass
PY
}

progress_bar() {
    local done_count=$1
    local total_count=$2
    local width=${3:-36}
    local filled=0
    if (( total_count > 0 )); then
        filled=$((done_count * width / total_count))
    fi
    local empty=$((width - filled))
    printf '['
    printf '%*s' "${filled}" '' | tr ' ' '#'
    printf '%*s' "${empty}" '' | tr ' ' '.'
    printf '] %d/%d done, remaining %d' "${done_count}" "${total_count}" "$((total_count - done_count))"
}

write_status_from_result() {
    local task_name=$1
    local rc=$2
    local result_json=$3
    local log_file=$4
    local tmp_file="${STATUS_DIR}/${task_name}.status.tmp.$$"
    local status_file="${STATUS_DIR}/${task_name}.status"

    python - "${task_name}" "${rc}" "${result_json}" "${log_file}" "${TEST_NUM}" > "${tmp_file}" <<'PY'
import json
import sys
from pathlib import Path

task, rc_s, result_s, log_s, expected_s = sys.argv[1:]
rc = int(rc_s)
expected = float(expected_s)
result_path = Path(result_s)
succ = 0.0
total = 0.0
rate = 0.0
error = ""

if result_path.exists():
    try:
        data = json.loads(result_path.read_text())
        succ = float(data.get("succ_num", 0.0))
        total = float(data.get("total_num", 0.0))
        rate = float(data.get("succ_rate", 0.0))
    except Exception as exc:
        error = f"failed_to_parse_result:{exc}"
else:
    error = "missing_result_json"

complete = int(rc == 0 and total >= expected)
print(
    "\t".join(
        [
            task,
            str(rc),
            f"{succ:g}",
            f"{total:g}",
            f"{rate:.10f}",
            str(complete),
            str(result_path),
            str(log_s),
            error,
        ]
    )
)
PY
    mv "${tmp_file}" "${status_file}"
}

start_local_servers() {
    local -n _gpu_ids_ref=$1
    local node_rank=$2
    local server_log_dir="${RUN_ROOT}/node${node_rank}/server_logs"
    local server_output_dir="${RUN_ROOT}/node${node_rank}/server_outputs"
    mkdir -p "${server_log_dir}" "${server_output_dir}"

    local python_bin="${DREAMZERO_VENV}/bin/python"
    if [[ ! -x "${python_bin}" ]]; then
        python_bin=python
    fi

    SERVER_PIDS=()
    for local_slot in "${!_gpu_ids_ref[@]}"; do
        local gpu_id="${_gpu_ids_ref[${local_slot}]}"
        local port=$((START_PORT + local_slot))
        local current_master_port=$((MASTER_PORT + node_rank * 100 + local_slot))
        local log_file="${server_log_dir}/server_slot${local_slot}_gpu${gpu_id}.log"
        local extra_args=()
        if [[ -n "${MAX_CHUNK_SIZE}" ]]; then
            extra_args+=(--max_chunk_size "${MAX_CHUNK_SIZE}")
        fi
        if [[ "${SAVE_SERVER_VIDEO}" == "1" ]]; then
            extra_args+=(--save_server_video)
        fi

        echo "[node ${node_rank}] start server slot=${local_slot} gpu=${gpu_id} port=${port} log=${log_file}"
        (
            cd "${REPO_ROOT}"
            export VIRTUAL_ENV="${DREAMZERO_VENV}"
            export PATH="${DREAMZERO_VENV}/bin:${PATH}"
            export CUDA_VISIBLE_DEVICES="${gpu_id}"
            "${python_bin}" -m torch.distributed.run \
                --nproc_per_node 1 \
                --master_port "${current_master_port}" \
                socket_test_optimized_aloha_x5lite_bimanual.py \
                --model_path "${CKPT}" \
                --output_root "${server_output_dir}" \
                --port "${port}" \
                "${extra_args[@]}"
        ) > "${log_file}" 2>&1 &
        SERVER_PIDS+=("$!")
        sleep "${SERVER_LAUNCH_STAGGER_SEC}"
    done
}

wait_for_local_servers() {
    local -n _gpu_ids_ref=$1
    local node_rank=$2
    local deadline=$((SECONDS + SERVER_READY_TIMEOUT_SEC))

    while true; do
        local ready=0
        for local_slot in "${!_gpu_ids_ref[@]}"; do
            local port=$((START_PORT + local_slot))
            if websocket_is_ready 127.0.0.1 "${port}"; then
                ready=$((ready + 1))
            fi
        done

        printf '\r[node %s] waiting for servers: %d/%d ready' "${node_rank}" "${ready}" "${#_gpu_ids_ref[@]}"
        if (( ready == ${#_gpu_ids_ref[@]} )); then
            printf '\n'
            return 0
        fi

        if (( SECONDS >= deadline )); then
            printf '\n'
            echo "[node ${node_rank}] timed out waiting for policy servers" >&2
            return 1
        fi
        sleep 5
    done
}

stop_local_servers() {
    cleanup_local_eval_processes
}

run_client_slot() {
    local node_rank=$1
    local local_slot=$2
    local gpu_id=$3
    local global_slot=$4
    local total_slots=$5
    local client_output_dir="${RUN_ROOT}/node${node_rank}/client_outputs"
    local client_log_dir="${RUN_ROOT}/node${node_rank}/client_logs"
    mkdir -p "${client_output_dir}" "${client_log_dir}"

    local python_bin="${ROBOTWIN_VENV}/bin/python"
    if [[ ! -x "${python_bin}" ]]; then
        python_bin=python
    fi

    local slot_log="${client_log_dir}/slot${local_slot}_gpu${gpu_id}.log"
    {
        echo "[node ${node_rank} slot ${local_slot}] global_slot=${global_slot}/${total_slots} gpu=${gpu_id}"
        cd "${REPO_ROOT}"
        export VIRTUAL_ENV="${ROBOTWIN_VENV}"
        export PATH="${ROBOTWIN_VENV}/bin:${PATH}"
        export LD_LIBRARY_PATH="/usr/lib64:/usr/lib:${LD_LIBRARY_PATH:-}"
        export CUDA_VISIBLE_DEVICES="${gpu_id}"
        export EPISODE_TIMEOUT_SEC
        export SAVE_COMPARISON_VIDEO

        for task_idx in "${!TASK_NAMES[@]}"; do
            local assigned_slot
            if (( TOTAL_TASKS >= total_slots )); then
                assigned_slot=$((task_idx % total_slots))
            else
                assigned_slot=$((task_idx * total_slots / TOTAL_TASKS))
            fi
            if (( assigned_slot != global_slot )); then
                continue
            fi

            local task_name="${TASK_NAMES[${task_idx}]}"
            local task_log="${client_log_dir}/${task_name}.log"
            local result_json="${client_output_dir}/stseed-$((10000 * (1 + SEED)))/metrics/${task_name}/res.json"
            rm -f "${STATUS_DIR}/${task_name}.status"
            echo "[node ${node_rank} slot ${local_slot}] start task=${task_name} port=$((START_PORT + local_slot)) gpu=${gpu_id}"

            local rc=0
            set +e
            PYTHONWARNINGS=ignore::UserWarning \
                XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
                "${python_bin}" -m example.robotwin.eval_polict_client_openpi \
                    --config "policy/${POLICY_NAME}/deploy_policy.yml" \
                    --host 127.0.0.1 \
                    --port "$((START_PORT + local_slot))" \
                    --save_root "${client_output_dir}" \
                    --video_guidance_scale "${VIDEO_GUIDANCE_SCALE}" \
                    --action_guidance_scale "${ACTION_GUIDANCE_SCALE}" \
                    --test_num "${TEST_NUM}" \
                    --overrides \
                    --task_name "${task_name}" \
                    --task_config "${TASK_CONFIG}" \
                    --train_config_name "${TRAIN_CONFIG_NAME}" \
                    --model_name "${MODEL_NAME}" \
                    --ckpt_setting "${MODEL_NAME}" \
                    --seed "${SEED}" \
                    --policy_name "${POLICY_NAME}" \
                    > "${task_log}" 2>&1
            rc=$?
            set -e

            write_status_from_result "${task_name}" "${rc}" "${result_json}" "${task_log}"
            echo "[node ${node_rank} slot ${local_slot}] finished task=${task_name} rc=${rc}"
            if [[ "${rc}" != "0" || ! -f "${result_json}" ]]; then
                echo "[node ${node_rank} slot ${local_slot}] aborting slot after task=${task_name}: rc=${rc}, result_json=${result_json}"
                exit 1
            fi
        done
    } >> "${slot_log}" 2>&1
}

node_main() {
    cd "${REPO_ROOT}"
    load_tasks

    local node_rank=${NODE_RANK:?NODE_RANK is required for node role}
    local total_nodes=${TOTAL_NODES:?TOTAL_NODES is required for node role}
    local slots_per_node=${SLOTS_PER_NODE:?SLOTS_PER_NODE is required for node role}
    local total_slots=$((total_nodes * slots_per_node))

    mapfile -t gpu_ids < <(detect_gpu_ids)
    if (( ${#gpu_ids[@]} > slots_per_node )); then
        gpu_ids=("${gpu_ids[@]:0:${slots_per_node}}")
    fi
    if (( ${#gpu_ids[@]} < slots_per_node )); then
        echo "[node ${node_rank}] only detected ${#gpu_ids[@]} GPUs, expected ${slots_per_node}" >&2
        return 1
    fi

    SERVER_PIDS=()
    WORKER_PIDS=()
    trap cleanup_local_eval_processes EXIT
    trap 'cleanup_local_eval_processes; exit 130' INT TERM

    pre_cleanup_local_eval_processes
    start_local_servers gpu_ids "${node_rank}"
    wait_for_local_servers gpu_ids "${node_rank}"

    for local_slot in "${!gpu_ids[@]}"; do
        local global_slot=$((node_rank * slots_per_node + local_slot))
        run_client_slot "${node_rank}" "${local_slot}" "${gpu_ids[${local_slot}]}" "${global_slot}" "${total_slots}" &
        WORKER_PIDS+=("$!")
    done

    local status=0
    for pid in "${WORKER_PIDS[@]}"; do
        if ! wait "${pid}"; then
            status=1
        fi
    done

    touch "${RUN_ROOT}/node${node_rank}.done"
    return "${status}"
}

parse_nodes() {
    if [[ -n "${ROBOTWIN_NODES:-}" ]]; then
        tr ', ' '\n' <<<"${ROBOTWIN_NODES}" | awk 'NF'
        return
    fi
    if [[ -n "${NODES_FILE:-}" ]]; then
        awk 'NF && $1 !~ /^#/' "${NODES_FILE}"
        return
    fi
}

collect_status_line() {
    local status_file=$1
    IFS=$'\t' read -r task rc succ total rate complete result_json log_file error < "${status_file}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${task}" "${rc}" "${succ}" "${total}" "${rate}" "${complete}" "${result_json}" "${log_file}" "${error}"
}

monitor_progress() {
    local -A reported=()
    local done_count=0

    while (( done_count < TOTAL_TASKS )); do
        local changed=0
        for task_name in "${TASK_NAMES[@]}"; do
            if [[ -n "${reported[${task_name}]:-}" ]]; then
                continue
            fi
            local status_file="${STATUS_DIR}/${task_name}.status"
            if [[ ! -f "${status_file}" ]]; then
                continue
            fi

            local line
            line=$(collect_status_line "${status_file}")
            IFS=$'\t' read -r task rc succ total rate complete result_json log_file error <<< "${line}"
            reported["${task_name}"]=1
            done_count=$((done_count + 1))
            changed=1
            printf '\r%*s\r' 120 ''
            python - "${task}" "${rc}" "${succ}" "${total}" "${rate}" "${complete}" "${error}" <<'PY'
import sys

task, rc, succ, total, rate, complete, error = sys.argv[1:]
rate_pct = float(rate) * 100.0
suffix = "complete" if complete == "1" else "incomplete"
if error:
    suffix += f", {error}"
label = "task done" if complete == "1" else "task failed"
print(f"[{label}] {task}: acc={rate_pct:.2f}% ({succ}/{total}) rc={rc} {suffix}")
PY
        done

        progress_bar "${done_count}" "${TOTAL_TASKS}"
        if (( changed == 0 )); then
            sleep "${PROGRESS_INTERVAL_SEC}"
        fi
    done
    printf '\n'
}

write_final_summary() {
    python - "${TASK_FILE}" "${STATUS_DIR}" "${SUMMARY_DIR}" "${TEST_NUM}" <<'PY'
import sys
from pathlib import Path

task_file = Path(sys.argv[1])
status_dir = Path(sys.argv[2])
summary_dir = Path(sys.argv[3])
expected = float(sys.argv[4])

tasks = [line.strip() for line in task_file.read_text().splitlines() if line.strip() and not line.startswith("#")]
rows = []
missing = []

for task in tasks:
    status_path = status_dir / f"{task}.status"
    if not status_path.exists():
        missing.append(task)
        rows.append(
            {
                "task": task,
                "rc": None,
                "succ_num": 0.0,
                "total_num": 0.0,
                "succ_rate": 0.0,
                "complete": False,
                "result_json": "",
                "log_file": "",
                "error": "missing_status",
            }
        )
        continue

    parts = status_path.read_text().rstrip("\n").split("\t")
    parts += [""] * (9 - len(parts))
    task_name, rc, succ, total, rate, complete, result_json, log_file, error = parts[:9]
    rows.append(
        {
            "task": task_name,
            "rc": int(rc),
            "succ_num": float(succ),
            "total_num": float(total),
            "succ_rate": float(rate),
            "complete": complete == "1",
            "result_json": result_json,
            "log_file": log_file,
            "error": error,
        }
    )

completed = [row for row in rows if row["complete"]]
attempted = [row for row in rows if row["total_num"] > 0]
total_succ = sum(row["succ_num"] for row in rows)
total_num = sum(row["total_num"] for row in rows)
micro_acc = total_succ / total_num if total_num else 0.0
macro_acc = sum(row["succ_rate"] for row in attempted) / len(attempted) if attempted else 0.0

summary_dir.mkdir(parents=True, exist_ok=True)
summary_path = summary_dir / "summary_result.md"
with summary_path.open("w") as f:
    f.write("# RoboTwin Eval Summary\n\n")
    f.write(f"- tasks: {len(tasks)}\n")
    f.write(f"- completed_tasks: {len(completed)}\n")
    f.write(f"- attempted_tasks: {len(attempted)}\n")
    f.write(f"- expected_episodes_per_task: {expected:g}\n")
    f.write(f"- micro_acc: {micro_acc * 100:.2f}% ({total_succ:g}/{total_num:g})\n")
    f.write(f"- macro_acc: {macro_acc * 100:.2f}%\n")
    if missing:
        f.write(f"- missing_tasks: {', '.join(missing)}\n")
    f.write("\n")
    f.write("| task | acc | succ/total | complete | rc | error | result_json | log_file |\n")
    f.write("| --- | ---: | ---: | --- | ---: | --- | --- | --- |\n")
    for row in rows:
        rc = "" if row["rc"] is None else str(row["rc"])
        f.write(
            f"| {row['task']} | {row['succ_rate'] * 100:.2f}% | "
            f"{row['succ_num']:g}/{row['total_num']:g} | {row['complete']} | "
            f"{rc} | {row['error']} | {row['result_json']} | {row['log_file']} |\n"
        )

print(f"Final summary: completed {len(completed)}/{len(tasks)} tasks")
print(f"Micro acc: {micro_acc * 100:.2f}% ({total_succ:g}/{total_num:g})")
print(f"Macro acc: {macro_acc * 100:.2f}%")
print(f"Summary: {summary_path}")
PY
}

coordinator_main() {
    cd "${REPO_ROOT}"
    write_task_file
    load_tasks

    mapfile -t nodes < <(parse_nodes)
    local node_count=${#nodes[@]}
    local local_mode=0
    if (( node_count == 0 )); then
        local_mode=1
        nodes=("local")
        node_count=1
    fi

    if (( local_mode == 1 )); then
        if [[ -z "${NUM_GPUS:-}" ]]; then
            mapfile -t local_gpu_ids < <(detect_gpu_ids)
            NUM_GPUS=${#local_gpu_ids[@]}
        fi
        SLOTS_PER_NODE=${SLOTS_PER_NODE:-${NUM_GPUS}}
    else
        SLOTS_PER_NODE=${SLOTS_PER_NODE:-${NUM_GPUS:-8}}
    fi

    cat > "${RUN_ROOT}/run_config.env" <<EOF
CKPT=${CKPT}
RUN_ROOT=${RUN_ROOT}
TASK_FILE=${TASK_FILE}
TEST_NUM=${TEST_NUM}
TOTAL_TASKS=${TOTAL_TASKS}
TOTAL_NODES=${node_count}
SLOTS_PER_NODE=${SLOTS_PER_NODE}
START_PORT=${START_PORT}
MASTER_PORT=${MASTER_PORT}
AUTO_CLEANUP=${AUTO_CLEANUP}
PRE_CLEANUP=${PRE_CLEANUP}
EOF

    echo "Run root: ${RUN_ROOT}"
    echo "Tasks: ${TOTAL_TASKS}, episodes per task: ${TEST_NUM}"
    echo "Nodes: ${node_count}, slots per node: ${SLOTS_PER_NODE}"
    echo "Task file: ${TASK_FILE}"

    COORDINATOR_NODES=("${nodes[@]}")
    COORDINATOR_NODE_PIDS=()
    MONITOR_PID=""
    trap cleanup_coordinator EXIT
    trap 'cleanup_coordinator; exit 130' INT TERM

    for node_rank in "${!nodes[@]}"; do
        local node="${nodes[${node_rank}]}"
        local node_log="${NODE_LOG_DIR}/node${node_rank}.log"
        rm -f "${RUN_ROOT}/node${node_rank}.done"

        if [[ "${node}" == "local" ]]; then
            echo "[coordinator] launch local node rank=${node_rank} log=${node_log}"
            (
                export ROBOTWIN_EVAL_ROLE=node
                export NODE_RANK="${node_rank}"
                export TOTAL_NODES="${node_count}"
                export SLOTS_PER_NODE
                export RUN_ROOT TASK_FILE STATUS_DIR SUMMARY_DIR
                export DREAMZERO_VENV ROBOTWIN_VENV START_PORT MASTER_PORT TEST_NUM SEED
                export MAX_CHUNK_SIZE SERVER_READY_TIMEOUT_SEC SERVER_LAUNCH_STAGGER_SEC
                export EPISODE_TIMEOUT_SEC SAVE_COMPARISON_VIDEO SAVE_SERVER_VIDEO
                export AUTO_CLEANUP PRE_CLEANUP KEEP_SERVERS
                export POLICY_NAME TASK_CONFIG TRAIN_CONFIG_NAME MODEL_NAME
                export ACTION_GUIDANCE_SCALE VIDEO_GUIDANCE_SCALE
                bash "${SCRIPT_PATH}" "${CKPT}"
            ) > "${node_log}" 2>&1 &
        else
            echo "[coordinator] launch remote node=${node} rank=${node_rank} log=${node_log}"
            ssh "${SSH_USER}@${node}" "cd '${REMOTE_REPO_ROOT}' && \
                ROBOTWIN_EVAL_ROLE=node \
                NODE_RANK='${node_rank}' \
                TOTAL_NODES='${node_count}' \
                SLOTS_PER_NODE='${SLOTS_PER_NODE}' \
                RUN_ROOT='${RUN_ROOT}' \
                TASK_FILE='${TASK_FILE}' \
                STATUS_DIR='${STATUS_DIR}' \
                SUMMARY_DIR='${SUMMARY_DIR}' \
                DREAMZERO_VENV='${DREAMZERO_VENV}' \
                ROBOTWIN_VENV='${ROBOTWIN_VENV}' \
                START_PORT='${START_PORT}' \
                MASTER_PORT='${MASTER_PORT}' \
                TEST_NUM='${TEST_NUM}' \
                SEED='${SEED}' \
                MAX_CHUNK_SIZE='${MAX_CHUNK_SIZE}' \
                SERVER_READY_TIMEOUT_SEC='${SERVER_READY_TIMEOUT_SEC}' \
                SERVER_LAUNCH_STAGGER_SEC='${SERVER_LAUNCH_STAGGER_SEC}' \
                EPISODE_TIMEOUT_SEC='${EPISODE_TIMEOUT_SEC}' \
                SAVE_COMPARISON_VIDEO='${SAVE_COMPARISON_VIDEO}' \
                SAVE_SERVER_VIDEO='${SAVE_SERVER_VIDEO}' \
                AUTO_CLEANUP='${AUTO_CLEANUP}' \
                PRE_CLEANUP='${PRE_CLEANUP}' \
                KEEP_SERVERS='${KEEP_SERVERS}' \
                POLICY_NAME='${POLICY_NAME}' \
                TASK_CONFIG='${TASK_CONFIG}' \
                TRAIN_CONFIG_NAME='${TRAIN_CONFIG_NAME}' \
                MODEL_NAME='${MODEL_NAME}' \
                ACTION_GUIDANCE_SCALE='${ACTION_GUIDANCE_SCALE}' \
                VIDEO_GUIDANCE_SCALE='${VIDEO_GUIDANCE_SCALE}' \
                bash '${REMOTE_SCRIPT_PATH}' '${CKPT}'" > "${node_log}" 2>&1 &
        fi
        COORDINATOR_NODE_PIDS+=("$!")
    done

    monitor_progress &
    MONITOR_PID=$!

    local status=0
    for pid in "${COORDINATOR_NODE_PIDS[@]}"; do
        if ! wait "${pid}"; then
            status=1
        fi
    done

    local all_statuses_written=0
    for _ in 1 2 3; do
        if ! kill -0 "${MONITOR_PID}" >/dev/null 2>&1; then
            break
        fi
        local status_count
        status_count=$(find "${STATUS_DIR}" -maxdepth 1 -name '*.status' 2>/dev/null | wc -l | awk '{print $1}')
        if (( status_count >= TOTAL_TASKS )); then
            all_statuses_written=1
            break
        fi
        sleep 2
    done
    if (( all_statuses_written == 1 )); then
        wait "${MONITOR_PID}" || true
    elif kill -0 "${MONITOR_PID}" >/dev/null 2>&1; then
        printf '\n[coordinator] stopping progress monitor; some tasks may be missing because a node exited early.\n'
        kill "${MONITOR_PID}" >/dev/null 2>&1 || true
        wait "${MONITOR_PID}" || true
    else
        wait "${MONITOR_PID}" || true
    fi
    write_final_summary
    return "${status}"
}

if [[ "${ROBOTWIN_EVAL_ROLE:-coordinator}" == "node" ]]; then
    node_main
else
    coordinator_main
fi
