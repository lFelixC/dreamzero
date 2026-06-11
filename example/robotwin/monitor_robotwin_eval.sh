#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash example/robotwin/monitor_robotwin_eval.sh [RUN_ROOT]
  bash example/robotwin/monitor_robotwin_eval.sh --watch [RUN_ROOT]

Env:
  RUN_ROOT=...            Eval run directory. Used if RUN_ROOT argument is omitted.
  EVAL_RUN_BASE=...       Parent run directory for auto-detecting the latest run.
  INTERVAL=10             Refresh interval for --watch.
EOF
}

WATCH=0
INTERVAL=${INTERVAL:-10}
RUN_ROOT_ARG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        -w|--watch)
            WATCH=1
            shift
            ;;
        --interval)
            INTERVAL=${2:?--interval requires a value}
            shift 2
            ;;
        *)
            RUN_ROOT_ARG=$1
            shift
            ;;
    esac
done

if [[ -n "${RUN_ROOT_ARG}" ]]; then
    RUN_ROOT=${RUN_ROOT_ARG}
elif [[ -n "${RUN_ROOT:-}" ]]; then
    RUN_ROOT=${RUN_ROOT}
else
    if [[ -d /2023133163/checkpoints/dreamzero/robotwin_eval_runs ]]; then
        EVAL_RUN_BASE=${EVAL_RUN_BASE:-/2023133163/checkpoints/dreamzero/robotwin_eval_runs}
    else
        SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
        REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
        EVAL_RUN_BASE=${EVAL_RUN_BASE:-${REPO_ROOT}/outputs/robotwin_eval_runs}
    fi
    RUN_ROOT=$(find "${EVAL_RUN_BASE}" -maxdepth 1 -type d -name 'robotwin_all_tasks_*' 2>/dev/null | sort | tail -1 || true)
fi

if [[ -z "${RUN_ROOT}" ]]; then
    echo "No RUN_ROOT found. Pass RUN_ROOT explicitly." >&2
    exit 2
fi

render_once() {
    python - "${RUN_ROOT}" <<'PY'
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

run_root = Path(sys.argv[1])
if not run_root.exists():
    print(f"RUN_ROOT does not exist: {run_root}", file=sys.stderr)
    sys.exit(2)

task_file = run_root / "tasks.txt"
status_dir = run_root / "task_status"
summary_path = run_root / "summary" / "summary_result.md"

def read_text(path: Path, limit_bytes: int | None = None) -> str:
    try:
        if limit_bytes is None:
            return path.read_text(errors="replace")
        size = path.stat().st_size
        with path.open("rb") as f:
            if size > limit_bytes:
                f.seek(-limit_bytes, os.SEEK_END)
            data = f.read()
        return data.decode(errors="replace")
    except Exception:
        return ""

def tail_lines(path: Path, limit: int = 80) -> list[str]:
    text = read_text(path, 64 * 1024)
    return text.splitlines()[-limit:]

def rel(path: Path) -> str:
    try:
        return str(path.relative_to(run_root))
    except ValueError:
        return str(path)

def age(path: Path) -> str:
    try:
        seconds = max(0, int(time.time() - path.stat().st_mtime))
    except OSError:
        return "?"
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"

def last_interesting_line(path: Path) -> str:
    ignore = (
        "DeprecationWarning",
        "UserWarning",
        "pkg_resources is deprecated",
        "====",
    )
    for line in reversed(tail_lines(path, 120)):
        stripped = line.strip().replace("\r", "")
        if not stripped:
            continue
        if any(token in stripped for token in ignore):
            continue
        return stripped[-180:]
    return ""

tasks = []
if task_file.exists():
    tasks = [
        line.strip()
        for line in read_text(task_file).splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

status_rows = {}
for status_path in sorted(status_dir.glob("*.status")) if status_dir.exists() else []:
    parts = read_text(status_path).rstrip("\n").split("\t")
    parts += [""] * (9 - len(parts))
    task, rc, succ, total, rate, complete, result_json, log_file, error = parts[:9]
    status_rows[task] = {
        "path": status_path,
        "rc": rc,
        "succ": succ,
        "total": total,
        "rate": rate,
        "complete": complete == "1",
        "error": error,
        "log_file": log_file,
    }

reported = len(status_rows)
completed = sum(1 for row in status_rows.values() if row["complete"])
failed = sum(1 for row in status_rows.values() if not row["complete"])
missing = max(0, len(tasks) - reported) if tasks else 0

print(f"RUN_ROOT: {run_root}")
print(
    f"TASKS: reported {reported}/{len(tasks) if tasks else '?'} | "
    f"complete {completed} | failed {failed} | missing {missing}"
)

if summary_path.exists():
    for line in read_text(summary_path).splitlines()[:7]:
        if line.startswith("- micro_acc") or line.startswith("- macro_acc") or line.startswith("- completed_tasks"):
            print(f"SUMMARY: {line[2:]}")

node_ids = set()
for p in run_root.glob("node_logs/node*.log"):
    m = re.search(r"node(\d+)\.log$", p.name)
    if m:
        node_ids.add(int(m.group(1)))
for p in run_root.glob("node[0-9]*"):
    m = re.search(r"node(\d+)$", p.name)
    if m and p.is_dir():
        node_ids.add(int(m.group(1)))

print("\nNODES:")
if not node_ids:
    print("  no node logs yet")
for node_id in sorted(node_ids):
    node_dir = run_root / f"node{node_id}"
    node_log = run_root / "node_logs" / f"node{node_id}.log"
    node_text = read_text(node_log, 128 * 1024)
    ready_matches = re.findall(r"waiting for servers:\s*(\d+)/(\d+)\s*ready", node_text)
    if ready_matches:
        ready, expected = ready_matches[-1]
    else:
        server_logs = list((node_dir / "server_logs").glob("*.log"))
        expected = str(len(server_logs)) if server_logs else "?"
        ready_count = sum(
            1
            for log in server_logs
            if re.search(r"server listening|Serving .* websocket server", read_text(log, 64 * 1024))
        )
        ready = str(ready_count) if server_logs else "?"

    slot_logs = sorted((node_dir / "client_logs").glob("slot*.log"))
    active = []
    slot_failed = []
    slot_finished = 0
    for slot_log in slot_logs:
        text = read_text(slot_log, 64 * 1024)
        starts = re.findall(r"start task=([^\s]+).*gpu=([^\s]+)", text)
        finishes = re.findall(r"finished task=([^\s]+) rc=([0-9]+)", text)
        if finishes:
            slot_finished += len(finishes)
            task, rc = finishes[-1]
            if rc != "0":
                slot_failed.append(f"{slot_log.stem}:{task}:rc{rc}")
        if starts:
            task, gpu = starts[-1]
            if not finishes or finishes[-1][0] != task:
                active.append(f"{slot_log.stem}:{task}@gpu{gpu}")

    server_errors = 0
    for log in (node_dir / "server_logs").glob("*.log"):
        text = "\n".join(tail_lines(log, 180))
        if re.search(r"Traceback|ERROR|RuntimeError|EADDRINUSE|failed|CUDA|Vulkan", text, re.IGNORECASE):
            server_errors += 1

    done = "done" if (run_root / f"node{node_id}.done").exists() else "running"
    last_node_line = last_interesting_line(node_log) if node_log.exists() else ""
    active_text = ", ".join(active[:4]) if active else "-"
    if len(active) > 4:
        active_text += f", +{len(active) - 4}"
    print(
        f"  node{node_id}: servers {ready}/{expected} | active {len(active)} | "
        f"slot_finished {slot_finished} | slot_failed {len(slot_failed)} | "
        f"server_err_logs {server_errors} | {done}"
    )
    if active_text != "-":
        print(f"    active: {active_text}")
    if slot_failed:
        print(f"    slot failures: {', '.join(slot_failed[:4])}")
    if last_node_line:
        print(f"    node log: {last_node_line}")

print("\nLATEST STATUS:")
latest_status = sorted(
    status_rows.values(),
    key=lambda row: row["path"].stat().st_mtime if row["path"].exists() else 0,
    reverse=True,
)[:10]
if not latest_status:
    print("  no task status files yet")
else:
    for row in latest_status:
        task = row["path"].stem
        try:
            pct = float(row["rate"]) * 100.0
            rate = f"{pct:.2f}%"
        except Exception:
            rate = row["rate"]
        state = "OK" if row["complete"] else "FAIL"
        err = f" err={row['error']}" if row["error"] else ""
        print(f"  {state:4s} {task:32s} acc={rate:>8s} ({row['succ']}/{row['total']}) rc={row['rc']}{err}")

print("\nSHARD PROGRESS:")
shard_root = status_dir / "shards"
shard_dirs = []
if shard_root.exists():
    shard_dirs = sorted(
        [p for p in shard_root.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )[:8]
if not shard_dirs:
    print("  no sharded task progress")
else:
    for shard_dir in shard_dirs:
        shard_statuses = sorted(shard_dir.glob("slot*.status"))
        shard_succ = 0.0
        shard_total = 0.0
        shard_failed = 0
        for shard_status in shard_statuses:
            parts = read_text(shard_status).rstrip("\n").split("\t")
            parts += [""] * (10 - len(parts))
            _, _, rc, succ, total, _, complete, *_ = parts[:10]
            try:
                shard_succ += float(succ)
                shard_total += float(total)
            except ValueError:
                pass
            if rc and (rc != "0" or complete != "1"):
                shard_failed += 1
        aggregate_done = (status_dir / f"{shard_dir.name}.status").exists()
        state = "done" if aggregate_done else "running"
        print(
            f"  {shard_dir.name:32s} shards={len(shard_statuses)} "
            f"episodes={shard_total:g} succ={shard_succ:g} failed_shards={shard_failed} {state}"
        )

print("\nLATEST CLIENT LOGS:")
client_logs = [
    p
    for p in run_root.glob("node*/client_logs/*.log")
    if not p.name.startswith("slot")
]
client_logs = sorted(client_logs, key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)[:10]
if not client_logs:
    print("  no task client logs yet")
else:
    for path in client_logs:
        print(f"  {age(path):>5s} {rel(path)} :: {last_interesting_line(path)}")

print("\nRECENT ERRORS:")
error_re = re.compile(
    r"Traceback|ERROR|RuntimeError|AttributeError|ImportError|ModuleNotFound|"
    r"FileNotFound|CUDA error|Vulkan|Render Error|EADDRINUSE|address already in use|"
    r"Failed to collect|timed out|No such file|killed",
    re.IGNORECASE,
)
error_logs = []
for pattern in ("node_logs/node*.log", "node*/server_logs/*.log", "node*/client_logs/*.log"):
    error_logs.extend(run_root.glob(pattern))
error_logs = sorted(error_logs, key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)

seen = set()
printed = 0
for path in error_logs:
    for line in reversed(tail_lines(path, 240)):
        clean = line.strip().replace("\r", "")
        if not clean or not error_re.search(clean):
            continue
        key = (rel(path), clean)
        if key in seen:
            continue
        seen.add(key)
        print(f"  {rel(path)} :: {clean[-220:]}")
        printed += 1
        if printed >= 20:
            break
    if printed >= 20:
        break
if printed == 0:
    print("  no recent error signatures found")
PY
}

while true; do
    if (( WATCH == 1 )); then
        printf '\033[H\033[2J'
        date '+%F %T'
    fi
    render_once
    if (( WATCH == 0 )); then
        break
    fi
    sleep "${INTERVAL}"
done
