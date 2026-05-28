#!/bin/bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <sweep_log> <batch_size>" >&2
  exit 2
fi

SWEEP_LOG="$1"
BATCH_SIZE="$2"
SWEEP_SESSION="${SWEEP_SESSION:-cbs_mot_droid_sweep_base1024}"
POLL_SECONDS="${POLL_SECONDS:-20}"
WATCH_LOG="${WATCH_LOG:-${SWEEP_LOG%/*}/pause_after_bs${BATCH_SIZE}_$(date -u +%Y%m%d_%H%M%S).log}"
MARK="----- Finished CBS branch: global batch ${BATCH_SIZE} -----"

log() {
  echo "[pause-watcher] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "${WATCH_LOG}"
}

terminate_sweep() {
  local root_pid="${1}"
  local pgid

  pgid="$(ps -o pgid= -p "${root_pid}" | tr -d ' ' || true)"
  if [[ -z "${pgid}" ]]; then
    log "no process group found for root pid ${root_pid}"
    return 0
  fi

  log "terminating sweep process group -${pgid}"
  kill -TERM -- "-${pgid}" 2>/dev/null || true
  sleep 8
  if ps -g "${pgid}" -o pid= 2>/dev/null | grep -q .; then
    log "force killing residual sweep process group -${pgid}"
    kill -KILL -- "-${pgid}" 2>/dev/null || true
  fi
}

log "started; watching ${SWEEP_LOG}"
log "will stop session ${SWEEP_SESSION} after marker: ${MARK}"

while true; do
  if grep -Fq -- "${MARK}" "${SWEEP_LOG}"; then
    log "detected completion marker for batch ${BATCH_SIZE}"
    root_pid="$(tmux list-panes -t "${SWEEP_SESSION}" -F '#{pane_pid}' 2>/dev/null | head -n 1 || true)"
    if [[ -n "${root_pid}" ]]; then
      terminate_sweep "${root_pid}"
    else
      log "sweep session pane is already gone"
    fi
    log "done"
    exit 0
  fi

  if ! tmux has-session -t "${SWEEP_SESSION}" 2>/dev/null; then
    log "sweep session ended before completion marker"
    exit 1
  fi

  sleep "${POLL_SECONDS}"
done
