#!/usr/bin/env bash
set -euo pipefail

# Capacity probe for one node, 8 GPUs, Wan2.2 5B, 320x640, ZeRO-2 without offload.
# Default runs exactly one batch size. Pass CANDIDATE_BS="32 16 8" only when
# you intentionally want an automatic sweep.

DREAMZERO_ROOT="${DREAMZERO_ROOT:-/2024234320/dreamzero}"
cd "${DREAMZERO_ROOT}"

# shellcheck disable=SC1091
source scripts/env/ngc_droid_wan22.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export DEEPSPEED_CFG="${DEEPSPEED_CFG:-zero2}"
export MAX_STEPS="${MAX_STEPS:-3}"
export SAVE_STRATEGY="${SAVE_STRATEGY:-no}"
export SAVE_STEPS="${SAVE_STEPS:-1000000}"
export REPORT_TO="${REPORT_TO:-none}"
export SWANLAB_SYNC_WANDB="${SWANLAB_SYNC_WANDB:-0}"
export DREAMZERO_SKIP_FINAL_SAVE="${DREAMZERO_SKIP_FINAL_SAVE:-1}"
export LOGGING_STEPS="${LOGGING_STEPS:-1}"
export MAX_CHUNK_SIZE="${MAX_CHUNK_SIZE:-2}"

CANDIDATE_BS="${CANDIDATE_BS:-${PER_DEVICE_BS:-32}}"
STOP_AFTER_FIRST_SUCCESS="${STOP_AFTER_FIRST_SUCCESS:-1}"
GPU_POLL_INTERVAL="${GPU_POLL_INTERVAL:-2}"
LOG_ROOT="${DREAMZERO_LOG_ROOT:-${DREAMZERO_ROOT}/experiment_logs}"
PROBE_ROOT="${LOG_ROOT}/batchsize_probe"
mkdir -p "${PROBE_ROOT}"

monitor_gpu() {
  local output_csv="$1"
  echo "timestamp,gpu_index,used_mib,total_mib,utilization_gpu" > "${output_csv}"
  while true; do
    local ts
    ts="$(date -Iseconds)"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits \
      | awk -v ts="${ts}" -F, '{for (i=1;i<=NF;i++) gsub(/^ +| +$/, "", $i); print ts "," $1 "," $2 "," $3 "," $4}' \
      >> "${output_csv}" || true
    sleep "${GPU_POLL_INTERVAL}"
  done
}

summarize_run() {
  local run_dir="$1"
  local status="$2"
  local bs="$3"
  local peak
  peak="$(awk -F, 'NR > 1 && $3 > max {max=$3} END {print max+0}' "${run_dir}/gpu_mem.csv" 2>/dev/null || echo 0)"
  {
    echo "{"
    echo "  \"per_device_batch_size\": ${bs},"
    echo "  \"global_batch_size\": $((8 * bs)),"
    echo "  \"max_steps\": ${MAX_STEPS},"
    echo "  \"deepspeed\": \"${DEEPSPEED_CFG}\","
    echo "  \"status\": ${status},"
    echo "  \"peak_gpu_mem_mib\": ${peak},"
    echo "  \"finished_at\": \"$(date -Iseconds)\""
    echo "}"
  } > "${run_dir}/summary.json"
}

for bs in ${CANDIDATE_BS}; do
  run_name="bs${bs}_zero2_8gpu_320x640_steps${MAX_STEPS}_$(date +%Y%m%d_%H%M%S)"
  run_dir="${PROBE_ROOT}/${run_name}"
  output_dir="/defaultShare/dreamzero_outputs/${run_name}"
  mkdir -p "${run_dir}"

  {
    echo "run_name=${run_name}"
    echo "per_device_bs=${bs}"
    echo "global_batch_size=$((8 * bs))"
    echo "max_steps=${MAX_STEPS}"
    echo "deepspeed=${DEEPSPEED_CFG}"
    echo "output_dir=${output_dir}"
  } > "${run_dir}/run.env"

  monitor_gpu "${run_dir}/gpu_mem.csv" &
  monitor_pid=$!

  set +e
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  PER_DEVICE_BS="${bs}" \
  OUTPUT_DIR="${output_dir}" \
  DREAMZERO_METRICS_DIR="${PROBE_ROOT}" \
  DREAMZERO_METRICS_RUN_NAME="${run_name}" \
  bash scripts/train/droid_training_full_finetune_wan22_local_320x640.sh \
    2>&1 | tee "${run_dir}/stdout.log"
  status=${PIPESTATUS[0]}
  set -e

  kill "${monitor_pid}" >/dev/null 2>&1 || true
  wait "${monitor_pid}" >/dev/null 2>&1 || true

  summarize_run "${run_dir}" "${status}" "${bs}"

  if [ "${status}" -eq 0 ]; then
    echo "PASS bs=${bs}; logs: ${run_dir}"
    if [ "${STOP_AFTER_FIRST_SUCCESS}" = "1" ]; then
      break
    fi
  else
    echo "FAIL bs=${bs}; logs: ${run_dir}"
  fi
done
