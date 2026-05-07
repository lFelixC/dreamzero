#!/bin/bash

# Shared helpers for AI Station MPI jobs.
# AI Station runs the selected script in a launcher container. These helpers
# re-enter the same script with mpirun so one rank lands on each worker node.

ai_station_mpi_is_uint() {
  [[ "${1:-}" =~ ^[0-9]+$ ]]
}

ai_station_mpi_first_host_from_file() {
  local hostfile="$1"
  local host
  if [[ -z "${hostfile}" || ! -f "${hostfile}" ]]; then
    return 1
  fi
  host="$(awk 'NF && $1 !~ /^#/ {print $1; exit}' "${hostfile}")"
  host="${host%%:*}"
  if [[ -n "${host}" ]]; then
    echo "${host}"
    return 0
  fi
  return 1
}

ai_station_mpi_find_hostfile() {
  local hostfile
  for hostfile in \
    "${OMPI_MCA_orte_default_hostfile:-}" \
    "${OMPI_MCA_prte_default_hostfile:-}" \
    "${MPI_HOSTFILE:-}" \
    "${HOSTFILE:-}" \
    "${PBS_NODEFILE:-}" \
    /etc/mpi/hostfile \
    /etc/mpi/mpi-hosts \
    /job/hostfile; do
    if ai_station_mpi_first_host_from_file "${hostfile}" >/dev/null; then
      echo "${hostfile}"
      return 0
    fi
  done
  return 1
}

ai_station_mpi_count_unique_hosts() {
  local hostfile="$1"
  awk 'NF && $1 !~ /^#/ {host=$1; sub(/:.*/, "", host); print host}' "${hostfile}" \
    | sort -u \
    | wc -l \
    | awk '{print $1}'
}

ai_station_mpi_inside_mpi_rank() {
  [[ -n "${OMPI_COMM_WORLD_SIZE:-}" || -n "${PMI_SIZE:-}" || -n "${PMIX_SIZE:-}" ]]
}

ai_station_mpi_maybe_relaunch() {
  local script_path="$1"
  shift

  local hostfile host_count
  if [[ "${AI_STATION_MPI_REENTERED:-0}" == "1" ]] || ai_station_mpi_inside_mpi_rank; then
    return 0
  fi

  if ! command -v mpirun >/dev/null 2>&1; then
    echo "WARN: mpirun not found; running this script in the current container only"
    return 0
  fi

  if ! hostfile="$(ai_station_mpi_find_hostfile)"; then
    echo "WARN: MPI hostfile not found; running this script in the current container only"
    return 0
  fi

  host_count="$(ai_station_mpi_count_unique_hosts "${hostfile}")"
  if ! ai_station_mpi_is_uint "${host_count}" || (( host_count < 2 )); then
    echo "WARN: MPI hostfile ${hostfile} has ${host_count:-0} unique host; running locally"
    return 0
  fi

  export AI_STATION_MPI_REENTERED=1
  export OMPI_ALLOW_RUN_AS_ROOT=1
  export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

  echo "========== AI Station mpirun launcher =========="
  echo "HOSTNAME=${HOSTNAME:-unknown}"
  echo "HOSTFILE=${hostfile}"
  echo "NNODES=${host_count}"
  echo "SCRIPT=${script_path}"
  echo "================================================"

  exec mpirun \
    --allow-run-as-root \
    --hostfile "${hostfile}" \
    --map-by ppr:1:node \
    -np "${host_count}" \
    -x AI_STATION_MPI_REENTERED \
    -x DREAMZERO_PROFILE \
    -x DREAMZERO_ROOT \
    -x CHECKPOINT_ROOT \
    -x DATASET_ROOT \
    -x PYTHON_BIN \
    -x PYTHONPATH \
    -x PATH \
    -x LD_LIBRARY_PATH \
    -x CUDA_VISIBLE_DEVICES \
    -x SWANLAB_SYNC_WANDB \
    -x SWANLAB_API_KEY \
    -x WANDB_MODE \
    -x WANDB_PROJECT \
    -x MASTER_PORT \
    -x OUTPUT_DIR \
    bash "${script_path}" "$@"
}

ai_station_mpi_rank0_addr_file() {
  local job_token
  job_token="${OMPI_MCA_orte_ess_jobid:-${OMPI_MCA_orte_hnp_uri:-${PMIX_NAMESPACE:-default}}}"
  job_token="$(printf '%s' "${job_token}" | tr -c 'A-Za-z0-9_.-' '_')"
  echo "${MPI_MASTER_ADDR_FILE:-${DREAMZERO_ROOT}/.mpi_master_addr_${job_token}}"
}

ai_station_mpi_prepare_master_addr() {
  local start_epoch="$1"
  local world_size="$2"
  local global_rank="$3"
  local addr_file addr tmp_addr mtime

  if [[ -n "${MASTER_ADDR:-}" ]]; then
    return 0
  fi

  if (( world_size <= 1 )); then
    export MASTER_ADDR=127.0.0.1
    return 0
  fi

  addr_file="$(ai_station_mpi_rank0_addr_file)"
  if [[ "${global_rank}" == "0" ]]; then
    mkdir -p "$(dirname "${addr_file}")"
    rm -f "${addr_file}"
    tmp_addr="$(hostname -I 2>/dev/null | awk '{print $1}')"
    addr="${tmp_addr:-$(hostname -f 2>/dev/null || hostname)}"
    printf '%s\n' "${addr}" > "${addr_file}"
    export MASTER_ADDR="${addr}"
    return 0
  fi

  for _ in $(seq 1 180); do
    if [[ -s "${addr_file}" ]]; then
      mtime="$(stat -c %Y "${addr_file}" 2>/dev/null || echo 0)"
      if ai_station_mpi_is_uint "${mtime}" && (( mtime + 300 >= start_epoch )); then
        MASTER_ADDR="$(head -n 1 "${addr_file}")"
        export MASTER_ADDR
        return 0
      fi
    fi
    sleep 1
  done

  echo "ERROR: Timed out waiting for rank0 MASTER_ADDR file: ${addr_file}"
  return 1
}
