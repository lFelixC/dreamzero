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

ai_station_mpi_host_rank_from_file() {
  local hostfile="$1"
  local current_hosts rank host
  current_hosts="$(hostname 2>/dev/null; hostname -s 2>/dev/null; hostname -f 2>/dev/null; hostname -I 2>/dev/null | tr ' ' '\n')"
  rank=0
  while IFS= read -r host; do
    host="${host%%:*}"
    if printf '%s\n' "${current_hosts}" | awk -v h="${host}" '$0 == h {found=1} END {exit found ? 0 : 1}'; then
      echo "${rank}"
      return 0
    fi
    rank=$((rank + 1))
  done < <(awk 'NF && $1 !~ /^#/ {host=$1; sub(/:.*/, "", host); if (!seen[host]++) print host}' "${hostfile}")
  return 1
}

ai_station_mpi_inside_mpi_rank() {
  [[ -n "${OMPI_COMM_WORLD_SIZE:-}" || -n "${PMI_SIZE:-}" || -n "${PMIX_SIZE:-}" ]]
}

ai_station_mpi_find_mpirun() {
  local candidate
  for candidate in \
    mpirun \
    mpiexec \
    /usr/local/mpi/bin/mpirun \
    /usr/local/openmpi/bin/mpirun \
    /opt/mpi/bin/mpirun \
    /opt/openmpi/bin/mpirun \
    /usr/lib64/openmpi/bin/mpirun; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      command -v "${candidate}"
      return 0
    fi
  done
  return 1
}

ai_station_mpi_maybe_relaunch() {
  local script_path="$1"
  shift

  local hostfile host_count mpirun_bin current_rank
  if [[ "${AI_STATION_MPI_REENTERED:-0}" == "1" ]] || ai_station_mpi_inside_mpi_rank; then
    return 0
  fi

  if ! hostfile="$(ai_station_mpi_find_hostfile)"; then
    if [[ "${AI_STATION_ALLOW_LOCAL_FALLBACK:-0}" == "1" ]]; then
      echo "WARN: MPI hostfile not found; running this script in the current container only"
      return 0
    fi
    echo "ERROR: MPI hostfile not found and this process is not inside an MPI rank."
    echo "Set AI_STATION_ALLOW_LOCAL_FALLBACK=1 only for intentional single-node debugging."
    exit 1
  fi

  host_count="$(ai_station_mpi_count_unique_hosts "${hostfile}")"

  if ! mpirun_bin="$(ai_station_mpi_find_mpirun)"; then
    if ai_station_mpi_is_uint "${host_count}" \
      && (( host_count >= 1 )) \
      && current_rank="$(ai_station_mpi_host_rank_from_file "${hostfile}")"; then
      export OMPI_COMM_WORLD_SIZE="${host_count}"
      export OMPI_COMM_WORLD_RANK="${current_rank}"
      export OMPI_COMM_WORLD_LOCAL_RANK=0
      export OMPI_COMM_WORLD_LOCAL_SIZE=1
      export AI_STATION_MPI_REENTERED=1
      echo "WARN: mpirun not found; derived rank ${current_rank}/${host_count} from ${hostfile}"
      return 0
    fi

    if [[ "${AI_STATION_ALLOW_LOCAL_FALLBACK:-0}" == "1" ]]; then
      echo "WARN: mpirun not found; running this script in the current container only"
      return 0
    fi
    echo "ERROR: mpirun not found and current host is not listed in ${hostfile}."
    echo "Install OpenMPI/mpirun in the image, or make AI Station run this script on worker hosts."
    exit 1
  fi

  if ! ai_station_mpi_is_uint "${host_count}" || (( host_count < 2 )); then
    if [[ "${AI_STATION_ALLOW_LOCAL_FALLBACK:-0}" == "1" ]]; then
      echo "WARN: MPI hostfile ${hostfile} has ${host_count:-0} unique host; running locally"
      return 0
    fi
    echo "ERROR: MPI hostfile ${hostfile} has ${host_count:-0} unique host."
    echo "Set AI_STATION_ALLOW_LOCAL_FALLBACK=1 only for intentional single-node debugging."
    exit 1
  fi

  if current_rank="$(ai_station_mpi_host_rank_from_file "${hostfile}")"; then
    export OMPI_COMM_WORLD_SIZE="${host_count}"
    export OMPI_COMM_WORLD_RANK="${current_rank}"
    export OMPI_COMM_WORLD_LOCAL_RANK=0
    export OMPI_COMM_WORLD_LOCAL_SIZE=1
    export AI_STATION_MPI_REENTERED=1
    echo "INFO: current host is listed in ${hostfile}; derived rank ${current_rank}/${host_count}"
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

  exec "${mpirun_bin}" \
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
