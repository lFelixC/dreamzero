#!/bin/bash
set -euo pipefail

# AI Station/OpenMPI wrapper for the A800 first-frame MoT experiment.
# Submit this script as the MPI job script. It keeps the experiment settings in
# run_mot_first_frame_2node.sh and only maps MPI topology to torchrun topology.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DREAMZERO_PROFILE="${DREAMZERO_PROFILE:-/etc/profile.d/dreamzero-uv.sh}"
if [[ -f "${DREAMZERO_PROFILE}" ]]; then
  source "${DREAMZERO_PROFILE}"
else
  echo "WARN: ${DREAMZERO_PROFILE} not found; using existing environment/default paths"
fi

export DREAMZERO_ROOT="${DREAMZERO_ROOT:-/data/dreamzero_mot}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/data/checkpoints/dreamzero}"
export DATASET_ROOT="${DATASET_ROOT:-/data/datasets/dreamzero/droid_lerobot}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MASTER_PORT="${MASTER_PORT:-29410}"
export OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_ROOT}/dreamzero_droid_wan22_mot_a800_first_frame_mpi}"

source "${SCRIPT_DIR}/ai_station_mpi_common.sh"
ai_station_mpi_maybe_relaunch "${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")" "$@"

is_uint() {
  [[ "${1:-}" =~ ^[0-9]+$ ]]
}

as_uint_or() {
  local value="${1:-}"
  local default_value="$2"
  if is_uint "${value}"; then
    echo "${value}"
  else
    echo "${default_value}"
  fi
}

first_host_from_list() {
  local hosts="${1:-}"
  local host
  host="$(printf '%s\n' "${hosts}" | tr ',;' '\n' | awk 'NF {print $1; exit}')"
  host="${host%%:*}"
  if [[ -n "${host}" ]]; then
    echo "${host}"
    return 0
  fi
  return 1
}

first_host_from_file() {
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

resolve_master_addr() {
  local env_name host hostfile

  if [[ -n "${MASTER_ADDR:-}" ]]; then
    echo "${MASTER_ADDR}"
    return 0
  fi

  for env_name in MPI_MASTER_ADDR PET_MASTER_ADDR TORCH_MASTER_ADDR CHIEF_IP MASTER_IP MASTER_NODE_ADDR; do
    host="${!env_name:-}"
    if [[ -n "${host}" ]]; then
      echo "${host%%:*}"
      return 0
    fi
  done

  for env_name in VC_WORKER_HOSTS WORKER_HOSTS MPI_WORKER_HOSTS HOSTS; do
    if host="$(first_host_from_list "${!env_name:-}")"; then
      echo "${host}"
      return 0
    fi
  done

  for hostfile in \
    "${OMPI_MCA_orte_default_hostfile:-}" \
    "${OMPI_MCA_prte_default_hostfile:-}" \
    "${MPI_HOSTFILE:-}" \
    "${HOSTFILE:-}" \
    "${PBS_NODEFILE:-}" \
    /etc/mpi/hostfile \
    /etc/mpi/mpi-hosts \
    /job/hostfile; do
    if host="$(first_host_from_file "${hostfile}")"; then
      echo "${host}"
      return 0
    fi
  done

  return 1
}

resolve_master_addr_via_shared_file() {
  local start_epoch="$1"
  local job_token addr_file addr tmp_addr mtime
  job_token="${OMPI_MCA_orte_ess_jobid:-${OMPI_MCA_orte_hnp_uri:-${PMIX_NAMESPACE:-default}}}"
  job_token="$(printf '%s' "${job_token}" | tr -c 'A-Za-z0-9_.-' '_')"
  addr_file="${MPI_MASTER_ADDR_FILE:-${DREAMZERO_ROOT}/.mpi_master_addr_${job_token}}"

  if [[ "${MPI_GLOBAL_RANK}" == "0" ]]; then
    mkdir -p "$(dirname "${addr_file}")"
    rm -f "${addr_file}"
    tmp_addr="$(hostname -I 2>/dev/null | awk '{print $1}')"
    addr="${tmp_addr:-$(hostname -f 2>/dev/null || hostname)}"
    printf '%s\n' "${addr}" > "${addr_file}"
    echo "${addr}"
    return 0
  fi

  for _ in $(seq 1 180); do
    if [[ -s "${addr_file}" ]]; then
      mtime="$(stat -c %Y "${addr_file}" 2>/dev/null || echo 0)"
      if is_uint "${mtime}" && (( mtime + 300 >= start_epoch )); then
        head -n 1 "${addr_file}"
        return 0
      fi
    fi
    sleep 1
  done

  return 1
}

MPI_WRAPPER_START_EPOCH="$(date +%s)"
MPI_WORLD_SIZE="$(as_uint_or "${OMPI_COMM_WORLD_SIZE:-${PMI_SIZE:-${PMIX_SIZE:-${MPI_WORLD_SIZE:-${WORLD_SIZE:-1}}}}}" 1)"
MPI_GLOBAL_RANK="$(as_uint_or "${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${PMIX_RANK:-${MPI_RANK:-${RANK:-0}}}}}" 0)"
MPI_LOCAL_RANK="$(as_uint_or "${OMPI_COMM_WORLD_LOCAL_RANK:-${PMI_LOCAL_RANK:-${MPI_LOCALRANKID:-${LOCAL_RANK:-${SLURM_LOCALID:-0}}}}}" 0)"
MPI_LOCAL_SIZE="$(as_uint_or "${OMPI_COMM_WORLD_LOCAL_SIZE:-${PMI_LOCAL_SIZE:-${MPI_LOCALNRANKS:-${MPI_LOCAL_SIZE:-1}}}}" 1)"

ai_station_mpi_prepare_master_addr "${MPI_WRAPPER_START_EPOCH}" "${MPI_WORLD_SIZE}" "${MPI_GLOBAL_RANK}"

if (( MPI_LOCAL_SIZE < 1 )); then
  MPI_LOCAL_SIZE=1
fi

DERIVED_NNODES=$(((MPI_WORLD_SIZE + MPI_LOCAL_SIZE - 1) / MPI_LOCAL_SIZE))
DERIVED_NODE_RANK=$((MPI_GLOBAL_RANK / MPI_LOCAL_SIZE))

export NNODES="${NNODES:-${DERIVED_NNODES}}"
export NODE_RANK="${NODE_RANK:-${DERIVED_NODE_RANK}}"

if (( MPI_LOCAL_RANK != 0 )); then
  echo "MPI local rank ${MPI_LOCAL_RANK} exits; local rank 0 launches torchrun for this node."
  exit 0
fi

if ! MASTER_ADDR_RESOLVED="$(resolve_master_addr)"; then
  if ! MASTER_ADDR_RESOLVED="$(resolve_master_addr_via_shared_file "${MPI_WRAPPER_START_EPOCH}")"; then
    echo "ERROR: Could not resolve MASTER_ADDR. Set MASTER_ADDR in the AI Station job env."
    exit 1
  fi
fi
export MASTER_ADDR="${MASTER_ADDR_RESOLVED}"

echo "========== AI Station MPI wrapper =========="
echo "SCRIPT=$(basename "${BASH_SOURCE[0]}")"
echo "MPI_WORLD_SIZE=${MPI_WORLD_SIZE}"
echo "MPI_GLOBAL_RANK=${MPI_GLOBAL_RANK}"
echo "MPI_LOCAL_RANK=${MPI_LOCAL_RANK}"
echo "MPI_LOCAL_SIZE=${MPI_LOCAL_SIZE}"
echo "NNODES=${NNODES}"
echo "NODE_RANK=${NODE_RANK}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "============================================"

exec bash "${SCRIPT_DIR}/run_mot_first_frame_2node.sh" "$@"
