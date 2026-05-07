#!/bin/bash

# Shared environment bootstrap for the non-MPI A800 launchers.

dreamzero_a800_prepend_path() {
  local dir="$1"
  if [[ -n "${dir}" && -d "${dir}" ]]; then
    case ":${PATH:-}:" in
      *":${dir}:"*) ;;
      *) export PATH="${dir}:${PATH:-}" ;;
    esac
  fi
}

dreamzero_a800_source_env() {
  local profile="${DREAMZERO_PROFILE:-/etc/profile.d/dreamzero-uv.sh}"

  if [[ -f "${profile}" ]]; then
    # shellcheck disable=SC1090
    source "${profile}"
  else
    echo "WARN: ${profile} not found; using existing environment/default paths"
  fi

  if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ -x /opt/venvs/dreamzero/bin/python ]]; then
      export VIRTUAL_ENV=/opt/venvs/dreamzero
    else
      export VIRTUAL_ENV=/data/dreamzero/.venv
    fi
  fi

  export PYTHON_BIN="${PYTHON_BIN:-${VIRTUAL_ENV}/bin/python}"
  dreamzero_a800_prepend_path /root/.local/bin
  dreamzero_a800_prepend_path /usr/local/bin
  dreamzero_a800_prepend_path "${VIRTUAL_ENV}/bin"

  if [[ -z "${DREAMZERO_ROOT:-}" ]]; then
    if [[ -d /2023133163/liuf/dreamzero ]]; then
      export DREAMZERO_ROOT=/2023133163/liuf/dreamzero
    else
      export DREAMZERO_ROOT=/data/dreamzero_mot
    fi
  fi

  if [[ -z "${DATASET_ROOT:-}" ]]; then
    if [[ -d /2023133163/datasets/dreamzero ]]; then
      export DATASET_ROOT=/2023133163/datasets/dreamzero
    else
      export DATASET_ROOT=/data/datasets/dreamzero/droid_lerobot
    fi
  fi

  if [[ -z "${CHECKPOINT_ROOT:-}" ]]; then
    if [[ -d /2023133163/checkpoints/dreamzero ]]; then
      export CHECKPOINT_ROOT=/2023133163/checkpoints/dreamzero
    else
      export CHECKPOINT_ROOT=/data/checkpoints/dreamzero
    fi
  fi

  export PYTHONPATH="${DREAMZERO_ROOT}:${PYTHONPATH:-}"
  export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
  export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"

  if command -v nvcc >/dev/null 2>&1; then
    export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
  elif [[ -z "${CUDA_HOME:-}" && -d /usr/local/cuda-12.9 ]]; then
    export CUDA_HOME=/usr/local/cuda-12.9
  fi

  if [[ -n "${CUDA_HOME:-}" ]]; then
    dreamzero_a800_prepend_path "${CUDA_HOME}/bin"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
  fi
}
