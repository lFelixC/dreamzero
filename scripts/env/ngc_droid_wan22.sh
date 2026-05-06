#!/usr/bin/env bash
# NGC-oriented DreamZero DROID + Wan2.2 environment.
# Source from any shell before launching training:
#   source /2024234320/dreamzero/scripts/env/ngc_droid_wan22.sh

export DREAMZERO_ROOT="${DREAMZERO_ROOT:-/2024234320/dreamzero}"
export PYTHON_BIN="${PYTHON_BIN:-${DREAMZERO_ROOT}/.venv/bin/python}"

export DROID_DATA_ROOT="${DROID_DATA_ROOT:-${DREAMZERO_ROOT}/data/droid_lerobot}"
export WAN22_CKPT_DIR="${WAN22_CKPT_DIR:-${DREAMZERO_ROOT}/checkpoints/Wan2.2-TI2V-5B}"
export IMAGE_ENCODER_DIR="${IMAGE_ENCODER_DIR:-${DREAMZERO_ROOT}/checkpoints/Wan2.1-I2V-14B-480P}"
export TOKENIZER_DIR="${TOKENIZER_DIR:-${DREAMZERO_ROOT}/checkpoints/umt5-xxl}"
export OUTPUT_DIR="${OUTPUT_DIR:-/defaultShare/dreamzero_outputs/dreamzero_droid_wan22_full_finetune}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export CUDA_COMPAT_LIB_DIR="${CUDA_COMPAT_LIB_DIR:-${_CUDA_COMPAT_PATH:-/usr/local/cuda/compat}/lib}"
export PATH="${CUDA_HOME}/bin:${DREAMZERO_ROOT}/.venv/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_COMPAT_LIB_DIR}:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

export HPCX_ROOT="${HPCX_ROOT:-/opt/hpcx}"
_dreamzero_hpcx_ld_paths=""
for _dreamzero_hpcx_lib_dir in \
  "${HPCX_ROOT}/ucx/lib" \
  "${HPCX_ROOT}/ucc/lib" \
  "${HPCX_ROOT}/ompi/lib"; do
  if [ -d "${_dreamzero_hpcx_lib_dir}" ]; then
    _dreamzero_hpcx_ld_paths="${_dreamzero_hpcx_ld_paths:+${_dreamzero_hpcx_ld_paths}:}${_dreamzero_hpcx_lib_dir}"
  fi
done
if [ -n "${_dreamzero_hpcx_ld_paths}" ]; then
  export LD_LIBRARY_PATH="${_dreamzero_hpcx_ld_paths}:${LD_LIBRARY_PATH}"
fi
unset _dreamzero_hpcx_ld_paths _dreamzero_hpcx_lib_dir

export HF_HOME="${HF_HOME:-/defaultShare/dreamzero_cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/defaultShare/dreamzero_cache/pip}"
export UV_INDEX_URL="${UV_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/defaultShare/dreamzero_cache/uv}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export SWANLAB_SYNC_WANDB="${SWANLAB_SYNC_WANDB:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"

if [ -z "${NUM_GPUS:-}" ]; then
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -r -a _dreamzero_gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
    export NUM_GPUS="${#_dreamzero_gpu_ids[@]}"
    unset _dreamzero_gpu_ids
  elif command -v nvidia-smi >/dev/null 2>&1; then
    export NUM_GPUS="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
  else
    export NUM_GPUS=1
  fi
fi

if [ -f "${DREAMZERO_ROOT}/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${DREAMZERO_ROOT}/.venv/bin/activate"
fi
