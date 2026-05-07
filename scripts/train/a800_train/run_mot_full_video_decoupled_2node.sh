#!/bin/bash
set -euo pipefail

# A800 2-node MoT experiment: full-video attention with decoupled video/action noise.
# Run this script on every node in the experiment with the same MASTER_ADDR
# and MASTER_PORT, changing only NODE_RANK.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${SCRIPT_DIR}/dreamzero_env_common.sh"
dreamzero_a800_source_env

export DREAMZERO_ROOT="${DREAMZERO_ROOT:-/data/dreamzero_mot}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/data/checkpoints/dreamzero}"
export DATASET_ROOT="${DATASET_ROOT:-/data/datasets/dreamzero/droid_lerobot}"

export NNODES="${NNODES:-2}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29430}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

export MOT_ACTION_VIDEO_ATTENTION="full_video"
export MOT_DECOUPLE_VIDEO_ACTION_NOISE="true"
export MOT_VIDEO_NOISE_BETA_ALPHA="${MOT_VIDEO_NOISE_BETA_ALPHA:-3.0}"
export MOT_VIDEO_NOISE_BETA_BETA="${MOT_VIDEO_NOISE_BETA_BETA:-1.0}"
export MOT_DECOUPLED_INFERENCE_VIDEO_FINAL_NOISE="${MOT_DECOUPLED_INFERENCE_VIDEO_FINAL_NOISE:-0.8}"
export MOT_DECOUPLED_INFERENCE_VIDEO_REFRESH_STEPS="${MOT_DECOUPLED_INFERENCE_VIDEO_REFRESH_STEPS:-8}"
export MOT_INFERENCE_VIDEO_MODE="${MOT_INFERENCE_VIDEO_MODE:-auto}"
export OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_ROOT}/dreamzero_droid_wan22_mot_a800_full_video_decoupled_2node}"

exec bash "${TRAIN_DIR}/run_dreamzero_mot_multinode.sh" "$@"
