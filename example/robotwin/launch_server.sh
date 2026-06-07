#!/bin/bash
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

START_PORT=${START_PORT:-29056}
MASTER_PORT=${MASTER_PORT:-29061}
CKPT=${CKPT:-/data/checkpoints/dreamzero/dreamzero_robotwin/checkpoint-30000}
SERVER_GPU=${SERVER_GPU:-0}
SERVER_NPROC=${SERVER_NPROC:-1}
MAX_CHUNK_SIZE=${MAX_CHUNK_SIZE:-}
SAVE_SERVER_VIDEO=${SAVE_SERVER_VIDEO:-0}

save_root=${OUTPUT_ROOT:-'visualization/'}
mkdir -p $save_root

extra_args=()
if [[ -n "${MAX_CHUNK_SIZE}" ]]; then
    extra_args+=(--max_chunk_size "${MAX_CHUNK_SIZE}")
fi
if [[ "${SAVE_SERVER_VIDEO}" == "1" ]]; then
    extra_args+=(--save_server_video)
fi

CUDA_VISIBLE_DEVICES=${SERVER_GPU} python -m torch.distributed.run \
    --nproc_per_node ${SERVER_NPROC} \
    --master_port $MASTER_PORT \
    socket_test_optimized_aloha_x5lite_bimanual.py \
    --model_path "$CKPT" \
    --port $START_PORT \
    --output_root "$save_root" \
    "${extra_args[@]}"
