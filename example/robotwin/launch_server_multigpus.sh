#!/bin/bash
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

START_PORT=${START_PORT:-29556}
MASTER_PORT=${MASTER_PORT:-29661}
NUM_GPUS=${NUM_GPUS:-8}
CKPT=${CKPT:-/data/checkpoints/dreamzero/dreamzero_robotwin/checkpoint-30000}
MAX_CHUNK_SIZE=${MAX_CHUNK_SIZE:-}
SAVE_SERVER_VIDEO=${SAVE_SERVER_VIDEO:-0}
LOG_DIR=${LOG_DIR:-'./logs'}
mkdir -p $LOG_DIR

save_root=${OUTPUT_ROOT:-'./visualization/'}
mkdir -p $save_root

batch_time=$(date +%Y%m%d_%H%M%S)


for ((i=0; i<NUM_GPUS; i++)); do
    CURRENT_PORT=$((START_PORT + i))
    CURRENT_MASTER_PORT=$((MASTER_PORT + i))

    LOG_FILE="${LOG_DIR}/server_${i}_${batch_time}.log"
    echo "[Server ${i}] GPU: ${i} | PORT: ${CURRENT_PORT} | MASTER_PORT: ${CURRENT_MASTER_PORT} | Log: ${LOG_FILE}"

    extra_args=()
    if [[ -n "${MAX_CHUNK_SIZE}" ]]; then
        extra_args+=(--max_chunk_size "${MAX_CHUNK_SIZE}")
    fi
    if [[ "${SAVE_SERVER_VIDEO}" == "1" ]]; then
        extra_args+=(--save_server_video)
    fi

    CUDA_VISIBLE_DEVICES=$i  \
    nohup python -m torch.distributed.run \
        --nproc_per_node 1 \
        --master_port $CURRENT_MASTER_PORT \
        socket_test_optimized_aloha_x5lite_bimanual.py \
        --model_path "$CKPT" \
        --output_root "$save_root" \
        --port $CURRENT_PORT \
        "${extra_args[@]}" > $LOG_FILE 2>&1 &
    sleep 2;
done

echo "All ${NUM_GPUS} instances have been launched in the background."
wait
