#!/bin/bash
set -euo pipefail

# Fixed local setup. Edit only these lines if your paths or GPU selection change.
DREAMZERO_ROOT="/data/dreamzero"
CHECKPOINT_ROOT="/data/checkpoints/dreamzero"
DATASET_ROOT="/data/datasets/dreamzero"

WAN22_CKPT_DIR="${CHECKPOINT_ROOT}/Wan2.2-TI2V-5B"
IMAGE_ENCODER_DIR="${CHECKPOINT_ROOT}/Wan2.1-I2V-14B-480P"
TOKENIZER_DIR="${CHECKPOINT_ROOT}/umt5-xxl"
OUTPUT_DIR="${CHECKPOINT_ROOT}/dreamzero_droid_wan22_full_finetune"

PYTHON_BIN="${DREAMZERO_ROOT}/.venv/bin/python"
EXPERIMENT_PY="${DREAMZERO_ROOT}/groot/vla/experiment/experiment.py"

export CUDA_VISIBLE_DEVICES="4,5,6,7"
export CUDA_HOME="/usr/local/cuda-12.9"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export HYDRA_FULL_ERROR=1
export SWANLAB_SYNC_WANDB=1      # 让 wandb 日志同时同步到 SwanLab
export WANDB_MODE=offline        # wandb 本地离线落盘；SwanLab 通过 sync_wandb 接管展示

PER_DEVICE_BS=64                 # 每张 GPU 一次 forward/backward 真正装进去的 micro-batch
DEEPSPEED_CFG="zero2"            # DeepSpeed 配置；显存紧张时用 offload 更稳
MAX_STEPS=50000                  # 总训练步数

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#GPU_IDS[@]}"
GLOBAL_BATCH_SIZE=$((NUM_GPUS * PER_DEVICE_BS))  # 有效 batch size；默认等于“不做梯度累积”

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "ERROR: Python not found at ${PYTHON_BIN}"
    echo "Please make sure /data/dreamzero/.venv is created."
    exit 1
fi

if [ ! -f "${EXPERIMENT_PY}" ]; then
    echo "ERROR: experiment.py not found at ${EXPERIMENT_PY}"
    exit 1
fi

resolve_dataset_root() {
    local base="$1"
    local candidate

    if [ -f "${base}/meta/modality.json" ]; then
        echo "${base}"
        return 0
    fi

    if [ -f "${base}/droid_lerobot/meta/modality.json" ]; then
        echo "${base}/droid_lerobot"
        return 0
    fi

    candidate="$(find "${base}" -maxdepth 3 -path '*/meta/modality.json' 2>/dev/null | head -n 1 || true)"
    if [ -n "${candidate}" ]; then
        dirname "$(dirname "${candidate}")"
        return 0
    fi

    return 1
}

if ! DROID_DATA_ROOT="$(resolve_dataset_root "${DATASET_ROOT}")"; then
    echo "ERROR: Could not find meta/modality.json under ${DATASET_ROOT}"
    echo "Expected either:"
    echo "  ${DATASET_ROOT}/meta/modality.json"
    echo "or"
    echo "  ${DATASET_ROOT}/droid_lerobot/meta/modality.json"
    exit 1
fi

echo "Using DROID_DATA_ROOT=${DROID_DATA_ROOT}"

if [ ! -d "${WAN22_CKPT_DIR}" ] || [ -z "$(ls -A "${WAN22_CKPT_DIR}" 2>/dev/null)" ]; then
    echo "Wan2.2-TI2V-5B not found at ${WAN22_CKPT_DIR}. Downloading..."
    huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir "${WAN22_CKPT_DIR}"
fi

if [ ! -d "${TOKENIZER_DIR}" ] || [ -z "$(ls -A "${TOKENIZER_DIR}" 2>/dev/null)" ]; then
    echo "umt5-xxl tokenizer not found at ${TOKENIZER_DIR}. Downloading..."
    huggingface-cli download google/umt5-xxl --local-dir "${TOKENIZER_DIR}"
fi

if [ ! -f "${IMAGE_ENCODER_DIR}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" ]; then
    echo "CLIP image encoder not found at ${IMAGE_ENCODER_DIR}. Downloading Wan2.1 checkpoint for CLIP..."
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "${IMAGE_ENCODER_DIR}"
fi

mkdir -p "${OUTPUT_DIR}"
cd "${DREAMZERO_ROOT}"

TRAIN_OVERRIDES=(
    "report_to=wandb"                                                           # 日志上报到 wandb；不想联网可改成 none
    "data=dreamzero/droid_relative_wan22"                                       # 使用的 Hydra 数据配置：DROID + Wan2.2 分辨率版本
    "wandb_project=dreamzero"                                                   # wandb project 名称
    "train_architecture=full"                                                   # 全量微调，不是 LoRA
    "num_frames=33"                                                             # 每个样本输入的视频帧数
    "action_horizon=24"                                                         # 每次预测的动作 horizon 长度
    "num_views=3"                                                               # 相机视角数量
    "model=dreamzero/vla"                                                       # 顶层模型配置
    "model/dreamzero/action_head=wan_flow_matching_action_tf_wan22"             # 使用 Wan2.2 5B action head 配置
    "model/dreamzero/transform=dreamzero_cotrain"                               # 数据增强 / 预处理配置
    "num_frame_per_block=2"                                                     # 每个 video block 包含的帧数
    "num_action_per_block=24"                                                   # 每个 block 对应的动作 token 数
    "num_state_per_block=1"                                                     # 每个 block 对应的状态 token 数
    "seed=42"                                                                   # 随机种子
    "training_args.learning_rate=1e-5"                                          # 学习率
    "training_args.deepspeed=groot/vla/configs/deepspeed/${DEEPSPEED_CFG}.json" # DeepSpeed 配置文件
    "save_steps=1000"                                                           # 每多少 step 存一次 checkpoint
    "training_args.warmup_ratio=0.05"                                           # warmup 比例
    "output_dir=${OUTPUT_DIR}"                                                  # checkpoint / 日志输出目录
    "per_device_train_batch_size=${PER_DEVICE_BS}"                              # 每张 GPU 的 micro-batch size
    "global_batch_size=${GLOBAL_BATCH_SIZE}"                                    # 目标有效 batch size；训练器会自动换算梯度累积
    "max_steps=${MAX_STEPS}"                                                    # 最大训练步数
    "weight_decay=1e-5"                                                         # 权重衰减
    "save_total_limit=10"                                                       # 最多保留多少个 checkpoint
    "upload_checkpoints=false"                                                  # 不自动上传 checkpoint
    "bf16=true"                                                                 # 训练时使用 bfloat16
    "tf32=true"                                                                 # 允许 matmul 使用 TF32
    "eval_bf16=true"                                                            # eval 也使用 bfloat16
    "dataloader_pin_memory=true"                                                # dataloader 开启 pin_memory
    "dataloader_num_workers=4"                                                  # dataloader worker 数
    "image_resolution_width=320"                                                # 训练输入宽度
    "image_resolution_height=160"                                               # 训练输入高度
    "save_lora_only=false"                                                      # 全量微调时保存完整权重，不只保存 LoRA
    "max_chunk_size=4"                                                          # 数据 / 序列 chunk 上限
    "save_strategy=steps"                                                       # 按 step 保存，而不是按 epoch
    "droid_data_root=${DROID_DATA_ROOT}"                                        # DROID LeRobot 数据集根目录
    "dit_version=${WAN22_CKPT_DIR}"                                             # Wan2.2 主权重目录
    "text_encoder_pretrained_path=${WAN22_CKPT_DIR}/models_t5_umt5-xxl-enc-bf16.pth" # Wan2.2 里的文本编码器权重
    "image_encoder_pretrained_path=${IMAGE_ENCODER_DIR}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" # CLIP 图像编码器权重
    "vae_pretrained_path=${WAN22_CKPT_DIR}/Wan2.2_VAE.pth"                      # Wan2.2 VAE 权重
    "tokenizer_path=${TOKENIZER_DIR}"                                           # UMT5 tokenizer 目录
)

"${PYTHON_BIN}" -m torch.distributed.run --nproc_per_node "${NUM_GPUS}" --standalone "${EXPERIMENT_PY}" "${TRAIN_OVERRIDES[@]}"
