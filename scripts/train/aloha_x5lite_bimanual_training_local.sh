#!/bin/bash
set -euo pipefail

# Fixed local setup. Edit only these lines if your paths, GPU selection, or batch size change.
DREAMZERO_ROOT="/data/dreamzero"
CHECKPOINT_ROOT="/data/checkpoints/dreamzero"
ALOHA_DATA_ROOT="/data/datasets/dreamzero/1k_demo_lerobot_merged_v4_shuffle_only_clipped"
PRETRAINED_MODEL_PATH="${CHECKPOINT_ROOT}/DreamZero-AgiBot"
WAN21_CKPT_DIR="${CHECKPOINT_ROOT}/Wan2.1-I2V-14B-480P"
TOKENIZER_DIR="${CHECKPOINT_ROOT}/umt5-xxl"
OUTPUT_DIR="${CHECKPOINT_ROOT}/dreamzero_aloha_x5lite_bimanual_full_finetune_shuffle"

PYTHON_BIN="${DREAMZERO_ROOT}/.venv/bin/python"
EXPERIMENT_PY="${DREAMZERO_ROOT}/groot/vla/experiment/experiment.py"

export CUDA_VISIBLE_DEVICES="4,5,6,7"
export CUDA_HOME="/usr/local/cuda-12.9"
export PATH="${DREAMZERO_ROOT}/.venv/bin:${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export HYDRA_FULL_ERROR=1
export SWANLAB_SYNC_WANDB=1      # 让 wandb 日志同时同步到 SwanLab
export WANDB_MODE=offline        # wandb 本地离线落盘；SwanLab 通过 sync_wandb 接管展示

PER_DEVICE_BS=2                  # 每张 GPU 的 micro-batch；若 OOM，手动改成 16
GRAD_ACCUM_STEPS=1               # 梯度累积步数；>1 时会增大全局 batch 而不增加单卡显存
DEEPSPEED_CFG="zero2"            # ALOHA + DreamZero-AgiBot 更稳的默认值
MAX_STEPS=50000                  # 总训练步数
SAVE_STEPS=1000                  # 每多少步保存一次 checkpoint；减少保存频率更稳
IMAGE_RESOLUTION_WIDTH=320       # 训练输入宽度
IMAGE_RESOLUTION_HEIGHT=176      # 训练输入高度
MAX_CHUNK_SIZE=4                 # 数据 chunk 上限
DATALOADER_NUM_WORKERS=1         # 8 卡视频训练时先用保守 worker 数，降低 host RAM 压力
DATALOADER_PIN_MEMORY=false      # checkpoint 保存时 pinned memory 更容易放大内存峰值
DATALOADER_PERSISTENT_WORKERS=false  # 降低长期驻留 worker 的内存占用

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#GPU_IDS[@]}"
GLOBAL_BATCH_SIZE=$((NUM_GPUS * PER_DEVICE_BS * GRAD_ACCUM_STEPS))  # 有效 batch size = GPU数 * micro-batch * 梯度累积步数

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "ERROR: Python not found at ${PYTHON_BIN}"
    echo "Please make sure /data/dreamzero/.venv is created."
    exit 1
fi

if [ ! -f "${EXPERIMENT_PY}" ]; then
    echo "ERROR: experiment.py not found at ${EXPERIMENT_PY}"
    exit 1
fi

if [ ! -f "${ALOHA_DATA_ROOT}/meta/modality.json" ]; then
    echo "ERROR: DreamZero metadata missing at ${ALOHA_DATA_ROOT}/meta/modality.json"
    echo "Run the merged dataset build and convert_lerobot_to_gear step first."
    exit 1
fi

if [ ! -d "${PRETRAINED_MODEL_PATH}" ]; then
    echo "ERROR: pretrained model path not found at ${PRETRAINED_MODEL_PATH}"
    exit 1
fi

if [ ! -f "${WAN21_CKPT_DIR}/models_t5_umt5-xxl-enc-bf16.pth" ]; then
    echo "ERROR: Wan2.1 text encoder missing at ${WAN21_CKPT_DIR}"
    exit 1
fi

if [ ! -f "${WAN21_CKPT_DIR}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" ]; then
    echo "ERROR: Wan2.1 image encoder missing at ${WAN21_CKPT_DIR}"
    exit 1
fi

if [ ! -f "${WAN21_CKPT_DIR}/Wan2.1_VAE.pth" ]; then
    echo "ERROR: Wan2.1 VAE missing at ${WAN21_CKPT_DIR}"
    exit 1
fi

if [ ! -d "${TOKENIZER_DIR}" ]; then
    echo "ERROR: tokenizer path not found at ${TOKENIZER_DIR}"
    exit 1
fi

echo "Using ALOHA_DATA_ROOT=${ALOHA_DATA_ROOT}"
echo "Using PRETRAINED_MODEL_PATH=${PRETRAINED_MODEL_PATH}"
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Using PER_DEVICE_BS=${PER_DEVICE_BS}, GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS}, GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"

mkdir -p "${OUTPUT_DIR}"
cd "${DREAMZERO_ROOT}"

TRAIN_OVERRIDES=(
    "report_to=wandb"                                                           # 日志上报到 wandb
    "data=dreamzero/aloha_x5lite_bimanual_relative"                             # ALOHA X5lite 双臂数据配置
    "wandb_project=dreamzero"                                                   # wandb project 名称
    "train_architecture=full"                                                   # 全量微调，不是 LoRA
    "num_frames=33"                                                             # 每个样本输入的视频帧数
    "action_horizon=24"                                                         # 每次预测的动作 horizon 长度
    "num_views=3"                                                               # 相机视角数量
    "model=dreamzero/vla"                                                       # 顶层模型配置
    "model/dreamzero/action_head=wan_flow_matching_action_tf"                   # Wan2.1 flow matching action head
    "model/dreamzero/transform=dreamzero_cotrain"                               # 预处理 / 文本模板配置
    "num_frame_per_block=2"                                                     # 每个 video block 包含的帧数
    "num_action_per_block=24"                                                   # 每个 block 对应的动作 token 数
    "num_state_per_block=1"                                                     # 每个 block 对应的状态 token 数
    "seed=42"                                                                   # 随机种子
    "training_args.learning_rate=1e-5"                                          # 学习率
    "training_args.deepspeed=groot/vla/configs/deepspeed/${DEEPSPEED_CFG}.json"
    "save_steps=${SAVE_STEPS}"                                                  # 每多少 step 存一次 checkpoint
    "training_args.warmup_ratio=0.05"                                           # warmup 比例
    "output_dir=${OUTPUT_DIR}"                                                  # checkpoint / 日志输出目录
    "per_device_train_batch_size=${PER_DEVICE_BS}"                              # 每张 GPU 的 micro-batch size
    "global_batch_size=${GLOBAL_BATCH_SIZE}"                                    # 目标有效 batch size
    "max_steps=${MAX_STEPS}"                                                    # 最大训练步数
    "weight_decay=1e-5"                                                         # 权重衰减
    "save_total_limit=100"                                                      # 最多保留多少个 checkpoint
    "upload_checkpoints=false"                                                  # 不自动上传 checkpoint
    "bf16=true"                                                                 # 训练时使用 bfloat16
    "tf32=true"                                                                 # 允许 matmul 使用 TF32
    "eval_bf16=true"                                                            # eval 也使用 bfloat16
    "dataloader_pin_memory=${DATALOADER_PIN_MEMORY}"                            # 更保守的 dataloader 内存策略
    "dataloader_num_workers=${DATALOADER_NUM_WORKERS}"                          # dataloader worker 数
    "dataloader_persistent_workers=${DATALOADER_PERSISTENT_WORKERS}"            # 避免 worker 常驻放大内存峰值
    "image_resolution_width=${IMAGE_RESOLUTION_WIDTH}"                          # 训练输入宽度
    "image_resolution_height=${IMAGE_RESOLUTION_HEIGHT}"                        # 训练输入高度
    "save_lora_only=false"                                                      # 全量微调时保存完整权重
    "max_chunk_size=${MAX_CHUNK_SIZE}"                                          # 数据 / 序列 chunk 上限
    "save_strategy=steps"                                                       # 按 step 保存
    "aloha_data_root=${ALOHA_DATA_ROOT}"                                        # 合并后的 ALOHA LeRobot 根目录
    "pretrained_model_path=${PRETRAINED_MODEL_PATH}"                            # DreamZero-AgiBot 热启动权重
    "dit_version=${WAN21_CKPT_DIR}"                                             # Wan2.1 主权重目录
    "text_encoder_pretrained_path=${WAN21_CKPT_DIR}/models_t5_umt5-xxl-enc-bf16.pth" # Wan2.1 文本编码器
    "image_encoder_pretrained_path=${WAN21_CKPT_DIR}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
    "vae_pretrained_path=${WAN21_CKPT_DIR}/Wan2.1_VAE.pth"                      # Wan2.1 VAE 权重
    "tokenizer_path=${TOKENIZER_DIR}"                                           # UMT5 tokenizer 目录
    "++action_head_cfg.config.skip_component_loading=true"                      # 跳过缺失 action head 子组件的硬加载
    "++action_head_cfg.config.defer_lora_injection=true"                        # 延后 LoRA 注入，兼容 DreamZero-AgiBot 初始化
)

"${PYTHON_BIN}" -m torch.distributed.run --nproc_per_node "${NUM_GPUS}" --standalone "${EXPERIMENT_PY}" "${TRAIN_OVERRIDES[@]}"
