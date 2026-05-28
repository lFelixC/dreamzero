#!/bin/bash
set -euo pipefail

# Lightweight local training entry for the AgileX ALOHA X5lite LeRobot shards.
# It keeps the ALOHA embodiment/key contract, but uses the same sharded mixture
# path style as the DROID training scripts by passing a glob of dataset roots.

DREAMZERO_ROOT="${DREAMZERO_ROOT:-/data/dreamzero}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/data/checkpoints/dreamzero}"
AGILEX_ALOHA_DATASET_ROOT="${AGILEX_ALOHA_DATASET_ROOT:-/data/datasets/dreamzero/dreamzero_agilex_aloha/datasets}"
AGILEX_ALOHA_DATA_GLOB="${AGILEX_ALOHA_DATA_GLOB:-${AGILEX_ALOHA_DATASET_ROOT}/*}"
export AGILEX_ALOHA_DATASET_ROOT
export AGILEX_ALOHA_DATA_GLOB

WAN22_CKPT_DIR="${WAN22_CKPT_DIR:-${CHECKPOINT_ROOT}/Wan2.2-TI2V-5B}"
IMAGE_ENCODER_DIR="${IMAGE_ENCODER_DIR:-${WAN22_CKPT_DIR}}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${CHECKPOINT_ROOT}/umt5-xxl}"
OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_ROOT}/dreamzero_agilex_aloha_wan22_full_finetune}"

if [ ! -d "${TOKENIZER_DIR}" ] && [ -d "${WAN22_CKPT_DIR}/google/umt5-xxl" ]; then
    TOKENIZER_DIR="${WAN22_CKPT_DIR}/google/umt5-xxl"
fi

PYTHON_BIN="${PYTHON_BIN:-${DREAMZERO_ROOT}/.venv/bin/python}"
EXPERIMENT_PY="${DREAMZERO_ROOT}/groot/vla/experiment/experiment.py"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.9}"
export PATH="${DREAMZERO_ROOT}/.venv/bin:${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export SWANLAB_SYNC_WANDB="${SWANLAB_SYNC_WANDB:-1}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"

REPORT_TO="${REPORT_TO:-wandb}"
PER_DEVICE_BS="${PER_DEVICE_BS:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
DEEPSPEED_CFG="${DEEPSPEED_CFG:-zero2}"
MAX_STEPS="${MAX_STEPS:-50000}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-20}"

IMAGE_RESOLUTION_WIDTH="${IMAGE_RESOLUTION_WIDTH:-320}"
IMAGE_RESOLUTION_HEIGHT="${IMAGE_RESOLUTION_HEIGHT:-160}"
NUM_FRAMES="${NUM_FRAMES:-33}"
ACTION_HORIZON="${ACTION_HORIZON:-24}"
NUM_VIEWS="${NUM_VIEWS:-3}"
MAX_CHUNK_SIZE="${MAX_CHUNK_SIZE:-4}"
DATASET_SHARD_SAMPLING_RATE="${DATASET_SHARD_SAMPLING_RATE:-0.1}"
DATASET_SHARD_SAMPLING_STRATEGY="${DATASET_SHARD_SAMPLING_STRATEGY:-random}"
DATASET_SHARD_SAMPLING_BLOCK_SIZE="${DATASET_SHARD_SAMPLING_BLOCK_SIZE:-64}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
DATALOADER_PIN_MEMORY="${DATALOADER_PIN_MEMORY:-false}"
DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-false}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-$((12000 + RANDOM % 20000))}"

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${NUM_GPUS:-${#GPU_IDS[@]}}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((NUM_GPUS * PER_DEVICE_BS * GRAD_ACCUM_STEPS))}"

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "ERROR: Python not found at ${PYTHON_BIN}"
    exit 1
fi

if [ ! -f "${EXPERIMENT_PY}" ]; then
    echo "ERROR: experiment.py not found at ${EXPERIMENT_PY}"
    exit 1
fi

if [ ! -d "${AGILEX_ALOHA_DATASET_ROOT}" ]; then
    echo "ERROR: dataset root not found at ${AGILEX_ALOHA_DATASET_ROOT}"
    exit 1
fi

if [ ! -f "${WAN22_CKPT_DIR}/models_t5_umt5-xxl-enc-bf16.pth" ]; then
    echo "ERROR: Wan2.2 text encoder missing under ${WAN22_CKPT_DIR}"
    exit 1
fi

if [ ! -f "${IMAGE_ENCODER_DIR}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" ]; then
    echo "ERROR: CLIP image encoder missing under ${IMAGE_ENCODER_DIR}"
    exit 1
fi

if [ ! -f "${WAN22_CKPT_DIR}/Wan2.2_VAE.pth" ]; then
    echo "ERROR: Wan2.2 VAE missing under ${WAN22_CKPT_DIR}"
    exit 1
fi

if [ ! -d "${TOKENIZER_DIR}" ]; then
    echo "ERROR: tokenizer path not found at ${TOKENIZER_DIR}"
    exit 1
fi

if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
    "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

import pyarrow.parquet as pq

root = Path(os.environ["AGILEX_ALOHA_DATASET_ROOT"])
dataset_roots = sorted(p for p in root.iterdir() if p.is_dir())
required_meta = [
    "info.json",
    "modality.json",
    "embodiment.json",
    "tasks.jsonl",
    "episodes.jsonl",
    "stats.json",
    "relative_stats_dreamzero.json",
]
video_keys = [
    "observation.images.cam_high",
    "observation.images.cam_left",
    "observation.images.cam_right",
]
expected_columns = {
    "observation.state",
    "action",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
}

fatal = []
warnings = []
total_episodes = 0
total_frames = 0
total_parquets = 0
total_mp4s = 0
min_len = None
max_len = 0

for dataset_path in dataset_roots:
    meta_dir = dataset_path / "meta"
    missing = [name for name in required_meta if not (meta_dir / name).exists()]
    if missing:
        fatal.append((dataset_path.name, "missing_meta", missing))
        continue

    try:
        info = json.load(open(meta_dir / "info.json"))
        modality = json.load(open(meta_dir / "modality.json"))
        embodiment = json.load(open(meta_dir / "embodiment.json"))
        stats = json.load(open(meta_dir / "stats.json"))
        relative_stats = json.load(open(meta_dir / "relative_stats_dreamzero.json"))
    except Exception as exc:
        fatal.append((dataset_path.name, "bad_meta_json", repr(exc)))
        continue

    if embodiment.get("embodiment_tag") != "aloha_x5lite_bimanual":
        fatal.append((dataset_path.name, "bad_embodiment", embodiment))
    if info.get("fps") != 30:
        fatal.append((dataset_path.name, "bad_fps", info.get("fps")))

    expected_modality = {
        "video": {"cam_high", "cam_left", "cam_right"},
        "state": {"left_joint_pos", "left_gripper_pos", "right_joint_pos", "right_gripper_pos"},
        "action": {"left_joint_pos", "left_gripper_pos", "right_joint_pos", "right_gripper_pos"},
    }
    for group, keys in expected_modality.items():
        got = set(modality.get(group, {}))
        if got != keys:
            fatal.append((dataset_path.name, f"bad_{group}_keys", sorted(got)))

    for key in ["observation.state", "action"]:
        if key not in stats:
            fatal.append((dataset_path.name, "missing_stats", key))
    for key in expected_modality["action"]:
        if key not in relative_stats:
            fatal.append((dataset_path.name, "missing_relative_stats", key))

    episodes = []
    with open(meta_dir / "episodes.jsonl") as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line))
    episode_ids = {int(ep["episode_index"]) for ep in episodes}
    episode_lengths = {int(ep["episode_index"]): int(ep["length"]) for ep in episodes}
    discarded_ids = set(int(i) for i in info.get("discarded_episode_indices", []))

    if len(episodes) != info.get("total_episodes"):
        fatal.append((dataset_path.name, "episode_count_mismatch", (len(episodes), info.get("total_episodes"))))
    if sum(episode_lengths.values()) != info.get("total_frames"):
        fatal.append((dataset_path.name, "frame_count_mismatch", (sum(episode_lengths.values()), info.get("total_frames"))))

    total_episodes += len(episodes)
    total_frames += sum(episode_lengths.values())
    if episode_lengths:
        values = list(episode_lengths.values())
        min_len = min(values) if min_len is None else min(min_len, min(values))
        max_len = max(max_len, max(values))

    parquets = sorted((dataset_path / "data").glob("chunk-*/*.parquet"))
    parquet_ids = {int(p.stem.split("_")[-1]) for p in parquets}
    total_parquets += len(parquets)
    missing_parquet_ids = episode_ids - parquet_ids
    missing_required_parquet_ids = missing_parquet_ids - discarded_ids
    extra_parquet_ids = parquet_ids - episode_ids
    if missing_required_parquet_ids or extra_parquet_ids:
        fatal.append(
            (
                dataset_path.name,
                "parquet_episode_id_mismatch",
                {
                    "missing": sorted(missing_required_parquet_ids)[:10],
                    "extra": sorted(extra_parquet_ids)[:10],
                },
            )
        )

    for video_key in video_keys:
        videos = sorted((dataset_path / "videos" / "chunk-000" / video_key).glob("*.mp4"))
        video_ids = {int(p.stem.split("_")[-1]) for p in videos}
        total_mp4s += len(videos)
        missing_videos = episode_ids - video_ids
        extra_videos = video_ids - episode_ids
        if missing_videos:
            fatal.append((dataset_path.name, f"missing_videos:{video_key}", sorted(missing_videos)[:10]))
        if extra_videos:
            warnings.append((dataset_path.name, f"extra_videos:{video_key}", sorted(extra_videos)[:10]))

    sample_checked = False
    for parquet_path in parquets:
        episode_id = int(parquet_path.stem.split("_")[-1])
        try:
            parquet_file = pq.ParquetFile(parquet_path)
            names = set(parquet_file.schema_arrow.names)
            if not expected_columns <= names:
                fatal.append((dataset_path.name, "parquet_missing_columns", parquet_path.name, sorted(expected_columns - names)))
                break
            expected_rows = episode_lengths.get(episode_id)
            if parquet_file.metadata.num_rows != expected_rows:
                fatal.append((dataset_path.name, "parquet_row_count_mismatch", parquet_path.name, parquet_file.metadata.num_rows, expected_rows))
                break
            if not sample_checked and episode_id not in discarded_ids:
                table = pq.read_table(parquet_path, columns=["observation.state", "action"])
                state = table.column("observation.state")[0].as_py()
                action = table.column("action")[0].as_py()
                if len(state) != 14 or len(action) != 14:
                    fatal.append((dataset_path.name, "bad_state_action_dim", parquet_path.name, len(state), len(action)))
                sample_checked = True
        except Exception as exc:
            if episode_id in discarded_ids:
                warnings.append((dataset_path.name, "discarded_unreadable_parquet", parquet_path.name, repr(exc)))
            else:
                fatal.append((dataset_path.name, "unreadable_parquet", parquet_path.name, repr(exc)))
                break

print(
    "Preflight dataset summary: "
    f"roots={len(dataset_roots)}, episodes={total_episodes}, frames={total_frames}, "
    f"parquets={total_parquets}, mp4s={total_mp4s}, min_len={min_len}, max_len={max_len}"
)

if warnings:
    print(f"Preflight warnings ({len(warnings)}):")
    for item in warnings[:20]:
        print("  WARN", item)
    if len(warnings) > 20:
        print(f"  ... {len(warnings) - 20} more warnings")

if fatal:
    print(f"Preflight fatal issues ({len(fatal)}):")
    for item in fatal[:50]:
        print("  FATAL", item)
    raise SystemExit(1)
PY
fi

if [ "${PREFLIGHT_ONLY:-0}" = "1" ]; then
    echo "PREFLIGHT_ONLY=1, skipping training launch."
    exit 0
fi

echo "Using AGILEX_ALOHA_DATA_GLOB=${AGILEX_ALOHA_DATA_GLOB}"
echo "Using WAN22_CKPT_DIR=${WAN22_CKPT_DIR}"
echo "Using OUTPUT_DIR=${OUTPUT_DIR}"
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Using PER_DEVICE_BS=${PER_DEVICE_BS}, GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS}, GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
echo "Using MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

mkdir -p "${OUTPUT_DIR}"
cd "${DREAMZERO_ROOT}"

TRAIN_OVERRIDES=(
    "report_to=${REPORT_TO}"
    "data=dreamzero/aloha_x5lite_bimanual_relative"
    "wandb_project=dreamzero"
    "train_architecture=full"
    "num_frames=${NUM_FRAMES}"
    "action_horizon=${ACTION_HORIZON}"
    "num_views=${NUM_VIEWS}"
    "model=dreamzero/vla"
    "model/dreamzero/action_head=wan_flow_matching_action_tf_wan22"
    "model/dreamzero/transform=dreamzero_cotrain"
    "num_frame_per_block=2"
    "num_action_per_block=24"
    "num_state_per_block=1"
    "seed=42"
    "training_args.learning_rate=1e-5"
    "training_args.deepspeed=groot/vla/configs/deepspeed/${DEEPSPEED_CFG}.json"
    "save_steps=${SAVE_STEPS}"
    "training_args.warmup_ratio=0.05"
    "output_dir=${OUTPUT_DIR}"
    "per_device_train_batch_size=${PER_DEVICE_BS}"
    "global_batch_size=${GLOBAL_BATCH_SIZE}"
    "max_steps=${MAX_STEPS}"
    "weight_decay=1e-5"
    "save_total_limit=${SAVE_TOTAL_LIMIT}"
    "upload_checkpoints=false"
    "bf16=true"
    "tf32=true"
    "eval_bf16=true"
    "dataloader_pin_memory=${DATALOADER_PIN_MEMORY}"
    "dataloader_num_workers=${DATALOADER_NUM_WORKERS}"
    "dataloader_persistent_workers=${DATALOADER_PERSISTENT_WORKERS}"
    "image_resolution_width=${IMAGE_RESOLUTION_WIDTH}"
    "image_resolution_height=${IMAGE_RESOLUTION_HEIGHT}"
    "save_lora_only=false"
    "max_chunk_size=${MAX_CHUNK_SIZE}"
    "dataset_shard_sampling_rate=${DATASET_SHARD_SAMPLING_RATE}"
    "save_strategy=${SAVE_STRATEGY}"
    "aloha_data_root=${AGILEX_ALOHA_DATA_GLOB}"
    "dit_version=${WAN22_CKPT_DIR}"
    "text_encoder_pretrained_path=${WAN22_CKPT_DIR}/models_t5_umt5-xxl-enc-bf16.pth"
    "image_encoder_pretrained_path=${IMAGE_ENCODER_DIR}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
    "vae_pretrained_path=${WAN22_CKPT_DIR}/Wan2.2_VAE.pth"
    "tokenizer_path=${TOKENIZER_DIR}"
    "++train_dataset.mixture_kwargs.shard_sampling_strategy=${DATASET_SHARD_SAMPLING_STRATEGY}"
    "++train_dataset.mixture_kwargs.shard_sampling_block_size=${DATASET_SHARD_SAMPLING_BLOCK_SIZE}"
)

if [ "${SKIP_COMPONENT_LOADING:-0}" = "1" ]; then
    TRAIN_OVERRIDES+=("++action_head_cfg.config.skip_component_loading=true")
fi

if [ "${DEFER_LORA_INJECTION:-0}" = "1" ]; then
    TRAIN_OVERRIDES+=("++action_head_cfg.config.defer_lora_injection=true")
fi

"${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node "${NUM_GPUS}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    "${EXPERIMENT_PY}" \
    "${TRAIN_OVERRIDES[@]}"
