# RoboTwin Wan2.2 MoT Decoupled Full-Video Training

This document records the local training and resume commands for:

```text
scripts/train/robotwin_wan22_mot_decoupled_full_video.sh
```

Default experiment output:

```text
/data/checkpoints/dreamzero/dreamzero_robotwin_wan22_mot_decoupled_full_video
```

## 1. Preconditions

Run from the DreamZero repo:

```bash
cd /data/dreamzero
```

Expected local assets:

```text
/data/datasets/dreamzero/robotwin_unified_dreamzero_180x320/meta/modality.json
/data/checkpoints/dreamzero/Wan2.2-TI2V-5B
```

The launcher uses `/data/dreamzero/.venv/bin/python` by default and runs 8 GPUs
unless overridden.

## 2. Start Or Resume Training (tmux)

The training framework automatically checks `OUTPUT_DIR`:

- If `OUTPUT_DIR/config.json` exists, it treats the final model as ready and skips training.
- Otherwise it resumes from the largest `checkpoint-*` under `OUTPUT_DIR`.
- If there is no checkpoint, it starts a new run.

Use tmux so the run survives SSH disconnects:

```bash
tmux new -d -s dz_robotwin_wan22_mot_resume "bash -lc 'set -o pipefail; cd /data/dreamzero && GPU_IDS=0,1,2,3,4,5,6,7 CHECKPOINT_ROOT=/data/checkpoints/dreamzero OUTPUT_DIR=/data/checkpoints/dreamzero/dreamzero_robotwin_wan22_mot_decoupled_full_video ROBOTWIN_DATA_ROOT=/data/datasets/dreamzero/robotwin_unified_dreamzero_180x320 bash scripts/train/robotwin_wan22_mot_decoupled_full_video.sh 2>&1 | tee -a /data/checkpoints/dreamzero/dreamzero_robotwin_wan22_mot_decoupled_full_video/resume_$(date +%Y%m%d_%H%M%S).log'"
```

Attach to session:

```bash
tmux attach -t dz_robotwin_wan22_mot_resume
```

List sessions:

```bash
tmux ls
```

Expected resume log when checkpoints already exist:

```text
Resuming training from /data/checkpoints/dreamzero/dreamzero_robotwin_wan22_mot_decoupled_full_video/checkpoint-5000
```

The exact checkpoint number changes as training progresses.

If another training job is using the same distributed port, override it (also in tmux):

```bash
tmux new -d -s dz_robotwin_wan22_mot_resume_p29637 "bash -lc 'set -o pipefail; cd /data/dreamzero && MASTER_PORT=29637 GPU_IDS=0,1,2,3,4,5,6,7 CHECKPOINT_ROOT=/data/checkpoints/dreamzero OUTPUT_DIR=/data/checkpoints/dreamzero/dreamzero_robotwin_wan22_mot_decoupled_full_video ROBOTWIN_DATA_ROOT=/data/datasets/dreamzero/robotwin_unified_dreamzero_180x320 bash scripts/train/robotwin_wan22_mot_decoupled_full_video.sh 2>&1 | tee -a /data/checkpoints/dreamzero/dreamzero_robotwin_wan22_mot_decoupled_full_video/resume_$(date +%Y%m%d_%H%M%S).log'"
```

## 3. Important Defaults

The script passes these important Hydra overrides and environment defaults:

```bash
architecture=mot
MOT_ACTION_VIDEO_ATTENTION=full_video
MOT_DECOUPLE_VIDEO_ACTION_NOISE=true
MOT_INFERENCE_VIDEO_MODE=decoupled_denoise
STATIC_VIDEO_INIT=true
STATIC_VIDEO_INIT_NOISE=0.0

NUM_FRAMES=17
ACTION_HORIZON=24
MAX_CHUNK_SIZE=2
NUM_FRAME_PER_BLOCK=2
NUM_ACTION_PER_BLOCK=24
NUM_STATE_PER_BLOCK=1

IMAGE_RESOLUTION_HEIGHT=160
IMAGE_RESOLUTION_WIDTH=320
MODEL_TARGET_HEIGHT=320
MODEL_TARGET_WIDTH=640
MODEL_FRAME_SEQLEN=200

PER_DEVICE_BS=64
GLOBAL_BATCH_SIZE=512
MAX_STEPS=60000
SAVE_STEPS=1000
SAVE_TOTAL_LIMIT=10
DEEPSPEED_CFG=zero2
LEARNING_RATE=4e-5
```

These values preserve the DreamZero block alignment:

```text
one observed video block <-> one action chunk <-> one state token group
```

Do not change `NUM_FRAMES`, `ACTION_HORIZON`, `NUM_FRAME_PER_BLOCK`,
`NUM_ACTION_PER_BLOCK`, `NUM_STATE_PER_BLOCK`, image resolution, or
`MODEL_FRAME_SEQLEN` casually. Those values must stay aligned with the VAE
latent token layout and the action/state register layout.

## 4. Smoke / Debug Run (唯一直接运行训练命令)

Use this direct, non-tmux command only for a quick launcher test. Point
`OUTPUT_DIR` at a separate smoke directory so it does not pollute the real run.
For mature training or resume, use the tmux commands above.

```bash
cd /data/dreamzero

GPU_IDS=0,1 \
NUM_GPUS=2 \
PER_DEVICE_BS=1 \
GLOBAL_BATCH_SIZE=2 \
MAX_STEPS=1 \
SAVE_STEPS=1 \
EVAL_STEPS=1 \
DATALOADER_NUM_WORKERS=1 \
DATALOADER_PREFETCH_FACTOR=1 \
DATALOADER_PERSISTENT_WORKERS=false \
SWANLAB_SYNC_WANDB=0 \
WANDB_MODE=offline \
OUTPUT_DIR=/data/checkpoints/dreamzero/dreamzero_robotwin_wan22_mot_decoupled_full_video_smoke \
ROBOTWIN_DATA_ROOT=/data/datasets/dreamzero/robotwin_unified_dreamzero_180x320 \
bash scripts/train/robotwin_wan22_mot_decoupled_full_video.sh
```

## 5. Useful Checks

Find the latest checkpoint:

```bash
find /data/checkpoints/dreamzero/dreamzero_robotwin_wan22_mot_decoupled_full_video \
  -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1
```

Check whether training will skip because a final model exists:

```bash
ls -la /data/checkpoints/dreamzero/dreamzero_robotwin_wan22_mot_decoupled_full_video/config.json
```

Check current training state:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("/data/checkpoints/dreamzero/dreamzero_robotwin_wan22_mot_decoupled_full_video")
ckpts = sorted(root.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
latest = ckpts[-1]
state = json.loads((latest / "trainer_state.json").read_text())
print(latest)
print("global_step =", state.get("global_step"))
PY
```

Check GPU usage:

```bash
nvidia-smi
```

## 6. Logging Notes

The script defaults to:

```bash
WANDB_MODE=offline
SWANLAB_SYNC_WANDB=1
WANDB_PROJECT=dreamzero
```

Training resume is controlled by checkpoint discovery in `OUTPUT_DIR`, not by
the wandb or SwanLab run id. If you need wandb to resume into an existing run,
set `WANDB_RUN_ID` explicitly before launching.
