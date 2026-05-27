# DROID MoT CBS Restart And NSYS Notes

This note records the current DROID MoT critical-batch-size setup and the commands to restart either the full pipeline or short profiling runs.

## Current Training Shape

- Model: DreamZero MoT WAM on DROID, Wan2.2 backbone.
- GPUs: 8 local GPUs.
- Warmup: global batch 128, per-device batch 16, grad accumulation 1.
- Branch sweep target global batches: 128, 256, 512, 1024.
- Branch per-device batch cap: 32.
- Branch effective batches:
  - global 128: per-device 16, grad accumulation 1.
  - global 256: per-device 32, grad accumulation 1.
  - global 512: per-device 32, grad accumulation 2.
  - global 1024: per-device 32, grad accumulation 4.
- LR rule: AdamW sqrt scaling from base batch 128 and base LR 1.5e-5.
- Branch scheduler: constant LR, warmup disabled.
- MoT action loss weight: 2.0.
- Gradient checkpointing: enabled.
- Loss-gradient conflict logging: disabled. Do not enable it for CBS timing or memory tests.

## Full Pipeline In Tmux

Start the whole workflow from scratch:

```bash
cd /data/dreamzero
RUN_ID="cbs_mot_droid_$(date -u +%Y%m%d_%H%M%S)_tmux_full"
tmux new-session -d -s cbs_mot_droid_pipeline \
  "cd /data/dreamzero && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CBS_RUN_ID=${RUN_ID} bash scripts/train/run_droid_mot_cbs_pipeline.sh"
```

Attach or detach:

```bash
tmux attach -t cbs_mot_droid_pipeline
# detach: Ctrl-b d
```

The pipeline log is written under:

```text
/data/checkpoints/dreamzero/cbs_mot_droid/<RUN_ID>/logs/
```

The pipeline runs warmup first, then the four branches sequentially. It skips warmup if the expected `checkpoint-5000` already exists.

## Restart From An Existing Warmup Checkpoint

Use this after warmup has completed:

```bash
cd /data/dreamzero
export WARMUP_CKPT=/data/checkpoints/dreamzero/cbs_mot_droid/<RUN_ID>/warmup_b128_lr1.5e-5_steps5000/checkpoint-5000

tmux new-session -d -s cbs_mot_droid_sweep \
  "cd /data/dreamzero && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WARMUP_CKPT=${WARMUP_CKPT} bash scripts/train/run_droid_mot_cbs_sweep.sh"
```

To resume only a subset:

```bash
CBS_BATCHES="512 1024" WARMUP_CKPT="${WARMUP_CKPT}" bash scripts/train/run_droid_mot_cbs_sweep.sh
```

## Short NSYS Warmup Profile

Stop any active training first, then profile a short warmup run. Keep the run short; profiling all ranks produces large traces.

```bash
cd /data/dreamzero
mkdir -p /data/checkpoints/dreamzero/nsys

RUN_ID="nsys_mot_droid_$(date -u +%Y%m%d_%H%M%S)"
nsys profile \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt,cublas,cudnn \
  --sample=none \
  --cpuctxsw=none \
  --output="/data/checkpoints/dreamzero/nsys/${RUN_ID}" \
  bash -lc "cd /data/dreamzero && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CBS_RUN_ID=${RUN_ID} MAX_STEPS=30 SAVE_STEPS=30 PER_DEVICE_BS=16 USE_GRADIENT_CHECKPOINTING=true bash scripts/train/run_droid_mot_cbs_warmup.sh"
```

For a smaller trace, lower `MAX_STEPS` after the first successful profiling run. The first steps include startup and cache effects, so use enough steps to see steady training.

## Expected Durations

Approximate timings from observed H200 runs:

- Warmup, 5000 steps, per-device batch 16: about 16-17 hours.
- Branch 128, 4096 steps, per-device batch 16: about 13-14 hours.
- Branch 256, 2048 steps, per-device batch 32: about 9-11 hours.
- Branch 512, 1024 steps, per-device batch 32, grad accumulation 2: about 9-11 hours.
- Branch 1024, 512 steps, per-device batch 32, grad accumulation 4: about 9-11 hours.

Total full pipeline estimate: about 67-73 hours, plus checkpoint save overhead.
