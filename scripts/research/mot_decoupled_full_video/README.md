# MoT Decoupled Full-Video 1-Step

这组脚本只做薄封装，新的 checkpoint 仍然复用现有入口：

- server: `/data/dreamzero/socket_test_robolab_AR.py`（RoboArena/RoboLab 协议；真机 DROID client 用 `WebsocketClientPolicy` 读 `PolicyServerConfig`，必须用这个入口而不是原生 `socket_test_optimized_AR.py`）
- client: `/data/dreamzero/droid-client/scripts/main_dreamzero_test.py`

对齐契约：

- 训练：MoT action expert 使用 `full_video` K/V。
- 训练：video 固定到 one-step high-noise timestep，action 使用 independent uniform timestep。
- 推理：`decoupled_denoise`，video 只 refresh 1 step，action 每个 scheduler step 都用 refreshed full-video K/V denoise。
- websocket：DROID/AR，两路 exterior camera、一路 wrist camera、7D joint + 1D gripper。

## 1. Train

```bash
bash scripts/research/mot_decoupled_full_video/train.sh
```

训练 wrapper 固定的关键默认值：

- `OUTPUT_DIR=/data/checkpoints/dreamzero/dreamzero_droid_wan22_mot_decoupled_full_video`
- `MOT_ACTION_VIDEO_ATTENTION=full_video`
- `MOT_DECOUPLE_VIDEO_ACTION_NOISE=true`
- `MOT_INFERENCE_VIDEO_MODE=auto`
- `MOT_DECOUPLED_INFERENCE_VIDEO_REFRESH_STEPS=1`

最小 1-step 调试训练：

```bash
MAX_STEPS=1 SAVE_STEPS=1 EVAL_STEPS=1 \
PER_DEVICE_BS=1 GLOBAL_BATCH_SIZE=2 NUM_GPUS=2 GPU_IDS=0,1 CUDA_VISIBLE_DEVICES=0,1 \
DATALOADER_NUM_WORKERS=1 DATALOADER_PREFETCH_FACTOR=1 DATALOADER_PERSISTENT_WORKERS=false \
bash scripts/research/mot_decoupled_full_video/train.sh
```

The expected training log contains:

```text
[NOISE] Mode=MOT_DECOUPLED | Video: fixed one-step ... | Action: INDEPENDENT Uniform ...
```

## 2. Server

推荐直接复用现有 server 入口：

```bash
cd /data/dreamzero

CUDA_VISIBLE_DEVICES=0,1 \
MOT_INFERENCE_VIDEO_MODE=decoupled_denoise \
MOT_DECOUPLED_INFERENCE_VIDEO_REFRESH_STEPS=1 \
NUM_DIT_STEPS=16 \
DISABLE_TORCH_COMPILE=true \
PYTHONPATH=/data/dreamzero:${PYTHONPATH:-} \
/data/dreamzero/.venv/bin/python -m torch.distributed.run \
  --nproc_per_node 2 \
  --standalone \
  socket_test_robolab_AR.py \
  --model-path <NEW_CHECKPOINT_PATH> \
  --architecture auto \
  --host 0.0.0.0 \
  --port 8130 \
  --handshake-timeout-seconds 0
```

等价一键 wrapper：

```bash
MODEL_PATH=/data/checkpoints/dreamzero/dreamzero_droid_wan22_mot_decoupled_full_video/checkpoint-30000 \
bash scripts/research/mot_decoupled_full_video/serve_ar.sh
```

如果不传 `MODEL_PATH`，wrapper 会从 `OUTPUT_DIR` 下自动选最新的 `checkpoint-*`。

Expected server log:

```text
[MoT] inference_video_mode=decoupled_denoise ...
[MoT] decoupled_denoise: video_final_noise=0.800, video_refresh_steps=1/1, action_steps=every_step
[MoT] decoupled_denoise compute: video_refresh_steps=1, action_steps=16
```

`decoupled_denoise` is cache-order-sensitive and does not support RTC guidance.
The AR server advertises this in metadata so the real client disables RTC and
async prefetch.

## 3. Client

Run this on the DROID robot laptop environment:

```bash
REMOTE_HOST=<server_host> REMOTE_PORT=8130 \
bash scripts/research/mot_decoupled_full_video/client_droid_real.sh
```

推荐直接复用现有 DROID client 入口：

```bash
PYTHONPATH=/data/dreamzero:/data/dreamzero/droid-client:/data/dreamzero/droid-client/scripts:${PYTHONPATH:-} \
/data/dreamzero/.venv/bin/python /data/dreamzero/droid-client/scripts/main_dreamzero_test.py \
  --remote-host <server_host> \
  --remote-port 8130 \
  --open-loop-horizon 24 \
  --control-frequency 15
```

等价一键 wrapper：

```bash
REMOTE_HOST=<server_host> REMOTE_PORT=8130 \
bash scripts/research/mot_decoupled_full_video/client_droid_real.sh
```

That client keeps a 25-frame history and sends DROID-style sampled frames
`[0, 3, 6, ..., 24]` for each steady request, matching the training block
duration.

Do not enable `--use-rtc` or `--enable-async-prefetch` for this decoupled
checkpoint. `decoupled_denoise` needs in-order causal cache updates, and the
server advertises this as `cache_order_sensitive=true`.

Camera defaults in `main_dreamzero_test.py` are preserved:

- `right_camera_id=36517165`
- `left_camera_id=None`
- `missing_left_camera_strategy=mask_left`
- `wrist_camera_id=13337231`

Override them through the wrapper environment variables only if the robot
camera IDs differ.
