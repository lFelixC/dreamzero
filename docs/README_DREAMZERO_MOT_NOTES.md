# DreamZero MoT 当前实现

这份文档只记录 `/data/dreamzero_mot` 当前保留的 MoT 主链路。历史调研、规划草案、small split 实验和 ablation 操作说明已经移除，避免和生产路径混在一起。

## 架构概览

DreamZero MoT 是一个 video/action 双 expert 架构：

- `architecture=joint` 保留原始共享 `CausalWanModel` baseline。
- `architecture=mot` 使用 `MoTCausalWanModel`，保留 Wan video expert，并新增独立 action expert。
- video expert 每层输出 video K/V，action expert 用自己的 Q 和 state/action K/V 做 mixed attention。
- state/action token 由独立 action expert 编码和解码，顺序为 `[state_tokens | action_tokens]`，不再复用 video expert 的 FFN、residual path 或 joint action/state 模块。

核心实现：

- `groot/vla/model/dreamzero/modules/dreamzero_mot.py`
- `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py`
- `groot/vla/configs/model/dreamzero/action_head/wan_flow_matching_action_tf_wan22_mot.yaml`

## 保留开关

MoT 主链路保留必要结构参数和一个 KI 梯度路由开关：
MoT 推理额外保留 video denoise 模式开关，用于在只需要 action 的场景加速。

- `architecture`: `joint | mot`
- `mot_action_hidden_dim`
- `mot_action_ffn_dim`
- `mot_action_num_layers`
- `mot_action_num_heads`
- `mot_action_video_attention`: `first_frame | full_video | none`
- `mot_action_video_ki`: `true | false`
- `mot_inference_video_mode`: `auto | denoise | cache_only | decoupled_denoise`
- `mot_decouple_video_action_noise`: `true | false`
- `mot_video_noise_beta_alpha`
- `mot_video_noise_beta_beta`
- `mot_decoupled_inference_video_final_noise`
- `mot_decoupled_inference_video_refresh_steps`

`mot_action_video_attention` 含义：

- `first_frame`: action expert 只看首帧 video K/V，当前默认生产设置。
- `full_video`: action expert 看当前 video block 的完整 K/V。
- `none`: action expert 不看 video K/V，只走 action/state 自身 token。

`causal` alias 已删除；如需原先等价行为，请使用 `full_video`。

`mot_action_video_ki` 含义：

- `false`: action expert 可读取 video K/V，但 action loss 不通过 video K/V 回传到 video expert，当前默认。
- `true`: action loss 保留到 video K/V 的梯度，用于让 action supervision 共同更新 video expert。

`mot_inference_video_mode` 含义：

- `auto`: 当前默认。`first_frame` 和 `none` 推理时只用 cached video K/V 做 action denoise；`full_video` 保持 video/action 一起 denoise。
- `denoise`: 保持原始推理行为，video/action 每个采样步一起 denoise。
- `cache_only`: 强制不做 future video denoise，只用已建立的 video K/V cache 做 action conditioning。RTC guidance 当前会自动退回 `denoise`。
- `decoupled_denoise`: MoT full-video decoupled 专用推理路径。video 只按 refresh mask 跳步刷新，action 每个采样步都重新 denoise。

如果训练使用 `mot_action_video_attention=full_video`，但推理只想用 cached video K/V 加速，可以显式设置 `mot_inference_video_mode=cache_only` 或环境变量 `MOT_INFERENCE_VIDEO_MODE=cache_only`。

`mot_decouple_video_action_noise=true` 只支持 `architecture=mot` 且 `mot_action_video_attention=full_video`。训练时 video timestep 使用 `Beta(mot_video_noise_beta_alpha, mot_video_noise_beta_beta)` 偏向高噪声，action timestep 独立 Uniform 采样。`mot_inference_video_mode=auto` 会自动选择 `decoupled_denoise`。

`droid_random_drop_exterior_view_prob` 是 DROID 数据增强开关，默认 `0.0`。设置为 `0.5` 时，50% 训练样本会随机把 left/right exterior 其中一个置黑；设置为 `1.0` 时，每个训练样本都 drop 一个 exterior view。该增强只在训练态 DROID 三视角拼图时生效，不 drop wrist view。

## 固定主链路

以下实验开关已从配置入口移除，并固定为当前生产默认：

- action shared context: 关闭。
- video state context: 开启。
- action expert gate 初始化: AdaLN-zero 默认初始化。
- 训练 noise/timestep: 默认 video/action 使用标准耦合采样；仅当显式开启 `mot_decouple_video_action_noise` 时进入 MoT full-video decoupled 采样。
- 推理模式: 默认 `auto`，`first_frame`/`none` 可跳过 future video denoise，`full_video` 默认保持 video/action 一起 denoise，full-video decoupled checkpoint 默认使用 `decoupled_denoise`。

当前 MoT 主链路仍不恢复旧的 action-only cache refresh/no-denoise 诊断组合；新增的 decoupled 路径只服务 full-video 高噪声训练和对应的跳步 video 推理。

## 训练入口

保留的 MoT 训练脚本：

```bash
bash scripts/train/droid_wan22_mot_full.sh
```

常用可配环境变量仍保留：

- `GPU_IDS`
- `NUM_GPUS`
- `PER_DEVICE_BS`
- `GLOBAL_BATCH_SIZE`
- `MAX_STEPS`
- `SAVE_STEPS`
- `DEEPSPEED_CFG`
- `DATALOADER_NUM_WORKERS`
- `DATALOADER_PREFETCH_FACTOR`
- `DATALOADER_PERSISTENT_WORKERS`
- `USE_GRADIENT_CHECKPOINTING`
- `MOT_ACTION_VIDEO_ATTENTION`
- `MOT_ACTION_VIDEO_KI`，也可用短别名 `MOT_KI`
- `MOT_INFERENCE_VIDEO_MODE`
- `MOT_DECOUPLE_VIDEO_ACTION_NOISE`
- `MOT_VIDEO_NOISE_BETA_ALPHA`
- `MOT_VIDEO_NOISE_BETA_BETA`
- `MOT_DECOUPLED_INFERENCE_VIDEO_FINAL_NOISE`
- `MOT_DECOUPLED_INFERENCE_VIDEO_REFRESH_STEPS`
- `DROID_RANDOM_DROP_EXTERIOR_VIEW_PROB`

非 MoT 训练脚本仍保留；MoT small、joint small 和 MoT ablation 脚本已删除。

## 推理行为

普通推理和 RTC 兼容路径保留。MoT 下 `mot_inference_video_mode=auto` 会在 `first_frame`/`none` 下跳过 future video denoise，只做 action denoise；`full_video` 默认仍然 video/action 同步 denoise；`mot_decouple_video_action_noise=true` 的 full-video checkpoint 会自动使用 `decoupled_denoise`。需要强制行为时可在训练或 checkpoint 配置中设置 `mot_inference_video_mode=denoise|cache_only|decoupled_denoise`，也可在推理进程里用环境变量 `MOT_INFERENCE_VIDEO_MODE` 临时覆盖。`decoupled_denoise` 暂不支持 RTC guidance。

## 两卡推理命令

`socket_test_optimized_AR.py` 会从 checkpoint 的 `config.json` 读取 `architecture` 和 `mot_action_video_attention`。因此 first-frame 与 full-video 需要使用对应配置的 checkpoint，或使用只改 `config.json` 的 checkpoint view。

本机已验证的 smoke checkpoint：

- `first_frame`: `/data/checkpoints/dreamzero/dreamzero_droid_wan22_mot_smoke_first/checkpoint-1`
- `full_video`: `/data/checkpoints/dreamzero/dreamzero_droid_wan22_mot_smoke_full_config/checkpoint-1`

first-frame 两卡 server：

```bash
cd /data/dreamzero_mot
CUDA_VISIBLE_DEVICES=0,1 \
PYTHONPATH=/data/dreamzero_mot:${PYTHONPATH:-} \
TORCH_COMPILE_BACKEND= \
/data/dreamzero/.venv/bin/python -m torch.distributed.run \
  --nproc_per_node 2 \
  --standalone \
  socket_test_optimized_AR.py \
  --model-path /data/checkpoints/dreamzero/dreamzero_droid_wan22_mot_smoke_first/checkpoint-1 \
  --architecture mot \
  --host 127.0.0.1 \
  --port 8120 \
  --index 9200 \
  --max-chunk-size 1 \
  --timeout-seconds 900 \
  --handshake-timeout-seconds 0
```

full-video 两卡 server：

```bash
cd /data/dreamzero_mot
CUDA_VISIBLE_DEVICES=0,1 \
PYTHONPATH=/data/dreamzero_mot:${PYTHONPATH:-} \
TORCH_COMPILE_BACKEND= \
/data/dreamzero/.venv/bin/python -m torch.distributed.run \
  --nproc_per_node 2 \
  --standalone \
  socket_test_optimized_AR.py \
  --model-path /data/checkpoints/dreamzero/dreamzero_droid_wan22_mot_smoke_full_config/checkpoint-1 \
  --architecture mot \
  --host 127.0.0.1 \
  --port 8121 \
  --index 9201 \
  --max-chunk-size 1 \
  --timeout-seconds 900 \
  --handshake-timeout-seconds 0
```

实机客户端，在机器人环境的另一个终端执行：

```bash
cd /data/dreamzero_mot
PYTHONPATH=/data/dreamzero_mot:${PYTHONPATH:-} \
/data/dreamzero/.venv/bin/python example/remote_infer/main_dreamzero.py \
  --remote-host 127.0.0.1 \
  --remote-port 8120 \
  --open-loop-horizon 8 \
  --control-frequency 15 \
  --missing-right-camera-strategy duplicate_left \
  --log-timing \
  --video-output-dir /data/checkpoints/dreamzero/dreamzero_mot_remote_client_videos \
  --results-dir /data/checkpoints/dreamzero/dreamzero_mot_remote_client_results
```

测 full-video 时把客户端端口改成 `8121`。如果 server 在远端机器上，推荐先做 SSH 本地端口转发，然后保持 `--remote-host 127.0.0.1`，把 `--remote-port` 设为本地转发端口。

无机器人依赖的 smoke 测试，在另一个终端执行：

```bash
cd /data/dreamzero_mot
PYTHONPATH=/data/dreamzero_mot:${PYTHONPATH:-} \
/data/dreamzero/.venv/bin/python test_client_AR.py \
  --host 127.0.0.1 \
  --port 8120 \
  --num-chunks 1 \
  --use-zero-images \
  --prompt "pick up the object"
```

测 full-video 时把 smoke 客户端端口改成 `8121`。本次验证中，两个 server 都成功返回了 shape 为 `(1, 8)` 的 action。

## Full-Video Decoupled Smoke

本机已验证的 full-video decoupled smoke checkpoint：

- `full_video_decoupled`: `/data/checkpoints/dreamzero/dreamzero_droid_wan22_mot_decoupled_smoke/checkpoint-1`

最小训练并保存 checkpoint：

```bash
cd /data/dreamzero_mot

CUDA_VISIBLE_DEVICES=0,1 \
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
USE_GRADIENT_CHECKPOINTING=true \
MOT_ACTION_VIDEO_ATTENTION=full_video \
MOT_DECOUPLE_VIDEO_ACTION_NOISE=true \
MOT_VIDEO_NOISE_BETA_ALPHA=3.0 \
MOT_VIDEO_NOISE_BETA_BETA=1.0 \
MOT_DECOUPLED_INFERENCE_VIDEO_FINAL_NOISE=0.8 \
MOT_DECOUPLED_INFERENCE_VIDEO_REFRESH_STEPS=8 \
OUTPUT_DIR=/data/checkpoints/dreamzero/dreamzero_droid_wan22_mot_decoupled_smoke \
WANDB_MODE=offline \
SWANLAB_SYNC_WANDB=0 \
bash scripts/train/droid_wan22_mot_full.sh
```

训练成功时日志里应看到：

```text
[NOISE] Mode=MOT_DECOUPLED | Video: Beta(3.0,1.0) ... | Action: INDEPENDENT Uniform ...
```

decoupled 两卡 server：

```bash
cd /data/dreamzero_mot

CUDA_VISIBLE_DEVICES=0,1 \
MOT_INFERENCE_VIDEO_MODE=decoupled_denoise \
PYTHONPATH=/data/dreamzero_mot:${PYTHONPATH:-} \
TORCH_COMPILE_BACKEND= \
DISABLE_TORCH_COMPILE=true \
/data/dreamzero/.venv/bin/python -m torch.distributed.run \
  --nproc_per_node 2 \
  --standalone \
  socket_test_optimized_AR.py \
  --model-path /data/checkpoints/dreamzero/dreamzero_droid_wan22_mot_decoupled_smoke/checkpoint-1 \
  --architecture auto \
  --host 127.0.0.1 \
  --port 8130 \
  --index 9300 \
  --max-chunk-size 1 \
  --timeout-seconds 900 \
  --handshake-timeout-seconds 0
```

无机器人依赖的 smoke client：

```bash
cd /data/dreamzero_mot

PYTHONPATH=/data/dreamzero_mot:${PYTHONPATH:-} \
/data/dreamzero/.venv/bin/python test_client_AR.py \
  --host 127.0.0.1 \
  --port 8130 \
  --num-chunks 1 \
  --use-zero-images \
  --prompt "pick up the object"
```

推理成功时 server 日志里应看到：

```text
[MoT] inference_video_mode=decoupled_denoise (configured=decoupled_denoise, action_video_attention=full_video)
[MoT] decoupled_denoise: video_final_noise=0.800, video_refresh_steps=8/16, action_steps=every_step
[MoT] decoupled_denoise compute: video_refresh_steps=8, action_steps=16
```

本次 smoke client 返回了 shape 为 `(1, 8)` 的 action。

如果需要做新的诊断实验，建议新建临时实验分支或独立脚本，不再把诊断开关接回生产 action head 配置。
