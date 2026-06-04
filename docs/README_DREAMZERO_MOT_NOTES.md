# DreamZero MoT 当前实现

这份文档只记录 `/data/dreamzero_mot` 当前保留的 MoT 主链路。历史调研、规划草案、small split 实验和 ablation 操作说明已经移除，避免和生产路径混在一起。

## 架构概览

DreamZero MoT 是一个 video/action 双 expert 架构：

- `architecture=joint` 保留原始共享 `CausalWanModel` baseline。
- `architecture=mot` 使用 `MoTCausalWanModel`，保留 Wan video expert，并新增独立 action expert。
- video/action expert 每层分别构造 Q/K/V，`full_video` 下拼接后按 block-causal mask 做双向 mixed attention，并在 mixed attention 后共用同一次 Wan text/image cross-attn；`full_video_unidirectional` 下 action 读完整 block-causal video，video 不读 action/state register。
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
- `mot_action_video_attention`: `first_frame | full_video | full_video_unidirectional | none`
- `mot_action_video_ki`: `true | false`
- `mot_inference_video_mode`: `auto | denoise | cache_only | decoupled_denoise`
- `mot_decouple_video_action_noise`: `true | false`
- `mot_video_noise_beta_alpha`
- `mot_video_noise_beta_beta`
- `mot_decoupled_inference_video_final_noise`
- `mot_decoupled_inference_video_refresh_steps`
- `activation_checkpointing_policy`: `off | mixed | both`

`mot_action_video_attention` 含义：

- `full_video`: video/action 按 block-causal 可见性双向融合。video block 与 action block 只能看当前/过去 video，以及当前 block 的 state/action register，不能看未来 block。
- `full_video_unidirectional`: action expert 按 block-causal 可见性看完整 video K/V 和当前 block 的 state/action register；video 不读 action/state register。
- `first_frame`: action expert 只看首帧 video K/V，用于复现实验或降低推理成本；video 不读 action/state register。
- `none`: action expert 不看 video K/V，只走 action/state 自身 token；video 不读 action/state register。

`causal` alias 已删除；如需双向行为请使用 `full_video`，如需 action 单向看完整 video 请使用 `full_video_unidirectional`。

`mot_action_video_ki` 含义：

- `false`: 使用单次 joint mixed attention，action loss 可经 video K/V 回传到 video expert，当前默认。
- `true`: action 分支读取 detached video K/V，保留 video/action 梯度独立的兼容路径；`full_video` 下 video 分支仍按 mask 读取当前 block 的 action/state register。

`mot_inference_video_mode` 含义：

- `auto`: 当前默认。`first_frame`、`none` 和 `full_video_unidirectional` 推理时可用 cached video K/V 做 action denoise；`full_video` 因为 video 也读取 action，保持 video/action 一起 denoise。
- `denoise`: 保持原始推理行为，video/action 每个采样步一起 denoise。
- `cache_only`: 强制不做 future video denoise，只用已建立的 video K/V cache 做 action conditioning。适用于 `first_frame`/`none`/`full_video_unidirectional`；RTC guidance 当前会自动退回 `denoise`。
- `decoupled_denoise`: 只适用于 `full_video_unidirectional`；双向 `full_video` 会拒绝该模式。

如果训练使用双向 `mot_action_video_attention=full_video`，推理不能切到 `cache_only` 或 `decoupled_denoise`。这两个模式会让 video cache 跳过当前 action token，造成训练/推理不一致；代码会直接报错。

`mot_decouple_video_action_noise=true` 支持 `architecture=mot` 且 `mot_action_video_attention=full_video|full_video_unidirectional`。训练时 video timestep 使用 `Beta(mot_video_noise_beta_alpha, mot_video_noise_beta_beta)` 偏向高噪声，action timestep 独立 Uniform 采样。双向 `full_video` 下 `auto` 会选择 `denoise`；`full_video_unidirectional` 下 `auto` 会选择 `decoupled_denoise`。

`droid_random_drop_exterior_view_prob` 是 DROID 数据增强开关，默认 `0.0`。设置为 `0.5` 时，50% 训练样本会随机把 left/right exterior 其中一个置黑；设置为 `1.0` 时，每个训练样本都 drop 一个 exterior view。该增强只在训练态 DROID 三视角拼图时生效，不 drop wrist view。

`activation_checkpointing_policy` 是 MoT 固定 Selective Activation Checkpointing 开关，默认 `off`，且只有在 `USE_GRADIENT_CHECKPOINTING=true` 时可用：

- `off`: 保持普通 block-level gradient checkpointing 行为。
- `mixed`: 在 checkpoint 内保存 MoT `_scaled_dot_product_mixed_attention()` 的 attention 输出，减少 backward 重算 mixed video/action attention。
- `both`: 在 `mixed` 基础上，额外保存当前路径中的 video self-attention 输出。该模式显存增量明显更高，建议先小 batch smoke 再扩大训练。

默认 `full_video` MoT 主路径主要命中 mixed attention；`both` 只有在当前训练路径存在独立 video self-attention 时才会比 `mixed` 多保存。排查 SAC 是否实际命中时可临时设置 `DREAMZERO_SAC_DEBUG=1`，需要查看底层 op 路径时可设置 `DREAMZERO_SAC_TRACE=1`。

## 固定主链路

以下实验开关已从配置入口移除，并固定为当前生产默认：

- `full_video` text cross-attn: video/action 共用 Wan cross-attn，action token 通过 projection bridge 进入 video hidden space 后读取 text/image context；非 `full_video` action direct text context 仍关闭。
- video cross-attn state context: 关闭；state 只通过 action expert 的 state tokens 进入 mixed attention。
- action expert gate 初始化: AdaLN-zero 默认初始化。
- 训练 noise/timestep: 默认 video/action 使用标准耦合采样；仅当显式开启 `mot_decouple_video_action_noise` 时进入 MoT 完整 video-action 模式的独立 timestep 采样。
- 推理模式: 默认 `auto`，双向 `full_video` 默认保持 video/action 一起 denoise；`first_frame`/`none`/`full_video_unidirectional` 可跳过 future video denoise。

当前 MoT 主链路仍不恢复旧的 action-only cache refresh/no-denoise 诊断组合；双向 `full_video` 必须让 video/action 在每个采样步使用同一组当前 token。

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
- `ACTIVATION_CHECKPOINTING_POLICY`
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

普通推理和 RTC 兼容路径保留。MoT 下 `mot_inference_video_mode=auto` 会在 `first_frame`/`none`/`full_video_unidirectional` 下跳过 future video denoise，只做 action denoise；双向 `full_video` 默认仍然 video/action 同步 denoise。需要强制行为时可在训练或 checkpoint 配置中设置 `mot_inference_video_mode=denoise|cache_only|decoupled_denoise`，也可在推理进程里用环境变量 `MOT_INFERENCE_VIDEO_MODE` 临时覆盖；双向 `full_video` 会拒绝 `cache_only` 和 `decoupled_denoise`，`decoupled_denoise` 仅适用于 `full_video_unidirectional` 且暂不支持 RTC guidance。

## 两卡推理命令

`socket_test_optimized_AR.py` 会从 checkpoint 的 `config.json` 读取 `architecture` 和 `mot_action_video_attention`。因此 first-frame、full-video 双向与 full-video 单向需要使用对应配置的 checkpoint，或使用只改 `config.json` 的 checkpoint view。

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

## Full-Video Independent Noise Smoke

本机已验证的 full-video 独立 timestep smoke checkpoint：

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

两卡 server：

```bash
cd /data/dreamzero_mot

CUDA_VISIBLE_DEVICES=0,1 \
MOT_INFERENCE_VIDEO_MODE=denoise \
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
[MoT] inference_video_mode=denoise (configured=denoise, action_video_attention=full_video)
```

本次 smoke client 返回了 shape 为 `(1, 8)` 的 action。

如果需要做新的诊断实验，建议新建临时实验分支或独立脚本，不再把诊断开关接回生产 action head 配置。
