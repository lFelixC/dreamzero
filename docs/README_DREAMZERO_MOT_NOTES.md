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

- `architecture`: `joint | mot`
- `mot_action_hidden_dim`
- `mot_action_ffn_dim`
- `mot_action_num_layers`
- `mot_action_num_heads`
- `mot_action_video_attention`: `first_frame | full_video | none`
- `mot_action_video_ki`: `true | false`

`mot_action_video_attention` 含义：

- `first_frame`: action expert 只看首帧 video K/V，当前默认生产设置。
- `full_video`: action expert 看当前 video block 的完整 K/V。
- `none`: action expert 不看 video K/V，只走 action/state 自身 token。

`causal` alias 已删除；如需原先等价行为，请使用 `full_video`。

`mot_action_video_ki` 含义：

- `false`: action expert 可读取 video K/V，但 action loss 不通过 video K/V 回传到 video expert，当前默认。
- `true`: action loss 保留到 video K/V 的梯度，用于让 action supervision 共同更新 video expert。

## 固定主链路

以下实验开关已从配置入口移除，并固定为当前生产默认：

- action shared context: 关闭。
- video state context: 开启。
- action expert gate 初始化: AdaLN-zero 默认初始化。
- 训练 noise/timestep: video/action 使用标准耦合采样。
- 推理模式: video/action 一起 denoise。

因此当前 MoT 主链路没有 no-denoise、action-only cache refresh、video refresh stride 或 noise decouple 分支。

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

非 MoT 训练脚本仍保留；MoT small、joint small 和 MoT ablation 脚本已删除。

## 推理行为

普通推理和 RTC 兼容路径保留。MoT 下 action prediction 与 video denoising 同步执行，不再暴露单独关闭 video denoising 的运行时开关。

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

如果需要做新的诊断实验，建议新建临时实验分支或独立脚本，不再把诊断开关接回生产 action head 配置。
