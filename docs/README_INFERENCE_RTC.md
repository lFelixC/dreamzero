# DreamZero Inference RTC

这份文档对应当前的 inference-only RTC 改造，覆盖：

- 单臂 server: `/data/dreamzero/socket_test_optimized_AR.py`
- 双臂 server: `/data/dreamzero/socket_test_optimized_aloha_x5lite_bimanual.py`
- 单臂 client: `/data/dreamzero/example/remote_infer/main_dreamzero.py`
- 双臂 client: `/data/aloha_agilex_arx5/python/examples/X5lite_old/inference_dreamzero_aloha_sync.py`
- 双臂启动脚本: `/data/aloha_agilex_arx5/tools/inference_dreamzero_aloha_sync.sh`

## 1. 改造原则

- RTC 只走 inference 路径，不改训练流程。
- RTC 的真正开关由 client 决定。
- server 侧 RTC 参数名保持不变，仍然负责 guidance 超参数。
- server 是否走 RTC，最终按请求里是否带 `rtc_step_idx` 决定；也就是说，不带这个字段就是普通 chunk inference。
- 两个 client 都支持可选的 action interpolation，用来把一个 chunk 拉长成更多控制周期。

## 2. 推荐的 interpolation 策略

当前已知情况是：`24` 个 action chunk 的一次推理大约需要 `1.5s`。只靠 RTC 不能完全消除等待，因此 client 侧新增了可控插值。

当前实现选择：

- 只对关节位姿做线性插值。
- 插进去的中间步保持 gripper 不变。
- 这样比“连 gripper 一起线插”稳定，尤其是真机上夹爪更不容易来回抖。

单臂：

- 关节维度 `0:7` 线性插值。
- gripper 维度 `7` 在插值步保持前一时刻目标。

双臂：

- 左臂关节维度 `0:6` 线性插值。
- 右臂关节维度 `7:13` 线性插值。
- gripper 维度 `6` 和 `13` 在插值步保持前一时刻目标。

推荐起点：

- 控制频率先用 `15 Hz`。
- `--interpolate-actions`
- `--interpolation-substeps 1`

这样 `24` 个原始 action 会扩成 `47` 个控制步，15 Hz 下大约能覆盖 `3.13s`，通常足够把下一次推理接上。  
如果还经常等推理，可以继续试 `--interpolation-substeps 2`，但动作会更“软”，跟手性也会再下降一点。

## 3. Server 运行命令

下面命令只展示 RTC 相关部分。后端切换、`torch.compile`、`FA2/TE` 的 recipe 见：

- `/data/dreamzero/docs/README_INFERENCE_BACKENDS.md`

### 3.1 单臂 server

```bash
cd /data/dreamzero

torchrun --standalone --nproc_per_node 1 \
  socket_test_optimized_AR.py \
  --model-path /path/to/checkpoint \
  --port 8000 \
  --enable-dit-cache \
  --rtc-execution-horizon 10 \
  --rtc-max-guidance-weight 10.0 \
  --rtc-prefix-attention-schedule EXP \
  --rtc-guidance-max-steps 4 \
  --rtc-guidance-step-stride 1
```

### 3.2 双臂 server

```bash
cd /data/dreamzero

torchrun --standalone --nproc_per_node 1 \
  socket_test_optimized_aloha_x5lite_bimanual.py \
  --model_path /path/to/checkpoint \
  --port 8000 \
  --enable-dit-cache \
  --rtc-execution-horizon 10 \
  --rtc-max-guidance-weight 10.0 \
  --rtc-prefix-attention-schedule EXP \
  --rtc-guidance-max-steps 4 \
  --rtc-guidance-step-stride 1
```

说明：

- server 端保留了 `--use-rtc`，但现在不再作为真正的 on/off 开关。
- 是否进入 RTC，取决于 client 请求里有没有带 `rtc_step_idx`。
- 如果 client 不开 RTC，server 会自动回到普通 chunk 推理。

## 4. Client 运行命令

### 4.1 单臂 client

```bash
cd /data/dreamzero

python example/remote_infer/main_dreamzero.py \
  --remote-host 127.0.0.1 \
  --remote-port 8000 \
  --control-frequency 15 \
  --open-loop-horizon 8 \
  --use-rtc \
  --enable-async-prefetch \
  --rtc-inference-delay-steps 2 \
  --rtc-use-measured-delay \
  --rtc-handoff-joint-blend 0.6 \
  --interpolate-actions \
  --interpolation-substeps 1 \
  --log-timing
```

如果想看逐步 trace：

```bash
cd /data/dreamzero

python example/remote_infer/main_dreamzero.py \
  --remote-host 127.0.0.1 \
  --remote-port 8000 \
  --control-frequency 15 \
  --use-rtc \
  --rtc-inference-delay-steps 2 \
  --rtc-use-measured-delay \
  --interpolate-actions \
  --interpolation-substeps 1 \
  --rtc-debug-trace-path /tmp/dreamzero_single_arm_rtc.jsonl
```

### 4.2 双臂 client: 推荐用 shell

```bash
cd /data/aloha_agilex_arx5

PROMPT="Pick up the target object and finish the task." \
POLICY_HOST=127.0.0.1 \
POLICY_PORT=8000 \
CONTROL_RATE=15 \
OPEN_LOOP_HORIZON=8 \
USE_RTC=1 \
ENABLE_ASYNC_PREFETCH=1 \
RTC_INFERENCE_DELAY_STEPS=2 \
RTC_USE_MEASURED_DELAY=1 \
RTC_HANDOFF_JOINT_BLEND=0.6 \
INTERPOLATE_ACTIONS=1 \
INTERPOLATION_SUBSTEPS=1 \
bash tools/inference_dreamzero_aloha_sync.sh --yes
```

如果想写 trace：

```bash
cd /data/aloha_agilex_arx5

PROMPT="Pick up the target object and finish the task." \
POLICY_HOST=127.0.0.1 \
POLICY_PORT=8000 \
CONTROL_RATE=15 \
USE_RTC=1 \
RTC_INFERENCE_DELAY_STEPS=2 \
INTERPOLATE_ACTIONS=1 \
INTERPOLATION_SUBSTEPS=1 \
RTC_DEBUG_TRACE_PATH=/tmp/dreamzero_aloha_rtc.jsonl \
bash tools/inference_dreamzero_aloha_sync.sh --yes
```

### 4.3 双臂 client: 直接跑 python

```bash
cd /data/aloha_agilex_arx5

python python/examples/X5lite_old/inference_dreamzero_aloha_sync.py \
  --policy-host 127.0.0.1 \
  --policy-port 8000 \
  --prompt "Pick up the target object and finish the task." \
  --control-rate 15 \
  --open-loop-horizon 8 \
  --use-rtc \
  --enable-async-prefetch \
  --rtc-inference-delay-steps 2 \
  --rtc-use-measured-delay \
  --rtc-handoff-joint-blend 0.6 \
  --interpolate-actions \
  --interpolation-substeps 1 \
  --yes
```

## 5. 参数说明

### 5.1 Client 侧 RTC 参数

- `use_rtc`
  真正的 RTC 开关。打开后 client 会带上 `rtc_step_idx` 和 `rtc_inference_delay_steps`。

- `enable_async_prefetch`
  是否提前发下一次请求。RTC 打开时，client 内部也会自动把 prefetch 打开。

- `rtc_inference_delay_steps`
  估计推理结果返回时，当前 chunk 已经额外消耗了多少个“原始 action step”。

- `rtc_use_measured_delay`
  用实际 round-trip time 动态更新 `rtc_inference_delay_steps`。建议打开。

- `rtc_handoff_joint_blend`
  chunk 切换第一步的 joint blend 系数。`1.0` 表示不做 smoothing。推荐从 `0.6` 开始。

- `rtc_debug_trace_path`
  可选 JSONL trace 输出路径。

- `interpolate_actions`
  是否开启 client 侧插值扩展。

- `interpolation_substeps`
  每两个原始 action 之间插入多少个额外控制步。推荐先从 `1` 开始。

### 5.2 Server 侧 RTC 参数

- `rtc_execution_horizon`
  约束 prefix 的有效执行范围。

- `rtc_max_guidance_weight`
  prefix guidance 的最大权重。

- `rtc_prefix_attention_schedule`
  prefix 权重衰减策略，当前常用 `EXP`。

- `rtc_guidance_max_steps`
  只在 diffusion 后几步打开 RTC guidance。

- `rtc_guidance_step_stride`
  RTC guidance 的步长。

- `use_rtc`
  兼容保留参数。现在建议把它当成“server 支持 RTC guidance 配置”而不是最终开关。

## 6. 建议的起步配置

如果你现在的 24-step chunk 推理大约 `1.5s`，建议先从下面这个组合开始：

- control frequency: `15`
- `use_rtc=true`
- `rtc_use_measured_delay=true`
- `rtc_inference_delay_steps=2`
- `rtc_handoff_joint_blend=0.6`
- `interpolate_actions=true`
- `interpolation_substeps=1`

如果表现是：

- 还是经常等下一块结果：优先试 `interpolation_substeps=2`
- 切块时第一步有点冲：把 `rtc_handoff_joint_blend` 从 `0.6` 降到 `0.4`
- 动作太“软”或滞后感明显：把 `interpolation_substeps` 降回 `1` 或直接关闭 interpolation

## 7. 当前实现要点

- RTC 的 left-over 计算和 delay trim 都在 inference 路径里做。
- training 相关代码没有被改成依赖 RTC。
- server 和 client 仍然兼容非 RTC 的普通 chunk 推理。
- 双臂 shell 已经支持把 RTC/interpolation 参数透传到 python client。
