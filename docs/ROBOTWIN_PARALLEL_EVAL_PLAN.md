# RoboTwin Parallel Eval Roadmap

## V1 Summary

第一版目标已经完成：在不修改 `third_party/lerobot` 和
`third_party/RoboTwin` 的前提下，实现了 **RoboTwin 多 env subprocess 并行
rollout + DreamZero websocket batched inference**。

落地文件：

- `example/robotwin/robotwin_fast_env.py`：本仓库内的 RoboTwin fast-step /
  `step_chunk` wrapper，只在 chunk 末尾或 done 时取 obs。
- `example/robotwin/parallel_env_worker.py`：RoboTwin subprocess worker，
  worker 内部再 import RoboTwin / LeRobot，避免父进程 fork CUDA / SAPIEN。
- `example/robotwin/parallel_eval.py`：同步 wave controller，收集
  `[B,T,H,W,3]` obs batch，发送给单 websocket server，接收 `[B,H,14]`。
- `example/robotwin/run_robotwin_eval.sh`：统一入口；`MODE=parallel`
  一键启动 server、等待端口、
  运行并行 eval、写 report。
- `socket_test_optimized_aloha_x5lite_bimanual.py` 和
  `eval_utils/serve_dreamzero_wan22.py`：支持 batched observation/action
  conversion 和 batch resize helper。

第一版明确不做 Ray、thread-based RoboTwin env、third-party 修改、异步 refill、
mixed-task batch、per-env temporal cache、pipeline overlap、并行视频保存和 PPO/GRPO。

## V1 Validation

固定测试条件：

```text
task: beat_block_hammer
episodes: 8
episode_length: 400
open_loop_horizon: 8
max_steps: 50
save_video: 0
checkpoint: /data/checkpoints/dreamzero/dreamzero_robotwin/checkpoint-10000
```

串行 baseline：

```text
server GPU: 0
RoboTwin client GPU: 1
output: /data/checkpoints/dreamzero/robotwin_eval_runs/bench_serial_01_20260515_043413
client-only wall time: 851.73 s
episodes: 8 / 8
steps: all 400
successes: 0
```

并行 V1：

```text
server GPU: 2
env worker GPU: 3
num_envs: 4
output: /data/checkpoints/dreamzero/robotwin_eval_runs/bench_parallel_23_20260515_045037
controller-only wall time: 375.68 s
episodes: 8 / 8
steps: all 400
successes: 0
```

结果：

```text
speedup = 851.73 / 375.68 = 2.27x
```

这超过了第一版 `1.8x` 的验收线。server log 确认 batch 推理生效：

```text
action_shape=(4, 24, 14)
```

第一版结论：

- 当前工程已经具备 RoboTwin env 并行推理能力。
- 这个能力可以作为后续 RL rollout collector 的基础。
- 这个能力也已经能作为 RoboTwin eval 加速的第一版工具使用。

## V2 Goal

第二版目标同时服务 RoboTwin eval 加速和后续 RL collector：

1. **RoboTwin Eval Accelerator**
   - 更快跑 50-task / multi-seed eval。
   - 支持 benchmark、profiling、失败重试、断点续跑、自动汇总。
   - 后续支持多 server shard + 多 env GPU，提高任务吞吐。
2. **RoboTwin RL Collector Foundation**
   - 在并行 rollout 中稳定收集 episode / transition 数据。
   - 明确 reward、done、action chunk、obs、seed、task、policy version 字段。
   - 为 rejection-SFT、reward-weighted finetune、PPO / GRPO 做数据接口准备。

核心原则：

- eval 加速和 RL collector 共用 env worker / server shard / scheduler 基础设施。
- eval 输出继续兼容当前 `report.json` / `report.csv`。
- RL 输出新增 trajectory dataset，不污染 eval report。
- 默认仍不要求 Ray；单机多 GPU 用 subprocess。多机或 trainer 资源调度再评估 Ray。

## V2 Review Adoption

采纳评审的主体意见：

- **A1 Benchmark Harness**：新增 GPU idle precheck、完整 CLI/config snapshot、
  `benchmark.json` / `benchmark.md`，记录 GPU 型号、显存、utilization、
  git commit 和 timestamp。
- **A2 Env GPU Distribution**：episode/report 记录 worker 的
  `CUDA_VISIBLE_DEVICES`，按 env GPU 汇总 episode 数、worker 数、
  `env_step_time`、`get_obs_time`、`episode_wall_time`，并报告 imbalance。
- **A3 Persistent Env Reuse**：接受风险提示，但 V2 不默认开启 reuse。先做
  fresh-vs-reuse 对比实验，后续再加入 `max_env_reuse_episodes=20` 和失败率
  >5% 回退 cold reset。
- **A4 Client-Side Image Resize**：采用 websocket 连接后的 server metadata
  `image_resolution`，并提供 `--client-image-resolution none|auto|HxW`。默认
  `none`，profile/实验时显式开启。
- **A5 Multi-Server Sharding**：后续实现 `MultiServerClient` 或 shard client dict、
  自动端口、shard 子目录聚合和显存预算。
- **A6 Resume / Retry**：后续 `run_state.json` 记录 timestamp、git commit、
  config snapshot；worker restart 保持同一 `CUDA_VISIBLE_DEVICES`。
- **B1/B2/B4**：RL dataset 第一版采用 `manifest.jsonl + episode/images.npz`，
  默认 sparse reward，dense reward 只做可选 hook；B1 必须排在 B4 之前。
- **B5**：V2 不做 intra-batch async refill。真正 per-env temporal cache、
  action head causal cache save/restore、`DistributedWorkerLoop` barrier 改造
  延期到 V3，V2 只产出技术 spike 文档。
- **Cross-track**：加入跨 task temporal cache reset smoke、V1 2.27x 可复现
  benchmark、profile-guided optimization。

延期或不采纳：

- 不把 Ray 加入 V2 eval path。
- 不把 env reuse 设为默认。
- 不默认实现 dense reward。
- 不在 V2 做真正 per-env async refill；V2 只做 shard-level/task-level parallelism。

## V2 First Slice Implemented

本切片把 V1 加强成可复现 benchmark / profile 工具：

- 新增 `example/robotwin/benchmark_parallel_eval.py`：
  - `precheck`：用 `nvidia-smi` 对选定 GPU 做 idle 检查和 snapshot。
  - `summarize`：聚合 serial / parallel 输出，写 `benchmark.json` 和
    `benchmark.md`。
- 新增 `example/robotwin/run_robotwin_eval.sh MODE=benchmark`：
  - 默认 serial 用 server GPU `0`、env GPU `1`。
  - 默认 parallel 用 server GPU `2`、env GPU `3`、`NUM_ENVS=4`。
  - 排除 server 启动时间，用 serial client wall time 对比 parallel
    controller wall time。
- 扩展 `example/robotwin/run_robotwin_eval.sh MODE=parallel`：
  - 写 `timings.json`，包含 server startup、controller wall、total wall。
  - 透传 `PROFILE` 和 `CLIENT_IMAGE_RESOLUTION`。
- 扩展 `example/robotwin/parallel_eval.py`：
  - 新增 `--profile`。
  - 新增 `--client-image-resolution none|auto|HxW`。
  - 写 `run_config.json`，记录 argv、args、server metadata、workers、git commit。
  - episode JSON 增加 worker GPU 和 `env_reuse_mode: cold_reset`。
  - report 增加 env GPU 汇总和 wall-time imbalance。
  - profile 模式记录 payload build/pack、websocket roundtrip、action normalize、
    worker wait。
- 扩展 `example/robotwin/robotwin_fast_env.py`：
  - 增加无新依赖的 client-side image resize helper。
  - `stack_robotwin_payloads(..., image_resolution=...)` 可在 controller 侧 resize。

## V2 Implementation Order

后续顺序固定为：

```text
A1 → A2 → [A3 experiment + B1 design] → A4 → [B2 + B3] → A5 → [A6 + B4] → B5 spike
```

其中：

- A1/A2/A4 的第一切片已经落地。
- A3 先做实验，不默认启用。
- B1 schema 要早于 collector mode。
- B5 从 V2 开始写 spike，但真正 async refill 延期到 V3。

## RL Dataset Draft

第一版 RL collector schema 先采用无新依赖方案：

```text
rollouts/
  manifest.jsonl
  task_name/
    episode_000000/
      episode.json
      chunks.npz
      images.npz
```

约定：

- `manifest.jsonl` 存 task、seed、episode index、policy version、checkpoint、
  success、reward sum、step count、路径引用。
- `chunks.npz` 存 action chunk、state、reward_per_step、done_per_step、
  chunk timing。
- `images.npz` 存 `chunk_{i}_cam_{name}`，避免大量碎片图像文件。
- 默认 sparse reward：中间 action step 为 `0.0`，terminal step 使用 RoboTwin
  success/reward。
- dense reward hook 后续预留在 `example/robotwin/reward.py`，默认 disabled。

## Validation Commands

静态检查：

```bash
python -m py_compile \
  example/robotwin/parallel_eval.py \
  example/robotwin/parallel_env_worker.py \
  example/robotwin/robotwin_fast_env.py \
  example/robotwin/benchmark_parallel_eval.py \
  socket_test_optimized_aloha_x5lite_bimanual.py
bash -n example/robotwin/run_robotwin_eval.sh
```

Dry-run worker smoke：

```bash
NUM_ENVS=1 EPISODES=1 TASKS=beat_block_hammer DRY_RUN_ACTIONS=1 \
  MODE=parallel bash example/robotwin/run_robotwin_eval.sh
```

Main benchmark acceptance：

```bash
TASKS=beat_block_hammer EPISODES=8 NUM_ENVS=4 SAVE_VIDEO=0 OPEN_LOOP_HORIZON=8 \
SERIAL_SERVER_CUDA=0 SERIAL_ENV_CUDA=1 \
PARALLEL_SERVER_CUDA=2 PARALLEL_ENV_CUDA=3 \
OUTPUT_ROOT=/data/checkpoints/dreamzero/robotwin_eval_runs \
MODE=benchmark bash example/robotwin/run_robotwin_eval.sh
```

Pass criteria：

- 所有 8 个 episodes 都有 episode JSON。
- parallel report 有 batched action shape，例如 `[4,H,14]`。
- 排除 server 启动后，parallel wall time 相比 serial 至少 `1.8x`。
- `benchmark.json` 包含 GPU snapshot、CLI/config snapshot、git commit、timestamp。
- 每个 episode/report 可追溯 worker GPU。
- GPU precheck 不通过时脚本明确失败，不自动占用忙卡。

Cross-task smoke：

```bash
TASKS=beat_block_hammer,pick_dual_bottles EPISODES=2 NUM_ENVS=2 SAVE_VIDEO=0 \
SERVER_CUDA=2 ENV_CUDA=3 \
MODE=parallel bash example/robotwin/run_robotwin_eval.sh
```
