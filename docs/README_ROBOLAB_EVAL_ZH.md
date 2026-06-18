# DreamZero 在 RoboLab 上的评测流程

本文档记录 DreamZero AR server 配合 RoboLab 进行本地/远程评测的完整流程。命令按“可以直接复制执行”的方式组织。

> ⚠️ **两个 server 不要混用。** DreamZero 提供两个 websocket 入口：
> - `socket_test_robolab_AR.py`：RoboArena/RoboLab 协议（`observation/*` + `session_id` + `endpoint`，连接后下发 `PolicyServerConfig`）。**RoboLab 评测用这个**，支持逐 `session_id` 状态隔离和 server 端 batching。
> - `socket_test_optimized_AR.py`：原生 DROID 协议（`video.*` / `state.*` / `annotation.*`，下发 `policy_metadata`），单 session、无 batching。只给老的 `video.*` 原生 client 用。
>
> 两者**不兼容**，不能把 RoboLab client 指向原生 server，反之亦然。本文档所有命令都使用 `socket_test_robolab_AR.py`。

## 结论先看

RoboLab 默认 benchmark 集合就是 120 个任务：

```bash
find /workspace/robolab/robolab/tasks/benchmark \
  -maxdepth 1 \
  -type f \
  -name '*.py' \
  ! -name '__init__.py' \
  | wc -l
```

预期输出：

```text
120
```

运行 `policies/dreamzero/run.py` 时，如果不传 `--task` / `--tag`，RoboLab 会默认跑 `benchmark` 目录下的全部任务，也就是这 120 个任务。

### 并发 session 与 server 端 batching

`socket_test_robolab_AR.py` 启动的 AR server **不是单 temporal state**：

- 它通过 `supports_parallel_sessions=True` + 逐 session 的 `ActionHeadSessionStore`，为每条连接上的 episode 独立保存 causal / frame buffer / KV-cache 等状态，按 `session_id` 区分，互不污染。
- 它开启 server 端 batching：多个 client 的请求会在 server 进程内聚合成一个 `batch_size>1` 的 forward（受 `--batch-max-size` / `--batch-timeout-ms` 控制），用以提升单 GPU 的吞吐。

因此：

- **状态安全**：可以对同一个 DreamZero server 并发跑多个 RoboLab 进程，或在单个进程里 `--num-envs > 1`，前提是每个连接传不同的 `session_id`（RoboLab 客户端默认会传）。两者都不会互相污染 temporal/cache 状态。
- **填满 batch 的前提**：只有**多个独立进程/连接同时发请求**才会被 server 聚合成 batch。单进程内 `--num-envs > 1` **不行**——RoboLab 客户端的 `infer_many` 默认顺序逐个 env 查询，请求是串行到达 server 的，填不满 batch。要吞吐请用 `run_parallel.py` 起 N 个 worker 子进程。
- 想最快看到单条结果、不在意吞吐时，仍可以用 `--num-envs 1` 单串。
- 只有当 server 关闭了 batching（例如设了 `DYNAMIC_CACHE_SCHEDULE=true`，或用的不是 AR server 而是 `serve_dreamzero_wan22.py` 这类单 session 入口）时，并发安全但不再聚批。

## 机器和端口约定

本文档假设：

- DreamZero server 跑在远端机器 `liuf-nc118`。
- DreamZero server 远端监听 `127.0.0.1:8000`。
- 本地 RoboLab 容器通过 SSH tunnel 访问它。
- 本地 tunnel 端口用 `5000`，即：

```text
本地 127.0.0.1:5000 -> 远端 liuf-nc118 的 127.0.0.1:8000
```

如果你把 tunnel 开成 `8000:127.0.0.1:8000`，则后续 RoboLab 命令里的 `--remote-port` 改成 `8000`。

## 1. 远端启动 DreamZero Server

在远端 DreamZero 推理机器上执行。建议在 `tmux` 里跑，避免 SSH 断开导致 server 退出。

```bash
tmux new -s dreamzero_server
```

进入 DreamZero 仓库并启动 server：

```bash
cd /workspace/dreamzero

CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --standalone \
  --nproc_per_node=2 \
  socket_test_robolab_AR.py \
  --host 127.0.0.1 \
  --port 8000 \
  --model-path /path/to/dreamzero/checkpoint \
  --batch-max-size 8 \
  --batch-timeout-ms 8.0
```

需要按实际机器调整：

- `CUDA_VISIBLE_DEVICES=0,1`
- `--nproc_per_node=2`
- `--model-path /path/to/dreamzero/checkpoint`
- `--batch-max-size`：server 端单次 forward 最多聚合多少 client 请求。并发连接数越多，设大一点越能填满 batch。
- `--batch-timeout-ms`：等待凑批的超时（毫秒）。请求到达节奏不齐时调大（如 8~10）能提高实际 batch size，代价是首请求延迟略增。

如果只用 1 张 GPU：

```bash
cd /workspace/dreamzero

CUDA_VISIBLE_DEVICES=0 torchrun \
  --standalone \
  --nproc_per_node=1 \
  socket_test_robolab_AR.py \
  --host 127.0.0.1 \
  --port 8000 \
  --model-path /path/to/dreamzero/checkpoint \
  --batch-max-size 8 \
  --batch-timeout-ms 8.0
```

看到类似下面日志说明 server 已经开始监听：

```text
Serving DreamZero AR ... websocket server on ws://127.0.0.1:8000
```

## 2. 本地配置 SSH

本地 SSH config 文件：

```text
~/.ssh/config
```

这次使用的配置如下：

```sshconfig
Host nc118
  HostName 223.167.85.140
  User user
  Port 40006
  ForwardAgent yes

Host liuf-nc118
  HostName 127.0.0.1
  Port 22218
  User root
  ProxyJump nc118
  RemoteForward 18080 10.15.88.65:8080
  ForwardAgent yes
```

确保权限正确：

```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/config
```

检查 SSH 配置是否能解析：

```bash
ssh -G liuf-nc118 | grep -E '^(user|hostname|port|proxyjump|remoteforward) '
```

## 3. 本地启动 SSH Tunnel

在 RoboLab 机器上开一个新终端，执行：

```bash
ssh -N -L 5000:127.0.0.1:8000 liuf-nc118
```

这个终端需要一直保持打开。

确认本地端口已经监听：

```bash
ss -ltnp | grep ':5000'
```

应该能看到类似：

```text
LISTEN ... 127.0.0.1:5000 ... users:(("ssh",pid=...,fd=...))
```

如果你看到 `Connection refused`，优先检查 RoboLab 命令里的端口是否和 tunnel 本地端口一致。

例如你开的是：

```bash
ssh -N -L 5000:127.0.0.1:8000 liuf-nc118
```

那 RoboLab 必须使用：

```text
--remote-port 5000
```

不是 `8000`。

## 4. 先跑单任务 Smoke Test

进入 RoboLab 容器/环境：

```bash
cd /workspace/robolab
```

RoboLab Isaac 容器里通常没有 `uv`，请使用 Isaac Sim 的 Python wrapper：

```bash
/workspace/isaaclab/_isaac_sim/python.sh policies/dreamzero/run.py \
  --headless \
  --task BananaInBowlTask \
  --num-envs 1 \
  --remote-host 127.0.0.1 \
  --remote-port 5000
```

如果你的本地 tunnel 是 `8000:127.0.0.1:8000`，则改成：

```bash
--remote-port 8000
```

输出会写到：

```text
/workspace/robolab/output/<timestamp>_dreamzero/
```

## 5. 跑 120 个 Benchmark 任务

不传 `--task` / `--tag` 就会跑默认 benchmark 全量任务，也就是 120 个任务。

建议指定一个固定 output 文件夹名，方便断点续跑：

```bash
cd /workspace/robolab

/workspace/isaaclab/_isaac_sim/python.sh policies/dreamzero/run.py \
  --headless \
  --num-envs 1 \
  --num-runs 1 \
  --remote-host 127.0.0.1 \
  --remote-port 5000 \
  --output-folder-name dreamzero_120_eval
```

如果中途中断，重新执行同一条命令即可。RoboLab 会读取：

```text
/workspace/robolab/output/dreamzero_120_eval/episode_results.jsonl
```

已经完成的 task / episode 会被跳过。

### 不保存视频，加快评测并节省磁盘

全量 120 任务保存视频会占用较多磁盘。如果只关心 success rate，可以加：

```bash
--video-mode none
```

完整命令：

```bash
cd /workspace/robolab

/workspace/isaaclab/_isaac_sim/python.sh policies/dreamzero/run.py \
  --headless \
  --num-envs 1 \
  --num-runs 1 \
  --remote-host 127.0.0.1 \
  --remote-port 5000 \
  --output-folder-name dreamzero_120_eval \
  --video-mode none
```

如果希望 dashboard 里能看 episode 视频，保留默认 `--video-mode all`，或只保存 sensor：

```bash
--video-mode sensor
```

## 6. 多 Episode 评测

AR server 支持并发 session（每个连接独立保存 causal/KV-cache 状态）。并发是安全的，但要区分两件事：

- **状态安全**：单进程 `--num-envs > 1`、或多个 RoboLab 进程连同一个 server，都不会互相污染 temporal/cache 状态（按 `session_id` 隔离）。
- **能否填满 server 端 batch**：RoboLab 客户端的 `infer_many` 默认是**顺序**逐个 env 查询的（`DreamZeroClient` 没有并发 override），所以**单进程 `--num-envs > 1` 不会并发发请求、也填不满 batch**。真正能让 server 端 batch 拼满的是**多进程并发**：用 `run_parallel.py` 起 N 个 worker 子进程，每个 worker 各持一条独立 websocket 连接，这些连接的请求才会在 server 端同时到达、聚合成 `batch_size>1` 的 forward。

换句话说：想要吞吐就走 `run_parallel.py` 多 worker；只是想多跑几个 episode、不在意吞吐，单进程 `--num-envs > 1` + 多 `--num-runs` 也可以（state-safe，但 batch 基本不会被填满）。

如果每个任务要跑 5 个 episode，单进程单 env 串行（最简单）：

```bash
cd /workspace/robolab

/workspace/isaaclab/_isaac_sim/python.sh policies/dreamzero/run.py \
  --headless \
  --num-envs 1 \
  --num-runs 5 \
  --remote-host 127.0.0.1 \
  --remote-port 5000 \
  --output-folder-name dreamzero_120_eval_5runs
```

总 episode 数：

```text
120 tasks * 5 runs * 1 env = 600 episodes
```

## 7. Adaptive Sampling

RoboLab 支持按置信区间自适应采样。

例如每个任务最多跑 200 个 episode，CI 足够窄时提前停止：

```bash
cd /workspace/robolab

/workspace/isaaclab/_isaac_sim/python.sh policies/dreamzero/run.py \
  --headless \
  --num-envs 1 \
  --num-episodes-adaptive 200 \
  --ci-pp-width 0.14 \
  --remote-host 127.0.0.1 \
  --remote-port 5000 \
  --output-folder-name dreamzero_120_adaptive_200 \
  --video-mode none
```

注意：单进程时 adaptive sampling 仍是串行的，比较耗时。要真正加速，需要用 `run_parallel.py` 起**多个 worker 子进程**（每个一条独立连接）去并发填满 server 端 batch；单进程内提高 `--num-envs` 只是 state-safe 的串行，不会并发填 batch。

## 8. Dashboard 看板

Dashboard 只读取 RoboLab output，不连接 DreamZero server。

启动：

```bash
cd /workspace/robolab

/workspace/isaaclab/_isaac_sim/python.sh -m dashboard.cli \
  --output-dir /workspace/robolab/output \
  --host 0.0.0.0 \
  --port 8080
```

如果浏览器就在同一台机器上：

```text
http://127.0.0.1:8080
```

如果浏览器在你的本地电脑，而 dashboard 跑在远端/容器里，再开一个 tunnel：

```bash
ssh -N -L 8080:127.0.0.1:8080 liuf-nc118
```

然后本地浏览器打开：

```text
http://127.0.0.1:8080
```

Dashboard 会扫描：

```text
/workspace/robolab/output/
```

下面的实验目录，例如：

```text
/workspace/robolab/output/dreamzero_120_eval
/workspace/robolab/output/2026-06-15_16-31-56_dreamzero
```

## 9. 结果文件在哪里

一次评测的典型目录结构：

```text
/workspace/robolab/output/<output-folder-name>/
  episode_results.jsonl
  <TaskName>/
    env_cfg.json
    run_0.hdf5
    *.mp4
```

其中：

- `episode_results.jsonl`: 每个 episode 的 success/failure、score、timing 等结果。
- `run_0.hdf5`: episode 轨迹数据。
- `*.mp4`: sensor / viewport 视频，取决于 `--video-mode`。

## 10. 常见问题

### `uv: command not found`

RoboLab Isaac 容器里不一定安装 `uv`。

使用：

```bash
/workspace/isaaclab/_isaac_sim/python.sh policies/dreamzero/run.py ...
```

不要使用：

```bash
uv run python policies/dreamzero/run.py ...
```

### `Connection refused`

典型错误：

```text
[DreamZeroClient] Connecting to DreamZero server at ws://127.0.0.1:8000...
ConnectionRefusedError: [Errno 111] Connection refused
```

排查顺序：

1. DreamZero server 是否在远端启动成功。
2. SSH tunnel 是否还开着。
3. 本地监听端口是否正确：

```bash
ss -ltnp | grep -E ':(5000|8000)'
```

4. RoboLab 的 `--remote-port` 是否等于 tunnel 的本地端口。

如果 tunnel 是：

```bash
ssh -N -L 5000:127.0.0.1:8000 liuf-nc118
```

RoboLab 必须使用：

```text
--remote-port 5000
```

### `Warning: remote port forwarding failed for listen port 18080`

这是因为 SSH config 里有：

```sshconfig
RemoteForward 18080 10.15.88.65:8080
```

`RemoteForward` 是让远端监听端口 `18080`，不是 RoboLab 连接 DreamZero server 所需要的 tunnel。

DreamZero 评测需要的是 `LocalForward`：

```bash
ssh -N -L 5000:127.0.0.1:8000 liuf-nc118
```

如果不需要 `RemoteForward 18080 ...`，可以从 SSH config 里删掉或注释掉。

### 多开 RoboLab 进程 / 多 env 连同一个 DreamZero server 安全吗

安全。`socket_test_robolab_AR.py` 为每条连接按 `session_id` 独立保存 temporal cache / frame buffer / reset 状态（`ActionHeadSessionStore`），并发请求互不污染，并会在 server 端聚合成 batch forward。每个 RoboLab 进程 / 每个 env 只要用不同的 `session_id`（客户端默认会传）即可。

只有在以下情况才需要串行 `--num-envs 1`，或多起几套独立 server：

- 用的不是 AR server，而是 `serve_dreamzero_wan22.py` 这类单 session 入口（它们没有逐 session 状态隔离，也没有 server 端 batching）。
- server 被显式关闭了 batching（例如设了 `DYNAMIC_CACHE_SCHEDULE=true`）。

如果确实要多起独立 server 分担负载：

```text
server A: remote 8000 -> local 5000
server B: remote 8001 -> local 5001
server C: remote 8002 -> local 5002
```

每个 RoboLab 进程连接不同的 `--remote-port`。

## 11. 快速命令汇总

远端启动 server：

```bash
cd /workspace/dreamzero
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  socket_test_robolab_AR.py \
  --host 127.0.0.1 \
  --port 8000 \
  --model-path /path/to/dreamzero/checkpoint \
  --batch-max-size 8 \
  --batch-timeout-ms 8.0
```

本地开 tunnel：

```bash
ssh -N -L 5000:127.0.0.1:8000 liuf-nc118
```

单任务测试：

```bash
cd /workspace/robolab
/workspace/isaaclab/_isaac_sim/python.sh policies/dreamzero/run.py \
  --headless \
  --task BananaInBowlTask \
  --num-envs 1 \
  --remote-host 127.0.0.1 \
  --remote-port 5000
```

120 任务全量评测：

```bash
cd /workspace/robolab
/workspace/isaaclab/_isaac_sim/python.sh policies/dreamzero/run.py \
  --headless \
  --num-envs 1 \
  --num-runs 1 \
  --remote-host 127.0.0.1 \
  --remote-port 5000 \
  --output-folder-name dreamzero_120_eval
```

Dashboard：

```bash
cd /workspace/robolab
/workspace/isaaclab/_isaac_sim/python.sh -m dashboard.cli \
  --output-dir /workspace/robolab/output \
  --host 0.0.0.0 \
  --port 8080
```
