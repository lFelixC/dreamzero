# ALOHA DreamZero Remote Inference

这份说明对应 ALOHA 机械臂侧的 DreamZero client/server 入口：

- server: `/data/dreamzero/socket_test_optimized_aloha_x5lite_bimanual.py`

- shell: `/data/aloha_agilex_arx5/tools/inference_dreamzero_aloha_sync.sh`
- python: `/data/aloha_agilex_arx5/python/examples/X5lite_old/inference_dreamzero_aloha_sync.py`

## 0. 先起 DreamZero server

单卡最简单的启动方式：

```bash
cd /data/dreamzero

torchrun --standalone --nproc_per_node 1 \
  /data/dreamzero/socket_test_optimized_aloha_x5lite_bimanual.py \
  --model_path /path/to/your/checkpoint_dir \
  --port 8000
```

如果需要多卡挂模型，把 `--nproc_per_node 1` 改成实际卡数即可，例如：

```bash
cd /data/dreamzero

torchrun --standalone --nproc_per_node 8 \
  /data/dreamzero/socket_test_optimized_aloha_x5lite_bimanual.py \
  --model_path /path/to/your/checkpoint_dir \
  --port 8000
```

这个 server 会：

- 用 `EmbodimentTag.ALOHA_X5LITE_BIMANUAL` 加载 checkpoint
- 接收三路图像：`video.cam_high / video.cam_left / video.cam_right`
- 接收双臂状态：`state.left_* / state.right_*`
- 输出和 ALOHA client 对齐的 `14D absolute action`

## 1. 再做 SSH 端口转发

如果 DreamZero server 跑在远端机器的 `8000` 端口，本机执行：

```bash
ssh -N -L 8000:127.0.0.1:8000 <remote_host>
```

转发建立后，ALOHA 主机上的 client 直接连接本地：

- `POLICY_HOST=127.0.0.1`
- `POLICY_PORT=8000`

## 2. 推荐运行命令

在 ALOHA 主机执行：

```bash
cd /data/aloha_agilex_arx5

PROMPT="Pick up the target object and finish the task." \
POLICY_HOST=127.0.0.1 \
POLICY_PORT=8000 \
bash tools/inference_dreamzero_aloha_sync.sh --yes
```

## 3. 常用可调参数

如果 server 返回的是 packed 14D action，但顺序是 `right_first`，运行时加：

```bash
SERVER_ACTION_ORDER=right_first bash tools/inference_dreamzero_aloha_sync.sh --yes
```

如果确认 server 只吃一套键，可以关掉重复发送来减少带宽：

```bash
SEND_MODEL_KEYS=1 SEND_RAW_KEYS=0 bash tools/inference_dreamzero_aloha_sync.sh --yes
```

或者：

```bash
SEND_MODEL_KEYS=0 SEND_RAW_KEYS=1 bash tools/inference_dreamzero_aloha_sync.sh --yes
```

## 4. 当前 client 默认行为

- 三路图像来自 `cam_high / cam_left / cam_right`
- 双臂状态来自 ALOHA SDK 当前 follower `qpos`
- 首次请求发单帧，后续请求默认发最近 `4` 帧
- 默认每执行 `8` 个 action 就重新向 server 请求新 chunk
- 期望 server 最终返回 `14D` 双臂 action，或返回可解构为左右臂的 action 字典
