# RoboTwin Eval Docker 固化安装教程

这份文档用于在另一台服务器上准备 DreamZero + RoboTwin live eval 环境。目标是把环境固化在 Docker 镜像里，运行时只挂载代码、checkpoint 和输出目录。

当前 A800 训练脚本默认使用：

```bash
VIRTUAL_ENV=/opt/venvs/dreamzero
PYTHON_BIN=/opt/venvs/dreamzero/bin/python
DREAMZERO_ROOT=/2023133163/liuf/dreamzero
DATASET_ROOT=/2023133163/datasets/dreamzero
CHECKPOINT_ROOT=/2023133163/checkpoints/dreamzero
```

RoboTwin live eval 建议在同一个 Docker 镜像里固化两个 venv：

```text
/opt/venvs/dreamzero    # DreamZero policy server，加载模型权重
/opt/venvs/robotwin310  # RoboTwin/SAPIEN client，跑仿真、渲染、controller
```

不要把 venv 放在项目目录 `.venv` 里。项目目录通常是挂载盘，保存镜像时不一定会保存进去；`/opt/venvs/*` 更稳。

## 1. 推荐目录

在 A800 集群上建议统一使用：

```bash
export DREAMZERO_ROOT=/2023133163/liuf/dreamzero
export CHECKPOINT_ROOT=/2023133163/checkpoints/dreamzero
export OUTPUT_ROOT=/2023133163/checkpoints/dreamzero/robotwin_eval_runs
```

如果你在 `/data/dreamzero_mot` 调试，也可以把文中的 `/2023133163/liuf/dreamzero` 替换成 `/data/dreamzero_mot`。

## 2. 启动 Docker 容器

优先使用已经固化过 DreamZero 训练环境的镜像，例如：

```bash
docker run -it --name dreamzero-robotwin-eval \
  --gpus all \
  --network host \
  --ipc host \
  --shm-size 128g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -v /2023133163:/2023133163 \
  dreamzero-uv:latest \
  /bin/bash
```

如果没有现成镜像，用你们平台的 CUDA/PyTorch 基础镜像也可以。关键是宿主机 NVIDIA driver 要正常，容器里能看到 GPU：

```bash
nvidia-smi
```

## 3. 基础系统依赖

进入容器后安装编译、下载、OpenGL/EGL、视频和 SAPIEN 相关依赖：

```bash
apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates curl git wget unzip rsync \
  build-essential cmake ninja-build pkg-config \
  ffmpeg \
  libgl1 libglvnd0 libegl1 libegl-dev libgles2 \
  libglib2.0-0 libx11-6 libxext6 libxrender1 libsm6 \
  libxrandr2 libxi6 libxinerama1 libxcursor1 libxkbcommon0 \
  libvulkan1 mesa-vulkan-drivers \
  xvfb
```

如果容器 DNS 不通，先临时写公网 DNS：

```bash
cp /etc/resolv.conf /etc/resolv.conf.bak.$(date +%s) 2>/dev/null || true
cat > /etc/resolv.conf <<'EOF'
nameserver 223.5.5.5
nameserver 114.114.114.114
nameserver 8.8.8.8
options timeout:2 attempts:3 rotate
EOF
```

## 4. 准备代码

如果镜像里还没有代码，把项目放到固定路径：

```bash
mkdir -p /2023133163/liuf
cd /2023133163/liuf
git clone <your_dreamzero_repo_url> dreamzero
cd /2023133163/liuf/dreamzero
```

如果不能直接 git clone，可以从已有机器同步：

```bash
rsync -a --info=progress2 \
  --exclude checkpoints \
  --exclude outputs \
  /data/dreamzero_mot/ \
  <user>@<a800_node>:/2023133163/liuf/dreamzero/
```

## 5. 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
hash -r
uv --version
```

可选：配置 pip/uv 镜像源：

```bash
cat > /etc/pip.conf <<'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
EOF

mkdir -p /root/.config/uv
cat > /root/.config/uv/uv.toml <<'EOF'
index-url = "https://pypi.tuna.tsinghua.edu.cn/simple"
EOF
```

## 6. DreamZero server 环境

如果你的基础镜像已经来自 A800 训练环境，通常已经有：

```bash
/opt/venvs/dreamzero
```

先检查：

```bash
source /opt/venvs/dreamzero/bin/activate
which python
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
PY
```

如果没有这个 venv，可以按训练镜像文档安装，核心路径必须保持：

```bash
export VIRTUAL_ENV=/opt/venvs/dreamzero
export PATH=$VIRTUAL_ENV/bin:/usr/local/bin:/root/.local/bin:$PATH

uv venv --python 3.10 "$VIRTUAL_ENV"
source "$VIRTUAL_ENV/bin/activate"

cd /2023133163/liuf/dreamzero
uv pip install --upgrade pip setuptools wheel ninja packaging
uv pip install torch torchvision torchaudio
uv pip install -r docs/requirements_droid_wan22_uv.txt
uv pip install -e . --no-deps
```

实际集群上更推荐直接复用已经能跑 `scripts/train/a800_train/*.sh` 的固化镜像，不要临时重装 DreamZero server 环境。

## 7. RoboTwin client 环境

RoboTwin/SAPIEN 依赖和 DreamZero 训练依赖有版本冲突风险，所以单独建一个 Python 3.10 venv：

```bash
export ROBOTWIN_VENV=/opt/venvs/robotwin310
uv venv --python 3.10 "$ROBOTWIN_VENV"
source "$ROBOTWIN_VENV/bin/activate"

python --version
which python
```

进入项目并准备 RoboTwin：

```bash
cd /2023133163/liuf/dreamzero
mkdir -p third_party

if [ ! -d third_party/RoboTwin ]; then
  git clone https://github.com/RoboTwin-Platform/RoboTwin.git third_party/RoboTwin
fi

cd third_party/RoboTwin
git checkout 0aeea2d669c0f8516f4d5785f0aa33ba812c14b4
```

安装 RoboTwin 官方依赖：

```bash
cd /2023133163/liuf/dreamzero/third_party/RoboTwin
source /opt/venvs/robotwin310/bin/activate

uv pip install --upgrade pip setuptools wheel ninja packaging
pip install -r script/requirements.txt
```

官方脚本还会安装 `pytorch3d`、`curobo` 并 patch `sapien/mplib`。新环境建议直接跑：

```bash
cd /2023133163/liuf/dreamzero/third_party/RoboTwin
source /opt/venvs/robotwin310/bin/activate

bash script/_install.sh
```

说明：

- `_install.sh` 会 clone `envs/curobo`，如果之前已经 clone 过，可以先删除 `third_party/RoboTwin/envs/curobo` 再跑。
- `pytorch3d` 编译失败时，当前 DreamZero eval path 通常仍能跑，但最好在镜像制作阶段修好，避免后续官方工具报 warning。

## 8. 下载 RoboTwin assets

RoboTwin live env 必须有 assets：

```bash
cd /2023133163/liuf/dreamzero/third_party/RoboTwin
source /opt/venvs/robotwin310/bin/activate

bash script/_download_assets.sh
```

如果资产已经在另一台机器上下载好，也可以直接同步：

```bash
rsync -a --info=progress2 \
  /data/dreamzero_mot/third_party/RoboTwin/assets/ \
  <user>@<a800_node>:/2023133163/liuf/dreamzero/third_party/RoboTwin/assets/

cd /2023133163/liuf/dreamzero/third_party/RoboTwin
source /opt/venvs/robotwin310/bin/activate
python ./script/update_embodiment_config_path.py
```

检查关键目录：

```bash
ls third_party/RoboTwin/assets/embodiments
ls third_party/RoboTwin/assets/objects
```

## 9. 写入固定环境变量

把常用路径写进镜像：

```bash
cat > /etc/profile.d/dreamzero-robotwin-eval.sh <<'EOF'
export DREAMZERO_ROOT=/2023133163/liuf/dreamzero
export CHECKPOINT_ROOT=/2023133163/checkpoints/dreamzero
export DATASET_ROOT=/2023133163/datasets/dreamzero
export OUTPUT_ROOT=/2023133163/checkpoints/dreamzero/robotwin_eval_runs

export DREAMZERO_VENV=/opt/venvs/dreamzero
export ROBOTWIN_VENV=/opt/venvs/robotwin310
export PYTHON_BIN=/opt/venvs/dreamzero/bin/python

export LD_LIBRARY_PATH=/usr/lib64:/usr/lib:${LD_LIBRARY_PATH:-}
export NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
EOF

source /etc/profile.d/dreamzero-robotwin-eval.sh
```

注意：不要在同一个 shell 里同时 activate 两个 venv。server shell 使用 `/opt/venvs/dreamzero`，client shell 使用 `/opt/venvs/robotwin310`。

## 10. 环境自检

DreamZero server 环境：

```bash
cd /2023133163/liuf/dreamzero
source /opt/venvs/dreamzero/bin/activate

python - <<'PY'
import torch
import groot
import websockets
print("dreamzero server env ok")
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
PY
```

RoboTwin client 环境：

```bash
cd /2023133163/liuf/dreamzero
source /opt/venvs/robotwin310/bin/activate

python - <<'PY'
import sapien
import mplib
import gymnasium
import imageio
from example.robotwin.eval_polict_client_openpi import import_robotwin_runtime
import_robotwin_runtime()
print("robotwin client env ok")
PY
```

渲染 smoke：

```bash
cd /2023133163/liuf/dreamzero
source /opt/venvs/robotwin310/bin/activate

CUDA_VISIBLE_DEVICES=0 python example/robotwin/test_render.py
```

## 11. 保存 Docker 镜像

确认两个 venv 和 assets 都在容器内或挂载目录正确后保存镜像：

```bash
docker ps
docker commit dreamzero-robotwin-eval dreamzero-robotwin-eval:latest
docker save dreamzero-robotwin-eval:latest -o dreamzero-robotwin-eval.tar
```

到其它节点加载：

```bash
docker load -i dreamzero-robotwin-eval.tar
```

如果 `third_party/RoboTwin/assets` 放在挂载盘里，不会被 `docker commit` 保存；这通常是好事，但每台节点必须能访问同一路径或提前 rsync。

## 12. 单节点 eval smoke

启动 1 个 server。使用 DreamZero venv：

```bash
cd /2023133163/liuf/dreamzero
source /opt/venvs/dreamzero/bin/activate

CKPT=/2023133163/checkpoints/dreamzero/dreamzero_robotwin/checkpoint-30000 \
SERVER_GPU=0 \
START_PORT=29556 \
MASTER_PORT=29661 \
MAX_CHUNK_SIZE=24 \
OUTPUT_ROOT=/2023133163/checkpoints/dreamzero/robotwin_eval_runs/smoke_server_outputs \
bash example/robotwin/launch_server.sh
```

另开一个 shell，启动 1 个 client。使用 RoboTwin venv：

```bash
cd /2023133163/liuf/dreamzero
source /opt/venvs/robotwin310/bin/activate

CUDA_VISIBLE_DEVICES=0 \
HOST=127.0.0.1 \
PORT=29556 \
TEST_NUM=1 \
SEED=0 \
EPISODE_TIMEOUT_SEC=300 \
SAVE_COMPARISON_VIDEO=0 \
bash example/robotwin/launch_client.sh \
  /2023133163/checkpoints/dreamzero/robotwin_eval_runs/smoke_client \
  click_bell
```

预期：

- client 生成 `res.json`
- server log 里模型只加载一次
- 单 episode 开头有 policy reset
- 没有 timeout 时 `timeout=false`

## 13. 单节点 8 卡 smoke

server shell：

```bash
cd /2023133163/liuf/dreamzero
source /opt/venvs/dreamzero/bin/activate

NUM_GPUS=8 \
START_PORT=29556 \
MASTER_PORT=29661 \
CKPT=/2023133163/checkpoints/dreamzero/dreamzero_robotwin/checkpoint-30000 \
MAX_CHUNK_SIZE=24 \
OUTPUT_ROOT=/2023133163/checkpoints/dreamzero/robotwin_eval_runs/node0_server_outputs \
LOG_DIR=/2023133163/checkpoints/dreamzero/robotwin_eval_runs/node0_server_logs \
bash example/robotwin/launch_server_multigpus.sh
```

client shell：

```bash
cd /2023133163/liuf/dreamzero
source /opt/venvs/robotwin310/bin/activate

NUM_SERVERS=8 \
CLIENT_GPUS=0,1,2,3,4,5,6,7 \
START_PORT=29556 \
HOST=127.0.0.1 \
TASKS="click_alarmclock click_bell shake_bottle_horizontally open_laptop" \
TEST_NUM=1 \
EPISODE_TIMEOUT_SEC=300 \
SAVE_COMPARISON_VIDEO=0 \
LOG_DIR=/2023133163/checkpoints/dreamzero/robotwin_eval_runs/node0_client_logs \
bash example/robotwin/launch_client_multigpus.sh \
  /2023133163/checkpoints/dreamzero/robotwin_eval_runs/node0_client_outputs
```

## 14. 4 节点 / 8 节点 eval 建议

当前 `launch_server_multigpus.sh` 和 `launch_client_multigpus.sh` 是单节点脚本。多节点时每个节点都可以本地启动 8 个 server + 8 个 client：

```text
node0: slot 0-7
node1: slot 8-15
node2: slot 16-23
node3: slot 24-31
```

如果要跑：

```text
50 tasks * 50 episodes = 2500 jobs
```

建议后续使用 job-level sharding：

```text
job = (task, seed)
global_slot_id = node_rank * 8 + local_gpu_id
job_index % total_slots == global_slot_id
```

这样不会像“每轮同 task barrier”一样被某个 seed 长尾拖住。这个逻辑可以做成一个很薄的 multinode launcher，不需要改 server 架构。

## 15. 常见问题

### 15.1 server 和 client 用哪个 Python？

server：

```bash
source /opt/venvs/dreamzero/bin/activate
```

client：

```bash
source /opt/venvs/robotwin310/bin/activate
```

不要混用。DreamZero server 依赖和 RoboTwin/SAPIEN 依赖容易冲突。

### 15.2 为什么需要 `--network host`？

server/client 用 websocket port 通信。单节点本地连 `127.0.0.1` 时最简单；多容器或跨节点时也更容易排查端口问题。

### 15.3 `sapien` 或 OpenGL 报错怎么办？

先确认容器启动时有：

```bash
--gpus all
-e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
```

再确认系统库已安装：

```bash
ldconfig -p | grep -E 'libEGL|libGL|vulkan'
```

### 15.4 某个 episode 拖很久怎么办？

现在 client 支持：

```bash
EPISODE_TIMEOUT_SEC=900
```

超时后该 episode 记 fail 并继续，不会拖住几个小时。设为 `0` 可以关闭 timeout。

### 15.5 结果和 timing 在哪里？

每个 seed 会写：

```text
<save_root>/stseed-*/metrics/<task>/res.json
<save_root>/stseed-*/timings/<task>/episodes.jsonl
```

`episodes.jsonl` 里包含：

```text
T_expert_filter
T_env_reset
T_policy_infer
T_take_action
T_get_obs
episode_elapsed
infer_count
final_step
timeout
```

用它可以判断慢在 model infer，还是慢在 SAPIEN env step/get_obs。

