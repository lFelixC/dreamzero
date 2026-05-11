# A800 Multi-Node Training Scripts

这个目录保留 A800 多机训练入口。所有脚本都是自包含的，可以直接运行，不依赖其它训练启动脚本或环境 helper。

默认环境固定为：

```bash
VIRTUAL_ENV=/opt/venvs/dreamzero
PYTHON_BIN=/opt/venvs/dreamzero/bin/python
DREAMZERO_ROOT=/2023133163/liuf/dreamzero
DATASET_ROOT=/2023133163/datasets/dreamzero
CHECKPOINT_ROOT=/2023133163/checkpoints/dreamzero
```

这些变量仍然可以在命令前覆盖。脚本会自动设置 `PATH` 和 `PYTHONPATH`。

## 入口脚本

| Script | Experiment | Default port | Default output |
| --- | --- | --- | --- |
| `run_mot_full_video_2node.sh` | MoT full-video attention | `29420` | `${CHECKPOINT_ROOT}/dreamzero_droid_wan22_mot_a800_full_video_2node` |
| `run_mot_full_video_decoupled_2node.sh` | MoT full-video attention + decoupled video/action noise | `29430` | `${CHECKPOINT_ROOT}/dreamzero_droid_wan22_mot_a800_full_video_decoupled_2node` |
| `run_joint_drop_2node.sh` | joint baseline + DROID exterior-view drop | `29440` | `${CHECKPOINT_ROOT}/dreamzero_droid_wan22_joint_drop_a800_2node` |
| `run_robotwin_joint_cost_group_2node.sh` | RoboTwin joint; infra knobs default off, cost-group can be enabled explicitly | `29444` | `${CHECKPOINT_ROOT}/dreamzero_robotwin_wan22_joint_a800_2node` |
| `run_robotwin_joint_no_infra_2node.sh` | RoboTwin joint baseline with runtime infra knobs disabled | `29445` | `${CHECKPOINT_ROOT}/dreamzero_robotwin_wan22_joint_no_infra_a800_2node` |

## 首次进入容器检查

如果脚本是从 Windows/网页编辑器/平台同步过来的，先在每个容器里跑一次：

```bash
cd /2023133163/liuf/dreamzero

find scripts/train/a800_train -type f \( -name "*.sh" -o -name "*.py" \) -exec sed -i 's/\r$//' {} +

find scripts/train/a800_train -type f \( -name "*.sh" -o -name "*.py" \) -exec chmod +x {} +
```

第一行用于修复 CRLF 换行，否则可能出现 `/bin/bash^M: bad interpreter` 或 `set: pipefail` 这类异常。第二行给脚本加可执行权限，否则直接执行时可能出现 `Permission denied`。

## 配置公网 DNS

如果容器无法解析公网域名，例如 `pypi.org`、`huggingface.co`、`swanlab.cn`，可以在容器启动后临时改 `/etc/resolv.conf`：

```bash
cp /etc/resolv.conf /etc/resolv.conf.bak.$(date +%s) 2>/dev/null || true

cat > /etc/resolv.conf <<'EOF'
nameserver 223.5.5.5
nameserver 114.114.114.114
nameserver 8.8.8.8
options timeout:2 attempts:3 rotate
EOF
```

检查 DNS 是否生效：

```bash
getent hosts pypi.org
getent hosts huggingface.co
getent hosts swanlab.cn
```

检查是否真的能访问外网：

```bash
python - <<'PY'
import socket
import urllib.request

for host in ["pypi.org", "huggingface.co", "swanlab.cn"]:
    print(host, socket.gethostbyname(host))

for url in ["https://pypi.org/simple/pip/", "https://swanlab.cn"]:
    with urllib.request.urlopen(url, timeout=10) as resp:
        print(url, resp.status)
PY
```

如果 `getent hosts` 有输出，但 `urllib` 访问超时，说明 DNS 已经好了，问题是平台没有放开外网出口或需要 HTTP/HTTPS 代理。此时需要在任务环境里配置代理，例如：

```bash
export HTTP_PROXY=http://<proxy_host>:<proxy_port>
export HTTPS_PROXY=http://<proxy_host>:<proxy_port>
export NO_PROXY=localhost,127.0.0.1
```

注意：公网 DNS 只负责解析公网域名，通常不能解析 AI Station/Kubernetes 内部的 `worker-*` 主机名。多机训练的 `MASTER_ADDR` 仍然建议使用 rank0 节点 IP，或者通过平台内部 DNS/`/etc/hosts` 解决。

## 两机启动

在两台机器上使用相同的 `MASTER_ADDR`、`MASTER_PORT` 和 `NNODES`，只改变 `NODE_RANK`。`MASTER_ADDR` 必须是 rank0 节点能被其它节点访问到的 IP。

Rank 0：

```bash
cd /2023133163/liuf/dreamzero

MASTER_ADDR=<rank0_ip> \
NODE_RANK=0 \
NNODES=2 \
bash scripts/train/a800_train/run_mot_full_video_2node.sh
```

Rank 1：

```bash
cd /2023133163/liuf/dreamzero

MASTER_ADDR=<rank0_ip> \
NODE_RANK=1 \
NNODES=2 \
bash scripts/train/a800_train/run_mot_full_video_2node.sh
```

更换实验时只需要换脚本名：

```bash
bash scripts/train/a800_train/run_mot_full_video_decoupled_2node.sh
bash scripts/train/a800_train/run_joint_drop_2node.sh
```

## 常用覆盖项

默认每个节点使用 8 张卡：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

常用训练参数可以在命令前覆盖：

```bash
MASTER_ADDR=<rank0_ip> \
NODE_RANK=0 \
NNODES=2 \
PER_DEVICE_BS=16 \
MAX_STEPS=60000 \
SAVE_STEPS=5000 \
DEEPSPEED_CFG=zero2 \
DATASET_ROOT=/2023133163/datasets/dreamzero \
CHECKPOINT_ROOT=/2023133163/checkpoints/dreamzero \
OUTPUT_DIR=/2023133163/checkpoints/dreamzero/my_exp \
bash scripts/train/a800_train/run_mot_full_video_2node.sh
```

如果多个实验并跑，请确保每个实验使用不同的 `MASTER_PORT`。三个脚本默认端口已经错开。

## 实验参数

Decoupled 脚本可覆盖：

```bash
MOT_VIDEO_NOISE_BETA_ALPHA=5.0 \
MOT_VIDEO_NOISE_BETA_BETA=1.0 \
MOT_DECOUPLED_INFERENCE_VIDEO_FINAL_NOISE=0.85 \
MOT_DECOUPLED_INFERENCE_VIDEO_REFRESH_STEPS=6 \
bash scripts/train/a800_train/run_mot_full_video_decoupled_2node.sh
```

Joint-drop 脚本默认：

```bash
DROID_RANDOM_DROP_EXTERIOR_VIEW_PROB=0.15
```

需要调整时：

```bash
DROID_RANDOM_DROP_EXTERIOR_VIEW_PROB=1.0 \
bash scripts/train/a800_train/run_joint_drop_2node.sh
```

三个脚本都会在 rank0 准备共享模型资产，其它节点等待资产出现。已经提前准备好模型时，可以设置：

```bash
PREPARE_ASSETS=false
```

## SwanLab 上传

训练脚本默认关闭 SwanLab 同步：

```bash
SWANLAB_SYNC_WANDB=0
WANDB_MODE=offline
```

训练完成后，如果需要上传 Trainer 的 scalar history，在能访问 SwanLab 的机器上执行：

```bash
cd /2023133163/liuf/dreamzero

python scripts/train/a800_train/upload_trainer_state_to_swanlab.py \
  /2023133163/checkpoints/dreamzero/dreamzero_droid_wan22_joint_drop_a800_2node \
  --project dreamzero \
  --experiment-name dreamzero_droid_wan22_joint_drop_a800_2node
```

脚本会自动读取 `OUTPUT_DIR/trainer_state.json` 或最新 `checkpoint-*/trainer_state.json`。如果确实需要训练时实时同步，可以在启动命令前覆盖：

```bash
SWANLAB_SYNC_WANDB=1
```
