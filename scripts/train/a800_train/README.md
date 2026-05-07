# A800 Multi-Node Training Scripts

这个目录只保留三个 A800 多机训练入口。每个脚本都是自包含的，可以直接运行，不依赖其它训练启动脚本或环境 helper。

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

## 首次进入容器检查

如果脚本是从 Windows/网页编辑器/平台同步过来的，先在每个容器里跑一次：

```bash
cd /2023133163/liuf/dreamzero

find scripts/train/a800_train -type f -name "*.sh" -exec sed -i 's/\r$//' {} \;
chmod +x scripts/train/a800_train/*.sh
```

第一行用于修复 CRLF 换行，否则可能出现 `/bin/bash^M: bad interpreter`。第二行给 shell 脚本加可执行权限，否则直接执行时可能出现 `Permission denied`。

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
