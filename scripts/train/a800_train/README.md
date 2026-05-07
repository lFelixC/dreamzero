# A800 2-Node Training Scripts

这个目录放 A800 集群上跑 MoT 对比实验的薄封装脚本。默认假设是 64 卡 A800 拆成 4 个实验组，每个实验占用 2 个节点，每个节点 8 张卡，也就是每个实验 16 卡。

当前已经提供四个实验入口：

| Script | Attention | Default port | Default output |
| --- | --- | --- | --- |
| `run_mot_first_frame_2node.sh` | `MOT_ACTION_VIDEO_ATTENTION=first_frame` | `29410` | `/data/checkpoints/dreamzero/dreamzero_droid_wan22_mot_a800_first_frame_2node` |
| `run_mot_full_video_2node.sh` | `MOT_ACTION_VIDEO_ATTENTION=full_video` | `29420` | `/data/checkpoints/dreamzero/dreamzero_droid_wan22_mot_a800_full_video_2node` |
| `run_mot_full_video_decoupled_2node.sh` | `full_video` + decoupled video/action noise | `29430` | `/data/checkpoints/dreamzero/dreamzero_droid_wan22_mot_a800_full_video_decoupled_2node` |
| `run_joint_drop_2node.sh` | `architecture=joint` + DROID exterior-view drop | `29440` | `/data/checkpoints/dreamzero/dreamzero_droid_wan22_joint_drop_a800_2node` |

三个 MoT 脚本会调用上一层的 `../run_dreamzero_mot_multinode.sh`。`run_joint_drop_2node.sh` 是 joint baseline 的独立入口，默认保持 joint 训练设置，并额外开启 DROID exterior-view drop。

## 环境初始化

非 MPI 的 `*_2node.sh` 现在会自动读取 `DREAMZERO_PROFILE`，默认是 `/etc/profile.d/dreamzero-uv.sh`。因此 Docker 镜像里配置的 `VIRTUAL_ENV`、`PYTHON_BIN`、`DREAMZERO_ROOT`、`DATASET_ROOT`、`CHECKPOINT_ROOT`、`PYTHONPATH`、CUDA 路径和 pip/uv 镜像都会自动生效。

手动两机启动时，外面通常只需要设置拓扑变量：

```bash
MASTER_ADDR=<rank0_ip> \
NODE_RANK=0 \
NNODES=2 \
scripts/train/a800_train/run_mot_first_frame_2node.sh
```

第二台机器把 `NODE_RANK=1` 即可。`CUDA_VISIBLE_DEVICES` 默认是 `0,1,2,3,4,5,6,7`，只在需要换卡组时覆盖。

## First Frame 实验

在同一个实验的两个节点上分别执行，`MASTER_ADDR` 必须填写 rank0 节点能被另一个节点访问到的 IP。

Rank 0 节点：

```bash
cd /data/dreamzero_mot

MASTER_ADDR=<first_frame_rank0_ip> \
NODE_RANK=0 \
NNODES=2 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
scripts/train/a800_train/run_mot_first_frame_2node.sh
```

Rank 1 节点：

```bash
cd /data/dreamzero_mot

MASTER_ADDR=<first_frame_rank0_ip> \
NODE_RANK=1 \
NNODES=2 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
scripts/train/a800_train/run_mot_first_frame_2node.sh
```

## Full Video 实验

Rank 0 节点：

```bash
cd /data/dreamzero_mot

MASTER_ADDR=<full_video_rank0_ip> \
NODE_RANK=0 \
NNODES=2 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
scripts/train/a800_train/run_mot_full_video_2node.sh
```

Rank 1 节点：

```bash
cd /data/dreamzero_mot

MASTER_ADDR=<full_video_rank0_ip> \
NODE_RANK=1 \
NNODES=2 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
scripts/train/a800_train/run_mot_full_video_2node.sh
```

## Full Video Decoupled 实验

这个实验固定：

```bash
MOT_ACTION_VIDEO_ATTENTION=full_video
MOT_DECOUPLE_VIDEO_ACTION_NOISE=true
MOT_INFERENCE_VIDEO_MODE=auto
```

`auto` 推理时会使用 `decoupled_denoise`：video 按 refresh mask 跳步刷新，action 每一步都重新 denoise。

Rank 0 节点：

```bash
cd /data/dreamzero_mot

MASTER_ADDR=<full_video_decoupled_rank0_ip> \
NODE_RANK=0 \
NNODES=2 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
scripts/train/a800_train/run_mot_full_video_decoupled_2node.sh
```

Rank 1 节点：

```bash
cd /data/dreamzero_mot

MASTER_ADDR=<full_video_decoupled_rank0_ip> \
NODE_RANK=1 \
NNODES=2 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
scripts/train/a800_train/run_mot_full_video_decoupled_2node.sh
```

## Joint Drop 实验

这个实验固定：

```bash
architecture=joint
model/dreamzero/action_head=wan_flow_matching_action_tf_wan22
DROID_RANDOM_DROP_EXTERIOR_VIEW_PROB=0.5
```

`0.5` 表示 50% 的训练样本随机把 left/right exterior 其中一个置黑；如果要每个训练样本都 drop 一个 exterior view，可以覆盖成 `DROID_RANDOM_DROP_EXTERIOR_VIEW_PROB=1.0`。

Rank 0 节点：

```bash
cd /data/dreamzero_mot

MASTER_ADDR=<joint_drop_rank0_ip> \
NODE_RANK=0 \
NNODES=2 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
scripts/train/a800_train/run_joint_drop_2node.sh
```

Rank 1 节点：

```bash
cd /data/dreamzero_mot

MASTER_ADDR=<joint_drop_rank0_ip> \
NODE_RANK=1 \
NNODES=2 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
scripts/train/a800_train/run_joint_drop_2node.sh
```

## 常用覆盖项

这些环境变量可以在命令前覆盖，不需要改脚本：

```bash
PER_DEVICE_BS=64 \
GLOBAL_BATCH_SIZE=1024 \
MAX_STEPS=30000 \
SAVE_STEPS=5000 \
DEEPSPEED_CFG=zero2 \
DATASET_ROOT=/data/datasets/dreamzero/droid_lerobot \
CHECKPOINT_ROOT=/data/checkpoints/dreamzero \
WAN22_CKPT_DIR=/data/checkpoints/dreamzero/Wan2.2-TI2V-5B \
OUTPUT_DIR=/data/checkpoints/dreamzero/my_mot_exp \
scripts/train/a800_train/run_mot_first_frame_2node.sh
```

decoupled 实验常用覆盖项：

```bash
MOT_VIDEO_NOISE_BETA_ALPHA=5.0 \
MOT_DECOUPLED_INFERENCE_VIDEO_FINAL_NOISE=0.85 \
MOT_DECOUPLED_INFERENCE_VIDEO_REFRESH_STEPS=6 \
scripts/train/a800_train/run_mot_full_video_decoupled_2node.sh
```

随机 drop DROID exterior view 的训练增强在三个 MoT 脚本里默认关闭，在 `run_joint_drop_2node.sh` 里默认 `0.5`。需要给其它脚本开启时设置概率，`0.5` 表示 50% 的训练样本随机把 left/right exterior 其中一个置黑，`1.0` 表示每个训练样本都 drop 一个 exterior view：

```bash
DROID_RANDOM_DROP_EXTERIOR_VIEW_PROB=0.5 \
scripts/train/a800_train/run_mot_full_video_2node.sh
```

如果多个实验在同一批机器上并跑，请确保每个实验使用不同的 `MASTER_PORT`。当前四个脚本默认已经错开端口，后续第五个实验可以继续用 `29450` 这类端口。

## 单机多卡分组

如果你说的“两组卡”不是两个节点，而是在单机上拆 GPU 组，可以把 `NNODES=1`，并设置不同的 `CUDA_VISIBLE_DEVICES` 和 `MASTER_PORT`。

示例：

```bash
cd /data/dreamzero_mot

NNODES=1 \
NODE_RANK=0 \
MASTER_ADDR=127.0.0.1 \
MASTER_PORT=29410 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
scripts/train/a800_train/run_mot_first_frame_2node.sh
```

## 推理模式说明

训练时通常不用关心 `MOT_INFERENCE_VIDEO_MODE`，脚本默认写成 `auto`。真正挂 server 推理时，可以再通过环境变量覆盖成 `cache_only`、`denoise` 或 `decoupled_denoise`。

`first_frame` 训练出来的模型在 `auto` 推理下会优先走 `cache_only`，也就是 action 直接读首帧 video K/V，不完整去噪 future video。

`full_video` 训练出来的模型在 `auto` 推理下会走 `denoise`，因为 action 需要 full video K/V。

`full_video + MOT_DECOUPLE_VIDEO_ACTION_NOISE=true` 训练出来的模型在 `auto` 推理下会走 `decoupled_denoise`，video 跳步刷新 K/V，action 每步 denoise。
