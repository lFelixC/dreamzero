# ALOHA Dataset to DreamZero

这份文档说明：如果你手上有一批新的 ALOHA X5lite 双臂数据，如何把它转换成 DreamZero 可直接训练的数据集。

本文档默认你想复用当前仓库里已经注册好的 embodiment 和训练配置：

- `embodiment_tag = aloha_x5lite_bimanual`
- 数据配置：`dreamzero/aloha_x5lite_bimanual_relative`
- 训练脚本：`scripts/train/aloha_x5lite_bimanual_training_local.sh`

如果你的新数据仍然是同一种机器人语义：

- 双臂 14D `state`
- 双臂 14D `action`
- 三路相机：`cam_high / cam_left / cam_right`

那么你通常不需要再改 DreamZero 代码，只需要按下面步骤处理数据。

---

## 1. 目标格式

DreamZero 最终需要的是一个标准 LeRobot root，并且 `meta/` 下面补齐 DreamZero 训练需要的 metadata：

```text
your_dataset/
├── data/
├── videos/
└── meta/
    ├── info.json
    ├── tasks.jsonl
    ├── episodes.jsonl
    ├── episodes_stats.jsonl
    ├── modality.json
    ├── embodiment.json
    ├── stats.json
    └── relative_stats_dreamzero.json
```

其中：

- `info.json / tasks.jsonl / episodes.jsonl / episodes_stats.jsonl` 是 LeRobot 自带元数据
- `modality.json / embodiment.json / stats.json / relative_stats_dreamzero.json` 是 DreamZero 训练额外需要的元数据

---

## 2. 先明确当前 ALOHA 约定

### 2.1 原始 parquet 不改左右语义

当前这条 ALOHA 路线里，merged parquet 继续保留原始 14D packed vector，不在 parquet 里重排左右臂。

也就是说：

- `observation.state` 还是原始 14D
- `action` 还是原始 14D
- 左右臂顺序不在 parquet 里改写

### 2.2 DreamZero 暴露成 left-first 四个子键

DreamZero 训练时，通过 `modality.json` 把 packed vector 映射成 canonical 的四个子键：

| DreamZero key | original_key | start:end | 含义 |
|---|---|---:|---|
| `state.left_joint_pos` | `observation.state` | `7:13` | 左臂 6 维关节 |
| `state.left_gripper_pos` | `observation.state` | `13:14` | 左夹爪 1 维 |
| `state.right_joint_pos` | `observation.state` | `0:6` | 右臂 6 维关节 |
| `state.right_gripper_pos` | `observation.state` | `6:7` | 右夹爪 1 维 |
| `action.left_joint_pos` | `action` | `7:13` | 左臂动作 |
| `action.left_gripper_pos` | `action` | `13:14` | 左夹爪动作 |
| `action.right_joint_pos` | `action` | `0:6` | 右臂动作 |
| `action.right_gripper_pos` | `action` | `6:7` | 右夹爪动作 |

### 2.3 视频和语言键映射

| DreamZero key | original_key |
|---|---|
| `video.cam_high` | `observation.images.cam_high` |
| `video.cam_left` | `observation.images.cam_left` |
| `video.cam_right` | `observation.images.cam_right` |
| `annotation.task` | `task_index` |

这里 `annotation.task <- task_index` 的意思不是把任务文本写进 parquet，而是：

- parquet 里保留 `task_index`
- `tasks.jsonl` 里保留 `task_index -> task text`
- `modality.json` 里把 `annotation.task` 映射到 `task_index`

---

## 3. action 要不要提前转成 relative

不要。

这条 ALOHA 路线下，raw dataset 里的 `action` 应该保留为绝对动作，不要在 parquet 里提前改成 `delta` 或 `relative action`。

原因是当前 DreamZero 数据配置已经写了：

- `relative_action: true`
- `relative_action_keys: [left_joint_pos, left_gripper_pos, right_joint_pos, right_gripper_pos]`

这表示 DreamZero 会在加载样本时，按 `action - reference_state` 在线转成 relative action。

### 正确做法

- parquet 里保存 absolute action
- `convert_lerobot_to_gear.py` 只负责生成 metadata 和 relative stats
- 训练时由 DreamZero loader 做 relative conversion

### 不正确做法

- 先把 parquet 里的 `action` 改成 delta
- 同时训练配置里还保留 `relative_action: true`

这样会等于做两次 relative，语义会错。

### 一句话结论

如果你的新 ALOHA 数据集里的 `action` 仍然是绝对关节/夹爪目标，就直接沿用本文档。

如果你的新数据集里的 `action` 已经是 delta / relative action，那么不要直接用当前这套 `aloha_x5lite_bimanual_relative` 配置，必须单独改配置。

---

## 4. 路径一：原始数据还是多子任务 LeRobot，先合并再转换

如果你的输入目录类似这样：

```text
/data/datasets/dreamzero/my_new_aloha_dataset/
├── task_a/
│   └── lerobot_data/
├── task_b/
│   └── lerobot_data/
└── ...
```

那么先跑 merge/build 脚本，把多个子任务合成一个大 LeRobot root。

### 4.1 合并并重写原始数据

这一步会做真正需要改原始数据的部分：

- 多子任务合并成一个大 LeRobot
- `50 fps -> 30 fps` 重采样
- 视频转成 `H.264 mp4`
- 重建 `timestamp / frame_index / index / episode_index / task_index`
- 裁剪 gripper 到 `[0, 1]`

命令：

```bash
cd /data/dreamzero

/data/openpi/.venv/bin/python scripts/data/build_aloha_x5lite_bimanual_lerobot.py \
  --input-root /data/datasets/dreamzero/my_new_aloha_dataset \
  --output-root /data/datasets/dreamzero/my_new_aloha_dataset_30fps \
  --target-fps 30 \
  --video-codec h264 \
  --force
```

如果你只想合并部分 task，可以显式指定：

```bash
cd /data/dreamzero

/data/openpi/.venv/bin/python scripts/data/build_aloha_x5lite_bimanual_lerobot.py \
  --input-root /data/datasets/dreamzero/my_new_aloha_dataset \
  --output-root /data/datasets/dreamzero/my_new_aloha_dataset_30fps \
  --task-dirs task_a task_b task_c \
  --target-fps 30 \
  --video-codec h264 \
  --force
```

这一步完成后，你会得到一个标准单根 LeRobot 数据集，例如：

```text
/data/datasets/dreamzero/my_new_aloha_dataset_30fps
```

注意：这一步还没有生成 `modality.json` 等 DreamZero metadata。

---

## 5. 路径二：如果你已经有单个 LeRobot root，可直接跳到 convert

如果你的数据已经满足下面条件：

- 已经是单个标准 LeRobot root
- fps 已经是你想要的真实训练 fps
- 视频已经能被 `decord` 正常读取
- gripper 数值已经是 `[0, 1]`

那么可以跳过 build 脚本，直接做 metadata 转换。

---

## 6. 转成 DreamZero 可训练的数据集

不管你是从第 4 步 merge 出来的，还是原本就有单根 LeRobot root，这一步都要跑。

### 6.1 运行 convert_lerobot_to_gear

命令：

```bash
cd /data/dreamzero

/data/dreamzero/.venv/bin/python scripts/data/convert_lerobot_to_gear.py \
  --dataset-path /data/datasets/dreamzero/my_new_aloha_dataset_30fps \
  --embodiment-tag aloha_x5lite_bimanual \
  --state-keys '{"left_joint_pos":[7,13],"left_gripper_pos":[13,14],"right_joint_pos":[0,6],"right_gripper_pos":[6,7]}' \
  --action-keys '{"left_joint_pos":[7,13],"left_gripper_pos":[13,14],"right_joint_pos":[0,6],"right_gripper_pos":[6,7]}' \
  --relative-action-keys left_joint_pos left_gripper_pos right_joint_pos right_gripper_pos \
  --task-key task_index \
  --task-alias task \
  --force
```

### 6.2 这条命令做了什么

- 生成 `meta/modality.json`
- 生成 `meta/embodiment.json`
- 生成 `meta/stats.json`
- 生成 `meta/relative_stats_dreamzero.json`
- 保留和复用现有 `tasks.jsonl / episodes.jsonl`
- 不改 parquet
- 不改视频

### 6.3 `--task-key` 和 `--task-alias`

这里要特别注意：

- `--task-key task_index`
- `--task-alias task`

它们组合起来会在 `modality.json` 里写出：

```json
{
  "annotation": {
    "task": {
      "original_key": "task_index"
    }
  }
}
```

这就是当前 ALOHA 路线需要的 `annotation.task` 映射方式。

---

## 7. 转换完成后应该检查什么

执行：

```bash
ls /data/datasets/dreamzero/my_new_aloha_dataset_30fps/meta
```

你应该至少看到：

```text
embodiment.json
episodes.jsonl
episodes_stats.jsonl
info.json
modality.json
relative_stats_dreamzero.json
stats.json
tasks.jsonl
```

还可以快速检查：

```bash
python - <<'PY'
import json
from pathlib import Path

meta = Path("/data/datasets/dreamzero/my_new_aloha_dataset_30fps/meta")

for name in [
    "info.json",
    "modality.json",
    "embodiment.json",
    "stats.json",
    "relative_stats_dreamzero.json",
]:
    p = meta / name
    print(name, p.exists())

info = json.loads((meta / "info.json").read_text())
print("fps =", info["fps"])
print("total_episodes =", info["total_episodes"])
print("total_tasks =", info["total_tasks"])
PY
```

---

## 8. 训练命令

当前仓库里已经有一份正式训练脚本：

```text
/data/dreamzero/scripts/train/aloha_x5lite_bimanual_training_local.sh
```

这个脚本默认使用：

- `data=dreamzero/aloha_x5lite_bimanual_relative`
- `embodiment_tag=aloha_x5lite_bimanual`
- `pretrained_model_path=/data/checkpoints/dreamzero/DreamZero-AgiBot`
- `dit_version=/data/checkpoints/dreamzero/Wan2.1-I2V-14B-480P`
- `CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"`
- `PER_DEVICE_BS=32`

直接启动：

```bash
cd /data/dreamzero

bash scripts/train/aloha_x5lite_bimanual_training_local.sh
```

如果 8 卡 `PER_DEVICE_BS=32` 会爆显存，就把脚本顶部改成：

```bash
PER_DEVICE_BS=16
```

脚本会自动重新计算：

```bash
GLOBAL_BATCH_SIZE = NUM_GPUS * PER_DEVICE_BS
```

---

## 9. 从新数据集到训练的最短命令清单

### 9.1 如果输入是多子任务 ALOHA

```bash
cd /data/dreamzero

/data/openpi/.venv/bin/python scripts/data/build_aloha_x5lite_bimanual_lerobot.py \
  --input-root /data/datasets/dreamzero/my_new_aloha_dataset \
  --output-root /data/datasets/dreamzero/my_new_aloha_dataset_30fps \
  --target-fps 30 \
  --video-codec h264 \
  --force

/data/dreamzero/.venv/bin/python scripts/data/convert_lerobot_to_gear.py \
  --dataset-path /data/datasets/dreamzero/my_new_aloha_dataset_30fps \
  --embodiment-tag aloha_x5lite_bimanual \
  --state-keys '{"left_joint_pos":[7,13],"left_gripper_pos":[13,14],"right_joint_pos":[0,6],"right_gripper_pos":[6,7]}' \
  --action-keys '{"left_joint_pos":[7,13],"left_gripper_pos":[13,14],"right_joint_pos":[0,6],"right_gripper_pos":[6,7]}' \
  --relative-action-keys left_joint_pos left_gripper_pos right_joint_pos right_gripper_pos \
  --task-key task_index \
  --task-alias task \
  --force

bash scripts/train/aloha_x5lite_bimanual_training_local.sh
```

### 9.2 如果输入已经是单个 LeRobot root

```bash
cd /data/dreamzero

/data/dreamzero/.venv/bin/python scripts/data/convert_lerobot_to_gear.py \
  --dataset-path /data/datasets/dreamzero/my_new_aloha_dataset_30fps \
  --embodiment-tag aloha_x5lite_bimanual \
  --state-keys '{"left_joint_pos":[7,13],"left_gripper_pos":[13,14],"right_joint_pos":[0,6],"right_gripper_pos":[6,7]}' \
  --action-keys '{"left_joint_pos":[7,13],"left_gripper_pos":[13,14],"right_joint_pos":[0,6],"right_gripper_pos":[6,7]}' \
  --relative-action-keys left_joint_pos left_gripper_pos right_joint_pos right_gripper_pos \
  --task-key task_index \
  --task-alias task \
  --force

bash scripts/train/aloha_x5lite_bimanual_training_local.sh
```

---

## 10. 什么时候需要改代码，而不是只换路径

如果你的新数据和当前 ALOHA 约定不一致，就不能直接照抄本文档。

常见需要改代码的情况：

- 不是 `cam_high / cam_left / cam_right` 这三个相机名
- `state` / `action` 不是 14D packed vector
- 左右臂顺序不是当前的 right-first 原始布局
- `task_index` 不存在，语言键来自别的列
- `action` 已经是 relative / delta，不是 absolute action

这几种情况下，至少要同步修改：

- `convert_lerobot_to_gear.py` 的映射命令
- `groot/vla/configs/data/dreamzero/base_48_wan_fine_aug_relative.yaml`
- `groot/vla/configs/data/dreamzero/aloha_x5lite_bimanual_relative.yaml`

如果 embodiment 已经不是当前这条 ALOHA 形态，还需要再改：

- `groot/vla/data/schema/embodiment_tags.py`
- `groot/vla/configs/model/dreamzero/transform/base.yaml`
- `groot/vla/model/dreamzero/transform/dreamzero_cotrain.py`

---

## 11. 当前仓库中和这条流程相关的文件

- `scripts/data/build_aloha_x5lite_bimanual_lerobot.py`
- `scripts/data/convert_lerobot_to_gear.py`
- `groot/vla/configs/data/dreamzero/aloha_x5lite_bimanual_relative.yaml`
- `groot/vla/configs/data/dreamzero/base_48_wan_fine_aug_relative.yaml`
- `groot/vla/configs/model/dreamzero/transform/base.yaml`
- `scripts/train/aloha_x5lite_bimanual_training_local.sh`

