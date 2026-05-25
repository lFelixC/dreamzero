# DROID MoT Full-Video Denoising Attention Plan

本文档目标：用 DROID 数据解释 full-video MoT 中 video expert 给 action expert 提供了什么视觉信息，以及这些信息是否有因果作用。

## 核心问题

在每个 denoising step 的每一层，action expert 用 action token 的 Q 去 attend video K/V。要回答：

- action token 关注 video 的哪些时空区域？
- 这些关注随 denoise step 如何变化？
- 被关注的 video 信息是否**因果地**影响 action prediction？

## 前提

- checkpoint 必须是 `mot_action_video_attention=full_video` 训练出来的。
- 推理关闭 DiT 跳步：`NUM_DIT_STEPS=16`，`MOT_INFERENCE_VIDEO_MODE=denoise`。
- 数据：DROID held-out 50-100 条，覆盖 pick/place、drawer、wipe 等，成功/失败各半。
- 模型：Wan22 5B MoT，160x320，33 frames，3 views，latent grid 5x10 per frame。

## 实验一：Action-to-Video Attention 记录 + ROI 分析

**这是观察实验，回答"模型在看哪里"。**

### 记录内容

在 action expert mixed attention 中，对每个 denoise step 和选定 layer，记录 action query → video key 的 softmax 后 attention weight，按以下 ROI 聚合：

- **时间 ROI**：first frame / past frames / current block
- **视角 ROI**：left / right / wrist
- **语义 ROI**：gripper / object / target / background（用 Grounded-SAM 或手工标注 30-50 张的 bbox）

每条记录格式：

```json
{
  "episode_id": "...",
  "denoise_step": 0,
  "layer": 15,
  "action_token_group": "early",
  "mass_by_view": {"left": 0.0, "right": 0.0, "wrist": 0.0},
  "mass_by_time": {"first_frame": 0.0, "past": 0.0, "current_block": 0.0},
  "mass_by_roi": {"gripper": 0.0, "object": 0.0, "target": 0.0, "background": 0.0},
  "attention_entropy": 0.0
}
```

默认保存 head-mean、按 action token group（early/late）聚合。只对少量 case 保存完整 attention matrix 用于可视化 overlay。

### 分析维度

- 按 denoise step 画 `mass_by_roi` 和 `mass_by_time` 曲线，看 early→late step 关注是否从全局布局收敛到 gripper-object 局部。
- 按 layer group（early 0-7 / middle 8-21 / late 22-29）分组，看浅层是否更关注空间位置、深层是否更关注语义区域。
- 对少量样本做原图 attention overlay，定性验证 ROI 映射是否正确。

## 实验二：Causal Ablation（ROI Mask + K/V Replacement）

**这是干预实验，回答"模型看的那些信息真的有用吗"。**

### ROI mask ablation

在 action expert mixed attention 中，对 video K/V 按 ROI 做 mask（将对应 token 的 attention score 设为 -inf），比较 mask 前后的 action prediction：

- mask gripper / object / target / background K/V
- mask current block / past video K/V
- mask first frame K/V

度量指标：
- action chunk MSE 相对 baseline 的变化
- 如果 mask 高 attention ROI 后 action 误差显著增大，且 mask 低 attention ROI 影响很小 → 强因果证据

### K/V replacement

构造 donor/receiver episode pair（同 task 不同物体位置，或同场景不同 instruction），将 receiver 的某些 video K/V 替换为 donor 的 K/V：

- 替换所有层 vs 仅 early denoise steps vs 仅 middle layers
- 观察 action prediction 是否朝 donor 的目标/阶段偏移

这个实验最能区分 "video K/V 携带任务目标/物体位置" vs "只是背景先验"。

## 输出物

```
outputs/droid_mot_attention/<run_id>/
  config.json
  attention_roi_records.jsonl
  ablation_records.jsonl
  figures/
    denoise_step_roi_mass.png
    layer_step_heatmap.png
    ablation_bar_chart.png
    per_episode_attention_overlay/
  report.html
```

## 实现步骤

1. 确认 checkpoint、固定 seed、写 offline replay entrypoint。
2. 在 `_run_action_expert_block` 加可选 recorder（`DREAMZERO_RECORD_ACTION_VIDEO_ATTENTION=1` 时启用，softmax 后直接按 ROI 聚合，不保存全矩阵）。
3. 实现 token ROI mapper（frame_id、view_id、pixel_box、semantic mask → token grid soft mapping）。
4. Smoke test（3 episodes, 3 layers, 16 steps），确认 attention mass 归一化正确、overlay 位置正确。
5. 正式跑 50-100 episodes。
6. 跑 ROI mask ablation + K/V replacement。

## 结论标准

满足以下条件即可较有信心地解释 video→action 信息流：

1. action-to-video attention 在 denoise step/layer 上呈现稳定模式。
2. 高 attention ROI 与 gripper/object/target 等语义区域一致。
3. mask 高 attention ROI 显著改变 action prediction；mask 低 attention ROI 影响很小。
4. K/V replacement 将 action prediction 推向 donor 的目标/阶段。

---

## 评审意见

### 为什么只保留这两个实验

原始计划包含 5 个实验：Attention 记录、Token-ROI 映射、Denoise step 统计、Causal Ablation、Feature Probe。我保留了实验一（Attention 记录 + ROI）和实验四（Causal Ablation），把实验二的 token 映射并入实验一作为基础设施，删除了实验三和实验五。理由如下：

**实验三（Denoise step 统计）不是独立实验。** 它只是对实验一收集的数据按 step/layer 分组画图，属于分析环节而非独立实验。将其作为独立实验列出会稀释计划的 focus。

**实验五（Feature Probe）在这个场景下是弱证据。** Probe 的核心问题是：即使一个 linear probe 能从 video token 解码出 "gripper 位置"，也不能证明模型**实际使用了**这个信息来做 action prediction。Probe 容易产生 false positive（Belinkov 2022 等大量文献指出了这个问题）。相比之下，Causal Ablation 直接回答"去掉这个信息后 action 变差了吗"，是更强的因果证据。Probe 可以在 ablation 结果 inconclusive 时作为补充，但不应该放在核心计划里。

### 为什么实验一 + 实验四是"最强"组合

这是 mechanistic interpretability 的经典范式：**observe, then intervene**。

- **实验一（观察）**：告诉你模型在"看哪里"。没有这个，后面的 ablation 就是盲目的——你不知道该 mask 什么。
- **实验四（干预）**：告诉你"看的东西是否有因果作用"。没有这个，attention map 就只是相关性——高 attention 的区域不一定对 prediction 重要（例如模型可能在 attend background 做 positional reference，mask 掉反而没影响）。

两者结合构成了一个完整的论证链条：
1. 模型主要关注 gripper/object 区域（实验一）
2. Mask 这些区域后 action error 显著上升（实验二 ROI mask）
3. 替换 video K/V 后 action 朝 donor 偏移（实验二 K/V replacement）
4. → 结论：full-video MoT 中 video expert 向 action expert 提供了有因果作用的 gripper/object 时空信息

### 其他注意事项

- **Checkpoint 前提不可跳过**。如果 DROID checkpoint 是用 `first_frame` 训练的，临时 override 到 `full_video` 得到的 attention pattern 没有解释价值。务必先用 full-video config finetune 一版。
- **关闭 DiT 跳步是关键**。默认 16 step scheduler 只有 8 个真实 DiT step，中间 step 复用旧 flow 没有新 attention。不关跳步会导致一半 denoise step 数据缺失。
- **Semantic ROI 标注成本可控**。30-50 张手工 bbox 足够建立高可信的 ROI 映射，不需要全量自动标注。Grounded-SAM 可以作为 scale-up 选项但非必须。
- **先跑 smoke test 再 scale**。3 episode × 3 layer 的 smoke 花不了多少时间，但能提前发现 ROI 映射 bug、attention 归一化问题、recorder 对 inference 的干扰等，避免在大规模跑完后才发现数据不可用。
