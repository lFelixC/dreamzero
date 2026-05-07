# AGENTS.md

## 目的

这份文件记录 DreamZero 在本仓库中的算法结构和关键不变量。后续 agent 在修改代码前，应先用这里的算法地图理解：数据如何进入模型、模型在预测什么、action 和 video 如何对齐、训练与推理共享哪些隐式契约。

## 算法总览

DreamZero 在这里实现的是 World Action Model，而不是纯 action-only policy。

核心思想：

- 模型同时学习 world dynamics 和 robot policy。
- 输入是多视角视频、语言、当前 state、未来 action chunk。
- 视频经 VAE 进入 latent space，action/state 被编码成 DiT 里的 action-state register。
- DiT 对视频 latent 和 action register 做联合 flow matching / denoising。
- 训练时同时预测 video flow/noise 和 action flow/noise。
- 推理时给定真实观测帧和 state，模型在因果 KV cache 上逐块预测下一个 action chunk，并可同时生成 video latent 作为 world prediction。

高层路径：

```text
LeRobot + DreamZero meta
  -> modality transforms
  -> multi-view video grid + normalized state/action + language tokens
  -> VLA
  -> text/image/video conditioning
  -> CausalWanModel DiT with action-state register
  -> dynamics loss + action loss during training
  -> causal chunked action prediction during inference
```

## 数据表示

DreamZero 的数据契约由 `meta/`、Hydra data config 和 transform pipeline 共同定义。不要只看 tensor shape；modality key、view order、normalization 统计和 action horizon 同样是算法的一部分。

关键文件：

- `groot/vla/data/schema/`
- `groot/vla/data/dataset/`
- `groot/vla/data/transform/`
- `groot/vla/model/dreamzero/transform/dreamzero_cotrain.py`
- `groot/vla/configs/data/dreamzero/*.yaml`

### Modality Contract

每个 embodiment 的 dataset root 需要有 DreamZero metadata：

- `meta/modality.json` 定义 state/action/video/annotation 的 key 和 index range。
- `meta/stats.json` 提供普通 state/action normalization 统计。
- `meta/relative_stats_dreamzero.json` 提供 relative action 统计。
- `meta/embodiment.json` 给出 embodiment tag。

Hydra YAML 里的 `modality_keys` 必须和 `modality.json` 中的 key 精确匹配。例如 DROID 常见 key 是：

- video: `video.exterior_image_1_left`, `video.exterior_image_2_left`, `video.wrist_image_left`
- state: `state.joint_position`, `state.gripper_position`
- action: `action.joint_position`, `action.gripper_position`
- language: annotation language keys

### Multi-View Video Layout

`ConcatTransform` 先把多个 `video.*` key 按 config 顺序拼成 `video`，随后 `DreamTransform` 把多视角视频排成单张 grid image，再作为 Wan video model 的输入。

DROID / `EmbodimentTag.OXE_DROID` 的布局特殊：

```text
[ wrist view stretched across top row ]
[ left exterior | right exterior       ]
```

其他多视角 embodiment 通常使用 2x2 grid：

```text
[ view 0 | view 2 ]
[ view 1 | black  ]
```

这个布局会进入语言 prompt 中的 view description，也会影响视觉 token 的空间语义。修改 camera order、grid layout 或 prompt view description，会改变模型学到的跨视角对应关系。

### State / Action Representation

state 和 action 在 transform 中按固定 key order concat，并 pad 到模型配置的最大维度：

- `state` shape 语义是当前或短历史 proprio token。
- `action` shape 语义是未来 action chunk。
- `state_mask` 和 `action_mask` 标记真实维度，避免 padded dim 参与 loss。
- `has_real_action` 控制样本是否贡献 action loss。
- `embodiment_id` 表示 embodiment tag 映射后的整数 id，用于 action/state 编码路径。

数值归一化通常使用 `q99`，范围被裁剪到 `[-1, 1]`。action 进入 diffusion 前必须已经在该归一化空间内；推理输出也先在归一化空间，再由 policy wrapper unnormalize 回机器人动作空间。

### Language Processing

语言不是裸 task string。`DreamTransform` 和 collator 会按 embodiment 注入 view layout 描述，例如“multi-view video shows...”和每个 view 对应的相机含义。

这意味着 language prompt 同时承担两个角色：

- task instruction。
- 视觉 grid 语义说明。

如果改 view layout，必须同步改 language formalization，否则模型会收到互相矛盾的视觉语义。

## 模型结构

高层模型是 `VLA = backbone + action_head`，核心在 action head。

关键文件：

- `groot/vla/model/dreamzero/base_vla.py`
- `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py`
- `groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py`
- `groot/vla/model/dreamzero/modules/flow_match_scheduler.py`
- `groot/vla/model/dreamzero/modules/flow_unipc_multistep_scheduler.py`
- `groot/vla/model/n1_5/sim_policy.py`

### VLA Boundary

`base_vla.py` 只做模型级调度：

```text
inputs
  -> backbone.prepare_input(...)
  -> action_head.prepare_input(...)
  -> backbone(...)
  -> action_head(backbone_outputs, action_inputs)
```

当前 DreamZero 的主要算法负载在 `WANPolicyHead`，包括：

- text encoder
- image encoder
- video VAE
- CausalWanModel DiT
- flow scheduler
- action/state encoder and decoder
- causal KV cache lifecycle

### Conditioning

Action head 构造三类 conditioning：

- Text conditioning: UMT5/T5 text encoder 输出 prompt embedding。
- Image conditioning: 首帧经 CLIP image encoder 得到 `clip_feature`。
- Video latent conditioning: 输入视频经 Wan VAE 编码为 latent `y` 或 denoising target。

Wan2.1 和 Wan2.2 的 image/video conditioning 方式不同，但外部 policy API 不应因此改变。

### CausalWanModel Sequence

`CausalWanModel` 将视频 token 和机器人 action/state token 放入同一个 transformer 序列。

训练时的概念布局：

```text
[ first image latent tokens ]
[ causal video block 0 tokens ]
[ causal video block 1 tokens ]
...
[ action tokens for block 0 ]
[ action tokens for block 1 ]
...
[ state tokens for block 0 ]
[ state tokens for block 1 ]
...
```

其中每个 block 满足：

```text
one image block <-> one action chunk <-> one state token group
```

关键配置：

- `num_frame_per_block`: 一个 causal video block 中的 latent frame 数。
- `num_action_per_block`: 每个 video block 对应的 action token 数。
- `num_state_per_block`: 每个 video block 对应的 state token 数。
- `action_horizon`: 一次预测的 action steps 数。
- `frame_seqlen`: 单个 latent frame patch embedding 后的 token 数。

必须满足：

```text
num_image_blocks == num_action_blocks == num_state_blocks
action_horizon = num_image_blocks * num_action_per_block
state_horizon  = num_image_blocks * num_state_per_block
```

代码里还隐含训练 shape 约束：

```text
actions.shape[1] / (latent_frames - 1)
  == num_action_per_block / num_frame_per_block

(latent_frames - 1) / state_features.shape[1]
  == num_frame_per_block / num_state_per_block
```

如果这些关系被破坏，action register 会和 video block 错位，即使 tensor 能 broadcast，算法语义也是错的。

### Action-State Register

Action/state 不是作为普通条件向量加到 hidden state 上，而是作为 transformer 序列尾部的 register token：

```text
action_features = action_encoder(noisy_action, action_timestep, embodiment_id)
state_features  = state_encoder(state, embodiment_id)
action_register = concat(action_features, state_features)
x = concat(video_tokens, action_register)
```

DiT 输出后：

```text
video token slice  -> video flow/noise prediction
action token slice -> action_decoder(...) -> action flow/noise prediction
```

这就是 DreamZero 能够把 world prediction 和 policy prediction 绑定在同一个 denoising process 里的核心。

### Positional Encoding And Causality

Video token 使用 3D RoPE：time、height、width。Action/state register 使用 1D RoPE：沿 action/state temporal index 编码。

attention 是 blockwise causal：

- first image frame 是全局 conditioning anchor。
- 后续 video block 只能看过去和当前允许的 block。
- action/state token 按 block 对齐，只能访问对应 causal context。
- `local_attn_size` 可限制 KV 可见窗口，当前实现由 `max_chunk_size * num_frame_per_block + 1` 推导。

推理时 `current_start_frame` 决定 RoPE 的时间 offset 和 KV cache 位置。它不是日志变量，而是因果序列坐标。

## 训练目标

训练入口最终调用 `WANPolicyHead.forward(...)`。

### Video Flow Matching

视频处理流程：

```text
uint8 video
  -> [0, 1]
  -> normalize to [-1, 1]
  -> optional resize to target Wan resolution
  -> Wan VAE encode
  -> latent noise sampling
  -> scheduler.add_noise(...)
  -> DiT predicts video flow/noise
  -> dynamics MSE against scheduler.training_target(...)
```

`frame_seqlen` 必须和 VAE latent size 以及 DiT patch embedding 一致：

```text
frame_seqlen = (latent_height // 2) * (latent_width // 2)
```

Wan2.2 5B 常用 `160x320` 输入：

```text
VAE38 spatial downscale 16x
160x320 -> latent 10x20
patch stride (1,2,2) -> 5x10 tokens
frame_seqlen = 50
```

### Action Flow Matching

action 处理流程：

```text
raw action
  -> key-wise normalization
  -> concat and pad
  -> add Gaussian noise by action timestep
  -> action_encoder(noisy_action, timestep_action, embodiment_id)
  -> DiT action register
  -> action_decoder(...)
  -> action MSE against scheduler.training_target(...)
```

video timestep 和 action timestep 可以是 coupled，也可以通过 config decouple。默认语义是 action timestep 从 video block timestep 推导，以保持 action chunk 和 causal video block 对齐。

### Loss Composition

总 loss 是 dynamics loss 与 action loss 相加：

```text
loss = weighted_dynamics_loss + weighted_action_loss
```

重要细节：

- dynamics loss 按 latent frame mask 忽略 padded video frames。
- action loss 乘 `action_mask`，忽略 padded action dims。
- action loss 再乘 `has_real_action`，无真实 action 的样本只贡献 video dynamics。
- scheduler 的 training weight 会作用到 video/action 的 timestep loss。

这允许同一训练框架混合“有 action 的机器人样本”和“更偏 video dynamics 的样本”，但 action/no-action 的 mask 语义必须保持准确。

## 推理算法

推理主要走 `GrootSimPolicy.lazy_joint_forward_causal(...)` 到 `VLA.lazy_joint_video_action_causal(...)`，再到 `WANPolicyHead.lazy_joint_video_action(...)`。

### Causal Closed-Loop

每次 policy call 预测一个 action chunk：

```text
current observation frames + current state + prompt
  -> update or reset session conditioning
  -> warm/update KV cache with real video latent
  -> initialize noisy video latent and noisy action chunk
  -> run flow denoising steps
  -> return action_pred and optional video_pred
  -> current_start_frame += num_frame_per_block
```

机器人或 eval client 执行返回的 action chunk 的一部分或全部，再把新的真实观测发给下一次 policy call。

### First Call vs Later Calls

first call 和 later call 语义不同：

- first call 用首帧建立 CLIP conditioning、VAE conditioning 和 KV cache。
- later call 复用 language/image conditioning，并把新观测 block 追加进 causal cache。
- 如果 prompt/language 改变、输入退回单帧、或 `current_start_frame` 超过 local attention window，必须 reset causal state。

需要一起 reset 的状态包括：

- `current_start_frame`
- positive/negative KV cache
- cross-attention cache
- cached `clip_feas`
- cached `ys`
- cached language identity

只 reset 其中一个会造成时间坐标、conditioning 或 cache 内容不一致。

### Denoising Loop

推理使用 `FlowUniPCMultistepScheduler`：

- video latent 从 noise 逐步 denoise。
- action chunk 从 noise 逐步 denoise。
- CFG 下 video flow 使用 conditional/unconditional mixing。
- 当前实现的 action prediction 使用 conditional action branch。

输出：

- `action_pred`: normalized action chunk，后续由 policy wrapper unnormalize。
- `video_pred`: latent/video prediction，用于调试或视频评估，不应改变 action chunk 对齐。

## Wan2.1 与 Wan2.2

本仓库用同一个 action head 实现兼容 Wan2.1-I2V-14B 和 Wan2.2-TI2V-5B，主要差异来自 config、VAE 和 resolution。

| 项 | Wan2.1-I2V-14B | Wan2.2-TI2V-5B |
| --- | --- | --- |
| config | `wan_flow_matching_action_tf.yaml` | `wan_flow_matching_action_tf_wan22.yaml` |
| model type | `i2v` | `ti2v` |
| DiT dim | 5120 | 3072 |
| VAE latent channels | 16 | 48 |
| VAE class | `WanVideoVAE` | `WanVideoVAE38` |
| common resolution | config-dependent | `160x320` in current Wan22 config |
| common `frame_seqlen` | config-dependent | `50` |
| first-frame handling | may concat first-frame latent | CLIP first-frame conditioning, no latent concat |

不要把 backbone swap 当成外部接口变化。算法外壳仍然是：

```text
one observed block -> one action chunk
```

变化的是 latent channel、token count、VAE downscale、conditioning 细节和 checkpoint component path。

## Embodiment 适配

新增或修改 embodiment 时，真正需要对齐的是算法契约：

- `EmbodimentTag` 中的 tag。
- dataset `meta/modality.json` 中 state/action/video/language key。
- data YAML 中的 `modality_config_*` 和 `transform_*`。
- video view order 和 `DreamTransform` grid layout。
- language prompt 中对 view layout 的描述。
- state/action concat order。
- normalization mode 和统计。
- `max_state_dim`、`max_action_dim`、`state_horizon`、`action_horizon`。
- 推理 wrapper 中输出 action 的拆分、unnormalize 和机器人执行顺序。

对机器人而言，action dimension 对了不等于语义对了。left/right arm order、gripper sign、absolute/relative action、joint order、camera order 都是 policy 语义的一部分。

## 修改算法代码时必须保护的不变量

- `modality.json`、Hydra `modality_keys`、transform concat order 必须一致。
- 多视角 grid layout 和 language view description 必须一致。
- action/state normalization 和 unnormalization 必须互为逆变换。
- `action_mask`、`state_mask`、`has_real_action` 不应在 collate 或 device move 中丢失。
- `frame_seqlen` 必须匹配 VAE latent spatial size 和 DiT patch embedding。
- `num_frame_per_block`、`num_action_per_block`、`num_state_per_block` 必须让 image/action/state block 数相等。
- first frame conditioning 不能和 denoised future block 混为同一语义。
- `current_start_frame` 必须和 KV cache、RoPE time index 同步。
- 推理 reset 必须同时处理 language、CLIP/VAE conditioning、KV cache 和 cross-attn cache。
- 训练和推理使用的目标 video resolution 必须一致，否则 latent token count 会错。
- action output 必须停留在 normalized space，直到 policy wrapper 显式 unnormalize。

## 推荐阅读路径

理解算法时按这个顺序读代码：

1. `groot/vla/model/dreamzero/base_vla.py`
2. `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py`
3. `groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py`
4. `groot/vla/model/dreamzero/transform/dreamzero_cotrain.py`
5. `groot/vla/data/transform/concat.py`
6. `groot/vla/data/transform/state_action.py`
7. `groot/vla/model/n1_5/sim_policy.py`
8. `docs/WAN22_BACKBONE.md`
