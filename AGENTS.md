# AGENTS.md

## Purpose

This file records the DreamZero algorithm structure and key invariants in this repository. Before changing code, future agents should use this algorithm map to understand how data enters the model, what the model predicts, how action and video are aligned, and which implicit contracts are shared by training and inference.

## Algorithm Overview

DreamZero implements a World Action Model here, not a pure action-only policy.

Core idea:

- The model learns both world dynamics and robot policy.
- Inputs include multi-view video, language, current state, and a future action chunk.
- Video is encoded into latent space by the VAE, while action/state is encoded as an action-state register inside the DiT.
- The DiT performs joint flow matching / denoising over video latents and action registers.
- During training, the model predicts both video flow/noise and action flow/noise.
- During inference, given real observation frames and state, the model predicts the next action chunk block by block using a causal KV cache, and can also generate video latents as world prediction.

High-level path:

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

## Data Representation

The DreamZero data contract is jointly defined by `meta/`, Hydra data config, and the transform pipeline. Do not only check tensor shape; modality key, view order, normalization statistics, and action horizon are also part of the algorithm.

Key files:

- `groot/vla/data/schema/`
- `groot/vla/data/dataset/`
- `groot/vla/data/transform/`
- `groot/vla/model/dreamzero/transform/dreamzero_cotrain.py`
- `groot/vla/configs/data/dreamzero/*.yaml`

### Modality Contract

Each embodiment dataset root needs DreamZero metadata:

- `meta/modality.json` defines state/action/video/annotation keys and index ranges.
- `meta/stats.json` provides standard state/action normalization statistics.
- `meta/relative_stats_dreamzero.json` provides relative-action statistics.
- `meta/embodiment.json` provides the embodiment tag.

`modality_keys` in Hydra YAML must exactly match the keys in `modality.json`. Common DROID keys are:

- video: `video.exterior_image_1_left`, `video.exterior_image_2_left`, `video.wrist_image_left`
- state: `state.joint_position`, `state.gripper_position`
- action: `action.joint_position`, `action.gripper_position`
- language: annotation language keys

### Multi-View Video Layout

`ConcatTransform` first concatenates multiple `video.*` keys into `video` in config order. `DreamTransform` then lays the multi-view video out as a single grid image before passing it to the Wan video model.

DROID / `EmbodimentTag.OXE_DROID` has a special layout:

```text
[ wrist view stretched across top row ]
[ left exterior | right exterior       ]
```

Other multi-view embodiments usually use a 2x2 grid:

```text
[ view 0 | view 2 ]
[ view 1 | black  ]
```

This layout is also described in the language prompt and affects the spatial semantics of visual tokens. Changing camera order, grid layout, or prompt view description changes the cross-view correspondences the model learns.

### State / Action Representation

State and action are concatenated by fixed key order in the transform, then padded to the model-configured maximum dimension:

- `state` semantically represents current or short-history proprio tokens.
- `action` semantically represents the future action chunk.
- `state_mask` and `action_mask` mark real dimensions so padded dimensions do not contribute to loss.
- `has_real_action` controls whether a sample contributes action loss.
- `embodiment_id` is the integer id mapped from the embodiment tag and is used by the action/state encoding path.

Numerical normalization usually uses `q99` and clips values to `[-1, 1]`. Actions must already be in this normalized space before diffusion; inference outputs are also in normalized space until the policy wrapper explicitly unnormalizes them back to robot action space.

### Language Processing

Language is not a raw task string. `DreamTransform` and the collator inject view-layout descriptions by embodiment, such as "multi-view video shows..." and camera meanings for each view.

This means the language prompt has two roles:

- task instruction.
- visual grid semantic description.

If the view layout changes, the language formalization must change with it. Otherwise the model receives contradictory visual semantics.

## Model Structure

The high-level model is `VLA = backbone + action_head`; the action head contains the core algorithm.

Key files:

- `groot/vla/model/dreamzero/base_vla.py`
- `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py`
- `groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py`
- `groot/vla/model/dreamzero/modules/flow_match_scheduler.py`
- `groot/vla/model/dreamzero/modules/flow_unipc_multistep_scheduler.py`
- `groot/vla/model/n1_5/sim_policy.py`

### VLA Boundary

`base_vla.py` only performs model-level dispatch:

```text
inputs
  -> backbone.prepare_input(...)
  -> action_head.prepare_input(...)
  -> backbone(...)
  -> action_head(backbone_outputs, action_inputs)
```

Most DreamZero algorithm logic currently lives in `WANPolicyHead`, including:

- text encoder
- image encoder
- video VAE
- CausalWanModel DiT
- flow scheduler
- action/state encoder and decoder
- causal KV cache lifecycle

### Conditioning

The action head constructs three types of conditioning:

- Text conditioning: UMT5/T5 text encoder output prompt embeddings.
- Image conditioning: the first frame is passed through the CLIP image encoder to produce `clip_feature`.
- Video latent conditioning: input video is encoded by the Wan VAE into latent `y` or a denoising target.

Wan2.1 and Wan2.2 use different image/video conditioning details, but the external policy API should not change because of that.

### CausalWanModel Sequence

`CausalWanModel` puts video tokens and robot action/state tokens into one transformer sequence.

Conceptual training layout:

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

Each block satisfies:

```text
one image block <-> one action chunk <-> one state token group
```

Key config fields:

- `num_frame_per_block`: latent frames in one causal video block.
- `num_action_per_block`: action tokens corresponding to each video block.
- `num_state_per_block`: state tokens corresponding to each video block.
- `action_horizon`: action steps predicted per call.
- `frame_seqlen`: token count after patch embedding one latent frame.

These must hold:

```text
num_image_blocks == num_action_blocks == num_state_blocks
action_horizon = num_image_blocks * num_action_per_block
state_horizon  = num_image_blocks * num_state_per_block
```

The code also has implicit training shape constraints:

```text
actions.shape[1] / (latent_frames - 1)
  == num_action_per_block / num_frame_per_block

(latent_frames - 1) / state_features.shape[1]
  == num_frame_per_block / num_state_per_block
```

If these relationships are broken, the action register becomes misaligned with the video blocks. Even if tensors can broadcast, the algorithmic semantics are wrong.

### Action-State Register

Action/state is not added to hidden states as a normal conditioning vector. It is appended to the transformer sequence tail as register tokens:

```text
action_features = action_encoder(noisy_action, action_timestep, embodiment_id)
state_features  = state_encoder(state, embodiment_id)
action_register = concat(action_features, state_features)
x = concat(video_tokens, action_register)
```

After DiT output:

```text
video token slice  -> video flow/noise prediction
action token slice -> action_decoder(...) -> action flow/noise prediction
```

This is the core mechanism that lets DreamZero bind world prediction and policy prediction inside one denoising process.

### Positional Encoding And Causality

Video tokens use 3D RoPE: time, height, width. Action/state registers use 1D RoPE along the action/state temporal index.

Attention is blockwise causal:

- The first image frame is the global conditioning anchor.
- Later video blocks can only see past blocks and the currently allowed block.
- Action/state tokens are block-aligned and can only access the corresponding causal context.
- `local_attn_size` can restrict the visible KV window. The current implementation derives it from `max_chunk_size * num_frame_per_block + 1`.

During inference, `current_start_frame` determines the RoPE time offset and KV cache position. It is not a logging variable; it is the causal sequence coordinate.

## Training Objective

The training entrypoint eventually calls `WANPolicyHead.forward(...)`.

### Video Flow Matching

Video processing flow:

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

`frame_seqlen` must match VAE latent size and DiT patch embedding:

```text
frame_seqlen = (latent_height // 2) * (latent_width // 2)
```

Wan2.2 5B commonly uses `160x320` input:

```text
VAE38 spatial downscale 16x
160x320 -> latent 10x20
patch stride (1,2,2) -> 5x10 tokens
frame_seqlen = 50
```

### Action Flow Matching

Action processing flow:

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

Video timestep and action timestep can be coupled, or decoupled by config. The default semantics derive the action timestep from the video block timestep so the action chunk and causal video block stay aligned.

### Loss Composition

Total loss is the sum of dynamics loss and action loss:

```text
loss = weighted_dynamics_loss + weighted_action_loss
```

Important details:

- Dynamics loss uses the latent frame mask to ignore padded video frames.
- Action loss is multiplied by `action_mask`, ignoring padded action dimensions.
- Action loss is also multiplied by `has_real_action`; samples without real action only contribute video dynamics.
- The scheduler training weight is applied to video/action timestep loss.

This allows one training framework to mix robot samples with actions and samples that are more video-dynamics-oriented, but the action/no-action mask semantics must remain accurate.

## Inference Algorithm

Inference mainly goes through `GrootSimPolicy.lazy_joint_forward_causal(...)`, then `VLA.lazy_joint_video_action_causal(...)`, then `WANPolicyHead.lazy_joint_video_action(...)`.

### Inference Entrypoints

`eval_utils/serve_dreamzero_wan22.py` is the original official Wan2.2 websocket serve script. It is kept mostly as a reference/compatibility entrypoint. The current repository primarily uses the root-level socket server versions for evaluation and deployment, for example:

- `socket_test_optimized_AR.py`
- `socket_test_optimized_aloha_x5lite_bimanual.py`

When reviewing or changing inference logic, prioritize the contracts between these root-level socket servers and their corresponding clients. Do not infer the current main-path behavior from `eval_utils/serve_dreamzero_wan22.py` alone.

### Causal Closed Loop

Each policy call predicts one action chunk:

```text
current observation frames + current state + prompt
  -> update or reset session conditioning
  -> warm/update KV cache with real video latent
  -> initialize noisy video latent and noisy action chunk
  -> run flow denoising steps
  -> return action_pred and optional video_pred
  -> current_start_frame += num_frame_per_block
```

The robot or eval client executes part or all of the returned action chunk, then sends new real observations to the next policy call.

### First Call vs Later Calls

The first call and later calls have different semantics:

- The first call establishes CLIP conditioning, VAE conditioning, and KV cache from the first frame.
- Later calls reuse language/image conditioning and append the new observation block to the causal cache.
- If prompt/language changes, the input falls back to a single frame, or `current_start_frame` exceeds the local attention window, causal state must be reset.

State that must be reset together includes:

- `current_start_frame`
- positive/negative KV cache
- cross-attention cache
- cached `clip_feas`
- cached `ys`
- cached language identity

Resetting only one of these creates inconsistencies between time coordinates, conditioning, and cache contents.

### Denoising Loop

Inference uses `FlowUniPCMultistepScheduler`:

- Video latent is denoised step by step from noise.
- Action chunk is denoised step by step from noise.
- Under CFG, video flow uses conditional/unconditional mixing.
- The current implementation uses the conditional action branch for action prediction.

Outputs:

- `action_pred`: normalized action chunk, later unnormalized by the policy wrapper.
- `video_pred`: latent/video prediction for debugging or video evaluation. It should not change action chunk alignment.

## Wan2.1 And Wan2.2

This repository uses one action head implementation compatible with Wan2.1-I2V-14B and Wan2.2-TI2V-5B. The main differences come from config, VAE, and resolution.

| Item | Wan2.1-I2V-14B | Wan2.2-TI2V-5B |
| --- | --- | --- |
| config | `wan_flow_matching_action_tf.yaml` | `wan_flow_matching_action_tf_wan22.yaml` |
| model type | `i2v` | `ti2v` |
| DiT dim | 5120 | 3072 |
| VAE latent channels | 16 | 48 |
| VAE class | `WanVideoVAE` | `WanVideoVAE38` |
| common resolution | config-dependent | `160x320` in current Wan22 config |
| common `frame_seqlen` | config-dependent | `50` |
| first-frame handling | may concat first-frame latent | CLIP first-frame conditioning, no latent concat |

Do not treat a backbone swap as an external interface change. The algorithm shell remains:

```text
one observed block -> one action chunk
```

What changes is latent channel count, token count, VAE downscale, conditioning details, and checkpoint component paths.

## Embodiment Adaptation

When adding or modifying an embodiment, the real target is the algorithm contract:

- The tag in `EmbodimentTag`.
- State/action/video/language keys in dataset `meta/modality.json`.
- `modality_config_*` and `transform_*` in data YAML.
- Video view order and `DreamTransform` grid layout.
- View layout description in the language prompt.
- State/action concat order.
- Normalization mode and statistics.
- `max_state_dim`, `max_action_dim`, `state_horizon`, `action_horizon`.
- Action splitting, unnormalization, and robot execution order in the inference wrapper.

For a robot, a matching action dimension does not imply matching semantics. Left/right arm order, gripper sign, absolute vs relative action, joint order, and camera order are all policy semantics.

## Invariants To Protect When Modifying Algorithm Code

- `modality.json`, Hydra `modality_keys`, and transform concat order must match.
- Multi-view grid layout and language view description must match.
- Action/state normalization and unnormalization must be inverse transforms.
- `action_mask`, `state_mask`, and `has_real_action` must not be lost during collation or device moves.
- `frame_seqlen` must match VAE latent spatial size and DiT patch embedding.
- `num_frame_per_block`, `num_action_per_block`, and `num_state_per_block` must make image/action/state block counts equal.
- First-frame conditioning must not be mixed with denoised future-block semantics.
- `current_start_frame` must stay synchronized with KV cache and RoPE time index.
- Inference reset must handle language, CLIP/VAE conditioning, KV cache, and cross-attention cache together.
- Training and inference must use the same target video resolution, otherwise latent token count is wrong.
- Action output must remain in normalized space until the policy wrapper explicitly unnormalizes it.

## Recommended Reading Path

Read the code in this order to understand the algorithm:

1. `groot/vla/model/dreamzero/base_vla.py`
2. `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py`
3. `groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py`
4. `groot/vla/model/dreamzero/transform/dreamzero_cotrain.py`
5. `groot/vla/data/transform/concat.py`
6. `groot/vla/data/transform/state_action.py`
7. `groot/vla/model/n1_5/sim_policy.py`
8. `docs/WAN22_BACKBONE.md`
