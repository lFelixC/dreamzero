# DreamZero Refactor Framework

This document is a working architecture map for DreamZero. Its goal is to make future refactors safer by separating:

- what the system does today,
- where the important boundaries already exist,
- where the current code is too coupled,
- and what should be extracted first.

This is not a paper summary. It is an engineering document for codebase restructuring.

## 1. One-line system summary

DreamZero is a metadata-driven robot learning stack that:

1. converts robot datasets into a LeRobot-style format plus DreamZero metadata,
2. samples multi-view video, state, language, and action chunks,
3. applies modality-specific transforms,
4. trains or runs a VLA wrapper whose action head jointly denoises video latents and action sequences,
5. serves the model in closed loop with causal chunked inference and KV caching.

## 2. Current top-level layout

| Area | Main path | Responsibility |
|---|---|---|
| Data conversion | `scripts/data/` | Convert raw datasets and generate `meta/` files |
| Configs | `groot/vla/configs/` | Hydra configs for datasets, transforms, model, DeepSpeed |
| Dataset + metadata | `groot/vla/data/` | Metadata loading, LeRobot dataset adapters, sharded sampling, transforms |
| Model wrapper | `groot/vla/model/dreamzero/` | VLA wrapper, backbone, action head, Wan modules, model transforms |
| Policy runtime | `groot/vla/model/n1_5/sim_policy.py` | Checkpoint loading, eval transforms, inference-time unnormalize logic |
| Training orchestration | `groot/vla/experiment/` | Trainer loop, logging, loss aggregation |
| Serving + evaluation | `eval_utils/`, `socket_test_optimized_AR.py`, `example/remote_infer/` | Websocket serving, RoboArena compatibility, real robot client loop |

## 3. Main execution paths

### 3.1 Offline data conversion

```text
Raw robot dataset
  -> scripts/data/convert_droid.py
  -> LeRobot parquet/mp4 + meta/
  -> scripts/data/convert_lerobot_to_gear.py (for generic embodiments)
  -> DreamZero-ready dataset root
```

Key outputs:

- `data/chunk-*/episode_*.parquet`
- `videos/chunk-*/.../episode_*.mp4`
- `meta/info.json`
- `meta/modality.json`
- `meta/stats.json`
- `meta/relative_stats_dreamzero.json`
- `meta/tasks.jsonl`
- `meta/episodes.jsonl`

### 3.2 Training path

```text
Hydra config
  -> LeRobot/DROID dataset builder
  -> shard sampler + chunk sampler
  -> transforms + collator
  -> base_vla.py
  -> backbone
  -> wan_flow_matching_action_tf.py
  -> dynamics_loss + action_loss
  -> trainer / logging / checkpoint
```

Important files:

- `groot/vla/data/dataset/lerobot.py`
- `groot/vla/data/dataset/lerobot_sharded.py`
- `groot/vla/model/dreamzero/transform/dreamzero_cotrain.py`
- `groot/vla/model/dreamzero/base_vla.py`
- `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py`
- `groot/vla/experiment/base.py`

### 3.3 Real-world and websocket inference path

```text
Robot / RoboArena observation
  -> client-side history buffer
  -> websocket request
  -> server adapter
  -> GrootSimPolicy.lazy_joint_forward_causal(...)
  -> base_vla.py
  -> DreamZero action head causal inference
  -> action chunk + video chunk
  -> client executes open-loop horizon
  -> next real observation updates cache context
```

Important files:

- `example/remote_infer/main_dreamzero.py`
- `eval_utils/serve_dreamzero_wan22.py`
- `socket_test_optimized_AR.py`
- `groot/vla/model/n1_5/sim_policy.py`
- `groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py`

## 4. Current layer model

The codebase already has implicit layers, but they are not cleanly separated.

### Layer A: Data contract layer

Primary files:

- `groot/vla/data/schema/embodiment_tags.py`
- `groot/vla/data/schema/lerobot.py`
- `groot/vla/data/dataset/metadata.py`

What this layer should own:

- dataset metadata schema,
- modality definitions,
- state/action/video key contracts,
- embodiment tags,
- normalization statistics schema.

What this layer should not own:

- sampling policy,
- inference-time cache state,
- websocket protocol logic.

### Layer B: Dataset and sampling layer

Primary files:

- `groot/vla/data/dataset/lerobot.py`
- `groot/vla/data/dataset/lerobot_sharded.py`
- `groot/vla/data/dataset/registry.py`

What this layer does today:

- loads local metadata,
- maps metadata to modality keys,
- reads parquet and videos,
- constructs trajectories,
- applies DROID-specific sub-language chunk sampling,
- converts relative actions on the fly,
- mixes multiple datasets.

Main issue:

Generic LeRobot loading and DROID-specific chunk logic are currently mixed together in the same runtime surface.

### Layer C: Transform layer

Primary files:

- `groot/vla/data/transform/video.py`
- `groot/vla/data/transform/state_action.py`
- `groot/vla/data/transform/concat.py`
- `groot/vla/model/dreamzero/transform/dreamzero_cotrain.py`

What this layer does:

- modality-specific preprocessing,
- normalization and padding,
- multi-view composition,
- text tokenization,
- collate-time mask creation.

Main issue:

Model-specific transform logic is split across generic transforms and DreamZero transform code, which makes inference/train parity harder to reason about.

### Layer D: Model orchestration layer

Primary files:

- `groot/vla/model/dreamzero/base_vla.py`
- `groot/vla/model/n1_5/sim_policy.py`

What this layer does today:

- prepare model inputs,
- dispatch backbone and action head,
- load checkpoints,
- initialize eval transforms,
- convert normalized actions back to robot actions,
- expose training and inference entrypoints.

Main issue:

`sim_policy.py` is doing three jobs at once:

- checkpoint/bootstrap logic,
- eval preprocessing contract,
- policy runtime behavior.

### Layer E: Model kernel layer

Primary files:

- `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py`
- `groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py`
- `groot/vla/model/dreamzero/modules/flow_match_scheduler.py`
- `groot/vla/model/dreamzero/modules/flow_unipc_multistep_scheduler.py`
- `groot/vla/model/dreamzero/modules/wan_video_vae.py`
- `groot/vla/model/dreamzero/modules/wan_video_text_encoder.py`
- `groot/vla/model/dreamzero/modules/wan_video_image_encoder.py`

What this layer should own:

- Wan model math,
- diffusion scheduling,
- causal attention,
- action/state register logic,
- denoising loops,
- latent encode/decode,
- cache update primitives.

Main issue:

The current action head file mixes:

- module construction,
- train forward,
- inference forward,
- cache lifecycle,
- real-world special cases,
- profiling,
- conditional/unconditional CFG handling,
- and environment-variable feature flags.

That makes it the highest-priority refactor target.

### Layer F: Delivery adapters

Primary files:

- `eval_utils/policy_server.py`
- `eval_utils/serve_dreamzero_wan22.py`
- `socket_test_optimized_AR.py`
- `example/remote_infer/main_dreamzero.py`

What this layer should own:

- request/response protocol,
- session lifecycle,
- camera buffer assembly,
- open-loop execution horizon,
- deployment-specific server behavior.

Main issue:

There is duplicated serving logic between the simpler `eval_utils/serve_dreamzero_wan22.py` path and the more specialized `socket_test_optimized_AR.py` path.

## 5. The most important coupling problems

### 5.1 Action head state is mutable and spread across concerns

`wan_flow_matching_action_tf.py` stores inference session state inside the model instance:

- `kv_cache1`
- `kv_cache_neg`
- `crossattn_cache`
- `current_start_frame`
- `clip_feas`
- `ys`
- `language`

This is workable for a research prototype, but it tightly couples:

- model weights,
- session state,
- request order,
- and deployment mode.

This should become an explicit runtime state object.

### 5.2 Real-world special cases live inside the core model path

The action head contains branches specifically for real-world execution, such as:

- first-call reset behavior,
- reuse of image conditioning,
- chunk-specific latent construction,
- and cache warmup/update steps.

These behaviors should live in an inference runtime service, not inside the train/infer kernel.

### 5.3 Dataset policy and dataset storage are entangled

The DROID dataset class is not only reading data; it also defines a task-specific sampling policy:

- language-range grouping,
- chunk count logic,
- frame index resampling,
- relative-action conversion.

That logic should move into a `ChunkSamplingStrategy` or `TrajectorySamplingPolicy` abstraction.

### 5.4 Serving code duplicates session and buffering logic

Both serving entrypoints manage:

- frame buffers,
- session resets,
- request conversion,
- action conversion.

This should be one reusable serving/runtime adapter with thin protocol frontends.

### 5.5 Config behavior is distributed across YAML, runtime overrides, and env vars

Important behavior depends on several separate places:

- Hydra YAML,
- shell overrides,
- checkpoint `experiment_cfg`,
- environment variables like `ENABLE_DIT_CACHE`, `DYNAMIC_CACHE_SCHEDULE`, `LOAD_TRT_ENGINE`.

This should be consolidated behind one typed runtime config object per entrypoint.

## 6. Refactor north star

The goal is not "rewrite everything." The goal is:

1. preserve model behavior,
2. preserve checkpoint compatibility,
3. preserve server API contracts,
4. isolate research kernels from runtime orchestration,
5. make new embodiments and new serving modes cheaper to add.

## 7. Proposed target architecture

One practical target shape is:

```text
dreamzero/
  app/
    train/
    infer/
    serve/
  domain/
    contracts/
    metadata/
    session/
  data/
    adapters/
    sampling/
    transforms/
  model/
    runtime/
    backbone/
    action_head/
    kernels/
    schedulers/
  serving/
    protocols/
    adapters/
```

This does not require a big bang rewrite. It can be reached incrementally inside the existing package structure.

### 7.1 Suggested runtime contracts

Introduce explicit typed objects for:

- `DreamZeroSample`
- `DreamZeroBatch`
- `ActionChunk`
- `VideoChunk`
- `InferenceSessionState`
- `CausalCacheState`
- `PolicyRequest`
- `PolicyResponse`

Benefits:

- fewer hidden assumptions about tensor shapes,
- easier tests,
- less direct mutation of model internals,
- clearer train/infer boundaries.

### 7.2 Suggested services to extract

#### A. Metadata service

Responsibilities:

- resolve local vs global metadata,
- load stats,
- substitute relative stats,
- validate modality contracts.

Likely extraction source:

- `groot/vla/data/dataset/lerobot.py`

#### B. Sampling service

Responsibilities:

- choose trajectory and step indices,
- sample DROID sub-language chunks,
- compute per-chunk frame/state/action indices.

Likely extraction source:

- `groot/vla/data/dataset/lerobot_sharded.py`

#### C. Action normalization service

Responsibilities:

- normalize and unnormalize action/state,
- convert relative action to absolute action,
- own embodiment-specific rules.

Likely extraction source:

- `groot/vla/data/transform/state_action.py`
- `groot/vla/model/n1_5/sim_policy.py`

#### D. Causal inference runtime

Responsibilities:

- manage session state,
- own KV cache lifecycle,
- warm cache with real observations,
- run chunk denoising,
- return action/video chunks.

Likely extraction source:

- `wan_flow_matching_action_tf.py`
- `sim_policy.py`

#### E. Server adapter

Responsibilities:

- convert external request format to internal `PolicyRequest`,
- assemble multi-view frame history,
- manage open-loop horizon and reset behavior,
- convert internal `PolicyResponse` to websocket payload.

Likely extraction source:

- `socket_test_optimized_AR.py`
- `eval_utils/serve_dreamzero_wan22.py`
- `example/remote_infer/main_dreamzero.py`

## 8. What should stay stable during refactor

These are the invariants worth protecting:

### 8.1 Data invariants

- `meta/` file meaning and schema
- modality key names
- normalization semantics
- relative action conversion semantics

### 8.2 Model invariants

- checkpoint loading path
- `base_vla.py` user-facing forward contract
- action head output keys
- dynamics/action loss formulas
- causal chunk alignment between video, state, and action

### 8.3 Serving invariants

- RoboArena-compatible metadata and response contract
- action output shape `(N, 8)` for DROID serving
- `session_id` reset semantics
- open-loop client behavior

## 9. Recommended refactor order

The safest order is outside-in for contracts, then inside-out for implementation.

### Phase 0: Lock current behavior with smoke tests

Before moving code, add or preserve checks for:

- one training batch shape contract,
- one inference batch shape contract,
- real-world causal step contract,
- websocket server output contract,
- relative-action round-trip correctness.

Do not start with a large structural rename before these exist.

### Phase 1: Extract contracts and config objects

Target:

- typed request/response/session/cache dataclasses,
- typed runtime config objects for training and serving.

Why first:

- very low behavioral risk,
- immediately reduces implicit coupling.

### Phase 2: Split dataset loading from sampling policy

Target:

- generic LeRobot adapter remains generic,
- DROID chunk sampling becomes a pluggable policy.

Why second:

- high leverage,
- simplifies later multi-embodiment work.

### Phase 3: Split policy runtime from model kernel

Target:

- action head keeps model math,
- a separate runtime object owns session state and cache update strategy.

Why third:

- this is the most important architectural cleanup,
- but it is safer after contracts are explicit.

### Phase 4: Unify serving adapters

Target:

- one internal serving runtime,
- thin wrappers for RoboArena and local robot use cases.

Why fourth:

- depends on the runtime extraction from Phase 3.

### Phase 5: Clean feature flags and deployment knobs

Target:

- replace env-var-driven core logic with typed runtime options where possible.

Why last:

- less risky after structure is already separated.

## 10. First files to refactor

If only a small amount of cleanup is possible first, start here:

### Option A: Extract `InferenceSessionState`

Source:

- `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py`

Move out:

- current frame pointer,
- caches,
- prompt/language state,
- cached image conditioning.

This yields the highest long-term payoff.

### Option B: Extract `DroidChunkSamplingStrategy`

Source:

- `groot/vla/data/dataset/lerobot_sharded.py`

Move out:

- language-range selection,
- chunk-size alignment,
- frame/state/action resampling.

This makes the data layer much easier to test.

### Option C: Extract a shared serving adapter

Source:

- `socket_test_optimized_AR.py`
- `eval_utils/serve_dreamzero_wan22.py`

Move out:

- session reset behavior,
- frame buffering,
- observation key mapping,
- action dict to `(N, 8)` conversion.

This removes duplication without touching the diffusion core.

## 11. Suggested ownership map after refactor

| Module | Should own | Should not own |
|---|---|---|
| `data/schema` | metadata schemas, modality contracts | sampling rules |
| `data/adapters` | parquet/video access | DROID-specific policy logic |
| `data/sampling` | chunk/index policy | file I/O |
| `data/transforms` | preprocessing and normalization | session/cache state |
| `model/runtime` | inference session lifecycle | low-level DiT math |
| `model/kernels` | Wan forward math, schedulers, VAE, attention | websocket protocol |
| `serving/adapters` | protocol conversion and buffering | model internals |
| `app/train` | train assembly | tensor math details |
| `app/serve` | deployment assembly | checkpoint math internals |

## 12. Non-goals for the first refactor pass

These are tempting, but should not be mixed into the first restructuring step:

- changing loss definitions,
- changing sampling semantics,
- changing checkpoint key names,
- changing action-space meaning,
- changing external server protocol,
- changing the Wan model itself.

Refactor first. Research changes later.

## 13. Practical definition of success

A successful first refactor should make the following true:

1. a new engineer can trace train and infer entrypoints without reading the entire action head file,
2. serving logic can be tested without loading the full model,
3. session state is explicit instead of hidden in mutable model fields,
4. DROID-specific sampling is isolated from generic LeRobot loading,
5. existing checkpoints and websocket clients still work unchanged.

## 14. Short summary

DreamZero already has the right high-level pieces, but the current codebase is organized around research velocity rather than runtime boundaries. The best refactor path is not a rewrite. It is:

- stabilize contracts,
- isolate dataset sampling policy,
- isolate inference session state,
- and then unify serving around that cleaner core.

If you use this document as the refactor map, the best first deep cut is the boundary between:

- `sim_policy.py` and the DreamZero action head,
- and between generic LeRobot loading and DROID-specific chunk sampling.
