# RoboTwin RSL-RL Integration Plan

本文档描述如何把 RSL-RL 接入已打通的 RoboTwin RL pipeline，同时把 SFT 和 RL 框架解耦。

核心目标：

- SFT pipeline 只负责产出 DreamZero RoboTwin checkpoint。
- RL pipeline 只读取 SFT checkpoint 初始化策略，不复用 SFT 的 Trainer、DeepSpeed、dataset loader。
- 第一版优先跑通稳定的 online RL 闭环，而不是直接 PPO 微调整个 diffusion WAM。
- RL checkpoint 应清楚记录所依赖的 SFT checkpoint、RL 配置、actor/critic 权重和归一化状态。

## Current Repo Context

当前已有基础：

- SFT 训练入口：
  - `scripts/train/a800_train/run_robotwin_joint_no_infra_2node.sh`
  - `groot/vla/experiment/experiment.py`
- DreamZero RoboTwin checkpoint 推理入口：
  - `socket_test_optimized_aloha_x5lite_bimanual.py`
  - `groot/vla/model/n1_5/sim_policy.py`
- RoboTwin live env / parallel eval 基础设施：
  - `example/robotwin/robotwin_fast_env.py`
  - `example/robotwin/parallel_env_worker.py`
  - `example/robotwin/parallel_eval.py`
  - `example/robotwin/reward.py`
- 现有 parallel eval 已经具备多 env subprocess rollout 和 batched websocket inference 能力，可作为 RL collector 基础。

## Review Adoption

本次修订把方案收紧到 RSL-RL 当前 `VecEnv` / `OnPolicyRunner` 接口可以直接落地的形态：

- RSL-RL actor 输出的是 **residual action**，不是最终发送给 RoboTwin 的 action。
- `RoboTwinRslVecEnv.step(residual_action)` 内部读取 frozen SFT base action，组合成 `final_action = clip(base_action + scale * residual_action)`，再 step RoboTwin。
- PPO 的 log-prob、entropy、KL 都只对应 residual action，避免“训练分布”和“真实执行动作”不一致。
- 第一版 observation 给 RSL-RL 的默认 `ActorCritic` 时必须是 2D 向量 TensorDict，不直接塞 raw image。视觉信息先通过 SFT base action 间接进入 residual policy。
- SFT adapter 不走 websocket，作为 RL 进程内的 frozen module 使用；websocket path 仍保留给 eval / baseline 对比。

## Architecture Boundary

建议边界如下：

```text
SFT pipeline
  offline RoboTwin data
  Hydra config
  HF Trainer / DeepSpeed
  DreamZero checkpoint

          |
          | checkpoint directory only
          v

RL pipeline
  RSL-RL runner
  RoboTwin VecEnv adapter
  frozen SFT policy adapter
  residual actor + critic
  final action composition inside VecEnv
  RL checkpoint
```

不要让 RL 入口 import 或继承 SFT `BaseExperiment` / `VLATrainer`。

推荐新增目录：

```text
groot/vla/rl/
  __init__.py
  robotwin_vec_env.py
  sft_policy_adapter.py
  residual_policy.py
  robotwin_rsl_train.py
  configs.py

scripts/train/
  robotwin_rl_rsl.sh

groot/vla/configs/rl/
  robotwin_rsl.yaml
```

## Step 1: Dependency And Entry Isolation

实现内容：

- 将 `rsl_rl` 作为 RL optional dependency 或单独 requirements，不加入 SFT 必需依赖。
- 新建 RL entrypoint，例如 `groot/vla/rl/robotwin_rsl_train.py`。
- 新建 shell launcher，例如 `scripts/train/robotwin_rl_rsl.sh`。
- RL launcher 只接收 `--sft-checkpoint`、RoboTwin env 配置、RSL-RL 配置和输出目录。

验收：

- 不安装 `rsl_rl` 时，现有 SFT 训练脚本仍可启动到配置解析阶段。
- RL 环境中 `python -c "import rsl_rl"` 成功。
- `bash -n scripts/train/robotwin_rl_rsl.sh` 通过。
- `python -m py_compile groot/vla/rl/*.py` 通过。

## Step 2: SFT Checkpoint Policy Adapter

实现内容：

- 从 `socket_test_optimized_aloha_x5lite_bimanual.py` 中抽出非 websocket 的 policy adapter，训练时不通过 policy server。
- Adapter 负责：
  - 读取 SFT checkpoint。
  - 复用 `GrootSimPolicy`。
  - 执行 RoboTwin obs 到 DreamZero model input 的转换。
  - 输出 `[B, H, 14]` action chunk。
  - 提供 step-level base action，即当前 chunk 的第一个未消费 action。
  - 维护 session reset / temporal cache reset。
  - 保证 action 顺序为 `[left7, right7]`。
- 第一版不训练 adapter 内部的 SFT model，所有 SFT 参数冻结。

建议接口：

```python
class SFTPolicyAdapter:
    def reset(self, session_ids: list[str] | None = None) -> None: ...
    def act_chunk(self, obs_batch: dict, prompts: list[str], session_ids: list[str]) -> np.ndarray: ...
    def act_step(self, obs_batch: dict, prompts: list[str], session_ids: list[str]) -> np.ndarray: ...
```

验收：

- 单条 RoboTwin obs 输入输出 shape 为 `[1, H, 14]`。
- batched obs 输入输出 shape 为 `[B, H, 14]`。
- `act_step()` 输出 shape 为 `[B, 14]`，且等于当前 chunk 的第一个 executable action。
- 与现有 websocket server 在同 checkpoint、同 obs 下 action shape 一致。
- `requires_grad=False` 覆盖全部 SFT model 参数。
- session_id 改变时 causal cache 被 reset，跨 task smoke 不串缓存。

## Step 3: RoboTwin RSL VecEnv Adapter

实现内容：

- 实现 `RoboTwinRslVecEnv`，复用现有 subprocess worker，避免父进程 fork SAPIEN / CUDA。
- 第一版使用 step-level residual action：RSL-RL 每步输出 `[num_envs, 14]`，VecEnv 内部组合最终动作。
- Env 内部可以缓存 SFT chunk，但对 RSL-RL 暴露一步一动的标准接口。
- VecEnv 必须提供：
  - `num_envs`
  - `num_actions=14`
  - `max_episode_length`
  - `episode_length_buf`
  - `device`
  - `cfg`
  - `get_observations()`
  - `step(actions)`
- `step(actions)` 中的 `actions` 是 residual action，真实发送给 RoboTwin 的动作是：

```text
base_action = frozen_sft_policy.act_step(latest_raw_obs)
final_action = clip(base_action + residual_scale * residual_action, -1.0, 1.0)
```

- 观测返回 RSL-RL 可消费的 `TensorDict`，第一版只放 2D 向量：
  - `policy`: `[num_envs, obs_dim]`
  - `critic`: `[num_envs, critic_obs_dim]`，第一版可等同于 `policy`
- 第一版 observation 建议包含：
  - current joint state / gripper state
  - last SFT base action
  - last executed final action
  - normalized episode progress
  - task id 或 task embedding
  - success / timeout mask 不进入 actor，可放 extras
- `extras` 中记录：
  - `time_outs`
  - `log`
  - `success`
  - `task`
  - `seed`
  - `env_step_time`
  - `get_obs_time`

注意：

- RSL-RL `VecEnv` 接口要求 `get_observations()` 返回 `TensorDict`，`step()` 返回 observations、rewards、dones、extras。
- RSL-RL 默认 `ActorCritic` 只支持 1D observation，所以 TensorDict 中每个 observation group 应是 `[num_envs, dim]`。
- RoboTwin reset 仍使用现有 retry 逻辑。
- 第一版 reward 使用 sparse terminal reward，dense reward hook 后续再接。

验收：

- `num_envs=1` dry-run 能 reset、step、done。
- `num_envs=4` dry-run 能跑完整 episode。
- done 后 env 能自动 reset 或由 runner 控制 reset。
- `reward` shape 为 `[num_envs]`。
- `done` shape 为 `[num_envs]`。
- `time_outs` 正确区分 success termination 和 episode length truncation。
- TensorDict 中 `policy` / `critic` 都是 2D tensor。
- residual action 为零时，RoboTwin 收到的 action 与 frozen SFT base action 一致。
- worker 退出时无残留子进程。

## Step 4: Residual Actor And Critic

实现内容：

第一版不建议直接 PPO 训练整个 DreamZero diffusion WAM，因为现有 diffusion action inference 没有天然 PPO 所需的 action log-prob。

推荐策略：

```text
sft_action = frozen_sft_policy(obs)[first_step]
residual_action ~ GaussianResidualActor(vector_obs)
final_action = clip(sft_action + residual_scale * residual_action, action_low, action_high)
```

实现上，`GaussianResidualActor` 可以直接使用 RSL-RL 默认 `ActorCritic`。关键是把 RSL-RL 的 action 语义定义为 residual，而不是 final action；final action composition 放在 `RoboTwinRslVecEnv.step()` 内。

模块：

- Frozen SFT policy adapter：提供 base action。
- Gaussian residual actor：RSL-RL 训练对象，zero-mean init。
- Critic：MLP value function。
- 可选 action normalizer / obs encoder。

优点：

- 初始化行为等价 SFT。
- PPO 可以获得 residual 的 log-prob、entropy、KL。
- 训练风险小，不会第一版就破坏 WAM 权重。
- RL checkpoint 小，便于快速试错。
- 不需要改 RSL-RL PPO storage/action log-prob 逻辑。

验收：

- residual mean 为零、`residual_scale=0` 或 deterministic zero-residual smoke 时，`final_action == sft_action`。
- actor 输出 distribution 有 `sample()`、`log_prob()`、`entropy()`。
- critic 输出 `[num_envs, 1]`。
- rollout storage 中 action、log_prob、value shape 符合 RSL-RL runner。
- SFT 参数无梯度，residual actor / critic 有梯度。

## Step 5: RSL-RL Runner Integration

实现内容：

- 新建 RL train entry：
  - 初始化 distributed / device。
  - 加载 SFT adapter。
  - 构造 `RoboTwinRslVecEnv`。
  - 构造 RSL-RL actor_critic、algorithm、runner config。
  - 启动 RSL-RL `OnPolicyRunner`。
- Runner config 的 `obs_groups` 第一版建议：

```yaml
obs_groups:
  policy: ["policy"]
  critic: ["critic"]
```

- 日志输出：
  - PPO loss
  - value loss
  - entropy
  - KL
  - residual norm
  - action clip ratio
  - episode reward
  - success rate
  - env throughput

验收：

- `num_learning_iterations=1` 能完整跑通。
- rollout collection、PPO update、checkpoint save 都成功。
- loss / KL / entropy / value 不为 NaN。
- RSL-RL `OnPolicyRunner` 可以从 `env.get_observations()` 完成 algorithm construction。
- checkpoint 包含：
  - SFT checkpoint path
  - residual actor state dict
  - critic state dict
  - optimizer state
  - normalizer state
  - residual scale
  - RL config
  - git commit

## Step 6: Small-Scale Training Acceptance

建议第一轮固定：

```text
task: beat_block_hammer
num_envs: 4
episode_length: 300 或 task step limit
reward: sparse terminal success
sft_checkpoint: existing RoboTwin SFT checkpoint
residual_std_init: small
residual_scale: 0.05 或更小起步
max_iterations: 10-50
```

验收不以成功率提升为第一目标，先验收稳定性：

- 训练 10-50 iteration 不崩溃。
- GPU memory 稳定，无持续泄漏。
- `residual_norm` 不爆炸。
- action clip ratio 不长期接近 1。
- KL 在阈值内。
- final action 与 SFT base action 的平均偏移可控。
- episode JSON / report 可复现。
- 同一个 SFT checkpoint 跑 frozen baseline 与 RL policy eval，报告可比较。

## Step 7: Scale To Multi-Task And Evaluation

实现内容：

- 接入多 task sampling。
- 每个 task 保留 task-specific prompt、seed、success metric。
- RL eval 复用现有 `parallel_eval.py` 报告结构，新增 RL checkpoint metadata。
- 后续可扩展：
  - dense reward hook。
  - chunk-level residual action。
  - multi-server shard。
  - per-env temporal cache save/restore。
  - 解冻 LoRA 或 action expert 微调。

验收：

- 多 task smoke：至少两个 RoboTwin task，每个 2 episodes。
- report 中每个 episode 可追踪：
  - task
  - seed
  - SFT checkpoint
  - RL checkpoint
  - success
  - reward sum
  - steps
- frozen SFT baseline 与 RL policy eval 输出在同一 report schema 下可比较。
- 跨 task cache reset 正常。

## Suggested Milestone Checklist

- [ ] RL dependency isolated from SFT.
- [ ] SFT policy adapter extracted and tested.
- [ ] RoboTwin RSL VecEnv dry-run passes.
- [ ] Residual actor / critic zero-init baseline matches SFT.
- [ ] One-iteration RSL-RL PPO update passes.
- [ ] Single-task 10-50 iteration stability run passes.
- [ ] RL checkpoint can resume and evaluate.
- [ ] Multi-task smoke passes.

## First Version Non-Goals

第一版不要做：

- 直接 PPO 更新整个 DreamZero WAM。
- 直接修改 SFT `experiment.py` / `BaseTrainer`。
- 在 RSL-RL 里复用 SFT dataloader。
- 把 raw image 直接喂给 RSL-RL 默认 `ActorCritic`。
- 第一版训练视觉 encoder。
- 默认启用 dense reward。
- 默认启用 Ray。
- per-env asynchronous refill。
- per-env causal cache save/restore。

这些都可以在 residual PPO 闭环稳定后逐步增加。

## Validation Commands Draft

静态检查：

```bash
cd /data/dreamzero_mot

python -m py_compile \
  example/robotwin/parallel_eval.py \
  example/robotwin/parallel_env_worker.py \
  example/robotwin/robotwin_fast_env.py \
  socket_test_optimized_aloha_x5lite_bimanual.py

bash -n scripts/train/a800_train/run_robotwin_joint_no_infra_2node.sh
```

RL 新增文件后：

```bash
cd /data/dreamzero_mot

python -m py_compile groot/vla/rl/*.py
bash -n scripts/train/robotwin_rl_rsl.sh
python -c "import rsl_rl; print('rsl_rl ok')"
```

RoboTwin env dry-run：

```bash
cd /data/dreamzero_mot

DRY_RUN_ACTIONS=1 TASKS=beat_block_hammer EPISODES=1 NUM_ENVS=1 \
ENV_CUDA=0 SAVE_VIDEO=0 \
bash example/robotwin/run_parallel_eval.sh
```

Frozen SFT baseline eval：

```bash
cd /data/dreamzero_mot

TASKS=beat_block_hammer EPISODES=4 NUM_ENVS=2 \
SERVER_CUDA=6,7 SERVER_NPROC=2 ENV_CUDA=0,1 \
OPEN_LOOP_HORIZON=8 SAVE_VIDEO=0 PORT=8100 \
bash example/robotwin/run_parallel_eval.sh
```

RL one-iteration smoke command should be added after `robotwin_rsl_train.py` lands.

## References

- RSL-RL configuration guide: https://leggedrobotics.github.io/rsl_rl/guide/configuration.html
- RSL-RL repository: https://github.com/leggedrobotics/rsl_rl
- RSL-RL VecEnv source: https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/env/vec_env.py
