# RoboTwin On DreamZero

This folder contains the RoboTwin inference client and the one-command live
rollout smoke test. DreamZero training/server code keeps using
`/data/dreamzero/.venv`; RoboTwin simulation dependencies live in the isolated
environment `/data/envs/robotwin310`.

## 1. Convert Data

```bash
cd /data/dreamzero_mot

/data/dreamzero/.venv/bin/python scripts/data/convert_robotwin_v3_to_dreamzero.py \
  --input /data/datasets/dreamzero/robotwin_unified \
  --output /data/datasets/dreamzero/robotwin_unified_dreamzero \
  --retire-source rename-backup

/data/dreamzero/.venv/bin/python scripts/data/convert_lerobot_to_gear.py \
  --dataset-path /data/datasets/dreamzero/robotwin_unified_dreamzero \
  --embodiment-tag aloha_x5lite_bimanual \
  --state-keys '{"left_joint_pos":[0,6],"left_gripper_pos":[6,7],"right_joint_pos":[7,13],"right_gripper_pos":[13,14]}' \
  --action-keys '{"left_joint_pos":[0,6],"left_gripper_pos":[6,7],"right_joint_pos":[7,13],"right_gripper_pos":[13,14]}' \
  --relative-action-keys left_joint_pos left_gripper_pos right_joint_pos right_gripper_pos \
  --task-key task_index \
  --task-alias task \
  --force
```

The raw dataset is preserved as `/data/datasets/dreamzero/robotwin_unified.raw_backup`.

## 2. Train Smoke Checkpoints

```bash
cd /data/dreamzero_mot

SMOKE_TEST=1 ARCH=joint \
ROBOTWIN_DATA_ROOT=/data/datasets/dreamzero/robotwin_unified_dreamzero \
WAN22_CKPT_DIR=/data/checkpoints/dreamzero/Wan2.2-TI2V-5B \
bash scripts/train/robotwin_training_wan22.sh

SMOKE_TEST=1 ARCH=mot \
ROBOTWIN_DATA_ROOT=/data/datasets/dreamzero/robotwin_unified_dreamzero \
WAN22_CKPT_DIR=/data/checkpoints/dreamzero/Wan2.2-TI2V-5B \
bash scripts/train/robotwin_training_wan22.sh
```

Smoke outputs used by the live test:

```bash
/data/checkpoints/dreamzero/dreamzero_robotwin_wan22_joint_smoke/checkpoint-2
/data/checkpoints/dreamzero/dreamzero_robotwin_wan22_mot_smoke/checkpoint-2
```

## 3. Install RoboTwin Environment

Keep this separate from the DreamZero uv environment. The commands below are
the path used on this machine; they leave `/data/dreamzero/.venv` untouched.

```bash
sudo apt-get update
sudo apt-get install -y vulkan-tools libvulkan1 ffmpeg

mkdir -p /data/envs /data/dreamzero_mot/third_party

source /root/miniconda3/etc/profile.d/conda.sh
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
conda create -y -p /data/envs/robotwin310 python=3.10
conda activate /data/envs/robotwin310

python -m pip install -U pip setuptools wheel

cd /data/dreamzero_mot/third_party
git clone https://github.com/huggingface/lerobot.git
git clone https://github.com/RoboTwin-Platform/RoboTwin.git

cd /data/dreamzero_mot/third_party/lerobot
git checkout 0e6114ac36e23038fafbbcaed89c2917aeb00fc5
python -m pip install -e . --no-deps --ignore-requires-python
/data/envs/robotwin310/bin/python /data/dreamzero_mot/example/robotwin/patch_lerobot_py310.py \
  --lerobot-root /data/dreamzero_mot/third_party/lerobot

cd /data/dreamzero_mot/third_party/RoboTwin
bash script/_install.sh
python -m pip install -e /data/openpi/packages/openpi-client
python -m pip install \
  websockets imageio imageio-ffmpeg pandas pyarrow pyyaml \
  draccus==0.10.0 einops safetensors cmake \
  "setuptools<81,>=71" "numpy==1.26.4" "opencv-python-headless==4.11.0.86" "warp-lang==1.0.2"
```

Download assets. Full RoboTwin assets:

```bash
cd /data/dreamzero_mot/third_party/RoboTwin
bash script/_download_assets.sh
```

For the `demo_clean` smoke task used here, random background textures are not
needed. This smaller asset download is enough for `beat_block_hammer`:

```bash
cd /data/dreamzero_mot/third_party/RoboTwin/assets
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="TianxingChen/RoboTwin2.0",
    allow_patterns=["embodiments.zip", "objects.zip"],
    local_dir=".",
    repo_type="dataset",
    resume_download=True,
)
PY
unzip -o embodiments.zip && rm -f embodiments.zip
unzip -o objects.zip && rm -f objects.zip
cd /data/dreamzero_mot/third_party/RoboTwin
python script/update_embodiment_config_path.py
```

Environment checks:

```bash
/data/envs/robotwin310/bin/python -c "import gymnasium, sapien, lerobot; from lerobot.envs.robotwin import RoboTwinEnv"

PYTHONPATH=/data/dreamzero_mot:/data/dreamzero_mot/third_party/RoboTwin:/data/dreamzero_mot/third_party/lerobot/src:/data/openpi/packages/openpi-client/src:$PYTHONPATH \
CUDA_VISIBLE_DEVICES=2 /data/envs/robotwin310/bin/python - <<'PY'
from example.robotwin.client import make_robotwin_env

env = make_robotwin_env("beat_block_hammer", 0, 5)
obs, info = env.reset(seed=0)
print(type(env), sorted(obs["pixels"]), obs["agent_pos"].shape, info)
env.close()
PY
```

For Hugging Face mirror downloads:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Pinned third-party checkout:

```text
LeRobot repo: https://github.com/huggingface/lerobot.git
LeRobot commit: 0e6114ac36e23038fafbbcaed89c2917aeb00fc5
RoboTwin repo: https://github.com/RoboTwin-Platform/RoboTwin.git
RoboTwin commit: 0aeea2d669c0f8516f4d5785f0aa33ba812c14b4
Recorded: 2026-05-15
```

At the time this pin was recorded, the local LeRobot checkout also had
uncommitted changes outside the RoboTwin env wrapper:

```text
M src/lerobot/datasets/streaming_dataset.py
M src/lerobot/motors/motors_bus.py
M src/lerobot/processor/pipeline.py
M src/lerobot/utils/io_utils.py
```

If those edits are needed for a run, preserve them as a separate patch or commit
before recreating the checkout elsewhere.

## 4. Server Only

Run the DreamZero websocket server from the original uv environment:

```bash
cd /data/dreamzero_mot

CUDA_VISIBLE_DEVICES=1 /data/dreamzero/.venv/bin/torchrun --standalone --nproc_per_node 1 \
  socket_test_optimized_aloha_x5lite_bimanual.py \
  --model_path /data/checkpoints/dreamzero/dreamzero_robotwin_wan22_joint_smoke/checkpoint-2 \
  --host 0.0.0.0 \
  --port 8000 \
  --max-chunk-size 4
```

## 5. Dataset Replay Client

This does not require RoboTwin simulation. It validates the websocket contract
using converted dataset frames and sends split state keys.

```bash
cd /data/dreamzero_mot

/data/dreamzero/.venv/bin/python example/robotwin/client.py \
  --mode dataset \
  --remote-host 127.0.0.1 \
  --remote-port 8000 \
  --episodes 5 \
  --task-name "Lift the bottle with narrow top head-up from the table" \
  --output-dir /data/checkpoints/dreamzero/robotwin_eval_runs/dataset_replay
```

## 6. Live RoboTwin Client

Use the isolated RoboTwin environment for simulation:

```bash
cd /data/dreamzero_mot

CUDA_VISIBLE_DEVICES=2 \
PYTHONPATH=/data/dreamzero_mot:/data/dreamzero_mot/third_party/RoboTwin:/data/dreamzero_mot/third_party/lerobot/src:/data/openpi/packages/openpi-client/src:$PYTHONPATH \
/data/envs/robotwin310/bin/python example/robotwin/client.py \
  --mode robotwin \
  --remote-host 127.0.0.1 \
  --remote-port 8000 \
  --episodes 1 \
  --robotwin-task beat_block_hammer \
  --episode-length 300 \
  --max-steps 300 \
  --open-loop-horizon 1 \
  --output-dir /data/checkpoints/dreamzero/robotwin_eval_runs/manual_live
```

The client maps RoboTwin cameras as:

- `head_camera -> video.cam_high`
- `left_camera -> video.cam_left`
- `right_camera -> video.cam_right`

It sends split state keys, avoiding the packed `observation.state` fallback.
Returned actions must have shape `(N, 14)` in `[left7, right7]` order.

## 7. Parallel RoboTwin Eval

The first parallel eval path does not use Ray and does not modify
`third_party`. It runs a controller process, multiple RoboTwin subprocess
workers, and one DreamZero websocket server that receives batched observations.

`run_robotwin_eval.sh` is the only recommended shell entrypoint. It defaults to
parallel eval; serial benchmark/smoke shell modes were removed to keep this
folder focused on normal RoboTwin eval.

Run one task with the 6/7 server and env workers on card 7:

```bash
cd /data/dreamzero_mot

SERVER_GPU=6,7 CLIENT_GPU=7 TASK=beat_block_hammer NUM_ENVS=8 \
EPISODES=8 SAVE_VIDEO=0 OPEN_LOOP_HORIZON=8 \
bash example/robotwin/run_robotwin_eval.sh
```

`NUM_ENVS` is the main parallelism knob. On this machine, use `4/8/16/24` for
capacity experiments and watch GPU memory/utilization.

Task selection:

```bash
# Single task.
SERVER_GPU=6,7 CLIENT_GPU=7 TASK=beat_block_hammer NUM_ENVS=8 \
bash example/robotwin/run_robotwin_eval.sh

# Multiple tasks. Tasks run sequentially; each task uses same-task env parallelism.
SERVER_GPU=6,7 CLIENT_GPU=7 TASKS=beat_block_hammer,pick_dual_bottles NUM_ENVS=4 \
bash example/robotwin/run_robotwin_eval.sh

# All RoboTwin eval tasks from third_party/RoboTwin/task_config/_eval_step_limit.yml.
SERVER_GPU=6,7 CLIENT_GPU=7 TASKS=all NUM_ENVS=8 \
bash example/robotwin/run_robotwin_eval.sh

# Print the resolved task list without launching server/envs.
TASKS=all LIST_TASKS=1 bash example/robotwin/run_robotwin_eval.sh
```

Dry-run the env worker path without the policy server:

```bash
cd /data/dreamzero_mot

SERVER_GPU=6,7 CLIENT_GPU=7 TASK=beat_block_hammer NUM_ENVS=1 \
DRY_RUN_ACTIONS=1 EPISODES=1 EPISODE_LENGTH=8 MAX_STEPS=1 \
bash example/robotwin/run_robotwin_eval.sh
```

Run a small real smoke:

```bash
cd /data/dreamzero_mot

SERVER_GPU=6,7 CLIENT_GPU=7 TASK=beat_block_hammer NUM_ENVS=4 \
EPISODES=4 SAVE_VIDEO=0 OPEN_LOOP_HORIZON=8 \
bash example/robotwin/run_robotwin_eval.sh
```

Useful variables:

```bash
CKPT=/data/checkpoints/dreamzero/dreamzero_robotwin
OUTPUT_ROOT=/data/checkpoints/dreamzero/robotwin_eval_runs/my_run
SERVER_GPU=6,7
CLIENT_GPU=7
NUM_ENVS=8
TASK=beat_block_hammer
TASKS=beat_block_hammer,pick_dual_bottles
TASKS=all
EPISODES=8
OPEN_LOOP_HORIZON=8
SAVE_VIDEO=0
PORT=8100
```

`SERVER_GPU` replaces the older `SERVER_CUDA` name. `CLIENT_GPU` replaces the
older `ENV_CUDA`/`CLIENT_CUDA` names. The old names still work as aliases, but
new commands should use `SERVER_GPU` and `CLIENT_GPU`.

Parallel eval outputs keep the existing layout:

```text
${OUTPUT_ROOT}/beat_block_hammer/episode_000000.json
${OUTPUT_ROOT}/beat_block_hammer/summary.json
${OUTPUT_ROOT}/report.json
${OUTPUT_ROOT}/report.csv
${OUTPUT_ROOT}/logs/env_worker_0.log
```

Each episode JSON records `reset_time`, `infer_wait_time`, `env_step_time`,
`get_obs_time`, `episode_wall_time`, and `infer_payload_size_bytes`. Task
summaries include batch infer time, per-env infer time, episode length
mean/std/min/max, and sync idle steps.

V1 constraints:

- Batch entries must share the same task and prompt.
- The controller uses fixed-size synchronized waves. Done envs stay inactive
  until the wave ends, preserving the server temporal cache batch shape.
- RTC batch, Ray, pipeline overlap, per-env virtual websocket sessions, and
  parallel video saving are intentionally out of scope.

## Troubleshooting

- Docker must expose graphics capabilities for SAPIEN:
  `-e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics`.
- Check Vulkan with `vulkaninfo --summary`.
- A `missing pytorch3d` warning is acceptable for this RGB observation smoke
  path. Install PyTorch3D only if you enable RoboTwin paths that require it.
- If server startup is slow, inspect
  `/data/checkpoints/dreamzero/robotwin_eval_runs/*/logs/server.log`.
- If `PORT=8000` is already serving an old model, the test script exits before
  starting a new server. Stop the old server or choose another `PORT`.
- If client reset fails, inspect RoboTwin assets under
  `/data/dreamzero_mot/third_party/RoboTwin`.
