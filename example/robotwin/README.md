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

## 7. One-Command Full Test

Run one RoboTwin episode for both smoke checkpoints:

```bash
cd /data/dreamzero_mot

bash example/robotwin/run_full_infer_test.sh
```

Useful overrides:

```bash
CKPT=both EPISODES=1 EPISODE_LENGTH=4 MAX_STEPS=1 OPEN_LOOP_HORIZON=4 \
SERVER_CUDA=1 CLIENT_CUDA=2 PORT=8000 SAVE_VIDEO=0 \
bash example/robotwin/run_full_infer_test.sh

CKPT=joint bash example/robotwin/run_full_infer_test.sh
CKPT=mot bash example/robotwin/run_full_infer_test.sh
MAX_STEPS=20 SAVE_VIDEO=1 bash example/robotwin/run_full_infer_test.sh
SERVER_CUDA=1 CLIENT_CUDA=2 PORT=8000 bash example/robotwin/run_full_infer_test.sh
```

Outputs are written under:

```bash
/data/checkpoints/dreamzero/robotwin_eval_runs/joint_smoke
/data/checkpoints/dreamzero/robotwin_eval_runs/mot_smoke
```

Each episode JSON records task, seed, steps, success, reward sum, action shapes,
and checkpoint metadata.

Verified smoke outputs from this machine:

- `joint_smoke`: `steps=4`, `action_shapes=[[4, 14]]`
- `mot_smoke`: `steps=4`, `action_shapes=[[4, 14]]`

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
