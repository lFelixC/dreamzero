# RoboTwin Eval On DreamZero

This directory contains the current RoboTwin evaluation path for DreamZero:
a DreamZero websocket policy server plus synchronized RoboTwin subprocess
workers. The recommended local entrypoint is:

```bash
bash example/robotwin/run_robotwin_eval.sh
```

The launcher starts the policy server, waits for it to become ready, runs the
RoboTwin controller/workers, writes reports, and stops the server when the run
finishes.

Assumed local paths:

```text
/data/dreamzero                 DreamZero repo
/data/dreamzero/.venv               DreamZero server Python environment
/data/envs/robotwin310              RoboTwin client Python environment
/data/checkpoints/dreamzero         DreamZero checkpoints and eval outputs
```

## 1. Install Local Eval Environment

Run this once on the machine that will execute RoboTwin simulation. On a local
single-machine eval setup, this is the same machine that hosts the DreamZero
server.

```bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/dreamzero}"
ROBOTWIN_ENV="${ROBOTWIN_ENV:-/data/envs/robotwin310}"
THIRD_PARTY="${REPO_ROOT}/third_party"

sudo apt-get update
sudo apt-get install -y git git-lfs unzip ffmpeg vulkan-tools libvulkan1

mkdir -p /data/envs "${THIRD_PARTY}"

source /root/miniconda3/etc/profile.d/conda.sh
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
if [[ ! -x "${ROBOTWIN_ENV}/bin/python" ]]; then
  conda create -y -p "${ROBOTWIN_ENV}" python=3.10
fi
conda activate "${ROBOTWIN_ENV}"

python -m pip install -U pip setuptools wheel

if [[ ! -d "${THIRD_PARTY}/lerobot/.git" ]]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/huggingface/lerobot.git "${THIRD_PARTY}/lerobot"
fi
git -C "${THIRD_PARTY}/lerobot" fetch --all --tags
git -C "${THIRD_PARTY}/lerobot" checkout 0e6114ac36e23038fafbbcaed89c2917aeb00fc5
python -m pip install -e "${THIRD_PARTY}/lerobot" --no-deps --ignore-requires-python
"${ROBOTWIN_ENV}/bin/python" "${REPO_ROOT}/example/robotwin/patch_lerobot_py310.py" \
  --lerobot-root "${THIRD_PARTY}/lerobot"

if [[ ! -d "${THIRD_PARTY}/RoboTwin/.git" ]]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/RoboTwin-Platform/RoboTwin.git "${THIRD_PARTY}/RoboTwin"
fi
git -C "${THIRD_PARTY}/RoboTwin" fetch --all --tags
git -C "${THIRD_PARTY}/RoboTwin" checkout 0aeea2d669c0f8516f4d5785f0aa33ba812c14b4
cd "${THIRD_PARTY}/RoboTwin"
bash script/_install.sh

python -m pip install \
  msgpack websockets imageio imageio-ffmpeg pandas pyarrow pyyaml \
  draccus==0.10.0 einops safetensors "cmake<4.2.0,>=3.29.0.1" \
  "setuptools<81,>=71" "numpy==1.26.4" \
  "opencv-python-headless==4.11.0.86" "warp-lang==1.0.2"
```

If `bash script/_install.sh` fails while installing curobo with
`CUDA_HOME environment variable is not set`, the container can see the GPU
driver but does not have the CUDA Toolkit or `nvcc`. Install the matching CUDA
Toolkit with apt, then resume from curobo only:

```bash
conda activate /data/envs/robotwin310

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
PY
```

Use the matching toolkit package. For example, if `torch.version.cuda` prints
`12.1`, install CUDA 12.1:

```bash
apt-get update
apt-get install -y wget gnupg ca-certificates build-essential ninja-build

source /etc/os-release
distro="${ID}${VERSION_ID/./}"

wget "https://developer.download.nvidia.com/compute/cuda/repos/${distro}/x86_64/cuda-keyring_1.1-1_all.deb"
dpkg -i cuda-keyring_1.1-1_all.deb

apt-get update
apt-get install -y cuda-toolkit-12-1

export CUDA_HOME=/usr/local/cuda-12.1
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

which nvcc
nvcc --version
```

If `torch.version.cuda` prints `12.4`, use `cuda-toolkit-12-4` and
`/usr/local/cuda-12.4` instead. After `nvcc` is available, do not rerun the
whole RoboTwin install script; just finish the curobo editable install:

```bash
cd /data/dreamzero/third_party/RoboTwin/envs/curobo
python -m pip install -e . --no-build-isolation
```

Pinned third-party revisions:

```text
LeRobot:  0e6114ac36e23038fafbbcaed89c2917aeb00fc5
RoboTwin: 0aeea2d669c0f8516f4d5785f0aa33ba812c14b4
```

The source clones skip Git LFS smudge because eval only needs code here.
RoboTwin assets are downloaded explicitly in section 2.

## 2. Download Assets And Check The Client

Download RoboTwin assets on the simulation/client machine. Full assets are
recommended for normal eval:

```bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/dreamzero}"
ROBOTWIN_ROOT="${REPO_ROOT}/third_party/RoboTwin"

cd "${ROBOTWIN_ROOT}"
bash script/_download_assets.sh
python script/update_embodiment_config_path.py
```

If Hugging Face is slow from your network, set a mirror before downloading:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Run a client-side import and reset check:

```bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/dreamzero}"
ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-/data/envs/robotwin310/bin/python}"
ROBOTWIN_TASK_CONFIG="${ROBOTWIN_TASK_CONFIG:-demo_clean}"

PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/RoboTwin:${REPO_ROOT}/third_party/lerobot/src:${REPO_ROOT}/third_party/lerobot:${PYTHONPATH:-}" \
ROBOTWIN_TASK_CONFIG="${ROBOTWIN_TASK_CONFIG}" \
CUDA_VISIBLE_DEVICES="${CLIENT_GPU:-0}" \
"${ROBOTWIN_PYTHON}" - <<'PY'
from example.robotwin.robotwin_fast_env import make_robotwin_env

env = make_robotwin_env("beat_block_hammer", 0, 8)
obs, info = env.reset(seed=0)
print(type(env).__name__, sorted(obs["pixels"]), obs["agent_pos"].shape, info)
env.close()
PY
```

The expected cameras are:

```text
head_camera
left_camera
right_camera
```

DreamZero maps them as:

```text
head_camera  -> video.cam_high
left_camera  -> video.cam_left
right_camera -> video.cam_right
```

## 3. Run Local Parallel Eval

Set the checkpoint and GPU layout explicitly. `SERVER_GPU` is used by the
DreamZero policy server. `CLIENT_GPU` is used by RoboTwin env workers and can
be a comma-separated list; workers are assigned round-robin.

Verify the DreamZero server environment before a real eval:

```bash
test -x /data/dreamzero/.venv/bin/python
test -x /data/dreamzero/.venv/bin/torchrun
/data/dreamzero/.venv/bin/python - <<'PY'
import torch
import websockets

print("dreamzero server env ok", torch.__version__, websockets.__version__)
PY
```

If your server environment lives elsewhere, override `DREAMZERO_PYTHON` and
`DREAMZERO_TORCHRUN` when launching `run_robotwin_eval.sh`.

List available RoboTwin eval tasks:

```bash
cd /data/dreamzero

TASKS=all LIST_TASKS=1 bash example/robotwin/run_robotwin_eval.sh
```

Dry-run the env-worker path without launching the policy server:

```bash
cd /data/dreamzero

SERVER_GPU=6,7 CLIENT_GPU=7 \
TASK=beat_block_hammer NUM_ENVS=1 \
DRY_RUN_ACTIONS=1 EPISODES=1 EPISODE_LENGTH=8 MAX_STEPS=1 \
bash example/robotwin/run_robotwin_eval.sh
```

Run a small real local eval:

```bash
cd /data/dreamzero

CKPT=/data/checkpoints/dreamzero/dreamzero_robotwin \
SERVER_GPU=6,7 CLIENT_GPU=7 \
TASK=beat_block_hammer NUM_ENVS=4 EPISODES=4 \
SAVE_VIDEO=0 PORT=8100 \
bash example/robotwin/run_robotwin_eval.sh
```

Run all RoboTwin eval tasks:

```bash
cd /data/dreamzero

CKPT=/data/checkpoints/dreamzero/dreamzero_robotwin \
OUTPUT_ROOT=/data/checkpoints/dreamzero/robotwin_eval_runs/full_local_eval \
SERVER_GPU=6,7 CLIENT_GPU=7 \
TASKS=all NUM_ENVS=8 EPISODES=50 \
SAVE_VIDEO=0 PORT=8100 \
bash example/robotwin/run_robotwin_eval.sh
```

Useful launcher variables:

```text
CKPT                         checkpoint dir; if it contains checkpoint-10000, that subdir is used
OUTPUT_ROOT                  eval output root
SERVER_GPU                   DreamZero server CUDA_VISIBLE_DEVICES
SERVER_NPROC                 server process count; defaults to number of SERVER_GPU entries
CLIENT_GPU                   RoboTwin worker CUDA_VISIBLE_DEVICES list
NUM_ENVS                     synchronized env worker count
TASK                         one RoboTwin task
TASKS                        comma-separated tasks, or all
EPISODES                     episodes per task
SEED_START                   first candidate seed
EPISODE_LENGTH               override task step limit; 0 reads RoboTwin limits
MAX_STEPS                    max policy requests per episode; 0 derives from episode length
RESET_RETRIES                RoboTwin reset retry count
DRY_RUN_ACTIONS              skip policy server and send zero-action chunks when 1
EXPERT_FILTER_MAX_CANDIDATES expert-success seed search budget
OPEN_LOOP_HORIZON            zero-action chunk length for DRY_RUN_ACTIONS=1
MAX_CHUNK_SIZE               max returned policy action chunk
SAVE_VIDEO                   save client rollout videos when nonzero
VIDEO_FPS                    client rollout video fps
CLIENT_IMAGE_RESOLUTION      none, auto, or HxW
ROBOTWIN_TASK_CONFIG         demo_clean or demo_randomized
PORT                         websocket server port
HOST                         host used by the local controller to reach the server
SERVER_HOST                  interface that the local server binds
MASTER_PORT                  torchrun process-group port
SERVER_TIMEOUT               seconds to wait for server startup
KEEP_SERVER                  leave the server running when nonzero
PROFILE                      include detailed controller profile fields when 1
PROGRESS                     plain, tqdm, or none
ROBOTWIN_PROGRESS_INTERVAL   seconds between in-wave progress heartbeats
DREAMZERO_PYTHON             DreamZero Python executable
DREAMZERO_TORCHRUN           DreamZero torchrun executable
ROBOTWIN_ENV                 RoboTwin conda env path
ROBOTWIN_PYTHON              RoboTwin Python executable
```

For real policy eval, action chunk length is controlled by `MAX_CHUNK_SIZE`.
`OPEN_LOOP_HORIZON` only affects zero-action dry runs.
`ROBOTWIN_TASK_CONFIG=demo_clean` evaluates the clean/easy RoboTwin setting.
Use `ROBOTWIN_TASK_CONFIG=demo_randomized` for the randomized/hard setting.

Outputs:

```text
${OUTPUT_ROOT}/run_config.json
${OUTPUT_ROOT}/timings.json
${OUTPUT_ROOT}/report.json
${OUTPUT_ROOT}/report.csv
${OUTPUT_ROOT}/logs/server_${PORT}.log
${OUTPUT_ROOT}/logs/env_worker_${WORKER_ID}.log
${OUTPUT_ROOT}/server_outputs/
${OUTPUT_ROOT}/${TASK}/summary.json
${OUTPUT_ROOT}/${TASK}/episode_000000.json
```

Worker logs are written for `WORKER_ID=0..NUM_ENVS-1`.

Each episode JSON records reset time, expert-filter time, inference wait time,
env step time, observation render time, episode wall time, action shapes, seed,
prompt, success, and error state. Task and global reports include success rate,
batch infer time, per-env infer time, episode length statistics, sync idle
steps, and per-client-GPU summaries.

Current V1 constraints:

```text
Tasks run sequentially.
Within one task, episodes run in fixed-size synchronized waves.
All entries in one policy batch share the same task and prompt.
Completed envs stay inactive until the wave ends to preserve temporal-cache batch shape.
RTC batch, Ray, pipeline overlap, and per-env virtual websocket sessions are out of scope.
```

One wave is one synchronized batch of RoboTwin envs. With `EPISODES=40` and
`NUM_ENVS=40`, each task has one wave. With `EPISODES=100` and `NUM_ENVS=40`,
each task has three waves: 40, 40, and 20 episodes. Inside a wave, all active
envs send one batched policy request per inference step.

Progress logs are written to stderr. `PROGRESS=plain` prints robust SSH-friendly
lines and periodic "still waiting" heartbeats during long worker reset, expert
filter, server inference, env step, and video-save phases. `PROGRESS=tqdm`
uses live bars when `tqdm` is installed and falls back to plain logs otherwise.

Troubleshooting:

```text
Docker must expose graphics capabilities for SAPIEN:
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

Check Vulkan:
  vulkaninfo --summary

If the server does not become ready:
  tail -120 ${OUTPUT_ROOT}/logs/server_${PORT}.log

If an env worker exits during reset:
  tail -120 ${OUTPUT_ROOT}/logs/env_worker_${WORKER_ID}.log

If the client reset check prints "Falling back to local RoboTwinEnv
compatibility wrapper" with a LeRobot Python syntax error:
  /data/envs/robotwin310/bin/python /data/dreamzero/example/robotwin/patch_lerobot_py310.py \
    --lerobot-root /data/dreamzero/third_party/lerobot

If a port is already serving an old model:
  choose a new PORT or stop the old torchrun process.
```

## 4. Run Remote H200 Server And 4090 Client

Use this layout when you want H200 GPUs to serve DreamZero inference and a
separate 4090 machine to run RoboTwin simulation. Install sections 1 and 2 on
the 4090 client machine. The H200 machine needs the DreamZero repo,
`/data/dreamzero/.venv`, and the checkpoint; it does not need the RoboTwin
simulation environment for serving.

Network requirements:

```text
The 4090 client must reach the H200 server IP and websocket PORT.
Use 10GbE or faster if possible; 1GbE works for smoke tests but can bottleneck large batches.
Keep CLIENT_IMAGE_RESOLUTION=none unless you explicitly want client-side resizing.
```

Start the policy server on the H200 machine:

```bash
set -euo pipefail

cd /data/dreamzero

CKPT="${CKPT:-/data/checkpoints/dreamzero/dreamzero_robotwin}"
SERVER_GPU="${SERVER_GPU:-0,1}"
if [[ -z "${SERVER_NPROC:-}" ]]; then
  SERVER_NPROC="$(python - <<'PY' "${SERVER_GPU}"
import sys

print(sum(1 for item in sys.argv[1].split(",") if item.strip()))
PY
)"
fi
PORT="${PORT:-8100}"
MASTER_PORT="${MASTER_PORT:-29610}"
MAX_CHUNK_SIZE="${MAX_CHUNK_SIZE:-24}"
SERVER_OUTPUT_ROOT="${SERVER_OUTPUT_ROOT:-/data/checkpoints/dreamzero/robotwin_eval_runs/remote_server_outputs}"

if [[ -d "${CKPT}/checkpoint-10000" && ! -f "${CKPT}/config.json" ]]; then
  CKPT="${CKPT}/checkpoint-10000"
fi

CUDA_VISIBLE_DEVICES="${SERVER_GPU}" \
/data/dreamzero/.venv/bin/torchrun \
  --nnodes 1 \
  --nproc_per_node "${SERVER_NPROC}" \
  --master_addr 127.0.0.1 \
  --master_port "${MASTER_PORT}" \
  socket_test_optimized_aloha_x5lite_bimanual.py \
  --model_path "${CKPT}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --max-chunk-size "${MAX_CHUNK_SIZE}" \
  --output-root "${SERVER_OUTPUT_ROOT}"
```

Check from the 4090 machine that the H200 port is reachable:

```bash
H200_HOST="<H200_IP_OR_HOSTNAME>"
PORT="${PORT:-8100}"

/data/envs/robotwin310/bin/python - <<'PY' "${H200_HOST}" "${PORT}"
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
with socket.create_connection((host, port), timeout=10):
    print(f"reachable: {host}:{port}")
PY
```

Run the RoboTwin client/controller on the 4090 machine:

```bash
set -euo pipefail

cd /data/dreamzero

H200_HOST="<H200_IP_OR_HOSTNAME>"
PORT="${PORT:-8100}"
ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-/data/envs/robotwin310/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/checkpoints/dreamzero/robotwin_eval_runs/remote_4090_client}"
CLIENT_GPU="${CLIENT_GPU:-0,1,2,3,4,5,6,7}"
ROBOTWIN_TASK_CONFIG="${ROBOTWIN_TASK_CONFIG:-demo_clean}"

PYTHONPATH="/data/dreamzero:/data/dreamzero/third_party/RoboTwin:/data/dreamzero/third_party/lerobot/src:/data/dreamzero/third_party/lerobot:${PYTHONPATH:-}" \
"${ROBOTWIN_PYTHON}" example/robotwin/parallel_eval.py \
  --remote-host "${H200_HOST}" \
  --remote-port "${PORT}" \
  --tasks "${TASKS:-beat_block_hammer}" \
  --episodes "${EPISODES:-8}" \
  --num-envs "${NUM_ENVS:-8}" \
  --env-cuda "${CLIENT_GPU}" \
  --worker-python "${ROBOTWIN_PYTHON}" \
  --output-dir "${OUTPUT_ROOT}" \
  --task-config "${ROBOTWIN_TASK_CONFIG}" \
  --episode-length "${EPISODE_LENGTH:-0}" \
  --max-steps "${MAX_STEPS:-0}" \
  --seed-start "${SEED_START:-0}" \
  --open-loop-horizon "${OPEN_LOOP_HORIZON:-8}" \
  --reset-retries "${RESET_RETRIES:-5}" \
  --expert-filter-max-candidates "${EXPERT_FILTER_MAX_CANDIDATES:-1000}" \
  --checkpoint-label "dreamzero_robotwin_h200_server" \
  --client-image-resolution "${CLIENT_IMAGE_RESOLUTION:-none}" \
  --progress "${PROGRESS:-plain}" \
  --progress-interval "${ROBOTWIN_PROGRESS_INTERVAL:-30}"
```

For the full task set from the 4090 client, use the same direct controller
entrypoint with larger `TASKS`, `EPISODES`, and `NUM_ENVS`:

```bash
set -euo pipefail

cd /data/dreamzero

H200_HOST="<H200_IP_OR_HOSTNAME>"
PORT="${PORT:-8100}"
ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-/data/envs/robotwin310/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/checkpoints/dreamzero/robotwin_eval_runs/remote_full_eval}"
CLIENT_GPU="${CLIENT_GPU:-0,1,2,3,4,5,6,7}"
ROBOTWIN_TASK_CONFIG="${ROBOTWIN_TASK_CONFIG:-demo_clean}"

PYTHONPATH="/data/dreamzero:/data/dreamzero/third_party/RoboTwin:/data/dreamzero/third_party/lerobot/src:/data/dreamzero/third_party/lerobot:${PYTHONPATH:-}" \
"${ROBOTWIN_PYTHON}" example/robotwin/parallel_eval.py \
  --remote-host "${H200_HOST}" \
  --remote-port "${PORT}" \
  --tasks all \
  --episodes "${EPISODES:-50}" \
  --num-envs "${NUM_ENVS:-16}" \
  --env-cuda "${CLIENT_GPU}" \
  --worker-python "${ROBOTWIN_PYTHON}" \
  --output-dir "${OUTPUT_ROOT}" \
  --task-config "${ROBOTWIN_TASK_CONFIG}" \
  --episode-length "${EPISODE_LENGTH:-0}" \
  --max-steps "${MAX_STEPS:-0}" \
  --seed-start "${SEED_START:-0}" \
  --open-loop-horizon "${OPEN_LOOP_HORIZON:-8}" \
  --reset-retries "${RESET_RETRIES:-5}" \
  --expert-filter-max-candidates "${EXPERT_FILTER_MAX_CANDIDATES:-1000}" \
  --checkpoint-label "dreamzero_robotwin_h200_server" \
  --client-image-resolution "${CLIENT_IMAGE_RESOLUTION:-none}" \
  --progress "${PROGRESS:-plain}" \
  --progress-interval "${ROBOTWIN_PROGRESS_INTERVAL:-30}"
```

Remote tuning notes:

```text
Start with NUM_ENVS=8, then try 16/24/32 while watching H200 utilization and network throughput.
Use CLIENT_GPU=0,1,2,3,4,5,6,7 on an 8-card 4090 node.
Use PROGRESS=plain for robust SSH logs, or PROGRESS=tqdm if tqdm is installed and you want live bars.
Add --save-video only for small debugging runs; video writing can dominate client-side time.
If the H200 server throws an internal websocket error, keep the server log open and restart the server before rerunning.
If the 4090 client disconnects after a long infer call, check network stability and websocket reachability first.
```
