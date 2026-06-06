# RoboTwin Eval On DreamZero

This directory contains the RoboTwin evaluation path for DreamZero. The current
default is Lingbot-style eval:

- one RoboTwin task is evaluated one episode at a time;
- every episode has its own expert-filter result, prompt, policy reset,
  `session_id`, and observation history;
- `NUM_ENVS=1` is required in Lingbot mode;
- multi-GPU speedup is task-level parallelism: one task owns one policy server,
  one port, one master port, one output directory, and one GPU slot.

The recommended local entrypoint is:

```bash
cd /data/dreamzero_mot
bash example/robotwin/run_robotwin_eval.sh
```

The launcher starts the DreamZero websocket policy server, waits for it to
become ready, runs the RoboTwin controller/worker, writes reports, and stops
the server when the run finishes.

Assumed local paths:

```text
/data/dreamzero_mot                                 DreamZero repo
/data/dreamzero/.venv                               DreamZero server Python environment
/data/envs/robotwin310                              RoboTwin client Python environment
/data/checkpoints/dreamzero/dreamzero_robotwin/checkpoint-30000
                                                    default eval checkpoint
/data/checkpoints/dreamzero/robotwin_eval_runs      default eval output parent
```

## 1. Install Local Eval Environment

Run this once on the machine that executes RoboTwin simulation.

```bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/dreamzero_mot}"
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

Pinned third-party revisions:

```text
LeRobot:  0e6114ac36e23038fafbbcaed89c2917aeb00fc5
RoboTwin: 0aeea2d669c0f8516f4d5785f0aa33ba812c14b4
```

If `bash script/_install.sh` fails with `CUDA_HOME environment variable is not
set`, install the CUDA Toolkit matching `torch.version.cuda`, then finish only
the curobo editable install:

```bash
conda activate /data/envs/robotwin310
python - <<'PY'
import torch
print(torch.version.cuda)
PY

export CUDA_HOME=/usr/local/cuda-12.1
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

cd /data/dreamzero_mot/third_party/RoboTwin/envs/curobo
python -m pip install -e . --no-build-isolation
```

Use `/usr/local/cuda-12.4` or the matching toolkit if your PyTorch CUDA version
is not 12.1.

## 2. Download Assets And Check RoboTwin

Download RoboTwin assets on the simulation/client machine:

```bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/dreamzero_mot}"
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

REPO_ROOT="${REPO_ROOT:-/data/dreamzero_mot}"
ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-/data/envs/robotwin310/bin/python}"
ROBOTWIN_TASK_CONFIG="${ROBOTWIN_TASK_CONFIG:-demo_clean}"

PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/RoboTwin:${REPO_ROOT}/third_party/lerobot/src:${REPO_ROOT}/third_party/lerobot:${PYTHONPATH:-}" \
ROBOTWIN_TASK_CONFIG="${ROBOTWIN_TASK_CONFIG}" \
CUDA_VISIBLE_DEVICES="${CLIENT_GPU:-0}" \
"${ROBOTWIN_PYTHON}" - <<'PY'
from example.robotwin.robotwin_fast_env import make_robotwin_env

env = make_robotwin_env("beat_block_hammer", 0, 8)
obs, info = env.reset(seed=10000)
print(type(env).__name__, sorted(obs["pixels"]), obs["agent_pos"].shape, info)
env.close()
PY
```

Camera mapping:

```text
head_camera  -> video.cam_high
left_camera  -> video.cam_left
right_camera -> video.cam_right
```

## 3. Local Lingbot-Style Eval

Check the DreamZero server environment:

```bash
test -x /data/dreamzero/.venv/bin/python
test -x /data/dreamzero/.venv/bin/torchrun
/data/dreamzero/.venv/bin/python - <<'PY'
import torch
import websockets
print("dreamzero server env ok", torch.__version__, websockets.__version__)
PY
```

List RoboTwin eval tasks:

```bash
cd /data/dreamzero_mot

TASKS=all LIST_TASKS=1 bash example/robotwin/run_robotwin_eval.sh
```

Dry-run the env-worker path without launching the policy server. This still
runs RoboTwin expert filtering and env setup, but sends zero actions:

```bash
cd /data/dreamzero_mot

DRY_RUN_ACTIONS=1 \
TASK=beat_block_hammer \
EPISODES=2 \
NUM_ENVS=1 \
SEED_START=10000 \
MAX_STEPS=1 \
EXPERT_FILTER_MAX_CANDIDATES=20 \
OUTPUT_ROOT=/tmp/dreamzero_robotwin_lingbot_dryrun \
bash example/robotwin/run_robotwin_eval.sh
```

Run the checkpoint-30000 smoke used for validating this eval path:

```bash
cd /data/dreamzero_mot

CKPT=/data/checkpoints/dreamzero/dreamzero_robotwin/checkpoint-30000 \
SERVER_GPU=4 \
CLIENT_GPU=5 \
SERVER_NPROC=1 \
TASK=beat_block_hammer \
EPISODES=2 \
NUM_ENVS=1 \
SEED_START=10000 \
MAX_STEPS=1 \
MAX_CHUNK_SIZE=8 \
EXPERT_FILTER_MAX_CANDIDATES=50 \
SAVE_VIDEO=0 \
PORT=8100 \
MASTER_PORT=29610 \
OUTPUT_ROOT=/data/checkpoints/dreamzero/robotwin_eval_runs/lingbot_style_ckpt30000_smoke \
bash example/robotwin/run_robotwin_eval.sh
```

`MAX_STEPS=1` is only a fast link test. A 0/2 success rate in this smoke is not
a model-quality signal.

Run one full task with Lingbot semantics:

```bash
cd /data/dreamzero_mot

CKPT=/data/checkpoints/dreamzero/dreamzero_robotwin/checkpoint-30000 \
SERVER_GPU=4 \
CLIENT_GPU=5 \
SERVER_NPROC=1 \
TASK=beat_block_hammer \
EPISODES=100 \
NUM_ENVS=1 \
SEED_START=10000 \
MAX_STEPS=0 \
MAX_CHUNK_SIZE=24 \
SAVE_VIDEO=0 \
PORT=8100 \
MASTER_PORT=29610 \
OUTPUT_ROOT=/data/checkpoints/dreamzero/robotwin_eval_runs/beat_block_hammer_ckpt30000_lingbot \
bash example/robotwin/run_robotwin_eval.sh
```

Useful launcher variables:

```text
CKPT                         checkpoint dir; default checkpoint-30000
OUTPUT_ROOT                  eval output root
SERVER_GPU                   DreamZero server CUDA_VISIBLE_DEVICES
SERVER_NPROC                 server process count; defaults to number of SERVER_GPU entries
CLIENT_GPU                   RoboTwin worker CUDA_VISIBLE_DEVICES
NUM_ENVS                     must be 1 in Lingbot mode
EVAL_MODE                    lingbot by default; batch keeps the legacy wave evaluator
TASK                         one RoboTwin task
TASKS                        comma-separated tasks, or all
EPISODES                     episodes per task; default 100
SEED_START                   first candidate seed; default 10000
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
ROBOTWIN_PROGRESS_INTERVAL   seconds between progress heartbeats
DREAMZERO_PYTHON             DreamZero Python executable
DREAMZERO_TORCHRUN           DreamZero torchrun executable
ROBOTWIN_ENV                 RoboTwin conda env path
ROBOTWIN_PYTHON              RoboTwin Python executable
```

## 4. Task-Level Multi-GPU Eval

Use task-level parallelism for all-task eval. Each task gets an independent
`run_robotwin_eval.sh` process and an independent websocket server. Do not
share one server across multiple concurrent tasks.

One GPU per server slot:

```bash
cd /data/dreamzero_mot

CKPT=/data/checkpoints/dreamzero/dreamzero_robotwin/checkpoint-30000 \
TASKS=all \
EPISODES=100 \
SEED_START=10000 \
SERVER_GPUS=4,5,6,7 \
CLIENT_GPUS=4,5,6,7 \
START_PORT=8200 \
START_MASTER_PORT=29700 \
OUTPUT_ROOT=/data/checkpoints/dreamzero/robotwin_eval_runs/all_tasks_ckpt30000_lingbot \
bash example/robotwin/run_robotwin_eval_multigpu.sh
```

Two GPUs per server slot:

```bash
cd /data/dreamzero_mot

CKPT=/data/checkpoints/dreamzero/dreamzero_robotwin/checkpoint-30000 \
TASKS=all \
EPISODES=100 \
SEED_START=10000 \
SERVER_GPU_GROUPS='0,1;2,3' \
CLIENT_GPU_GROUPS='4;5' \
START_PORT=8200 \
START_MASTER_PORT=29700 \
OUTPUT_ROOT=/data/checkpoints/dreamzero/robotwin_eval_runs/all_tasks_ckpt30000_lingbot_2gpu_server \
bash example/robotwin/run_robotwin_eval_multigpu.sh
```

The task-level launcher writes:

```text
${OUTPUT_ROOT}/task_parallel_report.json
${OUTPUT_ROOT}/logs/pids_*.txt
${OUTPUT_ROOT}/logs/status_*.txt
${OUTPUT_ROOT}/000_${TASK}/report.json
${OUTPUT_ROOT}/000_${TASK}/${TASK}/episode_000000.json
```

## 5. Direct Remote Server And Client

The local launcher is preferred. Use this only when the policy server and
RoboTwin simulation run on different machines.

Start one single-session policy server on the inference machine:

```bash
set -euo pipefail

cd /data/dreamzero_mot

CKPT="${CKPT:-/data/checkpoints/dreamzero/dreamzero_robotwin/checkpoint-30000}"
SERVER_GPU="${SERVER_GPU:-0}"
SERVER_NPROC="${SERVER_NPROC:-1}"
PORT="${PORT:-8100}"
MASTER_PORT="${MASTER_PORT:-29610}"
MAX_CHUNK_SIZE="${MAX_CHUNK_SIZE:-24}"
SERVER_OUTPUT_ROOT="${SERVER_OUTPUT_ROOT:-/data/checkpoints/dreamzero/robotwin_eval_runs/remote_server_outputs}"

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

Run one task from the RoboTwin simulation machine:

```bash
set -euo pipefail

cd /data/dreamzero_mot

H200_HOST="<H200_IP_OR_HOSTNAME>"
PORT="${PORT:-8100}"
ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-/data/envs/robotwin310/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/checkpoints/dreamzero/robotwin_eval_runs/remote_beat_block_hammer_ckpt30000}"
CLIENT_GPU="${CLIENT_GPU:-0}"
ROBOTWIN_TASK_CONFIG="${ROBOTWIN_TASK_CONFIG:-demo_clean}"

PYTHONPATH="/data/dreamzero_mot:/data/dreamzero_mot/third_party/RoboTwin:/data/dreamzero_mot/third_party/lerobot/src:/data/dreamzero_mot/third_party/lerobot:${PYTHONPATH:-}" \
"${ROBOTWIN_PYTHON}" example/robotwin/parallel_eval.py \
  --remote-host "${H200_HOST}" \
  --remote-port "${PORT}" \
  --tasks "${TASKS:-beat_block_hammer}" \
  --episodes "${EPISODES:-100}" \
  --num-envs 1 \
  --eval-mode lingbot \
  --env-cuda "${CLIENT_GPU}" \
  --worker-python "${ROBOTWIN_PYTHON}" \
  --output-dir "${OUTPUT_ROOT}" \
  --task-config "${ROBOTWIN_TASK_CONFIG}" \
  --episode-length "${EPISODE_LENGTH:-0}" \
  --max-steps "${MAX_STEPS:-0}" \
  --seed-start "${SEED_START:-10000}" \
  --open-loop-horizon "${OPEN_LOOP_HORIZON:-24}" \
  --reset-retries "${RESET_RETRIES:-5}" \
  --expert-filter-max-candidates "${EXPERT_FILTER_MAX_CANDIDATES:-1000}" \
  --checkpoint-label "dreamzero_robotwin_remote_lingbot" \
  --client-image-resolution "${CLIENT_IMAGE_RESOLUTION:-none}" \
  --progress "${PROGRESS:-plain}" \
  --progress-interval "${ROBOTWIN_PROGRESS_INTERVAL:-30}"
```

For concurrent remote all-task eval, start one server per task slot and use the
task-level launcher pattern from section 4 on the machine that can start those
servers. One shared server must not receive interleaved requests from multiple
RoboTwin tasks.

## 6. Outputs And Troubleshooting

Outputs:

```text
${OUTPUT_ROOT}/run_config.json
${OUTPUT_ROOT}/timings.json
${OUTPUT_ROOT}/report.json
${OUTPUT_ROOT}/report.csv
${OUTPUT_ROOT}/logs/server_${PORT}.log
${OUTPUT_ROOT}/logs/env_worker_0.log
${OUTPUT_ROOT}/${TASK}/summary.json
${OUTPUT_ROOT}/${TASK}/episode_000000.json
```

Each episode JSON records seed, prompt, session id, action shapes, success,
expert-filter time, inference wait time, env step time, observation render
time, and episode wall time. In Lingbot mode, `avg_wave_batch_size` should be
`1.0`.

Troubleshooting:

```text
Docker must expose graphics capabilities for SAPIEN:
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

Check Vulkan:
  vulkaninfo --summary

If the server does not become ready:
  tail -120 ${OUTPUT_ROOT}/logs/server_${PORT}.log

If an env worker exits during reset:
  tail -120 ${OUTPUT_ROOT}/logs/env_worker_0.log

If NUM_ENVS>1 fails:
  This is expected in Lingbot mode. Use task-level parallelism instead.

If a port is already serving an old model:
  choose a new PORT or stop the old torchrun process.
```

Current Lingbot-style constraints:

```text
One task process owns one active policy server session.
Episodes are serial within each task.
NUM_ENVS must be 1.
Task-level parallelism requires independent ports and output directories.
Legacy synchronized-wave batch eval is still available only with EVAL_MODE=batch.
```
