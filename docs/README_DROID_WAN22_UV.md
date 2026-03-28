# DreamZero DROID Wan2.2-5B: UV Environment Setup

This README is a `uv`-first setup note for training **DreamZero on the DROID dataset with the Wan2.2-TI2V-5B backbone**.

It is distilled from:

- `README.md`
- `docs/WAN22_BACKBONE.md`
- `scripts/train/droid_training_wan22.sh`

Compared with the main README, this version is focused on **training** and intentionally skips **TensorRT / ModelOpt** inference-only packages.

## 1. Assumptions

- Repo path: `/data/dreamzero`
- Python: `3.11`
- GPU runtime: CUDA `12.9+`
- Training target: DROID + Wan2.2-TI2V-5B
- This guide is for **Linux**

If you want online logging, run `wandb login` first.
If you do not want online logging, set:

```bash
export WANDB_MODE=offline
```

For Hugging Face downloads, make sure you have access to the required repos and set:

```bash
export HF_TOKEN=<your_hf_token>
```

## 2. Create the UV Environment

```bash
cd /data/dreamzero

# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate the venv
uv venv --python 3.11 .venv
source .venv/bin/activate

# Basic build helpers used by flash-attn and other native packages
uv pip install --upgrade pip setuptools wheel ninja packaging
```

## 3. Install Training Dependencies

### 3.1 Install CUDA PyTorch

```bash
cd /data/dreamzero
source .venv/bin/activate

uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --torch-backend cu129
```

### 3.2 Install DreamZero training dependencies

```bash
cd /data/dreamzero
source .venv/bin/activate

uv pip install -r docs/requirements_droid_wan22_uv.txt
uv pip install -e . --no-deps
```

### 3.3 Install CUDA Toolkit for `flash-attn` builds

If `which nvcc` returns nothing, you only have the NVIDIA driver/runtime and still need the CUDA development toolkit.

This section is mainly needed when you want to build `flash-attn` from source.

Ubuntu 22.04 example:

```bash
apt-get update
apt-get install -y wget gnupg

cd /tmp
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update

apt-get install -y cuda-toolkit-12-9
```

Set the CUDA environment variables:

```bash
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
hash -r
```

Verify the toolkit is visible:

```bash
which nvcc
nvcc --version
echo $CUDA_HOME
```

To make the variables persistent:

```bash
cat >> ~/.bashrc <<'EOF'
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
EOF

source ~/.bashrc
```

### 3.4 Optional: FlashAttention

`flash-attn` is a performance optimization, not a hard runtime requirement for this repo.
The DreamZero code falls back to PyTorch SDPA when `flash-attn` is unavailable.

Check whether `nvcc` exists first:

```bash
which nvcc
echo ${CUDA_HOME:-not_set}
```

If `nvcc` is missing, skip this step for now and continue with training setup.
If `nvcc` exists, install `flash-attn` with:

```bash
cd /data/dreamzero
source .venv/bin/activate

MAX_JOBS=8 CUDA_HOME=/usr/local/cuda-12.9 uv pip install --no-build-isolation flash-attn
```

### 3.5 Optional: Transformer Engine

The upstream README marks this as **GB200 only**. For normal H100 training, you can skip it.

```bash
cd /data/dreamzero
source .venv/bin/activate

uv pip install --no-build-isolation "transformer_engine[pytorch]"
```

## 4. Quick Sanity Check

```bash
cd /data/dreamzero
source .venv/bin/activate

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
which nvcc || true
python -c "import deepspeed, diffusers, transformers; print('core imports ok')"
python -c "import groot; print('dreamzero package ok')"
```

Optional checks:

```bash
python -c "import flash_attn; print('flash-attn ok')" || true
python -c "import transformer_engine; print('transformer_engine ok')" || true
```

## 5. Download Training Assets

### 5.1 DROID dataset

Recommended command:

```bash
cd /data/dreamzero
source .venv/bin/activate

python scripts/data/download_droid_hf.py --local-dir ./data/droid_lerobot --max-workers 1
```

Fallback command from the original README:

```bash
huggingface-cli download GEAR-Dreams/DreamZero-DROID-Data --repo-type dataset --local-dir ./data/droid_lerobot
```

### 5.2 Wan2.2-TI2V-5B checkpoint

```bash
cd /data/dreamzero
source .venv/bin/activate

huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./checkpoints/Wan2.2-TI2V-5B
```

### 5.3 CLIP image encoder from Wan2.1

Wan2.2-TI2V-5B does not include the CLIP image encoder used by the DreamZero training script.

```bash
cd /data/dreamzero
source .venv/bin/activate

huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./checkpoints/Wan2.1-I2V-14B-480P
```

### 5.4 UMT5 tokenizer

```bash
cd /data/dreamzero
source .venv/bin/activate

huggingface-cli download google/umt5-xxl --local-dir ./checkpoints/umt5-xxl
```

## 6. Launch Training

### 6.1 Common environment variables

```bash
cd /data/dreamzero
source .venv/bin/activate

export DREAMZERO_ROOT=/data/dreamzero
export DROID_DATA_ROOT=/data/dreamzero/data/droid_lerobot
export WAN22_CKPT_DIR=/data/dreamzero/checkpoints/Wan2.2-TI2V-5B
export IMAGE_ENCODER_DIR=/data/dreamzero/checkpoints/Wan2.1-I2V-14B-480P
export TOKENIZER_DIR=/data/dreamzero/checkpoints/umt5-xxl
export OUTPUT_DIR=/data/dreamzero/checkpoints/dreamzero_droid_wan22_lora
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NUM_GPUS=4
```

### 6.2 LoRA / smoke-test style run

This is the closest match to the current Wan2.2 README path and is useful to validate that the environment is correct.

```bash
cd /data/dreamzero
source .venv/bin/activate

bash scripts/train/droid_training_wan22.sh
```

Notes:

- This script currently uses `train_architecture=lora`
- It is configured as a short run (`max_steps=100`)
- It is good for confirming that data loading, model loading, and distributed launch all work

### 6.3 Full training run

If you want the longer full-finetuning recipe:

```bash
cd /data/dreamzero
source .venv/bin/activate

export OUTPUT_DIR=/data/dreamzero/checkpoints/dreamzero_droid_wan22_full_finetune
export PER_DEVICE_BS=1
export GLOBAL_BATCH_SIZE=4
export DEEPSPEED_CFG=zero2_offload

bash scripts/train/droid_training_full_finetune_wan22.sh
```

## 7. What Is Different From the Main README

- Uses `uv venv` + `uv pip`
- Installs PyTorch with `--torch-backend cu129`
- Skips `tensorrt`, `nvidia-modelopt`, and `nvidia-modelopt-core`
- Adds a dedicated requirements file for the DROID + Wan2.2 training path
- Assumes you want a training environment first, not TensorRT inference/export

## 8. Troubleshooting

### `flash-attn` build fails

If the error contains `nvcc was not found` or `CUDA_HOME environment variable is not set`, your machine has a working NVIDIA driver/runtime but does not currently expose the CUDA development toolkit needed to compile `flash-attn`.

In that case, you have two choices:

- Skip `flash-attn` and continue training. DreamZero will fall back to SDPA, but training may be slower.
- Install a CUDA toolkit / use a CUDA `devel` image, then set `CUDA_HOME` and retry.

- Check that the venv is activated: `which python`
- Check CUDA visibility: `python -c "import torch; print(torch.version.cuda)"`
- Make sure `ninja` is installed in the venv
- Retry with a smaller build parallelism:

```bash
MAX_JOBS=4 uv pip install --no-build-isolation flash-attn
```

### Script still uses the wrong Python

The Wan2.2 training scripts in this repo now prefer the active virtual environment. If you still need to force it:

```bash
export PYTHON_BIN=$(which python)
```

Then rerun the training script.
