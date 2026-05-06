# DreamZero MOT: Bake the UV Environment Into a Docker Image

This note explains how to make the `uv` runtime and Python environment travel
with a Docker image, instead of creating `.venv` under the live project
directory every time.

It complements:

- `docs/README_DROID_WAN22_UV.md`
- `docs/requirements_droid_wan22_uv.txt`

## 1. Recommended Layout

Use image-owned paths for tools and dependencies:

```text
/usr/local/bin/uv                  # uv and uvx
/opt/venvs/dreamzero-mot           # Python virtual environment baked into image
/opt/dreamzero_mot_deps            # Build-time dependency metadata
```

Use platform storage only for mutable content:

```text
<user storage>/dreamzero_mot        # Code uploaded by SFTP / rsync
<user storage>/datasets             # Datasets
<user storage>/checkpoints          # Model weights and training outputs
```

Avoid putting the baked venv under `/data`, `/mnt/nianfs`, or the SFTP project
directory. Those paths are commonly mounted by the platform at runtime, and a
mount can hide files that were originally inside the image.

## 2. Build From the RDMA Base Image

Assume the RDMA-ready base image tar is:

```bash
/data/yesai_ubuntu_base_3.0_rdma.tar
```

Load it on a machine that has Docker:

```bash
docker load -i /data/yesai_ubuntu_base_3.0_rdma.tar
docker images | grep yesai_ubuntu_base
```

The expected base tag is:

```text
yesai_ubuntu_base:3.0-rdma
```

From the repo root:

```bash
cd /data/dreamzero_mot
mkdir -p docker/uv-image
```

Create `docker/uv-image/Dockerfile`:

```dockerfile
FROM yesai_ubuntu_base:3.0-rdma

SHELL ["/bin/bash", "-lc"]

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_INSTALL_DIR=/usr/local/bin
ENV VIRTUAL_ENV=/opt/venvs/dreamzero-mot
ENV PATH=/opt/venvs/dreamzero-mot/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin
ENV PYTHONUNBUFFERED=1

# Default runtime repo path. Override this on AI Station if your uploaded code
# lives elsewhere.
ENV DREAMZERO_ROOT=/workspace/dreamzero_mot
ENV PYTHONPATH=/workspace/dreamzero_mot

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    wget \
    build-essential \
    ninja-build \
    pkg-config \
 && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh \
 && uv --version

WORKDIR /opt/dreamzero_mot_deps

# Keep the build context small. This image bakes dependencies, not live code.
COPY pyproject.toml README.md ./
COPY docs/requirements_droid_wan22_uv.txt docs/requirements_droid_wan22_uv.txt

RUN uv venv --python 3.11 "${VIRTUAL_ENV}" \
 && uv pip install --upgrade pip setuptools wheel ninja packaging \
 && uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --torch-backend cu129 \
 && uv pip install -r docs/requirements_droid_wan22_uv.txt \
 && rm -rf /root/.cache/uv

RUN cat >/etc/profile.d/dreamzero-uv.sh <<'EOF'
export VIRTUAL_ENV=/opt/venvs/dreamzero-mot
export PATH=/opt/venvs/dreamzero-mot/bin:/usr/local/bin:$PATH
export DREAMZERO_ROOT=${DREAMZERO_ROOT:-/workspace/dreamzero_mot}
export PYTHONPATH=$DREAMZERO_ROOT:${PYTHONPATH:-}
EOF

WORKDIR /workspace
```

Build and export:

```bash
cd /data/dreamzero_mot

docker build \
  -f docker/uv-image/Dockerfile \
  -t yesai_ubuntu_base:3.0-rdma-uv \
  .

docker save yesai_ubuntu_base:3.0-rdma-uv \
  -o /data/yesai_ubuntu_base_3.0_rdma_uv.tar
```

Upload this tar to AI Station:

```text
/data/yesai_ubuntu_base_3.0_rdma_uv.tar
```

## 3. Runtime Usage on AI Station

After starting a container from the baked image, activate the venv and point
`DREAMZERO_ROOT` to the uploaded project directory.

Example:

```bash
source /opt/venvs/dreamzero-mot/bin/activate

cd /mnt/nianfs/user-fs/2023133163/liuf/dreamzero_mot
export DREAMZERO_ROOT=$PWD
export PYTHONPATH=$PWD:${PYTHONPATH:-}

python -c "import sys; print(sys.executable)"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import deepspeed, diffusers, transformers; print('core imports ok')"
python -c "import groot; print('dreamzero local code ok')"
```

If AI Station exposes your user storage with a different absolute path, replace
the `cd` path with the real path shown in the platform shell.

## 4. Why the Project Is Not Installed Into the Image

This Dockerfile intentionally bakes dependencies but not the live source tree.
That keeps the workflow simple:

- Docker image: stable OS packages, RDMA libraries, `uv`, Python packages
- SFTP / rsync: changing code
- User storage: datasets, checkpoints, logs

When running from the uploaded repo root, Python sees the current project first
through `PYTHONPATH=$PWD`, so code changes take effect without rebuilding the
image.

If you need console entry points or package metadata, run this after uploading
the code:

```bash
source /opt/venvs/dreamzero-mot/bin/activate
cd /mnt/nianfs/user-fs/2023133163/liuf/dreamzero_mot
uv pip install -e . --no-deps
```

This step is quick because the heavy dependencies are already baked into
`/opt/venvs/dreamzero-mot`.

## 5. Optional CUDA Native Packages

`docs/README_DROID_WAN22_UV.md` treats `flash-attn` and Transformer Engine as
optional. If you want to bake them into the image too, add the install commands
after the main dependency install step.

Example for `flash-attn`, assuming the image has `nvcc` and `CUDA_HOME`:

```dockerfile
ENV CUDA_HOME=/usr/local/cuda-12.9
ENV PATH=/usr/local/cuda-12.9/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH}

RUN MAX_JOBS=8 uv pip install --no-build-isolation flash-attn \
 && rm -rf /root/.cache/uv
```

Example for Transformer Engine:

```dockerfile
ENV CUDA_HOME=/usr/local/cuda-12.9
ENV NVTE_CUDA_INCLUDE_PATH=/usr/local/cuda-12.9/include

RUN uv pip install --no-build-isolation "transformer_engine[pytorch]==2.13.0" \
 && rm -rf /root/.cache/uv
```

At runtime, if using the TE backend, expose the cuDNN libraries bundled in the
venv:

```bash
export LD_LIBRARY_PATH=/opt/venvs/dreamzero-mot/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
```

## 6. If You Are Editing Inside an Existing Container

If you already have a running development container and plan to save it as an
AI Station image, use the same paths:

```bash
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl git wget build-essential ninja-build pkg-config

curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

export VIRTUAL_ENV=/opt/venvs/dreamzero-mot
export PATH=$VIRTUAL_ENV/bin:/usr/local/bin:$PATH

uv venv --python 3.11 "$VIRTUAL_ENV"
uv pip install --upgrade pip setuptools wheel ninja packaging
uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --torch-backend cu129

cd /path/to/dreamzero_mot
uv pip install -r docs/requirements_droid_wan22_uv.txt

cat >/etc/profile.d/dreamzero-uv.sh <<'EOF'
export VIRTUAL_ENV=/opt/venvs/dreamzero-mot
export PATH=/opt/venvs/dreamzero-mot/bin:/usr/local/bin:$PATH
export DREAMZERO_ROOT=${DREAMZERO_ROOT:-/workspace/dreamzero_mot}
export PYTHONPATH=$DREAMZERO_ROOT:${PYTHONPATH:-}
EOF
```

Then save the development environment as an image in AI Station. If you stop
the development environment before saving the image, these changes may be lost.

## 7. SFTP Ignore List

Do not upload local virtual environments or caches with VS Code SFTP:

```json
{
  "ignore": [
    ".venv",
    "__pycache__",
    "*.pyc",
    ".git",
    "wandb",
    "logs",
    "checkpoints",
    "datasets",
    "data",
    ".cache"
  ]
}
```

The baked environment should come from the image. The uploaded project directory
should contain source code and lightweight config files only.

## 8. Quick Checklist

- `uv` exists at `/usr/local/bin/uv`
- venv exists at `/opt/venvs/dreamzero-mot`
- `which python` points to `/opt/venvs/dreamzero-mot/bin/python`
- code is uploaded separately by SFTP / rsync
- `PYTHONPATH` points to the uploaded repo root
- datasets and checkpoints stay on user storage, not inside the image
