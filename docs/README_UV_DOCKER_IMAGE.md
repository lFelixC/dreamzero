# Docker 中安装并固化 DreamZero 的 uv 环境

这份文档用于在一个新的 Docker 容器里安装 DreamZero 训练环境，并在后续保存镜像后继续使用。

目标：

- `uv` 安装在镜像里的通用路径：`/usr/local/bin/uv`
- Python 虚拟环境安装在镜像里：`/opt/venvs/dreamzero`
- 项目训练目录固定为：`/2023133163/liuf/dreamzero`
- 数据集目录固定为：`/2023133163/datasets/dreamzero`
- checkpoint 目录固定为：`/2023133163/checkpoints/dreamzero`

不要把虚拟环境放在项目目录的 `.venv` 里。项目目录可能是平台挂载盘，保存镜像时不一定会把它作为镜像环境保存；把环境放到 `/opt/venvs/dreamzero` 更稳。

## 1. 进入容器并检查网络

```bash
curl -I https://github.com
```

如果能看到 `HTTP/2 200` 或 `HTTP/1.1 200`，说明 DNS 和外网正常。

如果报错 `Could not resolve host`，先修 DNS，例如：

```bash
cp /etc/resolv.conf /etc/resolv.conf.bak
cat > /etc/resolv.conf <<'EOF'
nameserver 223.5.5.5
nameserver 114.114.114.114
nameserver 8.8.8.8
options timeout:2 attempts:3
EOF
```

## 2. 准备项目目录

后续默认项目放在：

```bash
/2023133163/liuf/dreamzero
```

进入项目目录：

```bash
cd /2023133163/liuf/dreamzero
```

如果项目还没放到这个目录，先把代码上传、复制或 clone 到这里。该目录下应该能看到：

```bash
ls
# 应该包含 pyproject.toml、groot/、docs/、scripts/ 等
```

## 3. 安装 uv 到镜像路径

如果已经有 `uv`，可以先看版本：

```bash
which uv
uv --version
```

建议再安装一份到 `/usr/local/bin`，这样保存镜像后更稳定：

```bash
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
hash -r
uv --version
```

## 4. 创建镜像内虚拟环境

```bash
export VIRTUAL_ENV=/opt/venvs/dreamzero
export PATH=$VIRTUAL_ENV/bin:/usr/local/bin:/root/.local/bin:$PATH

uv venv --python 3.11 "$VIRTUAL_ENV"
source "$VIRTUAL_ENV/bin/activate"

python --version
which python
```

期望 `which python` 输出：

```text
/opt/venvs/dreamzero/bin/python
```

## 5. 永久配置 pip 和 uv 清华源

清华 PyPI 源：

```text
https://pypi.tuna.tsinghua.edu.cn/simple
```

把 pip 的系统级配置写到 `/etc/pip.conf`，这样后续所有 root shell 和虚拟环境里的 `pip install` 默认都会走清华源：

```bash
cat > /etc/pip.conf <<'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
EOF
```

把 uv 的 root 用户全局配置写到 `/root/.config/uv/uv.toml`，这样后续 root 用户执行 `uv pip install` 默认也会走清华源：

```bash
mkdir -p /root/.config/uv
cat > /root/.config/uv/uv.toml <<'EOF'
index-url = "https://pypi.tuna.tsinghua.edu.cn/simple"
EOF
```

再写一个 profile 脚本作为兜底，让登录 shell 也自动带上这两个环境变量：

```bash
cat > /etc/profile.d/pypi-mirror.sh <<'EOF'
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
EOF

source /etc/profile.d/pypi-mirror.sh
```

检查配置：

```bash
cat /etc/pip.conf
cat /root/.config/uv/uv.toml
echo "$PIP_INDEX_URL"
echo "$UV_DEFAULT_INDEX"
```

后续保存镜像后，这些配置文件都会保留。新容器里只要还是 root 用户，`pip install` 和 `uv pip install` 默认都会走清华源。

注意：下面安装 PyTorch 时用了 `--torch-backend cu129`，这类 CUDA 轮子可能仍会访问 PyTorch 官方源；普通 Python 包会优先走清华 PyPI 源。

## 6. 安装训练依赖

先装系统编译依赖：

```bash
apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates curl git wget \
  build-essential ninja-build pkg-config
```

进入项目目录并安装 Python 包：

```bash
cd /2023133163/liuf/dreamzero
source /opt/venvs/dreamzero/bin/activate

uv pip install --upgrade pip setuptools wheel ninja packaging

uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --torch-backend cu129

uv pip install -r docs/requirements_droid_wan22_uv.txt
uv pip install -e . --no-deps

rm -rf /root/.cache/uv
```

`uv pip install -e . --no-deps` 只把当前项目注册进环境，不重复安装大依赖。

## 7. 安装 CUDA Toolkit

`flash-attn` 编译需要 `nvcc`。如果 `which nvcc` 没有输出，说明容器里缺 CUDA 编译工具包，需要先安装 CUDA Toolkit。

先检查：

```bash
which nvcc || true
echo "${CUDA_HOME:-not_set}"
```

如果已经有 `nvcc`，直接跳到下一节。

如果没有 `nvcc`，Ubuntu 22.04 容器里安装 CUDA Toolkit 12.9：

```bash
apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates wget gnupg

cd /tmp
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb

apt-get update
apt-get install -y --no-install-recommends cuda-toolkit-12-9
```

设置环境变量：

```bash
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
hash -r
```

验证：

```bash
which nvcc
nvcc --version
echo "$CUDA_HOME"
```

注意：这里安装的是容器内的 CUDA 编译工具包，用来提供 `nvcc`。GPU 驱动仍然由宿主机/平台提供，不能只靠容器里安装 CUDA Toolkit 解决驱动问题。

## 8. 可选安装 FlashAttention

`flash-attn` 是性能优化，不是硬依赖。没有它时，代码会回退到 PyTorch SDPA，但训练可能慢一些。

如果有 `nvcc`，执行：

```bash
cd /2023133163/liuf/dreamzero
source /opt/venvs/dreamzero/bin/activate

if command -v nvcc >/dev/null 2>&1; then
  export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
elif [ -d /usr/local/cuda-12.9 ]; then
  export CUDA_HOME=/usr/local/cuda-12.9
fi

if [ -z "${CUDA_HOME:-}" ]; then
  echo "ERROR: CUDA_HOME not found; skip flash-attn install in this container."
else
  export PATH=$CUDA_HOME/bin:$PATH
  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

  MAX_JOBS=2 uv pip install --no-build-isolation flash-attn

  python -c "import flash_attn; print('flash-attn ok')"
fi
```

如果编译时内存不够或太慢，可以把 `MAX_JOBS=8` 改成 `MAX_JOBS=4`。

## 9. 写入自动环境变量

让以后进入容器后能直接用这个环境：

```bash
cat > /etc/profile.d/dreamzero-uv.sh <<'EOF'
export VIRTUAL_ENV=/opt/venvs/dreamzero
export PYTHON_BIN=/opt/venvs/dreamzero/bin/python
export PATH=/opt/venvs/dreamzero/bin:/usr/local/bin:/root/.local/bin:$PATH
export DREAMZERO_ROOT=/2023133163/liuf/dreamzero
export DATASET_ROOT=/2023133163/datasets/dreamzero
export CHECKPOINT_ROOT=/2023133163/checkpoints/dreamzero
export PYTHONPATH=/2023133163/liuf/dreamzero:${PYTHONPATH:-}
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple

if command -v nvcc >/dev/null 2>&1; then
  export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
elif [ -d /usr/local/cuda-12.9 ]; then
  export CUDA_HOME=/usr/local/cuda-12.9
fi

if [ -n "${CUDA_HOME:-}" ]; then
  export PATH=$CUDA_HOME/bin:$PATH
  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
fi
EOF

source /etc/profile.d/dreamzero-uv.sh
```

## 10. 验证环境

```bash
which uv
uv --version

which python
python -c "import sys; print(sys.executable)"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import deepspeed, diffusers, transformers; print('core imports ok')"
python -c "import groot; print('dreamzero import ok')"
python -c "import flash_attn; print('flash-attn ok')" || true

echo "$DREAMZERO_ROOT"
echo "$DATASET_ROOT"
echo "$CHECKPOINT_ROOT"
```

如果 `torch.cuda.is_available()` 是 `True`，说明 PyTorch 能看到 GPU。

如果 `import groot` 失败，通常是没有进入项目目录，或者 `PYTHONPATH` 没设置：

```bash
cd /2023133163/liuf/dreamzero
export PYTHONPATH=$PWD:${PYTHONPATH:-}
```

## 11. 训练时使用这个环境

进入项目目录：

```bash
cd /2023133163/liuf/dreamzero
source /etc/profile.d/dreamzero-uv.sh
```

因为第 9 步已经导出了 `PYTHON_BIN`、`DREAMZERO_ROOT`、`DATASET_ROOT` 和 `CHECKPOINT_ROOT`，后续 A800 脚本不需要改代码，可以直接运行：

```bash
bash scripts/train/run_dreamzero_mot_multinode.sh
```

如果跑 A800 的 2 节点脚本：

```bash
bash scripts/train/a800_train/run_mot_full_video_2node.sh
```

如果你没有执行 `source /etc/profile.d/dreamzero-uv.sh`，那就需要在命令前临时指定：

```bash
PYTHON_BIN=/opt/venvs/dreamzero/bin/python \
DREAMZERO_ROOT=/2023133163/liuf/dreamzero \
DATASET_ROOT=/2023133163/datasets/dreamzero \
CHECKPOINT_ROOT=/2023133163/checkpoints/dreamzero \
bash scripts/train/a800_train/run_mot_full_video_2node.sh
```

## 12. 保存镜像

完成安装并验证通过后，在宿主机或平台的镜像管理页面保存当前容器。

如果你能在宿主机执行 Docker 命令，可以这样保存：

```bash
docker ps
docker commit <容器ID或容器名> dreamzero-uv:latest
docker save dreamzero-uv:latest -o dreamzero-uv.tar
```

之后用 `dreamzero-uv:latest` 或 `dreamzero-uv.tar` 新建容器时，`uv` 和 `/opt/venvs/dreamzero` 都会保留。

注意：如果 `/2023133163/liuf/dreamzero` 是平台挂载目录，保存镜像只会保存环境，不一定保存这个目录里的代码、数据和 checkpoint。新容器里仍然需要保证项目目录存在。

## 13. 新容器中快速恢复

以后从保存好的镜像启动新容器后：

```bash
source /etc/profile.d/dreamzero-uv.sh
cd /2023133163/liuf/dreamzero

which python
python -c "import torch, groot; print('env ok')"
```

能通过就可以继续训练。
