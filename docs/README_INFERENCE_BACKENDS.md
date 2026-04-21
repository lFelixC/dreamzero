# DreamZero Inference Backend Recipes

This note collects the runtime flags for:

- `socket_test_optimized_AR.py`
- `socket_test_optimized_aloha_x5lite_bimanual.py`
- `eval_utils/serve_dreamzero_wan22.py`

It is intended for local server startup, especially when comparing:

- `FA2` vs `TE`
- `torch.compile` disabled vs enabled

## 1. Runtime Flags

The current inference entrypoints support the following environment variables:

- `ATTENTION_BACKEND=FA2`
  Uses the FlashAttention 2 path.
- `ATTENTION_BACKEND=TE`
  Uses the Transformer Engine cuDNN attention path.
- `DISABLE_TORCH_COMPILE=true`
  Disables the model-side `torch.compile(...)` wrappers.
- `DISABLE_TORCH_COMPILE=false`
  Enables the model-side `torch.compile(...)` wrappers.
- `TORCH_COMPILE_BACKEND=cudagraphs`
  Selects the safer compile backend when compile is enabled.

Important:

- `DISABLE_TORCH_COMPILE=true` takes precedence over the entrypoint's default backend patch.
- You may still see a log like `Using torch.compile backend=cudagraphs for this entrypoint`.
- That log only shows the default backend selection. It does not mean compile is active when `DISABLE_TORCH_COMPILE=true`.

## 2. Transformer Engine on H200

This repo has been patched to work with `Transformer Engine 2.13.0` on the H200 inference path.

Install it into the DreamZero venv:

```bash
cd /data/dreamzero
source .venv/bin/activate

export CUDA_HOME=/usr/local/cuda-12.9
export NVTE_CUDA_INCLUDE_PATH=$CUDA_HOME/include

pip install --no-build-isolation "transformer_engine[pytorch]==2.13.0"
```

At runtime, expose the cuDNN libraries bundled inside the venv:

```bash
export LD_LIBRARY_PATH=/data/dreamzero/.venv/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
```

Quick sanity check:

```bash
cd /data/dreamzero
source .venv/bin/activate

python - <<'PY'
import transformer_engine
print("transformer_engine", transformer_engine.__version__)
PY
```

## 3. AR Server Recipes

All examples below use two GPUs. Replace `0,1` and `--nproc_per_node 2` as needed.

### 3.1 FA2 without `torch.compile`

```bash
cd /data/dreamzero

CUDA_VISIBLE_DEVICES=0,1 \
ATTENTION_BACKEND=FA2 \
DISABLE_TORCH_COMPILE=true \
torchrun --standalone --nproc_per_node 2 \
  socket_test_optimized_AR.py \
  --port 5000 \
  --enable-dit-cache \
  --model-path <path/to/checkpoint>
```

### 3.2 TE without `torch.compile`

```bash
cd /data/dreamzero

CUDA_VISIBLE_DEVICES=0,1 \
ATTENTION_BACKEND=TE \
DISABLE_TORCH_COMPILE=true \
LD_LIBRARY_PATH=/data/dreamzero/.venv/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-} \
torchrun --standalone --nproc_per_node 2 \
  socket_test_optimized_AR.py \
  --port 5000 \
  --enable-dit-cache \
  --model-path <path/to/checkpoint>
```

### 3.3 Compile enabled with `cudagraphs`

```bash
cd /data/dreamzero

CUDA_VISIBLE_DEVICES=0,1 \
ATTENTION_BACKEND=FA2 \
DISABLE_TORCH_COMPILE=false \
TORCH_COMPILE_BACKEND=cudagraphs \
torchrun --standalone --nproc_per_node 2 \
  socket_test_optimized_AR.py \
  --port 5000 \
  --enable-dit-cache \
  --model-path <path/to/checkpoint>
```

Notes:

- `inductor` on this machine was not the stable choice for GPU compile.
- `cudagraphs` is the recommended compile backend for the current patched entrypoints.
- If you want the smallest number of moving parts for debugging, start from `FA2 + DISABLE_TORCH_COMPILE=true`.

## 4. ALOHA Server Recipes

The same backend flags apply to the ALOHA bimanual server.

### 4.1 FA2 without `torch.compile`

```bash
cd /data/dreamzero

CUDA_VISIBLE_DEVICES=0,1 \
ATTENTION_BACKEND=FA2 \
DISABLE_TORCH_COMPILE=true \
torchrun --standalone --nproc_per_node 2 \
  socket_test_optimized_aloha_x5lite_bimanual.py \
  --port 8000 \
  --enable-dit-cache \
  --model_path <path/to/checkpoint>
```

### 4.2 TE without `torch.compile`

```bash
cd /data/dreamzero

CUDA_VISIBLE_DEVICES=0,1 \
ATTENTION_BACKEND=TE \
DISABLE_TORCH_COMPILE=true \
LD_LIBRARY_PATH=/data/dreamzero/.venv/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-} \
torchrun --standalone --nproc_per_node 2 \
  socket_test_optimized_aloha_x5lite_bimanual.py \
  --port 8000 \
  --enable-dit-cache \
  --model_path <path/to/checkpoint>
```

### 4.3 Compile enabled with `cudagraphs`

```bash
cd /data/dreamzero

CUDA_VISIBLE_DEVICES=0,1 \
ATTENTION_BACKEND=FA2 \
DISABLE_TORCH_COMPILE=false \
TORCH_COMPILE_BACKEND=cudagraphs \
torchrun --standalone --nproc_per_node 2 \
  socket_test_optimized_aloha_x5lite_bimanual.py \
  --port 8000 \
  --enable-dit-cache \
  --model_path <path/to/checkpoint>
```

## 5. Recommended Starting Points

If you only want a stable server:

- Start with `FA2 + DISABLE_TORCH_COMPILE=true`.

If you want to test Transformer Engine on Hopper / H200:

- Use `TE + DISABLE_TORCH_COMPILE=true`.
- Make sure `LD_LIBRARY_PATH` includes the venv cuDNN path.

If you want to compare compile behavior:

- Enable compile explicitly with `DISABLE_TORCH_COMPILE=false`.
- Keep `TORCH_COMPILE_BACKEND=cudagraphs`.
