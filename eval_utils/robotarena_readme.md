# DreamZero 5B RoboArena Server Guide

This note documents the exact RoboArena-compatible serving flow for the DreamZero 5B checkpoint:

- Checkpoint: `/data/checkpoints/dreamzero/dreamzero_droid_wan22_5B_full_finetune/checkpoint-8000`
- Server entrypoint: `/data/dreamzero/socket_test_optimized_AR.py`
- GPUs: `CUDA_VISIBLE_DEVICES=6,7`
- Port: `8000`

## Environment

```bash
cd /data/dreamzero
source /data/dreamzero/.venv/bin/activate
```

Optional sanity checks:

```bash
python - <<'PY'
import importlib.util
for name in ["openpi_client", "torch", "websockets"]:
    print(name, bool(importlib.util.find_spec(name)))
PY

nvidia-smi -L
```

## Launch The Server

```bash
cd /data/dreamzero
source /data/dreamzero/.venv/bin/activate

CUDA_VISIBLE_DEVICES=6,7 \
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  /data/dreamzero/socket_test_optimized_AR.py \
  --port 8000 \
  --enable-dit-cache \
  --model-path /data/checkpoints/dreamzero/dreamzero_droid_wan22_5B_full_finetune/checkpoint-8000
```

Expected behavior:

- rank 0 logs the RoboArena websocket server config
- the server listens on `0.0.0.0:8000`
- two worker processes are launched and use the two visible GPUs

Useful checks from another shell:

```bash
ss -ltnp | rg ':8000'
nvidia-smi
```

## Official RoboArena Test

Clone the official repo into `/tmp`:

```bash
cd /tmp
rm -rf /tmp/roboarena_official
git clone https://github.com/robo-arena/roboarena /tmp/roboarena_official
```

Try the upstream test first:

```bash
cd /data/dreamzero
source /data/dreamzero/.venv/bin/activate

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy no_proxy NO_PROXY

PYTHONPATH=/tmp/roboarena_official \
python /tmp/roboarena_official/scripts/test_policy_server.py
```

If the upstream script does not connect reliably because it uses the default host, run the localhost copy:

```bash
cd /data/dreamzero
source /data/dreamzero/.venv/bin/activate

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy no_proxy NO_PROXY

PYTHONPATH=/tmp/roboarena_official \
python /tmp/roboarena_official/scripts/test_policy_server_localhost.py
```

Pass conditions:

- metadata parses as official `PolicyServerConfig`
- the response is a dict
- `"actions"` exists in the response
- the action tensor has shape `(N, 8)`

## Local DreamZero Smoke Test

This validates that the RoboArena compatibility fix still preserves the original DreamZero remote-inference contract.

```bash
cd /data/dreamzero
source /data/dreamzero/.venv/bin/activate

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy no_proxy NO_PROXY

python /data/dreamzero/test_client_AR.py \
  --host 127.0.0.1 \
  --port 8000 \
  --num-chunks 1
```

Expected behavior:

- metadata validates successfully
- one initial request plus one chunk request complete
- the local client still receives a raw action array and logs action shape `(N, 8)`

## Outputs And Logs

The server writes generated videos under the checkpoint parent:

```text
/data/checkpoints/dreamzero/dreamzero_droid_wan22_5B_full_finetune/real_world_eval_gen_<date>_<index>/checkpoint-8000/
```

The local smoke test reads debug videos from:

```text
/data/dreamzero/debug_image/
```

## Shutdown

If the server is running in the foreground, stop it with:

```bash
Ctrl+C
```

If it was launched in the background, stop the matching torchrun process:

```bash
ps -ef | rg 'socket_test_optimized_AR.py|torch.distributed.run'
kill <pid>
```

## Troubleshooting

- If the official test fails to import `roboarena`, make sure `PYTHONPATH=/tmp/roboarena_official` is set.
- If the official upstream test does not connect, use `test_policy_server_localhost.py`.
- If websocket handshake or model warm-up is slow, wait for checkpoint loading to finish before running tests.
- The RoboArena compatibility fix is intentionally done in the websocket boundary so existing DreamZero callers keep receiving raw action arrays.
