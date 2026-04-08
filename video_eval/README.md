# DreamZero Video Eval

This directory contains a thin evaluation wrapper around DreamZero's existing websocket inference flow. The single-sample scripts are still available, and `run_batch_eval.py` adds a one-command batch workflow for one converted dataset root.

## Files

- `/data/dreamzero/video_eval/server.py`
  Launch the DreamZero websocket inference server and save `pred.mp4` into `results_root/session_id/`.
- `/data/dreamzero/video_eval/client.py`
  Send a debug-image request or one converted RoboSet sample to the server and save `request.json`, `metadata.json`, `run.log`, and optional `gt.mp4`.
- `/data/dreamzero/video_eval/convert_roboset_to_dreamzero.py`
  Extract RoboSet archives and convert sampled `h5 + Trial` units into mini DROID-style LeRobot datasets plus a `manifest.jsonl`.
- `/data/dreamzero/video_eval/evaluate_video_metrics.py`
  Compute LPIPS, SSIM, and PSNR for `pred.mp4` vs `gt.mp4`, or split `pred.mp4` back into `left/right/wrist` and compare per view.
- `/data/dreamzero/video_eval/run_batch_eval.py`
  Discover samples under one dataset root, run batch inference, run batch metrics, and write dataset-level summaries.

## Environment

All commands below assume the DreamZero virtual environment is active first.

```bash
cd /data/dreamzero
source /data/dreamzero/.venv/bin/activate
```

If `lpips` is missing:

```bash
cd /data/dreamzero
source /data/dreamzero/.venv/bin/activate
uv pip install lpips
```

## Single-Sample Workflow

### Start Server

`pred.mp4` is written by the server, but the run directory name comes from the client's `session_id` or `--run-id`. In practice, server and client should point at the same `results_root`.

```bash
CHECKPOINT=/data/checkpoints/dreamzero/dreamzero_droid_wan22_full_finetune/checkpoint-5000  # DreamZero inference checkpoint
RESULTS_ROOT=/data/dreamzero/video_eval/video_results  # Shared output root for server and client
PORT=5001  # Websocket port used by both server and client

CUDA_VISIBLE_DEVICES=0 python /data/dreamzero/video_eval/server.py \
  --port "$PORT" \
  --model-path "$CHECKPOINT" \
  --results-root "$RESULTS_ROOT"
```

### Debug Client

```bash
HOST=127.0.0.1  # Server hostname
PORT=5001  # Must match the server port
RUN_ID=step1_debug_smoke  # Session directory name under results_root

python /data/dreamzero/video_eval/client.py \
  --host "$HOST" \
  --port "$PORT" \
  --input-mode debug_image \
  --num-chunks 1 \
  --run-id "$RUN_ID"
```

### RoboSet Conversion

```bash
DATASET_ROOT=/data/datasets/robotset  # RoboSet archive root; script falls back to /data/datasets/roboset if needed
SAMPLE_COUNT=50  # Number of Trial samples to keep per dataset
SEED=42  # Sampling seed for reproducibility

python /data/dreamzero/video_eval/convert_roboset_to_dreamzero.py \
  --dataset-root "$DATASET_ROOT" \
  --sample-count "$SAMPLE_COUNT" \
  --seed "$SEED"
```

Converted outputs are written under:

- `/data/datasets/robotset/extracted/<dataset_name>/`
- `/data/datasets/robotset/dreamzero_converted/<dataset_name>/`

### Manifest Client

```bash
HOST=127.0.0.1  # Server hostname
PORT=5001  # Must match the server port
MANIFEST=/data/datasets/robotset/dreamzero_converted/clean_kitchen_slide_close_drawer_scene_3/manifest.jsonl  # Converted dataset manifest
SAMPLE_ID=clean_kitchen_slide_close_drawer_scene_3__clean_kitchen_close_drawer_scene_3_20230405-110426__Trial1  # One manifest sample to run

python /data/dreamzero/video_eval/client.py \
  --host "$HOST" \
  --port "$PORT" \
  --input-mode manifest \
  --manifest-path "$MANIFEST" \
  --sample-id "$SAMPLE_ID"
```

### Metrics

Composite-vs-composite metrics:

```bash
RUN_DIR=/data/dreamzero/video_eval/video_results/pick_towel_manifest_smoke  # One completed single-sample run directory

python /data/dreamzero/video_eval/evaluate_video_metrics.py \
  --pred-video "$RUN_DIR/pred.mp4" \
  --gt-video "$RUN_DIR/gt.mp4" \
  --output-dir "$RUN_DIR" \
  --lpips-net squeeze
```

Per-view metrics that split the predicted composite into `left/right/wrist`:

```bash
PRED_VIDEO=/data/dreamzero/video_eval/video_results/pick_towel_manifest_smoke/pred.mp4  # Predicted DreamZero composite video
OUTPUT_DIR=/data/dreamzero/video_eval/video_results/pick_towel_manifest_smoke_per_view  # Directory for per-view metrics outputs
LEFT_GT=/data/datasets/robotset/dreamzero_converted/clean_kitchen_pick_towel_scene_3/videos/chunk-000/observation.images.exterior_image_1_left/episode_000000.mp4  # GT left camera video
RIGHT_GT=/data/datasets/robotset/dreamzero_converted/clean_kitchen_pick_towel_scene_3/videos/chunk-000/observation.images.exterior_image_2_left/episode_000000.mp4  # GT right camera video
WRIST_GT=/data/datasets/robotset/dreamzero_converted/clean_kitchen_pick_towel_scene_3/videos/chunk-000/observation.images.wrist_image_left/episode_000000.mp4  # GT wrist camera video

python /data/dreamzero/video_eval/evaluate_video_metrics.py \
  --pred-video "$PRED_VIDEO" \
  --output-dir "$OUTPUT_DIR" \
  --eval-mode per_view \
  --lpips-net squeeze \
  --gt-left-video "$LEFT_GT" \
  --gt-right-video "$RIGHT_GT" \
  --gt-wrist-video "$WRIST_GT"
```

## Batch Workflow

### What `run_batch_eval.py` Does

Give the script one converted dataset root and it will:

1. Discover samples from `--manifest_path`, or from `<gt_root>/manifest.jsonl`, or by scanning the standard converted DreamZero layout.
2. Start a local `server.py` automatically when `--server_mode auto_local` is used.
3. Run inference sample by sample.
4. Run metrics sample by sample.
5. Write `batch_manifest.json`, `overall_summary.json`, `overall_summary.txt`, `worst_5_cases.csv`, optional `per_task_lpips.csv`, and one `samples/sample_XXX/` directory per sample.

### Key Arguments

- `--gt_root`
  One converted dataset root, for example `/data/datasets/robotset/dreamzero_converted/clean_kitchen_pick_towel_scene_3`.
- `--checkpoint`
  DreamZero checkpoint used by the auto-launched server.
- `--output_root`
  Parent directory that will contain the batch run folder.
- `--manifest_path`
  Optional explicit manifest override. If omitted, the script first tries `<gt_root>/manifest.jsonl`.
- `--max_samples`
  Useful for smoke tests when you only want the first few samples.
- `--resume`
  Reuse existing `pred.mp4` and `metrics_results.json` when present, but always regenerate summaries.
- `--run_name`
  Batch output directory name. Required together with `--resume`.
- `--server_mode`
  `auto_local` starts and stops `server.py` for you. `external` expects a server to already be running.
- `--host`
  Hostname of the websocket server. This mostly matters in `external` mode.
- `--port`
  Port of the websocket server. This must match the server you want to talk to.
- `--metrics_eval_mode`
  `both` by default. This writes both composite and per-view metrics, while dataset-level summaries use the `per_view` summary block.
- `--lpips_net`
  LPIPS backbone used during evaluation. `squeeze` is the lightest option and is used by the smoke tests here.
- `--pred_wrist_mode`
  How `per_view` evaluation reconstructs the wrist crop from the predicted composite video.

### Discovery Rules

- If `--manifest_path` is passed, that file is used.
- Otherwise, if `<gt_root>/manifest.jsonl` exists, manifest mode is used.
- Otherwise, scanner mode looks for the standard converted layout:
  `data/chunk-*/episode_*.parquet`
  `videos/chunk-*/observation.images.exterior_image_1_left/episode_*.mp4`
  `videos/chunk-*/observation.images.exterior_image_2_left/episode_*.mp4`
  `videos/chunk-*/observation.images.wrist_image_left/episode_*.mp4`
- If neither manifest mode nor scanner mode can resolve samples, the script fails loudly instead of silently skipping work.

### Batch Command: Manifest Mode Smoke Test

```bash
GT_ROOT=/data/datasets/robotset/dreamzero_converted/clean_kitchen_pick_towel_scene_3  # One converted dataset root to evaluate
CHECKPOINT=/data/checkpoints/dreamzero/dreamzero_droid_wan22_full_finetune/checkpoint-5000  # DreamZero checkpoint used by the batch run
OUTPUT_ROOT=/data/dreamzero/video_eval/video_results  # Parent directory that will contain the batch run folder
RUN_NAME=batch_pick_towel_smoke  # Batch output directory name under output_root
MAX_SAMPLES=2  # Smoke test limit so only the first 2 samples are processed

python /data/dreamzero/video_eval/run_batch_eval.py \
  --gt_root "$GT_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --output_root "$OUTPUT_ROOT" \
  --run_name "$RUN_NAME" \
  --max_samples "$MAX_SAMPLES"
```

### Batch Command: Resume Existing Run

```bash
GT_ROOT=/data/datasets/robotset/dreamzero_converted/clean_kitchen_pick_towel_scene_3  # Same dataset root used by the original run
CHECKPOINT=/data/checkpoints/dreamzero/dreamzero_droid_wan22_full_finetune/checkpoint-5000  # Same checkpoint used for inference
OUTPUT_ROOT=/data/dreamzero/video_eval/video_results  # Same parent directory as the original run
RUN_NAME=batch_pick_towel_smoke  # Existing batch directory to resume
MAX_SAMPLES=2  # Optional smoke test limit; keep it aligned with the original run when resuming

python /data/dreamzero/video_eval/run_batch_eval.py \
  --gt_root "$GT_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --output_root "$OUTPUT_ROOT" \
  --run_name "$RUN_NAME" \
  --max_samples "$MAX_SAMPLES" \
  --resume
```

### Batch Command: Scanner Mode Smoke Test

Use scanner mode when the dataset root has the standard converted `data/`, `videos/`, and optional `meta/` layout, but no `manifest.jsonl`.

```bash
GT_ROOT=/tmp/dz_scanner_pick_towel  # Scanner-mode root with data/ and videos/ but no manifest.jsonl
CHECKPOINT=/data/checkpoints/dreamzero/dreamzero_droid_wan22_full_finetune/checkpoint-5000  # DreamZero checkpoint used by the batch run
OUTPUT_ROOT=/data/dreamzero/video_eval/video_results  # Parent directory that will contain the batch run folder
RUN_NAME=batch_pick_towel_scanner_smoke  # Batch output directory name under output_root
MAX_SAMPLES=1  # Smoke test limit so only the first discovered sample is processed

python /data/dreamzero/video_eval/run_batch_eval.py \
  --gt_root "$GT_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --output_root "$OUTPUT_ROOT" \
  --run_name "$RUN_NAME" \
  --max_samples "$MAX_SAMPLES"
```

### Batch Output Layout

For a batch run under `/data/dreamzero/video_eval/video_results/<run_name>/`:

- `samples/sample_000/`
  One sample directory containing `pred.mp4`, `gt.mp4`, `request.json`, `metadata.json`, `metrics_results.json`, `metrics_results.txt`, and `run.log`.
- `batch_manifest.json`
  Run-level metadata plus one status record per sample, including `inference_status`, `metrics_status`, `pred_video`, `gt_video`, `metrics_json`, `summary_metric_lpips`, `evaluated_frames`, `resize_applied`, and `truncated`.
- `overall_summary.json`
  Machine-readable dataset summary.
- `overall_summary.txt`
  Human-readable text summary.
- `per_task_lpips.csv`
  Average LPIPS grouped by task, written only when at least one sample has a non-empty task.
- `worst_5_cases.csv`
  The five samples with the highest summary LPIPS.
- `run.log`
  Batch-level log file. In `auto_local` mode this also includes the local server output.

### `overall_summary.json` Field Meanings

- `num_total_samples`
  How many samples were discovered before any failures or skips.
- `num_successful_inference`
  How many samples produced a usable `pred.mp4`.
- `num_successful_metrics`
  How many samples produced a usable `metrics_results.json`.
- `num_failed_samples`
  How many samples failed either inference or metrics.
- `mean_lpips`
  Dataset-average LPIPS over successful metric runs. Lower is better.
- `mean_ssim`
  Dataset-average SSIM over successful metric runs. Higher is better.
- `mean_psnr`
  Dataset-average PSNR over successful metric runs. Higher is better.
- `early_lpips`
  Average LPIPS over the first half of each successfully evaluated sample.
- `late_lpips`
  Average LPIPS over the second half of each successfully evaluated sample.
- `lpips_drift`
  `late_lpips - early_lpips`. Positive means quality got worse later in the video.
- `avg_evaluated_frames`
  Average number of frames that were actually compared after truncation rules.
- `truncation_rate`
  Fraction of successful samples where pred and GT lengths differed and the comparison was truncated to the shorter one.
- `resize_rate`
  Fraction of successful samples where frame sizes differed and the predicted frames were resized before scoring.

## Verified Examples

- Single-sample debug-image run:
  `/data/dreamzero/video_eval/video_results/step1_debug_smoke/`
- Converted dataset roots:
  `/data/datasets/robotset/dreamzero_converted/clean_kitchen_pick_towel_scene_3/`
  and
  `/data/datasets/robotset/dreamzero_converted/clean_kitchen_slide_close_drawer_scene_3/`
  with `50` manifest entries each.
- Single-sample manifest runs:
  `/data/dreamzero/video_eval/video_results/pick_towel_manifest_smoke/`
  and
  `/data/dreamzero/video_eval/video_results/close_drawer_manifest_smoke/`
- Single-sample per-view metrics:
  `/data/dreamzero/video_eval/video_results/pick_towel_manifest_smoke_per_view/metrics_results.json`
  and
  `/data/dreamzero/video_eval/video_results/pick_towel_manifest_smoke_per_view/metrics_results.txt`
- Batch manifest-mode smoke test:
  `/data/dreamzero/video_eval/video_results/batch_pick_towel_smoke/`
  with `2` samples, `mean_lpips=0.38618981312302975`, `mean_ssim=0.4157841494261902`, `mean_psnr=14.57769768378883`.
- Batch scanner-mode smoke test:
  `/data/dreamzero/video_eval/video_results/batch_pick_towel_scanner_smoke/`
  with `1` sample and successful outputs under
  `/data/dreamzero/video_eval/video_results/batch_pick_towel_scanner_smoke/samples/sample_000/`.
- Resume smoke test:
  reran `batch_pick_towel_smoke` with `--resume`, confirmed inference and metrics were skipped, and regenerated the summary files in place.

## Known Limitations

- RoboSet does not expose DROID-style language annotations in the archive contents inspected here, so the converter uses fixed prompt templates derived from the dataset name.
- RoboSet `ctrl_arm` and `ctrl_ee` are copied into DROID action slots as a compatibility mapping; they are not claimed to be semantically identical to native DROID actions.
- Batch scanner mode currently targets the converted DreamZero eval layout described above; it does not try to auto-adapt to arbitrary unknown directory structures.
