GT_ROOT=/data/datasets/robotset/dreamzero_converted/clean_kitchen_pick_towel_scene_3  # One converted dataset root to evaluate
CHECKPOINT=/data/checkpoints/dreamzero/DreamZero-DROID  # DreamZero checkpoint used by the batch run
OUTPUT_ROOT=/data/dreamzero/video_eval/video_results  # Parent directory that will contain the batch run folder
RUN_NAME=batch_pick_towel_dreamzero  # Batch output directory name under output_root
MAX_SAMPLES=50  # Number of samples to process in this batch run
PORT=5002  # Websocket server port; local torch.distributed port is auto-derived from this by default

CUDA_VISIBLE_DEVICES=6 python /data/dreamzero/video_eval/run_batch_eval.py \
  --gt_root "$GT_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --output_root "$OUTPUT_ROOT" \
  --run_name "$RUN_NAME" \
  --max_samples "$MAX_SAMPLES" \
  --port "$PORT"
