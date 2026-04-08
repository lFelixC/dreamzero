GT_ROOT=/data/datasets/robotset/dreamzero_converted/clean_kitchen_pick_towel_scene_3  # One converted dataset root to evaluate
CHECKPOINT=/data/checkpoints/dreamzero/dreamzero_droid_wan22_5B_full_finetune/checkpoint-8000  # DreamZero checkpoint used by the batch run
OUTPUT_ROOT=/data/dreamzero/video_eval/video_results  # Parent directory that will contain the batch run folder
RUN_NAME=batch_pick_towel_ckpt_test # Batch output directory name under output_root
MAX_SAMPLES=50  # Number of samples to process in this batch run

CUDA_VISIBLE_DEVICES=7 python /data/dreamzero/video_eval/run_batch_eval.py \
  --gt_root "$GT_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --output_root "$OUTPUT_ROOT" \
  --run_name "$RUN_NAME" \
  --max_samples "$MAX_SAMPLES"
