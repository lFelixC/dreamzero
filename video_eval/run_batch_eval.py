#!/usr/bin/env python3
"""Run single-dataset batch DreamZero inference and metrics end to end."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from client import ensure_gt_video_for_entry, fetch_server_metadata, run_inference_session
from common import (
    DEFAULT_DEBUG_PROMPT,
    DEFAULT_MODEL_PATH,
    DEFAULT_PORT,
    VIDEO_RESULTS_ROOT,
    append_run_log,
    build_chunk_schedule,
    ensure_dir,
    load_jsonl,
    utc_timestamp,
    write_json,
)
from evaluate_video_metrics import evaluate_video_metrics


LOGGER = logging.getLogger(__name__)
EPISODE_RE = re.compile(r"episode_(\d+)\.parquet$")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt_root", required=True, help="Single evaluation dataset root.")
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_MODEL_PATH),
        help="DreamZero checkpoint used for inference.",
    )
    parser.add_argument(
        "--output_root",
        default=str(VIDEO_RESULTS_ROOT),
        help="Root directory for batch outputs.",
    )
    parser.add_argument("--manifest_path", default=None, help="Optional manifest.jsonl override.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional limit for smoke tests.")
    parser.add_argument("--resume", action="store_true", help="Reuse existing sample outputs when present.")
    parser.add_argument("--run_name", default=None, help="Optional batch run directory name.")
    parser.add_argument(
        "--server_mode",
        choices=("auto_local", "external"),
        default="auto_local",
        help="Whether to auto-launch the local websocket server or use an external one.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server hostname.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server port.")
    parser.add_argument(
        "--dist_port",
        type=int,
        default=None,
        help="Optional torch.distributed master port for the auto-launched local server.",
    )
    parser.add_argument(
        "--metrics_eval_mode",
        choices=("composite", "per_view", "both"),
        default="both",
        help="Metrics mode used for each sample.",
    )
    parser.add_argument(
        "--lpips_net",
        choices=("squeeze", "alex", "vgg"),
        default="squeeze",
        help="LPIPS backbone used during evaluation.",
    )
    parser.add_argument(
        "--pred_wrist_mode",
        choices=("resize", "left_half"),
        default="resize",
        help="How to recover the wrist crop in per_view metrics mode.",
    )
    parser.add_argument(
        "--wait_timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for each sample pred.mp4.",
    )
    parser.add_argument(
        "--server_start_timeout",
        type=float,
        default=600.0,
        help="Seconds to wait for the auto-launched server to become ready.",
    )
    return parser


def load_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def maybe_load_episodes_map(gt_root: Path) -> dict[int, dict[str, Any]]:
    episodes_path = gt_root / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        return {}
    return {int(row["episode_index"]): row for row in load_jsonl(episodes_path)}


def resolve_task(
    entry: dict[str, Any],
    episodes_map: dict[int, dict[str, Any]],
    dataset_name: str | None,
) -> str | None:
    for key in ("task", "prompt"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    episode_index = entry.get("episode_index")
    if episode_index is not None:
        episode_row = episodes_map.get(int(episode_index))
        if episode_row is not None:
            tasks = episode_row.get("tasks")
            if isinstance(tasks, list) and tasks:
                first_task = tasks[0]
                if isinstance(first_task, str) and first_task.strip():
                    return first_task.strip()

    prompt_variants = entry.get("prompt_variants")
    if isinstance(prompt_variants, list) and prompt_variants:
        first_prompt = prompt_variants[0]
        if isinstance(first_prompt, str) and first_prompt.strip():
            return first_prompt.strip()

    if dataset_name:
        return dataset_name
    return None


def resolve_prompt(entry: dict[str, Any], task: str | None) -> str:
    for key in ("prompt",):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    prompt_variants = entry.get("prompt_variants")
    if isinstance(prompt_variants, list) and prompt_variants:
        first_prompt = prompt_variants[0]
        if isinstance(first_prompt, str) and first_prompt.strip():
            return first_prompt.strip()
    if task:
        return task
    return DEFAULT_DEBUG_PROMPT


def read_episode_row_count(parquet_path: Path) -> int:
    import pyarrow.parquet as pq

    return int(pq.read_metadata(parquet_path).num_rows)


def normalize_manifest_sample(
    entry: dict[str, Any],
    sample_index: int,
    gt_root: Path,
    manifest_path: Path,
    episodes_map: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    dataset_name = entry.get("dataset_name") or gt_root.name
    task = resolve_task(entry, episodes_map, str(dataset_name) if dataset_name else None)
    prompt = resolve_prompt(entry, task)
    normalized = dict(entry)
    normalized["sample_index"] = sample_index
    normalized["sample_id"] = str(entry["sample_id"])
    normalized["task"] = task
    normalized["prompt"] = prompt
    normalized["source_mode"] = "manifest"
    normalized["manifest_path"] = str(manifest_path)
    normalized["gt_root"] = str(gt_root)
    normalized["dataset_name"] = str(dataset_name)
    normalized["initial_frame_indices"] = entry.get("initial_frame_indices", [0])
    normalized["chunk_frame_indices"] = entry.get("chunk_frame_indices") or build_chunk_schedule(
        int(entry.get("frame_count", 0))
    )
    normalized["frame_count"] = int(entry.get("frame_count", 0))
    normalized["anchor_frame_index"] = int(entry.get("anchor_frame_index", 23))
    return normalized


def discover_manifest_samples(gt_root: Path, manifest_path: Path) -> list[dict[str, Any]]:
    rows = load_jsonl(manifest_path)
    if not rows:
        raise RuntimeError(f"No rows found in manifest: {manifest_path}")
    episodes_map = maybe_load_episodes_map(gt_root)
    return [
        normalize_manifest_sample(entry, sample_index, gt_root, manifest_path, episodes_map)
        for sample_index, entry in enumerate(rows)
    ]


def discover_scanned_samples(gt_root: Path) -> list[dict[str, Any]]:
    episodes_map = maybe_load_episodes_map(gt_root)
    parquet_paths = sorted((gt_root / "data").glob("chunk-*/episode_*.parquet"))
    if not parquet_paths:
        raise RuntimeError(
            f"No manifest found and no standard DreamZero eval structure detected under {gt_root}"
        )

    samples: list[dict[str, Any]] = []
    for sample_index, parquet_path in enumerate(parquet_paths):
        match = EPISODE_RE.search(parquet_path.name)
        if match is None:
            continue
        episode_suffix = match.group(0).replace(".parquet", ".mp4")
        episode_index = int(match.group(1))
        chunk_dir = parquet_path.parent.name
        video_paths = {
            "observation.images.exterior_image_1_left": str(
                gt_root
                / "videos"
                / chunk_dir
                / "observation.images.exterior_image_1_left"
                / episode_suffix
            ),
            "observation.images.exterior_image_2_left": str(
                gt_root
                / "videos"
                / chunk_dir
                / "observation.images.exterior_image_2_left"
                / episode_suffix
            ),
            "observation.images.wrist_image_left": str(
                gt_root
                / "videos"
                / chunk_dir
                / "observation.images.wrist_image_left"
                / episode_suffix
            ),
        }
        missing = [key for key, value in video_paths.items() if not Path(value).exists()]
        if missing:
            LOGGER.warning("Skipping episode %s because videos are missing: %s", parquet_path, missing)
            continue

        episode_row = episodes_map.get(episode_index, {})
        frame_count = int(episode_row.get("length") or read_episode_row_count(parquet_path))
        task = None
        tasks = episode_row.get("tasks")
        if isinstance(tasks, list) and tasks:
            first_task = tasks[0]
            if isinstance(first_task, str) and first_task.strip():
                task = first_task.strip()
        prompt = task or gt_root.name
        chunk_frame_indices = build_chunk_schedule(frame_count)
        anchor_frame_index = chunk_frame_indices[0][-1] if chunk_frame_indices else max(frame_count - 1, 0)
        samples.append(
            {
                "sample_index": sample_index,
                "sample_id": f"{gt_root.name}__episode_{episode_index:06d}",
                "task": task,
                "prompt": prompt,
                "parquet_path": str(parquet_path),
                "video_paths": video_paths,
                "frame_count": frame_count,
                "fps": 15,
                "anchor_frame_index": anchor_frame_index,
                "initial_frame_indices": [0],
                "chunk_frame_indices": chunk_frame_indices,
                "source_mode": "scan",
                "dataset_name": gt_root.name,
                "gt_root": str(gt_root),
                "episode_index": episode_index,
                "prompt_variants": [prompt],
            }
        )

    if not samples:
        raise RuntimeError(
            f"No manifest found and no valid DreamZero eval samples could be scanned under {gt_root}"
        )
    return samples


def discover_samples(gt_root: Path, manifest_path: Path | None, max_samples: int | None) -> tuple[list[dict[str, Any]], str]:
    if manifest_path is not None:
        samples = discover_manifest_samples(gt_root, manifest_path)
        mode = "manifest"
    else:
        auto_manifest = gt_root / "manifest.jsonl"
        if auto_manifest.exists():
            samples = discover_manifest_samples(gt_root, auto_manifest)
            mode = "manifest"
        else:
            samples = discover_scanned_samples(gt_root)
            mode = "scan"
    if max_samples is not None:
        samples = samples[:max_samples]
    for sample_index, sample in enumerate(samples):
        sample["sample_index"] = sample_index
    return samples, mode


def sample_dir_name(sample_index: int, total_samples: int) -> str:
    width = max(3, len(str(max(total_samples - 1, 0))))
    return f"sample_{sample_index:0{width}d}"


def read_metadata_server_resolution(metadata_path: Path) -> tuple[int, int] | None:
    if not metadata_path.exists():
        return None
    try:
        metadata = load_json_file(metadata_path)
        resolution = metadata.get("server_metadata", {}).get("image_resolution")
        if isinstance(resolution, list) and len(resolution) == 2:
            return int(resolution[0]), int(resolution[1])
    except Exception:
        return None
    return None


def sample_needs_server(sample_dir: Path, resume: bool) -> bool:
    pred_exists = (sample_dir / "pred.mp4").exists()
    metrics_exists = (sample_dir / "metrics_results.json").exists()
    gt_exists = (sample_dir / "gt.mp4").exists()
    if not resume or not pred_exists:
        return True
    if not metrics_exists and not gt_exists and read_metadata_server_resolution(sample_dir / "metadata.json") is None:
        return True
    return False


def start_local_server(
    checkpoint: str,
    samples_root: Path,
    host: str,
    port: int,
    dist_port: int | None,
    run_log_path: Path,
) -> tuple[subprocess.Popen[str], Any]:
    command = [
        sys.executable,
        "/data/dreamzero/video_eval/server.py",
        "--host",
        host,
        "--port",
        str(port),
        "--model-path",
        checkpoint,
        "--results-root",
        str(samples_root),
    ]
    if dist_port is not None:
        command.extend(["--dist-port", str(dist_port)])
    append_run_log(run_log_path, f"[batch] starting_local_server command={json.dumps(command)}")
    log_handle = run_log_path.open("a", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd="/data",
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    return process, log_handle


def stop_local_server(process: subprocess.Popen[str] | None, log_handle: Any | None, run_log_path: Path) -> None:
    if process is None:
        return
    if process.poll() is None:
        append_run_log(run_log_path, "[batch] stopping_local_server")
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            append_run_log(run_log_path, "[batch] local_server_terminate_timeout, killing")
            process.kill()
            process.wait(timeout=20)
    if log_handle is not None:
        log_handle.close()


def write_csv_file(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def get_summary_metric_block(metrics_results: dict[str, Any]) -> dict[str, Any]:
    if "per_view" in metrics_results:
        per_view = metrics_results["per_view"]
        return {
            "mean": per_view["overall_mean"],
            "per_frame": per_view["aggregated_per_frame"],
            "evaluated_frames": per_view["evaluated_frames"],
            "resize_applied": per_view["resize_applied"],
            "truncated": per_view["truncated"],
        }
    return {
        "mean": metrics_results["mean"],
        "per_frame": metrics_results["per_frame"],
        "evaluated_frames": metrics_results["evaluated_frames"],
        "resize_applied": metrics_results["resize_applied"],
        "truncated": metrics_results["truncated"],
    }


def compute_half_lpips(per_frame: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    if not per_frame:
        return None, None
    midpoint = max(1, len(per_frame) // 2)
    early = per_frame[:midpoint]
    late = per_frame[midpoint:]
    if not late:
        late = early
    early_value = float(sum(float(row["lpips"]) for row in early) / len(early))
    late_value = float(sum(float(row["lpips"]) for row in late) / len(late))
    return early_value, late_value


def write_overall_summary_text(path: Path, summary: dict[str, Any]) -> None:
    def render(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6f}"
        if value is None:
            return "null"
        return str(value)

    lines = [f"{key}: {render(value)}" for key, value in summary.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_batch_manifest(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)

    if args.resume and not args.run_name:
        raise ValueError("--resume requires --run_name so the batch directory is unambiguous")

    gt_root = Path(args.gt_root)
    if not gt_root.exists():
        raise FileNotFoundError(f"gt_root does not exist: {gt_root}")

    run_name = args.run_name or f"batch_{utc_timestamp()}"
    batch_dir = ensure_dir(Path(args.output_root) / run_name)
    samples_root = ensure_dir(batch_dir / "samples")
    run_log_path = batch_dir / "run.log"
    append_run_log(run_log_path, f"[batch] argv={json.dumps(sys.argv)}")

    manifest_path = Path(args.manifest_path) if args.manifest_path else None
    samples, discovery_mode = discover_samples(gt_root, manifest_path, args.max_samples)
    append_run_log(run_log_path, f"[batch] discovered_samples={len(samples)} mode={discovery_mode}")

    batch_manifest: dict[str, Any] = {
        "run_name": run_name,
        "gt_root": str(gt_root),
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "checkpoint": args.checkpoint,
        "output_root": str(Path(args.output_root)),
        "batch_dir": str(batch_dir),
        "server_mode": args.server_mode,
        "summary_metric_source": "per_view" if args.metrics_eval_mode == "both" else args.metrics_eval_mode,
        "metrics_eval_mode": args.metrics_eval_mode,
        "lpips_net": args.lpips_net,
        "pred_wrist_mode": args.pred_wrist_mode,
        "discovery_mode": discovery_mode,
        "samples": [],
    }

    total_samples = len(samples)
    server_process: subprocess.Popen[str] | None = None
    server_log_handle = None
    server_metadata: dict[str, Any] | None = None

    needs_server = any(sample_needs_server(samples_root / sample_dir_name(sample["sample_index"], total_samples), args.resume) for sample in samples)
    try:
        if args.server_mode == "auto_local" and needs_server:
            server_process, server_log_handle = start_local_server(
                checkpoint=args.checkpoint,
                samples_root=samples_root,
                host=args.host,
                port=args.port,
                dist_port=args.dist_port,
                run_log_path=run_log_path,
            )
            server_metadata = fetch_server_metadata(args.host, args.port, timeout_seconds=args.server_start_timeout)
            append_run_log(run_log_path, f"[batch] server_ready metadata={json.dumps(server_metadata, sort_keys=True)}")
        elif args.server_mode == "external" and needs_server:
            server_metadata = fetch_server_metadata(args.host, args.port, timeout_seconds=args.server_start_timeout)
            append_run_log(run_log_path, f"[batch] external_server_ready metadata={json.dumps(server_metadata, sort_keys=True)}")

        for sample in samples:
            sample_index = int(sample["sample_index"])
            sample_name = sample_dir_name(sample_index, total_samples)
            sample_dir = ensure_dir(samples_root / sample_name)
            sample_log = sample_dir / "run.log"
            record = {
                "sample_index": sample_index,
                "sample_id": sample["sample_id"],
                "task": sample.get("task"),
                "sample_dir": str(sample_dir),
                "source_mode": sample["source_mode"],
                "inference_status": "pending",
                "metrics_status": "pending",
                "pred_video": str(sample_dir / "pred.mp4"),
                "gt_video": str(sample_dir / "gt.mp4"),
                "metrics_json": str(sample_dir / "metrics_results.json"),
                "error_message": None,
                "summary_metric_lpips": None,
                "evaluated_frames": None,
                "resize_applied": None,
                "truncated": None,
            }
            append_run_log(run_log_path, f"[batch] sample_start index={sample_index} sample_id={sample['sample_id']}")

            pred_path = sample_dir / "pred.mp4"
            metrics_path = sample_dir / "metrics_results.json"
            gt_path = sample_dir / "gt.mp4"
            metadata_path = sample_dir / "metadata.json"

            try:
                if args.resume and pred_path.exists() and pred_path.stat().st_size > 0:
                    record["inference_status"] = "skipped"
                    append_run_log(run_log_path, f"[batch] inference_skipped sample_id={sample['sample_id']}")
                else:
                    if args.server_mode == "external" and server_metadata is None:
                        server_metadata = fetch_server_metadata(args.host, args.port, timeout_seconds=args.server_start_timeout)
                    run_inference_session(
                        run_dir=sample_dir,
                        host=args.host,
                        port=args.port,
                        checkpoint_path=args.checkpoint,
                        input_mode="manifest",
                        prompt_override=sample.get("prompt"),
                        num_chunks=None,
                        wait_timeout=args.wait_timeout,
                        session_id=sample_name,
                        manifest_path=Path(sample["manifest_path"]) if sample.get("manifest_path") else None,
                        manifest_entry=sample,
                        server_metadata=server_metadata,
                    )
                    record["inference_status"] = "success"

                if not gt_path.exists():
                    image_resolution = read_metadata_server_resolution(metadata_path)
                    if image_resolution is None and server_metadata is not None:
                        image_resolution = tuple(server_metadata["image_resolution"])
                    ensure_gt_video_for_entry(
                        run_dir=sample_dir,
                        entry=sample,
                        image_resolution=image_resolution,
                        log_path=sample_log,
                    )

                if args.resume and metrics_path.exists() and metrics_path.stat().st_size > 0:
                    record["metrics_status"] = "skipped"
                    metrics_results = load_json_file(metrics_path)
                    append_run_log(run_log_path, f"[batch] metrics_skipped sample_id={sample['sample_id']}")
                else:
                    metrics_results = evaluate_video_metrics(
                        pred_video=pred_path,
                        output_dir=sample_dir,
                        sample_id=str(sample["sample_id"]),
                        eval_mode=args.metrics_eval_mode,
                        lpips_net=args.lpips_net,
                        pred_wrist_mode=args.pred_wrist_mode,
                        gt_video=gt_path,
                        gt_left_video=Path(sample["video_paths"]["observation.images.exterior_image_1_left"]),
                        gt_right_video=Path(sample["video_paths"]["observation.images.exterior_image_2_left"]),
                        gt_wrist_video=Path(sample["video_paths"]["observation.images.wrist_image_left"]),
                    )
                    record["metrics_status"] = "success"

                summary_block = get_summary_metric_block(metrics_results)
                record["summary_metric_lpips"] = float(summary_block["mean"]["lpips"])
                record["evaluated_frames"] = int(summary_block["evaluated_frames"])
                record["resize_applied"] = bool(summary_block["resize_applied"])
                record["truncated"] = bool(summary_block["truncated"])
            except Exception as exc:  # pragma: no cover - runtime failure path
                message = f"{type(exc).__name__}: {exc}"
                record["error_message"] = message
                append_run_log(run_log_path, f"[batch] sample_failed sample_id={sample['sample_id']} error={message}")
                append_run_log(sample_log, f"[batch] error={message}")
                if record["inference_status"] == "pending":
                    record["inference_status"] = "failed"
                record["metrics_status"] = "failed"
            finally:
                batch_manifest["samples"].append(record)
                write_batch_manifest(batch_dir / "batch_manifest.json", batch_manifest)

        successful_metric_records = []
        for record in batch_manifest["samples"]:
            metrics_json = Path(record["metrics_json"])
            if record["metrics_status"] not in {"success", "skipped"} or not metrics_json.exists():
                continue
            try:
                metrics_results = load_json_file(metrics_json)
                summary_block = get_summary_metric_block(metrics_results)
                early_lpips, late_lpips = compute_half_lpips(summary_block["per_frame"])
                successful_metric_records.append(
                    {
                        "record": record,
                        "metrics": metrics_results,
                        "summary_block": summary_block,
                        "early_lpips": early_lpips,
                        "late_lpips": late_lpips,
                    }
                )
            except Exception as exc:  # pragma: no cover - corrupted metrics file path
                append_run_log(
                    run_log_path,
                    f"[batch] summary_skip sample_id={record['sample_id']} error={type(exc).__name__}: {exc}",
                )

        num_successful_inference = sum(
            1 for record in batch_manifest["samples"] if Path(record["pred_video"]).exists()
        )
        num_successful_metrics = len(successful_metric_records)
        num_failed_samples = sum(
            1
            for record in batch_manifest["samples"]
            if record["inference_status"] == "failed" or record["metrics_status"] == "failed"
        )

        if successful_metric_records:
            mean_lpips = float(
                sum(item["summary_block"]["mean"]["lpips"] for item in successful_metric_records) / num_successful_metrics
            )
            mean_ssim = float(
                sum(item["summary_block"]["mean"]["ssim"] for item in successful_metric_records) / num_successful_metrics
            )
            mean_psnr = float(
                sum(item["summary_block"]["mean"]["psnr"] for item in successful_metric_records) / num_successful_metrics
            )
            avg_evaluated_frames = float(
                sum(item["summary_block"]["evaluated_frames"] for item in successful_metric_records) / num_successful_metrics
            )
            truncation_rate = float(
                sum(1 for item in successful_metric_records if item["summary_block"]["truncated"]) / num_successful_metrics
            )
            resize_rate = float(
                sum(1 for item in successful_metric_records if item["summary_block"]["resize_applied"]) / num_successful_metrics
            )
            valid_early = [item["early_lpips"] for item in successful_metric_records if item["early_lpips"] is not None]
            valid_late = [item["late_lpips"] for item in successful_metric_records if item["late_lpips"] is not None]
            early_lpips = float(sum(valid_early) / len(valid_early)) if valid_early else None
            late_lpips = float(sum(valid_late) / len(valid_late)) if valid_late else None
        else:
            mean_lpips = None
            mean_ssim = None
            mean_psnr = None
            avg_evaluated_frames = None
            truncation_rate = None
            resize_rate = None
            early_lpips = None
            late_lpips = None

        lpips_drift = None
        if early_lpips is not None and late_lpips is not None:
            lpips_drift = float(late_lpips - early_lpips)

        overall_summary = {
            "num_total_samples": total_samples,
            "num_successful_inference": num_successful_inference,
            "num_successful_metrics": num_successful_metrics,
            "num_failed_samples": num_failed_samples,
            "mean_lpips": mean_lpips,
            "mean_ssim": mean_ssim,
            "mean_psnr": mean_psnr,
            "early_lpips": early_lpips,
            "late_lpips": late_lpips,
            "lpips_drift": lpips_drift,
            "avg_evaluated_frames": avg_evaluated_frames,
            "truncation_rate": truncation_rate,
            "resize_rate": resize_rate,
        }
        write_json(batch_dir / "overall_summary.json", overall_summary)
        write_overall_summary_text(batch_dir / "overall_summary.txt", overall_summary)

        task_rows = []
        grouped_tasks: dict[str, list[float]] = {}
        for item in successful_metric_records:
            task = item["record"].get("task")
            if isinstance(task, str) and task.strip():
                grouped_tasks.setdefault(task.strip(), []).append(float(item["summary_block"]["mean"]["lpips"]))
        if grouped_tasks:
            for task, values in sorted(grouped_tasks.items()):
                task_rows.append(
                    {
                        "task": task,
                        "N": len(values),
                        "mean_lpips": sum(values) / len(values),
                    }
                )
            write_csv_file(
                batch_dir / "per_task_lpips.csv",
                ["task", "N", "mean_lpips"],
                task_rows,
            )

        worst_rows = []
        worst_candidates = sorted(
            successful_metric_records,
            key=lambda item: float(item["summary_block"]["mean"]["lpips"]),
            reverse=True,
        )[:5]
        for item in worst_candidates:
            record = item["record"]
            worst_rows.append(
                {
                    "sample_id": record["sample_id"],
                    "task": record.get("task"),
                    "pred_video": record["pred_video"],
                    "gt_video": record["gt_video"],
                    "mean_lpips": item["summary_block"]["mean"]["lpips"],
                    "evaluated_frames": item["summary_block"]["evaluated_frames"],
                }
            )
        write_csv_file(
            batch_dir / "worst_5_cases.csv",
            ["sample_id", "task", "pred_video", "gt_video", "mean_lpips", "evaluated_frames"],
            worst_rows,
        )

        append_run_log(run_log_path, f"[batch] complete summary={json.dumps(overall_summary, ensure_ascii=False)}")
    finally:
        stop_local_server(server_process, server_log_handle, run_log_path)


if __name__ == "__main__":
    main()
