#!/usr/bin/env python3
"""Evaluate LPIPS, SSIM, and PSNR for a predicted video against ground truth."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from common import ensure_dir, read_video_frames, resize_frames, split_droid_views, write_json


LOGGER = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-video", required=True, help="Predicted video path.")
    parser.add_argument("--gt-video", default=None, help="Ground-truth video path.")
    parser.add_argument("--gt-frames-dir", default=None, help="Ground-truth frames directory.")
    parser.add_argument("--output-dir", required=True, help="Directory for metrics outputs.")
    parser.add_argument("--sample-id", default=None, help="Optional sample ID override.")
    parser.add_argument(
        "--eval-mode",
        choices=("composite", "per_view", "both"),
        default="composite",
        help="Compare composite videos directly, split pred into three views, or run both.",
    )
    parser.add_argument(
        "--lpips-net",
        choices=("squeeze", "alex", "vgg"),
        default="squeeze",
        help="LPIPS backbone network. Default uses the lighter squeeze network.",
    )
    parser.add_argument(
        "--pred-wrist-mode",
        choices=("resize", "left_half"),
        default="resize",
        help="How to recover the wrist view from pred's top strip in per_view mode.",
    )
    parser.add_argument("--gt-left-video", default=None, help="GT left-view video path for per_view mode.")
    parser.add_argument("--gt-right-video", default=None, help="GT right-view video path for per_view mode.")
    parser.add_argument("--gt-wrist-video", default=None, help="GT wrist-view video path for per_view mode.")
    parser.add_argument(
        "--gt-left-frames-dir",
        default=None,
        help="GT left-view frame directory for per_view mode.",
    )
    parser.add_argument(
        "--gt-right-frames-dir",
        default=None,
        help="GT right-view frame directory for per_view mode.",
    )
    parser.add_argument(
        "--gt-wrist-frames-dir",
        default=None,
        help="GT wrist-view frame directory for per_view mode.",
    )
    parser.add_argument(
        "--export-frame-pairs",
        action="store_true",
        help="Export aligned frame pairs under output-dir/frame_pairs.",
    )
    return parser


def load_frames_source(video_path: Path | None, frames_dir: Path | None, label: str) -> np.ndarray:
    if video_path is not None:
        return read_video_frames(video_path)
    if frames_dir is None:
        raise ValueError(f"Missing frame source for {label}.")
    frame_paths = sorted(path for path in frames_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not frame_paths:
        raise RuntimeError(f"No image frames found in {frames_dir}")
    frames = []
    for path in frame_paths:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Failed to read frame {path}")
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return np.stack(frames, axis=0)


def load_gt_frames(gt_video: Path | None, gt_frames_dir: Path | None) -> np.ndarray:
    return load_frames_source(gt_video, gt_frames_dir, "composite gt")


def get_lpips_model(device: torch.device, network: str):
    try:
        import lpips  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "lpips is not installed. Install it with: "
            "source /data/dreamzero/.venv/bin/activate && uv pip install lpips"
        ) from exc
    model = lpips.LPIPS(net=network)
    model = model.to(device)
    model.eval()
    return model


def to_lpips_tensor(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
    return tensor.to(device)


def evaluate_frame_pair(
    pred_frames: np.ndarray,
    gt_frames: np.ndarray,
    lpips_model,
    device: torch.device,
    frame_pairs_dir: Path | None = None,
) -> dict[str, object]:
    original_pred_frames = int(pred_frames.shape[0])
    original_gt_frames = int(gt_frames.shape[0])
    evaluated_frames = min(original_pred_frames, original_gt_frames)

    pred_eval = pred_frames[:evaluated_frames]
    gt_eval = gt_frames[:evaluated_frames]

    resize_applied = False
    original_pred_resolution = list(pred_eval.shape[1:3])
    if pred_eval.shape[1:3] != gt_eval.shape[1:3]:
        pred_eval = resize_frames(pred_eval, gt_eval.shape[1], gt_eval.shape[2], interpolation=cv2.INTER_LINEAR)
        resize_applied = True

    per_frame: list[dict[str, float | int]] = []
    for frame_index in range(evaluated_frames):
        pred_frame = pred_eval[frame_index]
        gt_frame = gt_eval[frame_index]
        with torch.no_grad():
            lpips_value = float(
                lpips_model(
                    to_lpips_tensor(pred_frame, device),
                    to_lpips_tensor(gt_frame, device),
                )
                .detach()
                .cpu()
                .item()
            )
        ssim_value = float(
            structural_similarity(
                gt_frame,
                pred_frame,
                channel_axis=2,
                data_range=255,
            )
        )
        psnr_value = float(peak_signal_noise_ratio(gt_frame, pred_frame, data_range=255))
        per_frame.append(
            {
                "frame_index": frame_index,
                "lpips": lpips_value,
                "ssim": ssim_value,
                "psnr": psnr_value,
            }
        )

        if frame_pairs_dir is not None:
            ensure_dir(frame_pairs_dir)
            pred_out = frame_pairs_dir / f"{frame_index:04d}_pred.png"
            gt_out = frame_pairs_dir / f"{frame_index:04d}_gt.png"
            cv2.imwrite(str(pred_out), cv2.cvtColor(pred_frame, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(gt_out), cv2.cvtColor(gt_frame, cv2.COLOR_RGB2BGR))

    mean_metrics = {
        "lpips": float(np.mean([row["lpips"] for row in per_frame])),
        "ssim": float(np.mean([row["ssim"] for row in per_frame])),
        "psnr": float(np.mean([row["psnr"] for row in per_frame])),
    }
    return {
        "original_pred_frames": original_pred_frames,
        "original_gt_frames": original_gt_frames,
        "evaluated_frames": evaluated_frames,
        "truncated": original_pred_frames != original_gt_frames,
        "resize_applied": resize_applied,
        "original_pred_resolution": original_pred_resolution,
        "pred_resolution_after_resize": list(pred_eval.shape[1:3]),
        "gt_resolution": list(gt_eval.shape[1:3]),
        "per_frame": per_frame,
        "mean": mean_metrics,
    }


def load_gt_view_frames(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, str | None]]:
    explicit_sources = {
        "left": (
            Path(args.gt_left_video) if args.gt_left_video else None,
            Path(args.gt_left_frames_dir) if args.gt_left_frames_dir else None,
        ),
        "right": (
            Path(args.gt_right_video) if args.gt_right_video else None,
            Path(args.gt_right_frames_dir) if args.gt_right_frames_dir else None,
        ),
        "wrist": (
            Path(args.gt_wrist_video) if args.gt_wrist_video else None,
            Path(args.gt_wrist_frames_dir) if args.gt_wrist_frames_dir else None,
        ),
    }
    if any(video_path is not None or frames_dir is not None for video_path, frames_dir in explicit_sources.values()):
        missing = [
            view_name
            for view_name, (video_path, frames_dir) in explicit_sources.items()
            if video_path is None and frames_dir is None
        ]
        if missing:
            raise ValueError(
                "Per-view GT input requires all three views. Missing: " + ", ".join(sorted(missing))
            )
        frames = {
            view_name: load_frames_source(video_path, frames_dir, f"{view_name} gt")
            for view_name, (video_path, frames_dir) in explicit_sources.items()
        }
        sources = {
            view_name: str(video_path) if video_path is not None else str(frames_dir)
            for view_name, (video_path, frames_dir) in explicit_sources.items()
        }
        return frames, sources

    gt_video = Path(args.gt_video) if args.gt_video else None
    gt_frames_dir = Path(args.gt_frames_dir) if args.gt_frames_dir else None
    composite_gt = load_gt_frames(gt_video, gt_frames_dir)
    return split_droid_views(composite_gt), {
        "left": str(gt_video or gt_frames_dir),
        "right": str(gt_video or gt_frames_dir),
        "wrist": str(gt_video or gt_frames_dir),
    }


def evaluate_per_view(
    pred_frames: np.ndarray,
    args: argparse.Namespace,
    lpips_model,
    device: torch.device,
    frame_pairs_root: Path | None,
) -> dict[str, object]:
    pred_views = split_droid_views(pred_frames, wrist_mode=args.pred_wrist_mode)
    gt_views, gt_sources = load_gt_view_frames(args)

    view_results: dict[str, dict[str, object]] = {}
    for view_name in ("wrist", "left", "right"):
        export_dir = frame_pairs_root / view_name if frame_pairs_root is not None else None
        result = evaluate_frame_pair(
            pred_views[view_name],
            gt_views[view_name],
            lpips_model,
            device,
            frame_pairs_dir=export_dir,
        )
        view_results[view_name] = result

    aggregated_frames = min(len(view_results[view_name]["per_frame"]) for view_name in ("wrist", "left", "right"))
    aggregated_per_frame: list[dict[str, float | int]] = []
    for frame_index in range(aggregated_frames):
        frame_rows = [view_results[view_name]["per_frame"][frame_index] for view_name in ("wrist", "left", "right")]
        aggregated_per_frame.append(
            {
                "frame_index": frame_index,
                "lpips": float(np.mean([row["lpips"] for row in frame_rows])),
                "ssim": float(np.mean([row["ssim"] for row in frame_rows])),
                "psnr": float(np.mean([row["psnr"] for row in frame_rows])),
            }
        )

    overall_mean = {
        "lpips": float(np.mean([row["lpips"] for row in aggregated_per_frame])),
        "ssim": float(np.mean([row["ssim"] for row in aggregated_per_frame])),
        "psnr": float(np.mean([row["psnr"] for row in aggregated_per_frame])),
    }
    return {
        "pred_wrist_mode": args.pred_wrist_mode,
        "gt_sources": gt_sources,
        "evaluated_frames": aggregated_frames,
        "truncated": any(bool(view_results[view_name]["truncated"]) for view_name in ("wrist", "left", "right")),
        "resize_applied": any(
            bool(view_results[view_name]["resize_applied"]) for view_name in ("wrist", "left", "right")
        ),
        "views": view_results,
        "aggregated_per_frame": aggregated_per_frame,
        "overall_mean": overall_mean,
    }


def format_composite_text(sample_id: str, pred_path: Path, gt_label: str, result: dict[str, object]) -> list[str]:
    mean_metrics = result["mean"]  # type: ignore[assignment]
    text_lines = [
        f"sample_id: {sample_id}",
        f"pred_video: {pred_path}",
        f"gt_video: {gt_label}",
        f"original_pred_frames: {result['original_pred_frames']}",
        f"original_gt_frames: {result['original_gt_frames']}",
        f"evaluated_frames: {result['evaluated_frames']}",
        f"resize_applied: {result['resize_applied']}",
        f"mean_lpips: {mean_metrics['lpips']:.6f}",
        f"mean_ssim: {mean_metrics['ssim']:.6f}",
        f"mean_psnr: {mean_metrics['psnr']:.6f}",
        "",
        "per_frame:",
    ]
    text_lines.extend(
        f"frame={row['frame_index']} lpips={row['lpips']:.6f} ssim={row['ssim']:.6f} psnr={row['psnr']:.6f}"
        for row in result["per_frame"]  # type: ignore[index]
    )
    return text_lines


def format_per_view_text(sample_id: str, pred_path: Path, result: dict[str, object]) -> list[str]:
    text_lines = [
        f"sample_id: {sample_id}",
        f"pred_video: {pred_path}",
        f"pred_wrist_mode: {result['pred_wrist_mode']}",
        f"evaluated_frames: {result['evaluated_frames']}",
        f"truncated: {result['truncated']}",
        f"resize_applied: {result['resize_applied']}",
        f"overall_mean_lpips: {result['overall_mean']['lpips']:.6f}",
        f"overall_mean_ssim: {result['overall_mean']['ssim']:.6f}",
        f"overall_mean_psnr: {result['overall_mean']['psnr']:.6f}",
        "",
    ]
    for view_name in ("wrist", "left", "right"):
        view_result = result["views"][view_name]
        mean_metrics = view_result["mean"]
        text_lines.extend(
            [
                f"[{view_name}] gt_source: {result['gt_sources'][view_name]}",
                f"[{view_name}] original_pred_frames: {view_result['original_pred_frames']}",
                f"[{view_name}] original_gt_frames: {view_result['original_gt_frames']}",
                f"[{view_name}] evaluated_frames: {view_result['evaluated_frames']}",
                f"[{view_name}] resize_applied: {view_result['resize_applied']}",
                f"[{view_name}] mean_lpips: {mean_metrics['lpips']:.6f}",
                f"[{view_name}] mean_ssim: {mean_metrics['ssim']:.6f}",
                f"[{view_name}] mean_psnr: {mean_metrics['psnr']:.6f}",
                "",
            ]
        )
    text_lines.append("aggregated_per_frame:")
    text_lines.extend(
        (
            f"frame={row['frame_index']} lpips={row['lpips']:.6f} "
            f"ssim={row['ssim']:.6f} psnr={row['psnr']:.6f}"
        )
        for row in result["aggregated_per_frame"]
    )
    return text_lines


def evaluate_video_metrics(
    pred_video: Path,
    output_dir: Path,
    sample_id: str | None = None,
    eval_mode: str = "composite",
    lpips_net: str = "squeeze",
    pred_wrist_mode: str = "resize",
    gt_video: Path | None = None,
    gt_frames_dir: Path | None = None,
    gt_left_video: Path | None = None,
    gt_right_video: Path | None = None,
    gt_wrist_video: Path | None = None,
    gt_left_frames_dir: Path | None = None,
    gt_right_frames_dir: Path | None = None,
    gt_wrist_frames_dir: Path | None = None,
    export_frame_pairs: bool = False,
) -> dict[str, object]:
    output_dir = ensure_dir(output_dir)
    pred_frames = read_video_frames(pred_video)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_model = get_lpips_model(device, lpips_net)
    resolved_sample_id = sample_id or output_dir.name

    args = argparse.Namespace(
        pred_video=str(pred_video),
        gt_video=str(gt_video) if gt_video is not None else None,
        gt_frames_dir=str(gt_frames_dir) if gt_frames_dir is not None else None,
        output_dir=str(output_dir),
        sample_id=resolved_sample_id,
        eval_mode=eval_mode,
        lpips_net=lpips_net,
        pred_wrist_mode=pred_wrist_mode,
        gt_left_video=str(gt_left_video) if gt_left_video is not None else None,
        gt_right_video=str(gt_right_video) if gt_right_video is not None else None,
        gt_wrist_video=str(gt_wrist_video) if gt_wrist_video is not None else None,
        gt_left_frames_dir=str(gt_left_frames_dir) if gt_left_frames_dir is not None else None,
        gt_right_frames_dir=str(gt_right_frames_dir) if gt_right_frames_dir is not None else None,
        gt_wrist_frames_dir=str(gt_wrist_frames_dir) if gt_wrist_frames_dir is not None else None,
        export_frame_pairs=export_frame_pairs,
    )

    results: dict[str, object] = {
        "sample_id": resolved_sample_id,
        "pred_video": str(pred_video),
        "lpips_net": lpips_net,
        "eval_mode": eval_mode,
    }
    text_lines: list[str] = []

    if eval_mode in {"composite", "both"}:
        gt_frames = load_gt_frames(gt_video, gt_frames_dir)
        composite_pairs_dir = ensure_dir(output_dir / "frame_pairs" / "composite") if export_frame_pairs else None
        composite_result = evaluate_frame_pair(
            pred_frames,
            gt_frames,
            lpips_model,
            device,
            frame_pairs_dir=composite_pairs_dir,
        )
        if eval_mode == "composite":
            results.update(
                {
                    "gt_video": str(gt_video) if gt_video is not None else None,
                    "gt_frames_dir": str(gt_frames_dir) if gt_frames_dir is not None else None,
                    **composite_result,
                }
            )
            text_lines = format_composite_text(
                resolved_sample_id,
                pred_video,
                str(gt_video) if gt_video is not None else str(gt_frames_dir),
                composite_result,
            )
        else:
            results["composite"] = {
                "gt_video": str(gt_video) if gt_video is not None else None,
                "gt_frames_dir": str(gt_frames_dir) if gt_frames_dir is not None else None,
                **composite_result,
            }

    if eval_mode in {"per_view", "both"}:
        per_view_pairs_dir = ensure_dir(output_dir / "frame_pairs" / "per_view") if export_frame_pairs else None
        per_view_result = evaluate_per_view(
            pred_frames,
            args,
            lpips_model,
            device,
            frame_pairs_root=per_view_pairs_dir,
        )
        if eval_mode == "per_view":
            results.update(per_view_result)
            text_lines = format_per_view_text(resolved_sample_id, pred_video, per_view_result)
        else:
            results["per_view"] = per_view_result

    if eval_mode == "both":
        composite_text = format_composite_text(
            resolved_sample_id,
            pred_video,
            str(gt_video) if gt_video is not None else str(gt_frames_dir),
            results["composite"],
        )
        per_view_text = format_per_view_text(resolved_sample_id, pred_video, results["per_view"])
        text_lines = composite_text + ["", "== per_view ==", ""] + per_view_text

    (output_dir / "metrics_results.txt").write_text("\n".join(text_lines), encoding="utf-8")
    write_json(output_dir / "metrics_results.json", results)
    LOGGER.info("Wrote metrics to %s", output_dir)
    return results


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)

    evaluate_video_metrics(
        pred_video=Path(args.pred_video),
        output_dir=Path(args.output_dir),
        sample_id=args.sample_id,
        eval_mode=args.eval_mode,
        lpips_net=args.lpips_net,
        pred_wrist_mode=args.pred_wrist_mode,
        gt_video=Path(args.gt_video) if args.gt_video else None,
        gt_frames_dir=Path(args.gt_frames_dir) if args.gt_frames_dir else None,
        gt_left_video=Path(args.gt_left_video) if args.gt_left_video else None,
        gt_right_video=Path(args.gt_right_video) if args.gt_right_video else None,
        gt_wrist_video=Path(args.gt_wrist_video) if args.gt_wrist_video else None,
        gt_left_frames_dir=Path(args.gt_left_frames_dir) if args.gt_left_frames_dir else None,
        gt_right_frames_dir=Path(args.gt_right_frames_dir) if args.gt_right_frames_dir else None,
        gt_wrist_frames_dir=Path(args.gt_wrist_frames_dir) if args.gt_wrist_frames_dir else None,
        export_frame_pairs=args.export_frame_pairs,
    )


if __name__ == "__main__":
    main()
