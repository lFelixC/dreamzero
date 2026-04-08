#!/usr/bin/env python3
"""Shared helpers for DreamZero video evaluation scripts."""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any

import cv2
import imageio
import numpy as np


REPO_ROOT = Path("/data/dreamzero")
VIDEO_EVAL_ROOT = REPO_ROOT / "video_eval"
VIDEO_RESULTS_ROOT = VIDEO_EVAL_ROOT / "video_results"
DEFAULT_MODEL_PATH = Path(
    "/data/checkpoints/dreamzero/dreamzero_droid_wan22_full_finetune/checkpoint-5000"
)
DEFAULT_DEBUG_IMAGE_DIR = REPO_ROOT / "debug_image"
DEFAULT_PORT = 5001
DEFAULT_DATASET_ROOT_CANDIDATES = (
    Path("/data/datasets/robotset"),
    Path("/data/datasets/roboset"),
)

CAMERA_FILES = {
    "observation/exterior_image_0_left": "exterior_image_1_left.mp4",
    "observation/exterior_image_1_left": "exterior_image_2_left.mp4",
    "observation/wrist_image_left": "wrist_image_left.mp4",
}
MANIFEST_VIDEO_KEYS = {
    "observation.images.exterior_image_1_left": "observation/exterior_image_0_left",
    "observation.images.exterior_image_2_left": "observation/exterior_image_1_left",
    "observation.images.wrist_image_left": "observation/wrist_image_left",
}
RELATIVE_OFFSETS = (-23, -16, -8, 0)
ACTION_HORIZON = 24
DEFAULT_DEBUG_PROMPT = (
    "Move the pan forward and use the brush in the middle of the plates to brush the inside "
    "of the pan"
)
DEFAULT_DATASET_PROMPTS = {
    "clean_kitchen_slide_close_drawer_scene_3": [
        "Close the kitchen drawer.",
        "Slide the drawer shut.",
        "Push the drawer closed.",
    ],
    "clean_kitchen_pick_towel_scene_3": [
        "Pick up the towel.",
        "Grab the towel from the kitchen scene.",
        "Lift the towel off the surface.",
    ],
}


def utc_timestamp() -> str:
    """Return a UTC timestamp suitable for run directories."""
    return _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def append_run_log(log_path: Path, message: str) -> None:
    ensure_dir(log_path.parent)
    timestamp = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{timestamp} {message}\n")


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def find_existing_dataset_root() -> Path:
    for candidate in DEFAULT_DATASET_ROOT_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find /data/datasets/robotset or /data/datasets/roboset."
    )


def build_chunk_schedule(total_frames: int, num_chunks: int | None = None) -> list[list[int]]:
    """Build the frame schedule used by the official DreamZero test client."""
    chunks: list[list[int]] = []
    current_frame = 23
    while True:
        if num_chunks is not None and len(chunks) >= num_chunks:
            break
        indices = [max(current_frame + off, 0) for off in RELATIVE_OFFSETS]
        if indices[-1] >= total_frames:
            break
        chunks.append(indices)
        current_frame += ACTION_HORIZON
    return chunks


def read_video_frames(video_path: Path) -> np.ndarray:
    """Read an MP4 into an RGB uint8 array shaped (T, H, W, 3)."""
    cap = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames loaded from {video_path}")
    return np.stack(frames, axis=0)


def resize_frames(
    frames: np.ndarray,
    target_height: int,
    target_width: int,
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    """Resize (H,W,C) or (T,H,W,C) frames to a target size."""
    if frames.ndim == 3:
        if frames.shape[:2] == (target_height, target_width):
            return frames
        return cv2.resize(frames, (target_width, target_height), interpolation=interpolation)
    if frames.shape[1:3] == (target_height, target_width):
        return frames
    return np.stack(
        [
            cv2.resize(frame, (target_width, target_height), interpolation=interpolation)
            for frame in frames
        ],
        axis=0,
    )


def compose_droid_views(
    left_frames: np.ndarray,
    right_frames: np.ndarray,
    wrist_frames: np.ndarray,
) -> np.ndarray:
    """Compose DROID's three camera views into DreamZero's composite layout."""
    if left_frames.ndim != 4 or right_frames.ndim != 4 or wrist_frames.ndim != 4:
        raise ValueError("Expected all camera arrays to be shaped (T, H, W, 3).")
    total_frames = min(left_frames.shape[0], right_frames.shape[0], wrist_frames.shape[0])
    left = left_frames[:total_frames]
    right = right_frames[:total_frames]
    wrist = wrist_frames[:total_frames]
    height, width = left.shape[1:3]
    composite = np.zeros((total_frames, height * 2, width * 2, 3), dtype=np.uint8)
    composite[:, :height, :, :] = np.repeat(wrist, 2, axis=2)
    composite[:, height:, :width, :] = left
    composite[:, height:, width:, :] = right
    return composite


def split_droid_views(
    composite_frames: np.ndarray,
    wrist_mode: str = "resize",
) -> dict[str, np.ndarray]:
    """Split DreamZero's composite layout into wrist/left/right view tensors."""
    if composite_frames.ndim != 4:
        raise ValueError(f"Expected frames shaped (T, H, W, C), got {composite_frames.shape}")
    total_height, total_width = composite_frames.shape[1:3]
    if total_height % 2 != 0 or total_width % 2 != 0:
        raise ValueError(
            "Composite frames must have even height and width so they can be split into three views."
        )

    view_height = total_height // 2
    view_width = total_width // 2
    wrist_strip = composite_frames[:, :view_height, :, :]
    if wrist_mode == "resize":
        wrist = resize_frames(wrist_strip, view_height, view_width, interpolation=cv2.INTER_LINEAR)
    elif wrist_mode == "left_half":
        wrist = wrist_strip[:, :, :view_width, :]
    else:
        raise ValueError(f"Unsupported wrist_mode: {wrist_mode}")

    return {
        "wrist": wrist,
        "left": composite_frames[:, view_height:, :view_width, :],
        "right": composite_frames[:, view_height:, view_width:, :],
    }


def save_video(
    frames: np.ndarray,
    output_path: Path,
    fps: int = 5,
) -> None:
    """Write frames to MP4 using imageio."""
    if frames.ndim != 4:
        raise ValueError(f"Expected frames shaped (T, H, W, C), got {frames.shape}")
    ensure_dir(output_path.parent)
    imageio.mimsave(
        str(output_path),
        [np.asarray(frame) for frame in frames],
        fps=fps,
        codec="libx264",
        macro_block_size=1,
    )


def describe_tree(root: Path, max_depth: int = 2) -> list[str]:
    """Return a compact directory tree preview."""
    lines = [str(root)]
    if not root.exists():
        return lines

    root_depth = len(root.parts)
    for path in sorted(root.rglob("*")):
        depth = len(path.parts) - root_depth
        if depth > max_depth:
            continue
        prefix = "  " * depth
        marker = "[D]" if path.is_dir() else "[F]"
        lines.append(f"{prefix}{marker} {path.name}")
    return lines


def strip_archive_suffix(path: Path) -> str:
    """Return a filename without archive suffixes like .tar.gz."""
    name = path.name
    for suffix in (".tar.gz", ".tgz", ".tar", ".zip"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def load_manifest_entry(manifest_path: Path, sample_id: str | None) -> dict[str, Any]:
    rows = load_jsonl(manifest_path)
    if not rows:
        raise RuntimeError(f"No rows found in manifest: {manifest_path}")
    if sample_id is None:
        return rows[0]
    for row in rows:
        if row.get("sample_id") == sample_id:
            return row
    raise KeyError(f"Sample ID '{sample_id}' not found in {manifest_path}")
