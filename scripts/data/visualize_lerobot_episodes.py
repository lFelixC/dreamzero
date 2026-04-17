#!/usr/bin/env python3
"""
Visualize DROID LeRobot episodes for manual filtering.

The script scans parquet files from disk instead of trusting meta/info.json, then
writes one visualization video per selected episode. Each output video contains
the selected state/action joint and gripper traces on top, and the LeRobot camera
videos stitched horizontally below.

Examples:
  # Preview dataset summary only. This does not create videos.
  python scripts/data/visualize_lerobot_episodes.py \\
      --dataset-root /data/dataset/dreamzero/droid_lerobot

  # Generate a small sample.
  python scripts/data/visualize_lerobot_episodes.py \\
      --dataset-root /data/dataset/dreamzero/droid_lerobot \\
      --episodes 0 1 2

  # Generate all episodes in one chunk with multiple workers.
  python scripts/data/visualize_lerobot_episodes.py \\
      --dataset-root /data/dataset/dreamzero/droid_lerobot \\
      --chunks chunk-000 --workers 32

  # Randomly sample 1000 episodes found on disk.
  python scripts/data/visualize_lerobot_episodes.py \\
      --dataset-root /data/dataset/dreamzero/droid_lerobot \\
      --random 1000 --workers 32
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import re
import sys
import textwrap
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_DATASET_ROOT = Path("/data/dataset/dreamzero/droid_lerobot")
DEFAULT_CAMERA_NAMES = [
    "exterior_image_1_left",
    "exterior_image_2_left",
    "wrist_image_left",
]
STATE_PLOT_KEYS = ["joint_position", "gripper_position"]
ACTION_PLOT_KEYS = ["joint_position", "gripper_position"]
THREAD_ENV_VARS = [
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
]

for env_var in THREAD_ENV_VARS:
    os.environ.setdefault(env_var, "1")


@dataclass(frozen=True)
class EpisodeFile:
    episode_index: int
    chunk_name: str
    parquet_path: Path


@dataclass(frozen=True)
class VideoSpec:
    name: str
    original_key: str


@dataclass(frozen=True)
class PlotSeries:
    label: str
    values: Any


@dataclass(frozen=True)
class WorkItem:
    dataset_root: Path
    output_dir: Path
    episode_file: EpisodeFile
    episode_meta: dict[str, Any]
    info: dict[str, Any]
    modality: dict[str, Any]
    plot_mode: str
    max_frames: int | None
    overwrite: bool


@dataclass(frozen=True)
class WorkResult:
    episode_index: int
    output_path: Path
    status: str
    message: str
    warnings: tuple[str, ...]


def load_json(path: Path, required: bool = True) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required metadata file: {path}")
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_jsonl_line(line: str, path: Path, line_number: int) -> dict[str, Any] | None:
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError as exc:
        print(f"warning: could not parse {path}:{line_number}: {exc}", file=sys.stderr)
        return None


def load_selected_episode_metadata(
    path: Path,
    episode_indices: set[int],
) -> dict[int, dict[str, Any]]:
    if not path.exists() or not episode_indices:
        return {}

    found: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            item = parse_jsonl_line(line, path, line_number)
            if not item:
                continue
            episode_index = item.get("episode_index")
            if episode_index in episode_indices:
                found[int(episode_index)] = item
                if len(found) == len(episode_indices):
                    break
    return found


def load_tasks_by_index(path: Path, task_indices: set[int]) -> dict[int, str]:
    if not path.exists() or not task_indices:
        return {}

    found: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            item = parse_jsonl_line(line, path, line_number)
            if not item:
                continue
            task_index = item.get("task_index")
            if task_index in task_indices:
                found[int(task_index)] = str(item.get("task", ""))
                if len(found) == len(task_indices):
                    break
    return found


def count_jsonl_lines(path: Path) -> int | None:
    if not path.exists():
        return None
    count = 0
    with path.open("rb") as f:
        for _ in f:
            count += 1
    return count


def episode_index_from_path(path: Path) -> int | None:
    match = re.search(r"episode_(\d+)\.parquet$", path.name)
    if not match:
        return None
    return int(match.group(1))


def scan_episode_files(dataset_root: Path) -> list[EpisodeFile]:
    files: list[EpisodeFile] = []
    for parquet_path in sorted((dataset_root / "data").glob("chunk-*/episode_*.parquet")):
        episode_index = episode_index_from_path(parquet_path)
        if episode_index is None:
            continue
        files.append(
            EpisodeFile(
                episode_index=episode_index,
                chunk_name=parquet_path.parent.name,
                parquet_path=parquet_path,
            )
        )
    return sorted(files, key=lambda item: (item.chunk_name, item.episode_index))


def normalize_chunk_name(raw: str) -> str:
    raw = raw.strip()
    if raw.isdigit():
        return f"chunk-{int(raw):03d}"
    return raw


def selection_requested(args: argparse.Namespace) -> bool:
    return bool(
        args.all
        or args.episodes
        or args.episode_range
        or args.chunks
        or args.random is not None
    )


def resolve_episode_selection(
    args: argparse.Namespace,
    episode_files: list[EpisodeFile],
) -> tuple[list[EpisodeFile], list[str]]:
    warnings: list[str] = []
    by_episode = {item.episode_index: item for item in episode_files}
    selected_indices: set[int] = set()

    if args.all:
        selected_indices.update(by_episode)

    if args.random is not None:
        population = list(by_episode)
        sample_count = min(args.random, len(population))
        if args.random > len(population):
            warnings.append(
                f"--random requested {args.random} episodes, but only "
                f"{len(population)} were found on disk; selecting all of them"
            )
        rng = random.Random(args.random_seed)
        selected_indices.update(rng.sample(population, sample_count))

    if args.episodes:
        for episode_index in args.episodes:
            if episode_index not in by_episode:
                warnings.append(f"requested episode {episode_index} was not found on disk")
                continue
            selected_indices.add(episode_index)

    if args.episode_range:
        start, end = args.episode_range
        if end < start:
            warnings.append(f"episode range {start}..{end} is empty because END < START")
        else:
            missing_in_range: list[int] = []
            for episode_index in range(start, end + 1):
                if episode_index in by_episode:
                    selected_indices.add(episode_index)
                else:
                    missing_in_range.append(episode_index)
            if 0 < len(missing_in_range) <= 10:
                for episode_index in missing_in_range:
                    warnings.append(f"requested episode {episode_index} was not found on disk")
            elif missing_in_range:
                preview = ", ".join(str(index) for index in missing_in_range[:10])
                warnings.append(
                    f"episode range {start}..{end} contains {len(missing_in_range)} "
                    f"episodes not found on disk; first missing: {preview}"
                )

    if args.chunks:
        chunk_names = {normalize_chunk_name(raw) for raw in args.chunks}
        found_chunks = {item.chunk_name for item in episode_files}
        for chunk_name in sorted(chunk_names - found_chunks):
            warnings.append(f"requested chunk {chunk_name} was not found on disk")
        selected_indices.update(
            item.episode_index for item in episode_files if item.chunk_name in chunk_names
        )

    selected = [by_episode[index] for index in sorted(selected_indices)]
    if args.max_episodes is not None:
        selected = selected[: args.max_episodes]
    return selected, warnings


def get_video_specs(modality: dict[str, Any]) -> list[VideoSpec]:
    video_modality = modality.get("video") or {}
    specs: list[VideoSpec] = []
    for name, item in video_modality.items():
        item = item or {}
        original_key = item.get("original_key") or f"observation.images.{name}"
        specs.append(VideoSpec(name=name, original_key=original_key))

    if specs:
        return specs

    return [
        VideoSpec(name=name, original_key=f"observation.images.{name}")
        for name in DEFAULT_CAMERA_NAMES
    ]


def chunk_number_from_name(chunk_name: str, episode_index: int, chunks_size: int) -> int:
    match = re.fullmatch(r"chunk-(\d+)", chunk_name)
    if match:
        return int(match.group(1))
    return episode_index // chunks_size


def format_video_path(
    dataset_root: Path,
    info: dict[str, Any],
    episode_file: EpisodeFile,
    video_spec: VideoSpec,
) -> Path:
    chunks_size = int(info.get("chunks_size", 1000) or 1000)
    episode_chunk = chunk_number_from_name(
        episode_file.chunk_name, episode_file.episode_index, chunks_size
    )
    pattern = info.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    )
    try:
        relative = pattern.format(
            episode_chunk=episode_chunk,
            episode_index=episode_file.episode_index,
            video_key=video_spec.original_key,
        )
    except (KeyError, IndexError, ValueError) as exc:
        print(
            f"warning: could not format video_path pattern {pattern!r}: {exc}; using default",
            file=sys.stderr,
        )
        relative = (
            f"videos/{episode_file.chunk_name}/{video_spec.original_key}/"
            f"episode_{episode_file.episode_index:06d}.mp4"
        )
    return dataset_root / relative


def resolve_output_dir(dataset_root: Path, raw_output_dir: str) -> Path:
    output_dir = Path(raw_output_dir)
    if output_dir.is_absolute():
        return output_dir
    return dataset_root / output_dir


def require_runtime_dependencies() -> None:
    missing: list[str] = []
    for module_name, package_name in [
        ("numpy", "numpy"),
        ("cv2", "opencv-python"),
        ("pyarrow.parquet", "pyarrow"),
        ("tqdm", "tqdm"),
    ]:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            top_level = module_name.split(".", maxsplit=1)[0]
            if exc.name == top_level:
                missing.append(package_name)
            else:
                raise

    if missing:
        packages = ", ".join(sorted(set(missing)))
        raise RuntimeError(
            "Missing runtime dependencies: "
            f"{packages}. Activate the DreamZero environment or install them first, "
            "for example: pip install numpy opencv-python pyarrow tqdm"
        )


def import_runtime_modules() -> tuple[Any, Any, Any]:
    import cv2
    import numpy as np
    import pyarrow.parquet as pq

    return cv2, np, pq


def table_column_to_2d_array(table: Any, column_name: str, np: Any) -> Any:
    values = table[column_name].combine_chunks().to_pylist()
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    return array


def table_column_to_vector(table: Any, column_name: str, np: Any) -> Any:
    values = table[column_name].combine_chunks().to_pylist()
    return np.asarray(values)


def plot_keys_for_mode(plot_mode: str) -> list[tuple[str, list[str]]]:
    keys: list[tuple[str, list[str]]] = []
    if plot_mode in {"state", "state-action"}:
        keys.append(("state", STATE_PLOT_KEYS))
    if plot_mode in {"action", "state-action"}:
        keys.append(("action", ACTION_PLOT_KEYS))
    return keys


def build_required_columns(modality: dict[str, Any], plot_mode: str) -> set[str]:
    required = {"timestamp", "task_index"}
    for section, names in plot_keys_for_mode(plot_mode):
        default_key = "observation.state" if section == "state" else "action"
        section_modality = modality.get(section) or {}
        for name in names:
            item = section_modality.get(name) or {}
            required.add(item.get("original_key") or default_key)

    annotation = modality.get("annotation") or {}
    for name, item in annotation.items():
        item = item or {}
        required.add(item.get("original_key") or f"annotation.{name}")
    return required


def make_series_label(section: str, name: str, dim_index: int, dim_count: int) -> str:
    prefix = "S" if section == "state" else "A"
    if "joint" in name:
        return f"{prefix}.j{dim_index}"
    if "gripper" in name and dim_count == 1:
        return f"{prefix}.g"
    if "gripper" in name:
        return f"{prefix}.g{dim_index}"
    return f"{prefix}.{name[:4]}{dim_index}"


def build_plot_series(
    table: Any,
    modality: dict[str, Any],
    plot_mode: str,
    np: Any,
) -> tuple[list[PlotSeries], list[str]]:
    warnings: list[str] = []
    series: list[PlotSeries] = []

    for section, names in plot_keys_for_mode(plot_mode):
        default_key = "observation.state" if section == "state" else "action"
        section_modality = modality.get(section) or {}
        for name in names:
            item = section_modality.get(name)
            if not item:
                warnings.append(f"modality.{section}.{name} is missing; skipping plot series")
                continue

            original_key = item.get("original_key") or default_key
            if original_key not in table.column_names:
                warnings.append(
                    f"parquet column {original_key!r} is missing; skipping {section}.{name}"
                )
                continue

            try:
                start = int(item["start"])
                end = int(item["end"])
            except KeyError:
                warnings.append(f"modality.{section}.{name} is missing start/end; skipping")
                continue

            values = table_column_to_2d_array(table, original_key, np)
            if start < 0 or end > values.shape[1] or end <= start:
                warnings.append(
                    f"invalid slice for modality.{section}.{name}: {start}:{end} "
                    f"against width {values.shape[1]}"
                )
                continue

            sliced = values[:, start:end]
            for dim_index in range(sliced.shape[1]):
                label = make_series_label(section, name, dim_index, sliced.shape[1])
                series.append(PlotSeries(label=label, values=sliced[:, dim_index]))

    return series, warnings


def collect_fallback_task_indices(table: Any) -> set[int]:
    task_indices: set[int] = set()
    annotation_cols = [
        name for name in table.column_names if name.startswith("annotation.language.")
    ]
    for column_name in [*annotation_cols, "task_index"]:
        if column_name not in table.column_names:
            continue
        values = table[column_name].combine_chunks().to_pylist()
        if not values:
            continue
        value = values[0]
        if value is None:
            continue
        try:
            task_indices.add(int(value))
        except (TypeError, ValueError):
            continue
    return task_indices


def episode_tasks(
    dataset_root: Path,
    episode_meta: dict[str, Any],
    table: Any,
) -> list[str]:
    raw_tasks = episode_meta.get("tasks")
    if isinstance(raw_tasks, list):
        tasks = [str(task) for task in raw_tasks if str(task).strip()]
        if tasks:
            return tasks
    elif isinstance(raw_tasks, str) and raw_tasks.strip():
        return [raw_tasks]

    task_indices = collect_fallback_task_indices(table)
    tasks_by_index = load_tasks_by_index(dataset_root / "meta" / "tasks.jsonl", task_indices)
    fallback = [
        tasks_by_index[index]
        for index in sorted(task_indices)
        if tasks_by_index.get(index) and tasks_by_index[index] != "not provided"
    ]
    return fallback or ["not provided"]


def episode_success(episode_meta: dict[str, Any]) -> str:
    if "success" not in episode_meta:
        return "not provided"
    value = episode_meta["success"]
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).lower()


def cv2_text_width(text: str, cv2: Any, font: int, scale: float, thickness: int) -> int:
    return cv2.getTextSize(text, font, scale, thickness)[0][0]


def wrap_text(
    text: str,
    max_width: int,
    cv2: Any,
    font: int,
    scale: float,
    thickness: int,
) -> list[str]:
    words = text.split()
    if not words:
        return [""]

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if cv2_text_width(candidate, cv2, font, scale, thickness) <= max_width:
            current = candidate
            continue
        lines.append(current)
        current = word
    lines.append(current)
    return lines


def build_static_text_lines(
    tasks: list[str],
    max_width: int,
    cv2: Any,
) -> list[str]:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 1
    lines: list[str] = []
    for index, task in enumerate(tasks, start=1):
        text = f"instruction {index}: {task}"
        lines.extend(wrap_text(text, max_width, cv2, font, scale, thickness))
    return lines


def normalize_series(values: Any, np: Any) -> Any:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return array
    finite_mask = np.isfinite(array)
    if not finite_mask.any():
        return np.full(array.shape, 0.5, dtype=np.float64)
    finite_values = array[finite_mask]
    min_value = float(finite_values.min())
    max_value = float(finite_values.max())
    if abs(max_value - min_value) < 1e-12:
        return np.full(array.shape, 0.5, dtype=np.float64)
    normalized = (array - min_value) / (max_value - min_value)
    normalized[~finite_mask] = 0.5
    return normalized


def draw_legend(
    image: Any,
    series: list[PlotSeries],
    colors: list[tuple[int, int, int]],
    origin: tuple[int, int],
    width: int,
    cv2: Any,
) -> None:
    x, y = origin
    row_height = 18
    item_width = 54
    items_per_row = max(1, width // item_width)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for index, item in enumerate(series):
        row = index // items_per_row
        col = index % items_per_row
        item_x = x + col * item_width
        item_y = y + row * row_height
        color = colors[index % len(colors)]
        cv2.line(image, (item_x, item_y - 4), (item_x + 14, item_y - 4), color, 2, cv2.LINE_AA)
        cv2.putText(
            image,
            item.label,
            (item_x + 18, item_y),
            font,
            0.4,
            (25, 25, 25),
            1,
            cv2.LINE_AA,
        )


def draw_plot(
    image: Any,
    series: list[PlotSeries],
    current_frame: int,
    plot_rect: tuple[int, int, int, int],
    cv2: Any,
    np: Any,
) -> None:
    left, top, right, bottom = plot_rect
    plot_width = max(1, right - left)
    plot_height = max(1, bottom - top)
    colors = [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
        (227, 119, 194),
        (127, 127, 127),
        (188, 189, 34),
        (23, 190, 207),
        (66, 146, 198),
        (251, 106, 74),
        (49, 163, 84),
        (117, 107, 177),
        (253, 174, 97),
        (0, 109, 44),
    ]
    bgr_colors = [(b, g, r) for r, g, b in colors]

    cv2.rectangle(image, (left, top), (right, bottom), (245, 245, 245), -1)
    cv2.rectangle(image, (left, top), (right, bottom), (80, 80, 80), 1)

    for frac in (0.25, 0.5, 0.75):
        y = int(bottom - plot_height * frac)
        cv2.line(image, (left, y), (right, y), (215, 215, 215), 1, cv2.LINE_AA)

    for index, item in enumerate(series):
        normalized = normalize_series(item.values, np)
        if normalized.size == 0:
            continue
        if normalized.size == 1:
            xs = np.array([left], dtype=np.float64)
        else:
            xs = left + np.linspace(0, plot_width, normalized.size)
        ys = bottom - normalized * plot_height
        points = np.column_stack((xs, ys)).astype(np.int32)
        color = bgr_colors[index % len(bgr_colors)]
        cv2.polylines(image, [points], False, color, 1, cv2.LINE_AA)

        cursor_index = min(max(current_frame, 0), normalized.size - 1)
        cv2.circle(
            image,
            (int(xs[cursor_index]), int(ys[cursor_index])),
            3,
            color,
            -1,
            cv2.LINE_AA,
        )

    if series:
        total_frames = len(series[0].values)
        if total_frames <= 1:
            cursor_x = left
        else:
            cursor_x = int(
                left
                + plot_width
                * min(max(current_frame, 0), total_frames - 1)
                / (total_frames - 1)
            )
        cv2.line(image, (cursor_x, top), (cursor_x, bottom), (20, 20, 20), 1, cv2.LINE_AA)

    cv2.putText(
        image,
        "normalized per dimension",
        (left, top - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (50, 50, 50),
        1,
        cv2.LINE_AA,
    )
    draw_legend(image, series, bgr_colors, (left + 190, top - 8), max(1, right - left - 190), cv2)


def render_top_panel(
    width: int,
    header_height: int,
    plot_height: int,
    static_lines: list[str],
    series: list[PlotSeries],
    episode_index: int,
    success: str,
    current_frame: int,
    total_frames: int,
    timestamp: float,
    plot_mode: str,
    cv2: Any,
    np: Any,
) -> Any:
    panel_height = header_height + plot_height + 34
    image = np.full((panel_height, width, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    status = (
        f"episode {episode_index:06d} | success: {success} | "
        f"frame: {current_frame}/{max(0, total_frames - 1)} ({current_frame + 1}/{total_frames}) | "
        f"time: {timestamp:.2f}s | plot: {plot_mode}"
    )
    cv2.putText(image, status, (12, 26), font, 0.58, (10, 10, 10), 1, cv2.LINE_AA)

    y = 52
    for line in static_lines:
        cv2.putText(image, line, (12, y), font, 0.48, (30, 30, 30), 1, cv2.LINE_AA)
        y += 21

    plot_rect = (58, header_height + 24, width - 18, header_height + plot_height + 8)
    cv2.putText(
        image,
        "1.0",
        (18, plot_rect[1] + 6),
        font,
        0.38,
        (80, 80, 80),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "0.0",
        (18, plot_rect[3]),
        font,
        0.38,
        (80, 80, 80),
        1,
        cv2.LINE_AA,
    )
    draw_plot(image, series, current_frame, plot_rect, cv2, np)
    return image


def make_placeholder_frame(
    width: int,
    height: int,
    title: str,
    reason: str,
    cv2: Any,
    np: Any,
) -> Any:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, title[:40], (12, 28), font, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    y = 58
    for line in wrap_text(reason, max(20, width - 24), cv2, font, 0.48, 1):
        cv2.putText(frame, line, (12, y), font, 0.48, (230, 230, 230), 1, cv2.LINE_AA)
        y += 22
    return frame


class SequentialVideoReader:
    def __init__(
        self,
        video_spec: VideoSpec,
        path: Path,
        target_size: tuple[int, int],
        cv2: Any,
        np: Any,
    ) -> None:
        self.video_spec = video_spec
        self.path = path
        self.target_width, self.target_height = target_size
        self.cv2 = cv2
        self.np = np
        self.error: str | None = None
        self.cap = None

        if not path.exists():
            self.error = f"missing video: {path}"
            return

        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            self.error = f"open failed: {path}"
            self.cap.release()
            self.cap = None

    def read(self, frame_index: int) -> Any:
        if self.error:
            return make_placeholder_frame(
                self.target_width,
                self.target_height,
                self.video_spec.name,
                self.error,
                self.cv2,
                self.np,
            )

        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.error = f"read failed at frame {frame_index}: {self.path}"
            return make_placeholder_frame(
                self.target_width,
                self.target_height,
                self.video_spec.name,
                self.error,
                self.cv2,
                self.np,
            )

        if frame.shape[1] != self.target_width or frame.shape[0] != self.target_height:
            frame = self.cv2.resize(
                frame,
                (self.target_width, self.target_height),
                interpolation=self.cv2.INTER_AREA,
            )
        else:
            frame = frame.copy()

        label_width = min(self.target_width, 12 + len(self.video_spec.name) * 9)
        self.cv2.rectangle(frame, (0, 0), (label_width, 24), (0, 0, 0), -1)
        self.cv2.putText(
            frame,
            self.video_spec.name,
            (8, 17),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (245, 245, 245),
            1,
            self.cv2.LINE_AA,
        )
        return frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()


def target_video_size(
    info: dict[str, Any],
    video_specs: list[VideoSpec],
    video_paths: list[Path],
    cv2: Any,
) -> tuple[int, int]:
    features = info.get("features") or {}
    for spec in video_specs:
        shape = (features.get(spec.original_key) or {}).get("shape")
        if isinstance(shape, list) and len(shape) >= 2:
            height, width = int(shape[0]), int(shape[1])
            if width > 0 and height > 0:
                return width, height

    for path in video_paths:
        if not path.exists():
            continue
        cap = cv2.VideoCapture(str(path))
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if width > 0 and height > 0:
                return width, height
        cap.release()
    return 320, 180


def resolve_fps(info: dict[str, Any], video_paths: list[Path], cv2: Any) -> float:
    fps = info.get("fps")
    if isinstance(fps, (int, float)) and fps > 0:
        return float(fps)

    for path in video_paths:
        if not path.exists():
            continue
        cap = cv2.VideoCapture(str(path))
        if cap.isOpened():
            video_fps = float(cap.get(cv2.CAP_PROP_FPS))
            cap.release()
            if video_fps > 0:
                return video_fps
        cap.release()
    return 15.0


def output_path_for_episode(output_dir: Path, episode_file: EpisodeFile) -> Path:
    return output_dir / episode_file.chunk_name / f"episode_{episode_file.episode_index:06d}.mp4"


def process_episode(work_item: WorkItem) -> WorkResult:
    try:
        return _process_episode(work_item)
    except Exception as exc:  # Keep one bad episode from terminating the whole batch.
        output_path = output_path_for_episode(work_item.output_dir, work_item.episode_file)
        return WorkResult(
            episode_index=work_item.episode_file.episode_index,
            output_path=output_path,
            status="error",
            message=f"{type(exc).__name__}: {exc}",
            warnings=(),
        )


def _process_episode(work_item: WorkItem) -> WorkResult:
    cv2, np, pq = import_runtime_modules()
    episode_file = work_item.episode_file
    output_path = output_path_for_episode(work_item.output_dir, episode_file)

    if output_path.exists() and not work_item.overwrite:
        return WorkResult(
            episode_index=episode_file.episode_index,
            output_path=output_path,
            status="skipped",
            message="output already exists; pass --overwrite to regenerate",
            warnings=(),
        )

    warnings: list[str] = []
    video_specs = get_video_specs(work_item.modality)
    video_paths = [
        format_video_path(work_item.dataset_root, work_item.info, episode_file, video_spec)
        for video_spec in video_specs
    ]

    schema = pq.read_schema(str(episode_file.parquet_path))
    available_columns = set(schema.names)
    required_columns = build_required_columns(work_item.modality, work_item.plot_mode)
    columns = [name for name in required_columns if name in available_columns]
    if "timestamp" not in columns and "timestamp" in available_columns:
        columns.append("timestamp")
    missing_columns = sorted(required_columns - available_columns)
    for column_name in missing_columns:
        warnings.append(f"parquet column {column_name!r} is missing")

    table = pq.read_table(str(episode_file.parquet_path), columns=columns or None)
    total_frames = table.num_rows
    if total_frames == 0:
        return WorkResult(
            episode_index=episode_file.episode_index,
            output_path=output_path,
            status="error",
            message="parquet has zero rows",
            warnings=tuple(warnings),
        )

    series, series_warnings = build_plot_series(table, work_item.modality, work_item.plot_mode, np)
    warnings.extend(series_warnings)
    if not series:
        warnings.append("no plot series were available; rendering an empty plot area")

    if "timestamp" in table.column_names:
        timestamps = table_column_to_vector(table, "timestamp", np).astype(np.float64)
    else:
        timestamps = None

    tasks = episode_tasks(work_item.dataset_root, work_item.episode_meta, table)
    success = episode_success(work_item.episode_meta)
    target_size = target_video_size(work_item.info, video_specs, video_paths, cv2)
    fps = resolve_fps(work_item.info, video_paths, cv2)
    frame_count = total_frames
    if work_item.max_frames is not None:
        frame_count = min(frame_count, work_item.max_frames)

    output_width = target_size[0] * max(1, len(video_specs))
    static_lines = build_static_text_lines(tasks, output_width - 24, cv2)
    header_height = max(96, 48 + 21 * len(static_lines))
    plot_height = 260 if len(series) <= 16 else 320
    output_height = header_height + plot_height + 34 + target_size[1]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (output_width, output_height),
    )
    if not writer.isOpened():
        return WorkResult(
            episode_index=episode_file.episode_index,
            output_path=output_path,
            status="error",
            message="failed to open cv2.VideoWriter",
            warnings=tuple(warnings),
        )

    readers = [
        SequentialVideoReader(video_spec, video_path, target_size, cv2, np)
        for video_spec, video_path in zip(video_specs, video_paths)
    ]

    try:
        for frame_index in range(frame_count):
            if timestamps is not None and frame_index < len(timestamps):
                timestamp = float(timestamps[frame_index])
            else:
                timestamp = frame_index / fps
            top_panel = render_top_panel(
                output_width,
                header_height,
                plot_height,
                static_lines,
                series,
                episode_file.episode_index,
                success,
                frame_index,
                total_frames,
                timestamp,
                work_item.plot_mode,
                cv2,
                np,
            )
            camera_frames = [reader.read(frame_index) for reader in readers]
            if not camera_frames:
                camera_frames = [
                    make_placeholder_frame(
                        target_size[0],
                        target_size[1],
                        "video",
                        "no video modalities provided",
                        cv2,
                        np,
                    )
                ]
            stitched = np.concatenate(camera_frames, axis=1)
            frame = np.vstack((top_panel, stitched))
            writer.write(frame)
    finally:
        writer.release()
        for reader in readers:
            reader.release()

    for reader in readers:
        if reader.error:
            warnings.append(reader.error)

    return WorkResult(
        episode_index=episode_file.episode_index,
        output_path=output_path,
        status="written",
        message=f"wrote {frame_count} frames at {fps:g} FPS",
        warnings=tuple(warnings),
    )


def print_dataset_summary(
    dataset_root: Path,
    info: dict[str, Any],
    modality: dict[str, Any],
    episode_files: list[EpisodeFile],
) -> None:
    print(f"Dataset root: {dataset_root}")
    print(f"Parquet episodes on disk: {len(episode_files)}")

    by_chunk: dict[str, int] = {}
    for item in episode_files:
        by_chunk[item.chunk_name] = by_chunk.get(item.chunk_name, 0) + 1
    if by_chunk:
        chunk_summary = ", ".join(f"{name}: {count}" for name, count in sorted(by_chunk.items()))
        print(f"Chunks on disk: {chunk_summary}")
    else:
        print("Chunks on disk: none")

    info_total = info.get("total_episodes")
    if info_total is not None:
        print(f"meta/info.json total_episodes: {info_total}")
        if info_total != len(episode_files):
            print(
                "warning: meta/info.json total_episodes does not match disk scan "
                f"({info_total} vs {len(episode_files)})"
            )

    info_chunks = info.get("total_chunks")
    if info_chunks is not None and by_chunk and info_chunks != len(by_chunk):
        print(
            "warning: meta/info.json total_chunks does not match disk scan "
            f"({info_chunks} vs {len(by_chunk)})"
        )

    episodes_jsonl_count = count_jsonl_lines(dataset_root / "meta" / "episodes.jsonl")
    tasks_jsonl_count = count_jsonl_lines(dataset_root / "meta" / "tasks.jsonl")
    if episodes_jsonl_count is not None:
        print(f"meta/episodes.jsonl lines: {episodes_jsonl_count}")
    if tasks_jsonl_count is not None:
        print(f"meta/tasks.jsonl lines: {tasks_jsonl_count}")

    video_specs = get_video_specs(modality)
    print(
        "Video modalities: "
        + ", ".join(f"{spec.name} ({spec.original_key})" for spec in video_specs)
    )
    camera_dirs = sorted(
        path for path in (dataset_root / "videos").glob("chunk-*/*") if path.is_dir()
    )
    print(f"Camera directories on disk: {len(camera_dirs)}")
    if camera_dirs:
        for path in camera_dirs[:12]:
            video_count = len(list(path.glob("episode_*.mp4")))
            print(f"  {path.relative_to(dataset_root)}: {video_count} mp4 files")
        if len(camera_dirs) > 12:
            print(f"  ... {len(camera_dirs) - 12} more camera directories")


def print_no_selection_help(dataset_root: Path) -> None:
    print()
    print("No videos requested, so no output was generated.")
    print(
        "Pass one of --all, --episodes, --episode-range, --chunks, or --random "
        "to write videos."
    )
    print("Examples:")
    print(
        "  python scripts/data/visualize_lerobot_episodes.py "
        f"--dataset-root {dataset_root} --episodes 0 1 2"
    )
    print(
        "  python scripts/data/visualize_lerobot_episodes.py "
        f"--dataset-root {dataset_root} --chunks chunk-000"
    )
    print(
        "  python scripts/data/visualize_lerobot_episodes.py "
        f"--dataset-root {dataset_root} --random 1000"
    )


def make_parser() -> argparse.ArgumentParser:
    epilog = """
Examples:
  Preview summary only:
    python scripts/data/visualize_lerobot_episodes.py \\
        --dataset-root /data/dataset/dreamzero/droid_lerobot

  Generate a short sample:
    python scripts/data/visualize_lerobot_episodes.py \\
        --dataset-root /data/dataset/dreamzero/droid_lerobot \\
        --episodes 0 1 2

  Generate all episodes in chunk-000:
    python scripts/data/visualize_lerobot_episodes.py \\
        --dataset-root /data/dataset/dreamzero/droid_lerobot \\
        --chunks chunk-000

  Randomly sample 1000 episodes found on disk:
    python scripts/data/visualize_lerobot_episodes.py \\
        --dataset-root /data/dataset/dreamzero/droid_lerobot \\
        --random 1000
"""
    parser = argparse.ArgumentParser(
        description=(
            "Visualize DROID LeRobot episodes with stitched cameras and "
            "state/action traces."
        ),
        epilog=textwrap.dedent(epilog),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"LeRobot dataset root. Default: {DEFAULT_DATASET_ROOT}",
    )
    parser.add_argument(
        "--output-dir",
        default="visualized_videos",
        help="Output directory. Relative paths are resolved under --dataset-root.",
    )
    parser.add_argument(
        "--episodes",
        nargs="+",
        type=int,
        help="Episode indices to visualize, for example: --episodes 0 1 2.",
    )
    parser.add_argument(
        "--episode-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Inclusive episode index range to visualize.",
    )
    parser.add_argument(
        "--chunks",
        nargs="+",
        help="Chunk names or numbers to visualize, for example: --chunks chunk-000 or --chunks 0.",
    )
    parser.add_argument("--all", action="store_true", help="Visualize every episode found on disk.")
    parser.add_argument(
        "--random",
        type=int,
        metavar="N",
        help="Randomly sample N episodes from all parquet episodes found on disk.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        help="Optional random seed for reproducible --random sampling.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Number of worker processes. Default: 32.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate output videos that already exist.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        help="Limit the number of selected episodes after sorting.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Limit output frames per episode for quick previews.",
    )
    parser.add_argument(
        "--plot-mode",
        choices=["state-action", "state", "action"],
        default="state-action",
        help="Which low-dimensional traces to render. Default: state-action.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.max_episodes is not None and args.max_episodes < 1:
        raise ValueError("--max-episodes must be >= 1")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max-frames must be >= 1")
    if args.random is not None and args.random < 1:
        raise ValueError("--random must be >= 1")


def run_generation(
    args: argparse.Namespace,
    info: dict[str, Any],
    modality: dict[str, Any],
    selected: list[EpisodeFile],
) -> int:
    from tqdm import tqdm

    output_dir = resolve_output_dir(args.dataset_root, args.output_dir)
    selected_indices = {item.episode_index for item in selected}
    episode_metadata = load_selected_episode_metadata(
        args.dataset_root / "meta" / "episodes.jsonl",
        selected_indices,
    )
    work_items = [
        WorkItem(
            dataset_root=args.dataset_root,
            output_dir=output_dir,
            episode_file=item,
            episode_meta=episode_metadata.get(item.episode_index, {}),
            info=info,
            modality=modality,
            plot_mode=args.plot_mode,
            max_frames=args.max_frames,
            overwrite=args.overwrite,
        )
        for item in selected
    ]

    print(f"Selected episodes: {len(work_items)}")
    print(f"Output directory: {output_dir}")

    results: list[WorkResult] = []
    if args.workers == 1:
        for work_item in tqdm(work_items, desc="Visualizing", unit="episode"):
            results.append(process_episode(work_item))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_episode, work_item) for work_item in work_items]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Visualizing",
                unit="episode",
            ):
                results.append(future.result())

    results.sort(key=lambda item: item.episode_index)
    status_counts: dict[str, int] = {}
    for result in results:
        status_counts[result.status] = status_counts.get(result.status, 0) + 1
        print(
            f"{result.status}: episode {result.episode_index:06d}: "
            f"{result.output_path} ({result.message})"
        )
        for warning in result.warnings:
            print(f"warning: episode {result.episode_index:06d}: {warning}", file=sys.stderr)

    summary = ", ".join(f"{status}={count}" for status, count in sorted(status_counts.items()))
    print(f"Done: {summary}")
    return 1 if status_counts.get("error") else 0


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
        args.dataset_root = args.dataset_root.expanduser().resolve()
        info = load_json(args.dataset_root / "meta" / "info.json")
        modality = load_json(args.dataset_root / "meta" / "modality.json")
        episode_files = scan_episode_files(args.dataset_root)
        print_dataset_summary(args.dataset_root, info, modality, episode_files)

        if not selection_requested(args):
            print_no_selection_help(args.dataset_root)
            return 0

        selected, warnings = resolve_episode_selection(args, episode_files)
        for warning in warnings:
            print(f"warning: {warning}", file=sys.stderr)
        if not selected:
            print("No matching episodes were selected.", file=sys.stderr)
            return 1

        require_runtime_dependencies()
        return run_generation(args, info, modality, selected)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
