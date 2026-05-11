#!/usr/bin/env python3
"""Convert RoboTwin/LeRobot v3 shards into a DreamZero-compatible layout.

The downloaded RoboTwin dataset uses the LeRobot v3 shard layout:

    data/chunk-000/file-000.parquet
    videos/observation.images.cam_left_wrist/chunk-000/file-000.mp4
    meta/episodes/chunk-000/file-000.parquet

DreamZero's current sharded loader expects one parquet/video entry per episode
with the LeRobot v2 placeholders `{episode_chunk}` and `{episode_index}`. This
script creates a v2-style view of RoboTwin:

    data/chunk-000/episode_000000.parquet
    videos/chunk-000/observation.images.cam_left/episode_000000.mp4

Low-dimensional parquet files are sliced per episode. Video files are linked
back to the original shard files, and per-episode timestamps are shifted so the
existing video reader lands on the correct segment inside the shared mp4.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

log = logging.getLogger(__name__)

SOURCE_VIDEO_KEYS = {
    "observation.images.cam_high": "observation.images.cam_high",
    "observation.images.cam_left": "observation.images.cam_left_wrist",
    "observation.images.cam_right": "observation.images.cam_right_wrist",
}
DATA_PATTERN = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATTERN = (
    "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
)
CHUNKS_SIZE = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/data/datasets/dreamzero/robotwin_unified"),
        help="RoboTwin LeRobot v3 dataset root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/data/datasets/dreamzero/robotwin_unified_dreamzero"),
        help="DreamZero-compatible output dataset root.",
    )
    parser.add_argument(
        "--retire-source",
        choices=["none", "rename-backup"],
        default="none",
        help="After a successful conversion, optionally rename the source to .raw_backup.",
    )
    parser.add_argument(
        "--link-mode",
        choices=["symlink", "hardlink", "copy"],
        default="symlink",
        help="How to reuse video shard files. Symlink is recommended.",
    )
    parser.add_argument(
        "--smoke-episodes",
        type=int,
        default=8,
        help="Number of episode indices to write into meta/smoke_episode_filter.json.",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing output directory.")
    parser.add_argument(
        "--validate-episodes",
        type=int,
        default=3,
        help="How many converted episodes to validate before retiring the source.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent, ensure_ascii=False)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_episode_metadata(source: Path) -> pd.DataFrame:
    episode_paths = sorted((source / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not episode_paths:
        raise FileNotFoundError(f"No v3 episode metadata parquet files found under {source}/meta/episodes")

    frames = [pd.read_parquet(path) for path in episode_paths]
    episodes = pd.concat(frames, ignore_index=True)
    episodes = episodes.sort_values("episode_index").reset_index(drop=True)
    return episodes


def load_tasks(source: Path) -> list[dict[str, Any]]:
    tasks_path = source / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        return [{"task_index": 0, "task": ""}]

    tasks_df = pd.read_parquet(tasks_path)
    records: list[dict[str, Any]] = []
    if "task_index" in tasks_df.columns:
        for index, row in tasks_df.iterrows():
            task_text = str(row["task"] if "task" in tasks_df.columns else index)
            records.append({"task_index": int(row["task_index"]), "task": task_text})
    elif {"task_index", "task"}.issubset(tasks_df.index.names):
        for task_index, task in tasks_df.index:
            records.append({"task_index": int(task_index), "task": str(task)})
    elif "task" in tasks_df.columns:
        for idx, row in tasks_df.iterrows():
            records.append({"task_index": int(idx), "task": str(row["task"])})
    else:
        records = [{"task_index": int(i), "task": str(idx)} for i, idx in enumerate(tasks_df.index)]

    seen: set[int] = set()
    unique_records: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: int(item["task_index"])):
        task_index = int(record["task_index"])
        if task_index in seen:
            continue
        seen.add(task_index)
        unique_records.append({"task_index": task_index, "task": str(record["task"])})
    return unique_records


def build_info(source_info: dict[str, Any]) -> dict[str, Any]:
    features = dict(source_info.get("features", {}))
    converted_features: dict[str, Any] = {}
    for key, value in features.items():
        if key == "observation.images.cam_left_wrist":
            converted_features["observation.images.cam_left"] = value
        elif key == "observation.images.cam_right_wrist":
            converted_features["observation.images.cam_right"] = value
        else:
            converted_features[key] = value

    info = dict(source_info)
    info["codebase_version"] = f"{source_info.get('codebase_version', 'v3.0')}-dreamzero-v2-layout"
    info["chunks_size"] = CHUNKS_SIZE
    info["data_path"] = DATA_PATTERN
    info["video_path"] = VIDEO_PATTERN
    info["features"] = converted_features
    return info


def maybe_link(src: Path, dst: Path, link_mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return

    if link_mode == "symlink":
        os.symlink(src, dst)
    elif link_mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError as exc:
            log.warning("Hardlink failed for %s -> %s (%s); falling back to symlink", src, dst, exc)
            os.symlink(src, dst)
    elif link_mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported link mode: {link_mode}")


def source_data_path(source: Path, chunk_index: int, file_index: int) -> Path:
    return source / "data" / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.parquet"


def source_video_path(source_for_links: Path, source_video_key: str, chunk_index: int, file_index: int) -> Path:
    return (
        source_for_links
        / "videos"
        / source_video_key
        / f"chunk-{chunk_index:03d}"
        / f"file-{file_index:03d}.mp4"
    )


def output_data_path(output: Path, episode_index: int) -> Path:
    return output / DATA_PATTERN.format(
        episode_chunk=episode_index // CHUNKS_SIZE,
        episode_index=episode_index,
    )


def output_video_path(output: Path, episode_index: int, video_key: str) -> Path:
    return output / VIDEO_PATTERN.format(
        episode_chunk=episode_index // CHUNKS_SIZE,
        episode_index=episode_index,
        video_key=video_key,
    )


def row_value(row: pd.Series, key: str, default: Any = None) -> Any:
    if key not in row:
        return default
    value = row[key]
    if isinstance(value, (list, tuple, dict)):
        return value
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    return value


def episode_tasks(row: pd.Series, task_by_index: dict[int, str]) -> list[str]:
    tasks = row_value(row, "tasks")
    if isinstance(tasks, np.ndarray):
        tasks = tasks.tolist()
    if isinstance(tasks, (list, tuple)) and len(tasks) > 0:
        return [str(task) for task in tasks]
    if isinstance(tasks, str) and tasks:
        return [tasks]
    task_index = row_value(row, "task_index")
    if task_index is not None:
        return [task_by_index.get(int(task_index), "")]
    return [""]


def copy_stats_if_available(source: Path, output: Path) -> None:
    stats_path = source / "meta" / "stats.json"
    if not stats_path.exists():
        return

    stats = read_json(stats_path)
    renamed: dict[str, Any] = {}
    for key, value in stats.items():
        if key == "observation.images.cam_left_wrist":
            renamed["observation.images.cam_left"] = value
        elif key == "observation.images.cam_right_wrist":
            renamed["observation.images.cam_right"] = value
        else:
            renamed[key] = value
    write_json(output / "meta" / "stats.json", renamed, indent=2)


def convert_dataset(
    source: Path,
    output: Path,
    source_for_links: Path,
    *,
    link_mode: str,
    smoke_episodes: int,
) -> None:
    source_info = read_json(source / "meta" / "info.json")
    episodes = load_episode_metadata(source)
    tasks = load_tasks(source)
    task_by_index = {int(record["task_index"]): str(record["task"]) for record in tasks}

    write_json(output / "meta" / "info.json", build_info(source_info), indent=2)
    write_jsonl(output / "meta" / "tasks.jsonl", tasks)
    copy_stats_if_available(source, output)

    episode_records: list[dict[str, Any]] = []
    offset_records: list[dict[str, Any]] = []

    grouped = episodes.groupby(["data/chunk_index", "data/file_index"], sort=True)
    for (data_chunk, data_file), group in tqdm(grouped, desc="Writing per-episode parquets"):
        data_chunk = int(data_chunk)
        data_file = int(data_file)
        src_data_path = source_data_path(source, data_chunk, data_file)
        if not src_data_path.exists():
            raise FileNotFoundError(src_data_path)

        data_df = pd.read_parquet(src_data_path)
        file_base_offset = int(group["dataset_from_index"].min())
        group = group.sort_values("episode_index")

        for _, row in group.iterrows():
            episode_index = int(row["episode_index"])
            start = int(row["dataset_from_index"]) - file_base_offset
            end = int(row["dataset_to_index"]) - file_base_offset
            length = int(row["length"])
            if end - start != length:
                raise ValueError(
                    f"Episode {episode_index} length mismatch: slice {end - start}, metadata {length}"
                )

            episode_df = data_df.iloc[start:end].copy()
            if len(episode_df) != length:
                raise ValueError(
                    f"Episode {episode_index} slice produced {len(episode_df)} rows, expected {length}"
                )
            if "timestamp" in episode_df.columns:
                ts_offset = float(
                    row_value(row, "videos/observation.images.cam_high/from_timestamp", 0.0)
                )
                episode_df["timestamp"] = episode_df["timestamp"].astype(float) + ts_offset
            episode_df["frame_index"] = range(length)
            episode_df["episode_index"] = episode_index
            episode_df["index"] = range(length)

            out_data_path = output_data_path(output, episode_index)
            out_data_path.parent.mkdir(parents=True, exist_ok=True)
            episode_df.to_parquet(out_data_path, index=False)

            for out_video_key, src_video_key in SOURCE_VIDEO_KEYS.items():
                video_chunk = int(row_value(row, f"videos/{src_video_key}/chunk_index", data_chunk))
                video_file = int(row_value(row, f"videos/{src_video_key}/file_index", data_file))
                actual_src_video = source_video_path(source, src_video_key, video_chunk, video_file)
                link_src_video = source_video_path(source_for_links, src_video_key, video_chunk, video_file)
                if not actual_src_video.exists():
                    raise FileNotFoundError(actual_src_video)
                maybe_link(link_src_video, output_video_path(output, episode_index, out_video_key), link_mode)

            tasks_for_episode = episode_tasks(row, task_by_index)
            episode_records.append(
                {
                    "episode_index": episode_index,
                    "tasks": tasks_for_episode,
                    "length": length,
                }
            )
            offset_records.append(
                {
                    "episode_index": episode_index,
                    "source_data_chunk_index": data_chunk,
                    "source_data_file_index": data_file,
                    "source_dataset_from_index": int(row["dataset_from_index"]),
                    "source_dataset_to_index": int(row["dataset_to_index"]),
                    "source_video_offsets": {
                        out_key: {
                            "source_key": src_key,
                            "chunk_index": int(row_value(row, f"videos/{src_key}/chunk_index", data_chunk)),
                            "file_index": int(row_value(row, f"videos/{src_key}/file_index", data_file)),
                            "from_timestamp": float(row_value(row, f"videos/{src_key}/from_timestamp", 0.0)),
                            "to_timestamp": float(row_value(row, f"videos/{src_key}/to_timestamp", 0.0)),
                        }
                        for out_key, src_key in SOURCE_VIDEO_KEYS.items()
                    },
                }
            )

    episode_records.sort(key=lambda item: int(item["episode_index"]))
    offset_records.sort(key=lambda item: int(item["episode_index"]))
    write_jsonl(output / "meta" / "episodes.jsonl", episode_records)
    write_jsonl(output / "meta" / "robotwin_v3_offsets.jsonl", offset_records)
    write_json(
        output / "meta" / "smoke_episode_filter.json",
        {"episode_indices": [record["episode_index"] for record in episode_records[:smoke_episodes]]},
        indent=2,
    )


def validate_output(output: Path, count: int, *, require_video_targets: bool = True) -> None:
    info = read_json(output / "meta" / "info.json")
    with (output / "meta" / "episodes.jsonl").open("r", encoding="utf-8") as f:
        episodes = [json.loads(line) for line in f if line.strip()]
    if not episodes:
        raise ValueError("Converted dataset has no episodes.")

    for record in episodes[:count]:
        episode_index = int(record["episode_index"])
        parquet_path = output_data_path(output, episode_index)
        if not parquet_path.exists():
            raise FileNotFoundError(parquet_path)
        df = pd.read_parquet(parquet_path)
        if len(df) != int(record["length"]):
            raise ValueError(f"Episode {episode_index} length mismatch after conversion")
        state0 = df["observation.state"].iloc[0]
        action0 = df["action"].iloc[0]
        if len(state0) != 14 or len(action0) != 14:
            raise ValueError(
                f"Episode {episode_index} expected state/action dims 14, got {len(state0)}/{len(action0)}"
            )
        for video_key in SOURCE_VIDEO_KEYS:
            path = output_video_path(output, episode_index, video_key)
            if require_video_targets and not path.exists():
                raise FileNotFoundError(path)
            if not require_video_targets and not (path.exists() or path.is_symlink()):
                raise FileNotFoundError(path)

    expected_keys = {
        "observation.images.cam_high",
        "observation.images.cam_left",
        "observation.images.cam_right",
    }
    feature_keys = set(info.get("features", {}))
    missing = expected_keys - feature_keys
    if missing:
        raise ValueError(f"Converted info.json is missing video features: {sorted(missing)}")


def retire_source(source: Path) -> None:
    backup = source.with_name(f"{source.name}.raw_backup")
    if backup.exists():
        raise FileExistsError(f"Backup path already exists: {backup}")
    source.rename(backup)
    log.info("Renamed source dataset to %s", backup)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.exists():
        backup = source.with_name(f"{source.name}.raw_backup")
        if backup.exists():
            raise FileNotFoundError(
                f"Source {source} is already retired; use {backup} as the source or the converted output."
            )
        raise FileNotFoundError(source)
    if not (source / "meta" / "info.json").exists():
        raise FileNotFoundError(source / "meta" / "info.json")

    if output.exists():
        if not args.force:
            raise FileExistsError(f"Output exists: {output}. Use --force to replace it.")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    backup = source.with_name(f"{source.name}.raw_backup")
    if args.retire_source == "rename-backup" and backup.exists():
        raise FileExistsError(f"Cannot retire source because backup already exists: {backup}")

    source_for_links = backup if args.retire_source == "rename-backup" else source
    convert_dataset(
        source,
        output,
        source_for_links,
        link_mode=args.link_mode,
        smoke_episodes=args.smoke_episodes,
    )
    validate_output(
        output,
        args.validate_episodes,
        require_video_targets=args.retire_source != "rename-backup",
    )

    if args.retire_source == "rename-backup":
        retire_source(source)
        validate_output(output, args.validate_episodes, require_video_targets=True)

    log.info("Converted RoboTwin dataset: %s", output)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log.error("%s", exc)
        sys.exit(1)
