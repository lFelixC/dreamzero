#!/usr/bin/env python3
"""Split RoboTwin shard-linked videos into true per-episode videos.

The existing RoboTwin DreamZero conversion creates one parquet per episode, but
the mp4 paths are hardlinks to the original LeRobot v3 shard videos. This script
keeps the same LeRobot/DreamZero directory layout and replaces each episode mp4
with a stream-copied segment for that episode. It also rewrites each episode
parquet timestamp column back to local episode time.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from decord import VideoReader, cpu
from tqdm import tqdm


LOG = logging.getLogger("split_robotwin_episode_videos")

VIDEO_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left",
    "observation.images.cam_right",
)

MANIFEST_NAME = "robotwin_episode_video_split_manifest.json"
ANCHOR_INDEX_NAME = "robotwin_episode_video_split_anchor_index.json"
ANCHOR_DIR_NAME = ".robotwin_episode_video_sources"


@dataclass(frozen=True)
class EpisodeRecord:
    episode_index: int
    length: int
    source_video_offsets: dict[str, dict[str, Any]]


@dataclass
class EpisodeResult:
    episode_index: int
    parquet_updated: int = 0
    parquet_skipped: int = 0
    videos_written: int = 0
    videos_skipped: int = 0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def format_data_path(dataset_root: Path, info: dict[str, Any], episode_index: int) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    pattern = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    return dataset_root / pattern.format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
    )


def format_video_path(
    dataset_root: Path,
    info: dict[str, Any],
    episode_index: int,
    video_key: str,
) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    pattern = info.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    )
    return dataset_root / pattern.format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
        video_key=video_key,
    )


def source_id(video_key: str, offset: dict[str, Any]) -> str:
    return "|".join(
        [
            video_key,
            str(offset.get("source_key", video_key)),
            str(int(offset["chunk_index"])),
            str(int(offset["file_index"])),
        ]
    )


def load_episode_records(dataset_root: Path) -> tuple[dict[str, Any], list[EpisodeRecord]]:
    info_path = dataset_root / "meta" / "info.json"
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    offsets_path = dataset_root / "meta" / "robotwin_v3_offsets.jsonl"
    for path in [info_path, episodes_path, offsets_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    info = read_json(info_path)
    lengths = {
        int(record["episode_index"]): int(record["length"])
        for record in iter_jsonl(episodes_path)
    }

    records: list[EpisodeRecord] = []
    for record in iter_jsonl(offsets_path):
        episode_index = int(record["episode_index"])
        if episode_index not in lengths:
            raise ValueError(f"Episode {episode_index} is missing from episodes.jsonl")
        source_offsets = record.get("source_video_offsets", {})
        missing = [key for key in VIDEO_KEYS if key not in source_offsets]
        if missing:
            raise ValueError(f"Episode {episode_index} is missing video offsets: {missing}")
        records.append(
            EpisodeRecord(
                episode_index=episode_index,
                length=lengths[episode_index],
                source_video_offsets=source_offsets,
            )
        )

    records.sort(key=lambda item: item.episode_index)
    return info, records


def video_needs_work_fast(video_path: Path) -> bool:
    if not video_path.exists():
        return True
    return video_path.stat().st_nlink != 1


def build_anchor_index(
    dataset_root: Path,
    info: dict[str, Any],
    records: list[EpisodeRecord],
) -> tuple[dict[str, str], Path, Path]:
    anchor_dir = dataset_root / ANCHOR_DIR_NAME
    index_path = dataset_root / "meta" / ANCHOR_INDEX_NAME

    needed_source_ids: set[str] = set()
    representative_paths: dict[str, Path] = {}
    for record in records:
        for video_key in VIDEO_KEYS:
            offset = record.source_video_offsets[video_key]
            sid = source_id(video_key, offset)
            video_path = format_video_path(dataset_root, info, record.episode_index, video_key)
            if video_needs_work_fast(video_path):
                needed_source_ids.add(sid)
                representative_paths.setdefault(sid, video_path)

    if not needed_source_ids:
        return {}, anchor_dir, index_path

    existing_index: dict[str, str] = {}
    if index_path.exists():
        raw_index = read_json(index_path)
        existing_index = {
            str(key): str(value)
            for key, value in raw_index.get("anchors", raw_index).items()
        }

    anchors: dict[str, str] = {}
    missing_existing: list[str] = []
    for sid in sorted(needed_source_ids):
        existing_anchor = existing_index.get(sid)
        if existing_anchor and Path(existing_anchor).exists():
            anchors[sid] = existing_anchor
        elif existing_anchor:
            missing_existing.append(sid)

    if missing_existing:
        LOG.warning(
            "Ignoring %d stale anchor index entries whose files are missing.",
            len(missing_existing),
        )

    for sid in sorted(needed_source_ids - set(anchors)):
        src_path = representative_paths[sid]
        if not src_path.exists():
            raise FileNotFoundError(f"Cannot create source anchor; missing {src_path}")
        if src_path.stat().st_nlink == 1:
            raise RuntimeError(
                f"Cannot create source anchor for {sid}: {src_path} is already a single-link file. "
                "The dataset may be partially converted without anchors."
            )

        anchor_path = anchor_dir / f"{sid.replace('|', '__')}.mp4"
        anchor_path.parent.mkdir(parents=True, exist_ok=True)
        if not anchor_path.exists():
            tmp_anchor = anchor_path.with_name(f".{anchor_path.name}.{os.getpid()}.tmp")
            if tmp_anchor.exists():
                tmp_anchor.unlink()
            os.link(src_path, tmp_anchor)
            os.replace(tmp_anchor, anchor_path)
        anchors[sid] = str(anchor_path)

    write_json_atomic(
        index_path,
        {
            "created_at": utc_now(),
            "dataset_root": str(dataset_root),
            "anchors": anchors,
        },
    )
    return anchors, anchor_dir, index_path


def parquet_is_local(df: pd.DataFrame, episode_index: int, fps: float) -> bool:
    if len(df) == 0:
        return False
    required = {"timestamp", "frame_index", "episode_index", "index"}
    if not required.issubset(df.columns):
        return False
    expected_ts = np.arange(len(df), dtype=np.float64) / float(fps)
    actual_ts = df["timestamp"].to_numpy(dtype=np.float64)
    if not np.allclose(actual_ts, expected_ts, atol=1e-5, rtol=0):
        return False
    expected_indices = np.arange(len(df), dtype=np.int64)
    if not np.array_equal(df["frame_index"].to_numpy(dtype=np.int64), expected_indices):
        return False
    if not np.array_equal(df["index"].to_numpy(dtype=np.int64), expected_indices):
        return False
    return bool(np.all(df["episode_index"].to_numpy(dtype=np.int64) == int(episode_index)))


def update_episode_parquet(
    dataset_root: Path,
    info: dict[str, Any],
    episode_index: int,
    expected_length: int,
    fps: float,
) -> bool:
    parquet_path = format_data_path(dataset_root, info, episode_index)
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)

    df = pd.read_parquet(parquet_path)
    if len(df) != expected_length:
        raise ValueError(
            f"Episode {episode_index} parquet length {len(df)} != metadata length {expected_length}"
        )
    if parquet_is_local(df, episode_index, fps):
        return False

    df = df.copy()
    df["timestamp"] = np.arange(expected_length, dtype=np.float64) / float(fps)
    df["frame_index"] = np.arange(expected_length, dtype=np.int64)
    df["episode_index"] = np.full(expected_length, int(episode_index), dtype=np.int64)
    df["index"] = np.arange(expected_length, dtype=np.int64)

    tmp_path = parquet_path.with_name(f".{parquet_path.name}.{os.getpid()}.tmp")
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, parquet_path)
    return True


def video_frame_count(video_path: Path) -> tuple[int, float]:
    vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    return len(vr), float(vr.get_avg_fps())


def video_is_complete(
    video_path: Path,
    expected_frames: int,
    fps: float,
    frame_tolerance: int,
) -> bool:
    if not video_path.exists():
        return False
    stat = video_path.stat()
    if stat.st_nlink != 1 or stat.st_size <= 0:
        return False
    try:
        frame_count, avg_fps = video_frame_count(video_path)
    except Exception:
        return False
    return abs(frame_count - expected_frames) <= frame_tolerance and abs(avg_fps - fps) <= 0.25


def run_ffmpeg_stream_copy(
    ffmpeg: str,
    src_path: Path,
    dst_path: Path,
    start_seconds: float,
    duration_seconds: float,
) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst_path.with_name(f".{dst_path.name}.{os.getpid()}.{time.time_ns()}.tmp.mp4")
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-ss",
        f"{start_seconds:.9f}",
        "-i",
        str(src_path),
        "-t",
        f"{duration_seconds:.9f}",
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "copy",
        "-avoid_negative_ts",
        "make_zero",
        "-reset_timestamps",
        "1",
        str(tmp_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        os.replace(tmp_path, dst_path)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"ffmpeg failed for {src_path} -> {dst_path}: {exc.stderr.strip()}"
        ) from exc
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


def split_episode_video(
    *,
    dataset_root: Path,
    info: dict[str, Any],
    anchors: dict[str, str],
    record: EpisodeRecord,
    video_key: str,
    fps: float,
    frame_tolerance: int,
    ffmpeg: str,
    force: bool,
) -> bool:
    dst_path = format_video_path(dataset_root, info, record.episode_index, video_key)
    if video_is_complete(dst_path, record.length, fps, frame_tolerance):
        return False
    if dst_path.exists() and dst_path.stat().st_nlink == 1 and not force:
        raise RuntimeError(f"Refusing to replace existing single-link video without --force: {dst_path}")
    if dst_path.exists() and dst_path.stat().st_nlink != 1 and not force:
        raise RuntimeError(f"Refusing to replace hardlinked shard video without --force: {dst_path}")

    offset = record.source_video_offsets[video_key]
    sid = source_id(video_key, offset)
    anchor = anchors.get(sid)
    if anchor is None:
        raise RuntimeError(f"No source anchor for {sid}; cannot split {dst_path}")
    src_path = Path(anchor)
    if not src_path.exists():
        raise FileNotFoundError(src_path)

    start_seconds = float(offset["from_timestamp"])
    duration_seconds = float(record.length) / float(fps)
    run_ffmpeg_stream_copy(ffmpeg, src_path, dst_path, start_seconds, duration_seconds)

    if not video_is_complete(dst_path, record.length, fps, frame_tolerance):
        try:
            frame_count, avg_fps = video_frame_count(dst_path)
        except Exception as exc:
            raise RuntimeError(f"Output video is not readable: {dst_path}: {exc}") from exc
        raise RuntimeError(
            f"Output video frame count mismatch for {dst_path}: "
            f"frames={frame_count}, fps={avg_fps}, expected={record.length}"
        )
    return True


def process_episode(
    record: EpisodeRecord,
    *,
    dataset_root: Path,
    info: dict[str, Any],
    anchors: dict[str, str],
    fps: float,
    frame_tolerance: int,
    ffmpeg: str,
    force: bool,
) -> EpisodeResult:
    result = EpisodeResult(episode_index=record.episode_index)
    if update_episode_parquet(dataset_root, info, record.episode_index, record.length, fps):
        result.parquet_updated += 1
    else:
        result.parquet_skipped += 1

    for video_key in VIDEO_KEYS:
        if split_episode_video(
            dataset_root=dataset_root,
            info=info,
            anchors=anchors,
            record=record,
            video_key=video_key,
            fps=fps,
            frame_tolerance=frame_tolerance,
            ffmpeg=ffmpeg,
            force=force,
        ):
            result.videos_written += 1
        else:
            result.videos_skipped += 1
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/data/datasets/dreamzero/robotwin_unified_dreamzero"),
    )
    parser.add_argument("--mode", choices=["stream-copy"], default="stream-copy")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument(
        "--frame-tolerance",
        type=int,
        default=8,
        help="Allowed frame-count difference after stream-copy cutting.",
    )
    parser.add_argument(
        "--keep-source-anchors",
        action="store_true",
        help="Keep hidden hardlinks to original shard videos after a successful run.",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(dataset_root)
    if shutil.which(args.ffmpeg) is None:
        raise FileNotFoundError(f"ffmpeg not found on PATH: {args.ffmpeg}")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    info, records = load_episode_records(dataset_root)
    fps = float(args.fps if args.fps is not None else info.get("fps", 30))
    if fps <= 0:
        raise ValueError(f"Invalid fps: {fps}")

    started_at = utc_now()
    manifest_path = dataset_root / "meta" / MANIFEST_NAME
    counters = {
        "episodes_processed": 0,
        "parquet_updated": 0,
        "parquet_skipped": 0,
        "videos_written": 0,
        "videos_skipped": 0,
    }
    failure: dict[str, Any] | None = None

    anchors, anchor_dir, anchor_index_path = build_anchor_index(dataset_root, info, records)
    LOG.info("Dataset root: %s", dataset_root)
    LOG.info("Episodes: %d | fps: %.3f | workers: %d", len(records), fps, args.workers)
    LOG.info("Source anchors needed: %d", len(anchors))

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    process_episode,
                    record,
                    dataset_root=dataset_root,
                    info=info,
                    anchors=anchors,
                    fps=fps,
                    frame_tolerance=int(args.frame_tolerance),
                    ffmpeg=args.ffmpeg,
                    force=bool(args.force),
                )
                for record in records
            ]
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="split episodes",
                unit="episode",
            ):
                try:
                    result = future.result()
                except Exception as exc:
                    failure = {"error": repr(exc)}
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise
                counters["episodes_processed"] += 1
                counters["parquet_updated"] += result.parquet_updated
                counters["parquet_skipped"] += result.parquet_skipped
                counters["videos_written"] += result.videos_written
                counters["videos_skipped"] += result.videos_skipped
    finally:
        status = "failed" if failure else "success"
        manifest = {
            "status": status,
            "started_at": started_at,
            "finished_at": utc_now(),
            "dataset_root": str(dataset_root),
            "mode": args.mode,
            "fps": fps,
            "video_keys": list(VIDEO_KEYS),
            "ffmpeg": args.ffmpeg,
            "frame_tolerance": int(args.frame_tolerance),
            "workers": int(args.workers),
            "anchors": {
                "count": len(anchors),
                "dir": str(anchor_dir),
                "index": str(anchor_index_path),
            },
            "counters": counters,
            "failure": failure,
        }
        write_json_atomic(manifest_path, manifest)

    if not args.keep_source_anchors:
        if anchor_dir.exists():
            shutil.rmtree(anchor_dir)
        if anchor_index_path.exists():
            anchor_index_path.unlink()
    LOG.info("Wrote manifest: %s", manifest_path)
    LOG.info("Completed: %s", json.dumps(counters, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        LOG.error("%s", exc)
        raise SystemExit(1)
