#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


RAW_ROOT = Path("/data/datasets/dreamzero/basic_pick_place")
OUTPUT_ROOT = Path("/data/datasets/dreamzero/basic_pick_place_ego_160x320")
VIDEO_KEY = "observation.images.ego"
CHUNKS_SIZE = 1000


@dataclass(frozen=True)
class EpisodeInfo:
    raw_id: int
    raw_hdf5: Path
    raw_mp4: Path
    length: int
    task_text: str
    attrs: dict[str, object]


@dataclass(frozen=True)
class VideoTask:
    src: Path
    dst: Path
    frames: int


class RunningStats:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.count = 0
        self.total = np.zeros(dim, dtype=np.float64)
        self.total_sq = np.zeros(dim, dtype=np.float64)
        self.min = np.full(dim, np.inf, dtype=np.float64)
        self.max = np.full(dim, -np.inf, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        rows = values.reshape(-1, self.dim).astype(np.float64, copy=False)
        if rows.size == 0:
            return
        self.count += rows.shape[0]
        self.total += rows.sum(axis=0)
        self.total_sq += np.square(rows).sum(axis=0)
        self.min = np.minimum(self.min, rows.min(axis=0))
        self.max = np.maximum(self.max, rows.max(axis=0))

    def as_dict(self) -> dict:
        if self.count == 0:
            zeros = [0.0] * self.dim
            return {
                "mean": zeros,
                "std": zeros,
                "min": zeros,
                "max": zeros,
                "q01": zeros,
                "q99": zeros,
            }
        mean = self.total / self.count
        variance = np.maximum((self.total_sq / self.count) - np.square(mean), 0.0)
        return {
            "mean": mean.tolist(),
            "std": np.sqrt(variance).tolist(),
            "min": self.min.tolist(),
            "max": self.max.tolist(),
            "q01": self.min.tolist(),
            "q99": self.max.tolist(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert raw basic_pick_place ego HDF5/MP4 episodes into a "
            "DreamZero/LeRobot video-only pretrain dataset with resized videos."
        )
    )
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--height", type=int, default=160)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--chunks-size", type=int, default=CHUNKS_SIZE)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/usr/bin/ffmpeg"))
    parser.add_argument("--ffprobe", type=Path, default=Path("/usr/bin/ffprobe"))
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--verify-existing-videos",
        action="store_true",
        help="Probe existing output videos before skipping them.",
    )
    return parser.parse_args()


def json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=json_default)
        handle.write("\n")


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=json_default) + "\n")


def decode_attr(value: object) -> object:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        decoded = [decode_attr(item) for item in value.tolist()]
        return decoded
    if isinstance(value, np.generic):
        return value.item()
    return value


def normalized_task_text(attrs: dict[str, object]) -> str:
    for key in ("llm_description", "description"):
        value = attrs.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() != "none":
            return text
    return "Perform a basic pick and place task."


def discover_episodes(raw_root: Path, limit: int) -> list[EpisodeInfo]:
    hdf5_paths = sorted(
        raw_root.glob("*.hdf5"),
        key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem,
    )
    if limit > 0:
        hdf5_paths = hdf5_paths[:limit]
    episodes: list[EpisodeInfo] = []
    for hdf5_path in tqdm(hdf5_paths, desc="scan:hdf5", unit="episode"):
        if not hdf5_path.stem.isdigit():
            continue
        raw_id = int(hdf5_path.stem)
        mp4_path = raw_root / f"{raw_id}.mp4"
        if not mp4_path.is_file():
            raise FileNotFoundError(mp4_path)
        with h5py.File(hdf5_path, "r") as handle:
            if "transforms/camera" not in handle:
                raise KeyError(f"{hdf5_path}: missing transforms/camera")
            length = int(handle["transforms/camera"].shape[0])
            attrs = {key: decode_attr(handle.attrs[key]) for key in handle.attrs.keys()}
        episodes.append(
            EpisodeInfo(
                raw_id=raw_id,
                raw_hdf5=hdf5_path,
                raw_mp4=mp4_path,
                length=length,
                task_text=normalized_task_text(attrs),
                attrs=attrs,
            )
        )
    if not episodes:
        raise ValueError(f"No episodes found in {raw_root}")
    return episodes


def prepare_output_root(output_root: Path, force: bool, resume: bool) -> None:
    if output_root.exists() and force:
        shutil.rmtree(output_root)
    if output_root.exists() and not force and not resume:
        raise FileExistsError(f"{output_root} exists. Use --force or --resume.")
    (output_root / "data").mkdir(parents=True, exist_ok=True)
    (output_root / "videos").mkdir(parents=True, exist_ok=True)
    (output_root / "meta").mkdir(parents=True, exist_ok=True)


def build_task_index(episodes: list[EpisodeInfo]) -> tuple[list[dict], dict[str, int]]:
    task_to_index: dict[str, int] = {}
    for episode in episodes:
        if episode.task_text not in task_to_index:
            task_to_index[episode.task_text] = len(task_to_index)
    tasks = [
        {"task_index": index, "task": task}
        for task, index in sorted(task_to_index.items(), key=lambda item: item[1])
    ]
    return tasks, task_to_index


def episode_data_path(output_root: Path, episode_index: int, chunks_size: int) -> Path:
    chunk = episode_index // chunks_size
    return output_root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"


def episode_video_path(output_root: Path, episode_index: int, chunks_size: int) -> Path:
    chunk = episode_index // chunks_size
    return (
        output_root
        / "videos"
        / f"chunk-{chunk:03d}"
        / VIDEO_KEY
        / f"episode_{episode_index:06d}.mp4"
    )


def write_episode_parquet(
    output_root: Path,
    output_episode_index: int,
    global_start_index: int,
    episode: EpisodeInfo,
    task_index: int,
    args: argparse.Namespace,
) -> Path:
    path = episode_data_path(output_root, output_episode_index, args.chunks_size)
    if path.exists() and args.resume:
        return path

    frame_index = np.arange(episode.length, dtype=np.int64)
    df = pd.DataFrame(
        {
            "timestamp": (frame_index.astype(np.float32) / float(args.fps)).astype(np.float32),
            "frame_index": frame_index,
            "episode_index": np.full(episode.length, output_episode_index, dtype=np.int64),
            "index": (global_start_index + frame_index).astype(np.int64),
            "task_index": np.full(episode.length, task_index, dtype=np.int64),
            "annotation.task": [episode.task_text] * episode.length,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def probe_video_ok(path: Path, args: argparse.Namespace, expected_frames: int | None = None) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    command = [
        str(args.ffprobe),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,nb_frames,avg_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        streams = json.loads(result.stdout).get("streams", [])
    except Exception:
        return False
    if not streams:
        return False
    stream = streams[0]
    if int(stream.get("width", -1)) != args.width:
        return False
    if int(stream.get("height", -1)) != args.height:
        return False
    if stream.get("codec_name") != "h264":
        return False
    if expected_frames is not None and str(stream.get("nb_frames", "")).isdigit():
        if int(stream["nb_frames"]) != int(expected_frames):
            return False
    return True


def convert_one_video(video_task: VideoTask, args: argparse.Namespace) -> tuple[str, str]:
    dst = video_task.dst
    if args.verify_existing_videos:
        if probe_video_ok(dst, args, video_task.frames):
            return "skipped", str(dst)
    elif dst.exists() and dst.stat().st_size > 0:
        return "skipped", str(dst)

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + f".tmp-{os.getpid()}")
    if tmp.exists():
        tmp.unlink()

    command = [
        str(args.ffmpeg),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-threads",
        "1",
        "-i",
        str(video_task.src),
        "-vf",
        f"scale={args.width}:{args.height}",
        "-an",
        "-r",
        str(args.fps),
        "-c:v",
        "libx264",
        "-preset",
        args.preset,
        "-crf",
        str(args.crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-f",
        "mp4",
        str(tmp),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        tmp.replace(dst)
        return "converted", str(dst)
    except subprocess.CalledProcessError as exc:
        if tmp.exists():
            tmp.unlink()
        message = (exc.stderr or exc.stdout or "").strip().splitlines()[-6:]
        return "failed", f"{video_task.src}: {' | '.join(message)}"


def convert_videos(video_tasks: list[VideoTask], args: argparse.Namespace, output_root: Path) -> None:
    converted = skipped = failed = 0
    failures: list[str] = []
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(convert_one_video, task, args) for task in video_tasks]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            status, detail = future.result()
            if status == "converted":
                converted += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
                failures.append(detail)
            if index == 1 or index % 500 == 0 or index == len(video_tasks):
                elapsed = max(time.time() - start, 1e-6)
                rate = index / elapsed
                eta = (len(video_tasks) - index) / rate if rate > 0 else 0.0
                print(
                    "[videos] "
                    f"{index}/{len(video_tasks)} converted={converted} skipped={skipped} "
                    f"failed={failed} rate={rate:.2f}/s eta={eta / 60:.1f}m",
                    flush=True,
                )
    if failures:
        failure_path = output_root / "video_conversion_failures.txt"
        failure_path.write_text("\n".join(failures) + "\n", encoding="utf-8")
        raise RuntimeError(f"{failed} video conversions failed; see {failure_path}")


def build_info(
    *,
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    chunks_size: int,
    height: int,
    width: int,
    fps: int,
) -> dict:
    video_info = {
        "video.height": height,
        "video.width": width,
        "video.codec": "h264",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False,
        "video.fps": fps,
        "video.channels": 3,
        "has_audio": False,
    }
    return {
        "codebase_version": "v2.1",
        "robot_type": "ego_video_only",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": total_episodes,
        "total_chunks": (total_episodes + chunks_size - 1) // chunks_size,
        "chunks_size": chunks_size,
        "fps": float(fps),
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            VIDEO_KEY: {
                "dtype": "video",
                "shape": [3, height, width],
                "names": ["channels", "height", "width"],
                "info": video_info,
            },
            "annotation.task": {
                "dtype": "string",
                "shape": [1],
                "names": None,
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }


def build_modality() -> dict:
    return {
        "state": {},
        "action": {},
        "video": {
            "ego": {
                "original_key": VIDEO_KEY,
            }
        },
        "annotation": {
            "task": {
                "original_key": "annotation.task",
            }
        },
    }


def build_empty_stats(total_frames: int, fps: int) -> dict:
    if total_frames <= 0:
        timestamps = np.array([0.0], dtype=np.float64)
    else:
        timestamps = np.arange(total_frames, dtype=np.float64) / float(fps)
    stats = RunningStats(1)
    stats.update(timestamps.reshape(-1, 1))
    return {
        "num_trajectories": 0,
        "total_trajectory_length": int(total_frames),
        "timestamp": stats.as_dict(),
    }


def build_dataset(episodes: list[EpisodeInfo], args: argparse.Namespace) -> dict:
    output_root = args.output_root.resolve()
    prepare_output_root(output_root, force=args.force, resume=args.resume)
    tasks_jsonl, task_to_index = build_task_index(episodes)

    total_frames = 0
    global_start_index = 0
    episodes_jsonl: list[dict] = []
    episodes_detail: list[dict] = []
    video_tasks: list[VideoTask] = []
    timestamp_stats = RunningStats(1)

    for output_episode_index, episode in enumerate(
        tqdm(episodes, desc="write:parquet", unit="episode")
    ):
        task_index = task_to_index[episode.task_text]
        write_episode_parquet(
            output_root=output_root,
            output_episode_index=output_episode_index,
            global_start_index=global_start_index,
            episode=episode,
            task_index=task_index,
            args=args,
        )
        timestamps = np.arange(episode.length, dtype=np.float64) / float(args.fps)
        timestamp_stats.update(timestamps.reshape(-1, 1))
        video_dst = episode_video_path(output_root, output_episode_index, args.chunks_size)
        video_tasks.append(VideoTask(src=episode.raw_mp4, dst=video_dst, frames=episode.length))
        episodes_jsonl.append(
            {
                "episode_index": output_episode_index,
                "tasks": [episode.task_text],
                "length": episode.length,
            }
        )
        detail = {
            "episode_index": output_episode_index,
            "raw_episode_index": episode.raw_id,
            "length": episode.length,
            "task_index": task_index,
            "task": episode.task_text,
            "attrs": episode.attrs,
        }
        episodes_detail.append(detail)
        global_start_index += episode.length
        total_frames += episode.length

    info = build_info(
        total_episodes=len(episodes_jsonl),
        total_frames=total_frames,
        total_tasks=len(tasks_jsonl),
        chunks_size=args.chunks_size,
        height=args.height,
        width=args.width,
        fps=args.fps,
    )
    write_json(output_root / "meta" / "info.json", info)
    write_json(output_root / "meta" / "modality.json", build_modality())
    write_json(
        output_root / "meta" / "embodiment.json",
        {"robot_type": "ego_video_only", "embodiment_tag": "mecka_hands"},
    )
    write_jsonl(output_root / "meta" / "tasks.jsonl", tasks_jsonl)
    write_jsonl(output_root / "meta" / "episodes.jsonl", episodes_jsonl)
    write_jsonl(output_root / "meta" / "episodes_detail_global_instruction.jsonl", episodes_detail)
    write_json(output_root / "meta" / "stats.json", {
        "num_trajectories": len(episodes_jsonl),
        "total_trajectory_length": total_frames,
        "timestamp": timestamp_stats.as_dict(),
    })
    write_json(output_root / "meta" / "relative_stats_dreamzero.json", {})
    write_json(output_root / "meta" / "smoke_episode_filter.json", [0, 1])

    print(f"[convert] videos={len(video_tasks)} workers={args.workers}", flush=True)
    convert_videos(video_tasks, args, output_root)

    summary = {
        "raw_root": str(args.raw_root.resolve()),
        "output_root": str(output_root),
        "episodes": len(episodes_jsonl),
        "frames": total_frames,
        "hours_at_fps": total_frames / float(args.fps) / 3600.0,
        "tasks": len(tasks_jsonl),
        "videos": len(video_tasks),
        "height": args.height,
        "width": args.width,
        "fps": args.fps,
        "video_key": VIDEO_KEY,
        "embodiment_tag": "mecka_hands",
        "format": "video_only_pretrain",
    }
    write_json(output_root / "meta" / "build_summary.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    args.raw_root = args.raw_root.resolve()
    args.output_root = args.output_root.resolve()
    if not args.raw_root.exists():
        raise FileNotFoundError(args.raw_root)
    if not args.ffmpeg.exists():
        raise FileNotFoundError(args.ffmpeg)
    if not args.ffprobe.exists():
        raise FileNotFoundError(args.ffprobe)

    print(f"[setup] raw_root={args.raw_root}", flush=True)
    print(f"[setup] output_root={args.output_root}", flush=True)
    print(f"[setup] target={args.height}x{args.width} fps={args.fps}", flush=True)
    episodes = discover_episodes(args.raw_root, args.limit)
    print(f"[setup] episodes={len(episodes)}", flush=True)
    summary = build_dataset(episodes, args)
    print("[done] " + json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
