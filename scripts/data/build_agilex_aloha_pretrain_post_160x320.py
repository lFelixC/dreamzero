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
import sys
import time
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm


SOURCE_ROOT = Path("/data/datasets/dreamzero/dreamzero_agilex_aloha/datasets")
PRETRAIN_OUTPUT = Path("/data/datasets/dreamzero/dreamzero_agilex_aloha_pretrain_160x320")
POST_OUTPUT = Path("/data/datasets/dreamzero/dreamzero_agilex_aloha_post_160x320")
POST_PREFIXES = ("lerobot_205__", "lerobot_mars__", "lerobot_table__")
VIDEO_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left",
    "observation.images.cam_right",
)
COMMON_COLUMNS = (
    "observation.state",
    "action",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
)
CHUNKS_SIZE = 1000


@dataclass(frozen=True)
class SourceDataset:
    root: Path
    info: dict
    task_map: dict[int, str]
    episode_map: dict[int, dict]
    discarded_episode_indices: set[int]

    def parquet_path(self, episode_index: int) -> Path:
        chunk = episode_index // int(self.info.get("chunks_size", CHUNKS_SIZE))
        return self.root / self.info["data_path"].format(
            episode_chunk=chunk,
            episode_index=episode_index,
        )

    def video_path(self, episode_index: int, video_key: str) -> Path:
        chunk = episode_index // int(self.info.get("chunks_size", CHUNKS_SIZE))
        return self.root / self.info["video_path"].format(
            episode_chunk=chunk,
            episode_index=episode_index,
            video_key=video_key,
        )


@dataclass(frozen=True)
class EpisodeRef:
    source: SourceDataset
    source_episode_index: int


@dataclass(frozen=True)
class VideoTask:
    src: Path
    dst: Path
    frames: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build physical 160x320 H264 pretrain/post DreamZero ALOHA datasets "
            "from staged AgileX ALOHA LeRobot roots."
        )
    )
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--pretrain-output", type=Path, default=PRETRAIN_OUTPUT)
    parser.add_argument("--post-output", type=Path, default=POST_OUTPUT)
    parser.add_argument(
        "--split",
        choices=["both", "pretrain", "post"],
        default="both",
        help="Which split(s) to build.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit episodes per selected split after shuffling. Intended for smoke tests.",
    )
    parser.add_argument("--height", type=int, default=160)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/opt/ffmpeg7/bin/ffmpeg"))
    parser.add_argument("--ffprobe", type=Path, default=Path("/opt/ffmpeg7/bin/ffprobe"))
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument(
        "--resample-timestamps",
        action="store_true",
        help=(
            "Resample rows by nearest timestamp before writing parquet. By default, "
            "all source parquet rows are preserved and timestamps are normalized to target fps."
        ),
    )
    parser.add_argument(
        "--relative-quantile-samples",
        type=int,
        default=2_000_000,
        help="Reservoir size forwarded to convert_lerobot_to_gear.py.",
    )
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="Skip DreamZero metadata conversion. Mostly useful for debugging video/parquet output.",
    )
    parser.add_argument(
        "--verify-existing-videos",
        action="store_true",
        help="Deprecated compatibility flag. Existing output videos are always probed before skipping.",
    )
    parser.add_argument("--force", action="store_true", help="Delete output split roots first.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Allow existing output roots and skip already converted videos.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_source_dataset(root: Path) -> SourceDataset:
    info = read_json(root / "meta" / "info.json")
    tasks = read_jsonl(root / "meta" / "tasks.jsonl")
    episodes = read_jsonl(root / "meta" / "episodes.jsonl")
    task_map = {int(task["task_index"]): str(task.get("task", "")) for task in tasks}
    episode_map = {int(episode["episode_index"]): episode for episode in episodes}
    return SourceDataset(
        root=root,
        info=info,
        task_map=task_map,
        episode_map=episode_map,
        discarded_episode_indices={
            int(index) for index in info.get("discarded_episode_indices", [])
        },
    )


def discover_sources(source_root: Path) -> tuple[list[SourceDataset], list[SourceDataset]]:
    if not source_root.exists():
        raise FileNotFoundError(source_root)
    pretrain: list[SourceDataset] = []
    post: list[SourceDataset] = []
    for root in sorted(source_root.iterdir()):
        if not root.is_dir():
            continue
        if not (root / "meta" / "info.json").is_file():
            continue
        source = load_source_dataset(root)
        if root.name.startswith(POST_PREFIXES):
            post.append(source)
        else:
            pretrain.append(source)
    return pretrain, post


def build_episode_plan(sources: list[SourceDataset], seed: int, limit: int) -> list[EpisodeRef]:
    plan: list[EpisodeRef] = []
    for source in sources:
        total_episodes = int(source.info["total_episodes"])
        for episode_index in range(total_episodes):
            if episode_index in source.discarded_episode_indices:
                continue
            if episode_index not in source.episode_map:
                raise KeyError(f"{source.root}: episode {episode_index} missing from episodes.jsonl")
            plan.append(EpisodeRef(source=source, source_episode_index=episode_index))
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(plan))
    shuffled = [plan[int(index)] for index in permutation]
    if limit > 0:
        shuffled = shuffled[:limit]
    return shuffled


def output_parquet_path(output_root: Path, episode_index: int) -> Path:
    chunk = episode_index // CHUNKS_SIZE
    return output_root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"


def output_video_path(output_root: Path, episode_index: int, video_key: str) -> Path:
    chunk = episode_index // CHUNKS_SIZE
    return (
        output_root
        / "videos"
        / f"chunk-{chunk:03d}"
        / video_key
        / f"episode_{episode_index:06d}.mp4"
    )


def compute_resample_indices(source_timestamps: np.ndarray, target_fps: int) -> tuple[np.ndarray, np.ndarray]:
    if source_timestamps.ndim != 1:
        raise ValueError("Expected a 1D timestamp array")
    if len(source_timestamps) == 0:
        raise ValueError("Expected at least one timestamp")
    relative_ts = source_timestamps.astype(np.float64) - float(source_timestamps[0])
    if np.any(np.diff(relative_ts) < -1e-6):
        raise ValueError("Source timestamps must be non-decreasing")
    last_ts = float(relative_ts[-1])
    target_length = max(1, int(np.floor(last_ts * target_fps + 1e-6)) + 1)
    target_timestamps = np.arange(target_length, dtype=np.float64) / float(target_fps)

    right = np.searchsorted(relative_ts, target_timestamps, side="left")
    right = np.clip(right, 0, len(relative_ts) - 1)
    left = np.clip(right - 1, 0, len(relative_ts) - 1)
    right_distance = np.abs(relative_ts[right] - target_timestamps)
    left_distance = np.abs(target_timestamps - relative_ts[left])
    selected_indices = np.where(right_distance < left_distance, right, left).astype(np.int64)
    return selected_indices, target_timestamps.astype(np.float32)


def task_text_for_source_index(source: SourceDataset, task_index: int, fallback: str) -> str:
    text = source.task_map.get(int(task_index), "")
    return text if text else fallback


def episode_fallback_task(source: SourceDataset, episode_index: int) -> str:
    episode = source.episode_map[episode_index]
    tasks = episode.get("tasks") or [""]
    return str(tasks[0])


def build_features(template_info: dict, height: int, width: int, fps: int) -> dict:
    state_feature = template_info["features"]["observation.state"]
    action_feature = template_info["features"]["action"]
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [14],
            "names": state_feature.get("names"),
        },
        "action": {
            "dtype": "float32",
            "shape": [14],
            "names": action_feature.get("names"),
        },
    }
    for key in VIDEO_KEYS:
        features[key] = {
            "dtype": "video",
            "shape": [3, height, width],
            "names": ["channels", "height", "width"],
            "info": {
                "video.height": height,
                "video.width": width,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": fps,
                "video.channels": 3,
                "has_audio": False,
            },
        }
    features.update(
        {
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        }
    )
    return features


def scalar_stats(values: np.ndarray) -> dict:
    values = values.reshape(-1)
    return {
        "min": [float(values.min())],
        "max": [float(values.max())],
        "mean": [float(values.mean())],
        "std": [float(values.std())],
        "count": [int(values.shape[0])],
    }


def vector_stats(values: np.ndarray) -> dict:
    return {
        "min": values.min(axis=0).astype(float).tolist(),
        "max": values.max(axis=0).astype(float).tolist(),
        "mean": values.mean(axis=0).astype(float).tolist(),
        "std": values.std(axis=0).astype(float).tolist(),
        "count": [int(values.shape[0])],
    }


def rewrite_episode(
    ref: EpisodeRef,
    output_root: Path,
    output_episode_index: int,
    global_start_index: int,
    task_text_to_index: dict[str, int],
    task_index_to_text: dict[int, str],
    target_fps: int,
    resample_timestamps: bool,
) -> tuple[dict, dict, list[VideoTask], int]:
    source = ref.source
    source_episode_index = ref.source_episode_index
    df = pd.read_parquet(source.parquet_path(source_episode_index))
    missing = [column for column in ("observation.state", "action", "timestamp", "task_index") if column not in df.columns]
    if missing:
        raise KeyError(f"{source.root} episode {source_episode_index} missing columns: {missing}")

    if resample_timestamps:
        selected_indices, target_timestamps = compute_resample_indices(
            df["timestamp"].to_numpy(dtype=np.float64, copy=True),
            target_fps,
        )
    else:
        selected_indices = np.arange(len(df), dtype=np.int64)
        target_timestamps = (selected_indices.astype(np.float32) / float(target_fps))
    length = int(selected_indices.shape[0])
    fallback_task = episode_fallback_task(source, source_episode_index)

    source_task_indices = df["task_index"].to_numpy(dtype=np.int64, copy=True)[selected_indices]
    remapped_task_indices = np.array(
        [
            task_text_to_index[
                task_text_for_source_index(source, int(task_index), fallback_task)
            ]
            for task_index in source_task_indices
        ],
        dtype=np.int64,
    )

    state = np.stack(df["observation.state"].to_numpy()[selected_indices]).astype(np.float32)
    action = np.stack(df["action"].to_numpy()[selected_indices]).astype(np.float32)
    frame_index = np.arange(length, dtype=np.int64)
    episode_index = np.full(length, output_episode_index, dtype=np.int64)
    global_index = np.arange(global_start_index, global_start_index + length, dtype=np.int64)

    output_df = pd.DataFrame(
        {
            "observation.state": list(state),
            "action": list(action),
            "timestamp": target_timestamps.astype(np.float32),
            "frame_index": frame_index,
            "episode_index": episode_index,
            "index": global_index,
            "task_index": remapped_task_indices,
        }
    )
    parquet_path = output_parquet_path(output_root, output_episode_index)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(parquet_path, index=False)

    task_texts = []
    seen_tasks = set()
    for task_index in remapped_task_indices.tolist():
        task_text = task_index_to_text[int(task_index)]
        if task_text not in seen_tasks:
            seen_tasks.add(task_text)
            task_texts.append(task_text)

    episode_record = {
        "episode_index": output_episode_index,
        "tasks": task_texts or [""],
        "length": length,
    }
    stats_record = {
        "episode_index": output_episode_index,
        "stats": {
            "observation.state": vector_stats(state),
            "action": vector_stats(action),
            "timestamp": scalar_stats(target_timestamps.astype(np.float64)),
            "frame_index": scalar_stats(frame_index.astype(np.float64)),
            "episode_index": scalar_stats(episode_index.astype(np.float64)),
            "index": scalar_stats(global_index.astype(np.float64)),
            "task_index": scalar_stats(remapped_task_indices.astype(np.float64)),
        },
    }

    video_tasks = [
        VideoTask(
            src=source.video_path(source_episode_index, video_key),
            dst=output_video_path(output_root, output_episode_index, video_key),
            frames=length,
        )
        for video_key in VIDEO_KEYS
    ]
    return episode_record, stats_record, video_tasks, length


def probe_ok(
    path: Path,
    ffprobe: Path,
    height: int,
    width: int,
    fps: int,
    expected_frames: int | None = None,
) -> bool:
    if path.is_symlink() or not path.exists() or path.stat().st_size <= 0:
        return False
    command = [
        str(ffprobe),
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,nb_frames,nb_read_frames",
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
    if (
        stream.get("codec_name") != "h264"
        or int(stream.get("width", -1)) != width
        or int(stream.get("height", -1)) != height
        or stream.get("avg_frame_rate") != f"{fps}/1"
    ):
        return False
    if expected_frames is None:
        return True
    frame_count = stream.get("nb_read_frames") or stream.get("nb_frames")
    if frame_count in (None, "N/A"):
        return False
    try:
        return int(frame_count) == int(expected_frames)
    except ValueError:
        return False


def convert_video(task: VideoTask, args: argparse.Namespace) -> tuple[str, str]:
    if not task.src.exists():
        return "failed", f"Missing source video: {task.src}"
    if task.dst.is_symlink() or task.dst.exists():
        if probe_ok(task.dst, args.ffprobe, args.height, args.width, args.fps, task.frames):
            return "skipped", str(task.dst)
        if task.dst.is_dir():
            return "failed", f"Output path is a directory: {task.dst}"
        task.dst.unlink()

    task.dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = task.dst.with_suffix(task.dst.suffix + f".tmp-{os.getpid()}")
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
        str(task.src),
        "-vf",
        (
            f"fps={args.fps}:round=near,"
            f"scale={args.width}:{args.height},"
            "tpad=stop_mode=clone:stop=-1"
        ),
        "-an",
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
        "-frames:v",
        str(task.frames),
        "-f",
        "mp4",
        str(tmp),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        if not probe_ok(tmp, args.ffprobe, args.height, args.width, args.fps, task.frames):
            tmp.unlink()
            return "failed", f"Converted video failed probe: {task.dst}"
        tmp.replace(task.dst)
        return "converted", str(task.dst)
    except subprocess.CalledProcessError as exc:
        if tmp.exists():
            tmp.unlink()
        tail = (exc.stderr or exc.stdout or "").strip().splitlines()[-5:]
        return "failed", f"{task.src}: {' | '.join(tail)}"


def convert_videos(video_tasks: list[VideoTask], args: argparse.Namespace, output_root: Path) -> None:
    converted = skipped = failed = 0
    failures: list[str] = []
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(convert_video, task, args) for task in video_tasks]
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


def ffmpeg_supports_encoder(ffmpeg: Path, encoder: str) -> bool:
    try:
        result = subprocess.run(
            [str(ffmpeg), "-hide_banner", "-encoders"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return False
    return encoder in result.stdout


def resolve_ffmpeg(args: argparse.Namespace) -> None:
    if ffmpeg_supports_encoder(args.ffmpeg, "libx264"):
        return
    fallback = Path("/usr/bin/ffmpeg")
    if fallback != args.ffmpeg and fallback.exists() and ffmpeg_supports_encoder(fallback, "libx264"):
        print(
            f"[setup] {args.ffmpeg} does not provide libx264; "
            f"using fallback {fallback}",
            flush=True,
        )
        args.ffmpeg = fallback
        return
    raise RuntimeError(
        f"{args.ffmpeg} does not provide libx264 and no usable fallback was found"
    )


def run_gear_conversion(output_root: Path, args: argparse.Namespace) -> None:
    script_path = Path(__file__).resolve().parent / "convert_lerobot_to_gear.py"
    command = [
        sys.executable,
        str(script_path),
        "--dataset-path",
        str(output_root),
        "--embodiment-tag",
        "aloha_x5lite_bimanual",
        "--state-keys",
        '{"left_joint_pos":[0,6],"left_gripper_pos":[6,7],"right_joint_pos":[7,13],"right_gripper_pos":[13,14]}',
        "--action-keys",
        '{"left_joint_pos":[0,6],"left_gripper_pos":[6,7],"right_joint_pos":[7,13],"right_gripper_pos":[13,14]}',
        "--relative-action-keys",
        "left_joint_pos",
        "left_gripper_pos",
        "right_joint_pos",
        "right_gripper_pos",
        "--task-key",
        "task_index",
        "--task-alias",
        "task",
        "--fps",
        str(float(args.fps)),
        "--action-horizon",
        "24",
        "--relative-quantile-samples",
        str(args.relative_quantile_samples),
        "--force",
    ]
    subprocess.run(command, check=True)


def prepare_output_root(path: Path, args: argparse.Namespace) -> None:
    if path.exists() and args.force:
        shutil.rmtree(path)
    if path.exists() and not args.resume and not args.force:
        raise FileExistsError(f"{path} already exists. Use --force or --resume.")
    (path / "data").mkdir(parents=True, exist_ok=True)
    (path / "videos").mkdir(parents=True, exist_ok=True)
    (path / "meta").mkdir(parents=True, exist_ok=True)


def build_task_index(plan: list[EpisodeRef]) -> tuple[list[dict], dict[str, int]]:
    task_text_to_index: dict[str, int] = {}
    for ref in plan:
        for task_text in ref.source.task_map.values():
            if task_text not in task_text_to_index:
                task_text_to_index[task_text] = len(task_text_to_index)
        episode = ref.source.episode_map[ref.source_episode_index]
        tasks = episode.get("tasks") or [""]
        for task in tasks:
            text = str(task)
            if text not in task_text_to_index:
                task_text_to_index[text] = len(task_text_to_index)
    tasks_jsonl = [
        {"task_index": index, "task": text}
        for text, index in sorted(task_text_to_index.items(), key=lambda item: item[1])
    ]
    return tasks_jsonl, task_text_to_index


def build_split(
    split_name: str,
    sources: list[SourceDataset],
    output_root: Path,
    args: argparse.Namespace,
) -> dict:
    print(f"[split:{split_name}] sources={len(sources)} output={output_root}")
    prepare_output_root(output_root, args)
    plan = build_episode_plan(sources, args.shuffle_seed, args.limit)
    tasks_jsonl, task_text_to_index = build_task_index(plan)
    task_index_to_text = {index: text for text, index in task_text_to_index.items()}

    if not sources:
        raise ValueError(f"No source datasets for split {split_name}")
    template_info = sources[0].info
    total_frames = 0
    episodes_jsonl: list[dict] = []
    episode_stats_jsonl: list[dict] = []
    video_tasks: list[VideoTask] = []
    global_start_index = 0
    for output_episode_index, ref in enumerate(
        tqdm(plan, desc=f"{split_name}:parquet", unit="episode")
    ):
        episode_record, stats_record, new_video_tasks, length = rewrite_episode(
            ref=ref,
            output_root=output_root,
            output_episode_index=output_episode_index,
            global_start_index=global_start_index,
            task_text_to_index=task_text_to_index,
            task_index_to_text=task_index_to_text,
            target_fps=args.fps,
            resample_timestamps=args.resample_timestamps,
        )
        episodes_jsonl.append(episode_record)
        episode_stats_jsonl.append(stats_record)
        video_tasks.extend(new_video_tasks)
        global_start_index += length
        total_frames += length

    info = {
        "codebase_version": "v2.1",
        "robot_type": "aloha",
        "total_episodes": len(episodes_jsonl),
        "total_frames": total_frames,
        "total_tasks": len(tasks_jsonl),
        "total_videos": len(video_tasks),
        "total_chunks": (len(episodes_jsonl) + CHUNKS_SIZE - 1) // CHUNKS_SIZE,
        "chunks_size": CHUNKS_SIZE,
        "fps": args.fps,
        "splits": {"train": f"0:{len(episodes_jsonl)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": build_features(template_info, args.height, args.width, args.fps),
    }
    write_json(output_root / "meta" / "info.json", info)
    write_jsonl(output_root / "meta" / "tasks.jsonl", tasks_jsonl)
    write_jsonl(output_root / "meta" / "episodes.jsonl", episodes_jsonl)
    write_jsonl(output_root / "meta" / "episodes_stats.jsonl", episode_stats_jsonl)
    smoke_filter = [episode["episode_index"] for episode in episodes_jsonl[: min(2, len(episodes_jsonl))]]
    write_json(output_root / "meta" / "smoke_episode_filter.json", smoke_filter)

    print(f"[split:{split_name}] videos={len(video_tasks)} workers={args.workers}")
    convert_videos(video_tasks, args, output_root)

    if not args.skip_convert:
        run_gear_conversion(output_root, args)

    summary = {
        "split": split_name,
        "source_root": str(args.source_root.resolve()),
        "output_root": str(output_root.resolve()),
        "source_roots": len(sources),
        "episodes": len(episodes_jsonl),
        "frames": total_frames,
        "videos": len(video_tasks),
        "tasks": len(tasks_jsonl),
        "height": args.height,
        "width": args.width,
        "fps": args.fps,
        "post_prefixes": list(POST_PREFIXES),
        "limit": args.limit,
    }
    write_json(output_root / "meta" / "build_summary.json", summary)
    print(f"[split:{split_name}] done: {summary}")
    return summary


def main() -> int:
    args = parse_args()
    if not args.ffmpeg.exists():
        raise FileNotFoundError(args.ffmpeg)
    if not args.ffprobe.exists():
        raise FileNotFoundError(args.ffprobe)
    resolve_ffmpeg(args)
    pretrain_sources, post_sources = discover_sources(args.source_root)
    print(f"[discover] pretrain_sources={len(pretrain_sources)} post_sources={len(post_sources)}")

    summaries = []
    if args.split in {"both", "pretrain"}:
        summaries.append(build_split("pretrain", pretrain_sources, args.pretrain_output, args))
    if args.split in {"both", "post"}:
        summaries.append(build_split("post", post_sources, args.post_output, args))

    print("[done] summaries:")
    for summary in summaries:
        print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
