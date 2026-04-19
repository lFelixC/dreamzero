#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
import pandas as pd
from tqdm import tqdm

LEROBOT_ROOT = Path("/data/openpi/third_party/lerobot")
if str(LEROBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(LEROBOT_ROOT))

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset_module
from lerobot.common.datasets.compute_stats import sample_indices
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

SOURCE_FPS = 50
STATE_GRIPPER_INDICES = (6, 13)
ACTION_GRIPPER_INDICES = (6, 13)
VIDEO_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left",
    "observation.images.cam_right",
)
DEFAULT_REPO_ID = "local/aloha_x5lite_bimanual_lerobot_30fps"


@dataclass
class SourceDataset:
    task_dir: Path
    root: Path
    info: dict
    task_map: dict[int, str]

    @property
    def total_episodes(self) -> int:
        return int(self.info["total_episodes"])

    def parquet_path(self, episode_index: int) -> Path:
        pattern = self.info["data_path"]
        chunk = episode_index // int(self.info.get("chunks_size", 1000))
        return self.root / pattern.format(episode_chunk=chunk, episode_index=episode_index)

    def video_path(self, episode_index: int, video_key: str) -> Path:
        pattern = self.info["video_path"]
        chunk = episode_index // int(self.info.get("chunks_size", 1000))
        return self.root / pattern.format(
            episode_chunk=chunk,
            episode_index=episode_index,
            video_key=video_key,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a merged ALOHA X5lite bimanual LeRobot dataset at 30 fps for DreamZero.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/data/datasets/dreamzero/1k_demo_lerobot"),
        help="Directory containing the 15 first-level ALOHA task folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/datasets/dreamzero/aloha_x5lite_bimanual_lerobot_30fps"),
        help="Path for the merged output LeRobot root.",
    )
    parser.add_argument(
        "--target-fps",
        type=int,
        default=30,
        help="Target fps for the merged dataset.",
    )
    parser.add_argument(
        "--video-codec",
        type=str,
        default="h264",
        choices=["h264", "hevc", "libsvtav1"],
        help="Codec used to encode the merged videos.",
    )
    parser.add_argument(
        "--task-dirs",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of task directory names under input-root. Defaults to all first-level dirs.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_ID,
        help="Synthetic repo_id recorded in the output dataset metadata.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete output-root if it already exists.",
    )
    return parser.parse_args()


def discover_task_dirs(input_root: Path, requested: list[str] | None) -> list[Path]:
    if requested:
        task_dirs = [input_root / name for name in requested]
    else:
        task_dirs = sorted(path for path in input_root.iterdir() if path.is_dir())

    valid = []
    for task_dir in task_dirs:
        dataset_root = task_dir / "lerobot_data"
        if (dataset_root / "meta" / "info.json").is_file():
            valid.append(task_dir)
    if not valid:
        raise FileNotFoundError(f"No LeRobot task directories found under {input_root}")
    return valid


def load_task_map(dataset_root: Path) -> dict[int, str]:
    task_map: dict[int, str] = {}
    tasks_path = dataset_root / "meta" / "tasks.jsonl"
    with tasks_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            task_map[int(record["task_index"])] = record["task"]
    return task_map


def build_source_datasets(task_dirs: list[Path]) -> list[SourceDataset]:
    datasets: list[SourceDataset] = []
    for task_dir in task_dirs:
        root = task_dir / "lerobot_data"
        with (root / "meta" / "info.json").open("r", encoding="utf-8") as handle:
            info = json.load(handle)
        datasets.append(
            SourceDataset(
                task_dir=task_dir,
                root=root,
                info=info,
                task_map=load_task_map(root),
            )
        )
    return datasets


def build_features(info: dict) -> dict:
    feature_order = (
        "observation.state",
        "action",
        "observation.velocity",
        "observation.effort",
        "observation.images.cam_high",
        "observation.images.cam_left",
        "observation.images.cam_right",
    )
    features = {}
    for key in feature_order:
        feature = info["features"][key]
        features[key] = {
            "dtype": feature["dtype"],
            "shape": tuple(feature["shape"]),
            "names": feature.get("names"),
        }
    return features


def patch_video_encoder(video_codec: str) -> None:
    ffmpeg_executable = imageio_ffmpeg.get_ffmpeg_exe()

    def _encode_with_codec(imgs_dir, video_path, fps, overwrite=False):
        codec_map = {
            "h264": "libx264",
            "hevc": "libx265",
            "libsvtav1": "libsvtav1",
        }
        if video_codec not in codec_map:
            raise ValueError(f"Unsupported video codec: {video_codec}")

        imgs_dir = Path(imgs_dir)
        video_path = Path(video_path)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            ffmpeg_executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y" if overwrite else "-n",
            "-framerate",
            str(fps),
            "-i",
            str(imgs_dir / "frame_%06d.png"),
            "-c:v",
            codec_map[video_codec],
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ]
        subprocess.run(command, check=True)

    lerobot_dataset_module.encode_video_frames = _encode_with_codec


def read_episode_dataframe(parquet_path: Path) -> pd.DataFrame:
    return pd.read_parquet(parquet_path)


def compute_resample_indices(source_timestamps: np.ndarray, target_fps: int) -> tuple[np.ndarray, np.ndarray]:
    if source_timestamps.ndim != 1:
        raise ValueError("Expected source timestamps to be a 1D array.")

    relative_ts = source_timestamps - source_timestamps[0]
    if np.any(np.diff(relative_ts) < -1e-6):
        raise ValueError("Source timestamps must be non-decreasing within an episode.")

    last_ts = float(relative_ts[-1]) if len(relative_ts) else 0.0
    target_length = max(1, int(np.floor(last_ts * target_fps + 1e-6)) + 1)
    target_timestamps = np.arange(target_length, dtype=np.float64) / float(target_fps)

    distances = np.abs(relative_ts[:, None] - target_timestamps[None, :])
    selected_indices = distances.argmin(axis=0)
    return selected_indices.astype(np.int64), target_timestamps.astype(np.float32)


def stack_column(df: pd.DataFrame, key: str) -> np.ndarray:
    return np.stack(df[key].to_numpy())


def clip_grippers(values: np.ndarray, indices: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int]]:
    clipped = values.copy()
    clip_counts: list[int] = []
    for idx in indices:
        original = clipped[:, idx].copy()
        clipped[:, idx] = np.clip(clipped[:, idx], 0.0, 1.0)
        clip_counts.append(int(np.count_nonzero(np.abs(original - clipped[:, idx]) > 1e-8)))
    return clipped, (clip_counts[0], clip_counts[1])


def get_task_strings(source: SourceDataset, task_indices: np.ndarray) -> list[str]:
    task_strings = []
    for task_idx in task_indices.tolist():
        if task_idx not in source.task_map:
            raise KeyError(
                f"task_index={task_idx} missing from {source.root / 'meta' / 'tasks.jsonl'}"
            )
        task_strings.append(source.task_map[task_idx])
    return task_strings


def log_episode_summary(
    logger: logging.Logger,
    merged_episode_index: int,
    source: SourceDataset,
    source_episode_index: int,
    source_length: int,
    target_length: int,
    timestamps: np.ndarray,
    state_clip_counts: tuple[int, int],
    action_clip_counts: tuple[int, int],
) -> None:
    step = float(timestamps[1] - timestamps[0]) if len(timestamps) > 1 else 0.0
    logger.info(
        "episode %03d <- %s/%06d | src=%d tgt=%d step=%.6f frame_index_ok=true "
        "state_clip[r=%d,l=%d] action_clip[r=%d,l=%d]",
        merged_episode_index,
        source.task_dir.name,
        source_episode_index,
        source_length,
        target_length,
        step,
        state_clip_counts[0],
        state_clip_counts[1],
        action_clip_counts[0],
        action_clip_counts[1],
    )


def transcode_episode_video(
    ffmpeg_executable: str,
    source_video_path: Path,
    output_video_path: Path,
    target_fps: int,
    video_codec: str,
) -> None:
    codec_map = {
        "h264": "libx264",
        "hevc": "libx265",
        "libsvtav1": "libsvtav1",
    }
    if video_codec not in codec_map:
        raise ValueError(f"Unsupported video codec: {video_codec}")

    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_video_path),
        "-vf",
        f"fps={target_fps}:round=near",
        "-an",
        "-c:v",
        codec_map[video_codec],
        "-pix_fmt",
        "yuv420p",
        str(output_video_path),
    ]
    subprocess.run(command, check=True)


def build_episode_buffer(
    dataset: LeRobotDataset,
    target_timestamps: np.ndarray,
    task_strings: list[str],
    state: np.ndarray,
    action: np.ndarray,
    velocity: np.ndarray,
    effort: np.ndarray,
) -> dict:
    episode_buffer = dataset.create_episode_buffer()
    episode_buffer["size"] = len(target_timestamps)
    episode_buffer["task"] = task_strings
    episode_buffer["timestamp"] = [np.float32(ts) for ts in target_timestamps.tolist()]
    episode_buffer["frame_index"] = [np.int64(i) for i in range(len(target_timestamps))]
    episode_buffer["observation.state"] = list(state)
    episode_buffer["action"] = list(action)
    episode_buffer["observation.velocity"] = list(velocity)
    episode_buffer["observation.effort"] = list(effort)
    return episode_buffer


def prepare_video_stats_images(
    dataset: LeRobotDataset,
    source: SourceDataset,
    source_episode_index: int,
    target_episode_index: int,
    selected_source_frame_indices: np.ndarray,
    target_length: int,
) -> dict[str, list[str]]:
    sample_positions = sample_indices(target_length)
    image_entries: dict[str, list[str]] = {}

    for video_key in VIDEO_KEYS:
        reader = imageio.get_reader(str(source.video_path(source_episode_index, video_key)))
        sample_paths: dict[int, str] = {}
        try:
            for sample_position in sample_positions:
                output_path = dataset._get_image_file_path(
                    episode_index=target_episode_index,
                    image_key=video_key,
                    frame_index=sample_position,
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                frame = reader.get_data(int(selected_source_frame_indices[sample_position]))
                imageio.imwrite(output_path, frame)
                sample_paths[sample_position] = str(output_path)
        finally:
            reader.close()

        nearest_sample_paths: list[str] = []
        sample_positions_arr = np.array(sample_positions, dtype=np.int64)
        for frame_index in range(target_length):
            insert_pos = int(np.searchsorted(sample_positions_arr, frame_index))
            if insert_pos <= 0:
                chosen = int(sample_positions_arr[0])
            elif insert_pos >= len(sample_positions_arr):
                chosen = int(sample_positions_arr[-1])
            else:
                left = int(sample_positions_arr[insert_pos - 1])
                right = int(sample_positions_arr[insert_pos])
                chosen = left if abs(frame_index - left) <= abs(right - frame_index) else right
            nearest_sample_paths.append(sample_paths[chosen])
        image_entries[video_key] = nearest_sample_paths

    return image_entries


def build_dataset(args: argparse.Namespace) -> None:
    logger = logging.getLogger("build_aloha_x5lite_bimanual_lerobot")
    task_dirs = discover_task_dirs(args.input_root, args.task_dirs)
    sources = build_source_datasets(task_dirs)

    total_episodes = sum(source.total_episodes for source in sources)
    logger.info("Building merged dataset from %d task folders and %d episodes.", len(sources), total_episodes)

    if args.output_root.exists():
        if not args.force:
            raise FileExistsError(f"{args.output_root} already exists. Use --force to overwrite.")
        shutil.rmtree(args.output_root)

    patch_video_encoder(args.video_codec)
    features = build_features(sources[0].info)
    robot_type = sources[0].info["robot_type"]

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.target_fps,
        root=args.output_root,
        robot_type=robot_type,
        features=features,
        use_videos=True,
        tolerance_s=1e-4,
    )

    merged_episode_index = 0
    total_state_clip = np.zeros(2, dtype=np.int64)
    total_action_clip = np.zeros(2, dtype=np.int64)

    for source in sources:
        source_fps = int(source.info["fps"])
        if source_fps != SOURCE_FPS:
            raise ValueError(f"Expected {SOURCE_FPS} fps, got {source_fps} in {source.root}")

        for source_episode_index in tqdm(
            range(source.total_episodes),
            desc=f"merge:{source.task_dir.name}",
            unit="episode",
        ):
            parquet_path = source.parquet_path(source_episode_index)
            df = read_episode_dataframe(parquet_path)
            source_timestamps = df["timestamp"].to_numpy(dtype=np.float64, copy=True)
            selected_indices, target_timestamps = compute_resample_indices(source_timestamps, args.target_fps)

            state = stack_column(df, "observation.state")[selected_indices].astype(np.float32, copy=False)
            action = stack_column(df, "action")[selected_indices].astype(np.float32, copy=False)
            velocity = stack_column(df, "observation.velocity")[selected_indices].astype(np.float32, copy=False)
            effort = stack_column(df, "observation.effort")[selected_indices].astype(np.float32, copy=False)
            task_indices = df["task_index"].to_numpy(dtype=np.int64, copy=True)[selected_indices]

            state, state_clip_counts = clip_grippers(state, STATE_GRIPPER_INDICES)
            action, action_clip_counts = clip_grippers(action, ACTION_GRIPPER_INDICES)
            total_state_clip += np.array(state_clip_counts, dtype=np.int64)
            total_action_clip += np.array(action_clip_counts, dtype=np.int64)

            task_strings = get_task_strings(source, task_indices)
            episode_buffer = build_episode_buffer(
                dataset=dataset,
                target_timestamps=target_timestamps,
                task_strings=task_strings,
                state=state,
                action=action,
                velocity=velocity,
                effort=effort,
            )
            selected_source_frame_indices = (
                df["frame_index"].to_numpy(dtype=np.int64, copy=True)[selected_indices]
            )

            for video_key in VIDEO_KEYS:
                source_video_path = source.video_path(source_episode_index, video_key)
                output_video_path = dataset.root / dataset.meta.get_video_file_path(
                    merged_episode_index,
                    video_key,
                )
                transcode_episode_video(
                    ffmpeg_executable=imageio_ffmpeg.get_ffmpeg_exe(),
                    source_video_path=source_video_path,
                    output_video_path=output_video_path,
                    target_fps=args.target_fps,
                    video_codec=args.video_codec,
                )

            episode_buffer.update(
                prepare_video_stats_images(
                    dataset=dataset,
                    source=source,
                    source_episode_index=source_episode_index,
                    target_episode_index=merged_episode_index,
                    selected_source_frame_indices=selected_source_frame_indices,
                    target_length=len(target_timestamps),
                )
            )
            dataset.episode_buffer = episode_buffer
            dataset.save_episode()
            log_episode_summary(
                logger=logger,
                merged_episode_index=merged_episode_index,
                source=source,
                source_episode_index=source_episode_index,
                source_length=len(df),
                target_length=len(target_timestamps),
                timestamps=target_timestamps,
                state_clip_counts=state_clip_counts,
                action_clip_counts=action_clip_counts,
            )
            merged_episode_index += 1

    info_path = args.output_root / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as handle:
        info = json.load(handle)

    logger.info(
        "Completed merged dataset at %s | fps=%s episodes=%s tasks=%s total_frames=%s",
        args.output_root,
        info["fps"],
        info["total_episodes"],
        info["total_tasks"],
        info["total_frames"],
    )
    logger.info(
        "Total gripper clips | state[right,left]=[%d,%d] action[right,left]=[%d,%d]",
        total_state_clip[0],
        total_state_clip[1],
        total_action_clip[0],
        total_action_clip[1],
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    build_dataset(args)


if __name__ == "__main__":
    main()
