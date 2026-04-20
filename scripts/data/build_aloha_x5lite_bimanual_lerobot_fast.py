#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

LEROBOT_ROOT = Path("/data/openpi/third_party/lerobot")
if str(LEROBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(LEROBOT_ROOT))

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


EpisodeRef = tuple[SourceDataset, int]


@dataclass(frozen=True)
class StreamVideoStatsSource:
    source_video_path: Path
    sampled_source_frame_indices: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a merged ALOHA X5lite bimanual LeRobot dataset at 30 fps for DreamZero with fast streaming video stats.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/data/datasets/dreamzero/1k_demo_lerobot"),
        help="Path to a single LeRobot dataset root, or a directory containing lerobot_data/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/datasets/dreamzero/aloha_x5lite_bimanual_lerobot_30fps_shuffle"),
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
        help="Deprecated and ignored. The script now treats input-root as a single merged LeRobot dataset.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_ID,
        help="Synthetic repo_id recorded in the output dataset metadata.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=42,
        help="Random seed for the global episode-level shuffle (default: 42).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete output-root if it already exists.",
    )
    return parser.parse_args()


def resolve_dataset_root(input_root: Path) -> Path:
    candidates = (
        input_root,
        input_root / "lerobot_data",
    )
    for candidate in candidates:
        if (candidate / "meta" / "info.json").is_file():
            return candidate
    raise FileNotFoundError(
        f"{input_root} is not a LeRobot dataset root. "
        "Expected meta/info.json under input-root or input-root/lerobot_data."
    )


def discover_task_dirs(input_root: Path, requested: list[str] | None) -> list[Path]:
    if requested:
        logging.getLogger("build_aloha_x5lite_bimanual_lerobot_fast").warning(
            "--task-dirs is ignored. input-root is treated as a single merged LeRobot dataset root."
        )
    resolve_dataset_root(input_root)
    return [input_root]


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
        root = resolve_dataset_root(task_dir)
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
    import imageio_ffmpeg
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset_module

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
    import pandas as pd

    return pd.read_parquet(parquet_path)


def compute_resample_indices(source_timestamps: np.ndarray, target_fps: int) -> tuple[np.ndarray, np.ndarray]:
    import numpy as np

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
    import numpy as np

    return np.stack(df[key].to_numpy())


def clip_grippers(values: np.ndarray, indices: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int]]:
    import numpy as np

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
    import numpy as np

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


def build_streaming_video_stats_sources(
    source: SourceDataset,
    source_episode_index: int,
    selected_source_frame_indices: np.ndarray,
    target_length: int,
) -> dict[str, StreamVideoStatsSource]:
    import numpy as np
    from lerobot.common.datasets.compute_stats import sample_indices

    sample_positions = sample_indices(target_length)
    sampled_source_frame_indices = selected_source_frame_indices[np.array(sample_positions, dtype=np.int64)]

    return {
        video_key: StreamVideoStatsSource(
            source_video_path=source.video_path(source_episode_index, video_key),
            sampled_source_frame_indices=tuple(int(idx) for idx in sampled_source_frame_indices.tolist()),
        )
        for video_key in VIDEO_KEYS
    }


def compute_streaming_video_feature_stats(video_source: StreamVideoStatsSource) -> dict[str, np.ndarray]:
    import imageio_ffmpeg
    import numpy as np
    from lerobot.common.datasets.compute_stats import auto_downsample_height_width

    if not video_source.sampled_source_frame_indices:
        raise ValueError(f"No sampled frame indices provided for {video_source.source_video_path}")

    frame_multiplicity = Counter(video_source.sampled_source_frame_indices)
    unique_frame_indices = sorted(frame_multiplicity)
    select_expr = "+".join(f"eq(n\\,{frame_index})" for frame_index in unique_frame_indices)
    frame_generator = imageio_ffmpeg.read_frames(
        video_source.source_video_path,
        pix_fmt="rgb24",
        output_params=["-vf", f"select={select_expr}", "-vsync", "0"],
    )

    channel_min = None
    channel_max = None
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sum_sq = np.zeros(3, dtype=np.float64)
    total_pixels = 0
    decoded_frames = 0

    try:
        metadata = next(frame_generator)
        width, height = metadata["size"]
        expected_frame_bytes = width * height * 3

        for frame_index in unique_frame_indices:
            frame_bytes = next(frame_generator, None)
            if frame_bytes is None:
                raise RuntimeError(
                    f"Decoded {decoded_frames} sampled frames from {video_source.source_video_path}, "
                    f"expected {len(unique_frame_indices)}."
                )
            if len(frame_bytes) != expected_frame_bytes:
                raise RuntimeError(
                    f"Unexpected frame byte size {len(frame_bytes)} for {video_source.source_video_path}, "
                    f"expected {expected_frame_bytes}."
                )

            frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(height, width, 3)
            frame = auto_downsample_height_width(frame.transpose(2, 0, 1))
            frame_float = frame.astype(np.float64, copy=False)
            multiplicity = frame_multiplicity[frame_index]

            frame_min = frame_float.min(axis=(1, 2))
            frame_max = frame_float.max(axis=(1, 2))
            if channel_min is None:
                channel_min = frame_min
                channel_max = frame_max
            else:
                channel_min = np.minimum(channel_min, frame_min)
                channel_max = np.maximum(channel_max, frame_max)

            channel_sum += frame_float.sum(axis=(1, 2)) * multiplicity
            channel_sum_sq += np.square(frame_float).sum(axis=(1, 2)) * multiplicity
            total_pixels += frame.shape[1] * frame.shape[2] * multiplicity
            decoded_frames += 1

        extra_frame = next(frame_generator, None)
        if extra_frame is not None:
            raise RuntimeError(
                f"Decoded more sampled frames than expected from {video_source.source_video_path}."
            )
    finally:
        frame_generator.close()

    if channel_min is None or channel_max is None or total_pixels == 0:
        raise RuntimeError(f"Failed to compute video stats for {video_source.source_video_path}")

    mean = channel_sum / total_pixels
    variance = np.maximum(channel_sum_sq / total_pixels - np.square(mean), 0.0)
    std = np.sqrt(variance)

    return {
        "min": (channel_min / 255.0).reshape(3, 1, 1),
        "max": (channel_max / 255.0).reshape(3, 1, 1),
        "mean": (mean / 255.0).reshape(3, 1, 1),
        "std": (std / 255.0).reshape(3, 1, 1),
        "count": np.array([len(video_source.sampled_source_frame_indices)], dtype=np.int64),
    }


def patch_compute_episode_stats() -> None:
    import numpy as np
    import lerobot.common.datasets.compute_stats as compute_stats_module
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset_module

    get_feature_stats = compute_stats_module.get_feature_stats
    sample_images = compute_stats_module.sample_images

    def _compute_episode_stats(
        episode_data: dict[str, list[str] | np.ndarray | StreamVideoStatsSource],
        features: dict,
    ) -> dict:
        ep_stats = {}
        for key, data in episode_data.items():
            if features[key]["dtype"] == "string":
                continue
            if features[key]["dtype"] in ["image", "video"]:
                if isinstance(data, StreamVideoStatsSource):
                    ep_stats[key] = compute_streaming_video_feature_stats(data)
                    continue

                ep_ft_array = sample_images(data)
                axes_to_reduce = (0, 2, 3)
                keepdims = True
            else:
                ep_ft_array = data
                axes_to_reduce = 0
                keepdims = data.ndim == 1

            ep_stats[key] = get_feature_stats(ep_ft_array, axis=axes_to_reduce, keepdims=keepdims)

            if features[key]["dtype"] in ["image", "video"]:
                ep_stats[key] = {
                    stat_key: stat_value if stat_key == "count" else np.squeeze(stat_value / 255.0, axis=0)
                    for stat_key, stat_value in ep_stats[key].items()
                }

        return ep_stats

    compute_stats_module.compute_episode_stats = _compute_episode_stats
    lerobot_dataset_module.compute_episode_stats = _compute_episode_stats


def build_episode_plan(sources: list[SourceDataset]) -> list[EpisodeRef]:
    episode_plan: list[EpisodeRef] = []
    for source in sources:
        source_fps = int(source.info["fps"])
        if source_fps <= 0:
            raise ValueError(f"Expected a positive fps, got {source_fps} in {source.root}")
        episode_plan.extend((source, source_episode_index) for source_episode_index in range(source.total_episodes))
    return episode_plan


def shuffle_episode_plan(episode_plan: list[EpisodeRef], shuffle_seed: int) -> list[EpisodeRef]:
    import numpy as np

    permutation = np.random.default_rng(shuffle_seed).permutation(len(episode_plan))
    return [episode_plan[idx] for idx in permutation.tolist()]


def format_episode_plan_preview(episode_plan: list[EpisodeRef], limit: int = 10) -> str:
    if not episode_plan:
        return "<empty>"

    preview = [
        f"{source.task_dir.name}/{source_episode_index:06d}"
        for source, source_episode_index in episode_plan[:limit]
    ]
    suffix = "" if len(episode_plan) <= limit else f" ... (+{len(episode_plan) - limit} more)"
    return ", ".join(preview) + suffix


def build_dataset(args: argparse.Namespace) -> None:
    import imageio_ffmpeg
    import numpy as np
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from tqdm import tqdm

    logger = logging.getLogger("build_aloha_x5lite_bimanual_lerobot_fast")
    task_dirs = discover_task_dirs(args.input_root, args.task_dirs)
    sources = build_source_datasets(task_dirs)
    episode_plan = build_episode_plan(sources)
    shuffled_plan = shuffle_episode_plan(episode_plan, args.shuffle_seed)

    source = sources[0]
    total_episodes = len(episode_plan)
    logger.info(
        "Building dataset from %s | source_fps=%s target_fps=%s episodes=%d",
        source.root,
        source.info["fps"],
        args.target_fps,
        total_episodes,
    )
    logger.info(
        "Global episode shuffle | seed=%d total_episodes=%d preview=%s",
        args.shuffle_seed,
        total_episodes,
        format_episode_plan_preview(shuffled_plan),
    )

    if args.output_root.exists():
        if not args.force:
            raise FileExistsError(f"{args.output_root} already exists. Use --force to overwrite.")
        shutil.rmtree(args.output_root)

    patch_video_encoder(args.video_codec)
    patch_compute_episode_stats()
    features = build_features(source.info)
    robot_type = source.info["robot_type"]

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.target_fps,
        root=args.output_root,
        robot_type=robot_type,
        features=features,
        use_videos=True,
        tolerance_s=1e-4,
    )

    total_state_clip = np.zeros(2, dtype=np.int64)
    total_action_clip = np.zeros(2, dtype=np.int64)

    for merged_episode_index, (source, source_episode_index) in enumerate(
        tqdm(shuffled_plan, desc="merge:shuffled", unit="episode")
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
        selected_source_frame_indices = df["frame_index"].to_numpy(dtype=np.int64, copy=True)[selected_indices]

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
            build_streaming_video_stats_sources(
                source=source,
                source_episode_index=source_episode_index,
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
