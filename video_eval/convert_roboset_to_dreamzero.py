#!/usr/bin/env python3
"""Extract RoboSet archives and convert sampled trials into mini DreamZero datasets."""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from common import (
    ACTION_HORIZON,
    DEFAULT_DATASET_PROMPTS,
    MANIFEST_VIDEO_KEYS,
    append_run_log,
    build_chunk_schedule,
    describe_tree,
    ensure_dir,
    find_existing_dataset_root,
    resize_frames,
    save_jsonl,
    save_video,
    strip_archive_suffix,
    write_json,
)


LOGGER = logging.getLogger(__name__)

SUPPORTED_ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz")
TARGET_HEIGHT = 180
TARGET_WIDTH = 320
TARGET_FPS = 15

VIDEO_MODALITY_JSON = {
    "state": {
        "cartesian_position": {"start": 0, "end": 6},
        "gripper_position": {"start": 6, "end": 7},
        "joint_position": {"start": 7, "end": 14},
    },
    "action": {
        "cartesian_position": {"start": 0, "end": 6},
        "cartesian_velocity": {"start": 6, "end": 12},
        "gripper_position": {"start": 12, "end": 13},
        "gripper_velocity": {"start": 13, "end": 14},
        "joint_position": {"start": 14, "end": 21},
        "joint_velocity": {"start": 21, "end": 28},
    },
    "video": {
        "exterior_image_1_left": {"original_key": "observation.images.exterior_image_1_left"},
        "exterior_image_2_left": {"original_key": "observation.images.exterior_image_2_left"},
        "wrist_image_left": {"original_key": "observation.images.wrist_image_left"},
    },
    "annotation": {
        "language.language_instruction": {},
        "language.language_instruction_2": {},
        "language.language_instruction_3": {},
    },
}

CAMERA_MAPPING = {
    "rgb_left": "observation.images.exterior_image_1_left",
    "rgb_right": "observation.images.exterior_image_2_left",
    "rgb_wrist": "observation.images.wrist_image_left",
}
UNUSED_VIEWS = ["rgb_top"]
REQUIRED_DATA_KEYS = ("rgb_left", "rgb_right", "rgb_wrist", "qp_arm", "qp_ee", "ctrl_arm", "ctrl_ee")


@dataclass(frozen=True)
class TrialReference:
    dataset_name: str
    archive_path: Path
    extracted_h5_path: Path
    h5_stem: str
    trial_name: str
    frame_count: int
    solved: bool | None

    @property
    def sample_id(self) -> str:
        return f"{self.dataset_name}__{self.h5_stem}__{self.trial_name}"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="RoboSet archive root. Defaults to /data/datasets/robotset or /data/datasets/roboset.",
    )
    parser.add_argument("--sample-count", type=int, default=50, help="Trials sampled per dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for sampling.")
    parser.add_argument("--height", type=int, default=TARGET_HEIGHT, help="Output per-view frame height.")
    parser.add_argument("--width", type=int, default=TARGET_WIDTH, help="Output per-view frame width.")
    parser.add_argument("--fps", type=int, default=TARGET_FPS, help="FPS used for converted videos.")
    parser.add_argument("--force", action="store_true", help="Re-extract and overwrite generated outputs.")
    return parser


def is_supported_archive(path: Path) -> bool:
    return any(path.name.endswith(suffix) for suffix in SUPPORTED_ARCHIVE_SUFFIXES)


def dataset_prompts(dataset_name: str) -> list[str]:
    prompts = DEFAULT_DATASET_PROMPTS.get(dataset_name)
    if prompts is not None:
        return prompts
    readable = dataset_name.replace("_", " ")
    return [
        f"Perform the task: {readable}.",
        f"Complete the roboset task: {readable}.",
        f"Execute the scene instruction: {readable}.",
    ]


def archive_relative_path(member_name: str, dataset_name: str) -> Path:
    parts = [part for part in Path(member_name).parts if part not in ("", ".")]
    if dataset_name in parts:
        index = parts.index(dataset_name)
        trimmed = parts[index + 1 :]
        return Path(*trimmed) if trimmed else Path(".")
    return Path(Path(member_name).name)


def extract_tar_archive(archive_path: Path, output_dir: Path) -> None:
    with tarfile.open(archive_path, "r:*") as archive:
        for member in archive.getmembers():
            rel_path = archive_relative_path(member.name, strip_archive_suffix(archive_path))
            if rel_path == Path("."):
                continue
            destination = output_dir / rel_path
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                continue
            with destination.open("wb") as target:
                shutil.copyfileobj(source, target)


def extract_zip_archive(archive_path: Path, output_dir: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            rel_path = archive_relative_path(member.filename, strip_archive_suffix(archive_path))
            if rel_path == Path("."):
                continue
            destination = output_dir / rel_path
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)


def extract_archive(archive_path: Path, output_dir: Path, force: bool) -> Path:
    marker_path = output_dir / ".extract_complete.json"
    if force and output_dir.exists():
        shutil.rmtree(output_dir)
    if marker_path.exists():
        LOGGER.info("Skipping extraction for %s; marker exists.", archive_path.name)
        return output_dir

    ensure_dir(output_dir)
    LOGGER.info("Extracting %s -> %s", archive_path, output_dir)
    if archive_path.name.endswith(".zip"):
        extract_zip_archive(archive_path, output_dir)
    else:
        extract_tar_archive(archive_path, output_dir)
    write_json(
        marker_path,
        {
            "archive_path": str(archive_path),
            "output_dir": str(output_dir),
        },
    )
    return output_dir


def list_trial_references(dataset_name: str, archive_path: Path, extracted_dir: Path) -> list[TrialReference]:
    references: list[TrialReference] = []
    for h5_path in sorted(extracted_dir.rglob("*.h5")):
        with h5py.File(h5_path, "r") as h5:
            for trial_name in sorted(key for key in h5.keys() if key.startswith("Trial")):
                data_group = h5[trial_name].get("data")
                if data_group is None:
                    continue
                if any(key not in data_group for key in REQUIRED_DATA_KEYS):
                    continue
                frame_count = min(int(data_group[key].shape[0]) for key in ("rgb_left", "rgb_right", "rgb_wrist"))
                solved = None
                if "config" in h5[trial_name] and "solved" in h5[trial_name]["config"]:
                    solved = bool(float(np.asarray(h5[trial_name]["config"]["solved"]).reshape(-1)[0]))
                references.append(
                    TrialReference(
                        dataset_name=dataset_name,
                        archive_path=archive_path,
                        extracted_h5_path=h5_path,
                        h5_stem=h5_path.stem,
                        trial_name=trial_name,
                        frame_count=frame_count,
                        solved=solved,
                    )
                )
    return references


def select_trials(trials: list[TrialReference], sample_count: int, seed: int) -> list[TrialReference]:
    if not trials:
        return []
    ordered = sorted(trials, key=lambda ref: ref.sample_id)
    actual_count = min(sample_count, len(ordered))
    rng = random.Random(seed)
    selected = rng.sample(ordered, k=actual_count)
    return sorted(selected, key=lambda ref: ref.sample_id)


def build_state_vector(qp_arm: np.ndarray, qp_ee: np.ndarray) -> np.ndarray:
    zeros_cartesian = np.zeros((qp_arm.shape[0], 6), dtype=np.float64)
    gripper = qp_ee.reshape(-1, 1).astype(np.float64)
    joints = qp_arm.astype(np.float64)
    return np.concatenate([zeros_cartesian, gripper, joints], axis=1)


def build_action_vector(ctrl_arm: np.ndarray, ctrl_ee: np.ndarray) -> np.ndarray:
    zeros_cartesian_position = np.zeros((ctrl_arm.shape[0], 6), dtype=np.float64)
    zeros_cartesian_velocity = np.zeros((ctrl_arm.shape[0], 6), dtype=np.float64)
    gripper_position = ctrl_ee.reshape(-1, 1).astype(np.float64)
    zeros_gripper_velocity = np.zeros((ctrl_arm.shape[0], 1), dtype=np.float64)
    joints = ctrl_arm.astype(np.float64)
    zeros_joint_velocity = np.zeros((ctrl_arm.shape[0], 7), dtype=np.float64)
    return np.concatenate(
        [
            zeros_cartesian_position,
            zeros_cartesian_velocity,
            gripper_position,
            zeros_gripper_velocity,
            joints,
            zeros_joint_velocity,
        ],
        axis=1,
    )


def make_episode_table(
    episode_index: int,
    frame_count: int,
    prompts: list[str],
    state_matrix: np.ndarray,
    action_matrix: np.ndarray,
    fps: int,
) -> pa.Table:
    return pa.table(
        {
            "observation.state": pa.array(state_matrix.tolist(), type=pa.list_(pa.float64())),
            "action": pa.array(action_matrix.tolist(), type=pa.list_(pa.float64())),
            "next.reward": pa.array([0.0] * frame_count, type=pa.float64()),
            "next.done": pa.array([False] * (frame_count - 1) + [True], type=pa.bool_()),
            "is_terminal": pa.array([False] * (frame_count - 1) + [True], type=pa.bool_()),
            "is_first": pa.array([True] + [False] * (frame_count - 1), type=pa.bool_()),
            "discount": pa.array([1.0] * frame_count, type=pa.float64()),
            "timestamp": pa.array([index / fps for index in range(frame_count)], type=pa.float64()),
            "episode_index": pa.array([episode_index] * frame_count, type=pa.int64()),
            "frame_index": pa.array(list(range(frame_count)), type=pa.int64()),
            "annotation.language.language_instruction": pa.array([1] * frame_count, type=pa.int64()),
            "annotation.language.language_instruction_2": pa.array([2] * frame_count, type=pa.int64()),
            "annotation.language.language_instruction_3": pa.array([3] * frame_count, type=pa.int64()),
            "task_index": pa.array([1] * frame_count, type=pa.int64()),
        }
    )


def save_episode_videos(
    episode_index: int,
    output_dir: Path,
    fps: int,
    left: np.ndarray,
    right: np.ndarray,
    wrist: np.ndarray,
) -> dict[str, str]:
    relative_paths = {}
    video_root = ensure_dir(output_dir / "videos" / "chunk-000")
    for dataset_key, frames in (
        ("observation.images.exterior_image_1_left", left),
        ("observation.images.exterior_image_2_left", right),
        ("observation.images.wrist_image_left", wrist),
    ):
        episode_filename = f"episode_{episode_index:06d}.mp4"
        target_path = video_root / dataset_key / episode_filename
        save_video(frames, target_path, fps=fps)
        relative_paths[dataset_key] = str(target_path)
    return relative_paths


def write_tasks_file(meta_dir: Path, prompts: list[str]) -> None:
    rows = [{"task_index": 0, "task": "not provided"}]
    rows.extend({"task_index": index + 1, "task": prompt} for index, prompt in enumerate(prompts))
    save_jsonl(meta_dir / "tasks.jsonl", rows)


def write_info_file(dataset_dir: Path, total_episodes: int, total_frames: int, fps: int, height: int, width: int) -> None:
    info = {
        "codebase_version": "v2.0",
        "robot_type": "droid",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 4,
        "total_videos": 3,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": "0:100"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.exterior_image_1_left": {
                "dtype": "video",
                "shape": [height, width, 3],
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": fps,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "observation.images.exterior_image_2_left": {
                "dtype": "video",
                "shape": [height, width, 3],
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": fps,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "observation.images.wrist_image_left": {
                "dtype": "video",
                "shape": [height, width, 3],
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": fps,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "observation.state": {
                "dtype": "float64",
                "shape": [14],
                "names": ["cartesian_position", "gripper_position", "joint_position"],
            },
            "action": {
                "dtype": "float64",
                "shape": [28],
                "names": [
                    "cartesian_position",
                    "cartesian_velocity",
                    "gripper_position",
                    "gripper_velocity",
                    "joint_position",
                    "joint_velocity",
                ],
            },
            "timestamp": {"dtype": "float64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "next.reward": {"dtype": "float64", "shape": [1]},
            "next.done": {"dtype": "bool", "shape": [1]},
            "is_terminal": {"dtype": "bool", "shape": [1]},
            "is_first": {"dtype": "bool", "shape": [1]},
            "discount": {"dtype": "float64", "shape": [1]},
            "annotation.language.language_instruction": {"dtype": "int64", "shape": [1]},
            "annotation.language.language_instruction_2": {"dtype": "int64", "shape": [1]},
            "annotation.language.language_instruction_3": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        },
    }
    write_json(dataset_dir / "meta" / "info.json", info)


def convert_dataset(
    dataset_root: Path,
    archive_path: Path,
    sample_count: int,
    seed: int,
    height: int,
    width: int,
    fps: int,
    force: bool,
) -> dict[str, Any]:
    dataset_name = strip_archive_suffix(archive_path)
    extracted_dir = dataset_root / "extracted" / dataset_name
    output_dir = dataset_root / "dreamzero_converted" / dataset_name
    log_path = output_dir / "conversion_run.log"

    extract_archive(archive_path, extracted_dir, force=force)
    extracted_tree = describe_tree(extracted_dir, max_depth=2)
    for line in extracted_tree:
        LOGGER.info("%s", line)

    references = list_trial_references(dataset_name, archive_path, extracted_dir)
    selected = select_trials(references, sample_count, seed)
    prompts = dataset_prompts(dataset_name)

    if force and output_dir.exists():
        shutil.rmtree(output_dir)
    ensure_dir(output_dir / "data" / "chunk-000")
    ensure_dir(output_dir / "meta")

    append_run_log(log_path, f"[convert] archive={archive_path}")
    append_run_log(log_path, f"[convert] available_trials={len(references)} selected_trials={len(selected)}")

    write_json(output_dir / "meta" / "modality.json", VIDEO_MODALITY_JSON)
    write_tasks_file(output_dir / "meta", prompts)

    episodes_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    total_frames = 0

    for episode_index, reference in enumerate(selected):
        append_run_log(log_path, f"[convert] processing sample_id={reference.sample_id}")
        with h5py.File(reference.extracted_h5_path, "r") as h5:
            data = h5[reference.trial_name]["data"]
            frame_count = min(reference.frame_count, int(data["ctrl_arm"].shape[0]), int(data["qp_arm"].shape[0]))

            left = resize_frames(np.asarray(data["rgb_left"][:frame_count]), height, width)
            right = resize_frames(np.asarray(data["rgb_right"][:frame_count]), height, width)
            wrist = resize_frames(np.asarray(data["rgb_wrist"][:frame_count]), height, width)

            state_matrix = build_state_vector(
                np.asarray(data["qp_arm"][:frame_count]),
                np.asarray(data["qp_ee"][:frame_count]),
            )
            action_matrix = build_action_vector(
                np.asarray(data["ctrl_arm"][:frame_count]),
                np.asarray(data["ctrl_ee"][:frame_count]),
            )

        table = make_episode_table(
            episode_index=episode_index,
            frame_count=frame_count,
            prompts=prompts,
            state_matrix=state_matrix,
            action_matrix=action_matrix,
            fps=fps,
        )
        parquet_path = output_dir / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        pq.write_table(table, parquet_path)

        video_paths = save_episode_videos(
            episode_index=episode_index,
            output_dir=output_dir,
            fps=fps,
            left=left,
            right=right,
            wrist=wrist,
        )

        chunk_frame_indices = build_chunk_schedule(frame_count)
        anchor_frame_index = chunk_frame_indices[0][-1] if chunk_frame_indices else max(frame_count - 1, 0)
        manifest_rows.append(
            {
                "sample_id": reference.sample_id,
                "dataset_name": dataset_name,
                "archive_path": str(reference.archive_path),
                "extracted_h5_path": str(reference.extracted_h5_path),
                "trial_name": reference.trial_name,
                "episode_index": episode_index,
                "frame_count": frame_count,
                "fps": fps,
                "anchor_frame_index": anchor_frame_index,
                "initial_frame_indices": [0],
                "chunk_frame_indices": chunk_frame_indices,
                "camera_mapping": CAMERA_MAPPING,
                "unused_views": UNUSED_VIEWS,
                "prompt_variants": prompts,
                "gt_source": (
                    "full_episode_composite_from_left_right_wrist_resized_to_server_resolution"
                ),
                "parquet_path": str(parquet_path),
                "video_paths": video_paths,
            }
        )
        episodes_rows.append(
            {
                "episode_index": episode_index,
                "tasks": prompts,
                "length": frame_count,
                "success": reference.solved,
            }
        )
        total_frames += frame_count

    save_jsonl(output_dir / "meta" / "episodes.jsonl", episodes_rows)
    save_jsonl(output_dir / "manifest.jsonl", manifest_rows)
    write_info_file(output_dir, len(selected), total_frames, fps, height, width)

    conversion_log = {
        "dataset_name": dataset_name,
        "archive_path": str(archive_path),
        "extracted_dir": str(extracted_dir),
        "converted_dir": str(output_dir),
        "sample_count_requested": sample_count,
        "sample_count_actual": len(selected),
        "seed": seed,
        "camera_mapping": CAMERA_MAPPING,
        "unused_views": UNUSED_VIEWS,
        "prompt_variants": prompts,
        "state_mapping": "observation.state = [zeros(6), qp_ee(1), qp_arm(7)]",
        "action_mapping": "action = [zeros(6), zeros(6), ctrl_ee(1), zeros(1), ctrl_arm(7), zeros(7)]",
        "notes": [
            "RoboSet ctrl_arm/ctrl_ee are copied into the DROID joint_position/gripper_position slots as a compatibility mapping.",
            "Missing cartesian_position and velocity channels are zero-filled.",
            "rgb_top is not used for inference and is only tracked in metadata.",
        ],
        "extracted_tree_depth_2": extracted_tree,
        "selected_sample_ids": [reference.sample_id for reference in selected],
    }
    write_json(output_dir / "conversion_log.json", conversion_log)
    return conversion_log


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)

    dataset_root = Path(args.dataset_root) if args.dataset_root else find_existing_dataset_root()
    archives = sorted(path for path in dataset_root.iterdir() if path.is_file() and is_supported_archive(path))
    if not archives:
        raise FileNotFoundError(f"No supported archives found under {dataset_root}")

    overall_log: list[dict[str, Any]] = []
    for archive_path in archives:
        overall_log.append(
            convert_dataset(
                dataset_root=dataset_root,
                archive_path=archive_path,
                sample_count=args.sample_count,
                seed=args.seed,
                height=args.height,
                width=args.width,
                fps=args.fps,
                force=args.force,
            )
        )

    write_json(dataset_root / "dreamzero_converted" / "conversion_summary.json", overall_log)
    LOGGER.info("Conversion complete for %d archives", len(overall_log))


if __name__ == "__main__":
    main()
