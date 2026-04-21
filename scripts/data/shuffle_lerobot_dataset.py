#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import copy
from dataclasses import dataclass
import json
import logging
import math
import os
from pathlib import Path
import shutil
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

DEFAULT_OUTPUT_ROOT = Path("/data/datasets/dreamzero/1k_demo_lerobot_merged_v4_shuffle_only")
DEFAULT_NUM_STEPS_PER_SHARD = int(1e4)
DEFAULT_GRIPPER_INDICES = (6, 13)
REINDEXED_JSONL_FILES = {
    "episodes.jsonl",
    "episodes_stats.jsonl",
    "episodes_detail_global_instruction.jsonl",
    "step_filter.jsonl",
}
STALE_META_FILES_WHEN_CLIPPED = {
    "relative_horizon_stats_dreamzero.json",
    "relative_stats.json",
    "relative_stats_dreamzero.json",
    "stats.json",
}


class ShuffleError(RuntimeError):
    pass


@dataclass(frozen=True)
class EpisodeDescriptor:
    source_episode_index: int
    length: int
    tasks: tuple[str, ...]

    @property
    def task_label(self) -> str:
        if not self.tasks:
            return ""
        return " | ".join(self.tasks)


@dataclass(frozen=True)
class ClipConfig:
    state_indices: tuple[int, ...]
    action_indices: tuple[int, ...]
    lower: float
    upper: float


@dataclass(frozen=True)
class EpisodeRewriteResult:
    episode_length: int
    updated_column_stats: dict[str, dict[str, list[float | int]]]
    state_clip_counts: tuple[int, ...] = ()
    action_clip_counts: tuple[int, ...] = ()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Shuffle a LeRobot dataset without re-encoding videos, while rewriting episode-level metadata.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Path to the source LeRobot dataset root, or a directory containing lerobot_data/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Path to the shuffled output dataset root.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=42,
        help="Random seed for per-task shuffling and interleaving order (default: 42).",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="task_round_robin",
        choices=["task_round_robin", "random"],
        help="Shuffle strategy. task_round_robin is designed to mix tasks across shards.",
    )
    parser.add_argument(
        "--video-mode",
        type=str,
        default="hardlink",
        choices=["hardlink", "copy", "symlink"],
        help="How to materialize videos in the output dataset (default: hardlink).",
    )
    parser.add_argument(
        "--num-steps-per-shard",
        type=int,
        default=DEFAULT_NUM_STEPS_PER_SHARD,
        help="Shard size used only for before/after diagnostics (default: 10000).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete output-root if it already exists.",
    )
    parser.add_argument(
        "--clip-grippers",
        action="store_true",
        help="Clip packed gripper dims in observation.state and action while rewriting parquet.",
    )
    parser.add_argument(
        "--state-gripper-indices",
        type=int,
        nargs="*",
        default=list(DEFAULT_GRIPPER_INDICES),
        help="Packed observation.state indices to clip when --clip-grippers is enabled.",
    )
    parser.add_argument(
        "--action-gripper-indices",
        type=int,
        nargs="*",
        default=list(DEFAULT_GRIPPER_INDICES),
        help="Packed action indices to clip when --clip-grippers is enabled.",
    )
    parser.add_argument(
        "--clip-min",
        type=float,
        default=0.0,
        help="Lower bound used by --clip-grippers (default: 0.0).",
    )
    parser.add_argument(
        "--clip-max",
        type=float,
        default=1.0,
        help="Upper bound used by --clip-grippers (default: 1.0).",
    )
    args = parser.parse_args()
    if args.clip_max < args.clip_min:
        parser.error("--clip-max must be greater than or equal to --clip-min.")
    if args.clip_grippers:
        if not args.state_gripper_indices:
            parser.error("--clip-grippers requires at least one --state-gripper-indices value.")
        if not args.action_gripper_indices:
            parser.error("--clip-grippers requires at least one --action-gripper-indices value.")
    return args


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


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4)
        handle.write("\n")


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
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
            handle.write(json.dumps(record) + "\n")


def detect_video_keys(info: dict) -> list[str]:
    features = info.get("features", {})
    return [key for key, value in features.items() if value.get("dtype") == "video"]


def get_feature_dim(info: dict, column: str) -> int:
    feature = info.get("features", {}).get(column)
    if feature is None:
        raise ShuffleError(f"Expected feature '{column}' in meta/info.json.")
    shape = feature.get("shape")
    if not shape:
        raise ShuffleError(f"Feature '{column}' is missing shape metadata in meta/info.json.")
    if not isinstance(shape, list) or not shape:
        raise ShuffleError(f"Unsupported shape for feature '{column}': {shape}")
    return int(shape[0])


def validate_clip_indices(info: dict, column: str, indices: tuple[int, ...]) -> None:
    dim = get_feature_dim(info, column)
    invalid = [idx for idx in indices if idx < 0 or idx >= dim]
    if invalid:
        raise ShuffleError(
            f"Clip indices {invalid} are out of range for '{column}' with shape[0]={dim}."
        )


def build_clip_config(args: argparse.Namespace, info: dict) -> ClipConfig | None:
    if not args.clip_grippers:
        return None

    state_indices = tuple(args.state_gripper_indices)
    action_indices = tuple(args.action_gripper_indices)
    validate_clip_indices(info, "observation.state", state_indices)
    validate_clip_indices(info, "action", action_indices)
    return ClipConfig(
        state_indices=state_indices,
        action_indices=action_indices,
        lower=float(args.clip_min),
        upper=float(args.clip_max),
    )


def get_episode_chunk(info: dict, episode_index: int) -> int:
    return int(episode_index) // int(info.get("chunks_size", 1000))


def get_parquet_path(root: Path, info: dict, episode_index: int) -> Path:
    return root / info["data_path"].format(
        episode_chunk=get_episode_chunk(info, episode_index),
        episode_index=episode_index,
    )


def get_video_path(root: Path, info: dict, episode_index: int, video_key: str) -> Path:
    pattern = info.get("video_path")
    if not pattern:
        raise ShuffleError("Dataset does not define video_path in meta/info.json.")
    return root / pattern.format(
        episode_chunk=get_episode_chunk(info, episode_index),
        episode_index=episode_index,
        video_key=video_key,
    )


def normalize_tasks(tasks: object) -> tuple[str, ...]:
    if not isinstance(tasks, list):
        return ("",)
    values = tuple(str(task) for task in tasks if str(task))
    return values if values else ("",)


def build_episode_descriptors(dataset_root: Path, info: dict) -> list[EpisodeDescriptor]:
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    total_episodes = int(info["total_episodes"])

    if episodes_path.is_file():
        episode_records = read_jsonl(episodes_path)
        if len(episode_records) != total_episodes:
            raise ShuffleError(
                f"episodes.jsonl has {len(episode_records)} records, expected {total_episodes}."
            )
        descriptors = [
            EpisodeDescriptor(
                source_episode_index=int(record["episode_index"]),
                length=int(record["length"]),
                tasks=normalize_tasks(record.get("tasks", [])),
            )
            for record in episode_records
        ]
    else:
        tasks_map = load_tasks_map(dataset_root / "meta" / "tasks.jsonl")
        descriptors = []
        for episode_index in tqdm(range(total_episodes), desc="scan:episodes", unit="episode"):
            parquet_path = get_parquet_path(dataset_root, info, episode_index)
            df = pd.read_parquet(parquet_path, columns=["task_index"])
            unique_tasks = sorted(int(value) for value in df["task_index"].dropna().unique().tolist())
            tasks = tuple(tasks_map.get(task_index, str(task_index)) for task_index in unique_tasks) or ("",)
            descriptors.append(
                EpisodeDescriptor(
                    source_episode_index=episode_index,
                    length=len(df),
                    tasks=tasks,
                )
            )

    expected_indices = list(range(total_episodes))
    actual_indices = [descriptor.source_episode_index for descriptor in descriptors]
    if actual_indices != expected_indices:
        raise ShuffleError(
            "episodes.jsonl must enumerate episode_index in ascending contiguous order. "
            f"Expected first values like {expected_indices[:5]}, got {actual_indices[:5]}."
        )
    return descriptors


def load_tasks_map(tasks_path: Path) -> dict[int, str]:
    if not tasks_path.is_file():
        return {}
    return {
        int(record["task_index"]): str(record["task"])
        for record in read_jsonl(tasks_path)
    }


def parse_split_ranges(split_spec: object) -> list[tuple[int, int]]:
    if split_spec is None:
        return []
    if not isinstance(split_spec, str):
        raise ShuffleError(f"Unsupported split spec type: {type(split_spec)}")
    ranges: list[tuple[int, int]] = []
    for chunk in split_spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ShuffleError(f"Invalid split range: {chunk}")
        start_str, end_str = chunk.split(":", 1)
        start = int(start_str)
        end = int(end_str)
        if end < start:
            raise ShuffleError(f"Invalid split range: {chunk}")
        ranges.append((start, end))
    return ranges


def build_split_groups(info: dict, episodes: list[EpisodeDescriptor]) -> list[tuple[str, list[EpisodeDescriptor]]]:
    splits = info.get("splits")
    if not splits:
        return [("all", episodes)]

    groups: list[tuple[str, list[EpisodeDescriptor]]] = []
    for split_name, split_spec in splits.items():
        split_episodes: list[EpisodeDescriptor] = []
        for start, end in parse_split_ranges(split_spec):
            split_episodes.extend(episodes[start:end])
        groups.append((split_name, split_episodes))
    return groups


def shuffle_random(episodes: list[EpisodeDescriptor], rng: np.random.Generator) -> list[EpisodeDescriptor]:
    permutation = rng.permutation(len(episodes))
    return [episodes[index] for index in permutation.tolist()]


def shuffle_task_round_robin(
    episodes: list[EpisodeDescriptor],
    rng: np.random.Generator,
) -> list[EpisodeDescriptor]:
    buckets: dict[tuple[str, ...], deque[EpisodeDescriptor]] = defaultdict(deque)
    for episode in episodes:
        buckets[episode.tasks].append(episode)

    for task_key, bucket in buckets.items():
        shuffled = [bucket[index] for index in rng.permutation(len(bucket)).tolist()]
        buckets[task_key] = deque(shuffled)

    active_keys = sorted(buckets, key=lambda key: repr(key))
    rng.shuffle(active_keys)

    shuffled_plan: list[EpisodeDescriptor] = []
    while active_keys:
        round_keys = list(active_keys)
        rng.shuffle(round_keys)
        next_active_keys: list[tuple[str, ...]] = []
        for task_key in round_keys:
            bucket = buckets[task_key]
            if not bucket:
                continue
            shuffled_plan.append(bucket.popleft())
            if bucket:
                next_active_keys.append(task_key)
        active_keys = next_active_keys
    return shuffled_plan


def shuffle_split(
    episodes: list[EpisodeDescriptor],
    rng: np.random.Generator,
    strategy: str,
) -> list[EpisodeDescriptor]:
    if strategy == "random":
        return shuffle_random(episodes, rng)
    if strategy == "task_round_robin":
        return shuffle_task_round_robin(episodes, rng)
    raise ValueError(f"Unsupported strategy: {strategy}")


def build_shuffle_plan(
    info: dict,
    episodes: list[EpisodeDescriptor],
    seed: int,
    strategy: str,
) -> tuple[list[EpisodeDescriptor], dict[str, int]]:
    rng = np.random.default_rng(seed)
    split_groups = build_split_groups(info, episodes)
    shuffled: list[EpisodeDescriptor] = []
    split_sizes: dict[str, int] = {}
    for split_name, split_episodes in split_groups:
        shuffled_split = shuffle_split(split_episodes, rng, strategy)
        shuffled.extend(shuffled_split)
        split_sizes[split_name] = len(shuffled_split)

    source_indices = [episode.source_episode_index for episode in shuffled]
    if len(source_indices) != len(episodes) or len(set(source_indices)) != len(episodes):
        raise ShuffleError("Shuffle plan does not contain each source episode exactly once.")
    return shuffled, split_sizes


def summarize_task_runs(episodes: list[EpisodeDescriptor]) -> dict[str, float | int]:
    if not episodes:
        return {"num_runs": 0, "max_run": 0, "mean_run": 0.0}

    run_lengths: list[int] = []
    current_label = episodes[0].task_label
    current_run = 1
    for episode in episodes[1:]:
        if episode.task_label == current_label:
            current_run += 1
            continue
        run_lengths.append(current_run)
        current_label = episode.task_label
        current_run = 1
    run_lengths.append(current_run)
    return {
        "num_runs": len(run_lengths),
        "max_run": max(run_lengths),
        "mean_run": float(sum(run_lengths) / len(run_lengths)),
    }


def summarize_shards(
    episodes: list[EpisodeDescriptor],
    num_steps_per_shard: int,
) -> dict[str, float | int]:
    if not episodes or num_steps_per_shard <= 0:
        return {
            "num_shards": 0,
            "min_unique_tasks_per_shard": 0,
            "mean_unique_tasks_per_shard": 0.0,
            "max_dominant_task_step_frac": 0.0,
        }

    total_steps = int(sum(episode.length for episode in episodes))
    num_shards = int(math.ceil(total_steps / num_steps_per_shard))
    cutoffs = np.linspace(0, total_steps, num_shards + 1)[1:]
    shard_episodes: list[list[EpisodeDescriptor]] = [[]]
    curr_steps = 0
    curr_cutoff = 0

    for episode in episodes:
        shard_episodes[-1].append(episode)
        curr_steps += episode.length
        if curr_cutoff < len(cutoffs) - 1 and curr_steps > cutoffs[curr_cutoff]:
            shard_episodes.append([])
            curr_cutoff += 1

    unique_counts: list[int] = []
    dominant_step_fracs: list[float] = []
    for shard in shard_episodes:
        step_counter: Counter[str] = Counter()
        shard_steps = 0
        for episode in shard:
            step_counter[episode.task_label] += episode.length
            shard_steps += episode.length
        unique_counts.append(len(step_counter))
        dominant_step_fracs.append(max(step_counter.values()) / shard_steps if shard_steps else 0.0)

    return {
        "num_shards": len(shard_episodes),
        "min_unique_tasks_per_shard": min(unique_counts),
        "mean_unique_tasks_per_shard": float(sum(unique_counts) / len(unique_counts)),
        "max_dominant_task_step_frac": max(dominant_step_fracs),
    }


def log_plan_summary(
    logger: logging.Logger,
    label: str,
    episodes: list[EpisodeDescriptor],
    num_steps_per_shard: int,
) -> None:
    run_stats = summarize_task_runs(episodes)
    shard_stats = summarize_shards(episodes, num_steps_per_shard)
    logger.info(
        "%s | runs[max=%d mean=%.2f total=%d] shards[count=%d min_unique_tasks=%d mean_unique_tasks=%.2f max_dominant_step_frac=%.3f]",
        label,
        run_stats["max_run"],
        run_stats["mean_run"],
        run_stats["num_runs"],
        shard_stats["num_shards"],
        shard_stats["min_unique_tasks_per_shard"],
        shard_stats["mean_unique_tasks_per_shard"],
        shard_stats["max_dominant_task_step_frac"],
    )


def maybe_delete_output(output_root: Path, force: bool) -> None:
    if not output_root.exists():
        return
    if not force:
        raise FileExistsError(f"{output_root} already exists. Use --force to overwrite.")
    shutil.rmtree(output_root)


def materialize_file(source: Path, destination: Path, mode: str, logger: logging.Logger) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, destination)
        return
    if mode == "symlink":
        os.symlink(source.resolve(), destination)
        return
    if mode == "hardlink":
        try:
            os.link(source, destination)
            return
        except OSError as exc:
            logger.warning(
                "Hardlink failed for %s -> %s (%s). Falling back to copy.",
                source,
                destination,
                exc,
            )
            shutil.copy2(source, destination)
            return
    raise ValueError(f"Unsupported materialization mode: {mode}")


def compute_column_stats(values: np.ndarray) -> dict[str, list[float | int]]:
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    return {
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "count": [int(values.shape[0])],
    }


def clip_dataframe_column(
    df: pd.DataFrame,
    column: str,
    indices: tuple[int, ...],
    lower: float,
    upper: float,
) -> tuple[tuple[int, ...], dict[str, list[float | int]]]:
    if column not in df.columns:
        raise ShuffleError(f"Cannot clip missing column '{column}'.")

    values = np.stack(df[column].to_numpy())
    clipped = values.copy()
    clip_counts: list[int] = []
    for idx in indices:
        original = clipped[:, idx].copy()
        clipped[:, idx] = np.clip(clipped[:, idx], lower, upper)
        clip_counts.append(int(np.count_nonzero(np.abs(original - clipped[:, idx]) > 1e-8)))

    df[column] = pd.Series(list(clipped), index=df.index)
    return tuple(clip_counts), compute_column_stats(clipped)


def rewrite_episode_table(
    source_parquet_path: Path,
    destination_parquet_path: Path,
    new_episode_index: int,
    global_index_start: int,
    clip_config: ClipConfig | None = None,
) -> EpisodeRewriteResult:
    df = pd.read_parquet(source_parquet_path)
    episode_length = len(df)
    updated_column_stats: dict[str, dict[str, list[float | int]]] = {}
    state_clip_counts: tuple[int, ...] = ()
    action_clip_counts: tuple[int, ...] = ()
    if clip_config is not None:
        state_clip_counts, updated_column_stats["observation.state"] = clip_dataframe_column(
            df=df,
            column="observation.state",
            indices=clip_config.state_indices,
            lower=clip_config.lower,
            upper=clip_config.upper,
        )
        action_clip_counts, updated_column_stats["action"] = clip_dataframe_column(
            df=df,
            column="action",
            indices=clip_config.action_indices,
            lower=clip_config.lower,
            upper=clip_config.upper,
        )
    df["episode_index"] = np.full(episode_length, new_episode_index, dtype=np.int64)
    df["index"] = np.arange(
        global_index_start,
        global_index_start + episode_length,
        dtype=np.int64,
    )
    destination_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(destination_parquet_path, index=False)
    return EpisodeRewriteResult(
        episode_length=episode_length,
        updated_column_stats=updated_column_stats,
        state_clip_counts=state_clip_counts,
        action_clip_counts=action_clip_counts,
    )


def build_updated_splits(info: dict, split_sizes: dict[str, int]) -> dict[str, str] | None:
    if not info.get("splits"):
        return None
    updated: dict[str, str] = {}
    start = 0
    for split_name in info["splits"]:
        size = split_sizes.get(split_name, 0)
        updated[split_name] = f"{start}:{start + size}"
        start += size
    return updated


def update_info_metadata(info: dict, split_sizes: dict[str, int], old_to_new: dict[int, int]) -> dict:
    updated = copy.deepcopy(info)
    updated_splits = build_updated_splits(info, split_sizes)
    if updated_splits is not None:
        updated["splits"] = updated_splits
    if "discarded_episode_indices" in updated:
        updated["discarded_episode_indices"] = sorted(
            old_to_new[old_episode_index]
            for old_episode_index in updated["discarded_episode_indices"]
            if old_episode_index in old_to_new
        )
    return updated


def rewrite_episodes_metadata(
    source_records: list[dict],
    plan: list[EpisodeDescriptor],
) -> list[dict]:
    source_by_episode_index = {
        int(record["episode_index"]): record
        for record in source_records
    }
    output_records: list[dict] = []
    for new_episode_index, descriptor in enumerate(plan):
        record = copy.deepcopy(source_by_episode_index[descriptor.source_episode_index])
        record["episode_index"] = new_episode_index
        output_records.append(record)
    return output_records


def consecutive_int_stats(start: int, length: int) -> dict[str, list[float | int]]:
    end = start + length - 1
    std = math.sqrt((length**2 - 1) / 12.0) if length > 1 else 0.0
    return {
        "min": [start],
        "max": [end],
        "mean": [float(start + end) / 2.0],
        "std": [float(std)],
        "count": [length],
    }


def patch_episode_stats_record(
    record: dict,
    new_episode_index: int,
    global_index_start: int,
    episode_length: int,
    updated_column_stats: dict[str, dict[str, list[float | int]]] | None = None,
) -> dict:
    patched = copy.deepcopy(record)
    patched["episode_index"] = new_episode_index
    stats = patched.get("stats", {})
    for column, column_stats in (updated_column_stats or {}).items():
        stats[column] = column_stats
    stats["episode_index"] = {
        "min": [new_episode_index],
        "max": [new_episode_index],
        "mean": [float(new_episode_index)],
        "std": [0.0],
        "count": [episode_length],
    }
    stats["index"] = consecutive_int_stats(global_index_start, episode_length)
    patched["stats"] = stats
    return patched


def rewrite_episode_stats_metadata(
    source_records: list[dict],
    plan: list[EpisodeDescriptor],
    episode_stats_updates: dict[int, dict[str, dict[str, list[float | int]]]] | None = None,
) -> list[dict]:
    source_by_episode_index = {
        int(record["episode_index"]): record
        for record in source_records
    }
    output_records: list[dict] = []
    global_index_start = 0
    for new_episode_index, descriptor in enumerate(plan):
        record = source_by_episode_index[descriptor.source_episode_index]
        output_records.append(
            patch_episode_stats_record(
                record=record,
                new_episode_index=new_episode_index,
                global_index_start=global_index_start,
                episode_length=descriptor.length,
                updated_column_stats=(episode_stats_updates or {}).get(new_episode_index),
            )
        )
        global_index_start += descriptor.length
    return output_records


def rewrite_generic_episode_jsonl(
    source_records: list[dict],
    plan: list[EpisodeDescriptor],
) -> list[dict]:
    source_by_episode_index = {
        int(record["episode_index"]): record
        for record in source_records
    }
    output_records: list[dict] = []
    for new_episode_index, descriptor in enumerate(plan):
        record = copy.deepcopy(source_by_episode_index[descriptor.source_episode_index])
        record["episode_index"] = new_episode_index
        output_records.append(record)
    return output_records


def normalize_initial_actions_payload(payload: np.ndarray) -> list[dict]:
    data = payload.tolist()
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    raise ShuffleError(f"Unsupported initial_actions.npz payload type: {type(data)}")


def rewrite_initial_actions(
    source_path: Path,
    destination_path: Path,
    old_to_new: dict[int, int],
) -> None:
    payload = np.load(str(source_path), allow_pickle=True)
    array = payload[payload.files[0]]
    initial_actions = normalize_initial_actions_payload(array)
    remapped_payload: list[dict] = []

    for dataset_actions in initial_actions:
        remapped_actions = {}
        for key, value in dataset_actions.items():
            old_episode_index = int(key)
            new_episode_index = old_to_new[old_episode_index]
            remapped_key = str(new_episode_index) if isinstance(key, str) else new_episode_index
            remapped_actions[remapped_key] = value
        remapped_payload.append(remapped_actions)

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(destination_path), remapped_payload)


def copy_top_level_extras(dataset_root: Path, output_root: Path) -> None:
    for path in dataset_root.iterdir():
        if path.name in {"data", "videos", "meta"}:
            continue
        destination = output_root / path.name
        if path.is_dir():
            shutil.copytree(path, destination)
        else:
            shutil.copy2(path, destination)


def rewrite_meta_files(
    dataset_root: Path,
    output_root: Path,
    info: dict,
    split_sizes: dict[str, int],
    plan: list[EpisodeDescriptor],
    old_to_new: dict[int, int],
    episode_stats_updates: dict[int, dict[str, dict[str, list[float | int]]]],
    clip_config: ClipConfig | None,
    logger: logging.Logger,
) -> None:
    source_meta_dir = dataset_root / "meta"
    destination_meta_dir = output_root / "meta"
    destination_meta_dir.mkdir(parents=True, exist_ok=True)

    for path in sorted(source_meta_dir.iterdir()):
        destination = destination_meta_dir / path.name

        if path.name == "info.json":
            updated_info = update_info_metadata(info, split_sizes, old_to_new)
            write_json(destination, updated_info)
            continue

        if path.name == "tasks.jsonl":
            shutil.copy2(path, destination)
            continue

        if path.name == "episodes.jsonl":
            source_records = read_jsonl(path)
            write_jsonl(destination, rewrite_episodes_metadata(source_records, plan))
            continue

        if path.name == "episodes_stats.jsonl":
            source_records = read_jsonl(path)
            write_jsonl(
                destination,
                rewrite_episode_stats_metadata(
                    source_records,
                    plan,
                    episode_stats_updates=episode_stats_updates,
                ),
            )
            continue

        if path.name.endswith(".jsonl") and path.name in REINDEXED_JSONL_FILES:
            source_records = read_jsonl(path)
            write_jsonl(destination, rewrite_generic_episode_jsonl(source_records, plan))
            continue

        if path.name == "initial_actions.npz":
            rewrite_initial_actions(path, destination, old_to_new)
            if clip_config is not None:
                logger.warning(
                    "Rekeyed initial_actions.npz without clipping payload values. "
                    "Regenerate it if your downstream consumes initial action caches."
                )
            continue

        if clip_config is not None and path.name in STALE_META_FILES_WHEN_CLIPPED:
            logger.warning(
                "Skipping %s because --clip-grippers changes underlying action/state statistics. "
                "Regenerate DreamZero metadata after shuffling.",
                path.name,
            )
            continue

        shutil.copy2(path, destination)
        logger.info("Copied extra meta file: %s", path.name)


def build_dataset(args: argparse.Namespace) -> None:
    logger = logging.getLogger("shuffle_lerobot_dataset")
    dataset_root = resolve_dataset_root(args.input_root)
    info = read_json(dataset_root / "meta" / "info.json")
    clip_config = build_clip_config(args, info)
    total_episodes = int(info["total_episodes"])
    video_keys = detect_video_keys(info)
    source_episodes = build_episode_descriptors(dataset_root, info)

    logger.info(
        "Source dataset: %s | episodes=%d total_frames=%d video_keys=%d strategy=%s clip_grippers=%s",
        dataset_root,
        total_episodes,
        int(info.get("total_frames", 0)),
        len(video_keys),
        args.strategy,
        clip_config is not None,
    )
    if clip_config is not None:
        logger.info(
            "Clip config | observation.state=%s action=%s range=[%.3f, %.3f]",
            list(clip_config.state_indices),
            list(clip_config.action_indices),
            clip_config.lower,
            clip_config.upper,
        )
    log_plan_summary(logger, "before", source_episodes, args.num_steps_per_shard)

    shuffle_plan, split_sizes = build_shuffle_plan(
        info=info,
        episodes=source_episodes,
        seed=args.shuffle_seed,
        strategy=args.strategy,
    )
    log_plan_summary(logger, "after", shuffle_plan, args.num_steps_per_shard)

    preview = ", ".join(
        f"{descriptor.source_episode_index:06d}->{new_episode_index:06d}:{descriptor.task_label}"
        for new_episode_index, descriptor in enumerate(shuffle_plan[:10])
    )
    logger.info("Shuffle preview: %s", preview)

    maybe_delete_output(args.output_root, args.force)
    args.output_root.mkdir(parents=True, exist_ok=True)
    copy_top_level_extras(dataset_root, args.output_root)

    old_to_new = {
        descriptor.source_episode_index: new_episode_index
        for new_episode_index, descriptor in enumerate(shuffle_plan)
    }

    total_state_clip = np.zeros(len(clip_config.state_indices), dtype=np.int64) if clip_config is not None else None
    total_action_clip = np.zeros(len(clip_config.action_indices), dtype=np.int64) if clip_config is not None else None
    episode_stats_updates: dict[int, dict[str, dict[str, list[float | int]]]] = {}
    global_index_start = 0
    for new_episode_index, descriptor in enumerate(
        tqdm(shuffle_plan, desc="shuffle:episodes", unit="episode")
    ):
        source_episode_index = descriptor.source_episode_index
        source_parquet_path = get_parquet_path(dataset_root, info, source_episode_index)
        destination_parquet_path = get_parquet_path(args.output_root, info, new_episode_index)

        rewrite_result = rewrite_episode_table(
            source_parquet_path=source_parquet_path,
            destination_parquet_path=destination_parquet_path,
            new_episode_index=new_episode_index,
            global_index_start=global_index_start,
            clip_config=clip_config,
        )
        if rewrite_result.episode_length != descriptor.length:
            raise ShuffleError(
                f"Episode length mismatch for source episode {source_episode_index}: "
                f"episodes.jsonl says {descriptor.length}, parquet says {rewrite_result.episode_length}."
            )
        if rewrite_result.updated_column_stats:
            episode_stats_updates[new_episode_index] = rewrite_result.updated_column_stats
        if total_state_clip is not None:
            total_state_clip += np.array(rewrite_result.state_clip_counts, dtype=np.int64)
        if total_action_clip is not None:
            total_action_clip += np.array(rewrite_result.action_clip_counts, dtype=np.int64)

        for video_key in video_keys:
            source_video_path = get_video_path(dataset_root, info, source_episode_index, video_key)
            destination_video_path = get_video_path(args.output_root, info, new_episode_index, video_key)
            materialize_file(source_video_path, destination_video_path, args.video_mode, logger)

        global_index_start += rewrite_result.episode_length

    rewrite_meta_files(
        dataset_root=dataset_root,
        output_root=args.output_root,
        info=info,
        split_sizes=split_sizes,
        plan=shuffle_plan,
        old_to_new=old_to_new,
        episode_stats_updates=episode_stats_updates,
        clip_config=clip_config,
        logger=logger,
    )

    if clip_config is not None and total_state_clip is not None and total_action_clip is not None:
        logger.info(
            "Total gripper clips | observation.state%s=%s action%s=%s",
            list(clip_config.state_indices),
            total_state_clip.tolist(),
            list(clip_config.action_indices),
            total_action_clip.tolist(),
        )
    logger.info(
        "Finished shuffle-only rewrite at %s | episodes=%d total_frames=%d video_mode=%s",
        args.output_root,
        total_episodes,
        global_index_start,
        args.video_mode,
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    build_dataset(args)


if __name__ == "__main__":
    main()
