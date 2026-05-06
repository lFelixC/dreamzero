#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_episodes(dataset_root: Path) -> list[dict]:
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if not episodes_path.is_file():
        raise FileNotFoundError(f"Missing episodes metadata: {episodes_path}")
    episodes = []
    with episodes_path.open("r") as f:
        for line in f:
            episodes.append(json.loads(line))
    return episodes


def write_split(path: Path, episode_indices: list[int], source_dataset: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_dataset": str(source_dataset),
        "seed": seed,
        "num_episodes": len(episode_indices),
        "episode_indices": episode_indices,
    }
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a fixed DROID train/val episode split.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/data/datasets/dreamzero/droid_lerobot"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/datasets/dreamzero/droid_lerobot_splits"),
    )
    parser.add_argument("--train-episodes", type=int, default=2000)
    parser.add_argument("--val-episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-length",
        type=int,
        default=32,
        help="Discard very short episodes that cannot provide the default DreamZero horizon.",
    )
    args = parser.parse_args()

    episodes = read_episodes(args.dataset_root)
    candidates = [
        int(ep["episode_index"])
        for ep in episodes
        if int(ep.get("length", 0)) >= args.min_length
    ]
    requested = args.train_episodes + args.val_episodes
    if requested > len(candidates):
        raise ValueError(
            f"Requested {requested} episodes, but only {len(candidates)} meet min_length={args.min_length}."
        )

    rng = np.random.default_rng(args.seed)
    shuffled = np.array(candidates, dtype=np.int64)
    rng.shuffle(shuffled)

    train_ids = sorted(shuffled[: args.train_episodes].astype(int).tolist())
    val_ids = sorted(
        shuffled[args.train_episodes : args.train_episodes + args.val_episodes]
        .astype(int)
        .tolist()
    )

    write_split(args.output_root / "train.json", train_ids, args.dataset_root, args.seed)
    write_split(args.output_root / "val.json", val_ids, args.dataset_root, args.seed)

    manifest = {
        "source_dataset": str(args.dataset_root),
        "seed": args.seed,
        "min_length": args.min_length,
        "total_episodes": len(episodes),
        "candidate_episodes": len(candidates),
        "train_episodes": len(train_ids),
        "val_episodes": len(val_ids),
        "train_path": str(args.output_root / "train.json"),
        "val_path": str(args.output_root / "val.json"),
    }
    with (args.output_root / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
