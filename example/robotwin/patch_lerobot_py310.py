#!/usr/bin/env python3
"""Apply small Python 3.10 compatibility fixes to a local LeRobot checkout.

The RoboTwin official environment is Python 3.10, while current LeRobot main may
contain a few Python 3.12-only type syntax forms. This script keeps the local
checkout importable without changing runtime behavior.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lerobot-root",
        type=Path,
        default=Path("/data/dreamzero_mot/third_party/lerobot"),
        help="Path to the local LeRobot checkout.",
    )
    return parser.parse_args()


def replace_once(path: Path, old: str, new: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return False
    if old not in text:
        return False
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return True


def ensure_contains(path: Path, marker: str, insertion_after: str, insertion: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return False
    if insertion_after not in text:
        raise ValueError(f"Cannot patch {path}: anchor not found: {insertion_after!r}")
    path.write_text(text.replace(insertion_after, insertion_after + insertion, 1), encoding="utf-8")
    return True


def main() -> None:
    args = parse_args()
    root = args.lerobot_root.expanduser().resolve()
    src = root / "src" / "lerobot"
    if not src.exists():
        raise FileNotFoundError(f"LeRobot src directory not found: {src}")

    changed = []

    io_utils = src / "utils" / "io_utils.py"
    if replace_once(io_utils, "from typing import Any\n", "from typing import Any, TypeVar\n"):
        changed.append(io_utils)
    if ensure_contains(
        io_utils,
        'T = TypeVar("T", bound=JsonLike)',
        'JsonLike = str | int | float | bool | None | list["JsonLike"] | dict[str, "JsonLike"] | tuple["JsonLike", ...]\n',
        'T = TypeVar("T", bound=JsonLike)\n',
    ):
        changed.append(io_utils)
    if replace_once(
        io_utils,
        "def deserialize_json_into_object[T: JsonLike](fpath: Path, obj: T) -> T:\n",
        "def deserialize_json_into_object(fpath: Path, obj: T) -> T:\n",
    ):
        changed.append(io_utils)

    streaming_dataset = src / "datasets" / "streaming_dataset.py"
    if replace_once(
        streaming_dataset,
        "from pathlib import Path\n",
        "from pathlib import Path\nfrom typing import Generic, TypeVar\n",
    ):
        changed.append(streaming_dataset)
    if ensure_contains(
        streaming_dataset,
        'T = TypeVar("T")',
        "from .video_utils import (\n    VideoDecoderCache,\n    decode_video_frames_torchcodec,\n)\n",
        '\nT = TypeVar("T")\n',
    ):
        changed.append(streaming_dataset)
    if replace_once(streaming_dataset, "class Backtrackable[T]:\n", "class Backtrackable(Generic[T]):\n"):
        changed.append(streaming_dataset)

    pipeline = src / "processor" / "pipeline.py"
    if replace_once(
        pipeline,
        "from typing import Any, TypedDict, TypeVar, cast\n",
        "from typing import Any, Generic, TypedDict, TypeVar, cast\n",
    ):
        changed.append(pipeline)
    if replace_once(
        pipeline,
        "class DataProcessorPipeline[TInput, TOutput](HubMixin):\n",
        "class DataProcessorPipeline(Generic[TInput, TOutput], HubMixin):\n",
    ):
        changed.append(pipeline)

    motors_bus = src / "motors" / "motors_bus.py"
    if replace_once(motors_bus, "type NameOrID = str | int\n", "NameOrID = str | int\n"):
        changed.append(motors_bus)
    if replace_once(motors_bus, "type Value = int | float\n", "Value = int | float\n"):
        changed.append(motors_bus)

    unique_changed = sorted({path.as_posix() for path in changed})
    if unique_changed:
        print("Patched LeRobot Python 3.10 syntax:")
        for path in unique_changed:
            print(f"  {path}")
    else:
        print("LeRobot Python 3.10 syntax patch already applied.")


if __name__ == "__main__":
    main()
