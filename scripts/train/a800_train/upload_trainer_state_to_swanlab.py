#!/usr/bin/env python3
"""Upload HuggingFace Trainer scalar history to SwanLab.

Run this on a machine that can reach SwanLab, after copying the training
output directory from the offline cluster.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path, help="Training output directory.")
    parser.add_argument("--project", default="dreamzero", help="SwanLab project name.")
    parser.add_argument("--workspace", default=None, help="SwanLab workspace/user/org name.")
    parser.add_argument("--experiment-name", default=None, help="SwanLab experiment name.")
    parser.add_argument("--resume-id", default=None, help="Existing SwanLab experiment id to resume.")
    parser.add_argument(
        "--resume",
        default="must",
        choices=["must", "allow", "never"],
        help="Resume behavior used with --resume-id.",
    )
    parser.add_argument(
        "--step-mode",
        default="auto",
        choices=["trainer", "auto"],
        help=(
            "Use trainer_state step as the x-axis, or omit explicit step so "
            "SwanLab increments once per uploaded log call."
        ),
    )
    parser.add_argument("--mode", default="cloud", choices=["cloud", "local", "offline", "disabled"])
    parser.add_argument("--logdir", default=None, help="Optional SwanLab local log directory.")
    parser.add_argument(
        "--raw-metric-names",
        action="store_true",
        help="Upload Trainer metric names as-is instead of matching HuggingFace/W&B names.",
    )
    return parser.parse_args()


def latest_trainer_state(output_dir: Path) -> Path:
    candidates = list(output_dir.glob("checkpoint-*/trainer_state.json"))
    direct_state = output_dir / "trainer_state.json"
    if direct_state.exists():
        candidates.append(direct_state)

    if not candidates:
        raise FileNotFoundError(f"No trainer_state.json found under {output_dir}")

    def sort_key(path: Path) -> tuple[int, float]:
        if path.parent.name.startswith("checkpoint-"):
            try:
                step = int(path.parent.name.rsplit("-", 1)[1])
            except ValueError:
                step = -1
        else:
            step = 10**18
        return step, path.stat().st_mtime

    return sorted(candidates, key=sort_key)[-1]


def rewrite_metric_name(key: str) -> str:
    """Match the train/eval/test prefixes used by HuggingFace's W&B logger."""
    if "/" in key:
        return key
    if key.startswith("eval_"):
        return f"eval/{key.removeprefix('eval_')}"
    if key.startswith("test_"):
        return f"test/{key.removeprefix('test_')}"
    return f"train/{key}"


def scalar_metrics(entry: dict[str, Any], *, raw_metric_names: bool = False) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in entry.items():
        if key in {"step", "epoch"}:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            metric_name = key if raw_metric_names else rewrite_metric_name(key)
            metrics[metric_name] = float(value)
    return metrics


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    state_path = latest_trainer_state(output_dir)

    with state_path.open("r", encoding="utf-8") as f:
        state = json.load(f)

    log_history = state.get("log_history", [])
    if not isinstance(log_history, list):
        raise ValueError(f"{state_path} does not contain a list log_history")

    import swanlab

    experiment_name = args.experiment_name or output_dir.name
    init_kwargs: dict[str, Any] = {
        "project": args.project,
        "experiment_name": experiment_name,
        "mode": args.mode,
        "config": {
            "source_output_dir": str(output_dir),
            "source_trainer_state": str(state_path),
        },
    }
    if args.workspace:
        init_kwargs["workspace"] = args.workspace
    if args.resume_id:
        init_kwargs["id"] = args.resume_id
        init_kwargs["resume"] = args.resume
    if args.logdir:
        init_kwargs["logdir"] = args.logdir

    run = swanlab.init(**init_kwargs)
    uploaded = 0
    for entry in log_history:
        if not isinstance(entry, dict):
            continue
        step = entry.get("step")
        if not isinstance(step, int):
            continue
        metrics = scalar_metrics(entry, raw_metric_names=args.raw_metric_names)
        if not metrics:
            continue
        if args.step_mode == "auto":
            swanlab.log(metrics)
        else:
            swanlab.log(metrics, step=step)
        uploaded += 1

    if hasattr(swanlab, "finish"):
        swanlab.finish()
    elif hasattr(run, "finish"):
        run.finish()

    print(f"Uploaded {uploaded} scalar log entries from {state_path}")


if __name__ == "__main__":
    main()
