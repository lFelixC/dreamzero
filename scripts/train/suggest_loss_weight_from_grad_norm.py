#!/usr/bin/env python3
"""Suggest a DreamZero action-loss weight from logged per-loss grad norms."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys


METRICS = {
    "joint": {
        "dynamics": "grad_conflict/joint/dynamics_grad_norm",
        "action": "grad_conflict/joint/action_grad_norm",
        "cosine": "grad_conflict/joint/dynamics_action_cosine",
        "conflict": "grad_conflict/joint/is_conflict",
    },
    "mot": {
        "dynamics": "grad_conflict/mot/dynamics_video_grad_norm",
        "action": "grad_conflict/mot/action_video_grad_norm",
        "action_branch": "grad_conflict/mot/action_action_grad_norm",
        "cosine": "grad_conflict/mot/video_dynamics_action_cosine",
        "conflict": "grad_conflict/mot/video_is_conflict",
    },
}


def _metric(entry: dict, key: str):
    if key in entry:
        return entry[key]
    train_key = f"train/{key}"
    return entry.get(train_key)


def _finite_float(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _resolve_log_path(path: Path) -> Path:
    if path.is_dir():
        return path / "loss_log.jsonl"
    return path


def _read_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def suggest(args: argparse.Namespace) -> dict:
    log_path = _resolve_log_path(Path(args.log))
    if not log_path.exists():
        raise FileNotFoundError(f"loss log not found: {log_path}")

    keys = METRICS[args.arch]
    ratios: list[float] = []
    dynamics_norms: list[float] = []
    action_norms: list[float] = []
    action_branch_norms: list[float] = []
    cosines: list[float] = []
    conflicts: list[float] = []

    for entry in _read_jsonl(log_path):
        step = int(entry.get("step", -1))
        if step < args.skip_first_steps:
            continue
        if args.step_min is not None and step < args.step_min:
            continue
        if args.step_max is not None and step > args.step_max:
            continue

        dynamics_norm = _finite_float(_metric(entry, keys["dynamics"]))
        action_norm = _finite_float(_metric(entry, keys["action"]))
        if dynamics_norm is None or action_norm is None:
            continue
        if action_norm <= args.eps or dynamics_norm < 0.0:
            continue

        dynamics_norms.append(dynamics_norm)
        action_norms.append(action_norm)
        ratios.append(dynamics_norm / action_norm)

        cosine = _finite_float(_metric(entry, keys["cosine"]))
        if cosine is not None:
            cosines.append(cosine)
        conflict = _finite_float(_metric(entry, keys["conflict"]))
        if conflict is not None:
            conflicts.append(conflict)

        action_branch_key = keys.get("action_branch")
        if action_branch_key is not None:
            action_branch_norm = _finite_float(_metric(entry, action_branch_key))
            if action_branch_norm is not None:
                action_branch_norms.append(action_branch_norm)

    if len(ratios) < args.min_samples:
        raise RuntimeError(
            f"Only found {len(ratios)} grad-norm samples in {log_path}; "
            f"need at least {args.min_samples}. Lower --min-samples, lower "
            "--skip-first-steps, or run calibration longer."
        )

    median_ratio = _median(ratios)
    assert median_ratio is not None
    equalized_weight = args.dynamics_loss_weight * median_ratio
    clipped_weight = min(max(equalized_weight, args.clip_min), args.clip_max)

    result = {
        "arch": args.arch,
        "log": str(log_path),
        "num_samples": len(ratios),
        "dynamics_loss_weight": args.dynamics_loss_weight,
        "equalized_action_loss_weight": equalized_weight,
        "suggested_action_loss_weight": clipped_weight,
        "clip_min": args.clip_min,
        "clip_max": args.clip_max,
        "median_dynamics_grad_norm": _median(dynamics_norms),
        "median_action_grad_norm": _median(action_norms),
        "median_dynamics_to_action_grad_ratio": median_ratio,
        "median_cosine": _median(cosines),
        "conflict_rate": (sum(1.0 for value in conflicts if value > 0.5) / len(conflicts))
        if conflicts
        else None,
    }
    if action_branch_norms:
        result["median_action_branch_grad_norm"] = _median(action_branch_norms)
        if result["median_action_branch_grad_norm"]:
            result["median_dynamics_to_action_branch_grad_ratio"] = (
                result["median_dynamics_grad_norm"] / result["median_action_branch_grad_norm"]
            )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=sorted(METRICS), required=True)
    parser.add_argument("--log", required=True, help="Path to loss_log.jsonl or an output directory.")
    parser.add_argument("--dynamics-loss-weight", type=float, default=1.0)
    parser.add_argument("--skip-first-steps", type=int, default=50)
    parser.add_argument("--step-min", type=int, default=None)
    parser.add_argument("--step-max", type=int, default=None)
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--clip-min", type=float, default=0.05)
    parser.add_argument("--clip-max", type=float, default=20.0)
    parser.add_argument("--format", choices=("json", "value", "shell"), default="json")
    args = parser.parse_args()

    result = suggest(args)
    if args.format == "value":
        print(f"{result['suggested_action_loss_weight']:.8g}")
    elif args.format == "shell":
        print(f"ACTION_LOSS_WEIGHT={result['suggested_action_loss_weight']:.8g}")
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
