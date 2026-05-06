#!/usr/bin/env python3
"""Summarize DreamZero JSONL metrics for batch-size and LR sweeps."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def finite_number(value):
    return isinstance(value, (int, float)) and value == value and value not in {float("inf"), float("-inf")}


def summarize_run(run_dir: Path) -> dict:
    meta_path = run_dir / "run_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    rows = read_jsonl(run_dir / "metrics.jsonl")
    losses = [row["loss"] for row in rows if finite_number(row.get("loss"))]
    grad_norms = [row["grad_norm"] for row in rows if finite_number(row.get("grad_norm"))]
    summary = {
        "run": run_dir.name,
        "learning_rate": meta.get("learning_rate", ""),
        "per_device_batch_size": meta.get("per_device_train_batch_size", ""),
        "global_batch_size": meta.get("global_batch_size", ""),
        "logged_steps": len(rows),
        "first_loss": losses[0] if losses else "",
        "last_loss": losses[-1] if losses else "",
        "min_loss": min(losses) if losses else "",
        "last_grad_norm": grad_norms[-1] if grad_norms else "",
        "max_grad_norm": max(grad_norms) if grad_norms else "",
    }
    if len(losses) >= 20:
        summary["last20_loss_mean"] = sum(losses[-20:]) / 20
        summary["first20_loss_mean"] = sum(losses[:20]) / 20
    else:
        summary["last20_loss_mean"] = ""
        summary["first20_loss_mean"] = ""
    return summary


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("experiment_logs/lr_sweep")
    if not root.exists():
        print(f"Run directory root does not exist: {root}", file=sys.stderr)
        return 1
    if not root.is_dir():
        print(f"Run directory root is not a directory: {root}", file=sys.stderr)
        return 1
    run_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    rows = [summarize_run(path) for path in run_dirs]
    if not rows:
        print(f"No run directories found under {root}", file=sys.stderr)
        return 1
    fieldnames = list(rows[0].keys())
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
