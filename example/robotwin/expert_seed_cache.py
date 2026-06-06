#!/usr/bin/env python3
"""Build fixed expert-accepted seed/prompt caches for RoboTwin eval."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
for extra in (
    REPO_ROOT / "third_party" / "RoboTwin",
    REPO_ROOT / "third_party" / "lerobot" / "src",
    REPO_ROOT / "third_party" / "lerobot",
):
    if extra.exists() and str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from example.robotwin.robotwin_fast_env import (  # noqa: E402
    DEFAULT_ROBOTWIN_TASK_CONFIG,
    ROBOTWIN_TASK_CONFIG_ENV,
    check_robotwin_expert_seed,
    generate_seen_instruction_deterministic,
)

DEFAULT_OUTPUT_ROOT = Path("/data/checkpoints/dreamzero/robotwin_eval_runs")


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else int(raw)


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def load_all_eval_tasks() -> list[str]:
    limit_path = REPO_ROOT / "third_party" / "RoboTwin" / "task_config" / "_eval_step_limit.yml"
    if not limit_path.exists():
        raise FileNotFoundError(f"Missing RoboTwin eval task limit file: {limit_path}")
    tasks: list[str] = []
    for line in limit_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        task = line.split(":", 1)[0].strip()
        if task:
            tasks.append(task)
    if not tasks:
        raise ValueError(f"No RoboTwin tasks found in {limit_path}")
    return tasks


def parse_tasks(raw: str) -> list[str]:
    parts = [item.strip() for item in raw.replace(",", " ").split() if item.strip()]
    if not parts:
        raise ValueError("--tasks must contain at least one RoboTwin task")
    if len(parts) == 1 and parts[0].lower() == "all":
        return load_all_eval_tasks()
    if any(item.lower() == "all" for item in parts):
        raise ValueError("Use --tasks all by itself, or provide explicit task names")
    return parts


def default_cache_root() -> Path:
    if os.environ.get("SEED_CACHE_ROOT"):
        return Path(os.environ["SEED_CACHE_ROOT"])
    return Path(os.environ.get("OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT))) / "expert_seed_cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default=os.environ.get("TASKS", os.environ.get("TASK", "all")))
    parser.add_argument("--cache-root", type=Path, default=default_cache_root())
    parser.add_argument(
        "--task-config",
        default=os.environ.get(ROBOTWIN_TASK_CONFIG_ENV, os.environ.get("TASK_CONFIG", DEFAULT_ROBOTWIN_TASK_CONFIG)),
    )
    parser.add_argument("--episodes-per-task", type=int, default=env_int("CACHE_EPISODES_PER_TASK", 200))
    parser.add_argument("--seed-start", type=int, default=env_int("SEED_START", 10000))
    parser.add_argument("--max-candidates", type=int, default=env_int("CACHE_MAX_CANDIDATES", 200000))
    parser.add_argument("--max-descriptions", type=int, default=env_int("CACHE_MAX_DESCRIPTIONS", 100))
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=env_bool("CACHE_OVERWRITE", False))
    parser.add_argument("--status-jsonl", type=Path, default=None)
    return parser.parse_args()


def read_existing(path: Path, *, task: str, task_config: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("task") != task:
            raise ValueError(f"{path}:{line_no} has task={row.get('task')!r}, expected {task!r}")
        if row.get("task_config") != task_config:
            raise ValueError(
                f"{path}:{line_no} has task_config={row.get('task_config')!r}, expected {task_config!r}"
            )
        rows.append(row)
    return rows


def write_rows_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")
    tmp.replace(path)


def append_status(status_path: Path | None, event: str, **payload: Any) -> None:
    row = {"timestamp": time.time(), "event": event, **payload}
    line = json.dumps(row, ensure_ascii=False, default=json_default)
    print(f"[seed-cache] {event} {json.dumps(payload, ensure_ascii=False, default=json_default)}", flush=True)
    if status_path is None:
        return
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with status_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def build_task_cache(args: argparse.Namespace, *, task: str) -> dict[str, Any]:
    cache_path = args.cache_root.expanduser().resolve() / args.task_config / f"{task}.jsonl"
    existing = [] if args.overwrite else read_existing(cache_path, task=task, task_config=args.task_config)
    if len(existing) >= args.episodes_per_task:
        append_status(
            args.status_jsonl,
            "task_cache_ready",
            task=task,
            path=cache_path,
            rows=len(existing),
            required=args.episodes_per_task,
        )
        return {"task": task, "path": cache_path.as_posix(), "rows": len(existing), "generated": 0}

    rows = list(existing)
    seen_seeds = {int(row["seed"]) for row in rows if "seed" in row}
    next_candidate_index = 0
    if rows:
        next_candidate_index = max(int(row.get("candidate_index", -1)) for row in rows) + 1

    append_status(
        args.status_jsonl,
        "task_cache_start",
        task=task,
        path=cache_path,
        existing=len(rows),
        required=args.episodes_per_task,
        seed_start=args.seed_start,
        next_candidate_index=next_candidate_index,
    )
    generated = 0
    checked = 0
    for candidate_index in range(next_candidate_index, args.max_candidates):
        if len(rows) >= args.episodes_per_task:
            break
        seed = int(args.seed_start + candidate_index)
        if seed in seen_seeds:
            continue
        checked += 1
        expert = check_robotwin_expert_seed(task, seed)
        if not bool(expert.get("accepted", False)):
            append_status(
                args.status_jsonl,
                "candidate_rejected",
                task=task,
                seed=seed,
                candidate_index=candidate_index,
                error_type=expert.get("error_type", ""),
                error=expert.get("error", ""),
            )
            continue
        episode_info = expert.get("episode_info")
        if not isinstance(episode_info, dict):
            append_status(
                args.status_jsonl,
                "candidate_rejected",
                task=task,
                seed=seed,
                candidate_index=candidate_index,
                error_type="MissingEpisodeInfo",
                error="accepted expert result did not include episode_info",
            )
            continue
        prompt = generate_seen_instruction_deterministic(
            task,
            episode_info,
            max_descriptions=args.max_descriptions,
            selection_index=len(rows),
        )
        row = {
            "task": task,
            "task_config": args.task_config,
            "episode_index": len(rows),
            "seed": seed,
            "prompt": prompt,
            "episode_info": episode_info,
            "candidate_index": candidate_index,
            "accepted": True,
            "expert": {
                "plan_success": bool(expert.get("plan_success", False)),
                "check_success": bool(expert.get("check_success", False)),
                "duration": float(expert.get("duration", 0.0) or 0.0),
            },
            "cache_version": 1,
            "generated_at": time.time(),
            "seed_start": int(args.seed_start),
            "max_descriptions": int(args.max_descriptions),
        }
        rows.append(row)
        seen_seeds.add(seed)
        generated += 1
        write_rows_atomic(cache_path, rows)
        append_status(
            args.status_jsonl,
            "candidate_accepted",
            task=task,
            seed=seed,
            candidate_index=candidate_index,
            episode_index=row["episode_index"],
            rows=len(rows),
        )

    if len(rows) < args.episodes_per_task:
        raise RuntimeError(
            f"Only found {len(rows)}/{args.episodes_per_task} accepted seeds for task={task!r} "
            f"after checking {checked} candidates from index {next_candidate_index}"
        )
    append_status(
        args.status_jsonl,
        "task_cache_done",
        task=task,
        path=cache_path,
        rows=len(rows),
        generated=generated,
    )
    return {"task": task, "path": cache_path.as_posix(), "rows": len(rows), "generated": generated}


def main() -> None:
    args = parse_args()
    if args.episodes_per_task <= 0:
        raise ValueError("--episodes-per-task must be positive")
    if args.max_candidates <= 0:
        raise ValueError("--max-candidates must be positive")
    os.environ[ROBOTWIN_TASK_CONFIG_ENV] = str(args.task_config)
    args.cache_root = args.cache_root.expanduser().resolve()
    if args.status_jsonl is not None:
        args.status_jsonl = args.status_jsonl.expanduser().resolve()
    tasks = parse_tasks(args.tasks)
    summaries = [build_task_cache(args, task=task) for task in tasks]
    done = {
        "mode": "robotwin_expert_seed_cache",
        "task_config": args.task_config,
        "cache_root": args.cache_root.as_posix(),
        "tasks": summaries,
    }
    print(json.dumps(done, indent=2, ensure_ascii=False, default=json_default))


if __name__ == "__main__":
    main()
