#!/usr/bin/env python3
"""Build fixed expert-accepted seed/prompt caches for RoboTwin eval."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import queue
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
    parser.add_argument("--num-workers", type=int, default=env_int("CACHE_NUM_WORKERS", 1))
    parser.add_argument("--prefetch-window", type=int, default=env_int("CACHE_PREFETCH_WINDOW", 0))
    parser.add_argument("--worker-gpus", default=os.environ.get("CACHE_WORKER_GPUS", ""))
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


def parse_worker_gpus(raw: str) -> list[str]:
    value = str(raw or "").strip()
    if not value:
        return []
    separator = ";" if ";" in value else ","
    return [item.strip() for item in value.split(separator) if item.strip()]


def make_cache_row(
    args: argparse.Namespace,
    *,
    task: str,
    seed: int,
    candidate_index: int,
    episode_info: dict[str, Any],
    expert: dict[str, Any],
    episode_index: int,
) -> dict[str, Any]:
    prompt = generate_seen_instruction_deterministic(
        task,
        episode_info,
        max_descriptions=args.max_descriptions,
        selection_index=episode_index,
    )
    return {
        "task": task,
        "task_config": args.task_config,
        "episode_index": episode_index,
        "seed": int(seed),
        "prompt": prompt,
        "episode_info": episode_info,
        "candidate_index": int(candidate_index),
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


def _parallel_worker_init(task_config: str, gpu_queue: Any | None) -> None:
    os.environ[ROBOTWIN_TASK_CONFIG_ENV] = str(task_config)
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    if gpu_queue is None:
        return
    try:
        gpu = str(gpu_queue.get_nowait()).strip()
    except (queue.Empty, EOFError, OSError):
        gpu = ""
    except Exception:
        gpu = ""
    if gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu


def _check_candidate_worker(task: str, task_config: str, seed: int, candidate_index: int) -> dict[str, Any]:
    os.environ[ROBOTWIN_TASK_CONFIG_ENV] = str(task_config)
    expert = check_robotwin_expert_seed(task, int(seed))
    return {
        "task": task,
        "seed": int(seed),
        "candidate_index": int(candidate_index),
        "expert": expert,
        "worker_pid": os.getpid(),
        "worker_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }


def _rejected_result_from_exception(task: str, seed: int, candidate_index: int, exc: BaseException) -> dict[str, Any]:
    return {
        "task": task,
        "seed": int(seed),
        "candidate_index": int(candidate_index),
        "expert": {
            "accepted": False,
            "error_type": exc.__class__.__name__,
            "error": str(exc),
            "duration": 0.0,
        },
        "worker_pid": 0,
        "worker_cuda_visible_devices": "",
    }


def commit_candidate_result(
    args: argparse.Namespace,
    *,
    task: str,
    cache_path: Path,
    rows: list[dict[str, Any]],
    seen_seeds: set[int],
    result: dict[str, Any],
) -> bool:
    seed = int(result["seed"])
    candidate_index = int(result["candidate_index"])
    expert = result.get("expert") if isinstance(result.get("expert"), dict) else {}
    if not bool(expert.get("accepted", False)):
        append_status(
            args.status_jsonl,
            "candidate_rejected",
            task=task,
            seed=seed,
            candidate_index=candidate_index,
            error_type=expert.get("error_type", ""),
            error=expert.get("error", ""),
            worker_pid=result.get("worker_pid", 0),
            worker_cuda_visible_devices=result.get("worker_cuda_visible_devices", ""),
        )
        return False
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
            worker_pid=result.get("worker_pid", 0),
            worker_cuda_visible_devices=result.get("worker_cuda_visible_devices", ""),
        )
        return False
    row = make_cache_row(
        args,
        task=task,
        seed=seed,
        candidate_index=candidate_index,
        episode_info=episode_info,
        expert=expert,
        episode_index=len(rows),
    )
    rows.append(row)
    seen_seeds.add(seed)
    write_rows_atomic(cache_path, rows)
    append_status(
        args.status_jsonl,
        "candidate_accepted",
        task=task,
        seed=seed,
        candidate_index=candidate_index,
        episode_index=row["episode_index"],
        rows=len(rows),
        worker_pid=result.get("worker_pid", 0),
        worker_cuda_visible_devices=result.get("worker_cuda_visible_devices", ""),
    )
    return True


def build_task_cache_serial(
    args: argparse.Namespace,
    *,
    task: str,
    cache_path: Path,
    rows: list[dict[str, Any]],
    seen_seeds: set[int],
    next_candidate_index: int,
) -> tuple[int, int]:
    generated = 0
    checked = 0
    for candidate_index in range(next_candidate_index, args.max_candidates):
        if len(rows) >= args.episodes_per_task:
            break
        seed = int(args.seed_start + candidate_index)
        if seed in seen_seeds:
            continue
        checked += 1
        result = _check_candidate_worker(task, args.task_config, seed, candidate_index)
        generated += int(
            commit_candidate_result(
                args,
                task=task,
                cache_path=cache_path,
                rows=rows,
                seen_seeds=seen_seeds,
                result=result,
            )
        )
    return generated, checked


def build_task_cache_parallel(
    args: argparse.Namespace,
    *,
    task: str,
    cache_path: Path,
    rows: list[dict[str, Any]],
    seen_seeds: set[int],
    next_candidate_index: int,
) -> tuple[int, int]:
    generated = 0
    checked = 0
    num_workers = max(int(args.num_workers), 1)
    prefetch_window = int(args.prefetch_window) if int(args.prefetch_window) > 0 else num_workers
    prefetch_window = max(prefetch_window, num_workers)
    worker_gpus = parse_worker_gpus(args.worker_gpus)
    manager: multiprocessing.managers.SyncManager | None = None
    gpu_queue = None
    if worker_gpus:
        manager = multiprocessing.Manager()
        gpu_queue = manager.Queue()
        for index in range(num_workers):
            gpu_queue.put(worker_gpus[index % len(worker_gpus)])

    append_status(
        args.status_jsonl,
        "parallel_start",
        task=task,
        num_workers=num_workers,
        prefetch_window=prefetch_window,
        worker_gpus=worker_gpus,
    )
    mp_context = multiprocessing.get_context("spawn")
    executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=mp_context,
        initializer=_parallel_worker_init,
        initargs=(args.task_config, gpu_queue),
    )
    futures: dict[concurrent.futures.Future, tuple[int, int]] = {}
    completed: dict[int, dict[str, Any]] = {}
    next_submit = int(next_candidate_index)
    next_commit = int(next_candidate_index)

    def submit_more() -> None:
        nonlocal next_submit
        while (
            len(futures) < prefetch_window
            and next_submit < int(args.max_candidates)
            and len(rows) < int(args.episodes_per_task)
        ):
            seed = int(args.seed_start + next_submit)
            candidate_index = int(next_submit)
            next_submit += 1
            if seed in seen_seeds:
                completed[candidate_index] = {
                    "task": task,
                    "seed": seed,
                    "candidate_index": candidate_index,
                    "expert": {"accepted": False, "error_type": "DuplicateSeed", "error": "seed already cached"},
                    "worker_pid": 0,
                    "worker_cuda_visible_devices": "",
                }
                continue
            future = executor.submit(_check_candidate_worker, task, args.task_config, seed, candidate_index)
            futures[future] = (candidate_index, seed)

    try:
        submit_more()
        while len(rows) < args.episodes_per_task and (futures or completed):
            if futures:
                done, _pending = concurrent.futures.wait(
                    futures,
                    timeout=1.0,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    candidate_index, seed = futures.pop(future)
                    try:
                        completed[candidate_index] = future.result()
                    except BaseException as exc:
                        completed[candidate_index] = _rejected_result_from_exception(
                            task,
                            seed,
                            candidate_index,
                            exc,
                        )
            while next_commit in completed and len(rows) < args.episodes_per_task:
                result = completed.pop(next_commit)
                checked += 1
                generated += int(
                    commit_candidate_result(
                        args,
                        task=task,
                        cache_path=cache_path,
                        rows=rows,
                        seen_seeds=seen_seeds,
                        result=result,
                    )
                )
                next_commit += 1
            submit_more()
            if next_submit >= int(args.max_candidates) and not futures:
                while next_commit in completed and len(rows) < args.episodes_per_task:
                    result = completed.pop(next_commit)
                    checked += 1
                    generated += int(
                        commit_candidate_result(
                            args,
                            task=task,
                            cache_path=cache_path,
                            rows=rows,
                            seen_seeds=seen_seeds,
                            result=result,
                        )
                    )
                    next_commit += 1
                break
    finally:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        if manager is not None:
            manager.shutdown()
    return generated, checked


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
        num_workers=args.num_workers,
        prefetch_window=args.prefetch_window,
    )
    if int(args.num_workers) <= 1:
        generated, checked = build_task_cache_serial(
            args,
            task=task,
            cache_path=cache_path,
            rows=rows,
            seen_seeds=seen_seeds,
            next_candidate_index=next_candidate_index,
        )
    else:
        generated, checked = build_task_cache_parallel(
            args,
            task=task,
            cache_path=cache_path,
            rows=rows,
            seen_seeds=seen_seeds,
            next_candidate_index=next_candidate_index,
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
