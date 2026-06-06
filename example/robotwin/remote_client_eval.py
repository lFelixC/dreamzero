#!/usr/bin/env python3
"""Launch RoboTwin microbatch client controllers against remote server slots."""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = Path("/data/checkpoints/dreamzero/robotwin_eval_runs/remote_client_microbatch")
DEFAULT_CKPT = Path("/data/checkpoints/dreamzero/dreamzero_robotwin/checkpoint-30000")


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else int(raw)


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else float(raw)


def pythonpath_env(base: dict[str, str]) -> str:
    parts = [
        str(REPO_ROOT),
        str(REPO_ROOT / "third_party" / "RoboTwin"),
        str(REPO_ROOT / "third_party" / "lerobot" / "src"),
        str(REPO_ROOT / "third_party" / "lerobot"),
    ]
    existing = base.get("PYTHONPATH")
    if existing:
        parts.append(existing)
    return ":".join(parts)


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


def parse_endpoints(args: argparse.Namespace) -> list[tuple[str, int]]:
    raw = str(args.server_endpoints or "").strip()
    if raw:
        endpoints: list[tuple[str, int]] = []
        for item in raw.replace(";", ",").split(","):
            item = item.strip()
            if not item:
                continue
            if ":" not in item:
                raise ValueError(f"Server endpoint must be host:port, got {item!r}")
            host, port = item.rsplit(":", 1)
            endpoints.append((host.strip(), int(port)))
        if not endpoints:
            raise ValueError("--server-endpoints resolved to no endpoints")
        return endpoints

    if not args.server_host:
        raise ValueError("--server-host is required when --server-endpoints is not set")
    if args.num_servers <= 0:
        raise ValueError("--num-servers must be positive when --server-endpoints is not set")
    return [(str(args.server_host), int(args.start_port) + index) for index in range(args.num_servers)]


def parse_group_list(raw: str) -> list[str]:
    if not raw:
        return []
    separator = ";" if ";" in raw else ","
    return [item.strip() for item in raw.split(separator) if item.strip()]


def parse_client_groups(raw_groups: str, raw_gpus: str, slots: int) -> list[str]:
    explicit = parse_group_list(raw_groups)
    if explicit:
        if len(explicit) != slots:
            raise ValueError(f"CLIENT_GPU_GROUPS has {len(explicit)} groups but server slots={slots}")
        return explicit
    gpus = [item.strip() for item in raw_gpus.split(",") if item.strip()]
    if not gpus:
        raise ValueError("CLIENT_GPUS or CLIENT_GPU_GROUPS is required")
    per_slot = max(len(gpus) // slots, 1)
    groups: list[str] = []
    cursor = 0
    for slot in range(slots):
        remaining_slots = slots - slot
        remaining_gpus = len(gpus) - cursor
        take = max(remaining_gpus // remaining_slots, 1)
        take = max(take, per_slot if remaining_gpus >= per_slot * remaining_slots else take)
        group = gpus[cursor : cursor + take]
        if not group:
            group = [gpus[slot % len(gpus)]]
        groups.append(",".join(group))
        cursor += len(group)
    return groups


def split_tasks(tasks: list[str], slots: int, mode: str) -> list[list[str]]:
    groups = [[] for _ in range(slots)]
    if mode == "contiguous":
        chunk = (len(tasks) + slots - 1) // slots
        for index in range(slots):
            groups[index] = tasks[index * chunk : (index + 1) * chunk]
        return groups
    if mode != "round_robin":
        raise ValueError(f"Unknown split mode: {mode}")
    for index, task in enumerate(tasks):
        groups[index % slots].append(task)
    return groups


def terminate_process_group(proc: subprocess.Popen | None, *, timeout: float = 5.0) -> None:
    if proc is None or proc.poll() is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=timeout)


@dataclass
class SlotLaunch:
    slot_id: int
    host: str
    port: int
    client_gpus: str
    tasks: list[str]
    output_dir: Path
    log_path: Path
    process: subprocess.Popen | None = None
    exit_code: int | None = None


def json_default(value: object) -> object:
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def build_controller_command(args: argparse.Namespace, slot: SlotLaunch) -> list[str]:
    cmd = [
        args.robotwin_python,
        "example/robotwin/parallel_eval.py",
        "--remote-host",
        slot.host,
        "--remote-port",
        str(slot.port),
        "--tasks",
        ",".join(slot.tasks),
        "--episodes",
        str(args.episodes),
        "--num-envs",
        str(args.sessions_per_server),
        "--eval-mode",
        "microbatch",
        "--env-cuda",
        slot.client_gpus,
        "--worker-python",
        args.robotwin_python,
        "--output-dir",
        str(slot.output_dir),
        "--task-config",
        args.task_config,
        "--episode-length",
        str(args.episode_length),
        "--max-steps",
        str(args.max_steps),
        "--open-loop-horizon",
        str(args.open_loop_horizon),
        "--reset-retries",
        str(args.reset_retries),
        "--expert-filter-max-candidates",
        str(args.expert_filter_max_candidates),
        "--checkpoint-label",
        "dreamzero_robotwin_remote_microbatch_client",
        "--checkpoint-path",
        str(args.ckpt),
        "--client-image-resolution",
        args.client_image_resolution,
        "--progress",
        args.progress,
        "--progress-interval",
        str(args.progress_interval),
        "--worker-timeout",
        str(args.worker_timeout),
        "--use-seed-cache",
        "--seed-cache-root",
        str(args.seed_cache_root),
    ]
    if args.profile:
        cmd.append("--profile")
    if args.save_video:
        cmd.extend(["--save-video", "--video-fps", str(args.video_fps)])
    if args.dry_run_actions:
        cmd.append("--dry-run-actions")
    return cmd


def read_slot_report(slot: SlotLaunch) -> dict[str, Any]:
    report_path = slot.output_dir / "report.json"
    if not report_path.exists():
        return {}
    return json.loads(report_path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-endpoints", default=os.environ.get("SERVER_ENDPOINTS", ""))
    parser.add_argument("--server-host", default=os.environ.get("SERVER_HOST", ""))
    parser.add_argument("--num-servers", type=int, default=env_int("NUM_SERVERS", 0))
    parser.add_argument("--start-port", type=int, default=env_int("START_PORT", 8200))
    parser.add_argument("--tasks", default=os.environ.get("TASKS", os.environ.get("TASK", "all")))
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)))
    parser.add_argument("--seed-cache-root", type=Path, default=Path(os.environ.get("SEED_CACHE_ROOT", "")) if os.environ.get("SEED_CACHE_ROOT") else None)
    parser.add_argument("--ckpt", type=Path, default=Path(os.environ.get("CKPT", str(DEFAULT_CKPT))))
    parser.add_argument("--episodes", type=int, default=env_int("EPISODES", 50))
    parser.add_argument("--sessions-per-server", type=int, default=env_int("SESSIONS_PER_SERVER", 8))
    parser.add_argument("--client-gpus", default=os.environ.get("CLIENT_GPUS", ""))
    parser.add_argument("--client-gpu-groups", default=os.environ.get("CLIENT_GPU_GROUPS", ""))
    parser.add_argument("--task-split", choices=("round_robin", "contiguous"), default=os.environ.get("TASK_SPLIT", "round_robin"))
    parser.add_argument("--task-config", default=os.environ.get("ROBOTWIN_TASK_CONFIG", os.environ.get("TASK_CONFIG", "demo_clean")))
    parser.add_argument("--episode-length", type=int, default=env_int("EPISODE_LENGTH", 0))
    parser.add_argument("--max-steps", type=int, default=env_int("MAX_STEPS", 0))
    parser.add_argument("--open-loop-horizon", type=int, default=env_int("OPEN_LOOP_HORIZON", 24))
    parser.add_argument("--reset-retries", type=int, default=env_int("RESET_RETRIES", 5))
    parser.add_argument("--expert-filter-max-candidates", type=int, default=env_int("EXPERT_FILTER_MAX_CANDIDATES", 1000))
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=env_bool("SAVE_VIDEO", False))
    parser.add_argument("--video-fps", type=float, default=env_float("VIDEO_FPS", 10.0))
    parser.add_argument("--profile", action=argparse.BooleanOptionalAction, default=env_bool("PROFILE", True))
    parser.add_argument("--client-image-resolution", default=os.environ.get("CLIENT_IMAGE_RESOLUTION", "none"))
    parser.add_argument("--progress", choices=("plain", "tqdm", "none"), default=os.environ.get("PROGRESS", os.environ.get("ROBOTWIN_PROGRESS", "plain")))
    parser.add_argument("--progress-interval", type=float, default=env_float("ROBOTWIN_PROGRESS_INTERVAL", 30.0))
    parser.add_argument("--worker-timeout", type=float, default=env_float("WORKER_TIMEOUT", 1800.0))
    parser.add_argument("--dry-run-actions", action=argparse.BooleanOptionalAction, default=env_bool("DRY_RUN_ACTIONS", False))
    robotwin_env = os.environ.get("ROBOTWIN_ENV", "/data/envs/robotwin310")
    parser.add_argument("--robotwin-python", default=os.environ.get("ROBOTWIN_PYTHON", str(Path(robotwin_env) / "bin" / "python")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.sessions_per_server <= 0:
        raise ValueError("--sessions-per-server must be positive")
    if args.open_loop_horizon <= 0:
        raise ValueError("--open-loop-horizon must be positive")
    if args.seed_cache_root is None:
        args.seed_cache_root = args.output_dir / "expert_seed_cache"
    args.output_dir = args.output_dir.expanduser().resolve()
    args.seed_cache_root = args.seed_cache_root.expanduser().resolve()
    args.ckpt = args.ckpt.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    endpoints = parse_endpoints(args)
    client_groups = parse_client_groups(args.client_gpu_groups, args.client_gpus, len(endpoints))
    tasks = parse_tasks(args.tasks)
    task_groups = split_tasks(tasks, len(endpoints), args.task_split)
    if not Path(args.robotwin_python).exists():
        raise FileNotFoundError(f"Missing ROBOTWIN_PYTHON executable: {args.robotwin_python}")

    slots: list[SlotLaunch] = []
    for slot_id, ((host, port), client_gpus, slot_tasks) in enumerate(
        zip(endpoints, client_groups, task_groups, strict=True)
    ):
        slot_output = args.output_dir / f"slot_{slot_id:02d}"
        slot_output.mkdir(parents=True, exist_ok=True)
        slots.append(
            SlotLaunch(
                slot_id=slot_id,
                host=host,
                port=port,
                client_gpus=client_gpus,
                tasks=slot_tasks,
                output_dir=slot_output,
                log_path=log_dir / f"controller_slot_{slot_id:02d}.log",
            )
        )

    launch_config = {
        "mode": "remote_client_microbatch_launcher",
        "timestamp": time.time(),
        "argv": sys.argv,
        "tasks": tasks,
        "seed_cache_root": args.seed_cache_root.as_posix(),
        "slots": [
            {
                "slot_id": slot.slot_id,
                "server": f"{slot.host}:{slot.port}",
                "client_gpus": slot.client_gpus,
                "tasks": slot.tasks,
                "output_dir": slot.output_dir.as_posix(),
                "log_path": slot.log_path.as_posix(),
            }
            for slot in slots
        ],
    }
    (args.output_dir / "remote_client_launch_config.json").write_text(
        json.dumps(launch_config, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )

    print(
        f"[remote-client] launching slots={len(slots)} tasks={len(tasks)} output={args.output_dir}",
        flush=True,
    )
    try:
        for slot in slots:
            if not slot.tasks:
                print(f"[remote-client] slot={slot.slot_id} has no tasks; skipping", flush=True)
                slot.exit_code = 0
                continue
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = slot.client_gpus
            env["PYTHONPATH"] = pythonpath_env(env)
            env["ROBOTWIN_TASK_CONFIG"] = str(args.task_config)
            env["USE_SEED_CACHE"] = "1"
            env["SEED_CACHE_ROOT"] = str(args.seed_cache_root)
            cmd = build_controller_command(args, slot)
            slot.log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = slot.log_path.open("ab", buffering=0)
            try:
                slot.process = subprocess.Popen(
                    cmd,
                    cwd=str(REPO_ROOT),
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            finally:
                log_file.close()
            print(
                f"[remote-client] slot={slot.slot_id} pid={slot.process.pid} "
                f"server={slot.host}:{slot.port} client_gpus={slot.client_gpus} tasks={len(slot.tasks)} "
                f"log={slot.log_path}",
                flush=True,
            )

        for slot in slots:
            if slot.process is None:
                continue
            slot.exit_code = slot.process.wait()
            print(f"[remote-client] slot={slot.slot_id} exit_code={slot.exit_code}", flush=True)
    finally:
        for slot in slots:
            terminate_process_group(slot.process)

    rows: list[dict[str, Any]] = []
    for slot in slots:
        report = read_slot_report(slot)
        rows.append(
            {
                "slot_id": slot.slot_id,
                "server": f"{slot.host}:{slot.port}",
                "client_gpus": slot.client_gpus,
                "tasks": len(slot.tasks),
                "exit_code": slot.exit_code,
                "episodes": int(report.get("episodes", 0) or 0) if report else 0,
                "successes": int(report.get("successes", 0) or 0) if report else 0,
                "tasks_finished": int(report.get("tasks_finished", 0) or 0) if report else 0,
                "tasks_complete": int(report.get("tasks_complete", 0) or 0) if report else 0,
                "microbatch_calls": int(report.get("microbatch_calls", 0) or 0) if report else 0,
                "avg_microbatch_size": float(report.get("avg_microbatch_size", 0.0) or 0.0) if report else 0.0,
                "report_path": (slot.output_dir / "report.json").as_posix(),
                "log_path": slot.log_path.as_posix(),
                "task_list": ",".join(slot.tasks),
            }
        )

    failed_slots = sum(1 for row in rows if int(row.get("exit_code") or 0) != 0)
    episodes = sum(int(row["episodes"]) for row in rows)
    successes = sum(int(row["successes"]) for row in rows)
    summary = {
        "mode": "remote_client_microbatch_launcher",
        "output_root": args.output_dir.as_posix(),
        "slots": len(slots),
        "failed_slots": failed_slots,
        "tasks": len(tasks),
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes if episodes else 0.0,
        "rows": rows,
    }
    (args.output_dir / "remote_client_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )
    with (args.output_dir / "remote_client_report.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "slot_id",
            "server",
            "client_gpus",
            "tasks",
            "exit_code",
            "episodes",
            "successes",
            "tasks_finished",
            "tasks_complete",
            "microbatch_calls",
            "avg_microbatch_size",
            "report_path",
            "log_path",
            "task_list",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=json_default))
    raise SystemExit(1 if failed_slots else 0)


if __name__ == "__main__":
    main()
