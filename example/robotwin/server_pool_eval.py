#!/usr/bin/env python3
"""Persistent DreamZero server pool with a FIFO RoboTwin task queue."""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = Path("/data/checkpoints/dreamzero/robotwin_eval_runs/server_pool_task_queue")
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


def trim(value: str) -> str:
    return value.strip()


def parse_groups(raw: str, *, split_mode: str) -> list[str]:
    if not raw:
        return []
    separator = ";" if split_mode == "semicolon" or (split_mode == "auto" and ";" in raw) else ","
    return [item for item in (trim(part) for part in raw.split(separator)) if item]


def gpu_count(group: str) -> int:
    return len([item for item in group.split(",") if item.strip()])


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
        raise ValueError("TASK/TASKS resolved to an empty task list")
    if len(parts) == 1 and parts[0].lower() == "all":
        return load_all_eval_tasks()
    if any(item.lower() == "all" for item in parts):
        raise ValueError("Use TASKS=all by itself, or provide explicit task names")
    return parts


def resolve_ckpt(path: str) -> Path:
    ckpt = Path(path).expanduser().resolve()
    nested = ckpt / "checkpoint-30000"
    if nested.is_dir() and not (ckpt / "config.json").exists():
        ckpt = nested
    return ckpt


def port_accepting(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def require_port_free(host: str, port: int) -> None:
    if port_accepting(host, port, timeout=1.0):
        raise RuntimeError(f"Port {host}:{port} is already accepting connections")


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


def probe_metadata(args: argparse.Namespace, *, port: int) -> dict[str, Any]:
    code = r"""
import json
import sys
from eval_utils.policy_client import WebsocketClientPolicy

client = WebsocketClientPolicy(
    host=sys.argv[1],
    port=int(sys.argv[2]),
    log_wait=False,
    open_timeout=float(sys.argv[3]),
)
try:
    print(json.dumps(client.get_server_metadata(), ensure_ascii=False))
finally:
    client.close()
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = pythonpath_env(env)
    completed = subprocess.run(
        [args.dreamzero_python, "-c", code, args.host, str(port), "5"],
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
        check=True,
    )
    return json.loads(completed.stdout)


def reset_server(args: argparse.Namespace, *, port: int, label: str) -> None:
    code = r"""
import sys
from eval_utils.policy_client import WebsocketClientPolicy

client = WebsocketClientPolicy(
    host=sys.argv[1],
    port=int(sys.argv[2]),
    log_wait=False,
    open_timeout=float(sys.argv[3]),
)
try:
    client.reset({"reset_all": True, "session_id": sys.argv[4], "prompt": ""})
finally:
    client.close()
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = pythonpath_env(env)
    subprocess.run(
        [
            args.dreamzero_python,
            "-c",
            code,
            args.host,
            str(port),
            "5",
            f"server-pool-{label}-{uuid.uuid4().hex[:8]}",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=True,
    )


@dataclass(frozen=True)
class TaskItem:
    index: int
    task: str


@dataclass
class TaskResult:
    task_index: int
    task: str
    slot_id: int
    status: str
    started_at: float
    finished_at: float
    output_dir: str
    log_path: str
    report_path: str = ""
    episodes: int = 0
    successes: int = 0
    success_rate: float = 0.0
    episode_errors: int = 0
    exit_code: int | None = None
    error: str = ""
    server_restarts: int = 0
    server_timing: dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        return {
            "task_index": self.task_index,
            "task": self.task,
            "slot_id": self.slot_id,
            "status": self.status,
            "episodes": self.episodes,
            "successes": self.successes,
            "success_rate": self.success_rate,
            "episode_errors": self.episode_errors,
            "exit_code": self.exit_code,
            "duration": max(self.finished_at - self.started_at, 0.0),
            "server_restarts": self.server_restarts,
            "server_timing": self.server_timing,
            "error": self.error,
            "output_dir": self.output_dir,
            "report": self.report_path,
            "log_path": self.log_path,
        }


@dataclass
class SlotState:
    slot_id: int
    server_gpus: str
    client_gpus: str
    port: int
    master_port: int
    server_nproc: int
    process: subprocess.Popen | None = None
    controller_process: subprocess.Popen | None = None
    restart_count: int = 0
    has_started_server: bool = False
    dead: bool = False
    server_metadata: dict[str, Any] = field(default_factory=dict)
    server_timing: dict[str, Any] = field(default_factory=dict)


class ServerPoolRunner:
    def __init__(self, args: argparse.Namespace, tasks: list[str], server_slots: list[str], client_slots: list[str]):
        self.args = args
        self.tasks = [TaskItem(index=i, task=task) for i, task in enumerate(tasks)]
        self.output_dir = args.output_dir.expanduser().resolve()
        self.log_dir = self.output_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.status_path = self.log_dir / "status.jsonl"
        self.status_lock = threading.Lock()
        self.results_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.results: list[TaskResult] = []
        self.task_queue: queue.Queue[TaskItem] = queue.Queue()
        for item in self.tasks:
            self.task_queue.put(item)
        self.slots = [
            SlotState(
                slot_id=i,
                server_gpus=server_slots[i],
                client_gpus=client_slots[i],
                port=args.start_port + i,
                master_port=args.start_master_port + i,
                server_nproc=gpu_count(server_slots[i]),
            )
            for i in range(len(server_slots))
        ]

    def log_status(self, event: str, **payload: Any) -> None:
        row = {"timestamp": time.time(), "event": event, **payload}
        line = json.dumps(row, ensure_ascii=False, default=str)
        with self.status_lock:
            with self.status_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        print(f"[pool] {event} {json.dumps(payload, ensure_ascii=False, default=str)}", flush=True)

    def server_alive(self, slot: SlotState) -> bool:
        return slot.process is not None and slot.process.poll() is None

    def start_server(self, slot: SlotState, *, reason: str) -> bool:
        if self.args.dry_run_actions:
            return True
        if slot.has_started_server:
            slot.restart_count += 1
        slot.has_started_server = True
        if slot.restart_count > self.args.server_restart_limit:
            slot.dead = True
            self.log_status(
                "slot_dead",
                slot_id=slot.slot_id,
                reason=f"restart limit exceeded during {reason}",
                restart_count=slot.restart_count,
            )
            return False

        require_port_free(self.args.host, slot.port)
        require_port_free("127.0.0.1", slot.master_port)
        server_log_path = self.log_dir / f"server_slot_{slot.slot_id}.log"
        server_output_root = self.output_dir / "server_outputs" / f"slot_{slot.slot_id}"
        server_output_root.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.args.dreamzero_torchrun,
            "--nnodes",
            "1",
            "--nproc_per_node",
            str(slot.server_nproc),
            "--master_addr",
            "127.0.0.1",
            "--master_port",
            str(slot.master_port),
            "example/robotwin/robotwin_microbatch_server.py"
            if self.args.microbatch
            else "socket_test_optimized_aloha_x5lite_bimanual.py",
            "--model_path",
            str(self.args.ckpt),
            "--host",
            self.args.server_host,
            "--port",
            str(slot.port),
            "--max-chunk-size",
            str(self.args.max_chunk_size),
            "--output-root",
            str(server_output_root),
        ]
        if self.args.microbatch:
            cmd.extend(
                [
                    "--microbatch-wait-ms",
                    str(self.args.microbatch_wait_ms),
                    "--microbatch-max-size",
                    str(self.args.microbatch_max_size),
                ]
            )
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = slot.server_gpus
        env["PYTHONPATH"] = pythonpath_env(env)
        log_file = server_log_path.open("ab", buffering=0)
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
        self.log_status(
            "server_start",
            slot_id=slot.slot_id,
            pid=slot.process.pid if slot.process else None,
            port=slot.port,
            master_port=slot.master_port,
            server_gpus=slot.server_gpus,
            restart_count=slot.restart_count,
            reason=reason,
            log_path=str(server_log_path),
        )
        return self.wait_server_ready(slot)

    def wait_server_ready(self, slot: SlotState) -> bool:
        deadline = time.time() + self.args.server_timeout
        last_error = ""
        while time.time() < deadline and not self.stop_event.is_set():
            if not self.server_alive(slot):
                last_error = f"server process exited with code {slot.process.poll() if slot.process else None}"
                break
            try:
                metadata = probe_metadata(self.args, port=slot.port)
                slot.server_metadata = metadata
                raw_timing = metadata.get("server_timing", {})
                slot.server_timing = raw_timing if isinstance(raw_timing, dict) else {}
                self.log_status(
                    "server_ready",
                    slot_id=slot.slot_id,
                    port=slot.port,
                    server_timing=slot.server_timing,
                )
                return True
            except Exception as exc:
                last_error = f"{exc.__class__.__name__}: {exc}"
                time.sleep(5)
        self.log_status("server_ready_failed", slot_id=slot.slot_id, port=slot.port, error=last_error)
        terminate_process_group(slot.process)
        slot.process = None
        return False

    def ensure_server(self, slot: SlotState, *, reason: str) -> bool:
        if self.args.dry_run_actions:
            return True
        if self.server_alive(slot):
            try:
                metadata = probe_metadata(self.args, port=slot.port)
                slot.server_metadata = metadata
                raw_timing = metadata.get("server_timing", {})
                slot.server_timing = raw_timing if isinstance(raw_timing, dict) else slot.server_timing
                return True
            except Exception as exc:
                self.log_status("server_probe_failed", slot_id=slot.slot_id, error=f"{exc.__class__.__name__}: {exc}")
        terminate_process_group(slot.process)
        slot.process = None
        while not slot.dead and not self.stop_event.is_set():
            try:
                if self.start_server(slot, reason=reason):
                    return True
            except Exception as exc:
                self.log_status("server_start_failed", slot_id=slot.slot_id, error=f"{exc.__class__.__name__}: {exc}")
                terminate_process_group(slot.process)
                slot.process = None
                if slot.restart_count > self.args.server_restart_limit:
                    slot.dead = True
                    break
            terminate_process_group(slot.process)
            slot.process = None
        return False

    def reset_slot_server(self, slot: SlotState, *, label: str) -> bool:
        if self.args.dry_run_actions:
            return True
        try:
            reset_server(self.args, port=slot.port, label=f"slot-{slot.slot_id}-{label}")
            return True
        except Exception as exc:
            self.log_status("server_reset_failed", slot_id=slot.slot_id, label=label, error=f"{exc.__class__.__name__}: {exc}")
            terminate_process_group(slot.process)
            slot.process = None
            return self.ensure_server(slot, reason=f"reset_failed:{label}")

    def controller_command(self, slot: SlotState, task: TaskItem, task_output: Path) -> list[str]:
        eval_mode = "microbatch" if self.args.microbatch else "lingbot"
        num_envs = self.args.sessions_per_server if self.args.microbatch else 1
        cmd = [
            self.args.robotwin_python,
            "example/robotwin/parallel_eval.py",
            "--remote-host",
            self.args.host,
            "--remote-port",
            str(slot.port),
            "--tasks",
            task.task,
            "--episodes",
            str(self.args.episodes),
            "--num-envs",
            str(num_envs),
            "--eval-mode",
            eval_mode,
            "--env-cuda",
            slot.client_gpus,
            "--worker-python",
            self.args.robotwin_python,
            "--output-dir",
            str(task_output),
            "--task-config",
            self.args.task_config,
            "--episode-length",
            str(self.args.episode_length),
            "--max-steps",
            str(self.args.max_steps),
            "--seed-start",
            str(self.args.seed_start),
            "--open-loop-horizon",
            str(self.args.open_loop_horizon),
            "--reset-retries",
            str(self.args.reset_retries),
            "--expert-filter-max-candidates",
            str(self.args.expert_filter_max_candidates),
            "--checkpoint-label",
            "dreamzero_robotwin_server_pool",
            "--checkpoint-path",
            str(self.args.ckpt),
            "--client-image-resolution",
            self.args.client_image_resolution,
            "--progress",
            self.args.progress,
            "--progress-interval",
            str(self.args.progress_interval),
            "--worker-timeout",
            str(self.args.worker_timeout),
        ]
        if self.args.microbatch:
            cmd.extend(["--use-seed-cache", "--seed-cache-root", str(self.args.seed_cache_root)])
        if self.args.profile:
            cmd.append("--profile")
        if self.args.save_video:
            cmd.extend(["--save-video", "--video-fps", str(self.args.video_fps)])
        if self.args.dry_run_actions:
            cmd.append("--dry-run-actions")
        return cmd

    def run_controller(self, slot: SlotState, task: TaskItem, task_output: Path, log_path: Path) -> int:
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = slot.client_gpus
        env["PYTHONPATH"] = pythonpath_env(env)
        env["ROBOTWIN_TASK_CONFIG"] = self.args.task_config
        if self.args.microbatch:
            env["USE_SEED_CACHE"] = "1"
            env["SEED_CACHE_ROOT"] = str(self.args.seed_cache_root)
        cmd = self.controller_command(slot, task, task_output)
        log_file = log_path.open("ab", buffering=0)
        try:
            slot.controller_process = subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            return slot.controller_process.wait()
        finally:
            log_file.close()
            slot.controller_process = None

    def read_task_report(self, task_output: Path) -> dict[str, Any]:
        report_path = task_output / "report.json"
        if not report_path.exists():
            return {}
        return json.loads(report_path.read_text(encoding="utf-8"))

    def count_episode_errors(self, task_output: Path) -> int:
        errors = 0
        for path in task_output.glob("*/episode_*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                errors += 1
                continue
            if str(data.get("error", "") or ""):
                errors += 1
        return errors

    def run_task(self, slot: SlotState, task: TaskItem) -> TaskResult:
        task_output = self.output_dir / f"{task.index:03d}_{task.task}"
        task_output.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f"controller_{task.index:03d}_{task.task}.log"
        started_at = time.time()
        self.log_status("task_start", slot_id=slot.slot_id, task_index=task.index, task=task.task, output_dir=str(task_output))

        if not self.ensure_server(slot, reason=f"task_start:{task.task}"):
            finished_at = time.time()
            return TaskResult(
                task_index=task.index,
                task=task.task,
                slot_id=slot.slot_id,
                status="failed",
                started_at=started_at,
                finished_at=finished_at,
                output_dir=str(task_output),
                log_path=str(log_path),
                exit_code=None,
                error="server unavailable before task",
                server_restarts=slot.restart_count,
                server_timing=slot.server_timing,
            )

        if not self.reset_slot_server(slot, label=f"pre-{task.task}"):
            finished_at = time.time()
            return TaskResult(
                task_index=task.index,
                task=task.task,
                slot_id=slot.slot_id,
                status="failed",
                started_at=started_at,
                finished_at=finished_at,
                output_dir=str(task_output),
                log_path=str(log_path),
                exit_code=None,
                error="server reset failed before task",
                server_restarts=slot.restart_count,
                server_timing=slot.server_timing,
            )
        exit_code = self.run_controller(slot, task, task_output, log_path)
        self.reset_slot_server(slot, label=f"post-{task.task}")

        report = self.read_task_report(task_output)
        episode_errors = self.count_episode_errors(task_output)
        rows = report.get("rows", []) if isinstance(report, dict) else []
        row = rows[0] if rows else {}
        finished_at = time.time()
        status = "ok" if exit_code == 0 and report and episode_errors == 0 else "failed"
        error_parts = []
        if exit_code != 0:
            error_parts.append(f"controller exit_code={exit_code}")
        if not report:
            error_parts.append("missing report.json")
        if episode_errors:
            error_parts.append(f"episode_errors={episode_errors}")
        result = TaskResult(
            task_index=task.index,
            task=task.task,
            slot_id=slot.slot_id,
            status=status,
            started_at=started_at,
            finished_at=finished_at,
            output_dir=str(task_output),
            log_path=str(log_path),
            report_path=str(task_output / "report.json") if report else "",
            episodes=int(row.get("episodes", report.get("episodes", 0) if isinstance(report, dict) else 0) or 0),
            successes=int(row.get("successes", report.get("successes", 0) if isinstance(report, dict) else 0) or 0),
            success_rate=float(row.get("success_rate", report.get("success_rate", 0.0) if isinstance(report, dict) else 0.0) or 0.0),
            episode_errors=episode_errors,
            exit_code=exit_code,
            error="; ".join(error_parts),
            server_restarts=slot.restart_count,
            server_timing=slot.server_timing,
        )
        self.log_status("task_done", **result.as_row())
        return result

    def slot_loop(self, slot: SlotState) -> None:
        try:
            if not self.ensure_server(slot, reason="initial"):
                return
            while not self.stop_event.is_set() and not slot.dead:
                try:
                    task = self.task_queue.get_nowait()
                except queue.Empty:
                    return
                result = self.run_task(slot, task)
                with self.results_lock:
                    self.results.append(result)
                if result.error and not self.args.dry_run_actions and not self.server_alive(slot):
                    self.ensure_server(slot, reason=f"after_failed_task:{task.task}")
        except Exception as exc:
            slot.dead = True
            self.log_status("slot_error", slot_id=slot.slot_id, error=f"{exc.__class__.__name__}: {exc}")
        finally:
            self.log_status("slot_exit", slot_id=slot.slot_id, dead=slot.dead, restart_count=slot.restart_count)

    def cleanup(self) -> None:
        self.stop_event.set()
        for slot in self.slots:
            terminate_process_group(slot.controller_process)
            terminate_process_group(slot.process)
            slot.controller_process = None
            slot.process = None

    def write_reports(self) -> dict[str, Any]:
        rows = [result.as_row() for result in sorted(self.results, key=lambda item: item.task_index)]
        completed_indices = {result.task_index for result in self.results}
        while True:
            try:
                task = self.task_queue.get_nowait()
            except queue.Empty:
                break
            if task.index in completed_indices:
                continue
            rows.append(
                TaskResult(
                    task_index=task.index,
                    task=task.task,
                    slot_id=-1,
                    status="failed",
                    started_at=time.time(),
                    finished_at=time.time(),
                    output_dir=str(self.output_dir / f"{task.index:03d}_{task.task}"),
                    log_path="",
                    error="not started because all server slots exited",
                ).as_row()
            )
        rows.sort(key=lambda item: int(item["task_index"]))
        episodes = sum(int(row.get("episodes", 0) or 0) for row in rows)
        successes = sum(int(row.get("successes", 0) or 0) for row in rows)
        failed_tasks = sum(1 for row in rows if row.get("status") != "ok")
        summary = {
            "mode": "server_pool_microbatch_task_queue" if self.args.microbatch else "server_pool_task_queue",
            "output_root": str(self.output_dir),
            "tasks_expected": len(self.tasks),
            "tasks_finished": len(rows),
            "tasks_failed": failed_tasks,
            "episodes": episodes,
            "successes": successes,
            "success_rate": successes / episodes if episodes else 0.0,
            "microbatch": {
                "enabled": bool(self.args.microbatch),
                "sessions_per_server": int(self.args.sessions_per_server),
                "microbatch_wait_ms": float(self.args.microbatch_wait_ms),
                "microbatch_max_size": int(self.args.microbatch_max_size),
                "seed_cache_root": str(self.args.seed_cache_root),
            },
            "slots": [
                {
                    "slot_id": slot.slot_id,
                    "server_gpus": slot.server_gpus,
                    "client_gpus": slot.client_gpus,
                    "port": slot.port,
                    "master_port": slot.master_port,
                    "restart_count": slot.restart_count,
                    "dead": slot.dead,
                    "server_timing": slot.server_timing,
                }
                for slot in self.slots
            ],
            "rows": rows,
        }
        for name in ("server_pool_report.json", "task_parallel_report.json"):
            (self.output_dir / name).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        fieldnames = [
            "task_index",
            "task",
            "slot_id",
            "status",
            "episodes",
            "successes",
            "success_rate",
            "episode_errors",
            "exit_code",
            "duration",
            "server_restarts",
            "error",
            "output_dir",
            "report",
            "log_path",
        ]
        with (self.output_dir / "server_pool_report.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
        return summary

    def run(self) -> int:
        self.log_status(
            "pool_start",
            tasks=len(self.tasks),
            slots=len(self.slots),
            output_dir=str(self.output_dir),
            dry_run_actions=self.args.dry_run_actions,
            microbatch=self.args.microbatch,
            sessions_per_server=self.args.sessions_per_server,
        )
        threads = [
            threading.Thread(target=self.slot_loop, args=(slot,), name=f"server-pool-slot-{slot.slot_id}")
            for slot in self.slots
        ]
        try:
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        finally:
            self.cleanup()
        summary = self.write_reports()
        self.log_status("pool_done", tasks_failed=summary["tasks_failed"], episodes=summary["episodes"])
        return 1 if summary["tasks_failed"] else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default=os.environ.get("TASKS", os.environ.get("TASK", "all")))
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)))
    parser.add_argument("--ckpt", type=Path, default=resolve_ckpt(os.environ.get("CKPT", str(DEFAULT_CKPT))))
    parser.add_argument("--episodes", type=int, default=env_int("EPISODES", 100))
    parser.add_argument("--seed-start", type=int, default=env_int("SEED_START", 10000))
    parser.add_argument("--episode-length", type=int, default=env_int("EPISODE_LENGTH", 0))
    parser.add_argument("--max-steps", type=int, default=env_int("MAX_STEPS", 0))
    parser.add_argument("--open-loop-horizon", type=int, default=env_int("OPEN_LOOP_HORIZON", 24))
    parser.add_argument("--reset-retries", type=int, default=env_int("RESET_RETRIES", 5))
    parser.add_argument("--expert-filter-max-candidates", type=int, default=env_int("EXPERT_FILTER_MAX_CANDIDATES", 1000))
    parser.add_argument("--max-chunk-size", type=int, default=env_int("MAX_CHUNK_SIZE", 24))
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=env_bool("SAVE_VIDEO", False))
    parser.add_argument("--video-fps", type=float, default=env_float("VIDEO_FPS", 10.0))
    parser.add_argument("--profile", action=argparse.BooleanOptionalAction, default=env_bool("PROFILE", False))
    parser.add_argument("--client-image-resolution", default=os.environ.get("CLIENT_IMAGE_RESOLUTION", "none"))
    parser.add_argument("--task-config", default=os.environ.get("ROBOTWIN_TASK_CONFIG", os.environ.get("TASK_CONFIG", "demo_clean")))
    parser.add_argument("--progress", choices=("plain", "tqdm", "none"), default=os.environ.get("PROGRESS", os.environ.get("ROBOTWIN_PROGRESS", "plain")))
    parser.add_argument("--progress-interval", type=float, default=env_float("ROBOTWIN_PROGRESS_INTERVAL", 30.0))
    parser.add_argument("--worker-timeout", type=float, default=env_float("WORKER_TIMEOUT", 900.0))
    parser.add_argument("--dry-run-actions", action=argparse.BooleanOptionalAction, default=env_bool("DRY_RUN_ACTIONS", False))
    parser.add_argument("--microbatch", action=argparse.BooleanOptionalAction, default=env_bool("MICROBATCH", False))
    parser.add_argument("--sessions-per-server", type=int, default=env_int("SESSIONS_PER_SERVER", 4))
    parser.add_argument("--microbatch-wait-ms", type=float, default=env_float("MICROBATCH_WAIT_MS", 10.0))
    parser.add_argument("--microbatch-max-size", type=int, default=env_int("MICROBATCH_MAX_SIZE", 4))
    parser.add_argument("--seed-cache-root", type=Path, default=Path(os.environ.get("SEED_CACHE_ROOT", "")) if os.environ.get("SEED_CACHE_ROOT") else None)
    parser.add_argument("--server-gpus", default=os.environ.get("SERVER_GPUS", os.environ.get("SERVER_GPU", "")))
    parser.add_argument("--server-gpu-groups", default=os.environ.get("SERVER_GPU_GROUPS", ""))
    parser.add_argument("--client-gpus", default=os.environ.get("CLIENT_GPUS", os.environ.get("CLIENT_GPU", os.environ.get("ENV_GPU", ""))))
    parser.add_argument("--client-gpu-groups", default=os.environ.get("CLIENT_GPU_GROUPS", ""))
    parser.add_argument("--start-port", type=int, default=env_int("START_PORT", 8200))
    parser.add_argument("--start-master-port", type=int, default=env_int("START_MASTER_PORT", 29700))
    parser.add_argument("--server-host", default=os.environ.get("SERVER_HOST", "0.0.0.0"))
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--server-timeout", type=float, default=env_float("SERVER_TIMEOUT", 1800.0))
    parser.add_argument("--server-restart-limit", type=int, default=env_int("SERVER_RESTART_LIMIT", 3))
    parser.add_argument("--dreamzero-torchrun", default=os.environ.get("DREAMZERO_TORCHRUN", "/data/dreamzero/.venv/bin/torchrun"))
    parser.add_argument("--dreamzero-python", default=os.environ.get("DREAMZERO_PYTHON", "/data/dreamzero/.venv/bin/python"))
    robotwin_env = os.environ.get("ROBOTWIN_ENV", "/data/envs/robotwin310")
    parser.add_argument("--robotwin-python", default=os.environ.get("ROBOTWIN_PYTHON", str(Path(robotwin_env) / "bin" / "python")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.ckpt = resolve_ckpt(str(args.ckpt))
    if not args.dry_run_actions and not args.ckpt.is_dir():
        raise FileNotFoundError(f"Missing checkpoint: {args.ckpt}")
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.open_loop_horizon <= 0:
        raise ValueError("--open-loop-horizon must be positive")
    if args.server_restart_limit < 0:
        raise ValueError("--server-restart-limit must be non-negative")
    if args.sessions_per_server <= 0:
        raise ValueError("--sessions-per-server must be positive")
    if args.microbatch_max_size <= 0:
        raise ValueError("--microbatch-max-size must be positive")
    if args.microbatch_wait_ms < 0:
        raise ValueError("--microbatch-wait-ms must be non-negative")
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.seed_cache_root is None:
        args.seed_cache_root = args.output_dir / "expert_seed_cache"
    else:
        args.seed_cache_root = args.seed_cache_root.expanduser().resolve()

    server_slots = parse_groups(args.server_gpu_groups, split_mode="semicolon") if args.server_gpu_groups else parse_groups(args.server_gpus, split_mode="comma")
    client_slots = parse_groups(args.client_gpu_groups, split_mode="semicolon") if args.client_gpu_groups else parse_groups(args.client_gpus, split_mode="comma")
    if not server_slots or not client_slots:
        raise ValueError("SERVER_GPUS/SERVER_GPU_GROUPS and CLIENT_GPUS/CLIENT_GPU_GROUPS are required")
    if len(server_slots) != len(client_slots):
        raise ValueError(f"SERVER and CLIENT slot counts must match: server={len(server_slots)} client={len(client_slots)}")
    if any(gpu_count(group) <= 0 for group in server_slots):
        raise ValueError(f"Invalid server GPU groups: {server_slots}")
    for path in (args.dreamzero_torchrun, args.dreamzero_python, args.robotwin_python):
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing executable: {path}")

    tasks = parse_tasks(args.tasks)
    runner = ServerPoolRunner(args, tasks, server_slots, client_slots)
    raise SystemExit(runner.run())


if __name__ == "__main__":
    main()
