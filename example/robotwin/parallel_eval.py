#!/usr/bin/env python3
"""LingBot-VA style synchronized-wave RoboTwin eval controller."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import csv
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from multiprocessing.connection import Connection, Listener
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

from eval_utils.policy_client import WebsocketClientPolicy  # noqa: E402
from example.robotwin.robotwin_fast_env import (  # noqa: E402
    DEFAULT_ROBOTWIN_TASK_CONFIG,
    ROBOTWIN_ACTION_DIM,
    ROBOTWIN_TASK_CONFIG_ENV,
    parse_image_resolution,
    robotwin_obs_sequences_to_batched_payload,
    robotwin_obs_sequence_to_payload,
)

from eval_utils import msgpack_numpy  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path("/data/checkpoints/dreamzero/robotwin_eval_runs/lingbot_style_eval")
KEYFRAMES_PER_CHUNK = 9
PROFILE_TIMER_FIELDS = (
    "T_load_model",
    "T_server_ready",
    "T_env_reset",
    "T_expert_seed_filter",
    "T_render_obs",
    "T_send_obs",
    "T_policy_infer",
    "T_recv_action",
    "T_step_chunk",
    "T_write_log/video",
    "T_microbatch_wait",
)


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


@dataclass
class WorkerHandle:
    worker_id: int
    process: subprocess.Popen
    conn: Connection
    log_path: Path
    cuda_visible_devices: str


def format_duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    minutes, sec = divmod(int(seconds), 60)
    hours, minute = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minute:02d}m{sec:02d}s"
    if minute:
        return f"{minute}m{sec:02d}s"
    return f"{sec}s"


class ProgressReporter:
    def __init__(
        self,
        *,
        mode: str,
        total_tasks: int,
        episodes_per_task: int,
        output_dir: Path,
        interval: float,
    ) -> None:
        self.mode = mode
        self.total_tasks = int(total_tasks)
        self.episodes_per_task = int(episodes_per_task)
        self.total_episodes = self.total_tasks * self.episodes_per_task
        self.output_dir = output_dir
        self.interval = max(float(interval), 1.0)
        self.started_at = time.perf_counter()
        self.global_completed = 0
        self.global_success = 0
        self.last_update_at = 0.0
        self.status_lock = threading.Lock()
        self.current_status: str | None = None
        self.current_status_started_at = 0.0
        self.heartbeat_stop = threading.Event()
        self.heartbeat_thread: threading.Thread | None = None
        self._tqdm_cls: Any | None = None
        self.task_bar: Any | None = None
        self.episode_bar: Any | None = None

        if self.mode == "tqdm":
            try:
                from tqdm import tqdm
            except Exception as exc:
                self.mode = "plain"
                self._emit(f"[progress] tqdm unavailable ({exc}); falling back to plain logs")
            else:
                self._tqdm_cls = tqdm
                self.task_bar = tqdm(
                    total=self.total_tasks,
                    desc="tasks",
                    unit="task",
                    dynamic_ncols=True,
                    file=sys.stderr,
                )

        self.log(
            f"enabled mode={self.mode} tasks={self.total_tasks} "
            f"episodes_per_task={self.episodes_per_task} output={self.output_dir}"
        )
        if self.mode != "none":
            self.heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                name="robotwin-progress-heartbeat",
                daemon=True,
            )
            self.heartbeat_thread.start()

    def _emit(self, message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    def log(self, message: str) -> None:
        if self.mode == "none":
            return
        line = f"[progress {time.strftime('%H:%M:%S')}] {message}"
        if self.mode == "tqdm" and self._tqdm_cls is not None:
            self._tqdm_cls.write(line, file=sys.stderr)
        else:
            self._emit(line)

    def _heartbeat_loop(self) -> None:
        while not self.heartbeat_stop.wait(self.interval):
            with self.status_lock:
                status = self.current_status
                started_at = self.current_status_started_at
            if not status:
                continue
            self.log(f"still waiting: {status} elapsed={format_duration(time.perf_counter() - started_at)}")

    def set_status(self, status: str) -> None:
        if self.mode == "none":
            return
        with self.status_lock:
            self.current_status = status
            self.current_status_started_at = time.perf_counter()

    def clear_status(self) -> None:
        if self.mode == "none":
            return
        with self.status_lock:
            self.current_status = None
            self.current_status_started_at = 0.0

    def close(self) -> None:
        self.heartbeat_stop.set()
        if self.heartbeat_thread is not None:
            self.heartbeat_thread.join(timeout=1.0)
            self.heartbeat_thread = None
        if self.episode_bar is not None:
            self.episode_bar.close()
            self.episode_bar = None
        if self.task_bar is not None:
            self.task_bar.close()
            self.task_bar = None

    def task_start(self, *, task_index: int, task: str, episode_length: int, total_waves: int) -> None:
        if self.mode == "tqdm" and self._tqdm_cls is not None:
            if self.episode_bar is not None:
                self.episode_bar.close()
            self.episode_bar = self._tqdm_cls(
                total=self.episodes_per_task,
                desc=f"{task[:28]}",
                unit="ep",
                leave=False,
                dynamic_ncols=True,
                file=sys.stderr,
            )
        self.log(
            f"task {task_index}/{self.total_tasks} start task={task} "
            f"episodes={self.episodes_per_task} episode_length={episode_length} waves={total_waves}"
        )

    def wave_phase(
        self,
        *,
        task_index: int,
        task: str,
        wave_id: int,
        total_waves: int,
        phase: str,
        detail: str,
    ) -> None:
        self.log(f"task {task_index}/{self.total_tasks} {task} wave {wave_id + 1}/{total_waves} {phase}: {detail}")

    def wave_step(
        self,
        *,
        task_index: int,
        task: str,
        wave_id: int,
        total_waves: int,
        infer_index: int,
        max_infer: int,
        active: int,
        batch_size: int,
        successes: int,
        errors: int,
        infer_time: float,
        wave_elapsed: float,
        force: bool = False,
    ) -> None:
        now = time.perf_counter()
        if not force and now - self.last_update_at < self.interval:
            return
        self.last_update_at = now
        postfix = (
            f"task={task} wave={wave_id + 1}/{total_waves} infer={infer_index}/{max_infer} "
            f"active={active}/{batch_size} success={successes}/{batch_size} errors={errors} "
            f"infer={infer_time:.2f}s elapsed={format_duration(wave_elapsed)}"
        )
        if self.mode == "tqdm" and self.episode_bar is not None:
            self.episode_bar.set_postfix_str(
                f"wave {wave_id + 1}/{total_waves} infer {infer_index}/{max_infer} "
                f"active {active}/{batch_size} ok {successes}/{batch_size}"
            )
            return
        self.log(f"task {task_index}/{self.total_tasks} {postfix}")

    def wave_done(
        self,
        *,
        task_index: int,
        task: str,
        wave_id: int,
        total_waves: int,
        results: list[dict[str, Any]],
        task_completed: int,
        wave_elapsed: float,
    ) -> None:
        completed = len(results)
        successes = sum(1 for item in results if bool(item.get("success")))
        errors = sum(1 for item in results if item.get("error"))
        self.global_completed += completed
        self.global_success += successes
        if self.episode_bar is not None and completed:
            self.episode_bar.update(completed)

        elapsed = time.perf_counter() - self.started_at
        rate = self.global_completed / elapsed if elapsed > 0 and self.global_completed else 0.0
        remaining = max(self.total_episodes - self.global_completed, 0)
        eta = remaining / rate if rate > 0 else 0.0
        success_rate = successes / completed if completed else 0.0
        self.log(
            f"task {task_index}/{self.total_tasks} {task} wave {wave_id + 1}/{total_waves} done "
            f"episodes={completed} success={successes}/{completed} ({success_rate:.1%}) errors={errors} "
            f"task_progress={task_completed}/{self.episodes_per_task} "
            f"global={self.global_completed}/{self.total_episodes} "
            f"speed={rate * 60:.2f} ep/min eta={format_duration(eta)} "
            f"wave_elapsed={format_duration(wave_elapsed)}"
        )

    def task_done(self, *, task_index: int, task: str, results: list[dict[str, Any]], elapsed: float) -> None:
        completed = len(results)
        successes = sum(1 for item in results if bool(item.get("success")))
        errors = sum(1 for item in results if item.get("error"))
        success_rate = successes / completed if completed else 0.0
        self.log(
            f"task {task_index}/{self.total_tasks} done task={task} "
            f"episodes={completed} success={successes}/{completed} ({success_rate:.1%}) "
            f"errors={errors} elapsed={format_duration(elapsed)}"
        )
        if self.episode_bar is not None:
            self.episode_bar.close()
            self.episode_bar = None
        if self.task_bar is not None:
            self.task_bar.update(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-host", default="127.0.0.1")
    parser.add_argument("--remote-port", type=int, default=8000)
    parser.add_argument("--tasks", default=os.environ.get("TASKS", os.environ.get("TASK", "beat_block_hammer")))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=1, help="Number of synchronized RoboTwin env workers.")
    parser.add_argument(
        "--eval-mode",
        choices=("lingbot", "batch", "microbatch"),
        default="lingbot",
        help=(
            "lingbot runs one episode at a time; batch keeps the legacy synchronized-wave evaluator; "
            "microbatch runs independent sessions against robotwin_microbatch_server."
        ),
    )
    parser.add_argument("--env-cuda", default=os.environ.get("ENV_CUDA", "0"))
    parser.add_argument("--worker-python", default=sys.executable)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--task-config",
        default=os.environ.get(ROBOTWIN_TASK_CONFIG_ENV, DEFAULT_ROBOTWIN_TASK_CONFIG),
        help="RoboTwin task config name, for example demo_clean or demo_randomized.",
    )
    parser.add_argument("--episode-length", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0, help="Max inference requests per episode; 0 derives from task step limit.")
    parser.add_argument("--open-loop-horizon", type=int, default=24, help="Dry-run action chunk length only.")
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--use-seed-cache", action=argparse.BooleanOptionalAction, default=env_flag("USE_SEED_CACHE", False))
    parser.add_argument("--seed-cache-root", type=Path, default=None)
    parser.add_argument("--reset-retries", type=int, default=5)
    parser.add_argument("--expert-filter-max-candidates", type=int, default=1000)
    parser.add_argument("--clip-action", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run-actions", action="store_true", help="Use zero actions and do not connect to the policy server.")
    parser.add_argument("--worker-timeout", type=float, default=900.0)
    parser.add_argument("--checkpoint-label", default="")
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--save-video", action="store_true", help="Save head-camera rollout videos from the worker.")
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--progress",
        choices=("plain", "tqdm", "none"),
        default=os.environ.get("ROBOTWIN_PROGRESS", "plain"),
        help="Progress display mode. tqdm falls back to plain if tqdm is unavailable.",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=float(os.environ.get("ROBOTWIN_PROGRESS_INTERVAL", "30")),
        help="Minimum seconds between in-wave progress heartbeats.",
    )
    parser.add_argument(
        "--client-image-resolution",
        default=os.environ.get("CLIENT_IMAGE_RESOLUTION", "none"),
        help="Client-side image resize target: none, auto, or HxW. auto uses websocket server metadata.",
    )
    return parser.parse_args()


def json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, set):
        return sorted(value)
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
    parts = [chunk.strip() for chunk in raw.replace(",", " ").split() if chunk.strip()]
    if not parts:
        raise ValueError("--tasks must contain at least one RoboTwin task")
    if len(parts) == 1 and parts[0].lower() == "all":
        return load_all_eval_tasks()
    if any(part.lower() == "all" for part in parts):
        raise ValueError("Use --tasks all by itself, or provide explicit task names")
    return parts


def resolve_seed_cache_root(args: argparse.Namespace) -> Path:
    if args.seed_cache_root is not None:
        return args.seed_cache_root.expanduser().resolve()
    return (args.output_dir / "expert_seed_cache").expanduser().resolve()


def load_seed_cache_for_task(args: argparse.Namespace, task: str) -> list[dict[str, Any]]:
    root = resolve_seed_cache_root(args)
    path = root / str(args.task_config) / f"{task}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing seed cache for task={task!r}: {path}. "
            "Run example/robotwin/expert_seed_cache.py first or pass --seed-cache-root."
        )
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("task") != task:
            raise ValueError(f"{path}:{line_no} has task={row.get('task')!r}, expected {task!r}")
        if row.get("task_config") != args.task_config:
            raise ValueError(
                f"{path}:{line_no} has task_config={row.get('task_config')!r}, expected {args.task_config!r}"
            )
        if not bool(row.get("accepted", False)):
            raise ValueError(f"{path}:{line_no} is not an accepted cache row")
        if "seed" not in row or "prompt" not in row:
            raise ValueError(f"{path}:{line_no} must contain seed and prompt")
        rows.append(row)
    if len(rows) < args.episodes:
        raise ValueError(
            f"Seed cache for task={task!r} has {len(rows)} rows but --episodes={args.episodes}. "
            "Generate more cache rows instead of reusing episodes implicitly."
        )
    return rows


def get_seed_cache_entry(args: argparse.Namespace, task: str, episode_index: int) -> dict[str, Any] | None:
    cache = getattr(args, "seed_cache_by_task", None)
    if not isinstance(cache, dict):
        return None
    rows = cache.get(task)
    if not rows:
        return None
    return rows[int(episode_index)]


def parse_cuda_list(raw: str, count: int) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        values = [os.environ.get("CUDA_VISIBLE_DEVICES", "")]
    return [values[i % len(values)] for i in range(count)]


def build_worker_env(args: argparse.Namespace, cuda_visible_devices: str) -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    env[ROBOTWIN_TASK_CONFIG_ENV] = str(args.task_config)
    pythonpath_parts = [
        str(REPO_ROOT),
        str(REPO_ROOT / "third_party" / "RoboTwin"),
        str(REPO_ROOT / "third_party" / "lerobot" / "src"),
        str(REPO_ROOT / "third_party" / "lerobot"),
    ]
    existing = env.get("PYTHONPATH")
    if existing:
        pythonpath_parts.append(existing)
    env["PYTHONPATH"] = ":".join(pythonpath_parts)
    return env


def start_workers(args: argparse.Namespace, log_dir: Path, progress: ProgressReporter | None = None) -> list[WorkerHandle]:
    log_dir.mkdir(parents=True, exist_ok=True)
    worker_count = max(int(args.num_envs), 1)
    authkey = secrets.token_bytes(16)
    listener = Listener(("127.0.0.1", 0), backlog=worker_count, authkey=authkey)
    with contextlib.suppress(AttributeError):
        listener._listener._socket.settimeout(1.0)
    host, port = listener.address
    cuda_values = parse_cuda_list(args.env_cuda, worker_count)
    worker_script = REPO_ROOT / "example" / "robotwin" / "parallel_env_worker.py"
    procs: dict[int, subprocess.Popen] = {}
    log_paths: dict[int, Path] = {}
    if progress is not None:
        progress.log(f"starting workers count={worker_count} logs={log_dir}")
    for worker_id in range(worker_count):
        log_path = log_dir / f"env_worker_{worker_id}.log"
        cmd = [
            args.worker_python,
            str(worker_script),
            "--host",
            str(host),
            "--port",
            str(port),
            "--authkey-hex",
            authkey.hex(),
            "--worker-id",
            str(worker_id),
        ]
        log_file = log_path.open("ab", buffering=0)
        try:
            procs[worker_id] = subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                env=build_worker_env(args, cuda_values[worker_id]),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            log_paths[worker_id] = log_path
        finally:
            log_file.close()

    deadline = time.time() + args.worker_timeout
    handles: dict[int, WorkerHandle] = {}
    try:
        if progress is not None:
            progress.set_status(f"worker connections 0/{worker_count}")
        while len(handles) < worker_count:
            if time.time() > deadline:
                missing = sorted(set(procs) - set(handles))
                paths = ", ".join(log_paths[item].as_posix() for item in missing)
                raise TimeoutError(f"Timed out waiting for workers {missing}; see {paths}")
            for worker_id, proc in procs.items():
                if worker_id not in handles and proc.poll() is not None:
                    raise RuntimeError(f"Worker {worker_id} exited before connecting; see {log_paths[worker_id]}")
            try:
                conn = listener.accept()
            except socket.timeout:
                continue
            ready = conn.recv()
            if not ready.get("ok") or ready.get("type") != "ready":
                raise RuntimeError(f"Unexpected worker handshake: {ready}")
            worker_id = int(ready["worker_id"])
            if worker_id not in procs:
                raise RuntimeError(f"Worker connected with unexpected id {worker_id}: {ready}")
            handles[worker_id] = WorkerHandle(
                worker_id=worker_id,
                process=procs[worker_id],
                conn=conn,
                log_path=log_paths[worker_id],
                cuda_visible_devices=str(ready.get("cuda_visible_devices", cuda_values[worker_id])),
            )
            if progress is not None:
                progress.log(
                    f"worker connected {len(handles)}/{worker_count} "
                    f"id={worker_id} pid={procs[worker_id].pid} cuda={handles[worker_id].cuda_visible_devices}"
                )
                if len(handles) < worker_count:
                    progress.set_status(f"worker connections {len(handles)}/{worker_count}")
        if progress is not None:
            progress.clear_status()
        return [handles[worker_id] for worker_id in sorted(handles)]
    except Exception:
        if progress is not None:
            progress.clear_status()
        for proc in procs.values():
            if proc.poll() is None:
                with contextlib.suppress(Exception):
                    os.killpg(proc.pid, signal.SIGTERM)
                with contextlib.suppress(Exception):
                    proc.terminate()
        raise
    finally:
        listener.close()


def stop_workers(workers: list[WorkerHandle]) -> None:
    for worker in workers:
        try:
            if worker.process.poll() is None:
                worker.conn.send({"cmd": "close"})
        except Exception:
            pass
    for worker in workers:
        try:
            if worker.conn.poll(5):
                worker.conn.recv()
        except Exception:
            pass
        with contextlib.suppress(Exception):
            worker.conn.close()
    for worker in workers:
        proc = worker.process
        if proc.poll() is not None:
            continue
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except Exception:
            proc.terminate()
    time.sleep(1)
    for worker in workers:
        proc = worker.process
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                proc.kill()


def recv_worker(worker: WorkerHandle, timeout: float) -> dict[str, Any]:
    if not worker.conn.poll(timeout):
        raise TimeoutError(f"Timed out waiting for worker {worker.worker_id}; see {worker.log_path}")
    response = worker.conn.recv()
    if not response.get("ok"):
        raise RuntimeError(
            f"Worker {worker.worker_id} failed with {response.get('error_type')}: {response.get('error')}\n"
            f"{response.get('traceback', '')}"
        )
    return response


def terminate_worker(worker: WorkerHandle) -> None:
    with contextlib.suppress(Exception):
        worker.conn.close()
    proc = worker.process
    if proc.poll() is not None:
        return
    with contextlib.suppress(Exception):
        os.killpg(proc.pid, signal.SIGTERM)
    with contextlib.suppress(Exception):
        proc.terminate()


def load_task_step_limit(task: str, fallback: int) -> int:
    if fallback > 0:
        return int(fallback)
    limit_path = REPO_ROOT / "third_party" / "RoboTwin" / "task_config" / "_eval_step_limit.yml"
    if limit_path.exists():
        for line in limit_path.read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            if key.strip() == task:
                with contextlib.suppress(ValueError):
                    return int(value.strip())
    return 300


def payload_size_bytes(payload: dict[str, Any]) -> int:
    payload_with_endpoint = dict(payload)
    payload_with_endpoint["endpoint"] = "infer"
    return len(msgpack_numpy.Packer().pack(payload_with_endpoint))


def normalize_action_sequence(action: Any) -> np.ndarray:
    arr = np.asarray(action, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim == 3:
        if arr.shape[0] != 1:
            raise ValueError(f"LingBot-style eval expects one action batch, got {arr.shape}")
        arr = arr[0]
    elif arr.ndim != 2:
        raise ValueError(f"Expected action shape [H,14], [1,H,14], or [14], got {arr.shape}")
    if arr.shape[-1] != ROBOTWIN_ACTION_DIM:
        raise ValueError(f"Expected action width {ROBOTWIN_ACTION_DIM}, got {arr.shape}")
    return np.ascontiguousarray(arr)


def normalize_batched_action_sequence(action: Any, batch_size: int) -> np.ndarray:
    arr = np.asarray(action, dtype=np.float32)
    if arr.ndim == 1:
        if batch_size != 1:
            raise ValueError(f"Expected batched action for B={batch_size}, got {arr.shape}")
        arr = arr.reshape(1, 1, -1)
    elif arr.ndim == 2:
        if arr.shape[-1] != ROBOTWIN_ACTION_DIM:
            raise ValueError(f"Expected action width {ROBOTWIN_ACTION_DIM}, got {arr.shape}")
        if batch_size == 1:
            arr = arr.reshape(1, arr.shape[0], arr.shape[1])
        elif arr.shape[0] == batch_size:
            arr = arr.reshape(batch_size, 1, arr.shape[1])
        else:
            raise ValueError(f"Cannot interpret action shape {arr.shape} for batch size {batch_size}")
    elif arr.ndim != 3:
        raise ValueError(f"Expected batched action shape [B,H,14], got {arr.shape}")
    if arr.shape[0] != batch_size:
        raise ValueError(f"Expected action batch size {batch_size}, got {arr.shape}")
    if arr.shape[-1] != ROBOTWIN_ACTION_DIM:
        raise ValueError(f"Expected action width {ROBOTWIN_ACTION_DIM}, got {arr.shape}")
    return np.ascontiguousarray(arr)


def checkpoint_result(args: argparse.Namespace) -> dict[str, str]:
    return {
        "label": str(args.checkpoint_label or ""),
        "path": str(args.checkpoint_path or ""),
    }


def server_timing_seconds(args: argparse.Namespace, key: str) -> float:
    server_timing = getattr(args, "server_timing", {}) or {}
    if not isinstance(server_timing, dict):
        return 0.0
    try:
        return float(server_timing.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def reset_profile_value(reset_response: dict[str, Any], key: str, default: float = 0.0) -> float:
    reset_profile = reset_response.get("reset_profile") or {}
    if not isinstance(reset_profile, dict):
        reset_profile = {}
    try:
        return float(reset_profile.get(key, default) or 0.0)
    except (TypeError, ValueError):
        return float(default)


def make_episode_metrics(
    reset_response: dict[str, Any],
    *,
    reset_start: float,
    now: float,
) -> dict[str, float | int]:
    reset_time = float(reset_response.get("reset_time", now - reset_start))
    reset_render_obs_time = reset_profile_value(reset_response, "get_obs_time", 0.0)
    env_reset_time = reset_profile_value(
        reset_response,
        "env_reset_time",
        max(reset_time - reset_render_obs_time, 0.0),
    )
    return {
        "reset_time": reset_time,
        "env_reset_time": env_reset_time,
        "expert_filter_time": float((reset_response.get("expert_filter") or {}).get("total_time", 0.0)),
        "infer_wait_time": 0.0,
        "env_step_time": 0.0,
        "get_obs_time": 0.0,
        "render_obs_time": reset_render_obs_time,
        "reset_render_obs_time": reset_render_obs_time,
        "step_chunk_time": 0.0,
        "infer_payload_size_bytes": 0,
        "payload_build_time": 0.0,
        "payload_pack_time": 0.0,
        "ws_roundtrip_time": 0.0,
        "send_obs_time": 0.0,
        "recv_action_time": 0.0,
        "policy_infer_time": 0.0,
        "server_policy_infer_time": 0.0,
        "action_normalize_time": 0.0,
        "worker_step_wait_time": 0.0,
        "batch_infer_time": 0.0,
        "per_env_infer_time": 0.0,
        "batch_infer_calls": 0,
        "microbatch_wait_time": 0.0,
        "microbatch_calls": 0,
        "microbatch_size_sum": 0,
        "microbatch_size_max": 0,
        "session_queue_depth_sum": 0,
        "session_queue_depth_max": 0,
        "server_vram_allocated_mb_max": 0.0,
        "server_vram_reserved_mb_max": 0.0,
        "sync_idle_steps": 0,
        "video_write_time": 0.0,
        "finish_episode_time": 0.0,
        "episode_result_write_time": 0.0,
        "write_log_video_time": 0.0,
    }


def add_client_server_timing(
    metrics: dict[str, float | int],
    client_timing: dict[str, Any],
    server_timing: dict[str, Any],
) -> None:
    metrics["send_obs_time"] = float(metrics.get("send_obs_time", 0.0)) + float(
        client_timing.get("T_send_obs", 0.0) or 0.0
    )
    metrics["recv_action_time"] = float(metrics.get("recv_action_time", 0.0)) + float(
        client_timing.get("T_recv_action", 0.0) or 0.0
    )
    policy_infer = float(
        server_timing.get("T_policy_infer", server_timing.get("policy_infer_time", 0.0)) or 0.0
    )
    metrics["policy_infer_time"] = float(metrics.get("policy_infer_time", 0.0)) + policy_infer
    metrics["server_policy_infer_time"] = float(metrics.get("server_policy_infer_time", 0.0)) + policy_infer
    microbatch_wait = float(server_timing.get("T_microbatch_wait", 0.0) or 0.0)
    microbatch_size = int(server_timing.get("microbatch_size", 0) or 0)
    queue_depth = int(server_timing.get("session_queue_depth", 0) or 0)
    vram_allocated = float(server_timing.get("server_vram_allocated_mb", 0.0) or 0.0)
    vram_reserved = float(server_timing.get("server_vram_reserved_mb", 0.0) or 0.0)
    if microbatch_wait or microbatch_size:
        metrics["microbatch_wait_time"] = float(metrics.get("microbatch_wait_time", 0.0)) + microbatch_wait
        metrics["microbatch_calls"] = int(metrics.get("microbatch_calls", 0)) + 1
        metrics["microbatch_size_sum"] = int(metrics.get("microbatch_size_sum", 0)) + microbatch_size
        metrics["microbatch_size_max"] = max(int(metrics.get("microbatch_size_max", 0)), microbatch_size)
        metrics["session_queue_depth_sum"] = int(metrics.get("session_queue_depth_sum", 0)) + queue_depth
        metrics["session_queue_depth_max"] = max(int(metrics.get("session_queue_depth_max", 0)), queue_depth)
    metrics["server_vram_allocated_mb_max"] = max(
        float(metrics.get("server_vram_allocated_mb_max", 0.0)),
        vram_allocated,
    )
    metrics["server_vram_reserved_mb_max"] = max(
        float(metrics.get("server_vram_reserved_mb_max", 0.0)),
        vram_reserved,
    )


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def resolve_client_image_resolution(raw: str, server_metadata: dict[str, Any] | None) -> tuple[int, int] | None:
    value = str(raw or "none").strip().lower()
    if value in {"", "none", "0", "false"}:
        return None
    if value == "auto":
        if not server_metadata or "image_resolution" not in server_metadata:
            raise ValueError("--client-image-resolution=auto requested but server metadata has no image_resolution")
        return parse_image_resolution(server_metadata["image_resolution"])
    return parse_image_resolution(value)


def write_run_config(
    args: argparse.Namespace,
    *,
    tasks: list[str],
    workers: list[WorkerHandle],
    server_metadata: dict[str, Any] | None,
) -> None:
    args_payload = {
        key: value
        for key, value in vars(args).items()
        if key not in {"seed_cache_by_task"}
    }
    mode = "robotwin_microbatch_eval" if args.eval_mode == "microbatch" else "lingbot_style_robotwin"
    payload = {
        "mode": mode,
        "timestamp": time.time(),
        "git_commit": git_commit(),
        "argv": sys.argv,
        "args": args_payload,
        "tasks": tasks,
        "server_metadata": server_metadata or {},
        "workers": [
            {
                "worker_id": worker.worker_id,
                "pid": worker.process.pid,
                "cuda_visible_devices": worker.cuda_visible_devices,
                "log_path": worker.log_path.as_posix(),
            }
            for worker in workers
        ],
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(payload, indent=2, default=json_default, ensure_ascii=False),
        encoding="utf-8",
    )


def write_episode_result(task_dir: Path, result: dict[str, Any]) -> float:
    task_dir.mkdir(parents=True, exist_ok=True)
    path = task_dir / f"episode_{int(result['episode_index']):06d}.json"
    write_start = time.perf_counter()
    path.write_text(json.dumps(result, indent=2, default=json_default, ensure_ascii=False), encoding="utf-8")
    write_time = time.perf_counter() - write_start
    profile = result.get("profile")
    timing = result.get("timing")
    if isinstance(profile, dict) and profile:
        profile["episode_result_write_time"] = write_time
        profile["T_write_log/video"] = float(profile.get("T_write_log/video", 0.0)) + write_time
        if isinstance(timing, dict):
            timing["episode_result_write_time"] = write_time
            timing["write_log_video_time"] = float(timing.get("write_log_video_time", 0.0)) + write_time
        path.write_text(json.dumps(result, indent=2, default=json_default, ensure_ascii=False), encoding="utf-8")
    return write_time


def aggregate_wave_batch_stats(results: list[dict[str, Any]]) -> dict[str, float | int]:
    calls_by_wave: dict[int, int] = {}
    batch_time_by_wave: dict[int, float] = {}
    for result in results:
        wave_id = int(result.get("wave_id", -1))
        if wave_id < 0:
            continue
        calls = int(result.get("batch_infer_calls", result.get("infer_calls", 0)))
        batch_time = float((result.get("timing") or {}).get("batch_infer_time", 0.0))
        calls_by_wave[wave_id] = max(calls_by_wave.get(wave_id, 0), calls)
        batch_time_by_wave[wave_id] = max(batch_time_by_wave.get(wave_id, 0.0), batch_time)
    total_calls = sum(calls_by_wave.values())
    total_batch_time = sum(batch_time_by_wave.values())
    return {
        "batch_infer_calls": total_calls,
        "batch_infer_time": total_batch_time,
        "avg_batch_infer_time": total_batch_time / total_calls if total_calls else 0.0,
    }


def write_task_summary(task_dir: Path, task: str, results: list[dict[str, Any]]) -> None:
    successes = sum(int(bool(result.get("success"))) for result in results)
    steps = [int(result.get("steps", 0)) for result in results]
    infer_counts = [int(result.get("infer_calls", 0)) for result in results]
    keyframes = [int(result.get("keyframe_count", 0)) for result in results]
    wave_ids = sorted({int(result.get("wave_id", -1)) for result in results if int(result.get("wave_id", -1)) >= 0})
    wave_batch_sizes_by_id = {
        wave_id: max(
            int(result.get("wave_batch_size", 1))
            for result in results
            if int(result.get("wave_id", -1)) == wave_id
        )
        for wave_id in wave_ids
    }
    wave_batch_sizes = list(wave_batch_sizes_by_id.values())
    wave_batch_stats = aggregate_wave_batch_stats(results)
    per_env_batch_call_count = sum(int(result.get("batch_infer_calls", result.get("infer_calls", 0))) for result in results)
    per_env_infer_time = sum(float((result.get("timing") or {}).get("per_env_infer_time", 0.0)) for result in results)
    sync_idle_steps = sum(int(result.get("sync_idle_steps", 0)) for result in results)
    microbatch_calls = sum(int((result.get("timing") or {}).get("microbatch_calls", 0)) for result in results)
    microbatch_size_sum = sum(
        float((result.get("timing") or {}).get("avg_microbatch_size", 0.0))
        * int((result.get("timing") or {}).get("microbatch_calls", 0))
        for result in results
    )
    session_queue_depth_sum = sum(
        float((result.get("timing") or {}).get("avg_session_queue_depth", 0.0))
        * int((result.get("timing") or {}).get("microbatch_calls", 0))
        for result in results
    )
    summary = {
        "mode": "robotwin_microbatch_eval"
        if results and str(results[0].get("eval_mode", "")) == "microbatch"
        else "lingbot_style_robotwin",
        "task": task,
        "episodes": len(results),
        "successes": successes,
        "success_rate": successes / len(results) if results else 0.0,
        "avg_steps": float(np.mean(steps)) if steps else 0.0,
        "episode_length_mean": float(np.mean(steps)) if steps else 0.0,
        "episode_length_std": float(np.std(steps)) if steps else 0.0,
        "episode_length_min": int(np.min(steps)) if steps else 0,
        "episode_length_max": int(np.max(steps)) if steps else 0,
        "avg_infer_calls": float(np.mean(infer_counts)) if infer_counts else 0.0,
        "avg_keyframes": float(np.mean(keyframes)) if keyframes else 0.0,
        "num_envs_requested": int(results[0].get("num_envs_requested", 1)) if results else 0,
        "waves": len(wave_ids),
        "avg_wave_batch_size": float(np.mean(wave_batch_sizes)) if wave_batch_sizes else 0.0,
        "batch_infer_calls": int(wave_batch_stats["batch_infer_calls"]),
        "avg_batch_infer_time": float(wave_batch_stats["avg_batch_infer_time"]),
        "avg_per_env_infer_time": per_env_infer_time / per_env_batch_call_count if per_env_batch_call_count else 0.0,
        "microbatch_calls": microbatch_calls,
        "avg_microbatch_size": microbatch_size_sum / microbatch_calls if microbatch_calls else 0.0,
        "max_microbatch_size": max(
            (int((result.get("timing") or {}).get("max_microbatch_size", 0)) for result in results),
            default=0,
        ),
        "avg_session_queue_depth": session_queue_depth_sum / microbatch_calls if microbatch_calls else 0.0,
        "max_session_queue_depth": max(
            (int((result.get("timing") or {}).get("max_session_queue_depth", 0)) for result in results),
            default=0,
        ),
        "server_vram_allocated_mb_max": max(
            (float((result.get("timing") or {}).get("server_vram_allocated_mb_max", 0.0)) for result in results),
            default=0.0,
        ),
        "server_vram_reserved_mb_max": max(
            (float((result.get("timing") or {}).get("server_vram_reserved_mb_max", 0.0)) for result in results),
            default=0.0,
        ),
        "sync_idle_steps": sync_idle_steps,
        "timestamp": time.time(),
    }
    for field in PROFILE_TIMER_FIELDS:
        values = [
            float((result.get("profile") or {}).get(field, 0.0) or 0.0)
            for result in results
        ]
        summary[f"avg_{field}"] = sum(values) / len(values) if values else 0.0
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=json_default, ensure_ascii=False),
        encoding="utf-8",
    )


def error_episode_result(
    args: argparse.Namespace,
    *,
    task: str,
    episode_index: int,
    worker: WorkerHandle,
    exc: BaseException,
    wave_id: int | None = None,
    batch_index: int | None = None,
    wave_batch_size: int | None = None,
) -> dict[str, Any]:
    return {
        "episode_index": episode_index,
        "worker_id": worker.worker_id,
        "worker_cuda_visible_devices": worker.cuda_visible_devices,
        "task": task,
        "prompt": "",
        "session_id": "",
        "steps": 0,
        "success": False,
        "reward_sum": 0.0,
        "done": False,
        "error": f"{exc.__class__.__name__}: {exc}",
        "checkpoint": checkpoint_result(args),
        "video_path": None,
        "video_frames": 0,
        "eval_mode": str(args.eval_mode),
        "lingbot_style_eval": True,
        "fast_eval": True,
        "env_reuse_mode": "cold_reset",
        "wave_id": int(wave_id) if wave_id is not None else -1,
        "batch_index": int(batch_index) if batch_index is not None else 0,
        "num_envs_requested": int(args.num_envs),
        "wave_batch_size": int(wave_batch_size) if wave_batch_size is not None else 1,
        "batch_infer_calls": 0,
        "avg_batch_infer_time": 0.0,
        "avg_per_env_infer_time": 0.0,
        "sync_idle_steps": 0,
        "timing": {
            "reset_time": 0.0,
            "env_reset_time": 0.0,
            "expert_filter_time": 0.0,
            "infer_wait_time": 0.0,
            "env_step_time": 0.0,
            "get_obs_time": 0.0,
            "render_obs_time": 0.0,
            "step_chunk_time": 0.0,
            "episode_wall_time": 0.0,
            "infer_payload_size_bytes": 0,
            "batch_infer_time": 0.0,
            "per_env_infer_time": 0.0,
            "send_obs_time": 0.0,
            "policy_infer_time": 0.0,
            "recv_action_time": 0.0,
            "microbatch_wait_time": 0.0,
            "avg_microbatch_size": 0.0,
            "max_microbatch_size": 0,
            "avg_session_queue_depth": 0.0,
            "max_session_queue_depth": 0,
            "server_vram_allocated_mb_max": 0.0,
            "server_vram_reserved_mb_max": 0.0,
            "write_log_video_time": 0.0,
        },
        "profile": {key: 0.0 for key in PROFILE_TIMER_FIELDS} if args.profile else {},
        "last_info": {},
    }


def finalize_episode(args: argparse.Namespace, state: dict[str, Any], completed_at: float) -> dict[str, Any]:
    metrics = state["metrics"]
    episode_wall_time = float(completed_at - state["episode_start"])
    batch_infer_calls = int(metrics.get("batch_infer_calls", state["infer_calls"]))
    batch_infer_time = float(metrics.get("batch_infer_time", 0.0))
    per_env_infer_time = float(metrics.get("per_env_infer_time", 0.0))
    write_log_video_time = float(metrics.get("write_log_video_time", 0.0))
    microbatch_calls = int(metrics.get("microbatch_calls", 0))
    avg_microbatch_size = (
        float(metrics.get("microbatch_size_sum", 0)) / microbatch_calls if microbatch_calls else 0.0
    )
    avg_session_queue_depth = (
        float(metrics.get("session_queue_depth_sum", 0)) / microbatch_calls if microbatch_calls else 0.0
    )
    timing = {
        "reset_time": float(metrics.get("reset_time", 0.0)),
        "env_reset_time": float(metrics.get("env_reset_time", metrics.get("reset_time", 0.0))),
        "expert_filter_time": float(metrics.get("expert_filter_time", 0.0)),
        "infer_wait_time": float(metrics.get("infer_wait_time", 0.0)),
        "env_step_time": float(metrics.get("env_step_time", 0.0)),
        "get_obs_time": float(metrics.get("get_obs_time", 0.0)),
        "render_obs_time": float(metrics.get("render_obs_time", metrics.get("get_obs_time", 0.0))),
        "reset_render_obs_time": float(metrics.get("reset_render_obs_time", 0.0)),
        "step_chunk_time": float(metrics.get("step_chunk_time", 0.0)),
        "episode_wall_time": episode_wall_time,
        "infer_payload_size_bytes": int(metrics.get("infer_payload_size_bytes", 0)),
        "batch_infer_time": batch_infer_time,
        "per_env_infer_time": per_env_infer_time,
        "send_obs_time": float(metrics.get("send_obs_time", 0.0)),
        "policy_infer_time": float(metrics.get("policy_infer_time", 0.0)),
        "recv_action_time": float(metrics.get("recv_action_time", 0.0)),
        "microbatch_wait_time": float(metrics.get("microbatch_wait_time", 0.0)),
        "microbatch_calls": microbatch_calls,
        "avg_microbatch_size": avg_microbatch_size,
        "max_microbatch_size": int(metrics.get("microbatch_size_max", 0)),
        "avg_session_queue_depth": avg_session_queue_depth,
        "max_session_queue_depth": int(metrics.get("session_queue_depth_max", 0)),
        "server_vram_allocated_mb_max": float(metrics.get("server_vram_allocated_mb_max", 0.0)),
        "server_vram_reserved_mb_max": float(metrics.get("server_vram_reserved_mb_max", 0.0)),
        "video_write_time": float(metrics.get("video_write_time", 0.0)),
        "finish_episode_time": float(metrics.get("finish_episode_time", 0.0)),
        "episode_result_write_time": float(metrics.get("episode_result_write_time", 0.0)),
        "write_log_video_time": write_log_video_time,
    }
    profile = {
        "T_load_model": server_timing_seconds(args, "T_load_model"),
        "T_server_ready": server_timing_seconds(args, "T_server_ready"),
        "T_env_reset": timing["env_reset_time"],
        "T_expert_seed_filter": timing["expert_filter_time"],
        "T_render_obs": timing["render_obs_time"],
        "T_send_obs": timing["send_obs_time"],
        "T_policy_infer": timing["policy_infer_time"],
        "T_recv_action": timing["recv_action_time"],
        "T_step_chunk": timing["step_chunk_time"],
        "T_write_log/video": write_log_video_time,
        "T_microbatch_wait": timing["microbatch_wait_time"],
        "avg_microbatch_size": avg_microbatch_size,
        "max_microbatch_size": timing["max_microbatch_size"],
        "avg_session_queue_depth": avg_session_queue_depth,
        "max_session_queue_depth": timing["max_session_queue_depth"],
        "server_vram_allocated_mb_max": timing["server_vram_allocated_mb_max"],
        "server_vram_reserved_mb_max": timing["server_vram_reserved_mb_max"],
        "payload_build_time": float(metrics.get("payload_build_time", 0.0)),
        "payload_pack_time": float(metrics.get("payload_pack_time", 0.0)),
        "ws_roundtrip_time": float(metrics.get("ws_roundtrip_time", 0.0)),
        "action_normalize_time": float(metrics.get("action_normalize_time", 0.0)),
        "worker_step_wait_time": float(metrics.get("worker_step_wait_time", 0.0)),
        "video_write_time": timing["video_write_time"],
        "finish_episode_time": timing["finish_episode_time"],
        "episode_result_write_time": timing["episode_result_write_time"],
    }
    return {
        "episode_index": int(state["episode_index"]),
        "worker_id": int(state["worker"].worker_id),
        "worker_cuda_visible_devices": state["worker"].cuda_visible_devices,
        "seed": int(state["seed"]),
        "reset_attempts": int(state["reset_attempts"]),
        "task": state["task"],
        "prompt": state["prompt"],
        "session_id": state["session_id"],
        "steps": int(state["steps"]),
        "infer_calls": int(state["infer_calls"]),
        "keyframe_count": int(state["keyframe_count"]),
        "success": bool(state["success"]),
        "reward_sum": float(state["reward_sum"]),
        "done": bool(state["done"]),
        "error": str(state.get("error", "")),
        "action_shapes": state["action_shapes"],
        "first_action": state["first_action"],
        "checkpoint": checkpoint_result(args),
        "video_path": state.get("video_path"),
        "video_frames": int(state.get("video_frames", 0)),
        "eval_mode": str(args.eval_mode),
        "lingbot_style_eval": True,
        "fast_eval": True,
        "env_reuse_mode": "cold_reset",
        "wave_id": int(state.get("wave_id", -1)),
        "batch_index": int(state.get("batch_index", 0)),
        "num_envs_requested": int(state.get("num_envs_requested", args.num_envs)),
        "wave_batch_size": int(state.get("wave_batch_size", 1)),
        "batch_infer_calls": batch_infer_calls,
        "avg_batch_infer_time": batch_infer_time / batch_infer_calls if batch_infer_calls else 0.0,
        "avg_per_env_infer_time": per_env_infer_time / batch_infer_calls if batch_infer_calls else 0.0,
        "sync_idle_steps": int(metrics.get("sync_idle_steps", 0)),
        "expert_filter": state.get("expert_filter", {"enabled": True}),
        "timing": timing,
        "profile": profile if args.profile else {},
        "last_info": state["last_info"],
    }


def reset_worker_for_episode(
    args: argparse.Namespace,
    *,
    task: str,
    worker: WorkerHandle,
    episode_index: int,
    episode_length: int,
    task_dir: Path,
    seed_start_override: int | None = None,
    worker_episode_index: int | None = None,
    episodes_override: int | None = None,
    cache_entry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    video_path = task_dir / "videos" / f"episode_{episode_index:06d}.mp4" if args.save_video else None
    payload = {
        "cmd": "reset",
        "task_name": task,
        "episode_index": int(worker_episode_index) if worker_episode_index is not None else episode_index,
        "episode_length": episode_length,
        "seed_start": int(seed_start_override) if seed_start_override is not None else args.seed_start,
        "episodes": int(episodes_override) if episodes_override is not None else args.episodes,
        "seed_stride": 1 if seed_start_override is not None else args.episodes,
        "reset_retries": args.reset_retries,
        "expert_filter": True,
        "expert_filter_max_candidates": args.expert_filter_max_candidates,
        "video_path": video_path.as_posix() if video_path is not None else "",
        "video_fps": args.video_fps,
    }
    if cache_entry is not None:
        payload.update(
            {
                "fixed_seed": int(cache_entry["seed"]),
                "fixed_prompt": str(cache_entry["prompt"]),
                "episode_info": cache_entry.get("episode_info", {}),
                "expert_filter": False,
            }
        )
    worker.conn.send(payload)
    return recv_worker(worker, args.worker_timeout)


def send_reset_worker_for_episode(
    args: argparse.Namespace,
    *,
    task: str,
    worker: WorkerHandle,
    episode_index: int,
    episode_length: int,
    task_dir: Path,
    seed_start_override: int | None = None,
    worker_episode_index: int | None = None,
    episodes_override: int | None = None,
    cache_entry: dict[str, Any] | None = None,
) -> float:
    video_path = task_dir / "videos" / f"episode_{episode_index:06d}.mp4" if args.save_video else None
    reset_start = time.perf_counter()
    payload = {
        "cmd": "reset",
        "task_name": task,
        "episode_index": int(worker_episode_index) if worker_episode_index is not None else episode_index,
        "episode_length": episode_length,
        "seed_start": int(seed_start_override) if seed_start_override is not None else args.seed_start,
        "episodes": int(episodes_override) if episodes_override is not None else args.episodes,
        "seed_stride": 1 if seed_start_override is not None else args.episodes,
        "reset_retries": args.reset_retries,
        "expert_filter": True,
        "expert_filter_max_candidates": args.expert_filter_max_candidates,
        "video_path": video_path.as_posix() if video_path is not None else "",
        "video_fps": args.video_fps,
    }
    if cache_entry is not None:
        payload.update(
            {
                "fixed_seed": int(cache_entry["seed"]),
                "fixed_prompt": str(cache_entry["prompt"]),
                "episode_info": cache_entry.get("episode_info", {}),
                "expert_filter": False,
            }
        )
    worker.conn.send(payload)
    return reset_start


def set_worker_prompt(args: argparse.Namespace, worker: WorkerHandle, prompt: str) -> str:
    worker.conn.send({"cmd": "set_prompt", "prompt": prompt})
    response = recv_worker(worker, args.worker_timeout)
    return str(response.get("prompt", prompt))


def make_wave_state(
    args: argparse.Namespace,
    *,
    task: str,
    worker: WorkerHandle,
    episode_index: int,
    episode_length: int,
    task_dir: Path,
    wave_id: int,
    batch_index: int,
    wave_batch_size: int,
    reset_start: float,
    reset_response: dict[str, Any],
) -> dict[str, Any]:
    now = time.perf_counter()
    prompt = str(reset_response["prompt"])
    return {
        "worker": worker,
        "episode_index": episode_index,
        "task": task,
        "prompt": prompt,
        "session_id": "",
        "seed": int(reset_response["seed"]),
        "reset_attempts": int(reset_response["reset_attempts"]),
        "expert_filter": reset_response.get("expert_filter", {"enabled": True}),
        "obs_sequence": [reset_response["obs"]],
        "done": False,
        "success": False,
        "reward_sum": 0.0,
        "steps": 0,
        "infer_calls": 0,
        "keyframe_count": 1,
        "action_shapes": [],
        "first_action": None,
        "last_info": reset_response.get("info", {}),
        "episode_start": reset_start,
        "video_path": (task_dir / "videos" / f"episode_{episode_index:06d}.mp4").as_posix()
        if args.save_video
        else None,
        "video_frames": 0,
        "wave_id": wave_id,
        "batch_index": batch_index,
        "num_envs_requested": int(args.num_envs),
        "wave_batch_size": wave_batch_size,
        "episode_length": episode_length,
        "metrics": make_episode_metrics(reset_response, reset_start=reset_start, now=now),
    }


def run_episode(
    args: argparse.Namespace,
    *,
    task: str,
    worker: WorkerHandle,
    client: WebsocketClientPolicy | None,
    episode_index: int,
    episode_length: int,
    task_dir: Path,
    wave_id: int | None = None,
    seed_start_override: int | None = None,
) -> dict[str, Any]:
    reset_start = time.perf_counter()
    reset_response = reset_worker_for_episode(
        args,
        task=task,
        worker=worker,
        episode_index=episode_index,
        episode_length=episode_length,
        task_dir=task_dir,
        seed_start_override=seed_start_override,
        worker_episode_index=0 if seed_start_override is not None else None,
        episodes_override=1 if seed_start_override is not None else None,
        cache_entry=get_seed_cache_entry(args, task, episode_index),
    )
    now = time.perf_counter()
    prompt = str(reset_response["prompt"])
    seed = int(reset_response["seed"])
    session_id = f"robotwin-lingbot-{task}-{seed}-{uuid.uuid4().hex[:8]}"
    if client is not None:
        client.reset({"session_id": session_id, "prompt": prompt})

    state: dict[str, Any] = {
        "worker": worker,
        "episode_index": episode_index,
        "task": task,
        "prompt": prompt,
        "session_id": session_id,
        "seed": seed,
        "reset_attempts": int(reset_response["reset_attempts"]),
        "expert_filter": reset_response.get("expert_filter", {"enabled": True}),
        "obs_sequence": [reset_response["obs"]],
        "done": False,
        "success": False,
        "reward_sum": 0.0,
        "steps": 0,
        "infer_calls": 0,
        "keyframe_count": 1,
        "action_shapes": [],
        "first_action": None,
        "last_info": reset_response.get("info", {}),
        "episode_start": reset_start,
        "video_path": (task_dir / "videos" / f"episode_{episode_index:06d}.mp4").as_posix()
        if args.save_video
        else None,
        "video_frames": 0,
        "wave_id": int(wave_id) if wave_id is not None else episode_index,
        "batch_index": 0,
        "num_envs_requested": 1,
        "wave_batch_size": 1,
        "metrics": make_episode_metrics(reset_response, reset_start=reset_start, now=now),
    }
    max_infer = args.max_steps if args.max_steps > 0 else max(int(episode_length), 1)

    for _ in range(max_infer):
        if state["done"]:
            break

        payload_build_start = time.perf_counter()
        payload = robotwin_obs_sequence_to_payload(
            state["obs_sequence"],
            prompt,
            session_id,
            image_resolution=args.resolved_client_image_resolution,
        )
        payload_build_time = time.perf_counter() - payload_build_start
        payload_pack_start = time.perf_counter()
        size_bytes = payload_size_bytes(payload)
        payload_pack_time = time.perf_counter() - payload_pack_start
        ws_roundtrip_time = 0.0
        action_normalize_time = 0.0
        client_timing: dict[str, Any] = {}
        server_timing: dict[str, Any] = {}

        if args.dry_run_actions:
            action_chunk = np.zeros((args.open_loop_horizon, ROBOTWIN_ACTION_DIM), dtype=np.float32)
        else:
            assert client is not None
            infer_start = time.perf_counter()
            raw_action, client_timing, server_timing = client.infer_timed(dict(payload))
            ws_roundtrip_time = time.perf_counter() - infer_start
            normalize_start = time.perf_counter()
            action_chunk = normalize_action_sequence(raw_action)
            action_normalize_time = time.perf_counter() - normalize_start

        infer_time = ws_roundtrip_time + action_normalize_time
        metrics = state["metrics"]
        state["infer_calls"] += 1
        metrics["infer_wait_time"] += infer_time
        metrics["infer_payload_size_bytes"] += size_bytes
        metrics["payload_build_time"] += payload_build_time
        metrics["payload_pack_time"] += payload_pack_time
        metrics["ws_roundtrip_time"] += ws_roundtrip_time
        metrics["action_normalize_time"] += action_normalize_time
        metrics["batch_infer_time"] += infer_time
        metrics["per_env_infer_time"] += infer_time
        metrics["batch_infer_calls"] += 1
        add_client_server_timing(metrics, client_timing, server_timing)
        state["action_shapes"].append(list(action_chunk.shape))
        if state["first_action"] is None and action_chunk.size:
            state["first_action"] = action_chunk[0].tolist()

        worker.conn.send(
            {
                "cmd": "step_chunk",
                "actions": action_chunk,
                "need_obs": True,
                "clip_action": bool(args.clip_action),
                "max_keyframes": KEYFRAMES_PER_CHUNK,
            }
        )
        worker_wait_start = time.perf_counter()
        step_response = recv_worker(worker, args.worker_timeout)
        worker_wait_time = time.perf_counter() - worker_wait_start

        actions_sent = int(step_response.get("actions_sent", 0))
        get_obs_time = float(step_response.get("get_obs_time", 0.0))
        state["metrics"]["worker_step_wait_time"] += worker_wait_time
        state["metrics"]["env_step_time"] += float(step_response.get("env_step_time", 0.0))
        state["metrics"]["get_obs_time"] += get_obs_time
        state["metrics"]["render_obs_time"] += get_obs_time
        state["metrics"]["step_chunk_time"] += float(
            step_response.get(
                "step_chunk_time",
                float(step_response.get("env_step_time", 0.0)) + get_obs_time,
            )
        )
        state["steps"] += actions_sent
        state["reward_sum"] += float(step_response.get("reward_sum", 0.0))
        state["done"] = bool(step_response.get("done", False))
        state["last_info"] = step_response.get("info", {})
        state["success"] = bool(state["last_info"].get("is_success", state["success"]))
        keyframes = step_response.get("keyframe_obs") or []
        if keyframes:
            state["obs_sequence"] = keyframes[-KEYFRAMES_PER_CHUNK:]
            state["keyframe_count"] += len(keyframes)
        elif step_response.get("obs") is not None:
            state["obs_sequence"] = [step_response["obs"]]
            state["keyframe_count"] += 1
        if actions_sent <= 0:
            state["done"] = True

    completed_at = time.perf_counter()
    if args.save_video:
        try:
            worker.conn.send({"cmd": "finish_episode"})
            video_response = recv_worker(worker, args.worker_timeout)
            state["video_path"] = video_response.get("video_path") or state.get("video_path")
            state["video_frames"] = int(video_response.get("video_frames", 0))
            state["metrics"]["video_write_time"] += float(video_response.get("video_write_time", 0.0))
            state["metrics"]["finish_episode_time"] += float(video_response.get("finish_episode_time", 0.0))
            state["metrics"]["write_log_video_time"] += float(video_response.get("video_write_time", 0.0))
        except Exception as exc:
            last_info = state.get("last_info", {})
            if not isinstance(last_info, dict):
                last_info = {"last_info": last_info}
            last_info["video_error"] = f"{exc.__class__.__name__}: {exc}"
            state["last_info"] = last_info
    return finalize_episode(args, state, completed_at)


def run_wave(
    args: argparse.Namespace,
    *,
    task: str,
    task_index: int,
    workers: list[WorkerHandle],
    client: WebsocketClientPolicy | None,
    episode_indices: list[int],
    episode_length: int,
    task_dir: Path,
    wave_id: int,
    total_waves: int,
    progress: ProgressReporter | None,
) -> list[dict[str, Any]]:
    planned_batch_size = len(episode_indices)
    if planned_batch_size <= 0:
        return []
    wave_started_at = time.perf_counter()
    wave_workers = workers[:planned_batch_size]
    results: list[dict[str, Any]] = []
    reset_starts: dict[int, float] = {}

    if progress is not None:
        first_ep = min(episode_indices)
        last_ep = max(episode_indices)
        progress.wave_phase(
            task_index=task_index,
            task=task,
            wave_id=wave_id,
            total_waves=total_waves,
            phase="start",
            detail=f"episodes={first_ep}-{last_ep} batch_size={planned_batch_size}",
        )

    for batch_index, (worker, episode_index) in enumerate(zip(wave_workers, episode_indices, strict=True)):
        try:
            reset_starts[batch_index] = send_reset_worker_for_episode(
                args,
                task=task,
                worker=worker,
                episode_index=episode_index,
                episode_length=episode_length,
                task_dir=task_dir,
                cache_entry=get_seed_cache_entry(args, task, episode_index),
            )
        except Exception as exc:
            terminate_worker(worker)
            results.append(
                error_episode_result(
                    args,
                    task=task,
                    episode_index=episode_index,
                    worker=worker,
                    exc=exc,
                    wave_id=wave_id,
                    batch_index=batch_index,
                    wave_batch_size=planned_batch_size,
                )
            )

    if progress is not None:
        progress.wave_phase(
            task_index=task_index,
            task=task,
            wave_id=wave_id,
            total_waves=total_waves,
            phase="reset",
            detail=f"waiting for {len(reset_starts)}/{planned_batch_size} env resets and expert filters",
        )
        progress.set_status(
            f"task {task_index}/{progress.total_tasks} {task} wave {wave_id + 1}/{total_waves} "
            f"reset/expert_filter ready=0/{len(reset_starts)}"
        )

    states: list[dict[str, Any]] = []
    for batch_index, (worker, episode_index) in enumerate(zip(wave_workers, episode_indices, strict=True)):
        if batch_index not in reset_starts:
            continue
        if progress is not None:
            progress.set_status(
                f"task {task_index}/{progress.total_tasks} {task} wave {wave_id + 1}/{total_waves} "
                f"reset/expert_filter worker={worker.worker_id} episode={episode_index} ready={len(states)}/{len(reset_starts)}"
            )
        try:
            reset_response = recv_worker(worker, args.worker_timeout)
            states.append(
                make_wave_state(
                    args,
                    task=task,
                    worker=worker,
                    episode_index=episode_index,
                    episode_length=episode_length,
                    task_dir=task_dir,
                    wave_id=wave_id,
                    batch_index=batch_index,
                    wave_batch_size=planned_batch_size,
                    reset_start=reset_starts[batch_index],
                    reset_response=reset_response,
                )
            )
        except Exception as exc:
            terminate_worker(worker)
            results.append(
                error_episode_result(
                    args,
                    task=task,
                    episode_index=episode_index,
                    worker=worker,
                    exc=exc,
                    wave_id=wave_id,
                    batch_index=batch_index,
                    wave_batch_size=planned_batch_size,
                )
            )

    if not states:
        if progress is not None:
            progress.clear_status()
        return results
    if progress is not None:
        progress.clear_status()
        progress.wave_phase(
            task_index=task_index,
            task=task,
            wave_id=wave_id,
            total_waves=total_waves,
            phase="reset_done",
            detail=f"ready={len(states)} reset_errors={len(results)}",
        )

    canonical_prompt = str(states[0]["prompt"])
    session_id = f"robotwin-lingbot-{task}-wave-{wave_id}-{uuid.uuid4().hex[:8]}"
    synced_states: list[dict[str, Any]] = []
    for batch_index, state in enumerate(states):
        state["batch_index"] = batch_index
        state["wave_batch_size"] = len(states)
        state["session_id"] = session_id
        state["prompt"] = canonical_prompt
        try:
            set_worker_prompt(args, state["worker"], canonical_prompt)
            synced_states.append(state)
        except Exception as exc:
            terminate_worker(state["worker"])
            results.append(
                error_episode_result(
                    args,
                    task=task,
                    episode_index=int(state["episode_index"]),
                    worker=state["worker"],
                    exc=exc,
                    wave_id=wave_id,
                    batch_index=batch_index,
                    wave_batch_size=len(states),
                )
            )

    states = synced_states
    if not states:
        return results
    for batch_index, state in enumerate(states):
        state["batch_index"] = batch_index
        state["wave_batch_size"] = len(states)

    if client is not None:
        if progress is not None:
            progress.set_status(
                f"task {task_index}/{progress.total_tasks} {task} wave {wave_id + 1}/{total_waves} policy reset"
            )
        try:
            client.reset({"session_id": session_id, "prompt": canonical_prompt})
        finally:
            if progress is not None:
                progress.clear_status()

    max_infer = args.max_steps if args.max_steps > 0 else max(int(episode_length), 1)
    batch_size = len(states)
    if progress is not None:
        progress.wave_phase(
            task_index=task_index,
            task=task,
            wave_id=wave_id,
            total_waves=total_waves,
            phase="infer",
            detail=f"batch_size={batch_size} max_infer={max_infer} session_id={session_id}",
        )
    for infer_index in range(max_infer):
        active_indices = [idx for idx, state in enumerate(states) if not bool(state["done"])]
        if not active_indices:
            break

        payload_build_start = time.perf_counter()
        payload = robotwin_obs_sequences_to_batched_payload(
            [state["obs_sequence"] for state in states],
            canonical_prompt,
            session_id,
            image_resolution=args.resolved_client_image_resolution,
        )
        payload_build_time = time.perf_counter() - payload_build_start
        payload_pack_start = time.perf_counter()
        size_bytes = payload_size_bytes(payload)
        payload_pack_time = time.perf_counter() - payload_pack_start
        ws_roundtrip_time = 0.0
        action_normalize_time = 0.0
        client_timing: dict[str, Any] = {}
        server_timing: dict[str, Any] = {}

        if args.dry_run_actions:
            action_batch = np.zeros((batch_size, args.open_loop_horizon, ROBOTWIN_ACTION_DIM), dtype=np.float32)
        else:
            assert client is not None
            infer_start = time.perf_counter()
            if progress is not None:
                progress.set_status(
                    f"task {task_index}/{progress.total_tasks} {task} wave {wave_id + 1}/{total_waves} "
                    f"infer {infer_index + 1}/{max_infer} batch={batch_size}"
                )
            try:
                raw_action, client_timing, server_timing = client.infer_timed(dict(payload))
            finally:
                if progress is not None:
                    progress.clear_status()
            ws_roundtrip_time = time.perf_counter() - infer_start
            normalize_start = time.perf_counter()
            action_batch = normalize_batched_action_sequence(raw_action, batch_size)
            action_normalize_time = time.perf_counter() - normalize_start

        infer_time = ws_roundtrip_time + action_normalize_time
        per_env_infer_time = infer_time / max(batch_size, 1)
        per_env_payload_size = int(round(size_bytes / max(batch_size, 1)))
        idle_count = batch_size - len(active_indices)

        for state in states:
            metrics = state["metrics"]
            metrics["batch_infer_calls"] += 1
            metrics["batch_infer_time"] += infer_time
            metrics["per_env_infer_time"] += per_env_infer_time
            if bool(state["done"]):
                metrics["sync_idle_steps"] += 1

        for batch_index in active_indices:
            state = states[batch_index]
            action_chunk = action_batch[batch_index]
            state["infer_calls"] += 1
            state["metrics"]["infer_wait_time"] += infer_time
            state["metrics"]["infer_payload_size_bytes"] += per_env_payload_size
            state["metrics"]["payload_build_time"] += payload_build_time
            state["metrics"]["payload_pack_time"] += payload_pack_time
            state["metrics"]["ws_roundtrip_time"] += ws_roundtrip_time
            state["metrics"]["action_normalize_time"] += action_normalize_time
            add_client_server_timing(state["metrics"], client_timing, server_timing)
            state["action_shapes"].append(list(action_chunk.shape))
            if state["first_action"] is None and action_chunk.size:
                state["first_action"] = action_chunk[0].tolist()
            try:
                state["worker"].conn.send(
                    {
                        "cmd": "step_chunk",
                        "actions": action_chunk,
                        "need_obs": True,
                        "clip_action": bool(args.clip_action),
                        "max_keyframes": KEYFRAMES_PER_CHUNK,
                    }
                )
            except Exception as exc:
                state["done"] = True
                state["error"] = f"{exc.__class__.__name__}: {exc}"
                terminate_worker(state["worker"])

        for batch_index in active_indices:
            state = states[batch_index]
            if state.get("error"):
                continue
            worker_wait_start = time.perf_counter()
            if progress is not None:
                progress.set_status(
                    f"task {task_index}/{progress.total_tasks} {task} wave {wave_id + 1}/{total_waves} "
                    f"step worker={state['worker'].worker_id} episode={state['episode_index']}"
                )
            try:
                step_response = recv_worker(state["worker"], args.worker_timeout)
            except Exception as exc:
                state["done"] = True
                state["error"] = f"{exc.__class__.__name__}: {exc}"
                terminate_worker(state["worker"])
                continue
            finally:
                if progress is not None:
                    progress.clear_status()
            worker_wait_time = time.perf_counter() - worker_wait_start

            actions_sent = int(step_response.get("actions_sent", 0))
            get_obs_time = float(step_response.get("get_obs_time", 0.0))
            state["metrics"]["worker_step_wait_time"] += worker_wait_time
            state["metrics"]["env_step_time"] += float(step_response.get("env_step_time", 0.0))
            state["metrics"]["get_obs_time"] += get_obs_time
            state["metrics"]["render_obs_time"] += get_obs_time
            state["metrics"]["step_chunk_time"] += float(
                step_response.get(
                    "step_chunk_time",
                    float(step_response.get("env_step_time", 0.0)) + get_obs_time,
                )
            )
            state["steps"] += actions_sent
            state["reward_sum"] += float(step_response.get("reward_sum", 0.0))
            state["done"] = bool(step_response.get("done", False))
            state["last_info"] = step_response.get("info", {})
            state["success"] = bool(state["last_info"].get("is_success", state["success"]))
            keyframes = step_response.get("keyframe_obs") or []
            if keyframes:
                state["obs_sequence"] = keyframes[-KEYFRAMES_PER_CHUNK:]
                state["keyframe_count"] += len(keyframes)
            elif step_response.get("obs") is not None:
                state["obs_sequence"] = [step_response["obs"]]
                state["keyframe_count"] += 1
            if actions_sent <= 0:
                state["done"] = True

        if progress is not None:
            active_after_step = sum(1 for state in states if not bool(state["done"]))
            successes = sum(1 for state in states if bool(state["success"]))
            errors = sum(1 for state in states if state.get("error"))
            progress.wave_step(
                task_index=task_index,
                task=task,
                wave_id=wave_id,
                total_waves=total_waves,
                infer_index=infer_index + 1,
                max_infer=max_infer,
                active=active_after_step,
                batch_size=batch_size,
                successes=successes,
                errors=errors,
                infer_time=infer_time,
                wave_elapsed=time.perf_counter() - wave_started_at,
                force=infer_index == 0 or active_after_step == 0,
            )

        if idle_count == batch_size:
            break

    completed_at = time.perf_counter()
    if args.save_video:
        pending_video_states: list[dict[str, Any]] = []
        for state in states:
            try:
                state["worker"].conn.send({"cmd": "finish_episode"})
                pending_video_states.append(state)
            except Exception as exc:
                last_info = state.get("last_info", {})
                if not isinstance(last_info, dict):
                    last_info = {"last_info": last_info}
                last_info["video_error"] = f"{exc.__class__.__name__}: {exc}"
                state["last_info"] = last_info
        for state in pending_video_states:
            if progress is not None:
                progress.set_status(
                    f"task {task_index}/{progress.total_tasks} {task} wave {wave_id + 1}/{total_waves} "
                    f"save_video worker={state['worker'].worker_id} episode={state['episode_index']}"
                )
            try:
                video_response = recv_worker(state["worker"], args.worker_timeout)
                state["video_path"] = video_response.get("video_path") or state.get("video_path")
                state["video_frames"] = int(video_response.get("video_frames", 0))
                state["metrics"]["video_write_time"] += float(video_response.get("video_write_time", 0.0))
                state["metrics"]["finish_episode_time"] += float(video_response.get("finish_episode_time", 0.0))
                state["metrics"]["write_log_video_time"] += float(video_response.get("video_write_time", 0.0))
            except Exception as exc:
                last_info = state.get("last_info", {})
                if not isinstance(last_info, dict):
                    last_info = {"last_info": last_info}
                last_info["video_error"] = f"{exc.__class__.__name__}: {exc}"
                state["last_info"] = last_info
            finally:
                if progress is not None:
                    progress.clear_status()

    results.extend(finalize_episode(args, state, completed_at) for state in states)
    return results


def run_microbatch_wave(
    args: argparse.Namespace,
    *,
    task: str,
    task_index: int,
    workers: list[WorkerHandle],
    clients: list[WebsocketClientPolicy] | None,
    episode_indices: list[int],
    episode_length: int,
    task_dir: Path,
    wave_id: int,
    total_waves: int,
    progress: ProgressReporter | None,
) -> list[dict[str, Any]]:
    planned_batch_size = len(episode_indices)
    if planned_batch_size <= 0:
        return []
    wave_started_at = time.perf_counter()
    wave_workers = workers[:planned_batch_size]
    wave_clients = clients[:planned_batch_size] if clients is not None else [None] * planned_batch_size
    results: list[dict[str, Any]] = []
    reset_starts: dict[int, float] = {}

    if progress is not None:
        progress.wave_phase(
            task_index=task_index,
            task=task,
            wave_id=wave_id,
            total_waves=total_waves,
            phase="start",
            detail=f"episodes={min(episode_indices)}-{max(episode_indices)} sessions={planned_batch_size}",
        )

    for batch_index, (worker, episode_index) in enumerate(zip(wave_workers, episode_indices, strict=True)):
        try:
            cache_entry = get_seed_cache_entry(args, task, episode_index)
            if args.use_seed_cache and cache_entry is None:
                raise RuntimeError(f"Missing seed cache entry task={task!r} episode_index={episode_index}")
            reset_starts[batch_index] = send_reset_worker_for_episode(
                args,
                task=task,
                worker=worker,
                episode_index=episode_index,
                episode_length=episode_length,
                task_dir=task_dir,
                cache_entry=cache_entry,
            )
        except Exception as exc:
            terminate_worker(worker)
            results.append(
                error_episode_result(
                    args,
                    task=task,
                    episode_index=episode_index,
                    worker=worker,
                    exc=exc,
                    wave_id=wave_id,
                    batch_index=batch_index,
                    wave_batch_size=planned_batch_size,
                )
            )

    states: list[dict[str, Any]] = []
    for batch_index, (worker, episode_index) in enumerate(zip(wave_workers, episode_indices, strict=True)):
        if batch_index not in reset_starts:
            continue
        if progress is not None:
            progress.set_status(
                f"task {task_index}/{progress.total_tasks} {task} microbatch wave {wave_id + 1}/{total_waves} "
                f"reset worker={worker.worker_id} episode={episode_index}"
            )
        try:
            reset_response = recv_worker(worker, args.worker_timeout)
            state = make_wave_state(
                args,
                task=task,
                worker=worker,
                episode_index=episode_index,
                episode_length=episode_length,
                task_dir=task_dir,
                wave_id=wave_id,
                batch_index=batch_index,
                wave_batch_size=planned_batch_size,
                reset_start=reset_starts[batch_index],
                reset_response=reset_response,
            )
            seed = int(state["seed"])
            state["session_id"] = f"robotwin-microbatch-{task}-{seed}-{uuid.uuid4().hex[:8]}"
            state["client"] = wave_clients[batch_index]
            state["num_envs_requested"] = int(args.num_envs)
            state["wave_batch_size"] = planned_batch_size
            states.append(state)
        except Exception as exc:
            terminate_worker(worker)
            results.append(
                error_episode_result(
                    args,
                    task=task,
                    episode_index=episode_index,
                    worker=worker,
                    exc=exc,
                    wave_id=wave_id,
                    batch_index=batch_index,
                    wave_batch_size=planned_batch_size,
                )
            )
        finally:
            if progress is not None:
                progress.clear_status()

    if not states:
        return results

    if clients is not None:
        for state in states:
            client = state.get("client")
            if client is None:
                continue
            if progress is not None:
                progress.set_status(
                    f"task {task_index}/{progress.total_tasks} {task} reset server session={state['session_id']}"
                )
            try:
                client.reset({"session_id": state["session_id"], "prompt": state["prompt"]})
            except Exception as exc:
                state["done"] = True
                state["error"] = f"{exc.__class__.__name__}: {exc}"
            finally:
                if progress is not None:
                    progress.clear_status()

    max_infer = args.max_steps if args.max_steps > 0 else max(int(episode_length), 1)
    if progress is not None:
        progress.wave_phase(
            task_index=task_index,
            task=task,
            wave_id=wave_id,
            total_waves=total_waves,
            phase="infer",
            detail=f"sessions={len(states)} max_infer={max_infer}",
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(len(states), 1)) as executor:
        for infer_index in range(max_infer):
            active_indices = [idx for idx, state in enumerate(states) if not bool(state["done"])]
            if not active_indices:
                break

            request_info: dict[int, dict[str, Any]] = {}
            action_by_index: dict[int, np.ndarray] = {}
            infer_elapsed_values: list[float] = []
            for batch_index in active_indices:
                state = states[batch_index]
                payload_build_start = time.perf_counter()
                payload = robotwin_obs_sequence_to_payload(
                    state["obs_sequence"],
                    state["prompt"],
                    state["session_id"],
                    image_resolution=args.resolved_client_image_resolution,
                )
                payload_build_time = time.perf_counter() - payload_build_start
                payload_pack_start = time.perf_counter()
                size_bytes = payload_size_bytes(payload)
                payload_pack_time = time.perf_counter() - payload_pack_start
                request_info[batch_index] = {
                    "payload": payload,
                    "payload_build_time": payload_build_time,
                    "payload_pack_time": payload_pack_time,
                    "size_bytes": size_bytes,
                }

            if args.dry_run_actions:
                for batch_index in active_indices:
                    action_by_index[batch_index] = np.zeros(
                        (args.open_loop_horizon, ROBOTWIN_ACTION_DIM),
                        dtype=np.float32,
                    )
                    infer_elapsed_values.append(0.0)
            else:
                futures: dict[concurrent.futures.Future, int] = {}
                for batch_index in active_indices:
                    state = states[batch_index]
                    client = state.get("client")
                    if client is None:
                        state["done"] = True
                        state["error"] = "missing websocket client"
                        continue
                    payload = request_info[batch_index]["payload"]

                    def infer_one(
                        ws_client: WebsocketClientPolicy,
                        request_payload: dict[str, Any],
                    ) -> tuple[np.ndarray, float, float, dict[str, Any], dict[str, Any]]:
                        infer_start = time.perf_counter()
                        raw_action, client_timing, server_timing = ws_client.infer_timed(dict(request_payload))
                        ws_roundtrip_time = time.perf_counter() - infer_start
                        normalize_start = time.perf_counter()
                        action_chunk = normalize_action_sequence(raw_action)
                        action_normalize_time = time.perf_counter() - normalize_start
                        return action_chunk, ws_roundtrip_time, action_normalize_time, client_timing, server_timing

                    futures[executor.submit(infer_one, client, payload)] = batch_index

                for future in concurrent.futures.as_completed(futures):
                    batch_index = futures[future]
                    state = states[batch_index]
                    try:
                        action_chunk, ws_roundtrip_time, action_normalize_time, client_timing, server_timing = future.result()
                    except Exception as exc:
                        state["done"] = True
                        state["error"] = f"{exc.__class__.__name__}: {exc}"
                        terminate_worker(state["worker"])
                        continue
                    action_by_index[batch_index] = action_chunk
                    infer_time = ws_roundtrip_time + action_normalize_time
                    infer_elapsed_values.append(infer_time)
                    info = request_info[batch_index]
                    metrics = state["metrics"]
                    state["infer_calls"] += 1
                    metrics["infer_wait_time"] += infer_time
                    metrics["infer_payload_size_bytes"] += int(info["size_bytes"])
                    metrics["payload_build_time"] += float(info["payload_build_time"])
                    metrics["payload_pack_time"] += float(info["payload_pack_time"])
                    metrics["ws_roundtrip_time"] += ws_roundtrip_time
                    metrics["action_normalize_time"] += action_normalize_time
                    metrics["batch_infer_time"] += infer_time
                    metrics["per_env_infer_time"] += infer_time
                    metrics["batch_infer_calls"] += 1
                    add_client_server_timing(metrics, client_timing, server_timing)

            if args.dry_run_actions:
                for batch_index in active_indices:
                    state = states[batch_index]
                    info = request_info[batch_index]
                    metrics = state["metrics"]
                    state["infer_calls"] += 1
                    metrics["infer_payload_size_bytes"] += int(info["size_bytes"])
                    metrics["payload_build_time"] += float(info["payload_build_time"])
                    metrics["payload_pack_time"] += float(info["payload_pack_time"])
                    metrics["batch_infer_calls"] += 1

            for batch_index, action_chunk in action_by_index.items():
                state = states[batch_index]
                state["action_shapes"].append(list(action_chunk.shape))
                if state["first_action"] is None and action_chunk.size:
                    state["first_action"] = action_chunk[0].tolist()
                try:
                    state["worker"].conn.send(
                        {
                            "cmd": "step_chunk",
                            "actions": action_chunk,
                            "need_obs": True,
                            "clip_action": bool(args.clip_action),
                            "max_keyframes": KEYFRAMES_PER_CHUNK,
                        }
                    )
                except Exception as exc:
                    state["done"] = True
                    state["error"] = f"{exc.__class__.__name__}: {exc}"
                    terminate_worker(state["worker"])

            for batch_index in list(action_by_index):
                state = states[batch_index]
                if state.get("error"):
                    continue
                worker_wait_start = time.perf_counter()
                if progress is not None:
                    progress.set_status(
                        f"task {task_index}/{progress.total_tasks} {task} microbatch wave {wave_id + 1}/{total_waves} "
                        f"step worker={state['worker'].worker_id} episode={state['episode_index']}"
                    )
                try:
                    step_response = recv_worker(state["worker"], args.worker_timeout)
                except Exception as exc:
                    state["done"] = True
                    state["error"] = f"{exc.__class__.__name__}: {exc}"
                    terminate_worker(state["worker"])
                    continue
                finally:
                    if progress is not None:
                        progress.clear_status()
                worker_wait_time = time.perf_counter() - worker_wait_start

                actions_sent = int(step_response.get("actions_sent", 0))
                get_obs_time = float(step_response.get("get_obs_time", 0.0))
                state["metrics"]["worker_step_wait_time"] += worker_wait_time
                state["metrics"]["env_step_time"] += float(step_response.get("env_step_time", 0.0))
                state["metrics"]["get_obs_time"] += get_obs_time
                state["metrics"]["render_obs_time"] += get_obs_time
                state["metrics"]["step_chunk_time"] += float(
                    step_response.get(
                        "step_chunk_time",
                        float(step_response.get("env_step_time", 0.0)) + get_obs_time,
                    )
                )
                state["steps"] += actions_sent
                state["reward_sum"] += float(step_response.get("reward_sum", 0.0))
                state["done"] = bool(step_response.get("done", False))
                state["last_info"] = step_response.get("info", {})
                state["success"] = bool(state["last_info"].get("is_success", state["success"]))
                keyframes = step_response.get("keyframe_obs") or []
                if keyframes:
                    state["obs_sequence"] = keyframes[-KEYFRAMES_PER_CHUNK:]
                    state["keyframe_count"] += len(keyframes)
                elif step_response.get("obs") is not None:
                    state["obs_sequence"] = [step_response["obs"]]
                    state["keyframe_count"] += 1
                if actions_sent <= 0:
                    state["done"] = True

            if progress is not None:
                active_after_step = sum(1 for state in states if not bool(state["done"]))
                successes = sum(1 for state in states if bool(state["success"]))
                errors = sum(1 for state in states if state.get("error"))
                progress.wave_step(
                    task_index=task_index,
                    task=task,
                    wave_id=wave_id,
                    total_waves=total_waves,
                    infer_index=infer_index + 1,
                    max_infer=max_infer,
                    active=active_after_step,
                    batch_size=len(states),
                    successes=successes,
                    errors=errors,
                    infer_time=max(infer_elapsed_values) if infer_elapsed_values else 0.0,
                    wave_elapsed=time.perf_counter() - wave_started_at,
                    force=infer_index == 0 or active_after_step == 0,
                )

    completed_at = time.perf_counter()
    if args.save_video:
        pending_video_states: list[dict[str, Any]] = []
        for state in states:
            try:
                state["worker"].conn.send({"cmd": "finish_episode"})
                pending_video_states.append(state)
            except Exception as exc:
                last_info = state.get("last_info", {})
                if not isinstance(last_info, dict):
                    last_info = {"last_info": last_info}
                last_info["video_error"] = f"{exc.__class__.__name__}: {exc}"
                state["last_info"] = last_info
        for state in pending_video_states:
            try:
                video_response = recv_worker(state["worker"], args.worker_timeout)
                state["video_path"] = video_response.get("video_path") or state.get("video_path")
                state["video_frames"] = int(video_response.get("video_frames", 0))
                state["metrics"]["video_write_time"] += float(video_response.get("video_write_time", 0.0))
                state["metrics"]["finish_episode_time"] += float(video_response.get("finish_episode_time", 0.0))
                state["metrics"]["write_log_video_time"] += float(video_response.get("video_write_time", 0.0))
            except Exception as exc:
                last_info = state.get("last_info", {})
                if not isinstance(last_info, dict):
                    last_info = {"last_info": last_info}
                last_info["video_error"] = f"{exc.__class__.__name__}: {exc}"
                state["last_info"] = last_info

    results.extend(finalize_episode(args, state, completed_at) for state in states)
    return results


def run_task(
    args: argparse.Namespace,
    *,
    task: str,
    task_index: int,
    workers: list[WorkerHandle],
    client: WebsocketClientPolicy | list[WebsocketClientPolicy] | None,
    progress: ProgressReporter | None,
) -> list[dict[str, Any]]:
    task_dir = args.output_dir / task
    task_dir.mkdir(parents=True, exist_ok=True)
    episode_length = load_task_step_limit(task, args.episode_length)
    results: list[dict[str, Any]] = []
    total_waves = (
        args.episodes
        if args.eval_mode == "lingbot"
        else max((args.episodes + args.num_envs - 1) // args.num_envs, 1)
    )
    task_started_at = time.perf_counter()
    if progress is not None:
        progress.task_start(
            task_index=task_index,
            task=task,
            episode_length=episode_length,
            total_waves=total_waves,
        )

    if args.eval_mode == "lingbot":
        worker = workers[0]
        single_client = client if isinstance(client, WebsocketClientPolicy) else None
        seed_cursor = int(args.seed_start)
        for episode_index in range(args.episodes):
            wave_started_at = time.perf_counter()
            try:
                result = run_episode(
                    args,
                    task=task,
                    worker=worker,
                    client=single_client,
                    episode_index=episode_index,
                    episode_length=episode_length,
                    task_dir=task_dir,
                    wave_id=episode_index,
                    seed_start_override=seed_cursor,
                )
                seed_cursor = int(result.get("seed", seed_cursor)) + 1
                episode_results = [result]
            except Exception as exc:
                episode_results = [
                    error_episode_result(
                        args,
                        task=task,
                        episode_index=episode_index,
                        worker=worker,
                        exc=exc,
                        wave_id=episode_index,
                        batch_index=0,
                        wave_batch_size=1,
                    )
                ]
                seed_cursor += max(int(args.expert_filter_max_candidates), int(args.reset_retries) + 1, 1)

            for result in episode_results:
                write_episode_result(task_dir, result)
                results.append(result)
            if progress is not None:
                progress.wave_done(
                    task_index=task_index,
                    task=task,
                    wave_id=episode_index,
                    total_waves=total_waves,
                    results=episode_results,
                    task_completed=len(results),
                    wave_elapsed=time.perf_counter() - wave_started_at,
                )

        if single_client is not None:
            with contextlib.suppress(Exception):
                single_client.reset({"session_id": f"robotwin-lingbot-{task}-final-flush"})
        write_task_summary(task_dir, task, results)
        if progress is not None:
            progress.task_done(
                task_index=task_index,
                task=task,
                results=results,
                elapsed=time.perf_counter() - task_started_at,
        )
        return results

    if args.eval_mode == "microbatch":
        microbatch_clients = client if isinstance(client, list) else None
        wave_id = 0
        for episode_start in range(0, args.episodes, args.num_envs):
            episode_indices = list(range(episode_start, min(episode_start + args.num_envs, args.episodes)))
            wave_started_at = time.perf_counter()
            try:
                wave_results = run_microbatch_wave(
                    args,
                    task=task,
                    task_index=task_index,
                    workers=workers,
                    clients=microbatch_clients,
                    episode_indices=episode_indices,
                    episode_length=episode_length,
                    task_dir=task_dir,
                    wave_id=wave_id,
                    total_waves=total_waves,
                    progress=progress,
                )
            except Exception as exc:
                wave_results = [
                    error_episode_result(
                        args,
                        task=task,
                        episode_index=episode_index,
                        worker=workers[batch_index % len(workers)],
                        exc=exc,
                        wave_id=wave_id,
                        batch_index=batch_index,
                        wave_batch_size=len(episode_indices),
                    )
                    for batch_index, episode_index in enumerate(episode_indices)
                ]
            for result in sorted(wave_results, key=lambda item: int(item.get("episode_index", 0))):
                write_episode_result(task_dir, result)
                results.append(result)
            if progress is not None:
                progress.wave_done(
                    task_index=task_index,
                    task=task,
                    wave_id=wave_id,
                    total_waves=total_waves,
                    results=wave_results,
                    task_completed=len(results),
                    wave_elapsed=time.perf_counter() - wave_started_at,
                )
            wave_id += 1

        if microbatch_clients is not None:
            for item in microbatch_clients:
                with contextlib.suppress(Exception):
                    item.reset({"reset_all": True})
        write_task_summary(task_dir, task, results)
        if progress is not None:
            progress.task_done(
                task_index=task_index,
                task=task,
                results=results,
                elapsed=time.perf_counter() - task_started_at,
            )
        return results

    single_client = client if isinstance(client, WebsocketClientPolicy) else None
    wave_id = 0
    for episode_start in range(0, args.episodes, args.num_envs):
        episode_indices = list(range(episode_start, min(episode_start + args.num_envs, args.episodes)))
        wave_started_at = time.perf_counter()
        try:
            wave_results = run_wave(
                args,
                task=task,
                task_index=task_index,
                workers=workers,
                client=single_client,
                episode_indices=episode_indices,
                episode_length=episode_length,
                task_dir=task_dir,
                wave_id=wave_id,
                total_waves=total_waves,
                progress=progress,
            )
        except Exception as exc:
            wave_results = [
                error_episode_result(
                    args,
                    task=task,
                    episode_index=episode_index,
                    worker=workers[batch_index % len(workers)],
                    exc=exc,
                    wave_id=wave_id,
                    batch_index=batch_index,
                    wave_batch_size=len(episode_indices),
                )
                for batch_index, episode_index in enumerate(episode_indices)
            ]
        for result in sorted(wave_results, key=lambda item: int(item.get("episode_index", 0))):
            write_episode_result(task_dir, result)
            results.append(result)
        if progress is not None:
            progress.wave_done(
                task_index=task_index,
                task=task,
                wave_id=wave_id,
                total_waves=total_waves,
                results=wave_results,
                task_completed=len(results),
                wave_elapsed=time.perf_counter() - wave_started_at,
            )
        wave_id += 1

    if single_client is not None:
        with contextlib.suppress(Exception):
            single_client.reset({"session_id": f"robotwin-lingbot-{task}-final-flush"})
    write_task_summary(task_dir, task, results)
    if progress is not None:
        progress.task_done(
            task_index=task_index,
            task=task,
            results=results,
            elapsed=time.perf_counter() - task_started_at,
        )
    return results


def finalize_gpu_summary(raw: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for gpu, item in sorted(raw.items()):
        episodes = int(item.get("episodes", 0))
        wall = float(item.get("episode_wall_time", 0.0))
        env_step = float(item.get("env_step_time", 0.0))
        get_obs = float(item.get("get_obs_time", 0.0))
        workers = sorted(str(worker) for worker in item.get("workers", set()) if str(worker))
        summary[gpu] = {
            "episodes": episodes,
            "workers": workers,
            "avg_env_step_time": env_step / episodes if episodes else 0.0,
            "avg_get_obs_time": get_obs / episodes if episodes else 0.0,
            "avg_episode_wall_time": wall / episodes if episodes else 0.0,
        }
    return summary


def write_report(output_root: Path, expected_episodes: int) -> None:
    task_dirs = sorted(p for p in output_root.iterdir() if p.is_dir() and (p / "summary.json").exists())
    rows: list[dict[str, Any]] = []
    total_success = 0
    total_episodes = 0
    complete_tasks = 0
    global_gpu_summary: dict[str, dict[str, Any]] = {}

    for task_dir in task_dirs:
        episode_paths = sorted(task_dir.glob("episode_*.json"))
        episode_results: list[dict[str, Any]] = []
        successes = 0
        rewards: list[float] = []
        steps: list[int] = []
        infer_calls: list[int] = []
        keyframes: list[int] = []
        wave_ids: set[int] = set()
        wave_batch_sizes_by_id: dict[int, int] = {}
        reset_times: list[float] = []
        expert_times: list[float] = []
        infer_times: list[float] = []
        batch_call_counts: list[int] = []
        per_env_batch_times: list[float] = []
        microbatch_calls_list: list[int] = []
        avg_microbatch_sizes: list[float] = []
        max_microbatch_sizes: list[int] = []
        avg_session_queue_depths: list[float] = []
        max_session_queue_depths: list[int] = []
        server_vram_allocated_values: list[float] = []
        server_vram_reserved_values: list[float] = []
        sync_idle_counts: list[int] = []
        env_times: list[float] = []
        obs_times: list[float] = []
        wall_times: list[float] = []
        payload_sizes: list[int] = []
        profile_timer_values: dict[str, list[float]] = {field: [] for field in PROFILE_TIMER_FIELDS}
        task_gpu_summary: dict[str, dict[str, Any]] = {}
        for path in episode_paths:
            data = json.loads(path.read_text(encoding="utf-8"))
            episode_results.append(data)
            timing = data.get("timing", {})
            profile = data.get("profile") or {}
            worker_gpu = str(data.get("worker_cuda_visible_devices", "unknown") or "unknown")
            successes += int(bool(data.get("success")))
            rewards.append(float(data.get("reward_sum", 0.0)))
            steps.append(int(data.get("steps", 0)))
            infer_calls.append(int(data.get("infer_calls", 0)))
            keyframes.append(int(data.get("keyframe_count", 0)))
            wave_id = int(data.get("wave_id", -1))
            if wave_id >= 0:
                wave_ids.add(wave_id)
                wave_batch_sizes_by_id[wave_id] = max(
                    wave_batch_sizes_by_id.get(wave_id, 0),
                    int(data.get("wave_batch_size", 1)),
                )
            reset_times.append(float(timing.get("reset_time", 0.0)))
            expert_times.append(float(timing.get("expert_filter_time", 0.0)))
            infer_times.append(float(timing.get("infer_wait_time", 0.0)))
            batch_call_counts.append(int(data.get("batch_infer_calls", data.get("infer_calls", 0))))
            per_env_batch_times.append(float(timing.get("per_env_infer_time", 0.0)))
            microbatch_calls_list.append(int(timing.get("microbatch_calls", 0)))
            avg_microbatch_sizes.append(float(timing.get("avg_microbatch_size", 0.0)))
            max_microbatch_sizes.append(int(timing.get("max_microbatch_size", 0)))
            avg_session_queue_depths.append(float(timing.get("avg_session_queue_depth", 0.0)))
            max_session_queue_depths.append(int(timing.get("max_session_queue_depth", 0)))
            server_vram_allocated_values.append(float(timing.get("server_vram_allocated_mb_max", 0.0)))
            server_vram_reserved_values.append(float(timing.get("server_vram_reserved_mb_max", 0.0)))
            sync_idle_counts.append(int(data.get("sync_idle_steps", 0)))
            env_times.append(float(timing.get("env_step_time", 0.0)))
            obs_times.append(float(timing.get("get_obs_time", 0.0)))
            wall_times.append(float(timing.get("episode_wall_time", 0.0)))
            payload_sizes.append(int(timing.get("infer_payload_size_bytes", 0)))
            for field in PROFILE_TIMER_FIELDS:
                profile_timer_values[field].append(float(profile.get(field, 0.0) or 0.0))
            for summary in (task_gpu_summary, global_gpu_summary):
                item = summary.setdefault(
                    worker_gpu,
                    {
                        "episodes": 0,
                        "workers": set(),
                        "env_step_time": 0.0,
                        "get_obs_time": 0.0,
                        "episode_wall_time": 0.0,
                    },
                )
                item["episodes"] += 1
                item["workers"].add(str(data.get("worker_id", "")))
                item["env_step_time"] += float(timing.get("env_step_time", 0.0))
                item["get_obs_time"] += float(timing.get("get_obs_time", 0.0))
                item["episode_wall_time"] += float(timing.get("episode_wall_time", 0.0))

        count = len(episode_paths)
        total_success += successes
        total_episodes += count
        complete = count == expected_episodes
        complete_tasks += int(complete)
        wave_batch_stats = aggregate_wave_batch_stats(episode_results)
        total_batch_calls = int(wave_batch_stats["batch_infer_calls"])
        per_env_batch_call_count = sum(batch_call_counts)
        total_microbatch_calls = sum(microbatch_calls_list)
        weighted_microbatch_size = sum(
            size * calls for size, calls in zip(avg_microbatch_sizes, microbatch_calls_list, strict=True)
        )
        weighted_queue_depth = sum(
            depth * calls for depth, calls in zip(avg_session_queue_depths, microbatch_calls_list, strict=True)
        )
        wave_batch_sizes = list(wave_batch_sizes_by_id.values())
        row = {
            "task": task_dir.name,
            "episodes": count,
            "successes": successes,
            "success_rate": successes / count if count else 0.0,
            "avg_reward": sum(rewards) / count if count else 0.0,
            "avg_steps": sum(steps) / count if count else 0.0,
            "avg_infer_calls": sum(infer_calls) / count if count else 0.0,
            "avg_keyframes": sum(keyframes) / count if count else 0.0,
            "waves": len(wave_ids),
            "avg_wave_batch_size": sum(wave_batch_sizes) / count if count else 0.0,
            "episode_length_std": float(np.std(steps)) if steps else 0.0,
            "avg_reset_time": sum(reset_times) / count if count else 0.0,
            "avg_expert_filter_time": sum(expert_times) / count if count else 0.0,
            "avg_infer_wait_time": sum(infer_times) / count if count else 0.0,
            "batch_infer_calls": total_batch_calls,
            "avg_batch_infer_time": float(wave_batch_stats["avg_batch_infer_time"]),
            "avg_per_env_infer_time": sum(per_env_batch_times) / per_env_batch_call_count
            if per_env_batch_call_count
            else 0.0,
            "microbatch_calls": total_microbatch_calls,
            "avg_microbatch_size": weighted_microbatch_size / total_microbatch_calls
            if total_microbatch_calls
            else 0.0,
            "max_microbatch_size": max(max_microbatch_sizes) if max_microbatch_sizes else 0,
            "avg_session_queue_depth": weighted_queue_depth / total_microbatch_calls
            if total_microbatch_calls
            else 0.0,
            "max_session_queue_depth": max(max_session_queue_depths) if max_session_queue_depths else 0,
            "server_vram_allocated_mb_max": max(server_vram_allocated_values) if server_vram_allocated_values else 0.0,
            "server_vram_reserved_mb_max": max(server_vram_reserved_values) if server_vram_reserved_values else 0.0,
            "sync_idle_steps": sum(sync_idle_counts),
            "avg_env_step_time": sum(env_times) / count if count else 0.0,
            "avg_get_obs_time": sum(obs_times) / count if count else 0.0,
            "avg_episode_wall_time": sum(wall_times) / count if count else 0.0,
            "avg_infer_payload_size_bytes": sum(payload_sizes) / count if count else 0.0,
            "env_gpu_summary": json.dumps(finalize_gpu_summary(task_gpu_summary), ensure_ascii=False, sort_keys=True),
            "complete": complete,
        }
        for field, values in profile_timer_values.items():
            row[f"avg_{field}"] = sum(values) / len(values) if values else 0.0
        rows.append(row)

    report_mode = "lingbot_style_robotwin"
    if rows and any(int(row.get("microbatch_calls", 0) or 0) > 0 for row in rows):
        report_mode = "robotwin_microbatch_eval"
    summary = {
        "mode": report_mode,
        "output_root": output_root.as_posix(),
        "tasks_finished": len(rows),
        "tasks_complete": complete_tasks,
        "expected_episodes_per_task": expected_episodes,
        "episodes": total_episodes,
        "successes": total_success,
        "success_rate": total_success / total_episodes if total_episodes else 0.0,
        "waves": sum(int(row.get("waves", 0)) for row in rows),
        "batch_infer_calls": sum(int(row.get("batch_infer_calls", 0)) for row in rows),
        "microbatch_calls": sum(int(row.get("microbatch_calls", 0)) for row in rows),
        "avg_microbatch_size": (
            sum(float(row.get("avg_microbatch_size", 0.0)) * int(row.get("microbatch_calls", 0)) for row in rows)
            / sum(int(row.get("microbatch_calls", 0)) for row in rows)
            if sum(int(row.get("microbatch_calls", 0)) for row in rows)
            else 0.0
        ),
        "server_vram_allocated_mb_max": max(
            (float(row.get("server_vram_allocated_mb_max", 0.0)) for row in rows),
            default=0.0,
        ),
        "server_vram_reserved_mb_max": max(
            (float(row.get("server_vram_reserved_mb_max", 0.0)) for row in rows),
            default=0.0,
        ),
        "sync_idle_steps": sum(int(row.get("sync_idle_steps", 0)) for row in rows),
        "env_gpu_summary": finalize_gpu_summary(global_gpu_summary),
        "rows": rows,
    }
    (output_root / "report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    fieldnames = [
        "task",
        "episodes",
        "successes",
        "success_rate",
        "avg_reward",
        "avg_steps",
        "avg_infer_calls",
        "avg_keyframes",
        "waves",
        "avg_wave_batch_size",
        "episode_length_std",
        "avg_reset_time",
        "avg_expert_filter_time",
        "avg_infer_wait_time",
        "batch_infer_calls",
        "avg_batch_infer_time",
        "avg_per_env_infer_time",
        "microbatch_calls",
        "avg_microbatch_size",
        "max_microbatch_size",
        "avg_session_queue_depth",
        "max_session_queue_depth",
        "server_vram_allocated_mb_max",
        "server_vram_reserved_mb_max",
        "sync_idle_steps",
        "avg_env_step_time",
        "avg_get_obs_time",
        "avg_episode_wall_time",
        "avg_infer_payload_size_bytes",
        *[f"avg_{field}" for field in PROFILE_TIMER_FIELDS],
        "env_gpu_summary",
        "complete",
    ]
    with (output_root / "report.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive")
    if args.eval_mode == "lingbot" and args.num_envs != 1:
        raise ValueError(
            "Lingbot-style eval is single-trajectory only: use --num-envs 1, "
            "or pass --eval-mode batch for the legacy synchronized-wave evaluator."
        )
    if args.open_loop_horizon <= 0:
        raise ValueError("--open-loop-horizon must be positive for dry-run chunks")
    if args.save_video and args.video_fps <= 0:
        raise ValueError("--video-fps must be positive when --save-video is enabled")
    if args.progress_interval <= 0:
        raise ValueError("--progress-interval must be positive")

    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.seed_cache_root = resolve_seed_cache_root(args)
    if args.eval_mode == "microbatch":
        args.use_seed_cache = True
    args.resolved_client_image_resolution = None
    args.server_timing = {}
    log_dir = args.output_dir / "logs"
    tasks = parse_tasks(args.tasks)
    args.seed_cache_by_task = {}
    if args.use_seed_cache:
        args.seed_cache_by_task = {
            task: load_seed_cache_for_task(args, task)
            for task in tasks
        }
    progress = ProgressReporter(
        mode=args.progress,
        total_tasks=len(tasks),
        episodes_per_task=args.episodes,
        output_dir=args.output_dir,
        interval=args.progress_interval,
    )
    workers: list[WorkerHandle] = []
    client: WebsocketClientPolicy | list[WebsocketClientPolicy] | None = None
    server_metadata: dict[str, Any] | None = None
    all_results: list[dict[str, Any]] = []
    try:
        workers = start_workers(args, log_dir, progress=progress)
        if not args.dry_run_actions:
            progress.log(f"connecting policy server ws://{args.remote_host}:{args.remote_port}")
            progress.set_status(f"policy server connect ws://{args.remote_host}:{args.remote_port}")
            try:
                if args.eval_mode == "microbatch":
                    clients = [
                        WebsocketClientPolicy(host=args.remote_host, port=args.remote_port)
                        for _ in workers
                    ]
                    client = clients
                    server_metadata = clients[0].get_server_metadata() if clients else {}
                else:
                    single_client = WebsocketClientPolicy(host=args.remote_host, port=args.remote_port)
                    client = single_client
                    server_metadata = single_client.get_server_metadata()
                raw_server_timing = server_metadata.get("server_timing", {}) if server_metadata else {}
                args.server_timing = raw_server_timing if isinstance(raw_server_timing, dict) else {}
                progress.log(f"policy server ready metadata={server_metadata}")
            finally:
                progress.clear_status()
        else:
            progress.log("dry-run actions enabled; policy server connection skipped")
        args.resolved_client_image_resolution = resolve_client_image_resolution(
            args.client_image_resolution,
            server_metadata,
        )
        progress.log(f"client_image_resolution={args.resolved_client_image_resolution or 'none'}")
        write_run_config(args, tasks=tasks, workers=workers, server_metadata=server_metadata)
        for task_index, task in enumerate(tasks, start=1):
            task_results = run_task(
                args,
                task=task,
                task_index=task_index,
                workers=workers,
                client=client,
                progress=progress,
            )
            all_results.extend(task_results)
        write_report(args.output_dir, args.episodes)
    finally:
        if isinstance(client, list):
            for item in client:
                with contextlib.suppress(Exception):
                    item.close()
        elif client is not None:
            client.close()
        if workers:
            stop_workers(workers)
        progress.close()

    done = {
        "mode": "robotwin_microbatch_eval" if args.eval_mode == "microbatch" else "lingbot_style_robotwin",
        "tasks": tasks,
        "episodes": len(all_results),
        "output_dir": args.output_dir.as_posix(),
        "report_json": (args.output_dir / "report.json").as_posix(),
        "report_csv": (args.output_dir / "report.csv").as_posix(),
    }
    print(json.dumps(done, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
