#!/usr/bin/env python3
"""Multi-env RoboTwin eval controller for DreamZero websocket inference."""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import pickle
import secrets
import signal
import socket
import subprocess
import sys
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
    Path("/data/openpi/packages/openpi-client/src"),
):
    if extra.exists() and str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from eval_utils.policy_client import WebsocketClientPolicy  # noqa: E402
from example.robotwin.robotwin_fast_env import (  # noqa: E402
    ROBOTWIN_ACTION_DIM,
    parse_image_resolution,
    stack_robotwin_payloads,
)

try:
    from openpi_client import msgpack_numpy
except Exception:  # pragma: no cover - fallback for static checks without openpi-client.
    msgpack_numpy = None


DEFAULT_OUTPUT_ROOT = Path("/data/checkpoints/dreamzero/robotwin_eval_runs/parallel_eval")


@dataclass
class WorkerHandle:
    worker_id: int
    process: subprocess.Popen
    conn: Connection
    log_path: Path
    cuda_visible_devices: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-host", default="127.0.0.1")
    parser.add_argument("--remote-port", type=int, default=8000)
    parser.add_argument("--tasks", default=os.environ.get("TASKS", os.environ.get("TASK", "beat_block_hammer")))
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--env-cuda", default=os.environ.get("ENV_CUDA", "0"))
    parser.add_argument("--worker-python", default=sys.executable)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--episode-length", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0, help="Max inference requests per episode; 0 derives from episode length.")
    parser.add_argument("--open-loop-horizon", type=int, default=8)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--reset-retries", type=int, default=5)
    parser.add_argument("--clip-action", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run-actions", action="store_true", help="Use zero actions and do not connect to the policy server.")
    parser.add_argument("--worker-timeout", type=float, default=900.0)
    parser.add_argument("--checkpoint-label", default="")
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--save-video", action="store_true", help="Reserved for compatibility; parallel eval v1 does not save videos.")
    parser.add_argument("--profile", action="store_true", help="Write finer controller-side timing fields.")
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
    parts: list[str] = []
    for chunk in raw.replace(",", " ").split():
        chunk = chunk.strip()
        if chunk:
            parts.append(chunk)
    if not parts:
        raise ValueError("--tasks must contain at least one RoboTwin task")
    if len(parts) == 1 and parts[0].lower() == "all":
        return load_all_eval_tasks()
    if any(part.lower() == "all" for part in parts):
        raise ValueError("Use --tasks all by itself, or provide explicit task names")
    return parts


def parse_cuda_list(raw: str, count: int) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        values = [os.environ.get("CUDA_VISIBLE_DEVICES", "")]
    return [values[i % len(values)] for i in range(count)]


def build_worker_env(cuda_visible_devices: str) -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    pythonpath_parts = [
        str(REPO_ROOT),
        str(REPO_ROOT / "third_party" / "RoboTwin"),
        str(REPO_ROOT / "third_party" / "lerobot" / "src"),
        str(REPO_ROOT / "third_party" / "lerobot"),
        "/data/openpi/packages/openpi-client/src",
    ]
    existing = env.get("PYTHONPATH")
    if existing:
        pythonpath_parts.append(existing)
    env["PYTHONPATH"] = ":".join(pythonpath_parts)
    return env


def start_workers(args: argparse.Namespace, log_dir: Path) -> list[WorkerHandle]:
    log_dir.mkdir(parents=True, exist_ok=True)
    authkey = secrets.token_bytes(16)
    listener = Listener(("127.0.0.1", 0), backlog=max(args.num_envs, 16), authkey=authkey)
    with contextlib.suppress(AttributeError):
        listener._listener._socket.settimeout(1.0)
    host, port = listener.address
    cuda_values = parse_cuda_list(args.env_cuda, args.num_envs)
    worker_script = REPO_ROOT / "example" / "robotwin" / "parallel_env_worker.py"
    processes: dict[int, subprocess.Popen] = {}
    logs: dict[int, Path] = {}

    for worker_id in range(args.num_envs):
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
            proc = subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                env=build_worker_env(cuda_values[worker_id]),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            log_file.close()
        processes[worker_id] = proc
        logs[worker_id] = log_path

    workers: dict[int, WorkerHandle] = {}
    deadline = time.time() + args.worker_timeout
    try:
        while len(workers) < args.num_envs:
            if time.time() > deadline:
                raise TimeoutError(f"Timed out waiting for {args.num_envs} workers to connect")
            for worker_id, proc in processes.items():
                if worker_id not in workers and proc.poll() is not None:
                    raise RuntimeError(
                        f"Worker {worker_id} exited before connecting; see {logs[worker_id]}"
                    )
            try:
                conn = listener.accept()
            except socket.timeout:
                continue
            ready = conn.recv()
            if not ready.get("ok") or ready.get("type") != "ready":
                raise RuntimeError(f"Unexpected worker handshake: {ready}")
            worker_id = int(ready["worker_id"])
            workers[worker_id] = WorkerHandle(
                worker_id=worker_id,
                process=processes[worker_id],
                conn=conn,
                log_path=logs[worker_id],
                cuda_visible_devices=str(ready.get("cuda_visible_devices", cuda_values[worker_id])),
            )
    except Exception:
        for proc in processes.values():
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except Exception:
                    proc.terminate()
        raise
    finally:
        listener.close()

    return [workers[i] for i in range(args.num_envs)]


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
        try:
            worker.conn.close()
        except Exception:
            pass
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
                try:
                    return int(value.strip())
                except ValueError:
                    break
    return 300


def payload_size_bytes(payload: dict[str, Any]) -> int:
    payload_with_endpoint = dict(payload)
    payload_with_endpoint["endpoint"] = "infer"
    if msgpack_numpy is not None:
        return len(msgpack_numpy.Packer().pack(payload_with_endpoint))
    return len(pickle.dumps(payload_with_endpoint))


def normalize_action_batch(action: Any, batch_size: int) -> np.ndarray:
    arr = np.asarray(action, dtype=np.float32)
    if arr.ndim == 1:
        if batch_size != 1:
            raise ValueError(f"Received 1D action for batch_size={batch_size}: {arr.shape}")
        arr = arr.reshape(1, 1, -1)
    elif arr.ndim == 2:
        if batch_size != 1:
            raise ValueError(f"Received 2D action for batch_size={batch_size}: {arr.shape}")
        arr = arr[None, ...]
    elif arr.ndim != 3:
        raise ValueError(f"Expected action shape [B,H,14] or [H,14], got {arr.shape}")
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


def resolve_client_image_resolution(
    raw: str,
    server_metadata: dict[str, Any] | None,
) -> tuple[int, int] | None:
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
    payload = {
        "mode": "parallel_robotwin",
        "timestamp": time.time(),
        "git_commit": git_commit(),
        "argv": sys.argv,
        "args": vars(args),
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


def write_episode_result(task_dir: Path, result: dict[str, Any]) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    path = task_dir / f"episode_{int(result['episode_index']):06d}.json"
    path.write_text(json.dumps(result, indent=2, default=json_default, ensure_ascii=False), encoding="utf-8")


def write_task_summary(task_dir: Path, task: str, results: list[dict[str, Any]], batch_records: list[dict[str, Any]]) -> None:
    successes = sum(int(bool(result.get("success"))) for result in results)
    steps = [int(result.get("steps", 0)) for result in results]
    summary = {
        "mode": "parallel_robotwin",
        "task": task,
        "episodes": len(results),
        "successes": successes,
        "success_rate": successes / len(results) if results else 0.0,
        "avg_steps": float(np.mean(steps)) if steps else 0.0,
        "episode_length_mean": float(np.mean(steps)) if steps else 0.0,
        "episode_length_std": float(np.std(steps)) if steps else 0.0,
        "episode_length_min": int(np.min(steps)) if steps else 0,
        "episode_length_max": int(np.max(steps)) if steps else 0,
        "sync_idle_steps": int(sum(record.get("sync_idle_steps", 0) for record in batch_records)),
        "batch_records": batch_records,
        "batch_session_ids": sorted({str(record.get("batch_session_id", "")) for record in batch_records if record.get("batch_session_id")}),
        "timestamp": time.time(),
    }
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=json_default, ensure_ascii=False),
        encoding="utf-8",
    )


def error_episode_result(
    args: argparse.Namespace,
    *,
    task: str,
    prompt: str,
    episode_index: int,
    worker: WorkerHandle,
    exc: BaseException,
) -> dict[str, Any]:
    return {
        "episode_index": episode_index,
        "worker_id": worker.worker_id,
        "worker_cuda_visible_devices": worker.cuda_visible_devices,
        "task": task,
        "prompt": prompt,
        "steps": 0,
        "success": False,
        "reward_sum": 0.0,
        "done": False,
        "error": f"{exc.__class__.__name__}: {exc}",
        "checkpoint": checkpoint_result(args),
        "parallel_eval": True,
        "fast_eval": True,
        "env_reuse_mode": "cold_reset",
        "timing": {
            "reset_time": 0.0,
            "infer_wait_time": 0.0,
            "env_step_time": 0.0,
            "get_obs_time": 0.0,
            "episode_wall_time": 0.0,
            "infer_payload_size_bytes": 0,
        },
        "profile": {},
    }


def finalize_episode(args: argparse.Namespace, state: dict[str, Any], wave_end_time: float) -> dict[str, Any]:
    metrics = state["metrics"]
    if state.get("completed_at") is None:
        state["completed_at"] = wave_end_time
    episode_wall_time = float(state["completed_at"] - state["episode_start"])
    timing = {
        "reset_time": float(metrics.get("reset_time", 0.0)),
        "infer_wait_time": float(metrics.get("infer_wait_time", 0.0)),
        "env_step_time": float(metrics.get("env_step_time", 0.0)),
        "get_obs_time": float(metrics.get("get_obs_time", 0.0)),
        "episode_wall_time": episode_wall_time,
        "infer_payload_size_bytes": int(metrics.get("infer_payload_size_bytes", 0)),
    }
    profile = {
        "payload_build_time": float(metrics.get("payload_build_time", 0.0)),
        "payload_pack_time": float(metrics.get("payload_pack_time", 0.0)),
        "ws_roundtrip_time": float(metrics.get("ws_roundtrip_time", 0.0)),
        "action_normalize_time": float(metrics.get("action_normalize_time", 0.0)),
        "worker_step_wait_time": float(metrics.get("worker_step_wait_time", 0.0)),
    }
    return {
        "episode_index": int(state["episode_index"]),
        "worker_id": int(state["worker"].worker_id),
        "worker_cuda_visible_devices": state["worker"].cuda_visible_devices,
        "seed": int(state["seed"]),
        "reset_attempts": int(state["reset_attempts"]),
        "task": state["task"],
        "prompt": state["prompt"],
        "steps": int(state["steps"]),
        "success": bool(state["success"]),
        "reward_sum": float(state["reward_sum"]),
        "done": bool(state["done"]),
        "action_shapes": state["action_shapes"],
        "first_action": state["first_action"],
        "checkpoint": checkpoint_result(args),
        "parallel_eval": True,
        "fast_eval": True,
        "env_reuse_mode": "cold_reset",
        "batch_session_id": state["batch_session_id"],
        "sync_idle_steps": int(metrics.get("sync_idle_steps", 0)),
        "timing": timing,
        "profile": profile if args.profile else {},
        "last_info": state["last_info"],
    }


def run_task(
    args: argparse.Namespace,
    *,
    task: str,
    workers: list[WorkerHandle],
    client: WebsocketClientPolicy | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    task_dir = args.output_dir / task
    task_dir.mkdir(parents=True, exist_ok=True)
    prompt = task.replace("_", " ")
    episode_length = load_task_step_limit(task, args.episode_length)
    max_infer = args.max_steps if args.max_steps > 0 else math.ceil(episode_length / args.open_loop_horizon)
    results: list[dict[str, Any]] = []
    batch_records: list[dict[str, Any]] = []

    for wave_start in range(0, args.episodes, args.num_envs):
        wave_workers = workers[: min(args.num_envs, args.episodes - wave_start)]
        batch_session_id = f"robotwin-parallel-{task}-{wave_start}-{uuid.uuid4().hex[:8]}"
        states: list[dict[str, Any]] = []

        for offset, worker in enumerate(wave_workers):
            episode_index = wave_start + offset
            worker.conn.send(
                {
                    "cmd": "reset",
                    "task_name": task,
                    "episode_index": episode_index,
                    "episode_length": episode_length,
                    "seed_start": args.seed_start,
                    "episodes": args.episodes,
                    "reset_retries": args.reset_retries,
                }
            )

        for offset, worker in enumerate(wave_workers):
            episode_index = wave_start + offset
            try:
                reset_response = recv_worker(worker, args.worker_timeout)
            except Exception as exc:
                result = error_episode_result(
                    args,
                    task=task,
                    prompt=prompt,
                    episode_index=episode_index,
                    worker=worker,
                    exc=exc,
                )
                write_episode_result(task_dir, result)
                results.append(result)
                continue
            now = time.perf_counter()
            states.append(
                {
                    "worker": worker,
                    "episode_index": episode_index,
                    "task": task,
                    "prompt": prompt,
                    "seed": reset_response["seed"],
                    "reset_attempts": reset_response["reset_attempts"],
                    "obs": reset_response["obs"],
                    "done": False,
                    "success": False,
                    "reward_sum": 0.0,
                    "steps": 0,
                    "action_shapes": [],
                    "first_action": None,
                    "last_info": reset_response.get("info", {}),
                    "episode_start": now - float(reset_response.get("reset_time", 0.0)),
                    "completed_at": None,
                    "batch_session_id": batch_session_id,
                    "metrics": {
                        "reset_time": float(reset_response.get("reset_time", 0.0)),
                        "infer_wait_time": 0.0,
                        "env_step_time": 0.0,
                        "get_obs_time": 0.0,
                        "infer_payload_size_bytes": 0,
                        "sync_idle_steps": 0,
                        "payload_build_time": 0.0,
                        "payload_pack_time": 0.0,
                        "ws_roundtrip_time": 0.0,
                        "action_normalize_time": 0.0,
                        "worker_step_wait_time": 0.0,
                    },
                }
            )

        if not states:
            continue
        if client is not None:
            client.reset({"session_id": batch_session_id})

        for infer_idx in range(max_infer):
            active_mask = [not state["done"] for state in states]
            active_count = sum(int(flag) for flag in active_mask)
            if active_count == 0:
                break

            payload_build_start = time.perf_counter()
            payload = stack_robotwin_payloads(
                [state["obs"] for state in states],
                prompt,
                batch_session_id,
                image_resolution=args.resolved_client_image_resolution,
            )
            payload_build_time = time.perf_counter() - payload_build_start
            payload_pack_start = time.perf_counter()
            size_bytes = payload_size_bytes(payload)
            payload_pack_time = time.perf_counter() - payload_pack_start
            action_normalize_time = 0.0
            ws_roundtrip_time = 0.0
            if args.dry_run_actions:
                action_batch = np.zeros(
                    (len(states), args.open_loop_horizon, ROBOTWIN_ACTION_DIM),
                    dtype=np.float32,
                )
            else:
                assert client is not None
                infer_start = time.perf_counter()
                raw_action = client.infer(dict(payload))
                ws_roundtrip_time = time.perf_counter() - infer_start
                normalize_start = time.perf_counter()
                action_batch = normalize_action_batch(raw_action, len(states))
                action_normalize_time = time.perf_counter() - normalize_start
            infer_time = ws_roundtrip_time + action_normalize_time
            action_horizon = min(args.open_loop_horizon, int(action_batch.shape[1]))
            if action_horizon <= 0:
                raise ValueError(f"Server returned empty action horizon: {action_batch.shape}")

            batch_idle_steps = 0
            for state, was_active in zip(states, active_mask, strict=True):
                if was_active:
                    state["metrics"]["infer_wait_time"] += infer_time
                    state["metrics"]["infer_payload_size_bytes"] += size_bytes
                    state["metrics"]["payload_build_time"] += payload_build_time
                    state["metrics"]["payload_pack_time"] += payload_pack_time
                    state["metrics"]["ws_roundtrip_time"] += ws_roundtrip_time
                    state["metrics"]["action_normalize_time"] += action_normalize_time
                else:
                    batch_idle_steps += action_horizon
                    state["metrics"]["sync_idle_steps"] += action_horizon

            pending: list[tuple[dict[str, Any], WorkerHandle, int]] = []
            for env_i, (state, was_active) in enumerate(zip(states, active_mask, strict=True)):
                if not was_active:
                    continue
                chunk = action_batch[env_i, :action_horizon]
                state["action_shapes"].append(list(chunk.shape))
                if state["first_action"] is None and chunk.size:
                    state["first_action"] = chunk[0].tolist()
                worker = state["worker"]
                worker.conn.send(
                    {
                        "cmd": "step_chunk",
                        "actions": chunk,
                        "need_obs": True,
                        "clip_action": bool(args.clip_action),
                    }
                )
                pending.append((state, worker, action_horizon))

            batch_worker_wait_time = 0.0
            for state, worker, sent_horizon in pending:
                worker_wait_start = time.perf_counter()
                try:
                    step_response = recv_worker(worker, args.worker_timeout)
                except Exception as exc:
                    state["done"] = True
                    state["last_info"] = {"error": f"{exc.__class__.__name__}: {exc}"}
                    state["completed_at"] = time.perf_counter()
                    continue
                worker_wait_time = time.perf_counter() - worker_wait_start
                batch_worker_wait_time += worker_wait_time

                actions_sent = int(step_response.get("actions_sent", 0))
                idle_from_partial_chunk = max(sent_horizon - actions_sent, 0)
                batch_idle_steps += idle_from_partial_chunk
                state["metrics"]["sync_idle_steps"] += idle_from_partial_chunk
                state["metrics"]["worker_step_wait_time"] += worker_wait_time
                state["metrics"]["env_step_time"] += float(step_response.get("env_step_time", 0.0))
                state["metrics"]["get_obs_time"] += float(step_response.get("get_obs_time", 0.0))
                state["steps"] += actions_sent
                state["reward_sum"] += float(step_response.get("reward_sum", 0.0))
                state["done"] = bool(step_response.get("done", False))
                state["last_info"] = step_response.get("info", {})
                state["success"] = bool(state["last_info"].get("is_success", state["success"]))
                if step_response.get("obs") is not None:
                    state["obs"] = step_response["obs"]
                if state["done"] and state["completed_at"] is None:
                    state["completed_at"] = time.perf_counter()

            lengths = [int(state["steps"]) for state in states]
            batch_records.append(
                {
                    "task": task,
                    "wave_start_episode": wave_start,
                    "batch_session_id": batch_session_id,
                    "infer_index": infer_idx,
                    "batch_size": len(states),
                    "active_count": active_count,
                    "batch_infer_time": infer_time,
                    "per_env_infer_time": infer_time / max(len(states), 1),
                    "infer_payload_size_bytes": size_bytes,
                    "payload_build_time": payload_build_time,
                    "payload_pack_time": payload_pack_time,
                    "ws_roundtrip_time": ws_roundtrip_time,
                    "action_normalize_time": action_normalize_time,
                    "worker_step_wait_time": batch_worker_wait_time,
                    "client_image_resolution": list(args.resolved_client_image_resolution)
                    if args.resolved_client_image_resolution is not None
                    else None,
                    "episode_length_mean": float(np.mean(lengths)) if lengths else 0.0,
                    "episode_length_std": float(np.std(lengths)) if lengths else 0.0,
                    "episode_length_min": int(np.min(lengths)) if lengths else 0,
                    "episode_length_max": int(np.max(lengths)) if lengths else 0,
                    "sync_idle_steps": int(batch_idle_steps),
                }
            )

            if all(state["done"] for state in states):
                break

        wave_end = time.perf_counter()
        for state in states:
            result = finalize_episode(args, state, wave_end)
            write_episode_result(task_dir, result)
            results.append(result)

    write_task_summary(task_dir, task, results, batch_records)
    return results, batch_records


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


def gpu_wall_time_imbalance(summary: dict[str, dict[str, Any]]) -> dict[str, float]:
    values = [float(item.get("avg_episode_wall_time", 0.0)) for item in summary.values() if item.get("episodes", 0)]
    if not values:
        return {"min": 0.0, "max": 0.0, "ratio": 0.0}
    min_value = min(values)
    max_value = max(values)
    return {
        "min": min_value,
        "max": max_value,
        "ratio": max_value / min_value if min_value > 0 else 0.0,
    }


def write_report(output_root: Path, expected_episodes: int) -> None:
    task_dirs = sorted(p for p in output_root.iterdir() if p.is_dir() and (p / "summary.json").exists())
    rows: list[dict[str, Any]] = []
    total_success = 0
    total_episodes = 0
    complete_tasks = 0
    global_gpu_summary: dict[str, dict[str, Any]] = {}

    for task_dir in task_dirs:
        episode_paths = sorted(task_dir.glob("episode_*.json"))
        successes = 0
        rewards: list[float] = []
        steps: list[int] = []
        reset_times: list[float] = []
        infer_times: list[float] = []
        env_times: list[float] = []
        obs_times: list[float] = []
        wall_times: list[float] = []
        payload_sizes: list[int] = []
        idle_steps: list[int] = []
        task_gpu_summary: dict[str, dict[str, Any]] = {}
        for path in episode_paths:
            data = json.loads(path.read_text(encoding="utf-8"))
            timing = data.get("timing", {})
            worker_gpu = str(data.get("worker_cuda_visible_devices", "unknown") or "unknown")
            successes += int(bool(data.get("success")))
            rewards.append(float(data.get("reward_sum", 0.0)))
            steps.append(int(data.get("steps", 0)))
            reset_times.append(float(timing.get("reset_time", 0.0)))
            infer_times.append(float(timing.get("infer_wait_time", 0.0)))
            env_times.append(float(timing.get("env_step_time", 0.0)))
            obs_times.append(float(timing.get("get_obs_time", 0.0)))
            wall_times.append(float(timing.get("episode_wall_time", 0.0)))
            payload_sizes.append(int(timing.get("infer_payload_size_bytes", 0)))
            idle_steps.append(int(data.get("sync_idle_steps", 0)))
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
        task_gpu_json = finalize_gpu_summary(task_gpu_summary)
        total_success += successes
        total_episodes += count
        complete = count == expected_episodes
        complete_tasks += int(complete)
        rows.append(
            {
                "task": task_dir.name,
                "episodes": count,
                "successes": successes,
                "success_rate": successes / count if count else 0.0,
                "avg_reward": sum(rewards) / count if count else 0.0,
                "avg_steps": sum(steps) / count if count else 0.0,
                "episode_length_std": float(np.std(steps)) if steps else 0.0,
                "avg_reset_time": sum(reset_times) / count if count else 0.0,
                "avg_infer_wait_time": sum(infer_times) / count if count else 0.0,
                "avg_env_step_time": sum(env_times) / count if count else 0.0,
                "avg_get_obs_time": sum(obs_times) / count if count else 0.0,
                "avg_episode_wall_time": sum(wall_times) / count if count else 0.0,
                "avg_infer_payload_size_bytes": sum(payload_sizes) / count if count else 0.0,
                "sync_idle_steps": sum(idle_steps),
                "env_gpu_summary": json.dumps(task_gpu_json, ensure_ascii=False, sort_keys=True),
                "complete": complete,
            }
        )

    global_gpu_json = finalize_gpu_summary(global_gpu_summary)
    summary = {
        "output_root": output_root.as_posix(),
        "tasks_finished": len(rows),
        "tasks_complete": complete_tasks,
        "expected_episodes_per_task": expected_episodes,
        "episodes": total_episodes,
        "successes": total_success,
        "success_rate": total_success / total_episodes if total_episodes else 0.0,
        "env_gpu_summary": global_gpu_json,
        "env_gpu_wall_time_imbalance": gpu_wall_time_imbalance(global_gpu_json),
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
        "episode_length_std",
        "avg_reset_time",
        "avg_infer_wait_time",
        "avg_env_step_time",
        "avg_get_obs_time",
        "avg_episode_wall_time",
        "avg_infer_payload_size_bytes",
        "sync_idle_steps",
        "env_gpu_summary",
        "complete",
    ]
    with (output_root / "report.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.save_video:
        raise ValueError("parallel_eval v1 does not support --save-video; use SAVE_VIDEO=0")
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive")
    if args.open_loop_horizon <= 0:
        raise ValueError("--open-loop-horizon must be positive")

    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.resolved_client_image_resolution = None
    log_dir = args.output_dir / "logs"
    tasks = parse_tasks(args.tasks)
    workers = start_workers(args, log_dir)
    client: WebsocketClientPolicy | None = None
    server_metadata: dict[str, Any] | None = None
    all_results: list[dict[str, Any]] = []
    all_batches: list[dict[str, Any]] = []
    try:
        if not args.dry_run_actions:
            client = WebsocketClientPolicy(host=args.remote_host, port=args.remote_port)
            server_metadata = client.get_server_metadata()
        args.resolved_client_image_resolution = resolve_client_image_resolution(
            args.client_image_resolution,
            server_metadata,
        )
        write_run_config(args, tasks=tasks, workers=workers, server_metadata=server_metadata)
        for task in tasks:
            task_results, task_batches = run_task(args, task=task, workers=workers, client=client)
            all_results.extend(task_results)
            all_batches.extend(task_batches)
        write_report(args.output_dir, args.episodes)
    finally:
        if client is not None:
            client.close()
        stop_workers(workers)

    done = {
        "mode": "parallel_robotwin",
        "tasks": tasks,
        "episodes": len(all_results),
        "batches": len(all_batches),
        "output_dir": args.output_dir.as_posix(),
        "report_json": (args.output_dir / "report.json").as_posix(),
        "report_csv": (args.output_dir / "report.csv").as_posix(),
    }
    print(json.dumps(done, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
