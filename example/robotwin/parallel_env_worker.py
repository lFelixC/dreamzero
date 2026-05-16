#!/usr/bin/env python3
"""Subprocess RoboTwin worker for parallel DreamZero eval."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any

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

from example.robotwin.robotwin_fast_env import DreamZeroRoboTwinEnv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--authkey-hex", required=True)
    parser.add_argument("--worker-id", type=int, required=True)
    return parser.parse_args()


def ok(**payload: Any) -> dict[str, Any]:
    payload["ok"] = True
    return payload


def err(exc: BaseException) -> dict[str, Any]:
    return {
        "ok": False,
        "error_type": exc.__class__.__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }


def main() -> None:
    args = parse_args()
    conn = Client((args.host, args.port), authkey=bytes.fromhex(args.authkey_hex))
    runner: DreamZeroRoboTwinEnv | None = None
    conn.send(
        ok(
            type="ready",
            worker_id=args.worker_id,
            pid=os.getpid(),
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        )
    )

    while True:
        cmd = conn.recv()
        name = cmd.get("cmd")
        try:
            if name == "close":
                if runner is not None:
                    runner.close()
                conn.send(ok(type="closed", worker_id=args.worker_id))
                return

            if name == "reset":
                if runner is not None:
                    runner.close()
                runner = DreamZeroRoboTwinEnv(
                    task_name=str(cmd["task_name"]),
                    episode_length=int(cmd["episode_length"]),
                    seed_start=int(cmd.get("seed_start", 0)),
                    episodes=int(cmd.get("episodes", 1)),
                    reset_retries=int(cmd.get("reset_retries", 0)),
                )
                result = runner.reset(int(cmd["episode_index"]))
                conn.send(ok(type="reset", worker_id=args.worker_id, **result))
                continue

            if name == "step_chunk":
                if runner is None:
                    raise RuntimeError("step_chunk received before reset")
                result = runner.step_chunk(
                    cmd["actions"],
                    need_obs=bool(cmd.get("need_obs", True)),
                    clip_action=bool(cmd.get("clip_action", True)),
                )
                conn.send(ok(type="step_chunk", worker_id=args.worker_id, **result))
                continue

            if name == "ping":
                conn.send(ok(type="pong", worker_id=args.worker_id))
                continue

            raise ValueError(f"Unknown worker command: {name!r}")
        except BaseException as exc:
            conn.send(err(exc))


if __name__ == "__main__":
    main()
