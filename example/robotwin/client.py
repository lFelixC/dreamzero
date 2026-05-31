#!/usr/bin/env python3
"""RoboTwin client for the existing ALOHA bimanual DreamZero server.

The important contract is that this client sends split state keys:

    state.left_joint_pos, state.left_gripper_pos,
    state.right_joint_pos, state.right_gripper_pos

That bypasses the server's packed observation.state fallback and preserves the
RoboTwin/LeRobot `[left7, right7]` ordering.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
import uuid
import warnings
from pathlib import Path
from typing import Any

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
for extra in (
    REPO_ROOT / "third_party" / "lerobot",
    REPO_ROOT / "third_party" / "lerobot" / "src",
    REPO_ROOT / "third_party" / "RoboTwin",
):
    if extra.exists() and str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from eval_utils.policy_client import WebsocketClientPolicy  # noqa: E402
# Keep these names available from client.py while sharing the implementation.
from example.robotwin.robotwin_fast_env import (  # noqa: E402,F401
    ROBOTWIN_ACTION_DIM,
    ROBOTWIN_ACTION_HIGH,
    ROBOTWIN_ACTION_LOW,
    ROBOTWIN_CAMERA_H,
    ROBOTWIN_CAMERA_NAMES,
    ROBOTWIN_CAMERA_TO_DREAMZERO,
    ROBOTWIN_CAMERA_W,
    RoboTwinCompatEnv,
    coalesce,
    ensure_robotwin_workdir,
    first_existing,
    initialize_robotwin_eval_state,
    is_unstable_reset_error,
    load_robotwin_setup_kwargs,
    load_robotwin_task,
    make_arm_tag,
    make_robotwin_env,
    model_image_sequence,
    nested_get,
    normalize_image,
    payload_from_observation,
    robotwin_fast_step,
    split_left_right_state,
)

VIDEO_KEYS = {
    "cam_high": "observation.images.cam_high",
    "cam_left": "observation.images.cam_left",
    "cam_right": "observation.images.cam_right",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-host", default="127.0.0.1")
    parser.add_argument("--remote-port", type=int, default=8000)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--task-name", default="")
    parser.add_argument("--output-dir", type=Path, default=Path("/data/checkpoints/dreamzero/robotwin_eval_runs"))
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/data/datasets/dreamzero/robotwin_unified_dreamzero"),
        help="DreamZero-converted RoboTwin dataset used by --mode dataset.",
    )
    parser.add_argument(
        "--mode",
        choices=["dataset", "gym", "robotwin"],
        default="dataset",
        help=(
            "dataset replays converted LeRobot observations; gym steps a generic Gymnasium env; "
            "robotwin steps LeRobot's RoboTwinEnv directly."
        ),
    )
    parser.add_argument("--env-id", default=None, help="Gymnasium env id for --mode gym.")
    parser.add_argument("--env-kwargs-json", default="{}", help="JSON kwargs passed to gym.make.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Inference calls per episode. Defaults to 1 for dataset mode and --episode-length for live env modes.",
    )
    parser.add_argument("--open-loop-horizon", type=int, default=1, help="Gym actions to execute per inference chunk.")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--robotwin-task", default="beat_block_hammer", help="RoboTwin task name for --mode robotwin.")
    parser.add_argument("--episode-length", type=int, default=300, help="Max live-env steps per episode.")
    parser.add_argument("--seed-start", type=int, default=0, help="First seed/episode index for live RoboTwin runs.")
    parser.add_argument(
        "--reset-retries",
        type=int,
        default=5,
        help="Retry live RoboTwin reset with later seeds when scene initialization is unstable.",
    )
    parser.add_argument(
        "--fast-eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For --mode robotwin, skip camera rendering on open-loop intermediate actions. "
            "The final action before the next inference still renders a fresh observation."
        ),
    )
    parser.add_argument(
        "--clip-action",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Clip server actions to the live environment action space before stepping.",
    )
    parser.add_argument("--save-video", action="store_true", help="Save an RGB rollout video for live RoboTwin runs.")
    parser.add_argument("--checkpoint-label", default="", help="Optional checkpoint label written to result JSON.")
    parser.add_argument("--checkpoint-path", default="", help="Optional checkpoint path written to result JSON.")
    return parser.parse_args()


def json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def dataset_episode_paths(dataset_root: Path, episode_index: int, info: dict[str, Any]) -> tuple[Path, dict[str, Path]]:
    chunk_size = int(info.get("chunks_size", 1000))
    chunk_index = episode_index // chunk_size
    parquet_path = dataset_root / info["data_path"].format(
        episode_chunk=chunk_index,
        episode_index=episode_index,
    )
    video_paths = {
        out_key: dataset_root
        / info["video_path"].format(
            episode_chunk=chunk_index,
            episode_index=episode_index,
            video_key=video_key,
        )
        for out_key, video_key in VIDEO_KEYS.items()
    }
    return parquet_path, video_paths


def read_frame(path: Path, timestamp: float, fps: float) -> np.ndarray:
    frame_index = max(int(round(timestamp * fps)), 0)
    reader = imageio.get_reader(path.as_posix())
    try:
        frame_count = reader.count_frames()
        if frame_count > 0:
            frame_index = min(frame_index, frame_count - 1)
        return normalize_image(reader.get_data(frame_index))
    finally:
        reader.close()


def select_dataset_episodes(dataset_root: Path, task_name: str, count: int) -> list[dict[str, Any]]:
    episodes = read_jsonl(dataset_root / "meta" / "episodes.jsonl")
    if task_name:
        filtered = [
            episode for episode in episodes
            if any(task_name.lower() in str(task).lower() for task in episode.get("tasks", []))
        ]
        if filtered:
            episodes = filtered
    return episodes[:count]


def run_dataset_mode(args: argparse.Namespace, client: WebsocketClientPolicy) -> list[dict[str, Any]]:
    dataset_root = args.dataset_root.expanduser().resolve()
    info = read_json(dataset_root / "meta" / "info.json")
    episodes = select_dataset_episodes(dataset_root, args.task_name, args.episodes)
    results: list[dict[str, Any]] = []
    max_steps = 1 if args.max_steps is None else args.max_steps

    for episode in episodes:
        episode_index = int(episode["episode_index"])
        prompt = args.task_name or str((episode.get("tasks") or [""])[0])
        parquet_path, video_paths = dataset_episode_paths(dataset_root, episode_index, info)
        df = pd.read_parquet(parquet_path)
        session_id = f"robotwin-dataset-{episode_index}-{uuid.uuid4().hex[:8]}"
        client.reset({"session_id": session_id})

        episode_actions = []
        for step_idx in range(min(max_steps, len(df))):
            row = df.iloc[step_idx]
            timestamp = float(row["timestamp"])
            obs = {
                "observation.state": row["observation.state"],
                "observation.images.cam_high": read_frame(video_paths["cam_high"], timestamp, args.fps),
                "observation.images.cam_left": read_frame(video_paths["cam_left"], timestamp, args.fps),
                "observation.images.cam_right": read_frame(video_paths["cam_right"], timestamp, args.fps),
            }
            payload = payload_from_observation(obs, prompt, session_id)
            action = np.asarray(client.infer(payload), dtype=np.float32)
            if action.ndim == 1:
                action = action.reshape(1, -1)
            if action.shape[-1] != 14:
                raise ValueError(f"Expected action width 14, got {action.shape}")
            episode_actions.append(action)

        result = {
            "episode_index": episode_index,
            "prompt": prompt,
            "num_requests": len(episode_actions),
            "action_shapes": [list(action.shape) for action in episode_actions],
            "first_action": episode_actions[0][0].tolist() if episode_actions else None,
            "checkpoint": checkpoint_result(args),
        }
        results.append(result)
        write_episode_result(args.output_dir, result)
    return results


def run_gym_mode(args: argparse.Namespace, client: WebsocketClientPolicy) -> list[dict[str, Any]]:
    if not args.env_id:
        raise ValueError("--env-id is required for --mode gym")
    import gymnasium as gym

    env_kwargs = json.loads(args.env_kwargs_json)
    results = []
    max_steps = args.episode_length if args.max_steps is None else args.max_steps
    for episode_idx in range(args.episodes):
        env = gym.make(args.env_id, **env_kwargs)
        reset_result = env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        prompt = args.task_name or getattr(env.unwrapped, "task_name", "")
        session_id = f"robotwin-gym-{episode_idx}-{uuid.uuid4().hex[:8]}"
        client.reset({"session_id": session_id})
        actions_sent = 0
        rewards = []
        done = False
        for _ in range(max_steps):
            payload = payload_from_observation(obs, prompt, session_id)
            action_chunk = np.asarray(client.infer(payload), dtype=np.float32)
            if action_chunk.ndim == 1:
                action_chunk = action_chunk.reshape(1, -1)
            if action_chunk.shape[-1] != 14:
                raise ValueError(f"Expected action width 14, got {action_chunk.shape}")
            for action in action_chunk[: args.open_loop_horizon]:
                if args.clip_action and hasattr(env, "action_space"):
                    action = np.clip(action, env.action_space.low, env.action_space.high)
                step_result = env.step(action)
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                    done = bool(terminated or truncated)
                else:
                    obs, reward, done, info = step_result
                actions_sent += 1
                rewards.append(float(reward))
                if done:
                    break
            if done:
                break
        result = {
            "episode_index": episode_idx,
            "prompt": prompt,
            "actions_sent": actions_sent,
            "reward_sum": float(sum(rewards)),
            "done": done,
            "checkpoint": checkpoint_result(args),
        }
        results.append(result)
        write_episode_result(args.output_dir, result)
        env.close()
    return results


def checkpoint_result(args: argparse.Namespace) -> dict[str, str]:
    return {
        "label": str(args.checkpoint_label or ""),
        "path": str(args.checkpoint_path or ""),
    }


def robotwin_obs_to_payload(obs: dict[str, Any], prompt: str, session_id: str) -> dict[str, Any]:
    pixels = obs.get("pixels")
    if not isinstance(pixels, dict):
        raise KeyError("RoboTwin observation must contain a 'pixels' dict")

    mapped_obs: dict[str, Any] = {
        dreamzero_key: pixels.get(robotwin_key)
        for dreamzero_key, robotwin_key in ROBOTWIN_CAMERA_TO_DREAMZERO.items()
    }
    mapped_obs["state"] = obs.get("agent_pos")
    return payload_from_observation(mapped_obs, prompt, session_id)


def save_rollout_video(output_dir: Path, episode_index: int, frames: list[np.ndarray], fps: float) -> str | None:
    if not frames:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"episode_{episode_index:06d}.mp4"
    imageio.mimsave(path.as_posix(), frames, fps=fps)
    return path.as_posix()


def reset_robotwin_env_with_retries(
    args: argparse.Namespace,
    episode_offset: int,
) -> tuple[gym.Env, int, int, dict[str, Any], dict[str, Any]]:
    retries = max(int(args.reset_retries), 0)
    base_seed = args.seed_start + episode_offset
    last_error: BaseException | None = None

    for attempt in range(retries + 1):
        seed = base_seed + attempt * max(int(args.episodes), 1)
        env = make_robotwin_env(args.robotwin_task, seed, args.episode_length)
        try:
            obs, info = env.reset(seed=seed)
            initialize_robotwin_eval_state(env)
            return env, seed, attempt, obs, info
        except Exception as exc:
            last_error = exc
            with contextlib.suppress(Exception):
                env.close()
            if not is_unstable_reset_error(exc) or attempt >= retries:
                raise
            warnings.warn(
                f"RoboTwin reset for task={args.robotwin_task!r} seed={seed} was unstable; "
                f"retrying with a later seed ({attempt + 1}/{retries}).",
                RuntimeWarning,
            )

    assert last_error is not None
    raise last_error


def run_robotwin_mode(args: argparse.Namespace, client: WebsocketClientPolicy) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    max_steps = args.episode_length if args.max_steps is None else args.max_steps
    prompt = args.task_name or args.robotwin_task.replace("_", " ")

    for episode_offset in range(args.episodes):
        env: gym.Env | None = None
        frames: list[np.ndarray] = []
        action_shapes: list[list[int]] = []
        first_action: list[float] | None = None
        rewards: list[float] = []
        actions_sent = 0
        success = False
        done = False
        info: dict[str, Any] = {}

        try:
            env, seed, reset_attempts, obs, info = reset_robotwin_env_with_retries(args, episode_offset)
            session_id = f"robotwin-live-{args.robotwin_task}-{seed}-{uuid.uuid4().hex[:8]}"
            client.reset({"session_id": session_id})
            for _ in range(max_steps):
                if args.save_video:
                    head_image = obs.get("pixels", {}).get("head_camera")
                    if head_image is not None:
                        frames.append(normalize_image(head_image))

                payload = robotwin_obs_to_payload(obs, prompt, session_id)
                action_chunk = np.asarray(client.infer(payload), dtype=np.float32)
                if action_chunk.ndim == 1:
                    action_chunk = action_chunk.reshape(1, -1)
                if action_chunk.shape[-1] != 14:
                    raise ValueError(f"Expected action width 14, got {action_chunk.shape}")
                action_shapes.append(list(action_chunk.shape))
                if first_action is None and action_chunk.size:
                    first_action = action_chunk[0].tolist()

                executable_actions = action_chunk[: args.open_loop_horizon]
                for action_index, action in enumerate(executable_actions):
                    if args.clip_action:
                        action = np.clip(action, env.action_space.low, env.action_space.high)
                    need_obs = action_index == len(executable_actions) - 1
                    if args.fast_eval:
                        next_obs, reward, terminated, truncated, info = robotwin_fast_step(
                            env,
                            action,
                            need_obs=need_obs,
                        )
                        if next_obs is not None:
                            obs = next_obs
                    else:
                        obs, reward, terminated, truncated, info = env.step(action)
                    actions_sent += 1
                    rewards.append(float(reward))
                    success = bool(info.get("is_success", success))
                    done = bool(terminated or truncated)
                    if done:
                        break
                if done:
                    break
        finally:
            if env is not None:
                env.close()

        video_path = save_rollout_video(args.output_dir, episode_offset, frames, args.fps) if args.save_video else None
        result = {
            "episode_index": episode_offset,
            "seed": seed,
            "reset_attempts": reset_attempts,
            "task": args.robotwin_task,
            "prompt": prompt,
            "steps": actions_sent,
            "success": success,
            "reward_sum": float(sum(rewards)),
            "done": done,
            "action_shapes": action_shapes,
            "first_action": first_action,
            "checkpoint": checkpoint_result(args),
            "video_path": video_path,
            "fast_eval": bool(args.fast_eval),
            "last_info": info,
        }
        results.append(result)
        write_episode_result(args.output_dir, result)
    return results


def write_episode_result(output_dir: Path, result: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"episode_{int(result['episode_index']):06d}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=json_default, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    client = WebsocketClientPolicy(host=args.remote_host, port=args.remote_port)
    try:
        if args.mode == "robotwin":
            results = run_robotwin_mode(args, client)
        elif args.mode == "gym":
            results = run_gym_mode(args, client)
        else:
            results = run_dataset_mode(args, client)
    finally:
        client.close()

    summary = {
        "mode": args.mode,
        "episodes": len(results),
        "output_dir": args.output_dir.as_posix(),
        "timestamp": time.time(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
