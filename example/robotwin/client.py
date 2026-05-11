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
import importlib
import json
import os
import sys
import time
import uuid
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import pandas as pd
from gymnasium import spaces

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

VIDEO_KEYS = {
    "cam_high": "observation.images.cam_high",
    "cam_left": "observation.images.cam_left",
    "cam_right": "observation.images.cam_right",
}

ROBOTWIN_CAMERA_TO_DREAMZERO = {
    "video.cam_high": "head_camera",
    "video.cam_left": "left_camera",
    "video.cam_right": "right_camera",
}
ROBOTWIN_CAMERA_NAMES = ("head_camera", "left_camera", "right_camera")
ROBOTWIN_ACTION_DIM = 14
ROBOTWIN_ACTION_LOW = -1.0
ROBOTWIN_ACTION_HIGH = 1.0
ROBOTWIN_CAMERA_H = 240
ROBOTWIN_CAMERA_W = 320


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
        "--clip-action",
        action=argparse.BooleanOptionalAction,
        default=True,
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


def split_left_right_state(state: Any) -> dict[str, np.ndarray]:
    arr = np.asarray(state, dtype=np.float32)
    if arr.ndim == 2:
        if arr.shape[0] != 1:
            raise ValueError(f"Packed state must be one row, got {arr.shape}")
        arr = arr[0]
    if arr.shape != (14,):
        raise ValueError(f"Expected packed RoboTwin state shape (14,), got {arr.shape}")
    left = arr[:7]
    right = arr[7:]
    return {
        "state.left_joint_pos": left[:6],
        "state.left_gripper_pos": left[6:7],
        "state.right_joint_pos": right[:6],
        "state.right_gripper_pos": right[6:7],
    }


def normalize_image(image: Any) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 4:
        arr = arr[-1]
    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise ValueError(f"Expected image shape (H, W, C), got {arr.shape}")
    arr = arr[..., :3]
    if np.issubdtype(arr.dtype, np.floating):
        if float(np.nanmax(arr)) <= 1.0:
            arr = (arr * 255.0).clip(0, 255)
        else:
            arr = arr.clip(0, 255)
    return np.ascontiguousarray(arr.astype(np.uint8))


def model_image_sequence(image: Any) -> np.ndarray:
    return normalize_image(image)[None, ...]


def first_existing(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def nested_get(mapping: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def payload_from_observation(obs: dict[str, Any], prompt: str, session_id: str) -> dict[str, Any]:
    images = {
        "video.cam_high": coalesce(
            first_existing(obs, ("video.cam_high", "observation.images.cam_high", "cam_high")),
            nested_get(obs, ("observation", "images", "cam_high")),
            nested_get(obs, ("images", "cam_high")),
        ),
        "video.cam_left": coalesce(
            first_existing(
                obs,
                ("video.cam_left", "observation.images.cam_left", "observation.images.cam_left_wrist", "cam_left", "cam_left_wrist"),
            ),
            nested_get(obs, ("observation", "images", "cam_left")),
            nested_get(obs, ("observation", "images", "cam_left_wrist")),
            nested_get(obs, ("images", "cam_left")),
        ),
        "video.cam_right": coalesce(
            first_existing(
                obs,
                ("video.cam_right", "observation.images.cam_right", "observation.images.cam_right_wrist", "cam_right", "cam_right_wrist"),
            ),
            nested_get(obs, ("observation", "images", "cam_right")),
            nested_get(obs, ("observation", "images", "cam_right_wrist")),
            nested_get(obs, ("images", "cam_right")),
        ),
    }
    missing_images = [key for key, value in images.items() if value is None]
    if missing_images:
        raise KeyError(f"Missing image observations: {missing_images}")

    split_state = {
        key: first_existing(obs, (key,))
        for key in (
            "state.left_joint_pos",
            "state.left_gripper_pos",
            "state.right_joint_pos",
            "state.right_gripper_pos",
        )
    }
    if any(value is None for value in split_state.values()):
        packed = first_existing(obs, ("observation.state", "state"))
        if packed is None:
            packed = nested_get(obs, ("observation", "state"))
        split_state = split_left_right_state(packed)

    payload = {
        "video.cam_high": model_image_sequence(images["video.cam_high"]),
        "video.cam_left": model_image_sequence(images["video.cam_left"]),
        "video.cam_right": model_image_sequence(images["video.cam_right"]),
        "prompt": prompt,
        "annotation.task": prompt,
        "session_id": session_id,
    }
    payload.update({key: np.asarray(value, dtype=np.float32) for key, value in split_state.items()})
    return payload


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


def load_robotwin_setup_kwargs(task_name: str) -> dict[str, Any]:
    ensure_robotwin_workdir()
    import yaml
    from envs import CONFIGS_PATH

    task_config = "demo_clean"
    with open(os.path.join(CONFIGS_PATH, f"{task_config}.yml"), encoding="utf-8") as f:
        args = yaml.safe_load(f)

    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), encoding="utf-8") as f:
        embodiment_types = yaml.safe_load(f)
    embodiment = args.get("embodiment", ["aloha-agilex"])
    if len(embodiment) == 1:
        robot_file = embodiment_types[embodiment[0]]["file_path"]
        args["left_robot_file"] = robot_file
        args["right_robot_file"] = robot_file
        args["dual_arm_embodied"] = True
    elif len(embodiment) == 3:
        args["left_robot_file"] = embodiment_types[embodiment[0]]["file_path"]
        args["right_robot_file"] = embodiment_types[embodiment[1]]["file_path"]
        args["embodiment_dis"] = embodiment[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError(f"embodiment must have 1 or 3 items, got {len(embodiment)}")

    with open(os.path.join(args["left_robot_file"], "config.yml"), encoding="utf-8") as f:
        args["left_embodiment_config"] = yaml.safe_load(f)
    with open(os.path.join(args["right_robot_file"], "config.yml"), encoding="utf-8") as f:
        args["right_embodiment_config"] = yaml.safe_load(f)

    with open(os.path.join(CONFIGS_PATH, "_camera_config.yml"), encoding="utf-8") as f:
        camera_config = yaml.safe_load(f)
    head_cam = args["camera"]["head_camera_type"]
    args["head_camera_h"] = camera_config[head_cam]["h"]
    args["head_camera_w"] = camera_config[head_cam]["w"]
    args["render_freq"] = 0
    args["task_name"] = task_name
    args["task_config"] = task_config
    return args


def load_robotwin_task(task_name: str) -> type:
    ensure_robotwin_workdir()
    module = importlib.import_module(f"envs.{task_name}")
    task_cls = getattr(module, task_name, None)
    if task_cls is None:
        raise AttributeError(f"Task class '{task_name}' not found in envs/{task_name}.py")
    return task_cls


def ensure_robotwin_workdir() -> None:
    robotwin_root = REPO_ROOT / "third_party" / "RoboTwin"
    if robotwin_root.exists() and Path.cwd().resolve() != robotwin_root.resolve():
        os.chdir(robotwin_root)


class RoboTwinCompatEnv(gym.Env):
    """Python 3.10 compatible wrapper for RoboTwin 2.0's SAPIEN API."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(
        self,
        task_name: str,
        episode_index: int = 0,
        camera_names: Sequence[str] = ROBOTWIN_CAMERA_NAMES,
        observation_height: int = ROBOTWIN_CAMERA_H,
        observation_width: int = ROBOTWIN_CAMERA_W,
        episode_length: int = 300,
    ) -> None:
        super().__init__()
        self.task_name = task_name
        self.task = task_name
        self.task_description = task_name.replace("_", " ")
        self.episode_index = episode_index
        self.camera_names = list(camera_names)
        self.observation_height = observation_height
        self.observation_width = observation_width
        self.episode_length = episode_length
        self._max_episode_steps = episode_length
        self._env: Any | None = None
        self._step_count = 0
        self._black_frame = np.zeros((self.observation_height, self.observation_width, 3), dtype=np.uint8)
        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Dict(
                    {
                        cam: spaces.Box(
                            low=0,
                            high=255,
                            shape=(self.observation_height, self.observation_width, 3),
                            dtype=np.uint8,
                        )
                        for cam in self.camera_names
                    }
                ),
                "agent_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(ROBOTWIN_ACTION_DIM,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Box(
            low=ROBOTWIN_ACTION_LOW,
            high=ROBOTWIN_ACTION_HIGH,
            shape=(ROBOTWIN_ACTION_DIM,),
            dtype=np.float32,
        )

    def _ensure_env(self) -> None:
        if self._env is None:
            self._env = load_robotwin_task(self.task_name)()

    def _get_obs(self) -> dict[str, Any]:
        assert self._env is not None
        raw = self._env.get_obs()
        cameras_raw = raw.get("observation", {})
        images: dict[str, np.ndarray] = {}
        for cam in self.camera_names:
            cam_data = cameras_raw.get(cam)
            img = cam_data.get("rgb") if cam_data else None
            if img is None:
                images[cam] = self._black_frame
                continue
            img = np.asarray(img, dtype=np.uint8)
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            elif img.shape[-1] != 3:
                img = img[..., :3]
            images[cam] = img

        joint_action = raw.get("joint_action") or {}
        vec = joint_action.get("vector")
        if vec is None:
            joint_state = np.zeros(ROBOTWIN_ACTION_DIM, dtype=np.float32)
        else:
            arr = np.asarray(vec, dtype=np.float32).ravel()
            joint_state = (
                arr[:ROBOTWIN_ACTION_DIM]
                if arr.size >= ROBOTWIN_ACTION_DIM
                else np.zeros(ROBOTWIN_ACTION_DIM, dtype=np.float32)
            )
        return {"pixels": images, "agent_pos": joint_state}

    def reset(self, seed: int | None = None, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        import torch

        self._ensure_env()
        super().reset(seed=seed)
        assert self._env is not None
        actual_seed = self.episode_index if seed is None else seed
        setup_kwargs = load_robotwin_setup_kwargs(self.task_name)
        setup_kwargs.update(seed=actual_seed, is_test=True)
        with torch.enable_grad():
            self._env.setup_demo(**setup_kwargs)
        self.episode_index += 1
        self._step_count = 0
        return self._get_obs(), {"is_success": False, "task": self.task_name}

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        import torch

        assert self._env is not None
        action = np.asarray(action, dtype=np.float32)
        if action.ndim != 1 or action.shape[0] != ROBOTWIN_ACTION_DIM:
            raise ValueError(f"Expected action shape ({ROBOTWIN_ACTION_DIM},), got {action.shape}")
        with torch.enable_grad():
            if hasattr(self._env, "take_action"):
                self._env.take_action(action)
            else:
                self._env.step(action)
        self._step_count += 1
        is_success = bool(getattr(self._env, "eval_success", False))
        if not is_success and hasattr(self._env, "check_success"):
            is_success = bool(self._env.check_success())
        obs = self._get_obs()
        truncated = self._step_count >= self.episode_length
        info = {"task": self.task_name, "is_success": is_success}
        return obs, float(is_success), is_success, truncated, info

    def close(self) -> None:
        if self._env is None:
            return
        with contextlib.suppress(Exception):
            if hasattr(self._env, "close"):
                self._env.close()
            elif hasattr(self._env, "close_env"):
                self._env.close_env()
        self._env = None


def make_robotwin_env(task_name: str, episode_index: int, episode_length: int) -> gym.Env:
    ensure_robotwin_workdir()
    try:
        from lerobot.envs.robotwin import RoboTwinEnv

        return RoboTwinEnv(
            task_name=task_name,
            episode_index=episode_index,
            episode_length=episode_length,
        )
    except Exception as exc:
        warnings.warn(
            f"Falling back to local RoboTwinEnv compatibility wrapper because LeRobot import failed: {exc}",
            RuntimeWarning,
        )
        return RoboTwinCompatEnv(
            task_name=task_name,
            episode_index=episode_index,
            episode_length=episode_length,
        )


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


def run_robotwin_mode(args: argparse.Namespace, client: WebsocketClientPolicy) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    max_steps = args.episode_length if args.max_steps is None else args.max_steps
    prompt = args.task_name or args.robotwin_task.replace("_", " ")

    for episode_offset in range(args.episodes):
        seed = args.seed_start + episode_offset
        env = make_robotwin_env(args.robotwin_task, seed, args.episode_length)
        session_id = f"robotwin-live-{args.robotwin_task}-{seed}-{uuid.uuid4().hex[:8]}"
        client.reset({"session_id": session_id})
        frames: list[np.ndarray] = []
        action_shapes: list[list[int]] = []
        first_action: list[float] | None = None
        rewards: list[float] = []
        actions_sent = 0
        success = False
        done = False
        info: dict[str, Any] = {}

        try:
            obs, info = env.reset(seed=seed)
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

                for action in action_chunk[: args.open_loop_horizon]:
                    if args.clip_action:
                        action = np.clip(action, env.action_space.low, env.action_space.high)
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
            env.close()

        video_path = save_rollout_video(args.output_dir, episode_offset, frames, args.fps) if args.save_video else None
        result = {
            "episode_index": episode_offset,
            "seed": seed,
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
