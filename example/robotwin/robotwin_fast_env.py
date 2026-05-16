#!/usr/bin/env python3
"""Shared RoboTwin helpers for DreamZero eval.

This module intentionally lives outside ``third_party``.  It wraps the pinned
LeRobot RoboTwinEnv interface and exposes a chunk-step API that renders only
when the next policy request needs a fresh observation.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import sys
import time
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
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


def ensure_robotwin_workdir() -> None:
    robotwin_root = REPO_ROOT / "third_party" / "RoboTwin"
    if robotwin_root.exists() and Path.cwd().resolve() != robotwin_root.resolve():
        os.chdir(robotwin_root)


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


def parse_image_resolution(value: str | Sequence[int] | None) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        raw = value.strip().lower()
        if not raw or raw == "none":
            return None
        for sep in ("x", ","):
            if sep in raw:
                left, right = raw.split(sep, 1)
                return int(left), int(right)
        raise ValueError(f"Expected image resolution as HxW, got {value!r}")
    if len(value) != 2:
        raise ValueError(f"Expected image resolution with 2 values, got {value!r}")
    height, width = int(value[0]), int(value[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"Image resolution must be positive, got {(height, width)}")
    return height, width


def resize_uint8_image(image: Any, resolution: tuple[int, int] | None) -> np.ndarray:
    arr = normalize_image(image)
    if resolution is None:
        return arr
    height, width = parse_image_resolution(resolution)
    if arr.shape[0] == height and arr.shape[1] == width:
        return arr

    try:
        import cv2

        interpolation = cv2.INTER_AREA if height <= arr.shape[0] and width <= arr.shape[1] else cv2.INTER_LINEAR
        return np.ascontiguousarray(cv2.resize(arr, (width, height), interpolation=interpolation))
    except Exception:
        pass

    try:
        from PIL import Image

        return np.ascontiguousarray(np.asarray(Image.fromarray(arr).resize((width, height), Image.BILINEAR)))
    except Exception:
        y_idx = np.linspace(0, arr.shape[0] - 1, height).round().astype(np.int64)
        x_idx = np.linspace(0, arr.shape[1] - 1, width).round().astype(np.int64)
        return np.ascontiguousarray(arr[y_idx][:, x_idx])


def model_image_sequence_resized(image: Any, resolution: tuple[int, int] | None) -> np.ndarray:
    return resize_uint8_image(image, resolution)[None, ...]


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


def split_left_right_state(state: Any) -> dict[str, np.ndarray]:
    arr = np.asarray(state, dtype=np.float32)
    if arr.ndim == 2:
        if arr.shape[0] != 1:
            raise ValueError(f"Packed state must be one row, got {arr.shape}")
        arr = arr[0]
    if arr.shape != (ROBOTWIN_ACTION_DIM,):
        raise ValueError(f"Expected packed RoboTwin state shape (14,), got {arr.shape}")
    left = arr[:7]
    right = arr[7:]
    return {
        "state.left_joint_pos": left[:6],
        "state.left_gripper_pos": left[6:7],
        "state.right_joint_pos": right[:6],
        "state.right_gripper_pos": right[6:7],
    }


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
                (
                    "video.cam_left",
                    "observation.images.cam_left",
                    "observation.images.cam_left_wrist",
                    "cam_left",
                    "cam_left_wrist",
                ),
            ),
            nested_get(obs, ("observation", "images", "cam_left")),
            nested_get(obs, ("observation", "images", "cam_left_wrist")),
            nested_get(obs, ("images", "cam_left")),
        ),
        "video.cam_right": coalesce(
            first_existing(
                obs,
                (
                    "video.cam_right",
                    "observation.images.cam_right",
                    "observation.images.cam_right_wrist",
                    "cam_right",
                    "cam_right_wrist",
                ),
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


def stack_robotwin_payloads(
    observations: Sequence[dict[str, Any]],
    prompt: str,
    session_id: str,
    *,
    image_resolution: tuple[int, int] | None = None,
) -> dict[str, Any]:
    if not observations:
        raise ValueError("Cannot build a batched payload from an empty observation list")

    single_payloads = [robotwin_obs_to_payload(obs, prompt, session_id) for obs in observations]
    batch: dict[str, Any] = {
        "session_id": session_id,
        # msgpack_numpy rejects object arrays, so the wire payload uses a Python
        # list.  The server converts it to a NumPy string array before transform.
        "prompt": [prompt for _ in single_payloads],
        "annotation.task": [prompt for _ in single_payloads],
    }
    for key in ("video.cam_high", "video.cam_left", "video.cam_right"):
        images = []
        for payload in single_payloads:
            arr = np.asarray(payload[key])
            if image_resolution is not None:
                frames = [resize_uint8_image(frame, image_resolution) for frame in arr.reshape((-1,) + arr.shape[-3:])]
                arr = np.stack(frames, axis=0).reshape(arr.shape[:-3] + frames[0].shape)
            images.append(np.ascontiguousarray(arr))
        batch[key] = np.stack(images, axis=0)
    for key in (
        "state.left_joint_pos",
        "state.left_gripper_pos",
        "state.right_joint_pos",
        "state.right_gripper_pos",
    ):
        values = []
        for payload in single_payloads:
            arr = np.asarray(payload[key], dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.ndim != 2:
                raise ValueError(f"{key} must be 1D or 2D before stacking, got {arr.shape}")
            values.append(arr[-1])
        batch[key] = np.stack(values, axis=0)[:, None, :]
    return batch


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
        obs, reward, terminated, truncated, info = robotwin_fast_step(self, action, need_obs=True)
        if obs is None:
            obs = self._get_obs()
        return obs, reward, terminated, truncated, info

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


def is_unstable_reset_error(exc: BaseException) -> bool:
    return exc.__class__.__name__ == "UnStableError" or "Objects is unstable" in str(exc)


def make_arm_tag(value: str) -> Any:
    try:
        ensure_robotwin_workdir()
        from envs.utils import ArmTag

        return ArmTag(value)
    except Exception:
        return value


def initialize_robotwin_eval_state(env: gym.Env) -> None:
    """Fill task fields that expert play_once normally creates before check_success."""
    inner_env = getattr(env, "_env", None)
    if inner_env is None:
        return

    task_name = str(getattr(inner_env, "task_name", getattr(env, "task_name", "")))
    if task_name == "open_laptop" and not hasattr(inner_env, "arm_tag") and hasattr(inner_env, "laptop"):
        try:
            ensure_robotwin_workdir()
            from envs.utils import ArmTag, get_face_prod

            face_prod = get_face_prod(inner_env.laptop.get_pose().q, [1, 0, 0], [1, 0, 0])
            inner_env.arm_tag = ArmTag("left" if face_prod > 0 else "right")
        except Exception:
            inner_env.arm_tag = make_arm_tag("left")

    elif task_name == "place_object_scale" and not hasattr(inner_env, "arm_tag") and hasattr(inner_env, "object"):
        inner_env.arm_tag = make_arm_tag("right" if inner_env.object.get_pose().p[0] > 0 else "left")

    elif task_name == "put_object_cabinet" and hasattr(inner_env, "object"):
        if not hasattr(inner_env, "arm_tag"):
            inner_env.arm_tag = make_arm_tag("right" if inner_env.object.get_pose().p[0] > 0 else "left")
        if not hasattr(inner_env, "origin_z"):
            inner_env.origin_z = float(inner_env.object.get_pose().p[2])


def reset_robotwin_env_with_retries(
    *,
    task_name: str,
    episode_offset: int,
    seed_start: int,
    episodes: int,
    episode_length: int,
    reset_retries: int,
) -> tuple[gym.Env, int, int, dict[str, Any], dict[str, Any], float]:
    retries = max(int(reset_retries), 0)
    base_seed = seed_start + episode_offset
    last_error: BaseException | None = None

    for attempt in range(retries + 1):
        seed = base_seed + attempt * max(int(episodes), 1)
        env = make_robotwin_env(task_name, seed, episode_length)
        reset_start = time.perf_counter()
        try:
            obs, info = env.reset(seed=seed)
            initialize_robotwin_eval_state(env)
            assert_robotwin_fast_interface(env, strict=True)
            reset_time = time.perf_counter() - reset_start
            return env, seed, attempt, obs, info, reset_time
        except Exception as exc:
            last_error = exc
            with contextlib.suppress(Exception):
                env.close()
            if not is_unstable_reset_error(exc) or attempt >= retries:
                raise
            warnings.warn(
                f"RoboTwin reset for task={task_name!r} seed={seed} was unstable; "
                f"retrying with a later seed ({attempt + 1}/{retries}).",
                RuntimeWarning,
            )

    assert last_error is not None
    raise last_error


def assert_robotwin_fast_interface(env: gym.Env, *, strict: bool) -> bool:
    missing: list[str] = []
    if getattr(env, "_env", None) is None:
        missing.append("_env")
    if not callable(getattr(env, "_get_obs", None)):
        missing.append("_get_obs()")
    if not hasattr(env, "_step_count"):
        missing.append("_step_count")
    if not hasattr(env, "episode_length") and not hasattr(env, "_max_episode_steps"):
        missing.append("episode_length/_max_episode_steps")

    if missing and strict:
        raise RuntimeError(
            "LeRobot RoboTwinEnv is incompatible with fast eval wrapper; missing "
            + ", ".join(missing)
            + ". Check the pinned LeRobot commit in example/robotwin/README.md."
        )
    return not missing


def take_robotwin_action_no_obs(
    env: gym.Env,
    action: np.ndarray,
) -> tuple[float, bool, bool, dict[str, Any]]:
    if not assert_robotwin_fast_interface(env, strict=False):
        obs, reward, terminated, truncated, info = env.step(action)
        del obs
        return reward, terminated, truncated, info

    import torch

    inner_env = getattr(env, "_env")
    action = np.asarray(action, dtype=np.float32)
    if action.ndim != 1 or action.shape[0] != ROBOTWIN_ACTION_DIM:
        raise ValueError(f"Expected action shape ({ROBOTWIN_ACTION_DIM},), got {action.shape}")

    with torch.enable_grad():
        if hasattr(inner_env, "take_action"):
            inner_env.take_action(action)
        else:
            inner_env.step(action)

    env._step_count += 1
    success = bool(getattr(inner_env, "eval_success", False))
    if not success and hasattr(inner_env, "check_success"):
        success = bool(inner_env.check_success())

    episode_length = int(getattr(env, "episode_length", getattr(env, "_max_episode_steps", 0)) or 0)
    truncated = bool(episode_length and env._step_count >= episode_length)
    terminated = bool(success)
    done = bool(terminated or truncated)
    info: dict[str, Any] = {
        "task": getattr(env, "task_name", getattr(env, "task", "")),
        "is_success": success,
        "step": int(env._step_count),
    }
    if done:
        info["final_info"] = {
            "task": info["task"],
            "is_success": success,
        }
    return float(success), terminated, truncated, info


def robotwin_fast_step(
    env: gym.Env,
    action: np.ndarray,
    *,
    need_obs: bool,
) -> tuple[dict[str, Any] | None, float, bool, bool, dict[str, Any]]:
    """Step RoboTwin without rendering unless the next inference needs obs."""
    if not assert_robotwin_fast_interface(env, strict=False):
        obs, reward, terminated, truncated, info = env.step(action)
        return obs, reward, terminated, truncated, info

    reward, terminated, truncated, info = take_robotwin_action_no_obs(env, action)
    done = bool(terminated or truncated)
    get_obs = getattr(env, "_get_obs", None)
    obs = get_obs() if need_obs and not done and callable(get_obs) else None
    return obs, reward, terminated, truncated, info


class DreamZeroRoboTwinEnv:
    """Episode-scoped helper used by subprocess workers."""

    def __init__(
        self,
        *,
        task_name: str,
        episode_length: int,
        seed_start: int,
        episodes: int,
        reset_retries: int,
    ) -> None:
        self.task_name = task_name
        self.episode_length = int(episode_length)
        self.seed_start = int(seed_start)
        self.episodes = int(episodes)
        self.reset_retries = int(reset_retries)
        self.env: gym.Env | None = None
        self.seed: int | None = None
        self.reset_attempts = 0
        self.obs: dict[str, Any] | None = None
        self.info: dict[str, Any] = {}
        self.done = False

    def close(self) -> None:
        if self.env is not None:
            with contextlib.suppress(Exception):
                self.env.close()
        self.env = None
        self.obs = None
        self.done = False

    def reset(self, episode_index: int) -> dict[str, Any]:
        self.close()
        env, seed, reset_attempts, obs, info, reset_time = reset_robotwin_env_with_retries(
            task_name=self.task_name,
            episode_offset=int(episode_index),
            seed_start=self.seed_start,
            episodes=self.episodes,
            episode_length=self.episode_length,
            reset_retries=self.reset_retries,
        )
        self.env = env
        self.seed = seed
        self.reset_attempts = reset_attempts
        self.obs = obs
        self.info = info
        self.done = False
        return {
            "obs": obs,
            "info": info,
            "seed": seed,
            "reset_attempts": reset_attempts,
            "reset_time": reset_time,
        }

    def step_chunk(
        self,
        actions: Any,
        *,
        need_obs: bool = True,
        clip_action: bool = True,
    ) -> dict[str, Any]:
        if self.env is None:
            raise RuntimeError("step_chunk called before reset")
        if self.done:
            return {
                "obs": self.obs,
                "reward_sum": 0.0,
                "done": True,
                "terminated": False,
                "truncated": False,
                "actions_sent": 0,
                "env_step_time": 0.0,
                "get_obs_time": 0.0,
                "info": self.info,
            }

        action_arr = np.asarray(actions, dtype=np.float32)
        if action_arr.ndim == 1:
            action_arr = action_arr.reshape(1, -1)
        if action_arr.ndim != 2 or action_arr.shape[1] != ROBOTWIN_ACTION_DIM:
            raise ValueError(f"Expected action chunk shape (H, {ROBOTWIN_ACTION_DIM}), got {action_arr.shape}")

        reward_sum = 0.0
        terminated = False
        truncated = False
        actions_sent = 0
        env_step_time = 0.0
        get_obs_time = 0.0
        info = self.info

        for action in action_arr:
            if clip_action and hasattr(self.env, "action_space"):
                action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
            step_start = time.perf_counter()
            reward, terminated, truncated, info = take_robotwin_action_no_obs(self.env, action)
            env_step_time += time.perf_counter() - step_start
            actions_sent += 1
            reward_sum += float(reward)
            self.info = info
            self.done = bool(terminated or truncated)
            if self.done:
                break

        if need_obs and not self.done:
            get_obs = getattr(self.env, "_get_obs")
            obs_start = time.perf_counter()
            self.obs = get_obs()
            get_obs_time += time.perf_counter() - obs_start

        return {
            "obs": self.obs,
            "reward_sum": reward_sum,
            "done": self.done,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "actions_sent": actions_sent,
            "env_step_time": env_step_time,
            "get_obs_time": get_obs_time,
            "info": self.info,
        }
