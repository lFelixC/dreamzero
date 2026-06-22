# ruff: noqa

"""Real-robot DROID client for the Cosmos3 RoboLab action policy server."""

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import contextlib
import csv
import dataclasses
import datetime
import faulthandler
import os
import signal
import threading
import time
from typing import Any, Literal
import uuid

import imageio.v2 as imageio
import numpy as np
from openpi_client import image_tools
import tqdm

faulthandler.enable()

DEFAULT_CAMERA_RESOLUTION = (360, 640)


@dataclasses.dataclass
class Args:
    # Hardware parameters
    left_camera_id: str = "36517165"
    right_camera_id: str | None = None
    wrist_camera_id: str = "13337231"
    missing_right_camera_strategy: Literal["duplicate_left", "mask_right"] = "duplicate_left"

    # Rollout parameters
    max_timesteps: int = 6000
    open_loop_horizon: int = 8
    control_frequency: int = 15
    enable_async_prefetch: bool = True
    interpolate_actions: bool = False
    interpolation_substeps: int = 1
    log_timing: bool = False
    timing_log_interval: int = 1

    # Cosmos3 observation layout. The server will compose a 540x640 image by
    # putting the wrist view on top and two half-size exterior views underneath.
    camera_image_height: int = DEFAULT_CAMERA_RESOLUTION[0]
    camera_image_width: int = DEFAULT_CAMERA_RESOLUTION[1]

    # Remote server parameters
    remote_host: str = "0.0.0.0"
    remote_port: int = 9000

    # Logging and outputs
    video_output_dir: str | None = None
    results_dir: str = "results"


@contextlib.contextmanager
def prevent_keyboard_interrupt():
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


def _normalize_timeout(timeout_seconds: float | None) -> float | None:
    if timeout_seconds is None or timeout_seconds <= 0:
        return None
    return timeout_seconds


class OpenPIWebsocketClientPolicy:
    """Small OpenPI-compatible websocket client without endpoint injection."""

    def __init__(
        self,
        host: str,
        port: int,
        open_timeout: float | None = 0.0,
        log_wait: bool = True,
    ) -> None:
        self._uri = f"ws://{host}:{port}"
        from openpi_client import msgpack_numpy

        self._msgpack_numpy = msgpack_numpy
        self._packer = msgpack_numpy.Packer()
        self._open_timeout = _normalize_timeout(open_timeout)
        self._log_wait = log_wait
        self._ws, self._server_metadata = self._wait_for_server()

    def _wait_for_server(self) -> tuple[Any, dict]:
        import websockets.sync.client as websocket_client

        if self._log_wait:
            print(f"Waiting for Cosmos3 server at {self._uri} ...")
        while True:
            try:
                conn = websocket_client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    open_timeout=self._open_timeout,
                    ping_interval=None,
                    ping_timeout=None,
                )
                metadata = self._msgpack_numpy.unpackb(conn.recv())
                if self._log_wait:
                    print(f"Connected to Cosmos3 server, metadata keys: {sorted(metadata.keys())}")
                return conn, metadata
            except (ConnectionRefusedError, OSError) as exc:
                if self._log_wait:
                    print(f"Still waiting for server ({exc})")
                time.sleep(5)

    def get_server_metadata(self) -> dict:
        return self._server_metadata

    def infer(self, obs: dict) -> dict:
        self._ws.send(self._packer.pack(obs))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return self._msgpack_numpy.unpackb(response)

    def close(self) -> None:
        self._ws.close()


@dataclasses.dataclass
class PendingPrefetch:
    future: Future
    request_step_idx: int


def _resize_image(image: np.ndarray, height: int, width: int) -> np.ndarray:
    return image_tools.resize_with_pad(image, height, width).astype(np.uint8)


def _build_cosmos_preview_frame(
    primary_image: np.ndarray,
    secondary_image: np.ndarray,
    wrist_image: np.ndarray,
    image_resolution: tuple[int, int],
) -> np.ndarray:
    height, width = image_resolution
    wrist = _resize_image(wrist_image, height, width)
    exterior_height = max(height // 2, 1)
    exterior_width = max(width // 2, 1)
    exterior_1 = _resize_image(primary_image, exterior_height, exterior_width)
    exterior_2 = _resize_image(secondary_image, exterior_height, exterior_width)
    return np.concatenate([wrist, np.concatenate([exterior_1, exterior_2], axis=1)], axis=0)


def _write_episode_video(output_path: str, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps, codec="libx264")


def _extract_observation(args: Args, obs_dict: dict) -> dict:
    image_observations = obs_dict["image"]
    left_image, right_image, wrist_image = None, None, None
    for key, value in image_observations.items():
        if args.left_camera_id in key and "left" in key:
            left_image = value
        elif args.right_camera_id and args.right_camera_id in key and "left" in key:
            right_image = value
        elif args.wrist_camera_id in key and "left" in key:
            wrist_image = value

    if left_image is None:
        raise ValueError(f"Could not find left camera '{args.left_camera_id}' in observation keys: {list(image_observations)}")
    if wrist_image is None:
        raise ValueError(
            f"Could not find wrist camera '{args.wrist_camera_id}' in observation keys: {list(image_observations)}"
        )

    left_image = left_image[..., :3][..., ::-1]
    wrist_image = wrist_image[..., :3][..., ::-1]
    if right_image is not None:
        right_image = right_image[..., :3][..., ::-1]

    robot_state = obs_dict["robot_state"]
    return {
        "left_image": left_image,
        "right_image": right_image,
        "wrist_image": wrist_image,
        "cartesian_position": np.asarray(robot_state["cartesian_position"]),
        "joint_position": np.asarray(robot_state["joint_positions"]),
        "gripper_position": np.asarray([robot_state["gripper_position"]]),
    }


def _resolve_secondary_image(
    args: Args,
    primary_image: np.ndarray,
    secondary_image: np.ndarray | None,
    warned_missing_right: bool,
) -> tuple[np.ndarray, bool]:
    if secondary_image is not None:
        return secondary_image, warned_missing_right

    if not warned_missing_right:
        if args.right_camera_id is None:
            missing_right_message = "Right camera id not provided"
        else:
            missing_right_message = f"Right camera '{args.right_camera_id}' not found in observations"

        if args.missing_right_camera_strategy == "mask_right":
            print(f"{missing_right_message}, filling Cosmos3 second exterior view with a black mask.")
        else:
            print(f"{missing_right_message}, duplicating left camera into Cosmos3 second exterior view.")
        warned_missing_right = True

    if args.missing_right_camera_strategy == "mask_right":
        return np.zeros_like(primary_image), warned_missing_right
    return primary_image, warned_missing_right


class Cosmos3RealPolicyClient:
    def __init__(
        self,
        remote_host: str,
        remote_port: int,
        open_loop_horizon: int,
        control_frequency: int,
        image_resolution: tuple[int, int],
        enable_async_prefetch: bool = True,
        interpolate_actions: bool = False,
        interpolation_substeps: int = 1,
    ) -> None:
        self._client = OpenPIWebsocketClientPolicy(host=remote_host, port=remote_port)
        self._open_loop_horizon = int(open_loop_horizon)
        self._control_frequency = int(control_frequency)
        self._image_resolution = image_resolution
        self._enable_async_prefetch = bool(enable_async_prefetch)
        self._interpolate_actions = bool(interpolate_actions)
        self._interpolation_substeps = max(int(interpolation_substeps), 0)
        if not self._interpolate_actions:
            self._interpolation_substeps = 0
        self._client_lock = threading.Lock()
        self._prefetch_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cosmos3_prefetch")

        self._pred_action_chunk: np.ndarray | None = None
        self._pred_action_step_deltas: np.ndarray | None = None
        self._actions_from_chunk_completed = 0
        self._base_actions_from_chunk_completed = 0
        self._base_chunk_size = 0
        self._pending_prefetch: PendingPrefetch | None = None
        self._episode_step_idx = 0
        self._session_id = str(uuid.uuid4())
        self._last_timing: dict[str, float | int | bool] = {}
        self._last_debug_step: dict[str, object] = {}
        self._next_chunk_id = 0
        self._active_chunk_id: int | None = None

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def image_resolution(self) -> tuple[int, int]:
        return self._image_resolution

    @property
    def last_timing(self) -> dict[str, float | int | bool]:
        return self._last_timing.copy()

    @property
    def last_debug_step(self) -> dict[str, object]:
        return self._last_debug_step.copy()

    def _reset_local_state(self) -> None:
        self._drain_pending_prefetch()
        self._pred_action_chunk = None
        self._pred_action_step_deltas = None
        self._actions_from_chunk_completed = 0
        self._base_actions_from_chunk_completed = 0
        self._base_chunk_size = 0
        self._pending_prefetch = None
        self._episode_step_idx = 0
        self._session_id = str(uuid.uuid4())
        self._last_debug_step = {}
        self._next_chunk_id = 0
        self._active_chunk_id = None

    def reset(self) -> None:
        self._reset_local_state()

    def _normalize_action_chunk(self, response: object) -> np.ndarray:
        if isinstance(response, dict):
            if "actions" in response:
                response = response["actions"]
            elif "action" in response:
                response = response["action"]
            else:
                raise TypeError(f"Policy response dict does not contain 'action' or 'actions'. Keys: {list(response.keys())}")

        pred_action_chunk = np.array(response, dtype=np.float32, copy=True)
        if pred_action_chunk.ndim != 2 or pred_action_chunk.shape[1] != 8:
            raise ValueError(f"Expected action chunk with shape (N, 8), got {pred_action_chunk.shape}")
        return pred_action_chunk

    def _interpolate_action_pair(
        self,
        start: np.ndarray,
        end: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        interpolated = np.array(start, dtype=np.float32, copy=True)
        interpolated[:7] = start[:7] + alpha * (end[:7] - start[:7])
        interpolated[7:] = start[7:]
        return interpolated

    def _expand_action_chunk_for_control(
        self,
        action_chunk: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self._interpolate_actions or self._interpolation_substeps <= 0 or action_chunk.shape[0] <= 1:
            return (
                np.array(action_chunk, dtype=np.float32, copy=True),
                np.ones((action_chunk.shape[0],), dtype=np.int32),
            )

        expanded_actions: list[np.ndarray] = []
        step_deltas: list[int] = []
        for index in range(action_chunk.shape[0] - 1):
            start = np.asarray(action_chunk[index], dtype=np.float32)
            end = np.asarray(action_chunk[index + 1], dtype=np.float32)
            expanded_actions.append(start.copy())
            step_deltas.append(1)
            for substep in range(1, self._interpolation_substeps + 1):
                alpha = substep / float(self._interpolation_substeps + 1)
                expanded_actions.append(self._interpolate_action_pair(start, end, alpha))
                step_deltas.append(0)

        expanded_actions.append(np.array(action_chunk[-1], dtype=np.float32, copy=True))
        step_deltas.append(1)
        return np.stack(expanded_actions, axis=0), np.asarray(step_deltas, dtype=np.int32)

    def _set_action_chunk(self, action_chunk: np.ndarray) -> None:
        expanded_chunk, step_deltas = self._expand_action_chunk_for_control(action_chunk)
        self._pred_action_chunk = expanded_chunk
        self._pred_action_step_deltas = step_deltas
        self._actions_from_chunk_completed = 0
        self._base_actions_from_chunk_completed = 0
        self._base_chunk_size = int(action_chunk.shape[0])
        self._next_chunk_id += 1
        self._active_chunk_id = self._next_chunk_id

    def _should_sync_query_policy(self) -> bool:
        if self._pred_action_chunk is None:
            return True
        if self._actions_from_chunk_completed >= len(self._pred_action_chunk):
            return True
        if self._enable_async_prefetch:
            return False
        return self._open_loop_horizon > 0 and self._base_actions_from_chunk_completed >= self._open_loop_horizon

    def _should_launch_prefetch(self) -> bool:
        if not self._enable_async_prefetch:
            return False
        if self._pred_action_chunk is None or self._pred_action_step_deltas is None:
            return False
        if self._pending_prefetch is not None:
            return False
        if self._actions_from_chunk_completed >= len(self._pred_action_chunk):
            return False
        if self._open_loop_horizon <= 0:
            return False
        return self._base_actions_from_chunk_completed >= self._open_loop_horizon

    def _build_request_data(
        self,
        primary_image: np.ndarray,
        secondary_image: np.ndarray,
        wrist_image: np.ndarray,
        joint_position: np.ndarray,
        gripper_position: np.ndarray,
        instruction: str,
    ) -> dict:
        height, width = self._image_resolution
        return {
            "observation/exterior_image_1_left": _resize_image(primary_image, height, width),
            "observation/exterior_image_2_left": _resize_image(secondary_image, height, width),
            "observation/wrist_image_left": _resize_image(wrist_image, height, width),
            "observation/joint_position": np.asarray(joint_position, dtype=np.float32),
            "observation/gripper_position": np.asarray(gripper_position, dtype=np.float32),
            "prompt": instruction,
        }

    def _run_policy_request(self, request_data: dict) -> dict[str, Any]:
        roundtrip_start = time.perf_counter()
        with self._client_lock:
            response = self._client.infer(request_data)
        policy_roundtrip_ms = (time.perf_counter() - roundtrip_start) * 1000
        return {
            "pred_action_chunk": self._normalize_action_chunk(response),
            "policy_roundtrip_ms": policy_roundtrip_ms,
            "server_timing": response.get("server_timing", {}) if isinstance(response, dict) else {},
        }

    def _activate_chunk(self, result: dict[str, Any]) -> int:
        self._set_action_chunk(np.asarray(result["pred_action_chunk"], dtype=np.float32))
        return int(self._active_chunk_id or -1)

    def _start_prefetch(self, request_data: dict, request_step_idx: int) -> None:
        if self._pending_prefetch is not None:
            return
        self._pending_prefetch = PendingPrefetch(
            future=self._prefetch_executor.submit(self._run_policy_request, request_data),
            request_step_idx=int(request_step_idx),
        )

    def _consume_pending_prefetch(self, wait: bool) -> tuple[bool, float, float]:
        if self._pending_prefetch is None:
            return False, 0.0, 0.0
        pending = self._pending_prefetch
        if not wait and not pending.future.done():
            return False, 0.0, 0.0

        wait_start = time.perf_counter()
        if wait:
            with prevent_keyboard_interrupt():
                result = pending.future.result()
        else:
            result = pending.future.result()
        policy_wait_ms = (time.perf_counter() - wait_start) * 1000
        self._pending_prefetch = None
        self._activate_chunk(result)
        return True, policy_wait_ms, float(result["policy_roundtrip_ms"])

    def _drain_pending_prefetch(self) -> None:
        if self._pending_prefetch is None:
            return
        try:
            with prevent_keyboard_interrupt():
                self._pending_prefetch.future.result()
        except Exception as exc:
            print(f"Warning: pending policy prefetch failed during reset: {exc}")
        finally:
            self._pending_prefetch = None

    def infer(
        self,
        primary_image: np.ndarray,
        secondary_image: np.ndarray,
        wrist_image: np.ndarray,
        joint_position: np.ndarray,
        gripper_position: np.ndarray,
        instruction: str,
    ) -> np.ndarray:
        infer_start = time.perf_counter()
        current_step_idx = int(self._episode_step_idx)
        joint_position_array = np.asarray(joint_position, dtype=np.float32).reshape(-1)
        gripper_position_array = np.asarray(gripper_position, dtype=np.float32).reshape(-1)

        queried_policy = False
        launched_prefetch = False
        waited_for_policy = False
        activated_chunk_this_step = False
        chunk_activation_source = "none"
        policy_wait_ms = 0.0
        build_request_ms = 0.0
        policy_roundtrip_ms = 0.0
        launched_prefetch_request_step_idx = -1

        if self._pending_prefetch is not None:
            wait_for_pending = self._pred_action_chunk is None or self._actions_from_chunk_completed >= len(self._pred_action_chunk)
            promoted_pending, waited_policy_ms, pending_roundtrip_ms = self._consume_pending_prefetch(wait=wait_for_pending)
            if promoted_pending:
                queried_policy = True
                waited_for_policy = waited_policy_ms > 0.0
                policy_wait_ms += waited_policy_ms
                policy_roundtrip_ms = pending_roundtrip_ms
                activated_chunk_this_step = True
                chunk_activation_source = "async_prefetch"

        if self._should_launch_prefetch():
            build_request_start = time.perf_counter()
            request_data = self._build_request_data(
                primary_image,
                secondary_image,
                wrist_image,
                joint_position,
                gripper_position,
                instruction,
            )
            build_request_ms += (time.perf_counter() - build_request_start) * 1000
            self._start_prefetch(request_data, request_step_idx=self._episode_step_idx)
            queried_policy = True
            launched_prefetch = True
            launched_prefetch_request_step_idx = int(self._episode_step_idx)

        if self._should_sync_query_policy():
            promoted_pending, waited_policy_ms, pending_roundtrip_ms = self._consume_pending_prefetch(wait=True)
            if promoted_pending:
                queried_policy = True
                waited_for_policy = True
                policy_wait_ms += waited_policy_ms
                policy_roundtrip_ms = pending_roundtrip_ms
                activated_chunk_this_step = True
                chunk_activation_source = "prefetch_on_sync_path"
            else:
                build_request_start = time.perf_counter()
                request_data = self._build_request_data(
                    primary_image,
                    secondary_image,
                    wrist_image,
                    joint_position,
                    gripper_position,
                    instruction,
                )
                build_request_ms += (time.perf_counter() - build_request_start) * 1000
                wait_start = time.perf_counter()
                with prevent_keyboard_interrupt():
                    result = self._run_policy_request(request_data)
                policy_wait_ms += (time.perf_counter() - wait_start) * 1000
                policy_roundtrip_ms = float(result["policy_roundtrip_ms"])
                waited_for_policy = True
                queried_policy = True
                self._activate_chunk(result)
                activated_chunk_this_step = True
                chunk_activation_source = "sync_request"

        action_start = time.perf_counter()
        chunk_local_step_idx = int(self._actions_from_chunk_completed)
        raw_action = np.asarray(self._pred_action_chunk[chunk_local_step_idx], dtype=np.float32)
        action = np.array(raw_action, dtype=np.float32, copy=True)
        chunk_id = self._active_chunk_id
        step_delta = 1
        if self._pred_action_step_deltas is not None:
            step_delta = int(self._pred_action_step_deltas[chunk_local_step_idx])
        self._actions_from_chunk_completed += 1
        self._base_actions_from_chunk_completed += step_delta
        self._episode_step_idx += step_delta

        action[-1] = 1.0 if action[-1].item() > 0.5 else 0.0

        infer_end = time.perf_counter()
        chunk_size = int(len(self._pred_action_chunk)) if self._pred_action_chunk is not None else 0
        self._last_timing = {
            "queried_policy": queried_policy,
            "build_request_ms": build_request_ms,
            "policy_wait_ms": policy_wait_ms,
            "policy_roundtrip_ms": policy_roundtrip_ms,
            "action_postprocess_ms": (infer_end - action_start) * 1000,
            "infer_total_ms": (infer_end - infer_start) * 1000,
            "chunk_size": chunk_size,
            "base_chunk_size": int(self._base_chunk_size),
            "remaining_actions_in_chunk": max(chunk_size - self._actions_from_chunk_completed, 0),
            "remaining_base_actions_in_chunk": max(self._base_chunk_size - self._base_actions_from_chunk_completed, 0),
            "launched_prefetch": launched_prefetch,
            "waited_for_policy": waited_for_policy,
            "prefetch_in_flight": self._pending_prefetch is not None,
            "interpolate_actions": self._interpolate_actions,
            "interpolation_substeps": self._interpolation_substeps,
        }
        self._last_debug_step = {
            "trace_type": "cosmos3_step",
            "session_id": self._session_id,
            "executed_step_idx": current_step_idx,
            "current_chunk_id": None if chunk_id is None else int(chunk_id),
            "chunk_local_step_idx": chunk_local_step_idx,
            "chunk_size": chunk_size,
            "base_chunk_size": int(self._base_chunk_size),
            "remaining_actions_after_step": max(chunk_size - self._actions_from_chunk_completed, 0),
            "remaining_base_actions_after_step": max(self._base_chunk_size - self._base_actions_from_chunk_completed, 0),
            "activated_chunk_this_step": bool(activated_chunk_this_step),
            "chunk_activation_source": chunk_activation_source,
            "queried_policy": bool(queried_policy),
            "launched_prefetch": bool(launched_prefetch),
            "waited_for_policy": bool(waited_for_policy),
            "prefetch_in_flight": bool(self._pending_prefetch is not None),
            "launched_prefetch_request_step_idx": launched_prefetch_request_step_idx,
            "policy_wait_ms": float(policy_wait_ms),
            "policy_roundtrip_ms": float(policy_roundtrip_ms),
            "build_request_ms": float(build_request_ms),
            "infer_total_ms": float((infer_end - infer_start) * 1000),
            "interpolate_actions": bool(self._interpolate_actions),
            "interpolation_substeps": int(self._interpolation_substeps),
            "joint_position": joint_position_array.astype(np.float32),
            "gripper_position": gripper_position_array.astype(np.float32),
            "raw_action": raw_action.astype(np.float32),
            "executed_action": action.astype(np.float32),
        }
        return action


def main(args: Args) -> None:
    try:
        from droid.robot_env import RobotEnv
    except ImportError as exc:
        raise ImportError(
            "Failed to import DROID RobotEnv. Please run this script in the robot laptop environment with DROID installed."
        ) from exc

    if args.video_output_dir:
        os.makedirs(args.video_output_dir, exist_ok=True)

    env = RobotEnv(action_space="joint_position", gripper_action_space="position")
    print("Created the droid env!")

    policy_client = Cosmos3RealPolicyClient(
        args.remote_host,
        args.remote_port,
        args.open_loop_horizon,
        args.control_frequency,
        image_resolution=(args.camera_image_height, args.camera_image_width),
        enable_async_prefetch=args.enable_async_prefetch,
        interpolate_actions=args.interpolate_actions,
        interpolation_substeps=args.interpolation_substeps,
    )

    records: list[dict] = []
    warned_missing_right = False

    while True:
        # instruction = input("Enter instruction: ")
        instruction = "pick up blue cube into red box"
        episode_session_id = policy_client.session_id
        episode_video_frames: list[np.ndarray] = []

        bar = tqdm.tqdm(range(args.max_timesteps))
        print("Running Cosmos3 rollout... press Ctrl+C to stop early.")
        for t_step in bar:
            loop_start = time.perf_counter()
            sleep_ms = 0.0
            try:
                obs_start = time.perf_counter()
                curr_obs = _extract_observation(args, env.get_observation())
                primary_image = curr_obs["left_image"]
                secondary_image = curr_obs["right_image"]
                secondary_image, warned_missing_right = _resolve_secondary_image(
                    args,
                    primary_image,
                    secondary_image,
                    warned_missing_right,
                )
                obs_done = time.perf_counter()

                video_frame_start = time.perf_counter()
                video_frame = _build_cosmos_preview_frame(
                    primary_image,
                    secondary_image,
                    curr_obs["wrist_image"],
                    policy_client.image_resolution,
                )
                episode_video_frames.append(video_frame)
                if t_step == 0:
                    imageio.imwrite("robot_camera_views_cosmos3.png", video_frame)
                video_frame_done = time.perf_counter()

                infer_start = time.perf_counter()
                action = policy_client.infer(
                    primary_image=primary_image,
                    secondary_image=secondary_image,
                    wrist_image=curr_obs["wrist_image"],
                    joint_position=curr_obs["joint_position"],
                    gripper_position=curr_obs["gripper_position"],
                    instruction=instruction,
                )
                infer_done = time.perf_counter()

                env_step_start = time.perf_counter()
                env.step(action)
                env_step_done = time.perf_counter()

                elapsed_time = time.perf_counter() - loop_start
                target_period = 1 / args.control_frequency
                if elapsed_time < target_period:
                    sleep_ms = (target_period - elapsed_time) * 1000
                    time.sleep(target_period - elapsed_time)

                if args.log_timing and t_step % args.timing_log_interval == 0:
                    infer_timing = policy_client.last_timing
                    print(
                        f"[timing step={t_step}] "
                        f"obs={(obs_done - obs_start) * 1000:.1f}ms "
                        f"compose={(video_frame_done - video_frame_start) * 1000:.1f}ms "
                        f"client_infer={(infer_done - infer_start) * 1000:.1f}ms "
                        f"(query={infer_timing.get('queried_policy', False)} "
                        f"prefetch_launch={bool(infer_timing.get('launched_prefetch', False))} "
                        f"prefetch_wait={bool(infer_timing.get('waited_for_policy', False))} "
                        f"prefetch_inflight={bool(infer_timing.get('prefetch_in_flight', False))} "
                        f"build={float(infer_timing.get('build_request_ms', 0.0)):.1f}ms "
                        f"ws_wait={float(infer_timing.get('policy_wait_ms', 0.0)):.1f}ms "
                        f"ws_roundtrip={float(infer_timing.get('policy_roundtrip_ms', 0.0)):.1f}ms "
                        f"post={float(infer_timing.get('action_postprocess_ms', 0.0)):.1f}ms "
                        f"chunk={int(infer_timing.get('chunk_size', 0))} "
                        f"base_chunk={int(infer_timing.get('base_chunk_size', 0))} "
                        f"remain={int(infer_timing.get('remaining_actions_in_chunk', 0))} "
                        f"base_remain={int(infer_timing.get('remaining_base_actions_in_chunk', 0))} "
                        f"interp={bool(infer_timing.get('interpolate_actions', False))}x"
                        f"{1 + int(infer_timing.get('interpolation_substeps', 0))}) "
                        f"env_step={(env_step_done - env_step_start) * 1000:.1f}ms "
                        f"sleep={sleep_ms:.1f}ms "
                        f"loop={(time.perf_counter() - loop_start) * 1000:.1f}ms"
                    )
            except KeyboardInterrupt:
                break

        if args.video_output_dir:
            input_video_path = os.path.join(args.video_output_dir, "inputs", f"{episode_session_id}.mp4")
            _write_episode_video(input_video_path, episode_video_frames, fps=args.control_frequency)
        else:
            input_video_path = None

        policy_client.reset()

        success: str | float | None = None
        while not isinstance(success, float):
            raw_success = input(
                "Did the rollout succeed? (enter y for 100%, n for 0%), or a numeric value 0-100 based on the evaluation spec"
            )
            if raw_success == "y":
                success = 1.0
            elif raw_success == "n":
                success = 0.0
            else:
                try:
                    success = float(raw_success) / 100
                except ValueError:
                    print(f"Could not parse success value: {raw_success}")
                    success = None
                    continue

            if isinstance(success, float) and not (0 <= success <= 1):
                print(f"Success must be a number in [0, 100] but got: {success * 100}")
                success = None

        records.append(
            {
                "success": success,
                "duration": t_step,
                "session_id": episode_session_id,
                "input_video_filename": input_video_path,
            }
        )

        if input("Do one more eval? (enter y or n) ").lower() != "y":
            break
        env.reset()

    os.makedirs(args.results_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%I:%M%p_%B_%d_%Y")
    csv_filename = os.path.join(args.results_dir, f"eval_cosmos3_{timestamp}.csv")
    with open(csv_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["success", "duration", "session_id", "input_video_filename"],
        )
        writer.writeheader()
        writer.writerows(records)
    print(f"Results saved to {csv_filename}")


def _parse_args() -> Args:
    defaults = Args()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-camera-id", default=defaults.left_camera_id)
    parser.add_argument("--right-camera-id", default=defaults.right_camera_id)
    parser.add_argument("--wrist-camera-id", default=defaults.wrist_camera_id)
    parser.add_argument(
        "--missing-right-camera-strategy",
        choices=["duplicate_left", "mask_right"],
        default=defaults.missing_right_camera_strategy,
    )
    parser.add_argument("--max-timesteps", type=int, default=defaults.max_timesteps)
    parser.add_argument("--open-loop-horizon", type=int, default=defaults.open_loop_horizon)
    parser.add_argument("--control-frequency", type=int, default=defaults.control_frequency)
    parser.add_argument(
        "--enable-async-prefetch",
        action=argparse.BooleanOptionalAction,
        default=defaults.enable_async_prefetch,
    )
    parser.add_argument(
        "--interpolate-actions",
        action=argparse.BooleanOptionalAction,
        default=defaults.interpolate_actions,
    )
    parser.add_argument("--interpolation-substeps", type=int, default=defaults.interpolation_substeps)
    parser.add_argument("--log-timing", action=argparse.BooleanOptionalAction, default=defaults.log_timing)
    parser.add_argument("--timing-log-interval", type=int, default=defaults.timing_log_interval)
    parser.add_argument("--camera-image-height", type=int, default=defaults.camera_image_height)
    parser.add_argument("--camera-image-width", type=int, default=defaults.camera_image_width)
    parser.add_argument("--remote-host", default=defaults.remote_host)
    parser.add_argument("--remote-port", type=int, default=defaults.remote_port)
    parser.add_argument("--video-output-dir", default=defaults.video_output_dir)
    parser.add_argument("--results-dir", default=defaults.results_dir)
    return Args(**vars(parser.parse_args()))


if __name__ == "__main__":
    main(_parse_args())
