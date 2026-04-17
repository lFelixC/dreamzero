# ruff: noqa

import contextlib
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import dataclasses
import datetime
import faulthandler
import json
import os
import signal
import threading
import time
from typing import Literal
import uuid

import imageio.v2 as imageio
import numpy as np
from openpi_client import image_tools
import pandas as pd
import tqdm
import tyro

from eval_utils.policy_client import WebsocketClientPolicy
from eval_utils.policy_server import PolicyServerConfig

faulthandler.enable()

FRAME_HISTORY_LENGTH = 24
FRAME_SELECTION_INDICES = (0, 7, 15, 23)
DEFAULT_IMAGE_RESOLUTION = (180, 320)


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
    use_rtc: bool = False
    enable_async_prefetch: bool = False
    rtc_inference_delay_steps: int = 0
    rtc_use_measured_delay: bool = False
    rtc_handoff_joint_blend: float = 0.6
    log_timing: bool = False
    timing_log_interval: int = 1
    rtc_debug_trace_path: str | None = None

    # Remote server parameters
    remote_host: str = "0.0.0.0"
    remote_port: int = 9000

    # Logging and outputs
    video_output_dir: str | None = None
    results_dir: str = "results"


@dataclasses.dataclass
class PendingPrefetch:
    future: Future
    request_step_idx: int
    activation_step_idx: int
    request_delay_steps: int


@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Delay Ctrl+C until the protected network call completes."""
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


def _resize_frames(frames: np.ndarray, height: int, width: int) -> np.ndarray:
    if frames.ndim == 3:
        return image_tools.resize_with_pad(frames, height, width)
    return np.stack([image_tools.resize_with_pad(frame, height, width) for frame in frames], axis=0)


def _select_history_frames(history: deque[np.ndarray]) -> np.ndarray:
    if not history:
        raise ValueError("Cannot build multi-frame request from empty history")

    window = list(history)
    if len(window) < FRAME_HISTORY_LENGTH:
        window = [window[0]] * (FRAME_HISTORY_LENGTH - len(window)) + window
    else:
        window = window[-FRAME_HISTORY_LENGTH:]

    return np.stack([window[index] for index in FRAME_SELECTION_INDICES], axis=0)


def _build_input_video_frame(
    primary_image: np.ndarray,
    secondary_image: np.ndarray,
    wrist_image: np.ndarray,
    image_resolution: tuple[int, int],
) -> np.ndarray:
    height, width = image_resolution
    exterior_0 = image_tools.resize_with_pad(primary_image, height, width)
    wrist = image_tools.resize_with_pad(wrist_image, height, width)
    exterior_1 = image_tools.resize_with_pad(secondary_image, height, width)
    return np.concatenate([exterior_0, wrist, exterior_1], axis=1)


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
            print(
                f"{missing_right_message}, filling DreamZero exterior_image_1_left with a black mask."
            )
        else:
            print(
                f"{missing_right_message}, duplicating left camera into DreamZero exterior_image_1_left."
            )
        warned_missing_right = True

    if args.missing_right_camera_strategy == "mask_right":
        return np.zeros_like(primary_image), warned_missing_right
    return primary_image, warned_missing_right


def _write_episode_video(output_path: str, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps, codec="libx264")


def _json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


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


class DreamZeroRealPolicyClient:
    def __init__(
        self,
        remote_host: str,
        remote_port: int,
        open_loop_horizon: int,
        control_frequency: int,
        use_rtc: bool = False,
        enable_async_prefetch: bool = False,
        rtc_inference_delay_steps: int = 0,
        rtc_use_measured_delay: bool = False,
        rtc_handoff_joint_blend: float = 0.6,
    ) -> None:
        self._client = WebsocketClientPolicy(host=remote_host, port=remote_port)
        metadata = self._client.get_server_metadata()
        self._server_config = PolicyServerConfig(**metadata)

        assert self._server_config.n_external_cameras == 2, (
            f"DreamZero real-world serve expects 2 external cameras, got {self._server_config.n_external_cameras}"
        )
        assert self._server_config.needs_wrist_camera, "DreamZero real-world serve must request wrist camera input"
        assert self._server_config.needs_session_id, "DreamZero real-world serve must require session_id"
        assert self._server_config.action_space == "joint_position", (
            f"Expected joint_position action space, got {self._server_config.action_space}"
        )

        self._image_resolution = self._server_config.image_resolution or DEFAULT_IMAGE_RESOLUTION
        self._open_loop_horizon = open_loop_horizon
        self._control_frequency = control_frequency
        self._use_rtc = use_rtc
        self._enable_async_prefetch = bool(enable_async_prefetch)
        self._rtc_inference_delay_steps = max(int(rtc_inference_delay_steps), 0)
        self._rtc_use_measured_delay = rtc_use_measured_delay
        self._rtc_handoff_joint_blend = float(np.clip(rtc_handoff_joint_blend, 0.0, 1.0))
        self._client_lock = threading.Lock()
        self._prefetch_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dreamzero_prefetch")

        self._history = {
            "primary": deque(maxlen=FRAME_HISTORY_LENGTH),
            "secondary": deque(maxlen=FRAME_HISTORY_LENGTH),
            "wrist": deque(maxlen=FRAME_HISTORY_LENGTH),
        }
        self._pred_action_chunk: np.ndarray | None = None
        self._actions_from_chunk_completed = 0
        self._has_sent_initial_request = False
        self._session_id = str(uuid.uuid4())
        self._episode_step_idx = 0
        self._last_runtime_rtc_delay_steps = self._rtc_inference_delay_steps
        self._pending_prefetch: PendingPrefetch | None = None
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
        self._history = {
            "primary": deque(maxlen=FRAME_HISTORY_LENGTH),
            "secondary": deque(maxlen=FRAME_HISTORY_LENGTH),
            "wrist": deque(maxlen=FRAME_HISTORY_LENGTH),
        }
        self._pred_action_chunk = None
        self._actions_from_chunk_completed = 0
        self._has_sent_initial_request = False
        self._session_id = str(uuid.uuid4())
        self._episode_step_idx = 0
        self._last_runtime_rtc_delay_steps = self._rtc_inference_delay_steps
        self._pending_prefetch = None
        self._last_debug_step = {}
        self._next_chunk_id = 0
        self._active_chunk_id = None

    def reset(self) -> None:
        self._drain_pending_prefetch()
        with self._client_lock:
            self._client.reset({})
        self._reset_local_state()

    def _append_history(self, primary_image: np.ndarray, secondary_image: np.ndarray, wrist_image: np.ndarray) -> None:
        self._history["primary"].append(primary_image)
        self._history["secondary"].append(secondary_image)
        self._history["wrist"].append(wrist_image)

    def _normalize_action_chunk(self, response: object) -> np.ndarray:
        if isinstance(response, dict):
            if "actions" in response:
                response = response["actions"]
            elif "action" in response:
                response = response["action"]
            else:
                joint_action = None
                gripper_action = None
                for key, value in response.items():
                    if not isinstance(key, str):
                        continue
                    if "joint_position" in key:
                        joint_action = np.asarray(value, dtype=np.float32)
                    elif "gripper_position" in key or key.endswith("gripper"):
                        gripper_action = np.asarray(value, dtype=np.float32)

                if joint_action is None:
                    raise TypeError(
                        f"Policy response dict does not contain an action array. Keys: {list(response.keys())}"
                    )

                if joint_action.ndim == 1:
                    joint_action = joint_action.reshape(1, -1)

                if gripper_action is None:
                    gripper_action = np.zeros((joint_action.shape[0], 1), dtype=np.float32)
                else:
                    if gripper_action.ndim == 0:
                        gripper_action = gripper_action.reshape(1, 1)
                    elif gripper_action.ndim == 1:
                        gripper_action = gripper_action.reshape(-1, 1)
                    if gripper_action.shape[-1] > 1:
                        gripper_action = gripper_action[..., :1]

                response = np.concatenate([joint_action, gripper_action], axis=-1)

        pred_action_chunk = np.array(response, dtype=np.float32, copy=True)
        if pred_action_chunk.ndim != 2 or pred_action_chunk.shape[1] != 8:
            raise ValueError(f"Expected action chunk with shape (N, 8), got {pred_action_chunk.shape}")
        return pred_action_chunk

    def _should_query_policy(self) -> bool:
        if self._pred_action_chunk is None:
            return True
        if self._actions_from_chunk_completed >= len(self._pred_action_chunk):
            return True
        return self._actions_from_chunk_completed >= self._open_loop_horizon

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
        if not self._has_sent_initial_request:
            exterior_0 = _resize_frames(primary_image, height, width)
            exterior_1 = _resize_frames(secondary_image, height, width)
            wrist = _resize_frames(wrist_image, height, width)
        else:
            exterior_0 = _resize_frames(_select_history_frames(self._history["primary"]), height, width)
            exterior_1 = _resize_frames(_select_history_frames(self._history["secondary"]), height, width)
            wrist = _resize_frames(_select_history_frames(self._history["wrist"]), height, width)

        request = {
            "observation/exterior_image_0_left": exterior_0,
            "observation/exterior_image_1_left": exterior_1,
            "observation/wrist_image_left": wrist,
            "observation/joint_position": joint_position.astype(np.float64),
            "observation/cartesian_position": np.zeros((6,), dtype=np.float64),
            "observation/gripper_position": gripper_position.astype(np.float64),
            "prompt": instruction,
            "session_id": self._session_id,
        }
        if self._use_rtc:
            request["rtc_step_idx"] = np.int32(self._episode_step_idx)
            request["rtc_inference_delay_steps"] = np.int32(self._last_runtime_rtc_delay_steps)
        return request

    def _compute_measured_delay_steps(self, policy_roundtrip_ms: float) -> int:
        control_period_ms = 1000.0 / max(self._control_frequency, 1)
        return max(int(np.ceil(policy_roundtrip_ms / control_period_ms)), 0)

    def _run_policy_request(self, request_data: dict) -> dict[str, object]:
        roundtrip_start = time.perf_counter()
        with self._client_lock:
            pred_action_chunk = self._client.infer(request_data)
        policy_roundtrip_ms = (time.perf_counter() - roundtrip_start) * 1000
        pred_action_chunk = self._normalize_action_chunk(pred_action_chunk)

        next_delay_steps = self._last_runtime_rtc_delay_steps
        if self._use_rtc and self._rtc_use_measured_delay:
            next_delay_steps = self._compute_measured_delay_steps(policy_roundtrip_ms)

        return {
            "pred_action_chunk": pred_action_chunk,
            "policy_roundtrip_ms": policy_roundtrip_ms,
            "next_delay_steps": next_delay_steps,
        }

    def _activate_chunk(self, result: dict[str, object]) -> int:
        self._pred_action_chunk = np.asarray(result["pred_action_chunk"], dtype=np.float32)
        self._actions_from_chunk_completed = 0
        self._has_sent_initial_request = True
        if self._use_rtc and self._rtc_use_measured_delay:
            self._last_runtime_rtc_delay_steps = int(result["next_delay_steps"])
        self._next_chunk_id += 1
        self._active_chunk_id = self._next_chunk_id
        return self._active_chunk_id

    def _start_prefetch(
        self,
        request_data: dict,
        request_step_idx: int,
        request_delay_steps: int,
    ) -> None:
        if self._pending_prefetch is not None:
            return
        self._pending_prefetch = PendingPrefetch(
            future=self._prefetch_executor.submit(self._run_policy_request, request_data),
            request_step_idx=int(request_step_idx),
            activation_step_idx=int(request_step_idx + max(request_delay_steps, 0)),
            request_delay_steps=int(max(request_delay_steps, 0)),
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

    def _should_launch_prefetch(self) -> bool:
        if not self._enable_async_prefetch:
            return False
        if self._pred_action_chunk is None:
            return False
        if self._pending_prefetch is not None:
            return False
        chunk_len = len(self._pred_action_chunk)
        if self._actions_from_chunk_completed >= chunk_len:
            return False
        launch_threshold = min(self._open_loop_horizon, max(chunk_len - 1, 0))
        return self._actions_from_chunk_completed >= launch_threshold

    def _should_activate_pending_prefetch(self) -> bool:
        if not self._use_rtc:
            return False
        if self._pending_prefetch is None:
            return False
        return self._episode_step_idx >= self._pending_prefetch.activation_step_idx

    def _smooth_rtc_handoff_action(
        self,
        action: np.ndarray,
        joint_position: np.ndarray,
    ) -> tuple[np.ndarray, bool, float, float]:
        if not self._use_rtc or self._rtc_handoff_joint_blend >= 1.0:
            return action, False, 0.0, 0.0

        obs_joint = np.asarray(joint_position, dtype=np.float32).reshape(-1)
        if obs_joint.shape[0] < 7 or action.shape[0] < 7:
            return action, False, 0.0, 0.0

        smoothed_action = np.array(action, dtype=np.float32, copy=True)
        current_joint = obs_joint[:7]
        raw_joint_target = smoothed_action[:7].copy()
        raw_delta = raw_joint_target - current_joint
        smoothed_action[:7] = current_joint + self._rtc_handoff_joint_blend * raw_delta
        smoothed_delta = smoothed_action[:7] - current_joint

        return (
            smoothed_action,
            True,
            float(np.max(np.abs(raw_delta))),
            float(np.max(np.abs(smoothed_delta))),
        )

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
        self._append_history(primary_image, secondary_image, wrist_image)
        history_append_done = time.perf_counter()
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
        request_rtc_delay_steps = self._last_runtime_rtc_delay_steps if self._use_rtc else 0
        handoff_smoothed = False
        handoff_joint_delta_max_before = 0.0
        handoff_joint_delta_max_after = 0.0
        launched_prefetch_request_step_idx = -1
        launched_prefetch_activation_step_idx = -1

        if self._should_activate_pending_prefetch():
            had_prior_chunk = self._pred_action_chunk is not None
            promoted_pending, waited_policy_ms, pending_roundtrip_ms = self._consume_pending_prefetch(wait=True)
            if promoted_pending:
                queried_policy = True
                waited_for_policy = waited_policy_ms > 0.0
                policy_wait_ms += waited_policy_ms
                policy_roundtrip_ms = pending_roundtrip_ms
                activated_chunk_this_step = had_prior_chunk
                chunk_activation_source = "rtc_prefetch_activation"

        if self._should_launch_prefetch():
            # If a pending chunk was activated earlier in this infer() call, measured-delay mode may
            # have updated the runtime delay estimate. Use the freshest value for the new prefetch.
            request_rtc_delay_steps = self._last_runtime_rtc_delay_steps if self._use_rtc else 0
            build_request_start = time.perf_counter()
            request_step_idx = self._episode_step_idx
            request_data = self._build_request_data(
                primary_image,
                secondary_image,
                wrist_image,
                joint_position,
                gripper_position,
                instruction,
            )
            build_request_ms = (time.perf_counter() - build_request_start) * 1000
            self._start_prefetch(
                request_data,
                request_step_idx=request_step_idx,
                request_delay_steps=request_rtc_delay_steps,
            )
            queried_policy = True
            launched_prefetch = True
            launched_prefetch_request_step_idx = int(request_step_idx)
            launched_prefetch_activation_step_idx = int(request_step_idx + max(request_rtc_delay_steps, 0))

        if self._pred_action_chunk is None or self._actions_from_chunk_completed >= len(self._pred_action_chunk):
            had_prior_chunk = self._pred_action_chunk is not None
            promoted_pending, waited_policy_ms, pending_roundtrip_ms = self._consume_pending_prefetch(wait=True)
            if promoted_pending:
                queried_policy = True
                waited_for_policy = True
                policy_wait_ms += waited_policy_ms
                policy_roundtrip_ms = pending_roundtrip_ms
                activated_chunk_this_step = had_prior_chunk
                chunk_activation_source = "prefetch_on_exhaustion"
            else:
                request_rtc_delay_steps = self._last_runtime_rtc_delay_steps if self._use_rtc else 0
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
                waited_for_policy = True
                queried_policy = True
                self._activate_chunk(result)
                policy_roundtrip_ms = float(result["policy_roundtrip_ms"])
                activated_chunk_this_step = had_prior_chunk
                chunk_activation_source = "sync_request"

        action_start = time.perf_counter()
        chunk_local_step_idx = int(self._actions_from_chunk_completed)
        raw_action = np.asarray(self._pred_action_chunk[chunk_local_step_idx], dtype=np.float32)
        action = np.array(raw_action, dtype=np.float32, copy=True)
        chunk_id = self._active_chunk_id
        if activated_chunk_this_step:
            action, handoff_smoothed, handoff_joint_delta_max_before, handoff_joint_delta_max_after = (
                self._smooth_rtc_handoff_action(action, joint_position_array)
            )
        self._actions_from_chunk_completed += 1
        self._episode_step_idx += 1

        if action[-1].item() > 0.5:
            action[-1] = 1.0
        else:
            action[-1] = 0.0

        infer_end = time.perf_counter()
        chunk_size = int(len(self._pred_action_chunk)) if self._pred_action_chunk is not None else 0
        self._last_timing = {
            "queried_policy": queried_policy,
            "history_append_ms": (history_append_done - infer_start) * 1000,
            "build_request_ms": build_request_ms,
            "policy_wait_ms": policy_wait_ms,
            "policy_roundtrip_ms": policy_roundtrip_ms,
            "action_postprocess_ms": (infer_end - action_start) * 1000,
            "infer_total_ms": (infer_end - infer_start) * 1000,
            "chunk_size": chunk_size,
            "remaining_actions_in_chunk": max(chunk_size - self._actions_from_chunk_completed, 0),
            "launched_prefetch": launched_prefetch,
            "waited_for_policy": waited_for_policy,
            "prefetch_in_flight": self._pending_prefetch is not None,
            "prefetch_activation_step_idx": (
                -1 if self._pending_prefetch is None else int(self._pending_prefetch.activation_step_idx)
            ),
            "handoff_smoothed": handoff_smoothed,
            "handoff_joint_delta_max_before": handoff_joint_delta_max_before,
            "handoff_joint_delta_max_after": handoff_joint_delta_max_after,
            "rtc_enabled": self._use_rtc,
            "rtc_step_idx": current_step_idx,
            "rtc_request_delay_steps": request_rtc_delay_steps,
            "rtc_next_delay_steps": self._last_runtime_rtc_delay_steps if self._use_rtc else 0,
        }
        self._last_debug_step = {
            "trace_type": "dreamzero_rtc_step",
            "session_id": self._session_id,
            "executed_step_idx": current_step_idx,
            "current_chunk_id": None if chunk_id is None else int(chunk_id),
            "chunk_local_step_idx": chunk_local_step_idx,
            "chunk_size": chunk_size,
            "remaining_actions_after_step": max(chunk_size - self._actions_from_chunk_completed, 0),
            "activated_chunk_this_step": bool(activated_chunk_this_step),
            "chunk_activation_source": chunk_activation_source,
            "queried_policy": bool(queried_policy),
            "launched_prefetch": bool(launched_prefetch),
            "waited_for_policy": bool(waited_for_policy),
            "prefetch_in_flight": bool(self._pending_prefetch is not None),
            "prefetch_pending_activation_step_idx": (
                -1 if self._pending_prefetch is None else int(self._pending_prefetch.activation_step_idx)
            ),
            "launched_prefetch_request_step_idx": launched_prefetch_request_step_idx,
            "launched_prefetch_activation_step_idx": launched_prefetch_activation_step_idx,
            "handoff_smoothed": bool(handoff_smoothed),
            "handoff_joint_delta_max_before": float(handoff_joint_delta_max_before),
            "handoff_joint_delta_max_after": float(handoff_joint_delta_max_after),
            "rtc_enabled": bool(self._use_rtc),
            "rtc_request_delay_steps": int(request_rtc_delay_steps),
            "rtc_next_delay_steps": int(self._last_runtime_rtc_delay_steps if self._use_rtc else 0),
            "policy_wait_ms": float(policy_wait_ms),
            "policy_roundtrip_ms": float(policy_roundtrip_ms),
            "build_request_ms": float(build_request_ms),
            "infer_total_ms": float((infer_end - infer_start) * 1000),
            "joint_position": joint_position_array.astype(np.float32),
            "gripper_position": gripper_position_array.astype(np.float32),
            "raw_action_before_handoff": raw_action.astype(np.float32),
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

    policy_client = DreamZeroRealPolicyClient(
        args.remote_host,
        args.remote_port,
        args.open_loop_horizon,
        args.control_frequency,
        use_rtc=args.use_rtc,
        enable_async_prefetch=(args.enable_async_prefetch or args.use_rtc),
        rtc_inference_delay_steps=args.rtc_inference_delay_steps,
        rtc_use_measured_delay=args.rtc_use_measured_delay,
        rtc_handoff_joint_blend=args.rtc_handoff_joint_blend,
    )

    records: list[dict] = []
    warned_missing_right = False
    trace_handle = None
    if args.rtc_debug_trace_path:
        trace_dir = os.path.dirname(args.rtc_debug_trace_path)
        if trace_dir:
            os.makedirs(trace_dir, exist_ok=True)
        trace_handle = open(args.rtc_debug_trace_path, "a", encoding="utf-8")

    while True:
        instruction = input("Enter instruction: ")
        episode_session_id = policy_client.session_id
        episode_video_frames: list[np.ndarray] = []

        bar = tqdm.tqdm(range(args.max_timesteps))
        print("Running rollout... press Ctrl+C to stop early.")
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
                video_frame = _build_input_video_frame(
                    primary_image,
                    secondary_image,
                    curr_obs["wrist_image"],
                    policy_client.image_resolution,
                )
                episode_video_frames.append(video_frame)
                if t_step == 0:
                    imageio.imwrite("robot_camera_views.png", video_frame)
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
                        f"history={float(infer_timing.get('history_append_ms', 0.0)):.1f}ms "
                        f"build={float(infer_timing.get('build_request_ms', 0.0)):.1f}ms "
                        f"ws_wait={float(infer_timing.get('policy_wait_ms', 0.0)):.1f}ms "
                        f"ws_roundtrip={float(infer_timing.get('policy_roundtrip_ms', 0.0)):.1f}ms "
                        f"post={float(infer_timing.get('action_postprocess_ms', 0.0)):.1f}ms "
                        f"chunk={int(infer_timing.get('chunk_size', 0))} "
                        f"remain={int(infer_timing.get('remaining_actions_in_chunk', 0))} "
                        f"prefetch_activate={int(infer_timing.get('prefetch_activation_step_idx', -1))} "
                        f"handoff_smooth={bool(infer_timing.get('handoff_smoothed', False))} "
                        f"joint_delta={float(infer_timing.get('handoff_joint_delta_max_before', 0.0)):.3f}->"
                        f"{float(infer_timing.get('handoff_joint_delta_max_after', 0.0)):.3f} "
                        f"rtc={bool(infer_timing.get('rtc_enabled', False))} "
                        f"rtc_step={int(infer_timing.get('rtc_step_idx', 0))} "
                        f"rtc_delay={int(infer_timing.get('rtc_request_delay_steps', 0))}->"
                        f"{int(infer_timing.get('rtc_next_delay_steps', 0))}) "
                        f"env_step={(env_step_done - env_step_start) * 1000:.1f}ms "
                        f"sleep={sleep_ms:.1f}ms "
                        f"loop={(time.perf_counter() - loop_start) * 1000:.1f}ms"
                    )

                if trace_handle is not None:
                    debug_step = policy_client.last_debug_step
                    trace_record = {
                        **debug_step,
                        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "episode_session_id": episode_session_id,
                        "instruction": instruction,
                        "loop_step": int(t_step),
                        "obs_ms": float((obs_done - obs_start) * 1000),
                        "compose_ms": float((video_frame_done - video_frame_start) * 1000),
                        "client_infer_ms": float((infer_done - infer_start) * 1000),
                        "env_step_ms": float((env_step_done - env_step_start) * 1000),
                        "sleep_ms": float(sleep_ms),
                        "loop_total_ms": float((time.perf_counter() - loop_start) * 1000),
                    }
                    trace_handle.write(json.dumps(trace_record, default=_json_default) + "\n")
                    trace_handle.flush()
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
    csv_filename = os.path.join(args.results_dir, f"eval_{timestamp}.csv")
    pd.DataFrame(records).to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")
    if trace_handle is not None:
        trace_handle.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
