# ruff: noqa

import contextlib
from collections import deque
import dataclasses
import datetime
import faulthandler
import os
import signal
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
    log_timing: bool = False
    timing_log_interval: int = 1

    # Remote server parameters
    remote_host: str = "0.0.0.0"
    remote_port: int = 9000

    # Logging and outputs
    video_output_dir: str | None = None
    results_dir: str = "results"


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
    def __init__(self, remote_host: str, remote_port: int, open_loop_horizon: int) -> None:
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

        self._history = {
            "primary": deque(maxlen=FRAME_HISTORY_LENGTH),
            "secondary": deque(maxlen=FRAME_HISTORY_LENGTH),
            "wrist": deque(maxlen=FRAME_HISTORY_LENGTH),
        }
        self._pred_action_chunk: np.ndarray | None = None
        self._actions_from_chunk_completed = 0
        self._has_sent_initial_request = False
        self._session_id = str(uuid.uuid4())
        self._last_timing: dict[str, float | int | bool] = {}

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def image_resolution(self) -> tuple[int, int]:
        return self._image_resolution

    @property
    def last_timing(self) -> dict[str, float | int | bool]:
        return self._last_timing.copy()

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

    def reset(self) -> None:
        self._client.reset({})
        self._reset_local_state()

    def _append_history(self, primary_image: np.ndarray, secondary_image: np.ndarray, wrist_image: np.ndarray) -> None:
        self._history["primary"].append(primary_image)
        self._history["secondary"].append(secondary_image)
        self._history["wrist"].append(wrist_image)

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

        return {
            "observation/exterior_image_0_left": exterior_0,
            "observation/exterior_image_1_left": exterior_1,
            "observation/wrist_image_left": wrist,
            "observation/joint_position": joint_position.astype(np.float64),
            "observation/cartesian_position": np.zeros((6,), dtype=np.float64),
            "observation/gripper_position": gripper_position.astype(np.float64),
            "prompt": instruction,
            "session_id": self._session_id,
        }

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

        queried_policy = self._should_query_policy()
        build_request_ms = 0.0
        policy_roundtrip_ms = 0.0

        if queried_policy:
            build_request_start = time.perf_counter()
            request_data = self._build_request_data(
                primary_image,
                secondary_image,
                wrist_image,
                joint_position,
                gripper_position,
                instruction,
            )
            build_request_ms = (time.perf_counter() - build_request_start) * 1000
            self._actions_from_chunk_completed = 0
            roundtrip_start = time.perf_counter()
            with prevent_keyboard_interrupt():
                pred_action_chunk = self._client.infer(request_data)
            policy_roundtrip_ms = (time.perf_counter() - roundtrip_start) * 1000
            pred_action_chunk = np.array(pred_action_chunk, dtype=np.float32, copy=True)

            if pred_action_chunk.ndim != 2 or pred_action_chunk.shape[1] != 8:
                raise ValueError(f"Expected action chunk with shape (N, 8), got {pred_action_chunk.shape}")

            self._pred_action_chunk = pred_action_chunk
            self._has_sent_initial_request = True

        action_start = time.perf_counter()
        action = np.asarray(self._pred_action_chunk[self._actions_from_chunk_completed], dtype=np.float32)
        self._actions_from_chunk_completed += 1

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
            "policy_roundtrip_ms": policy_roundtrip_ms,
            "action_postprocess_ms": (infer_end - action_start) * 1000,
            "infer_total_ms": (infer_end - infer_start) * 1000,
            "chunk_size": chunk_size,
            "remaining_actions_in_chunk": max(chunk_size - self._actions_from_chunk_completed, 0),
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

    policy_client = DreamZeroRealPolicyClient(args.remote_host, args.remote_port, args.open_loop_horizon)

    records: list[dict] = []
    warned_missing_right = False

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
                        f"history={float(infer_timing.get('history_append_ms', 0.0)):.1f}ms "
                        f"build={float(infer_timing.get('build_request_ms', 0.0)):.1f}ms "
                        f"ws_wait={float(infer_timing.get('policy_roundtrip_ms', 0.0)):.1f}ms "
                        f"post={float(infer_timing.get('action_postprocess_ms', 0.0)):.1f}ms "
                        f"chunk={int(infer_timing.get('chunk_size', 0))} "
                        f"remain={int(infer_timing.get('remaining_actions_in_chunk', 0))}) "
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
    csv_filename = os.path.join(args.results_dir, f"eval_{timestamp}.csv")
    pd.DataFrame(records).to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")


if __name__ == "__main__":
    main(tyro.cli(Args))
