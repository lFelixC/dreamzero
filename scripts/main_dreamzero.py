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
DEFAULT_IMAGE_RESOLUTION = (160, 320)


@dataclasses.dataclass
class Args:
    # Hardware parameters
    left_camera_id: str = "36517165"
    right_camera_id: str | None = None
    wrist_camera_id: str = "13337231"
    missing_right_camera_strategy: Literal["duplicate_left", "mask_right"] = "mask_right"

    # Rollout parameters
    max_timesteps: int = 6000
    open_loop_horizon: int = 24
    control_frequency: int = 15

    # Remote server parameters
    remote_host: str = "127.0.0.1"
    remote_port: int = 5002

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

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def image_resolution(self) -> tuple[int, int]:
        return self._image_resolution

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
        self._append_history(primary_image, secondary_image, wrist_image)

        if self._should_query_policy():
            request_data = self._build_request_data(
                primary_image,
                secondary_image,
                wrist_image,
                joint_position,
                gripper_position,
                instruction,
            )
            self._actions_from_chunk_completed = 0
            with prevent_keyboard_interrupt():
                pred_action_chunk = self._client.infer(request_data)
            pred_action_chunk = np.array(pred_action_chunk, dtype=np.float32, copy=True)

            if pred_action_chunk.ndim != 2 or pred_action_chunk.shape[1] != 8:
                raise ValueError(f"Expected action chunk with shape (N, 8), got {pred_action_chunk.shape}")

            self._pred_action_chunk = pred_action_chunk
            self._has_sent_initial_request = True

        action = np.asarray(self._pred_action_chunk[self._actions_from_chunk_completed], dtype=np.float32)
        self._actions_from_chunk_completed += 1

        if action[-1].item() > 0.5:
            action[-1] = 1.0
        else:
            action[-1] = 0.0
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
        # instruction = input("Enter instruction: ")
        instruction = "pick up the blue cube into the red box"
        episode_session_id = policy_client.session_id
        episode_video_frames: list[np.ndarray] = []

        bar = tqdm.tqdm(range(args.max_timesteps))
        print("Running rollout... press Ctrl+C to stop early.")
        for t_step in bar:
            start_time = time.time()
            try:
                curr_obs = _extract_observation(args, env.get_observation())
                primary_image = curr_obs["left_image"]
                secondary_image = curr_obs["right_image"]
                secondary_image, warned_missing_right = _resolve_secondary_image(
                    args,
                    primary_image,
                    secondary_image,
                    warned_missing_right,
                )

                video_frame = _build_input_video_frame(
                    primary_image,
                    secondary_image,
                    curr_obs["wrist_image"],
                    policy_client.image_resolution,
                )
                episode_video_frames.append(video_frame)
                if t_step == 0:
                    imageio.imwrite("robot_camera_views.png", video_frame)

                action = policy_client.infer(
                    primary_image=primary_image,
                    secondary_image=secondary_image,
                    wrist_image=curr_obs["wrist_image"],
                    joint_position=curr_obs["joint_position"],
                    gripper_position=curr_obs["gripper_position"],
                    instruction=instruction,
                )
                print(action)
                env.step(action)

                elapsed_time = time.time() - start_time
                target_period = 1 / args.control_frequency
                if elapsed_time < target_period:
                    time.sleep(target_period - elapsed_time)
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
