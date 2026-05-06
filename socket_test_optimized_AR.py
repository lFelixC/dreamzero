import dataclasses
import json
import logging
import socket
import asyncio
import os
import http
import logging
import shutil
import tempfile
import time
import traceback
import torch
import tyro
from einops import rearrange
import datetime
from pathlib import Path
from typing import Literal

from eval_utils.torch_compile_backend import configure_torch_compile_backend
from eval_utils.inference_rtc import (
    RTCSessionState,
    compute_prev_chunk_left_over,
    extract_optional_int,
    session_key,
    trim_action_chunk_for_delay,
)
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
from groot.vla.data.schema import EmbodimentTag
import imageio
import numpy as np

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames
from tianshou.data import Batch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

# Use roboarena policy server interface
from eval_utils.policy_server import WebsocketPolicyServer as RoboarenaServer
from eval_utils.policy_server import PolicyServerConfig
from eval_utils.serve_dreamzero_wan22 import (
    _get_expected_video_resolution,
    _resize_frames_to_resolution,
)
from groot.vla.utils.nvtx_utils import nvtx_range

DEFAULT_TORCH_COMPILE_BACKEND = configure_torch_compile_backend(default_backend="cudagraphs")
RESET_FLAG_KEY = "__reset_policy_state__"
WAN_JOINT_MODEL_TARGET = "groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk.CausalWanModel"
WAN_MOT_MODEL_TARGET = "groot.vla.model.dreamzero.modules.dreamzero_mot.MoTCausalWanModel"

logger = logging.getLogger(__name__)

@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    timeout_seconds: int = 604800  # 7 days default, configurable
    handshake_timeout_seconds: float | None = 0.0  # <= 0 disables the opening-handshake timeout.
    model_path: str = "./checkpoints/dreamzero"
    architecture: Literal["auto", "joint", "mot"] = "auto"
    allow_architecture_override: bool = False
    enable_dit_cache: bool = False
    index: int = 0
    max_chunk_size: int | None = None  # If None, use config value. Otherwise override max_chunk_size for inference.
    use_rtc: bool = False
    rtc_execution_horizon: int = 10
    rtc_max_guidance_weight: float = 10.0
    rtc_prefix_attention_schedule: str = "EXP"
    rtc_guidance_max_steps: int = 4
    rtc_guidance_step_stride: int = 1


def _action_head_inner_cfg(config: dict) -> dict | None:
    action_head_cfg = config.get("action_head_cfg")
    if not isinstance(action_head_cfg, dict):
        return None
    inner_cfg = action_head_cfg.get("config", action_head_cfg)
    return inner_cfg if isinstance(inner_cfg, dict) else None


def _read_checkpoint_architecture(model_path: str) -> str | None:
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return None
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    inner_cfg = _action_head_inner_cfg(config)
    if inner_cfg is None:
        return None
    architecture = inner_cfg.get("architecture")
    return str(architecture) if architecture in ("joint", "mot") else None


def _copy_checkpoint_with_architecture_override(
    model_path: str,
    architecture: Literal["joint", "mot"],
) -> tuple[str, tempfile.TemporaryDirectory]:
    """Create a temporary checkpoint view with only config.json patched.

    This is intentionally opt-in. Joint and MoT checkpoints have different
    parameter layouts, so overriding the architecture is mostly useful for
    recovering from an incorrectly saved config.
    """
    src_dir = Path(model_path).resolve()
    config_path = src_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing checkpoint config: {config_path}")

    temp_dir = tempfile.TemporaryDirectory(prefix="dreamzero_arch_override_")
    dst_dir = Path(temp_dir.name)

    for child in src_dir.iterdir():
        if child.name == "config.json":
            continue
        dst = dst_dir / child.name
        try:
            os.symlink(child, dst, target_is_directory=child.is_dir())
        except OSError:
            if child.is_dir():
                shutil.copytree(child, dst, symlinks=True)
            else:
                shutil.copy2(child, dst)

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    inner_cfg = _action_head_inner_cfg(config)
    if inner_cfg is None:
        raise ValueError(f"Could not find action_head_cfg.config in {config_path}")

    inner_cfg["architecture"] = architecture
    diffusion_cfg = inner_cfg.get("diffusion_model_cfg")
    if isinstance(diffusion_cfg, dict):
        diffusion_cfg["_target_"] = WAN_MOT_MODEL_TARGET if architecture == "mot" else WAN_JOINT_MODEL_TARGET

    patched_config_path = dst_dir / "config.json"
    with patched_config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    return str(dst_dir), temp_dir


def _resolve_model_path_for_architecture(args: Args) -> tuple[str, str, tempfile.TemporaryDirectory | None]:
    checkpoint_architecture = _read_checkpoint_architecture(args.model_path)
    if args.architecture == "auto":
        effective_architecture = checkpoint_architecture or "joint"
        logger.info(
            "Using checkpoint WAM architecture=%s (model_path=%s)",
            effective_architecture,
            args.model_path,
        )
        return args.model_path, effective_architecture, None

    requested_architecture = args.architecture
    if checkpoint_architecture == requested_architecture:
        logger.info(
            "Using requested WAM architecture=%s from checkpoint config",
            requested_architecture,
        )
        return args.model_path, requested_architecture, None

    mismatch = (
        f"Requested architecture={requested_architecture!r}, but checkpoint "
        f"{args.model_path!r} declares architecture={checkpoint_architecture or 'joint(default)'!r}."
    )
    if not args.allow_architecture_override:
        raise ValueError(
            mismatch
            + " Use a matching joint/MoT checkpoint, or pass --allow-architecture-override only if the saved config is wrong."
        )

    logger.warning("%s Forcing a temporary config override.", mismatch)
    override_path, temp_dir = _copy_checkpoint_with_architecture_override(
        args.model_path,
        requested_architecture,
    )
    return override_path, requested_architecture, temp_dir


class ARDroidRoboarenaPolicy:
    """Wrapper policy that implements roboarena.policy.BasePolicy interface for AR_droid.
    
    Handles:
    - Observation format conversion (roboarena -> AR_droid format)
    - Frame accumulation across calls (roboarena sends single frames, AR_droid expects multi-frame video)
    - Action format conversion (AR_droid dict -> roboarena array format)
    - Distributed inference coordination
    """
    
    # Number of frames to accumulate after the first call
    FRAMES_PER_CHUNK = 4
    
    def __init__(
        self,
        groot_policy: GrootSimPolicy,
        signal_group: dist.ProcessGroup,
        image_height: int,
        image_width: int,
        output_dir: str | None = None,
        max_chunk_size: int | None = None,
        use_rtc: bool = False,
        rtc_execution_horizon: int = 10,
        rtc_max_guidance_weight: float = 10.0,
        rtc_prefix_attention_schedule: str = "EXP",
        rtc_guidance_max_steps: int = 4,
        rtc_guidance_step_stride: int = 1,
    ) -> None:
        self._policy = groot_policy
        self._signal_group = signal_group
        self._image_height = image_height
        self._image_width = image_width
        self._output_dir = output_dir
        self._max_chunk_size = max_chunk_size
        self._use_rtc = use_rtc
        self._rtc_execution_horizon = rtc_execution_horizon
        self._rtc_max_guidance_weight = rtc_max_guidance_weight
        self._rtc_prefix_attention_schedule = rtc_prefix_attention_schedule
        self._rtc_guidance_max_steps = rtc_guidance_max_steps
        self._rtc_guidance_step_stride = rtc_guidance_step_stride
        
        # Frame buffers for accumulation (per camera view)
        self._frame_buffers: dict[str, list[np.ndarray]] = {
            "video.exterior_image_1_left": [],
            "video.exterior_image_2_left": [],
            "video.wrist_image_left": [],
        }
        self._call_count = 0
        self._is_first_call = True
        
        # Session tracking - reset state when new session starts
        self._current_session_id: str | None = None
        self._warned_single_external_fallback = False
        self._rtc_session_states: dict[str, RTCSessionState] = {}
        
        # Video across time for saving (similar to original server)
        self.video_across_time = []
        self._msg_index = 0
        
        # Create output directory if specified
        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)

    def _infer_phase(self, rtc_requested: bool, had_previous_chunk: bool) -> str:
        if not rtc_requested:
            return "rtc_off_first" if self._is_first_call else "rtc_off_steady"
        if had_previous_chunk:
            return "rtc_on_steady"
        return "rtc_on_first_chunk"

    @staticmethod
    def _normalize_image_array(data: np.ndarray, key: str) -> np.ndarray:
        arr = np.asarray(data)
        if arr.ndim not in (3, 4):
            raise ValueError(f"{key} must have shape (H, W, 3) or (T, H, W, 3), got {arr.shape}")
        if arr.shape[-1] != 3:
            raise ValueError(f"{key} must end with 3 color channels, got {arr.shape}")

        if np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float32)
            min_val = float(np.nanmin(arr))
            max_val = float(np.nanmax(arr))
            if min_val >= 0.0 and max_val <= 1.0:
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            elif min_val >= -1.0 and max_val <= 1.0:
                arr = ((arr + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)

        return np.ascontiguousarray(arr)

    def _append_frames_to_buffer(self, droid_key: str, data: np.ndarray) -> None:
        if data.ndim == 4:
            self._frame_buffers[droid_key].extend(list(data))
        else:
            self._frame_buffers[droid_key].append(data)

    @staticmethod
    def _extract_action_dict(action_chunk: Batch | dict) -> dict[str, np.ndarray | torch.Tensor]:
        if isinstance(action_chunk, dict):
            return {k: v for k, v in action_chunk.items() if isinstance(k, str) and k.startswith("action.")}

        try:
            return {
                k: v
                for k, v in action_chunk.items()
                if isinstance(k, str) and k.startswith("action.")
            }
        except Exception:
            action_dict: dict[str, np.ndarray | torch.Tensor] = {}
            for key in dir(action_chunk):
                if not key.startswith("action."):
                    continue
                try:
                    action_dict[key] = getattr(action_chunk, key)
                except AttributeError:
                    continue
            return action_dict

    def _reset_model_temporal_state(self) -> None:
        if hasattr(self._policy, "reset_inference_state"):
            self._policy.reset_inference_state()
            return
        if hasattr(self._policy.trained_model, "action_head") and hasattr(
            self._policy.trained_model.action_head, "current_start_frame"
        ):
            self._policy.trained_model.action_head.current_start_frame = 0
    
    def _convert_observation(self, obs: dict) -> dict:
        """Convert roboarena observation format to AR_droid format.
        
        Roboarena format:
            - observation/exterior_image_0_left: left exterior camera, (H, W, 3) or (T, H, W, 3)
            - observation/exterior_image_1_left: right exterior camera, (H, W, 3) or (T, H, W, 3)
            - observation/wrist_image_left: (H, W, 3) single frame
            - observation/joint_position: (7,)
            - observation/gripper_position: (1,)
            - prompt: str
        
        AR_droid format:
            - video.exterior_image_1_left: (T, H, W, 3) multi-frame
            - video.exterior_image_2_left: (T, H, W, 3) multi-frame
            - video.wrist_image_left: (T, H, W, 3) multi-frame
            - state.joint_position: (1, 7)
            - state.gripper_position: (1, 1)
            - annotation.language.action_text: str
        """
        converted = {}
        
        # Keep the DROID ordering expected by DreamZero:
        # obs 0 -> left exterior, obs 1 -> right exterior.
        image_key_mapping = {
            "observation/exterior_image_0_left": "video.exterior_image_1_left",
            "observation/exterior_image_1_left": "video.exterior_image_2_left",
            "observation/wrist_image_left": "video.wrist_image_left",
        }
        
        processed_images: dict[str, np.ndarray] = {}

        # Accumulate frames for each camera view
        for roboarena_key, droid_key in image_key_mapping.items():
            if roboarena_key in obs:
                data = self._normalize_image_array(obs[roboarena_key], roboarena_key)
                data = _resize_frames_to_resolution(data, self._image_height, self._image_width)
                processed_images[roboarena_key] = data
                self._append_frames_to_buffer(droid_key, data)

        # Accept either one or two exterior cameras from RoboArena.
        # If only one exterior stream is present, duplicate it into the missing DreamZero slot.
        exterior_0 = processed_images.get("observation/exterior_image_0_left")
        exterior_1 = processed_images.get("observation/exterior_image_1_left")
        if exterior_0 is None and exterior_1 is not None:
            if not self._warned_single_external_fallback:
                logger.warning(
                    "Only observation/exterior_image_1_left was provided; duplicating it to fill missing observation/exterior_image_0_left."
                )
                self._warned_single_external_fallback = True
            self._append_frames_to_buffer("video.exterior_image_1_left", exterior_1)
        elif exterior_1 is None and exterior_0 is not None:
            if not self._warned_single_external_fallback:
                logger.warning(
                    "Only observation/exterior_image_0_left was provided; duplicating it to fill missing observation/exterior_image_1_left."
                )
                self._warned_single_external_fallback = True
            self._append_frames_to_buffer("video.exterior_image_2_left", exterior_0)

        # Determine how many frames to use
        if self._is_first_call:
            # First call: use only 1 frame
            num_frames = 1
        else:
            # Subsequent calls: use exactly FRAMES_PER_CHUNK frames
            num_frames = self.FRAMES_PER_CHUNK
        
        # Build video tensors from accumulated frames
        for droid_key, buffer in self._frame_buffers.items():
            if len(buffer) > 0:
                if len(buffer) >= num_frames:
                    # Take the last num_frames frames
                    frames_to_use = buffer[-num_frames:]
                else:
                    # Pad by repeating the first frame to reach num_frames
                    frames_to_use = buffer.copy()
                    while len(frames_to_use) < num_frames:
                        # Prepend the first frame to pad
                        frames_to_use.insert(0, buffer[0])
                # Stack to (T, H, W, C)
                video = np.stack(frames_to_use, axis=0)
                converted[droid_key] = video
        
        # Convert state observations
        if "observation/joint_position" in obs:
            joint_pos = np.asarray(obs["observation/joint_position"])
            # Reshape to (1, 7) if needed
            if joint_pos.ndim == 1:
                joint_pos = joint_pos.reshape(1, -1)
            converted["state.joint_position"] = joint_pos.astype(np.float64)
        else:
            converted["state.joint_position"] = np.zeros((1, 7), dtype=np.float64)
        
        if "observation/gripper_position" in obs:
            gripper_pos = np.asarray(obs["observation/gripper_position"])
            # Reshape to (1, 1) if needed
            if gripper_pos.ndim == 1:
                gripper_pos = gripper_pos.reshape(1, -1)
            converted["state.gripper_position"] = gripper_pos.astype(np.float64)
        else:
            converted["state.gripper_position"] = np.zeros((1, 1), dtype=np.float64)
        
        # Convert prompt
        prompt = obs.get("prompt", "")
        if isinstance(prompt, np.ndarray):
            prompt = prompt.item() if prompt.size == 1 else prompt.reshape(-1)[0]
        converted["annotation.language.action_text"] = str(prompt)
        
        return converted
    
    def _convert_action(self, action_dict: dict) -> np.ndarray:
        """Convert AR_droid action dict to roboarena action array.
        
        AR_droid format:
            - action.joint_position: (N, 7)
            - action.gripper_position: (N,) or (N, 1)
        
        Roboarena format:
            - action: (N, 8) - 7 joint positions + 1 gripper
        """
        joint_action = None
        gripper_action = None
        
        # Extract actions from dict
        for key, value in action_dict.items():
            if "joint_position" in key:
                joint_action = value
            elif "gripper_position" in key or "gripper" in key:
                gripper_action = value
        
        if joint_action is None:
            # Fallback: return zeros
            return np.zeros((1, 8), dtype=np.float32)
        
        # Convert to numpy if tensor
        if isinstance(joint_action, torch.Tensor):
            joint_action = joint_action.cpu().numpy()
        
        # Ensure 2D shape (N, 7)
        if joint_action.ndim == 1:
            joint_action = joint_action.reshape(1, -1)
        
        N = joint_action.shape[0]
        
        # Handle gripper action
        if gripper_action is not None:
            if isinstance(gripper_action, torch.Tensor):
                gripper_action = gripper_action.cpu().numpy()
            # Reshape to (N, 1) if needed
            if gripper_action.ndim == 1:
                gripper_action = gripper_action.reshape(-1, 1)
            elif gripper_action.ndim == 0:
                gripper_action = gripper_action.reshape(1, 1)
            if gripper_action.shape[-1] > 1:
                gripper_action = gripper_action[..., :1]
        else:
            gripper_action = np.zeros((N, 1), dtype=np.float32)
        
        # Concatenate: (N, 7) + (N, 1) -> (N, 8)
        action = np.concatenate([joint_action, gripper_action], axis=-1).astype(np.float32)
        if self._max_chunk_size is not None and self._max_chunk_size > 0:
            action = action[: self._max_chunk_size]
        
        return action
    
    def _get_session_state(self, session_id: str | None) -> RTCSessionState:
        key = session_key(session_id)
        if key not in self._rtc_session_states:
            self._rtc_session_states[key] = RTCSessionState()
        return self._rtc_session_states[key]

    def _broadcast_payload_to_workers(self, payload: dict, phase: str = "unknown") -> None:
        """Broadcast batch data plus model kwargs from rank 0 to all other ranks."""
        import pickle

        with nvtx_range(f"dreamzero.ar.infer[{phase}].pickle_payload"):
            serialized = pickle.dumps(payload)
            data_size = len(serialized)

        with nvtx_range(f"dreamzero.ar.infer[{phase}].dist_broadcast.size"):
            size_tensor = torch.tensor([data_size], dtype=torch.int64, device="cuda")
            dist.broadcast(size_tensor, src=0)

        with nvtx_range(f"dreamzero.ar.infer[{phase}].dist_broadcast.payload"):
            data_tensor = torch.frombuffer(serialized, dtype=torch.uint8).cuda()
            dist.broadcast(data_tensor, src=0)

    def _set_reset_flag(self, converted_obs: dict, should_reset: bool) -> dict:
        worker_obs = dict(converted_obs)
        worker_obs[RESET_FLAG_KEY] = bool(should_reset)
        return worker_obs
    
    def infer(self, obs: dict) -> np.ndarray:
        """Infer actions from observations.
        
        Args:
            obs: Observation dict in roboarena format
            
        Returns:
            action: (N, 8) action array
        """
        # Check for session change - reset state if new session
        session_id = obs.get("session_id", None)
        should_reset = session_id is not None and session_id != self._current_session_id
        if should_reset:
            if self._current_session_id is not None:
                logger.info(f"Session changed from '{self._current_session_id}' to '{session_id}', resetting state")
                # Reset state for new session
                with nvtx_range("dreamzero.ar.reset.on_session_change"):
                    self._reset_state()
            else:
                logger.info(f"New session started: '{session_id}'")
            self._current_session_id = session_id

        rtc_step_idx = extract_optional_int(obs.get("rtc_step_idx", None))
        rtc_inference_delay_steps = max(
            extract_optional_int(obs.get("rtc_inference_delay_steps", 0), default=0) or 0,
            0,
        )
        rtc_requested = rtc_step_idx is not None
        if rtc_inference_delay_steps > 0 and not rtc_requested:
            logger.warning(
                "Received rtc_inference_delay_steps=%d without rtc_step_idx; ignoring RTC delay metadata for this call.",
                rtc_inference_delay_steps,
            )
        
        self._msg_index += 1
        self._call_count += 1
        
        session_state = self._get_session_state(session_id) if rtc_requested else None
        had_previous_chunk = bool(
            rtc_requested and session_state is not None and session_state.last_action_chunk_abs is not None
        )
        phase = self._infer_phase(rtc_requested, had_previous_chunk)
        with nvtx_range(f"dreamzero.ar.infer[{phase}].total"):
            with nvtx_range(f"dreamzero.ar.infer[{phase}].rtc_request_prepare"):
                prev_chunk_left_over_abs = None
                model_kwargs: dict[str, object] = {}
                if rtc_requested and session_state is not None:
                    prev_chunk_left_over_abs = compute_prev_chunk_left_over(session_state, rtc_step_idx)
                    model_kwargs = {
                        "use_rtc": True,
                        "prev_chunk_left_over_abs": prev_chunk_left_over_abs,
                        "inference_delay": rtc_inference_delay_steps,
                        "rtc_execution_horizon": self._rtc_execution_horizon,
                        "rtc_max_guidance_weight": self._rtc_max_guidance_weight,
                        "rtc_prefix_attention_schedule": self._rtc_prefix_attention_schedule,
                        "rtc_guidance_max_steps": self._rtc_guidance_max_steps,
                        "rtc_guidance_step_stride": self._rtc_guidance_step_stride,
                    }
                    if had_previous_chunk:
                        left_over_len = 0 if prev_chunk_left_over_abs is None else int(prev_chunk_left_over_abs.shape[0])
                        logger.info(
                            "RTC request session=%s step_idx=%d prev_left_over=%d delay=%d",
                            session_id,
                            rtc_step_idx,
                            left_over_len,
                            rtc_inference_delay_steps,
                        )
            with nvtx_range(f"dreamzero.ar.infer[{phase}].convert_observation"):
                converted_obs = self._convert_observation(obs)
                worker_obs = self._set_reset_flag(converted_obs, should_reset)

            # Signal workers to continue (0 = continue)
            signal_tensor = torch.zeros(1, dtype=torch.int32, device='cpu')
            with nvtx_range(f"dreamzero.ar.infer[{phase}].dist_broadcast.signal"):
                dist.broadcast(signal_tensor, src=0, group=self._signal_group)

            # Broadcast obs and model kwargs to workers
            self._broadcast_payload_to_workers(
                {
                    "obs": worker_obs,
                    "model_kwargs": model_kwargs,
                },
                phase=phase,
            )

            # Create batch for policy
            with nvtx_range(f"dreamzero.ar.infer[{phase}].batch_build"):
                batch = Batch(obs=converted_obs)

            # Distributed forward pass
            with nvtx_range(f"dreamzero.ar.infer[{phase}].dist_barrier.pre_forward"):
                dist.barrier()
            with nvtx_range(f"dreamzero.ar.infer[{phase}].policy_forward"):
                with torch.no_grad():
                    result_batch, video_pred = self._policy.lazy_joint_forward_causal(batch, **model_kwargs)
            with nvtx_range(f"dreamzero.ar.infer[{phase}].dist_barrier.post_forward"):
                dist.barrier()

            # Store video predictions for potential saving
            self.video_across_time.append(video_pred)

            # Extract and convert action
            with nvtx_range(f"dreamzero.ar.infer[{phase}].action_postprocess"):
                action_dict = self._extract_action_dict(result_batch.act)
                action = self._convert_action(action_dict)
                if rtc_requested and had_previous_chunk:
                    if rtc_inference_delay_steps >= action.shape[0]:
                        logger.warning(
                            "RTC inference delay %d exceeds chunk length %d; keeping the final action only.",
                            rtc_inference_delay_steps,
                            action.shape[0],
                        )
                    action, applied_delay_steps = trim_action_chunk_for_delay(
                        action,
                        rtc_inference_delay_steps,
                    )
                else:
                    applied_delay_steps = 0

                if rtc_requested and session_state is not None:
                    session_state.last_action_chunk_abs = action.copy()
                    session_state.request_count += 1
                    session_state.consumed_steps = 0
                    if rtc_step_idx is not None:
                        session_state.chunk_start_step_idx = rtc_step_idx + applied_delay_steps
                    else:
                        session_state.chunk_start_step_idx = None
                    logger.info(
                        "RTC stored executable chunk session=%s len=%d chunk_start_step_idx=%s delay_trim=%d",
                        session_id,
                        action.shape[0],
                        session_state.chunk_start_step_idx,
                        applied_delay_steps,
                    )
        
        # Update first call flag
        if self._is_first_call:
            self._is_first_call = False
        
        return action
    
    def _reset_state(self, save_video: bool = True) -> None:
        """Internal method to reset policy state.
        
        Args:
            save_video: Whether to save accumulated video before reset.
        """
        with nvtx_range("dreamzero.ar.reset.total"):
            # Optionally save accumulated video before reset
            if save_video and len(self.video_across_time) > 0 and self._output_dir:
                with nvtx_range("dreamzero.ar.reset.video_decode_save"):
                    try:
                        frame_list = []
                        video_across_time_cat = torch.cat(self.video_across_time, dim=2)
                        frames = self._policy.trained_model.action_head.vae.decode(
                            video_across_time_cat,
                            tiled=self._policy.trained_model.action_head.tiled,
                            tile_size=(self._policy.trained_model.action_head.tile_size_height, self._policy.trained_model.action_head.tile_size_width),
                            tile_stride=(self._policy.trained_model.action_head.tile_stride_height, self._policy.trained_model.action_head.tile_stride_width),
                        )
                        frames = rearrange(frames, "B C T H W -> B T H W C")
                        frames = frames[0]
                        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
                        for frame in frames:
                            frame_list.append(frame)

                        if len(frame_list) > 0:
                            sample_frame = frame_list[0]
                            if len(sample_frame.shape) == 3 and sample_frame.shape[2] in [1, 3, 4]:
                                save_dir = self._output_dir
                                os.makedirs(save_dir, exist_ok=True)
                                all_mp4_files = [f for f in os.listdir(save_dir) if f.endswith(".mp4")]
                                timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
                                num_frames = len(frame_list)
                                n = (num_frames - 1) // 8
                                output_path = os.path.join(save_dir, f'{len(all_mp4_files):06}_{timestamp}_n{n}.mp4')
                                imageio.mimsave(output_path, frame_list, fps=5, codec='libx264')
                                logger.info(f"Saved video on reset to: {output_path}")
                    except Exception as e:
                        logger.warning(f"Failed to save video on reset: {e}")

            with nvtx_range("dreamzero.ar.reset.state_clear"):
                for key in self._frame_buffers:
                    self._frame_buffers[key] = []

                self._call_count = 0
                self._is_first_call = True
                self.video_across_time = []
                self._warned_single_external_fallback = False
                self._rtc_session_states.clear()
                self._reset_model_temporal_state()
    
    def reset(self, reset_info: dict) -> None:
        """Reset the policy state for a new episode.
        
        Clears frame buffers and resets call count.
        """
        self._reset_state(save_video=True)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.
    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        output_dir: str | None = None,
        signal_group: dist.ProcessGroup | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._output_dir = output_dir
        logging.getLogger("websockets.server").setLevel(logging.INFO)
        self.video_across_time = []
        self._msg_index = 0
        self._signal_group = signal_group
        # Create output directory if specified
        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)
            os.makedirs(os.path.join(self._output_dir, "inputs"), exist_ok=True)

    def _reset_policy_temporal_state(self) -> None:
        if hasattr(self._policy, "reset_inference_state"):
            self._policy.reset_inference_state()
    
    def _save_input_obs(self, obs: dict) -> None:
        """Save incoming observation images per message.
        
        Expected format: THWC (Time, Height, Width, Channel) with 4 frames.
        Saves each frame as a separate PNG image: HWC format (uint8).
        
        Directory structure:
        output_dir/inputs/{msg_index:06d}_{timestamp}/{obs_key}/f{frame_idx:02d}.png
        """
        if not self._output_dir:
            return
        timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
        base_dir = os.path.join(self._output_dir, "inputs", f"{self._msg_index:06d}_{timestamp}")
        try:
            os.makedirs(base_dir, exist_ok=True)
        except Exception:
            return

        for key in ("video.exterior_image_1_left", "video.exterior_image_2_left", "video.wrist_image_left"):
            if key not in obs:
                continue
            value = obs[key]
            try:
                # Convert to numpy if tensor
                if isinstance(value, torch.Tensor):
                    arr = value.detach().cpu().numpy()
                else:
                    arr = np.asarray(value)
                
                # Expected format: THWC (Time, Height, Width, Channel)
                if arr.ndim != 4:
                    logger.warning(f"obs key '{key}' has shape {arr.shape}, expected 4D (T,H,W,C)")
                    continue
                
                # arr is (T, H, W, C)
                T, H, W, C = arr.shape
                
                # Normalize to uint8
                if arr.dtype == np.uint8:
                    frames_u8 = arr
                else:
                    f = arr.astype(np.float32)
                    # Common conventions: [-1,1] or [0,1]
                    min_val = float(np.nanmin(f))
                    max_val = float(np.nanmax(f))
                    if min_val >= -1.1 and max_val <= 1.1:
                        # Assume [-1,1] range
                        frames_u8 = ((f + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                    else:
                        # Min-max scaling
                        denom = (max_val - min_val) if (max_val - min_val) > 1e-6 else 1.0
                        frames_u8 = ((f - min_val) / denom * 255.0).clip(0, 255).astype(np.uint8)
                
                # Save each frame: frames_u8[i] is (H, W, C)
                key_dir = os.path.join(base_dir, key.replace("/", "_"))
                os.makedirs(key_dir, exist_ok=True)
                for frame_idx in range(T):
                    frame = frames_u8[frame_idx]  # (H, W, C)
                    # Handle grayscale (H, W) -> (H, W, 1)
                    if frame.ndim == 2:
                        frame = np.expand_dims(frame, axis=-1)
                    imageio.imwrite(os.path.join(key_dir, f"f{frame_idx:02d}.png"), frame)
                    
            except Exception as e:
                logger.warning(f"Failed to save obs key '{key}': {e}")
                continue



    def serve_forever(self, rank: int = 0) -> None:
        asyncio.run(self.run(rank))

    async def run(self, rank: int = 0):
        if rank == 0:
            async with _server.serve(
                self._handler,
                self._host,
                self._port,
                compression=None,
                max_size=None,
                process_request=_health_check,
                ping_interval=None,
            ) as server:
                await server.serve_forever()
        else:
            # Non-rank-0 processes run a worker loop
            await self._worker_loop()

    async def _worker_loop(self):
        """Worker loop for non-rank-0 processes to participate in distributed inference."""
        logger.info(f"Worker loop started for rank {dist.get_rank()}")
        signal_tensor = torch.zeros(1, dtype=torch.int32, device='cpu')
        while True:
            try:
                # Wait for obs broadcast from rank 0
                # Create a dummy obs dict structure - will be filled by broadcast
                # obs = {}

                with nvtx_range("dreamzero.ar.worker.dist_broadcast.signal"):
                    dist.broadcast(signal_tensor, src=0, group=self._signal_group)

                signal = signal_tensor.item()
                if signal == 1:
                    logger.info(f"Rank {dist.get_rank()} received shutdown signal")
                    break

                # --- ADD THIS ELIF BLOCK ---
                elif signal == 2:
                    logger.info(f"Rank {dist.get_rank()} received idle signal. Waiting for next client.")
                    # Loop back to the top and wait for the next signal
                    continue

                # Receive the batch data via broadcast/gather mechanism
                # This is a simplified version - the actual obs structure needs to be broadcasted
                with nvtx_range("dreamzero.ar.worker.receive_payload"):
                    batch, model_kwargs = self._receive_batch_from_rank0()
                reset_flag = bool(batch.obs.pop(RESET_FLAG_KEY, False))
                if reset_flag:
                    with nvtx_range("dreamzero.ar.worker.reset_state"):
                        self._reset_policy_temporal_state()
                # Participate in distributed forward pass
                with nvtx_range("dreamzero.ar.worker.dist_barrier.pre_forward"):
                    dist.barrier()
                with nvtx_range("dreamzero.ar.worker.policy_forward"):
                    with torch.no_grad():
                        result_batch, video_pred = self._policy.lazy_joint_forward_causal(batch, **model_kwargs)
                with nvtx_range("dreamzero.ar.worker.dist_barrier.post_forward"):
                    dist.barrier()

            except Exception as e:
                logger.error(f"Worker loop error on rank {dist.get_rank()}: {e}")
                traceback.print_exc()
                break

    def _receive_batch_from_rank0(self):
        """Receive batch data plus model kwargs from rank 0 using torch.distributed primitives."""
        import pickle

        # Receive the size of the pickled data first
        with nvtx_range("dreamzero.ar.worker.dist_broadcast.size"):
            size_tensor = torch.zeros(1, dtype=torch.int64, device='cuda')
            dist.broadcast(size_tensor, src=0)
            data_size = size_tensor.item()

        # Receive the actual data
        with nvtx_range("dreamzero.ar.worker.dist_broadcast.payload"):
            data_tensor = torch.zeros(data_size, dtype=torch.uint8, device='cuda')
            dist.broadcast(data_tensor, src=0)

        # Deserialize
        with nvtx_range("dreamzero.ar.worker.unpickle_payload"):
            payload = pickle.loads(data_tensor.cpu().numpy().tobytes())
        if isinstance(payload, dict) and "obs" in payload:
            obs = payload["obs"]
            model_kwargs = payload.get("model_kwargs", {})
        else:
            obs = payload
            model_kwargs = {}
        return Batch(obs=obs), model_kwargs

    def _broadcast_batch_to_workers(self, payload):
        """Broadcast batch data from rank 0 to all other ranks."""
        import pickle

        serialized = pickle.dumps(payload)
        data_size = len(serialized)

        # Broadcast size first
        size_tensor = torch.tensor([data_size], dtype=torch.int64, device='cuda')
        dist.broadcast(size_tensor, src=0)

        # Broadcast data
        data_tensor = torch.frombuffer(serialized, dtype=torch.uint8).cuda()
        dist.broadcast(data_tensor, src=0)

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        signal_tensor = torch.zeros(1, dtype=torch.int32, device='cpu')
        
        try:
            while True:
                try:
                    start_time = time.perf_counter()
                    data = await websocket.recv()
                    recv_done = time.perf_counter()
                    obs = msgpack_numpy.unpackb(data)
                    print(f"Wait Time: {recv_done - start_time:.2f} seconds")
                    self._msg_index += 1

                    infer_start_time = time.perf_counter()

                    # Signal other ranks to continue (0 = continue)
                    signal_tensor.zero_() 
                    dist.broadcast(signal_tensor, src=0, group=self._signal_group) # <-- USE GLOO GROUP

                    # Broadcast the obs to all ranks for distributed inference
                    self._broadcast_batch_to_workers(obs)
                    batch = Batch(obs=obs)

                    # All ranks need to participate in the forward pass
                    dist.barrier()
                    forward_start_time = time.perf_counter()
                    with torch.no_grad():
                        result_batch, video_pred = self._policy.lazy_joint_forward_causal(batch)
                    dist.barrier()
                    print(f"Forward Time: {time.perf_counter() - forward_start_time:.2f} seconds")

                    action_chunk_dict = result_batch.act
                    video_chunk = video_pred

                    print(f"Inference Time: {time.perf_counter() - infer_start_time:.2f} seconds")

                    self.video_across_time.append(video_chunk)

                    if len(self.video_across_time) > 10:
                        frame_list = []
                        video_across_time_cat = torch.cat(self.video_across_time, dim=2)
                        frames = self._policy.trained_model.action_head.vae.decode(
                            video_across_time_cat,
                            tiled=self._policy.trained_model.action_head.tiled,
                            tile_size=(self._policy.trained_model.action_head.tile_size_height, self._policy.trained_model.action_head.tile_size_width),
                            tile_stride=(self._policy.trained_model.action_head.tile_stride_height, self._policy.trained_model.action_head.tile_stride_width),
                        )
                        frames = rearrange(frames, "B C T H W -> B T H W C")
                        frames = frames[0]
                        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
                        # Add each frame individually to the list
                        for frame in frames:
                            frame_list.append(frame)

                        sample_frame = frame_list[0]
                        if len(sample_frame.shape) == 3 and sample_frame.shape[2] in [1, 3, 4]:
                            # Save all frames as a single MP4 file
                            save_dir = self._output_dir if self._output_dir else "."
                            os.makedirs(save_dir, exist_ok=True)
                            all_mp4_files = [f for f in os.listdir(save_dir) if f.endswith(".mp4")]
                            timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
                            num_frames = len(frame_list)
                            n = (num_frames - 1) // 8  # num_frames = 8n+1, so n = (num_frames-1)/8
                            output_path = os.path.join(save_dir, f'{len(all_mp4_files):06}_{timestamp}_n{n}.mp4')
                            imageio.mimsave(output_path, frame_list, fps=5, codec='libx264')
                            print(f"Saved video to: {output_path}")
                        else:
                            print(f"Warning: Invalid frame shape {sample_frame.shape}. Expected (H, W, C) with C in [1, 3, 4]. Skipping video save.")

                        self.video_across_time = []
                    elif self._policy.trained_model.action_head.current_start_frame == 1 + self._policy.trained_model.action_head.num_frame_per_block and len(self.video_across_time) > 1:
                        print("current_start_frame == 1 + num_frame_per_block and len(self.video_across_time) > 1")
                        frame_list = []
                        video_across_time_cat = torch.cat(self.video_across_time[:-1], dim=2)
                        frames = self._policy.trained_model.action_head.vae.decode(
                            video_across_time_cat,
                            tiled=self._policy.trained_model.action_head.tiled,
                            tile_size=(self._policy.trained_model.action_head.tile_size_height, self._policy.trained_model.action_head.tile_size_width),
                            tile_stride=(self._policy.trained_model.action_head.tile_stride_height, self._policy.trained_model.action_head.tile_stride_width),
                        )
                        frames = rearrange(frames, "B C T H W -> B T H W C")
                        frames = frames[0]
                        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
                        # Add each frame individually to the list
                        for frame in frames:
                            frame_list.append(frame)
                        sample_frame = frame_list[0]
                        if len(sample_frame.shape) == 3 and sample_frame.shape[2] in [1, 3, 4]:
                            # Save all frames as a single MP4 file
                            save_dir = self._output_dir if self._output_dir else "."
                            os.makedirs(save_dir, exist_ok=True)
                            all_mp4_files = [f for f in os.listdir(save_dir) if f.endswith(".mp4")]
                            timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
                            num_frames = len(frame_list)
                            n = (num_frames - 1) // 8  # num_frames = 8n+1, so n = (num_frames-1)/8
                            output_path = os.path.join(save_dir, f'{len(all_mp4_files):06}_{timestamp}_n{n}.mp4')
                            imageio.mimsave(output_path, frame_list, fps=5, codec='libx264')
                            print(f"Saved video to: {output_path}")
                        self.video_across_time = [video_chunk]

                    
                    def batch_to_dict(batch):
                        out = {}
                        for k in dir(batch):
                            if not k.startswith("action."):
                                continue
                            out[k] = getattr(batch, k)
                        return out
                    action_chunk_dict = batch_to_dict(action_chunk_dict)
                    await websocket.send(packer.pack(action_chunk_dict))

                except websockets.ConnectionClosed:
                    logger.info(f"Connection from {websocket.remote_address} closed")
                    if len(self.video_across_time) > 0:
                        frame_list = []
                        video_across_time_cat = torch.cat(self.video_across_time, dim=2)
                        frames = self._policy.trained_model.action_head.vae.decode(
                            video_across_time_cat,
                            tiled=self._policy.trained_model.action_head.tiled,
                            tile_size=(self._policy.trained_model.action_head.tile_size_height, self._policy.trained_model.action_head.tile_size_width),
                            tile_stride=(self._policy.trained_model.action_head.tile_stride_height, self._policy.trained_model.action_head.tile_stride_width),
                        )
                        frames = rearrange(frames, "B C T H W -> B T H W C")
                        frames = frames[0]
                        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
                        # Add each frame individually to the list
                        for frame in frames:
                            frame_list.append(frame)

                        sample_frame = frame_list[0]
                        if len(sample_frame.shape) == 3 and sample_frame.shape[2] in [1, 3, 4]:
                            # Save all frames as a single MP4 file
                            save_dir = self._output_dir if self._output_dir else "."
                            os.makedirs(save_dir, exist_ok=True)
                            all_mp4_files = [f for f in os.listdir(save_dir) if f.endswith(".mp4")]
                            timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
                            num_frames = len(frame_list)
                            n = (num_frames - 1) // 8  # num_frames = 8n+1, so n = (num_frames-1)/8
                            output_path = os.path.join(save_dir, f'{len(all_mp4_files):06}_{timestamp}_n{n}.mp4')
                            imageio.mimsave(output_path, frame_list, fps=5, codec='libx264')
                            print(f"Saved video to: {output_path}")
                        else:
                            print(f"Warning: Invalid frame shape {sample_frame.shape}. Expected (H, W, C) with C in [1, 3, 4]. Skipping video save.")

                    self.video_across_time = []
                    break
                except Exception:
                    await websocket.send(traceback.format_exc())
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="Internal server error. Traceback included in previous frame.",
                    )
                    raise
        finally:
            logger.info(f"Rank 0: Client session ended. Sending idle signal (2) to workers.")
            signal_tensor.fill_(2)  # Set tensor value to 2
            dist.broadcast(signal_tensor, src=0, group=self._signal_group)
            # When connection closes, signal other ranks to continue waiting for next connection
            # (or implement proper shutdown if needed)


def init_mesh() -> DeviceMesh:
    # env vars set by torchrun
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"Rank {rank}/{world_size} (PID: {os.getpid()}) setting device to {rank}")

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    mesh = init_device_mesh(
        device_type="cuda",
        mesh_shape=(world_size, ),
        mesh_dim_names=("ip", ),
    )
    print(f"Rank {rank}/{world_size} (PID: {os.getpid()}) using device {device}")

    return mesh

def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None


def main(args: Args) -> None:
    # Set environment variable for DIT cache.
    os.environ["ENABLE_DIT_CACHE"] = "true" if args.enable_dit_cache else "false"

    # Use TE cuDNN backend for attention.
    os.environ["ATTENTION_BACKEND"] = "TE"

    # Increase the recompile limit to 100 for inference due
    # to autoregressive nature of the model (several possible shapes).
    torch._dynamo.config.recompile_limit = 800

    embodiment_tag = "oxe_droid"
    model_path = args.model_path
    load_model_path, effective_architecture, _architecture_override_tmp = _resolve_model_path_for_architecture(args)
    policy_metadata = {
        "embodiment": embodiment_tag,
        "model_name": "dreamzero",
        "model_path": model_path,
        "architecture": effective_architecture,
    }

    with nvtx_range("dreamzero.ar.init.mesh"):
        device_mesh = init_mesh()
    rank = dist.get_rank()

    timeout_delta = datetime.timedelta(seconds=args.timeout_seconds)
    with nvtx_range("dreamzero.ar.init.signal_group"):
        signal_group = dist.new_group(backend="gloo", timeout=timeout_delta)
    logger.info(f"Rank {rank} initialized signal_group (gloo)")

    with nvtx_range("dreamzero.ar.init.policy_load"):
        policy = GrootSimPolicy(
            embodiment_tag=EmbodimentTag(embodiment_tag),
            model_path=load_model_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
            device_mesh=device_mesh,
        )
    loaded_architecture = getattr(getattr(policy.trained_model, "action_head", None), "config", None)
    loaded_architecture = getattr(loaded_architecture, "architecture", None)
    if loaded_architecture in ("joint", "mot") and loaded_architecture != effective_architecture:
        raise RuntimeError(
            f"Loaded WAM architecture {loaded_architecture!r}, expected {effective_architecture!r}."
        )
    logger.info("DreamZero WAM architecture active: %s", loaded_architecture or effective_architecture)
    image_height, image_width = _get_expected_video_resolution(policy)
    logger.info("Using checkpoint image resolution %dx%d", image_height, image_width)

    # Create server for all ranks - rank 0 handles websocket, others run worker loop
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    if rank == 0:
        logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)
        # Create output directory for videos
        # Extract parent directory and checkpoint name from model_path
        parent_dir = os.path.dirname(model_path)
        date_suffix = datetime.datetime.now().strftime("%Y%m%d")
        checkpoint_name = os.path.basename(model_path)
        output_dir = os.path.join(parent_dir, f"real_world_eval_gen_{date_suffix}_{args.index}", checkpoint_name)
        os.makedirs(output_dir, exist_ok=True)
        logging.info("Videos will be saved to: %s", output_dir)
    else:
        output_dir = None
        logging.info(f"Rank {rank} starting as worker for distributed inference...")
    
    # Create wrapper policy that converts between roboarena and AR_droid formats
    with nvtx_range("dreamzero.ar.init.wrapper_policy"):
        wrapper_policy = ARDroidRoboarenaPolicy(
            groot_policy=policy,
            signal_group=signal_group,
            image_height=image_height,
            image_width=image_width,
            output_dir=output_dir,
            max_chunk_size=args.max_chunk_size,
            use_rtc=args.use_rtc,
            rtc_execution_horizon=args.rtc_execution_horizon,
            rtc_max_guidance_weight=args.rtc_max_guidance_weight,
            rtc_prefix_attention_schedule=args.rtc_prefix_attention_schedule,
            rtc_guidance_max_steps=args.rtc_guidance_max_steps,
            rtc_guidance_step_stride=args.rtc_guidance_step_stride,
        )
    
    # Configure server for AR_droid (2 external cameras, wrist camera, joint position actions)
    server_config = PolicyServerConfig(
        image_resolution=(image_height, image_width),
        needs_wrist_camera=True,
        n_external_cameras=2,
        needs_stereo_camera=False,
        needs_session_id=True,  # Track session to reset state for new clients
        action_space="joint_position",
    )
    
    if rank == 0:
        logging.info("Using roboarena policy server interface")
        logging.info(f"Server config: {server_config}")
        logging.info("Serving DreamZero AR %s websocket server on ws://%s:%d", effective_architecture, args.host, args.port)
        roboarena_server = RoboarenaServer(
            policy=wrapper_policy,
            server_config=server_config,
            host=args.host,
            port=args.port,
            open_timeout=args.handshake_timeout_seconds,
        )
        roboarena_server.serve_forever()
    else:
        # Non-rank-0 processes need to run worker loop for distributed inference
        # We'll use the existing WebsocketPolicyServer's worker loop mechanism
        server = WebsocketPolicyServer(
            policy=policy,
            host=args.host,
            port=args.port,
            metadata=policy_metadata,
            output_dir=output_dir,
            signal_group=signal_group,
        )
        asyncio.run(server._worker_loop())
    


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    if DEFAULT_TORCH_COMPILE_BACKEND is not None:
        logger.info("Using torch.compile backend=%s for this entrypoint", DEFAULT_TORCH_COMPILE_BACKEND)
    args = tyro.cli(Args)
    main(args)
