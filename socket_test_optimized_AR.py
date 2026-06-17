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
from typing import Any, Literal

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
SESSION_ID_KEY = "__dreamzero_session_id__"
RESET_ALL_SESSIONS_KEY = "__reset_all_dreamzero_sessions__"
WAN_JOINT_MODEL_TARGET = "groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk.CausalWanModel"
WAN_MOT_MODEL_TARGET = "groot.vla.model.dreamzero.modules.dreamzero_mot.MoTCausalWanModel"

logger = logging.getLogger(__name__)


def _prefix_latent_frames_to_pixel_frames(prefix_latent_frames: int) -> int:
    if prefix_latent_frames <= 0:
        return 0
    return 1 + 4 * (prefix_latent_frames - 1)


def _decode_video_pred_chunks(action_head, chunks: list[tuple[torch.Tensor, int]]) -> list[np.ndarray]:
    frame_list: list[np.ndarray] = []
    for latents, prefix_latent_frames in chunks:
        with torch.no_grad():
            frames = action_head.vae.decode(
                latents,
                tiled=action_head.tiled,
                tile_size=(action_head.tile_size_height, action_head.tile_size_width),
                tile_stride=(action_head.tile_stride_height, action_head.tile_stride_width),
            )
        frames = rearrange(frames, "B C T H W -> B T H W C")[0]
        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        output_start = min(_prefix_latent_frames_to_pixel_frames(prefix_latent_frames), len(frames))
        frame_list.extend(list(frames[output_start:]))
    return frame_list


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
    save_input_vae_videos: bool = True
    max_saved_input_vae_videos: int = 0  # <= 0 means unlimited.
    input_vae_video_fps: int = 5
    input_vae_video_dir: str | None = None
    use_rtc: bool = False
    rtc_execution_horizon: int = 10
    rtc_max_guidance_weight: float = 10.0
    rtc_prefix_attention_schedule: str = "EXP"
    rtc_guidance_max_steps: int = 4
    rtc_guidance_step_stride: int = 1
    batch_max_size: int = 8
    batch_timeout_ms: float = 2.0


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


def _dreamzero_cache_mode(policy: GrootSimPolicy) -> tuple[str | None, bool]:
    action_head = getattr(getattr(policy, "trained_model", None), "action_head", None)
    if action_head is None:
        return None, False
    model = getattr(action_head, "model", None)
    if not getattr(model, "is_mot_wam", False):
        return None, False
    resolver = getattr(action_head, "_effective_mot_inference_video_mode", None)
    if callable(resolver):
        mode = str(resolver(use_rtc=False))
    else:
        mode = str(getattr(getattr(action_head, "config", None), "mot_inference_video_mode", "auto"))
    return mode, mode == "decoupled_denoise"


ACTION_HEAD_TEMPORAL_ATTRS = (
    "current_start_frame",
    "language",
    "clip_feas",
    "ys",
    "kv_cache1",
    "kv_cache_neg",
    "crossattn_cache",
    "crossattn_cache_neg",
    "skip_countdown",
    "last_video_pred_condition_latent_frames",
)
ACTION_HEAD_BATCH_DIM0_ATTRS = {"language", "clip_feas", "ys"}
ACTION_HEAD_BATCH_DIM1_ATTRS = {
    "kv_cache1",
    "kv_cache_neg",
    "crossattn_cache",
    "crossattn_cache_neg",
}


def _empty_frame_buffers() -> dict[str, list[np.ndarray]]:
    return {
        "video.exterior_image_1_left": [],
        "video.exterior_image_2_left": [],
        "video.wrist_image_left": [],
    }


def _get_action_head(policy: Any) -> Any | None:
    trained_model = getattr(policy, "trained_model", None)
    if trained_model is None:
        return None
    return getattr(trained_model, "action_head", None)


@dataclasses.dataclass
class ActionHeadTemporalState:
    values: dict[str, Any]


class BatchIncompatibleError(ValueError):
    """Raised before forward when a prepared request group cannot share one batch."""


def _temporal_scalar_signature(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return ("np", str(value.dtype), tuple(value.shape), value.tobytes())
    if torch.is_tensor(value):
        return ("torch-scalar", str(value.dtype), tuple(value.shape), value.detach().cpu().numpy().tobytes())
    return value


def _tensor_signature_without_dim(value: torch.Tensor, batch_dim: int) -> tuple:
    shape = list(value.shape)
    if len(shape) <= batch_dim:
        raise ValueError(f"Cannot remove batch dim {batch_dim} from tensor shape {tuple(value.shape)}.")
    del shape[batch_dim]
    return (str(value.dtype), str(value.device), tuple(shape))


def _temporal_attr_signature(attr: str, value: Any) -> Any:
    if value is None:
        return None
    if attr in ACTION_HEAD_BATCH_DIM0_ATTRS:
        if not torch.is_tensor(value):
            raise ValueError(f"Expected tensor temporal attr {attr}, got {type(value)!r}.")
        return _tensor_signature_without_dim(value, 0)
    if attr in ACTION_HEAD_BATCH_DIM1_ATTRS:
        if not isinstance(value, list):
            raise ValueError(f"Expected list temporal attr {attr}, got {type(value)!r}.")
        return tuple(_tensor_signature_without_dim(tensor, 1) for tensor in value)
    return _temporal_scalar_signature(value)


def _pack_temporal_attr(attr: str, values: list[Any]) -> Any:
    if attr in ACTION_HEAD_BATCH_DIM0_ATTRS:
        return torch.cat(values, dim=0)
    if attr in ACTION_HEAD_BATCH_DIM1_ATTRS:
        num_layers = len(values[0])
        return [
            torch.cat([value[layer_idx] for value in values], dim=1)
            for layer_idx in range(num_layers)
        ]
    return values[0]


def _split_temporal_attr(attr: str, value: Any, batch_size: int) -> list[Any]:
    if value is None:
        return [None for _ in range(batch_size)]
    if attr in ACTION_HEAD_BATCH_DIM0_ATTRS:
        return [value[index : index + 1] for index in range(batch_size)]
    if attr in ACTION_HEAD_BATCH_DIM1_ATTRS:
        return [
            [layer[:, index : index + 1] for layer in value]
            for index in range(batch_size)
        ]
    return [value for _ in range(batch_size)]


class ActionHeadSessionStore:
    """Per-session holder for DreamZero action-head temporal state."""

    def __init__(self, policy: GrootSimPolicy) -> None:
        self._policy = policy
        self._states: dict[str, ActionHeadTemporalState] = {}
        self._loaded_key: str | None = None

    def reset_loaded_state(self) -> None:
        if hasattr(self._policy, "reset_inference_state"):
            self._policy.reset_inference_state()
            return
        action_head = _get_action_head(self._policy)
        if action_head is None:
            return
        if hasattr(action_head, "reset_inference_state"):
            action_head.reset_inference_state()
            return
        if hasattr(action_head, "current_start_frame"):
            action_head.current_start_frame = 0
        if hasattr(action_head, "language"):
            action_head.language = None

    def restore(self, session_id: str | None, *, reset: bool = False) -> str:
        key = session_key(session_id)
        if reset:
            self._states.pop(key, None)

        state = self._states.get(key)
        if state is None:
            self.reset_loaded_state()
        else:
            action_head = _get_action_head(self._policy)
            if action_head is not None:
                for attr, value in state.values.items():
                    setattr(action_head, attr, value)
        self._loaded_key = key
        return key

    def state_signature(self, session_id: str | None, *, reset: bool = False) -> tuple:
        key = session_key(session_id)
        if reset or key not in self._states:
            return ("empty",)

        state = self._states[key]
        signature = []
        for attr in ACTION_HEAD_TEMPORAL_ATTRS:
            if attr not in state.values:
                signature.append((attr, "missing"))
            else:
                signature.append((attr, _temporal_attr_signature(attr, state.values[attr])))
        return tuple(signature)

    def restore_many(self, session_ids: list[str | None], *, resets: list[bool]) -> list[str]:
        if len(session_ids) != len(resets):
            raise ValueError("session_ids and resets must have the same length.")
        if len(session_ids) == 1:
            return [self.restore(session_ids[0], reset=resets[0])]

        keys = [session_key(session_id) for session_id in session_ids]
        for key, reset in zip(keys, resets, strict=True):
            if reset:
                self._states.pop(key, None)

        states = [self._states.get(key) for key in keys]
        if all(state is None for state in states):
            self.reset_loaded_state()
            self._loaded_key = "__batch__"
            return keys
        if any(state is None for state in states):
            raise BatchIncompatibleError("Cannot batch new and cached DreamZero sessions in one action-head state.")

        first_signature = self.state_signature(keys[0])
        for key in keys[1:]:
            signature = self.state_signature(key)
            if signature != first_signature:
                raise BatchIncompatibleError("Cannot batch DreamZero sessions with incompatible action-head state.")

        packed_values: dict[str, Any] = {}
        for attr in ACTION_HEAD_TEMPORAL_ATTRS:
            attr_values = [state.values.get(attr) for state in states if state is not None]
            if all(value is None for value in attr_values):
                continue
            if any(value is None for value in attr_values):
                raise BatchIncompatibleError(f"Cannot batch temporal attr {attr} with mixed missing values.")
            packed_values[attr] = _pack_temporal_attr(attr, attr_values)

        action_head = _get_action_head(self._policy)
        if action_head is not None:
            for attr, value in packed_values.items():
                setattr(action_head, attr, value)
        self._loaded_key = "__batch__"
        return keys

    def capture(self, session_id: str | None) -> None:
        key = session_key(session_id)
        action_head = _get_action_head(self._policy)
        if action_head is None:
            self._states[key] = ActionHeadTemporalState({})
        else:
            self._states[key] = ActionHeadTemporalState(
                {
                    attr: getattr(action_head, attr)
                    for attr in ACTION_HEAD_TEMPORAL_ATTRS
                    if hasattr(action_head, attr)
                }
            )
        self._loaded_key = key

    def capture_many(self, session_ids: list[str | None]) -> None:
        if len(session_ids) == 1:
            self.capture(session_ids[0])
            return

        action_head = _get_action_head(self._policy)
        keys = [session_key(session_id) for session_id in session_ids]
        batch_size = len(keys)
        if action_head is None:
            for key in keys:
                self._states[key] = ActionHeadTemporalState({})
            self._loaded_key = "__batch__"
            return

        per_session_values = [dict() for _ in keys]
        for attr in ACTION_HEAD_TEMPORAL_ATTRS:
            if not hasattr(action_head, attr):
                continue
            value = getattr(action_head, attr)
            values = _split_temporal_attr(attr, value, batch_size)
            for index, item_value in enumerate(values):
                per_session_values[index][attr] = item_value

        for key, values in zip(keys, per_session_values, strict=True):
            self._states[key] = ActionHeadTemporalState(values)
        self._loaded_key = "__batch__"

    def reset_session(self, session_id: str | None) -> None:
        key = session_key(session_id)
        self._states.pop(key, None)
        if self._loaded_key == key:
            self.reset_loaded_state()
            self._loaded_key = None

    def reset_all(self) -> None:
        self._states.clear()
        self.reset_loaded_state()
        self._loaded_key = None


@dataclasses.dataclass
class ARSessionState:
    frame_buffers: dict[str, list[np.ndarray]] = dataclasses.field(default_factory=_empty_frame_buffers)
    call_count: int = 0
    is_first_call: bool = True
    current_prompt: str | None = None
    warned_single_external_fallback: bool = False
    video_across_time: list[tuple[torch.Tensor, int]] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class PreparedARInference:
    index: int
    obs: dict
    session_id: str | None
    session_key: str
    converted_obs: dict
    model_kwargs: dict[str, object]
    reset_session_for_workers: bool
    reset_all_sessions_for_workers: bool
    rtc_requested: bool
    rtc_step_idx: int | None
    rtc_inference_delay_steps: int
    session_state: RTCSessionState | None
    had_previous_chunk: bool
    phase: str
    wrapper_state: ARSessionState


class ARDroidRoboarenaPolicy:
    """Wrapper policy that implements roboarena.policy.BasePolicy interface for AR_droid.
    
    Handles:
    - Observation format conversion (roboarena -> AR_droid format)
    - Frame accumulation across calls (roboarena sends single frames, AR_droid expects multi-frame video)
    - Action format conversion (AR_droid dict -> roboarena array format)
    - Distributed inference coordination
    """
    
    # DROID/Wan2.2 training samples each raw-video block as
    # [anchor, anchor+3, ..., anchor+24]. Wan VAE38 maps 9 raw frames to
    # 3 latent frames, so after the anchor latent this yields exactly
    # num_frame_per_block=2 latent frames for the causal block.
    RAW_FRAMES_PER_BLOCK = 8
    FRAMES_PER_CHUNK = RAW_FRAMES_PER_BLOCK + 1
    
    def __init__(
        self,
        groot_policy: GrootSimPolicy,
        signal_group: dist.ProcessGroup,
        image_height: int,
        image_width: int,
        output_dir: str | None = None,
        max_chunk_size: int | None = None,
        save_input_vae_videos: bool = True,
        max_saved_input_vae_videos: int = 0,
        input_vae_video_fps: int = 5,
        input_vae_video_dir: str | None = None,
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
        self._save_input_vae_videos = save_input_vae_videos
        self._max_saved_input_vae_videos = max_saved_input_vae_videos
        self._input_vae_video_fps = input_vae_video_fps
        self._input_vae_video_dir = input_vae_video_dir
        self._saved_input_vae_video_count = 0
        self._last_input_num_frames = 0
        self._use_rtc = use_rtc
        self._mot_inference_video_mode, self._cache_order_sensitive = _dreamzero_cache_mode(groot_policy)
        if self._cache_order_sensitive and self._use_rtc:
            logger.warning(
                "Disabling server-side RTC for cache-order-sensitive DreamZero mode %s.",
                self._mot_inference_video_mode,
            )
            self._use_rtc = False
        self._rtc_execution_horizon = rtc_execution_horizon
        self._rtc_max_guidance_weight = rtc_max_guidance_weight
        self._rtc_prefix_attention_schedule = rtc_prefix_attention_schedule
        self._rtc_guidance_max_steps = rtc_guidance_max_steps
        self._rtc_guidance_step_stride = rtc_guidance_step_stride
        
        # Frame buffers for accumulation (per camera view). These attributes
        # point at the currently loaded session state.
        self._frame_buffers: dict[str, list[np.ndarray]] = _empty_frame_buffers()
        self._call_count = 0
        self._is_first_call = True
        
        # Session tracking. RoboLab parallel eval interleaves env sessions, so
        # both the wrapper state and action-head temporal state are keyed by
        # session_id instead of being reset on every session switch.
        self._session_states: dict[str, ARSessionState] = {}
        self._model_session_store = ActionHeadSessionStore(self._policy)
        self._active_session_key: str | None = None
        self._current_session_id: str | None = None
        self._pending_worker_reset_all = False
        self._pending_worker_reset_sessions: set[str] = set()
        self._current_prompt: str | None = None
        self._warned_single_external_fallback = False
        self._rtc_session_states: dict[str, RTCSessionState] = {}
        
        # Video across time for saving (similar to original server)
        self.video_across_time = []
        self._msg_index = 0
        
        # Create output directory if specified
        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)
        if self._save_input_vae_videos:
            if self._input_vae_video_dir is None and self._output_dir:
                self._input_vae_video_dir = os.path.join(self._output_dir, "input_vae_roundtrip")
            if self._input_vae_video_dir:
                os.makedirs(self._input_vae_video_dir, exist_ok=True)

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

    def _make_input_vae_debug_spec(self, prompt: str) -> dict | None:
        if not self._save_input_vae_videos or not self._input_vae_video_dir:
            return None
        if self._max_saved_input_vae_videos > 0 and (
            self._saved_input_vae_video_count >= self._max_saved_input_vae_videos
        ):
            return None

        save_index = self._saved_input_vae_video_count
        self._saved_input_vae_video_count += 1
        timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S_%f")
        save_dir = os.path.join(
            self._input_vae_video_dir,
            f"{save_index:06d}_msg_{self._msg_index:06d}_{timestamp}_t{self._last_input_num_frames}",
        )

        action_head = getattr(getattr(self._policy, "trained_model", None), "action_head", None)
        model = getattr(action_head, "model", None)
        return {
            "save_dir": save_dir,
            "filename": "model_input_vae_roundtrip.mp4",
            "fps": self._input_vae_video_fps,
            "meta": {
                "save_index": save_index,
                "msg_index": self._msg_index,
                "call_count": self._call_count,
                "num_server_input_frames": self._last_input_num_frames,
                "prompt": prompt,
                "is_first_call": self._is_first_call,
                "current_start_frame_before_forward": getattr(action_head, "current_start_frame", None),
                "local_attn_size": getattr(model, "local_attn_size", None),
                "source": (
                    "post-transform action_input['images'] VAE roundtrip, followed by "
                    "constructed static-init latents decoded through the VAE"
                ),
            },
        }

    def _append_frames_to_buffer(self, droid_key: str, data: np.ndarray) -> None:
        if data.ndim == 4:
            self._frame_buffers[droid_key].extend(list(data))
        else:
            self._frame_buffers[droid_key].append(data)
        if len(self._frame_buffers[droid_key]) > self.FRAMES_PER_CHUNK:
            del self._frame_buffers[droid_key][:-self.FRAMES_PER_CHUNK]

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
        self._model_session_store.reset_loaded_state()

    def _apply_ar_session_state(self, state: ARSessionState) -> None:
        self._frame_buffers = state.frame_buffers
        self._call_count = state.call_count
        self._is_first_call = state.is_first_call
        self._current_prompt = state.current_prompt
        self._warned_single_external_fallback = state.warned_single_external_fallback
        self.video_across_time = state.video_across_time

    def _load_ar_session(self, session_id: str | None, *, reset: bool = False) -> str:
        key = session_key(session_id)
        if reset:
            self._session_states.pop(key, None)
            self._rtc_session_states.pop(key, None)

        state = self._session_states.setdefault(key, ARSessionState())
        self._apply_ar_session_state(state)
        self._model_session_store.restore(session_id, reset=reset)
        self._active_session_key = key
        self._current_session_id = None if session_id is None else str(session_id)
        return key

    def _capture_ar_session(self, session_id: str | None) -> None:
        key = session_key(session_id)
        self._session_states[key] = ARSessionState(
            frame_buffers=self._frame_buffers,
            call_count=self._call_count,
            is_first_call=self._is_first_call,
            current_prompt=self._current_prompt,
            warned_single_external_fallback=self._warned_single_external_fallback,
            video_across_time=self.video_across_time,
        )
        self._model_session_store.capture(session_id)
        self._active_session_key = key

    def _capture_ar_session_wrapper(self, session_id: str | None) -> None:
        key = session_key(session_id)
        self._session_states[key] = ARSessionState(
            frame_buffers=self._frame_buffers,
            call_count=self._call_count,
            is_first_call=self._is_first_call,
            current_prompt=self._current_prompt,
            warned_single_external_fallback=self._warned_single_external_fallback,
            video_across_time=self.video_across_time,
        )
        self._active_session_key = key

    def _clear_ar_session(self, session_id: str | None, *, save_video: bool) -> None:
        key = session_key(session_id)
        state = self._session_states.get(key)
        if state is not None:
            self._apply_ar_session_state(state)
            self._active_session_key = key
            self._current_session_id = None if session_id is None else str(session_id)
            self._reset_state(save_video=save_video)
        self._session_states.pop(key, None)
        self._rtc_session_states.pop(key, None)
        self._model_session_store.reset_session(session_id)

    def _clear_all_ar_sessions(self, *, save_video: bool) -> None:
        for key, state in list(self._session_states.items()):
            self._apply_ar_session_state(state)
            self._active_session_key = key
            self._current_session_id = None if key == "__default__" else key
            self._reset_state(save_video=save_video)

        self._session_states.clear()
        self._rtc_session_states.clear()
        self._model_session_store.reset_all()
        self._active_session_key = None
        self._current_session_id = None
        self._frame_buffers = _empty_frame_buffers()
        self._call_count = 0
        self._is_first_call = True
        self._current_prompt = None
        self._warned_single_external_fallback = False
        self.video_across_time = []

    def _should_send_single_anchor_frame(self, prompt: str) -> bool:
        if self._is_first_call:
            return True
        if self._current_prompt is not None and prompt != self._current_prompt:
            return True
        return False
    
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
        prompt = obs.get("prompt", "")
        if isinstance(prompt, np.ndarray):
            prompt = prompt.item() if prompt.size == 1 else prompt.reshape(-1)[0]
        prompt = str(prompt)
        
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

        # Determine how many frames to use. True episode starts and language
        # changes need a single anchor frame. Local-attention rollover keeps the
        # last observed DROID block so the action head can rebase as
        # [observed previous block -> predicted current block].
        if self._should_send_single_anchor_frame(prompt):
            num_frames = 1
        else:
            # Normal causal block: boundary/anchor frame plus 8 newly sampled
            # raw frames, matching DROID training's [0,3,...,24] block.
            num_frames = self.FRAMES_PER_CHUNK
        self._last_input_num_frames = num_frames
        
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
        converted["annotation.language.action_text"] = prompt
        self._current_prompt = prompt
        
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

    def _set_reset_flag(
        self,
        converted_obs: dict,
        should_reset: bool,
        *,
        session_id: str | None,
        reset_all_sessions: bool = False,
    ) -> dict:
        worker_obs = dict(converted_obs)
        worker_obs[RESET_FLAG_KEY] = bool(should_reset)
        worker_obs[SESSION_ID_KEY] = session_key(session_id)
        worker_obs[RESET_ALL_SESSIONS_KEY] = bool(reset_all_sessions)
        return worker_obs

    def _set_batch_reset_flags(
        self,
        converted_obs: dict,
        reset_flags: list[bool],
        *,
        session_ids: list[str | None],
        reset_all_sessions: bool = False,
    ) -> dict:
        if len(reset_flags) != len(session_ids):
            raise ValueError("reset_flags and session_ids must have the same length.")
        if len(session_ids) == 1:
            return self._set_reset_flag(
                converted_obs,
                reset_flags[0],
                session_id=session_ids[0],
                reset_all_sessions=reset_all_sessions,
            )

        worker_obs = dict(converted_obs)
        worker_obs[RESET_FLAG_KEY] = [bool(flag) for flag in reset_flags]
        worker_obs[SESSION_ID_KEY] = [session_key(session_id) for session_id in session_ids]
        worker_obs[RESET_ALL_SESSIONS_KEY] = bool(reset_all_sessions)
        return worker_obs

    @staticmethod
    def _pack_converted_observations(converted_obs_list: list[dict]) -> dict:
        if len(converted_obs_list) == 1:
            return converted_obs_list[0]

        keys = converted_obs_list[0].keys()
        packed: dict[str, Any] = {}
        for obs in converted_obs_list[1:]:
            if obs.keys() != keys:
                raise BatchIncompatibleError("Cannot batch observations with different converted keys.")

        for key in keys:
            values = [obs[key] for obs in converted_obs_list]
            first = values[0]
            if isinstance(first, np.ndarray):
                packed[key] = np.stack(values, axis=0)
            elif torch.is_tensor(first):
                packed[key] = torch.stack(values, dim=0)
            elif isinstance(first, str):
                packed[key] = np.asarray(values)
            else:
                packed[key] = np.asarray(values)
        return packed

    @staticmethod
    def _converted_video_frame_count(converted_obs: dict) -> int:
        for key, value in converted_obs.items():
            if key.startswith("video."):
                return int(value.shape[0])
        return 0

    @staticmethod
    def _slice_action_dict(
        action_dict: dict[str, np.ndarray | torch.Tensor],
        batch_index: int,
        batch_size: int,
    ) -> dict[str, np.ndarray | torch.Tensor]:
        if batch_size == 1:
            return action_dict

        sliced: dict[str, np.ndarray | torch.Tensor] = {}
        for key, value in action_dict.items():
            if torch.is_tensor(value):
                if value.ndim == 0 or value.shape[0] != batch_size:
                    sliced[key] = value
                else:
                    sliced[key] = value[batch_index]
            else:
                arr = np.asarray(value)
                if arr.ndim == 0 or arr.shape[0] != batch_size:
                    sliced[key] = arr
                else:
                    sliced[key] = arr[batch_index]
        return sliced

    @staticmethod
    def _snapshot_wrapper_state(
        frame_buffers: dict[str, list[np.ndarray]],
        call_count: int,
        is_first_call: bool,
        current_prompt: str | None,
        warned_single_external_fallback: bool,
        video_across_time: list[tuple[torch.Tensor, int]],
    ) -> ARSessionState:
        return ARSessionState(
            frame_buffers=frame_buffers,
            call_count=call_count,
            is_first_call=is_first_call,
            current_prompt=current_prompt,
            warned_single_external_fallback=warned_single_external_fallback,
            video_across_time=video_across_time,
        )
    
    def _prepare_inference(
        self,
        obs: dict,
        *,
        index: int,
        reset_all_sessions_for_workers: bool,
    ) -> PreparedARInference:
        session_id = obs.get("session_id", None)
        session_state_key = session_key(session_id)
        reset_session_for_workers = (
            reset_all_sessions_for_workers
            or session_state_key in self._pending_worker_reset_sessions
        )
        if reset_session_for_workers:
            self._pending_worker_reset_sessions.discard(session_state_key)

        new_session = session_state_key not in self._session_states
        if new_session:
            logger.info("New session started: '%s'", session_id)
        elif self._active_session_key != session_state_key:
            logger.info("Switching session from '%s' to '%s'", self._active_session_key, session_state_key)

        with nvtx_range("dreamzero.ar.session.restore"):
            self._load_ar_session(session_id, reset=reset_session_for_workers)

        rtc_step_idx = extract_optional_int(obs.get("rtc_step_idx", None))
        rtc_inference_delay_steps = max(
            extract_optional_int(obs.get("rtc_inference_delay_steps", 0), default=0) or 0,
            0,
        )
        rtc_requested = rtc_step_idx is not None
        if rtc_requested and self._cache_order_sensitive:
            logger.warning(
                "Ignoring RTC request because DreamZero mode %s requires in-order causal cache updates.",
                self._mot_inference_video_mode,
            )
            rtc_requested = False
            rtc_step_idx = None
            rtc_inference_delay_steps = 0
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
            prompt_for_debug = str(converted_obs.get("annotation.language.action_text", ""))
            input_vae_debug = self._make_input_vae_debug_spec(prompt_for_debug)
            if input_vae_debug is not None:
                model_kwargs["input_vae_debug"] = input_vae_debug

        wrapper_state = self._snapshot_wrapper_state(
            self._frame_buffers,
            self._call_count,
            self._is_first_call,
            self._current_prompt,
            self._warned_single_external_fallback,
            self.video_across_time,
        )

        return PreparedARInference(
            index=index,
            obs=obs,
            session_id=None if session_id is None else session_state_key,
            session_key=session_state_key,
            converted_obs=converted_obs,
            model_kwargs=model_kwargs,
            reset_session_for_workers=reset_session_for_workers,
            reset_all_sessions_for_workers=reset_all_sessions_for_workers,
            rtc_requested=rtc_requested,
            rtc_step_idx=rtc_step_idx,
            rtc_inference_delay_steps=rtc_inference_delay_steps,
            session_state=session_state,
            had_previous_chunk=had_previous_chunk,
            phase=phase,
            wrapper_state=wrapper_state,
        )

    def _batch_group_signature(self, prepared: PreparedARInference) -> tuple:
        if prepared.model_kwargs:
            return ("single", prepared.index)
        return (
            "batch",
            self._converted_video_frame_count(prepared.converted_obs),
            self._model_session_store.state_signature(
                prepared.session_id,
                reset=prepared.reset_session_for_workers,
            ),
        )

    def _build_inference_groups(self, prepared_items: list[PreparedARInference]) -> list[list[PreparedARInference]]:
        groups: list[list[PreparedARInference]] = []
        signature_to_group: dict[tuple, list[PreparedARInference]] = {}
        for prepared in prepared_items:
            signature = self._batch_group_signature(prepared)
            group = signature_to_group.get(signature)
            if group is None:
                group = []
                signature_to_group[signature] = group
                groups.append(group)
            group.append(prepared)
        return groups

    def _postprocess_group_actions(
        self,
        prepared_items: list[PreparedARInference],
        result_batch: Batch,
        video_pred: torch.Tensor | None,
    ) -> list[np.ndarray]:
        batch_size = len(prepared_items)
        action_dict = self._extract_action_dict(result_batch.act)
        actions: list[np.ndarray] = []

        for batch_index, prepared in enumerate(prepared_items):
            self._apply_ar_session_state(prepared.wrapper_state)
            sliced_action_dict = self._slice_action_dict(action_dict, batch_index, batch_size)
            action = self._convert_action(sliced_action_dict)

            if prepared.rtc_requested and prepared.had_previous_chunk:
                if prepared.rtc_inference_delay_steps >= action.shape[0]:
                    logger.warning(
                        "RTC inference delay %d exceeds chunk length %d; keeping the final action only.",
                        prepared.rtc_inference_delay_steps,
                        action.shape[0],
                    )
                action, applied_delay_steps = trim_action_chunk_for_delay(
                    action,
                    prepared.rtc_inference_delay_steps,
                )
            else:
                applied_delay_steps = 0

            if prepared.rtc_requested and prepared.session_state is not None:
                prepared.session_state.last_action_chunk_abs = action.copy()
                prepared.session_state.request_count += 1
                prepared.session_state.consumed_steps = 0
                if prepared.rtc_step_idx is not None:
                    prepared.session_state.chunk_start_step_idx = prepared.rtc_step_idx + applied_delay_steps
                else:
                    prepared.session_state.chunk_start_step_idx = None
                logger.info(
                    "RTC stored executable chunk session=%s len=%d chunk_start_step_idx=%s delay_trim=%d",
                    prepared.session_id,
                    action.shape[0],
                    prepared.session_state.chunk_start_step_idx,
                    applied_delay_steps,
                )

            if video_pred is not None:
                video_pred_for_session = (
                    video_pred[batch_index : batch_index + 1]
                    if batch_size > 1
                    else video_pred
                )
                video_chunk = self._prepare_video_pred_for_saving(video_pred_for_session)
                if video_chunk is not None:
                    self.video_across_time.append(video_chunk)

            if self._is_first_call:
                self._is_first_call = False
            self._capture_ar_session_wrapper(prepared.session_id)
            actions.append(action)

        return actions

    def _run_prepared_group(self, prepared_items: list[PreparedARInference]) -> list[np.ndarray]:
        if not prepared_items:
            return []
        phase = "batch" if len(prepared_items) > 1 else prepared_items[0].phase
        session_ids = [prepared.session_id for prepared in prepared_items]
        reset_flags = [prepared.reset_session_for_workers for prepared in prepared_items]
        reset_all_sessions = any(prepared.reset_all_sessions_for_workers for prepared in prepared_items)
        model_kwargs = prepared_items[0].model_kwargs
        if any(prepared.model_kwargs != model_kwargs for prepared in prepared_items):
            raise BatchIncompatibleError("Cannot batch requests with different model kwargs.")

        with nvtx_range(f"dreamzero.ar.infer[{phase}].total"):
            with nvtx_range(f"dreamzero.ar.infer[{phase}].session_restore"):
                self._model_session_store.restore_many(session_ids, resets=reset_flags)

            with nvtx_range(f"dreamzero.ar.infer[{phase}].batch_build"):
                converted_obs = self._pack_converted_observations(
                    [prepared.converted_obs for prepared in prepared_items]
                )
                worker_obs = self._set_batch_reset_flags(
                    converted_obs,
                    reset_flags,
                    session_ids=session_ids,
                    reset_all_sessions=reset_all_sessions,
                )
                batch = Batch(obs=converted_obs)

            signal_tensor = torch.zeros(1, dtype=torch.int32, device="cpu")
            with nvtx_range(f"dreamzero.ar.infer[{phase}].dist_broadcast.signal"):
                dist.broadcast(signal_tensor, src=0, group=self._signal_group)

            self._broadcast_payload_to_workers(
                {
                    "obs": worker_obs,
                    "model_kwargs": model_kwargs,
                },
                phase=phase,
            )

            with nvtx_range(f"dreamzero.ar.infer[{phase}].dist_barrier.pre_forward"):
                dist.barrier()
            with nvtx_range(f"dreamzero.ar.infer[{phase}].policy_forward"):
                with torch.no_grad():
                    result_batch, video_pred = self._policy.lazy_joint_forward_causal(batch, **model_kwargs)
            with nvtx_range(f"dreamzero.ar.infer[{phase}].dist_barrier.post_forward"):
                dist.barrier()

            with nvtx_range(f"dreamzero.ar.infer[{phase}].session_capture"):
                self._model_session_store.capture_many(session_ids)
            with nvtx_range(f"dreamzero.ar.infer[{phase}].action_postprocess"):
                return self._postprocess_group_actions(prepared_items, result_batch, video_pred)

    def infer_many(self, obs_list: list[dict]) -> list[np.ndarray]:
        if not obs_list:
            return []

        reset_all_for_next_group = self._pending_worker_reset_all
        if reset_all_for_next_group:
            self._pending_worker_reset_all = False
            self._pending_worker_reset_sessions.clear()

        prepared_items = [
            self._prepare_inference(
                obs,
                index=index,
                reset_all_sessions_for_workers=reset_all_for_next_group and index == 0,
            )
            for index, obs in enumerate(obs_list)
        ]

        actions_by_index: list[np.ndarray | None] = [None for _ in obs_list]
        for group in self._build_inference_groups(prepared_items):
            try:
                group_actions = self._run_prepared_group(group)
            except BatchIncompatibleError:
                if len(group) == 1:
                    raise
                group_actions = []
                for prepared in group:
                    group_actions.extend(self._run_prepared_group([prepared]))
            for prepared, action in zip(group, group_actions, strict=True):
                actions_by_index[prepared.index] = action

        if any(action is None for action in actions_by_index):
            raise RuntimeError("Missing action for one or more DreamZero batched requests.")
        return [action for action in actions_by_index if action is not None]

    def infer(self, obs: dict) -> np.ndarray:
        """Infer actions from one roboarena observation."""
        return self.infer_many([obs])[0]
    
    def _prepare_video_pred_for_saving(self, video_pred: torch.Tensor) -> tuple[torch.Tensor, int] | None:
        condition_latent_frames = int(
            getattr(
                self._policy.trained_model.action_head,
                "last_video_pred_condition_latent_frames",
                0,
            )
            or 0
        )
        if condition_latent_frames <= 0:
            return video_pred.detach(), 0
        if video_pred.ndim < 3:
            logger.warning("Unexpected video_pred shape %s; keeping it unchanged", tuple(video_pred.shape))
            return video_pred.detach(), 0
        condition_latent_frames = min(condition_latent_frames, video_pred.shape[2])
        logger.info(
            "Keeping %d condition/input latent frame(s) for per-chunk VAE decode; video_pred shape=%s",
            condition_latent_frames,
            tuple(video_pred.shape),
        )
        return video_pred.detach(), condition_latent_frames

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
                        frame_list = _decode_video_pred_chunks(
                            self._policy.trained_model.action_head,
                            self.video_across_time,
                        )

                        if len(frame_list) > 0:
                            sample_frame = frame_list[0]
                            if len(sample_frame.shape) == 3 and sample_frame.shape[2] in [1, 3, 4]:
                                save_dir = self._output_dir
                                os.makedirs(save_dir, exist_ok=True)
                                all_mp4_files = [f for f in os.listdir(save_dir) if f.endswith(".mp4")]
                                timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
                                num_frames = len(frame_list)
                                n = num_frames // self.RAW_FRAMES_PER_BLOCK
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
                self._current_prompt = None
                self._warned_single_external_fallback = False
                self._reset_model_temporal_state()
    
    def reset(self, reset_info: dict) -> None:
        """Reset the policy state for a new episode.
        
        Clears frame buffers and resets call count.
        """
        reset_all = bool(reset_info.get("reset_all_sessions", False))
        session_ids = reset_info.get("session_ids", None)
        if session_ids is None and reset_info.get("session_id", None) is not None:
            session_ids = [reset_info["session_id"]]

        if isinstance(session_ids, np.ndarray):
            session_ids = session_ids.reshape(-1).tolist()
        elif isinstance(session_ids, (str, bytes)):
            session_ids = [session_ids.decode() if isinstance(session_ids, bytes) else session_ids]

        if session_ids is None:
            reset_all = True

        if reset_all:
            self._clear_all_ar_sessions(save_video=True)
            self._pending_worker_reset_all = True
            self._pending_worker_reset_sessions.clear()
        else:
            for session_id in session_ids:
                session_id_str = session_id.decode() if isinstance(session_id, bytes) else str(session_id)
                self._clear_ar_session(session_id_str, save_video=True)
                self._pending_worker_reset_sessions.add(session_key(session_id_str))

        self._current_session_id = None
        logger.info(
            "policy reset requested with keys=%s reset_all=%s session_ids=%s",
            sorted(reset_info.keys()),
            reset_all,
            session_ids,
        )


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
        self._session_store = ActionHeadSessionStore(policy)
        # Create output directory if specified
        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)
            os.makedirs(os.path.join(self._output_dir, "inputs"), exist_ok=True)

    def _reset_policy_temporal_state(self) -> None:
        self._session_store.reset_loaded_state()
    
    def _save_input_obs(self, obs: dict) -> None:
        """Save incoming observation images per message.
        
        Expected format: THWC (Time, Height, Width, Channel).
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

    def _prepare_video_pred_for_saving(self, video_pred: torch.Tensor) -> tuple[torch.Tensor, int] | None:
        condition_latent_frames = int(
            getattr(
                self._policy.trained_model.action_head,
                "last_video_pred_condition_latent_frames",
                0,
            )
            or 0
        )
        if condition_latent_frames <= 0:
            return video_pred.detach(), 0
        if video_pred.ndim < 3:
            logger.warning("Unexpected video_pred shape %s; keeping it unchanged", tuple(video_pred.shape))
            return video_pred.detach(), 0
        condition_latent_frames = min(condition_latent_frames, video_pred.shape[2])
        logger.info(
            "Keeping %d condition/input latent frame(s) for per-chunk VAE decode; video_pred shape=%s",
            condition_latent_frames,
            tuple(video_pred.shape),
        )
        return video_pred.detach(), condition_latent_frames



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
                reset_flag_raw = batch.obs.pop(RESET_FLAG_KEY, False)
                reset_all_sessions = bool(batch.obs.pop(RESET_ALL_SESSIONS_KEY, False))
                worker_session_id = batch.obs.pop(SESSION_ID_KEY, None)
                with nvtx_range("dreamzero.ar.worker.session_restore"):
                    if reset_all_sessions:
                        self._session_store.reset_all()
                    if isinstance(worker_session_id, list):
                        if isinstance(reset_flag_raw, list):
                            reset_flags = [bool(flag) for flag in reset_flag_raw]
                        else:
                            reset_flags = [bool(reset_flag_raw) for _ in worker_session_id]
                        self._session_store.restore_many(
                            [str(session_id) for session_id in worker_session_id],
                            resets=reset_flags,
                        )
                    elif worker_session_id is not None:
                        self._session_store.restore(str(worker_session_id), reset=bool(reset_flag_raw))
                    elif bool(reset_flag_raw):
                        self._reset_policy_temporal_state()
                # Participate in distributed forward pass
                with nvtx_range("dreamzero.ar.worker.dist_barrier.pre_forward"):
                    dist.barrier()
                with nvtx_range("dreamzero.ar.worker.policy_forward"):
                    with torch.no_grad():
                        result_batch, video_pred = self._policy.lazy_joint_forward_causal(batch, **model_kwargs)
                with nvtx_range("dreamzero.ar.worker.dist_barrier.post_forward"):
                    dist.barrier()
                if isinstance(worker_session_id, list):
                    with nvtx_range("dreamzero.ar.worker.session_capture"):
                        self._session_store.capture_many([str(session_id) for session_id in worker_session_id])
                elif worker_session_id is not None:
                    with nvtx_range("dreamzero.ar.worker.session_capture"):
                        self._session_store.capture(str(worker_session_id))

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
                    video_chunk = (
                        self._prepare_video_pred_for_saving(video_pred)
                        if video_pred is not None
                        else None
                    )

                    print(f"Inference Time: {time.perf_counter() - infer_start_time:.2f} seconds")

                    if video_chunk is not None:
                        self.video_across_time.append(video_chunk)

                    if len(self.video_across_time) > 10:
                        frame_list = _decode_video_pred_chunks(
                            self._policy.trained_model.action_head,
                            self.video_across_time,
                        )

                        sample_frame = frame_list[0]
                        if len(sample_frame.shape) == 3 and sample_frame.shape[2] in [1, 3, 4]:
                            # Save all frames as a single MP4 file
                            save_dir = self._output_dir if self._output_dir else "."
                            os.makedirs(save_dir, exist_ok=True)
                            all_mp4_files = [f for f in os.listdir(save_dir) if f.endswith(".mp4")]
                            timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
                            num_frames = len(frame_list)
                            n = num_frames // 8
                            output_path = os.path.join(save_dir, f'{len(all_mp4_files):06}_{timestamp}_n{n}.mp4')
                            imageio.mimsave(output_path, frame_list, fps=5, codec='libx264')
                            print(f"Saved video to: {output_path}")
                        else:
                            print(f"Warning: Invalid frame shape {sample_frame.shape}. Expected (H, W, C) with C in [1, 3, 4]. Skipping video save.")

                        self.video_across_time = []
                    elif self._policy.trained_model.action_head.current_start_frame == 1 + self._policy.trained_model.action_head.num_frame_per_block and len(self.video_across_time) > 1:
                        print("current_start_frame == 1 + num_frame_per_block and len(self.video_across_time) > 1")
                        frame_list = _decode_video_pred_chunks(
                            self._policy.trained_model.action_head,
                            self.video_across_time[:-1],
                        )
                        sample_frame = frame_list[0]
                        if len(sample_frame.shape) == 3 and sample_frame.shape[2] in [1, 3, 4]:
                            # Save all frames as a single MP4 file
                            save_dir = self._output_dir if self._output_dir else "."
                            os.makedirs(save_dir, exist_ok=True)
                            all_mp4_files = [f for f in os.listdir(save_dir) if f.endswith(".mp4")]
                            timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
                            num_frames = len(frame_list)
                            n = num_frames // 8
                            output_path = os.path.join(save_dir, f'{len(all_mp4_files):06}_{timestamp}_n{n}.mp4')
                            imageio.mimsave(output_path, frame_list, fps=5, codec='libx264')
                            print(f"Saved video to: {output_path}")
                        self.video_across_time = [video_chunk] if video_chunk is not None else []

                    
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
                        frame_list = _decode_video_pred_chunks(
                            self._policy.trained_model.action_head,
                            self.video_across_time,
                        )

                        sample_frame = frame_list[0]
                        if len(sample_frame.shape) == 3 and sample_frame.shape[2] in [1, 3, 4]:
                            # Save all frames as a single MP4 file
                            save_dir = self._output_dir if self._output_dir else "."
                            os.makedirs(save_dir, exist_ok=True)
                            all_mp4_files = [f for f in os.listdir(save_dir) if f.endswith(".mp4")]
                            timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
                            num_frames = len(frame_list)
                            n = num_frames // 8
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


def _get_training_view_resolution(policy: GrootSimPolicy) -> tuple[int, int] | None:
    train_cfg = getattr(policy, "train_cfg", None)
    if train_cfg is None:
        return None

    height = getattr(train_cfg, "image_resolution_height", None)
    width = getattr(train_cfg, "image_resolution_width", None)
    if height is None or width is None:
        return None
    return int(height), int(width)


def _patch_eval_transform_input_resolution(
    policy: GrootSimPolicy,
    image_height: int,
    image_width: int,
) -> None:
    eval_transform = getattr(policy, "eval_transform", None)
    transforms = getattr(eval_transform, "transforms", None)
    if transforms is None:
        return

    image_height = int(image_height)
    image_width = int(image_width)
    expected_resolution = (image_width, image_height)
    for transform in transforms:
        if not hasattr(transform, "original_resolutions"):
            continue
        try:
            original_resolutions = getattr(transform, "original_resolutions")
        except AssertionError:
            continue
        if not original_resolutions:
            continue
        transform.original_resolutions = {
            key: expected_resolution for key in original_resolutions
        }
        transform_name = transform.__class__.__name__
        if transform_name not in {"VideoCrop", "VideoResize"}:
            continue

        old_size = (getattr(transform, "height", None), getattr(transform, "width", None))
        transform.height = image_height
        transform.width = image_width

        get_transform = getattr(transform, "get_transform", None)
        if not callable(get_transform):
            continue

        if getattr(transform, "backend", None) == "albumentations":
            import albumentations as A

            transform.train_transform = A.ReplayCompose(
                transforms=[get_transform(mode="train")]
            )
            eval_transform = get_transform(mode="eval")
            transform.eval_transform = (
                A.ReplayCompose(transforms=[eval_transform])
                if eval_transform is not None
                else None
            )
        else:
            transform.train_transform = get_transform(mode="train")
            transform.eval_transform = get_transform(mode="eval")

        logger.info(
            "Patched %s input resolution from %s to %s and rebuilt cached transforms",
            transform_name,
            old_size,
            (image_height, image_width),
        )


def _override_max_chunk_size(policy: GrootSimPolicy, max_chunk_size: int) -> None:
    action_head = getattr(getattr(policy, "trained_model", None), "action_head", None)
    model = getattr(action_head, "model", None)
    if model is None:
        return

    num_frame_per_block = int(getattr(model, "num_frame_per_block", 1))
    frame_seqlen = int(getattr(model, "frame_seqlen", 1))
    new_local_attn_size = (
        int(max_chunk_size) * num_frame_per_block + 1
        if int(max_chunk_size) != -1
        else -1
    )
    model.local_attn_size = new_local_attn_size

    blocks = getattr(model, "blocks", [])
    for block in blocks:
        block.local_attn_size = new_local_attn_size
        self_attn = getattr(block, "self_attn", None)
        if self_attn is None:
            continue
        self_attn.local_attn_size = new_local_attn_size
        self_attn.max_attention_size = (
            21 * frame_seqlen if new_local_attn_size == -1
            else new_local_attn_size * frame_seqlen
        )

    logger.info(
        "Overrode inference max_chunk_size=%s -> local_attn_size=%s (synced to %d blocks)",
        max_chunk_size,
        new_local_attn_size,
        len(blocks),
    )


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
    if args.max_chunk_size is not None:
        _override_max_chunk_size(policy, args.max_chunk_size)
    training_view_resolution = _get_training_view_resolution(policy)
    if training_view_resolution is not None:
        _patch_eval_transform_input_resolution(policy, *training_view_resolution)
        logger.info(
            "Using training single-view resolution %dx%d for websocket inputs",
            training_view_resolution[0],
            training_view_resolution[1],
        )
    loaded_architecture = getattr(getattr(policy.trained_model, "action_head", None), "config", None)
    loaded_architecture = getattr(loaded_architecture, "architecture", None)
    if loaded_architecture in ("joint", "mot") and loaded_architecture != effective_architecture:
        raise RuntimeError(
            f"Loaded WAM architecture {loaded_architecture!r}, expected {effective_architecture!r}."
        )
    logger.info("DreamZero WAM architecture active: %s", loaded_architecture or effective_architecture)
    image_height, image_width = _get_expected_video_resolution(policy)
    logger.info("Using websocket image resolution %dx%d", image_height, image_width)
    mot_inference_video_mode, cache_order_sensitive = _dreamzero_cache_mode(policy)
    if cache_order_sensitive and args.use_rtc:
        logger.warning(
            "Checkpoint mode %s is cache-order-sensitive; RTC requests will be ignored.",
            mot_inference_video_mode,
        )
    action_head = getattr(getattr(policy, "trained_model", None), "action_head", None)
    dynamic_cache_schedule = bool(getattr(action_head, "dynamic_cache_schedule", False))
    supports_batching = not dynamic_cache_schedule
    if dynamic_cache_schedule:
        logger.warning(
            "Disabling server-side request batching because DYNAMIC_CACHE_SCHEDULE uses batch-global skip state."
        )

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
            save_input_vae_videos=args.save_input_vae_videos,
            max_saved_input_vae_videos=args.max_saved_input_vae_videos,
            input_vae_video_fps=args.input_vae_video_fps,
            input_vae_video_dir=args.input_vae_video_dir,
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
        wam_architecture=loaded_architecture or effective_architecture,
        mot_inference_video_mode=mot_inference_video_mode,
        cache_order_sensitive=cache_order_sensitive,
        supports_rtc=not cache_order_sensitive,
        supports_async_prefetch=not cache_order_sensitive,
        supports_parallel_sessions=True,
        supports_batching=supports_batching,
        max_batch_size=args.batch_max_size,
        batch_timeout_ms=args.batch_timeout_ms,
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
