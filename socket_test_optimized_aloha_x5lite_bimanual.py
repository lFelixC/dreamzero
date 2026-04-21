from __future__ import annotations

import asyncio
import dataclasses
import datetime
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import tyro
from tianshou.data import Batch
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from eval_utils.torch_compile_backend import configure_torch_compile_backend
from eval_utils.policy_server import PolicyServerConfig, WebsocketPolicyServer
from eval_utils.serve_dreamzero_wan22 import _get_expected_video_resolution, _resize_frames_to_resolution
from groot.vla.data.schema import EmbodimentTag
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy

DEFAULT_TORCH_COMPILE_BACKEND = configure_torch_compile_backend(default_backend="cudagraphs")

logger = logging.getLogger(__name__)

RESET_FLAG_KEY = "__reset_policy_state__"
CONTINUE_SIGNAL = 0
SHUTDOWN_SIGNAL = 1


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    timeout_seconds: int = 604800
    handshake_timeout_seconds: float | None = 0.0
    model_path: str = "./checkpoints/dreamzero"
    tokenizer_path: str | None = None
    image_height: int | None = None
    image_width: int | None = None
    enable_dit_cache: bool = False
    max_chunk_size: int | None = None
    output_root: str | None = None
    index: int = 0


def _normalize_image_array(data: Any, key: str) -> np.ndarray:
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


def _ensure_2d_state(value: Any, key: str, width: int) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if arr.ndim != 2 or arr.shape[1] != width:
        raise ValueError(f"{key} must have shape ({1}, {width}) or ({width},), got {arr.shape}")
    return np.ascontiguousarray(arr.astype(np.float64, copy=False))


def _normalize_prompt(value: Any) -> str:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        value = value.reshape(-1)[0]
    if value is None:
        return ""
    return str(value)


def _extract_action_dict(action_chunk: Batch | dict) -> dict[str, Any]:
    if isinstance(action_chunk, dict):
        return {
            str(k): v
            for k, v in action_chunk.items()
            if isinstance(k, str) and k.startswith("action.")
        }

    try:
        return {
            str(k): v
            for k, v in action_chunk.items()
            if isinstance(k, str) and k.startswith("action.")
        }
    except Exception:
        action_dict: dict[str, Any] = {}
        for key in dir(action_chunk):
            if not key.startswith("action."):
                continue
            try:
                action_dict[key] = getattr(action_chunk, key)
            except AttributeError:
                continue
        return action_dict


def _to_numpy_2d(value: Any, key: str, width: int) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        if width != 1:
            raise ValueError(f"{key} expected width {width}, got scalar")
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        if width == 1:
            arr = arr.reshape(-1, 1)
        elif arr.shape[0] == width:
            arr = arr.reshape(1, width)
        else:
            raise ValueError(f"{key} expected trailing width {width}, got {arr.shape}")
    elif arr.ndim == 2 and arr.shape[1] == width:
        pass
    else:
        raise ValueError(f"{key} expected shape (N, {width}), got {arr.shape}")

    return np.ascontiguousarray(arr)


def _align_rows(arr: np.ndarray, rows: int, key: str) -> np.ndarray:
    if arr.shape[0] == rows:
        return arr
    if arr.shape[0] == 1:
        return np.repeat(arr, rows, axis=0)
    raise ValueError(f"{key} has {arr.shape[0]} rows but expected {rows}")


def _reset_model_temporal_state(policy: GrootSimPolicy) -> None:
    if hasattr(policy.trained_model, "action_head") and hasattr(
        policy.trained_model.action_head, "current_start_frame"
    ):
        policy.trained_model.action_head.current_start_frame = 0


def init_mesh(master_port: int | None = None) -> DeviceMesh:
    if not torch.cuda.is_available():
        raise RuntimeError("DreamZero server currently requires CUDA.")

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        if master_port is not None:
            os.environ["MASTER_PORT"] = str(master_port)
        else:
            os.environ.setdefault("MASTER_PORT", "29500")

        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            dist.init_process_group("nccl")
        else:
            dist.init_process_group("nccl", rank=0, world_size=1)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    logger.info("Rank %d/%d using cuda:%d", rank, world_size, local_rank)

    return init_device_mesh(
        device_type="cuda",
        mesh_shape=(world_size,),
        mesh_dim_names=("ip",),
    )


class AlohaBimanualPolicy:
    def __init__(
        self,
        groot_policy: GrootSimPolicy,
        signal_group: dist.ProcessGroup,
        image_height: int,
        image_width: int,
        output_dir: str | None = None,
        max_chunk_size: int | None = None,
    ) -> None:
        self._policy = groot_policy
        self._signal_group = signal_group
        self._image_height = image_height
        self._image_width = image_width
        self._output_dir = output_dir
        self._max_chunk_size = max_chunk_size
        self._current_session_id: str | None = None
        self._reset_next_infer = False
        self._video_pred_latents: list[torch.Tensor] = []
        self._current_prompt = ""

        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)

    def _extract_video(self, obs: dict[str, Any], model_key: str, raw_key: str) -> np.ndarray:
        if model_key in obs:
            data = obs[model_key]
        elif raw_key in obs:
            data = obs[raw_key]
        else:
            raise KeyError(f"Missing required image input: '{model_key}' or '{raw_key}'")

        arr = _normalize_image_array(data, model_key if model_key in obs else raw_key)
        arr = _resize_frames_to_resolution(arr, self._image_height, self._image_width)
        return np.ascontiguousarray(arr)

    def _extract_state(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        left_joint_key = "state.left_joint_pos"
        left_gripper_key = "state.left_gripper_pos"
        right_joint_key = "state.right_joint_pos"
        right_gripper_key = "state.right_gripper_pos"

        if all(key in obs for key in (left_joint_key, left_gripper_key, right_joint_key, right_gripper_key)):
            return {
                left_joint_key: _ensure_2d_state(obs[left_joint_key], left_joint_key, 6),
                left_gripper_key: _ensure_2d_state(obs[left_gripper_key], left_gripper_key, 1),
                right_joint_key: _ensure_2d_state(obs[right_joint_key], right_joint_key, 6),
                right_gripper_key: _ensure_2d_state(obs[right_gripper_key], right_gripper_key, 1),
            }

        if "observation.state" not in obs:
            raise KeyError(
                "Missing state keys. Expected either split keys state.left_* / state.right_* "
                "or packed observation.state."
            )

        packed = np.asarray(obs["observation.state"], dtype=np.float64)
        if packed.ndim == 2:
            if packed.shape[0] != 1:
                raise ValueError(f"observation.state must be 1D or single-row 2D, got {packed.shape}")
            packed = packed[0]
        if packed.shape != (14,):
            raise ValueError(f"observation.state must have shape (14,), got {packed.shape}")

        right = packed[:7]
        left = packed[7:]
        return {
            left_joint_key: left[:6].reshape(1, 6),
            left_gripper_key: left[6:7].reshape(1, 1),
            right_joint_key: right[:6].reshape(1, 6),
            right_gripper_key: right[6:7].reshape(1, 1),
        }

    def _convert_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        converted: dict[str, Any] = {
            "video.cam_high": self._extract_video(obs, "video.cam_high", "observation.images.cam_high"),
            "video.cam_left": self._extract_video(obs, "video.cam_left", "observation.images.cam_left"),
            "video.cam_right": self._extract_video(obs, "video.cam_right", "observation.images.cam_right"),
        }
        converted.update(self._extract_state(obs))

        prompt = _normalize_prompt(obs.get("prompt", obs.get("annotation.task", "")))
        if prompt:
            self._current_prompt = prompt
        converted["annotation.task"] = prompt
        return converted

    def _convert_action(self, action_dict: dict[str, Any]) -> np.ndarray:
        left_joint = None
        left_gripper = None
        right_joint = None
        right_gripper = None
        packed = None

        for key, value in action_dict.items():
            if "left_joint_pos" in key or "left_joint_position" in key:
                left_joint = value
            elif "left_gripper_pos" in key or "left_gripper_position" in key:
                left_gripper = value
            elif "right_joint_pos" in key or "right_joint_position" in key:
                right_joint = value
            elif "right_gripper_pos" in key or "right_gripper_position" in key:
                right_gripper = value
            elif "joint_position" in key and "gripper" not in key:
                packed = value

        if left_joint is not None and right_joint is not None:
            left_joint_arr = _to_numpy_2d(left_joint, "action.left_joint_pos", 6)
            right_joint_arr = _to_numpy_2d(right_joint, "action.right_joint_pos", 6)
            row_count = max(left_joint_arr.shape[0], right_joint_arr.shape[0])
            left_joint_arr = _align_rows(left_joint_arr, row_count, "action.left_joint_pos")
            right_joint_arr = _align_rows(right_joint_arr, row_count, "action.right_joint_pos")

            left_gripper_arr = (
                np.zeros((row_count, 1), dtype=np.float32)
                if left_gripper is None
                else _align_rows(_to_numpy_2d(left_gripper, "action.left_gripper_pos", 1), row_count, "action.left_gripper_pos")
            )
            right_gripper_arr = (
                np.zeros((row_count, 1), dtype=np.float32)
                if right_gripper is None
                else _align_rows(_to_numpy_2d(right_gripper, "action.right_gripper_pos", 1), row_count, "action.right_gripper_pos")
            )

            packed_action = np.concatenate(
                [
                    np.concatenate([left_joint_arr, left_gripper_arr], axis=-1),
                    np.concatenate([right_joint_arr, right_gripper_arr], axis=-1),
                ],
                axis=-1,
            )
        elif packed is not None:
            packed_action = _to_numpy_2d(packed, "action.joint_position", 14)
        else:
            raise KeyError(
                f"Could not construct ALOHA bimanual action from keys: {sorted(action_dict.keys())}"
            )

        if self._max_chunk_size is not None and self._max_chunk_size > 0:
            packed_action = packed_action[: self._max_chunk_size]

        return np.ascontiguousarray(packed_action.astype(np.float32))

    def _broadcast_batch_to_workers(self, obs: dict[str, Any]) -> None:
        serialized = pickle.dumps(obs)
        data_size = len(serialized)

        size_tensor = torch.tensor([data_size], dtype=torch.int64, device="cuda")
        dist.broadcast(size_tensor, src=0)

        data_tensor = torch.frombuffer(serialized, dtype=torch.uint8).cuda()
        dist.broadcast(data_tensor, src=0)

    def _set_reset_flag(self, converted_obs: dict[str, Any], should_reset: bool) -> dict[str, Any]:
        worker_obs = dict(converted_obs)
        worker_obs[RESET_FLAG_KEY] = bool(should_reset)
        return worker_obs

    def _save_predicted_video(self) -> None:
        if not self._output_dir or not self._video_pred_latents:
            return

        try:
            import imageio
            from einops import rearrange

            action_head = self._policy.trained_model.action_head
            latents = torch.cat(self._video_pred_latents, dim=2)
            with torch.no_grad():
                frames = action_head.vae.decode(
                    latents,
                    tiled=action_head.tiled,
                    tile_size=(action_head.tile_size_height, action_head.tile_size_width),
                    tile_stride=(action_head.tile_stride_height, action_head.tile_stride_width),
                )

            frames = rearrange(frames, "B C T H W -> B T H W C")[0]
            frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
            if len(frames) == 0:
                return

            os.makedirs(self._output_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
            existing = [f for f in os.listdir(self._output_dir) if f.endswith(".mp4")]
            safe_prompt = self._current_prompt.replace(" ", "_")
            safe_prompt = "".join(c for c in safe_prompt if c.isalnum() or c in "_-.")
            if len(safe_prompt) > 80:
                safe_prompt = safe_prompt[:80]
            if not safe_prompt:
                safe_prompt = "no_prompt"

            output_path = os.path.join(
                self._output_dir,
                f"{len(existing):06}_{safe_prompt}_{timestamp}.mp4",
            )
            imageio.mimsave(output_path, list(frames), fps=5, codec="libx264")
            logger.info("Saved video prediction (%d frames) to %s", len(frames), output_path)
        except Exception as e:
            logger.warning("Failed to save video prediction: %s", e)

    def _reset_local_state(self, save_video: bool = False) -> None:
        if save_video:
            self._save_predicted_video()
        self._video_pred_latents.clear()
        self._current_prompt = ""
        _reset_model_temporal_state(self._policy)

    def flush_pending_video(self) -> None:
        self._reset_local_state(save_video=True)

    def infer(self, obs: dict[str, Any]) -> np.ndarray:
        session_id = obs.get("session_id")
        should_reset = self._reset_next_infer or (
            session_id is not None and session_id != self._current_session_id
        )
        if should_reset:
            self._reset_local_state(save_video=not self._reset_next_infer)
            self._reset_next_infer = False
        if session_id is not None:
            self._current_session_id = str(session_id)

        converted_obs = self._convert_observation(obs)
        worker_obs = self._set_reset_flag(converted_obs, should_reset)

        signal_tensor = torch.tensor([CONTINUE_SIGNAL], dtype=torch.int32, device="cpu")
        dist.broadcast(signal_tensor, src=0, group=self._signal_group)
        self._broadcast_batch_to_workers(worker_obs)

        batch = Batch(obs=converted_obs)
        dist.barrier()
        with torch.no_grad():
            result_batch, video_pred = self._policy.lazy_joint_forward_causal(batch)
        dist.barrier()
        if video_pred is not None and self._output_dir:
            self._video_pred_latents.append(video_pred.detach())

        action_dict = _extract_action_dict(result_batch.act)
        action = self._convert_action(action_dict)
        logger.info(
            "infer session_id=%s prompt=%r action_shape=%s",
            self._current_session_id,
            converted_obs.get("annotation.task", ""),
            tuple(action.shape),
        )
        return action

    def reset(self, reset_info: dict[str, Any]) -> None:
        self._reset_local_state(save_video=True)
        self._reset_next_infer = True
        self._current_session_id = None
        logger.info("policy reset requested with keys=%s", sorted(reset_info.keys()))


class DistributedWorkerLoop:
    def __init__(self, policy: GrootSimPolicy, signal_group: dist.ProcessGroup) -> None:
        self._policy = policy
        self._signal_group = signal_group

    def _receive_batch_from_rank0(self) -> Batch:
        size_tensor = torch.zeros(1, dtype=torch.int64, device="cuda")
        dist.broadcast(size_tensor, src=0)
        data_size = size_tensor.item()

        data_tensor = torch.zeros(data_size, dtype=torch.uint8, device="cuda")
        dist.broadcast(data_tensor, src=0)
        obs = pickle.loads(data_tensor.cpu().numpy().tobytes())
        return Batch(obs=obs)

    async def run_forever(self) -> None:
        logger.info("Worker loop started for rank %d", dist.get_rank())
        signal_tensor = torch.zeros(1, dtype=torch.int32, device="cpu")
        while True:
            dist.broadcast(signal_tensor, src=0, group=self._signal_group)
            signal = int(signal_tensor.item())
            if signal == SHUTDOWN_SIGNAL:
                logger.info("Rank %d received shutdown signal", dist.get_rank())
                return
            if signal != CONTINUE_SIGNAL:
                logger.warning("Rank %d received unknown signal %s", dist.get_rank(), signal)
                continue

            batch = self._receive_batch_from_rank0()
            reset_flag = bool(batch.obs.pop(RESET_FLAG_KEY, False))
            if reset_flag:
                _reset_model_temporal_state(self._policy)

            dist.barrier()
            with torch.no_grad():
                self._policy.lazy_joint_forward_causal(batch)
            dist.barrier()


def _build_output_dir(output_root: str | None, model_path: str, index: int) -> str:
    checkpoint_path = Path(model_path.rstrip("/"))
    output_root_path = Path(output_root) if output_root is not None else checkpoint_path.parent
    timestamp = datetime.datetime.now().strftime("%Y%m%d")
    output_dir = output_root_path / f"real_world_eval_gen_{timestamp}_{index}" / checkpoint_path.name
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir)


def main(args: Args) -> None:
    os.environ["ENABLE_DIT_CACHE"] = "true" if args.enable_dit_cache else "false"
    os.environ.setdefault("ATTENTION_BACKEND", "TE")

    if hasattr(torch._dynamo.config, "recompile_limit"):
        torch._dynamo.config.recompile_limit = 800

    logging.basicConfig(level=logging.INFO, force=True)
    if DEFAULT_TORCH_COMPILE_BACKEND is not None:
        logger.info("Using torch.compile backend=%s for this entrypoint", DEFAULT_TORCH_COMPILE_BACKEND)

    device_mesh = init_mesh()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    timeout_delta = datetime.timedelta(seconds=args.timeout_seconds)
    signal_group = dist.new_group(backend="gloo", timeout=timeout_delta)
    logger.info("Rank %d/%d initialized signal_group", rank, world_size)

    embodiment_tag = EmbodimentTag.ALOHA_X5LITE_BIMANUAL
    logger.info("Loading DreamZero checkpoint from %s with embodiment=%s", args.model_path, embodiment_tag.value)
    policy = GrootSimPolicy(
        embodiment_tag=embodiment_tag,
        model_path=args.model_path,
        tokenizer_path_override=args.tokenizer_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        device_mesh=device_mesh,
    )

    if args.image_height is not None and args.image_width is not None:
        image_height, image_width = int(args.image_height), int(args.image_width)
        logger.info("Using CLI image resolution %dx%d", image_height, image_width)
    else:
        image_height, image_width = _get_expected_video_resolution(policy)
        logger.info("Using checkpoint image resolution %dx%d", image_height, image_width)

    output_dir = _build_output_dir(args.output_root, args.model_path, args.index) if rank == 0 else None
    if output_dir:
        logger.info("Server outputs will be written under %s", output_dir)

    wrapper_policy = AlohaBimanualPolicy(
        groot_policy=policy,
        signal_group=signal_group,
        image_height=image_height,
        image_width=image_width,
        output_dir=output_dir,
        max_chunk_size=args.max_chunk_size,
    )

    server_config = PolicyServerConfig(
        image_resolution=(image_height, image_width),
        needs_wrist_camera=True,
        n_external_cameras=2,
        needs_stereo_camera=False,
        needs_session_id=True,
        action_space="joint_position",
    )

    if rank == 0:
        logger.info("Serving ALOHA bimanual DreamZero websocket server on ws://%s:%d", args.host, args.port)
        server = WebsocketPolicyServer(
            policy=wrapper_policy,
            server_config=server_config,
            host=args.host,
            port=args.port,
            open_timeout=args.handshake_timeout_seconds,
        )
        try:
            server.serve_forever()
        finally:
            wrapper_policy.flush_pending_video()
            shutdown_signal = torch.tensor([SHUTDOWN_SIGNAL], dtype=torch.int32, device="cpu")
            dist.broadcast(shutdown_signal, src=0, group=signal_group)
    else:
        worker = DistributedWorkerLoop(policy=policy, signal_group=signal_group)
        asyncio.run(worker.run_forever())


if __name__ == "__main__":
    main(tyro.cli(Args))
