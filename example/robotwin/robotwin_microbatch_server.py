#!/usr/bin/env python3
"""RoboTwin-specific multi-session microbatch DreamZero websocket server."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import datetime
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import tyro
import websockets.asyncio.server
import websockets.frames
from tianshou.data import Batch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
for extra in (
    REPO_ROOT / "third_party" / "RoboTwin",
    REPO_ROOT / "third_party" / "lerobot" / "src",
    REPO_ROOT / "third_party" / "lerobot",
):
    if extra.exists() and str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from eval_utils import msgpack_numpy
from eval_utils.policy_server import PolicyServerConfig
from eval_utils.serve_dreamzero_wan22 import _get_expected_video_resolution, _resize_frames_to_resolution
from groot.vla.data.schema import EmbodimentTag
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
from socket_test_optimized_aloha_x5lite_bimanual import (
    CONTINUE_SIGNAL,
    RESET_FLAG_KEY,
    SHUTDOWN_SIGNAL,
    AlohaBimanualPolicy,
    DistributedWorkerLoop,
    _ensure_state_array,
    _extract_action_dict,
    _normalize_image_array,
    _normalize_prompt,
    _reset_model_temporal_state,
    configure_torch_compile_backend,
    init_mesh,
)

DEFAULT_TORCH_COMPILE_BACKEND = configure_torch_compile_backend(default_backend="cudagraphs")

logger = logging.getLogger(__name__)


@dataclass
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
    save_server_video: bool = False
    index: int = 0
    microbatch_wait_ms: float = 10.0
    microbatch_max_size: int = 4


@dataclass
class SessionState:
    session_id: str
    prompt: str = ""
    frame_buffers: dict[str, list[np.ndarray]] = field(
        default_factory=lambda: {key: [] for key in AlohaBimanualPolicy.VIDEO_KEYS}
    )
    is_first_call: bool = True
    last_reset_time: float = field(default_factory=time.time)
    infer_count: int = 0

    def clear(self, *, prompt: str = "") -> None:
        self.prompt = str(prompt or "")
        for values in self.frame_buffers.values():
            values.clear()
        self.is_first_call = True
        self.last_reset_time = time.time()
        self.infer_count = 0


@dataclass
class PendingRequest:
    obs: dict[str, Any]
    future: asyncio.Future
    enqueued_at: float
    session_id: str


def _session_id_value(value: Any) -> str:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return "__default__"
        value = value.reshape(-1)[0]
    if value is None:
        return "__default__"
    text = str(value)
    return text if text else "__default__"


def _pad_front_array(arr: np.ndarray, target_length: int) -> np.ndarray:
    if arr.shape[0] >= target_length:
        return np.ascontiguousarray(arr[-target_length:])
    pad_count = target_length - arr.shape[0]
    pad = np.repeat(arr[:1], pad_count, axis=0)
    return np.ascontiguousarray(np.concatenate([pad, arr], axis=0))


def _normalize_action_batch(action: np.ndarray, batch_size: int) -> np.ndarray:
    arr = np.asarray(action, dtype=np.float32)
    if arr.ndim == 2:
        if batch_size == 1:
            arr = arr[None, ...]
        elif arr.shape[0] == batch_size:
            arr = arr[:, None, :]
        else:
            raise ValueError(f"Cannot interpret action shape {arr.shape} for batch size {batch_size}")
    if arr.ndim != 3:
        raise ValueError(f"Expected action batch shape [B,H,D], got {arr.shape}")
    if arr.shape[0] != batch_size:
        raise ValueError(f"Expected action batch size {batch_size}, got {arr.shape}")
    return np.ascontiguousarray(arr)


class RobotwinMicrobatchPolicy(AlohaBimanualPolicy):
    def __init__(
        self,
        *,
        groot_policy: GrootSimPolicy,
        signal_group: dist.ProcessGroup,
        image_height: int,
        image_width: int,
        output_dir: str | None = None,
        max_chunk_size: int | None = None,
    ) -> None:
        super().__init__(
            groot_policy=groot_policy,
            signal_group=signal_group,
            image_height=image_height,
            image_width=image_width,
            output_dir=output_dir,
            max_chunk_size=max_chunk_size,
            use_rtc=False,
        )
        self._sessions: dict[str, SessionState] = {}

    def _get_session(self, session_id: str) -> SessionState:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionState(session_id=session_id)
        return self._sessions[session_id]

    def reset(self, reset_info: dict[str, Any]) -> None:
        session_id_raw = reset_info.get("session_id")
        prompt = _normalize_prompt(reset_info.get("prompt", reset_info.get("annotation.task", "")))
        if bool(reset_info.get("reset_all", False)) or session_id_raw is None:
            self._sessions.clear()
            logger.info("microbatch reset all sessions")
        else:
            session_id = _session_id_value(session_id_raw)
            self._get_session(session_id).clear(prompt=prompt)
            logger.info("microbatch reset session_id=%s prompt=%r", session_id, prompt)
        _reset_model_temporal_state(self._policy)

    def _extract_state(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        left_joint_key = "state.left_joint_pos"
        left_gripper_key = "state.left_gripper_pos"
        right_joint_key = "state.right_joint_pos"
        right_gripper_key = "state.right_gripper_pos"
        if all(key in obs for key in (left_joint_key, left_gripper_key, right_joint_key, right_gripper_key)):
            return {
                left_joint_key: _ensure_state_array(obs[left_joint_key], left_joint_key, 6),
                left_gripper_key: _ensure_state_array(obs[left_gripper_key], left_gripper_key, 1),
                right_joint_key: _ensure_state_array(obs[right_joint_key], right_joint_key, 6),
                right_gripper_key: _ensure_state_array(obs[right_gripper_key], right_gripper_key, 1),
            }
        return super()._extract_state(obs)

    def _extract_video_single(self, obs: dict[str, Any], model_key: str, raw_key: str) -> np.ndarray:
        if model_key in obs:
            data = obs[model_key]
        elif raw_key in obs:
            data = obs[raw_key]
        else:
            raise KeyError(f"Missing required image input: '{model_key}' or '{raw_key}'")
        arr = _normalize_image_array(data, model_key if model_key in obs else raw_key)
        if arr.ndim == 5:
            raise ValueError(
                "robotwin_microbatch_server expects one session per websocket infer request; "
                f"{model_key} received batched shape {arr.shape}"
            )
        return np.ascontiguousarray(_resize_frames_to_resolution(arr, self._image_height, self._image_width))

    def _append_video(self, session: SessionState, key: str, video: np.ndarray) -> None:
        if video.ndim == 3:
            session.frame_buffers[key].append(video)
        elif video.ndim == 4:
            session.frame_buffers[key].extend(list(video))
        else:
            raise ValueError(f"{key} expected 3D or 4D video for one session, got {video.shape}")
        if len(session.frame_buffers[key]) > self.FRAMES_PER_CHUNK:
            del session.frame_buffers[key][:-self.FRAMES_PER_CHUNK]

    def _window_from_session(self, session: SessionState, key: str, num_frames: int) -> np.ndarray:
        buffer = session.frame_buffers[key]
        if not buffer:
            raise ValueError(f"No frames buffered for session={session.session_id} key={key}")
        frames = buffer[-num_frames:] if len(buffer) >= num_frames else list(buffer)
        while len(frames) < num_frames:
            frames.insert(0, buffer[0])
        return np.ascontiguousarray(np.stack(frames, axis=0))

    def _convert_one(self, obs: dict[str, Any]) -> dict[str, Any]:
        session_id = _session_id_value(obs.get("session_id"))
        session = self._get_session(session_id)
        prompt = _normalize_prompt(obs.get("prompt", obs.get("annotation.task", session.prompt)))
        if prompt and session.prompt and prompt != session.prompt:
            logger.warning(
                "prompt changed without reset for session_id=%s; clearing per-session history",
                session_id,
            )
            session.clear(prompt=prompt)
        elif prompt and not session.prompt:
            session.prompt = prompt
        prompt = session.prompt or prompt

        videos = {
            "video.cam_high": self._extract_video_single(obs, "video.cam_high", "observation.images.cam_high"),
            "video.cam_left": self._extract_video_single(obs, "video.cam_left", "observation.images.cam_left"),
            "video.cam_right": self._extract_video_single(obs, "video.cam_right", "observation.images.cam_right"),
        }
        for key, video in videos.items():
            self._append_video(session, key, video)

        num_frames = 1 if session.is_first_call else self.FRAMES_PER_CHUNK
        converted = {key: self._window_from_session(session, key, num_frames) for key in self.VIDEO_KEYS}
        converted.update(self._extract_state(obs))
        converted["annotation.task"] = prompt
        converted["session_id"] = session_id
        session.is_first_call = False
        session.infer_count += 1
        return converted

    def _merge_converted(self, converted_items: list[dict[str, Any]]) -> dict[str, Any]:
        if not converted_items:
            raise ValueError("Cannot run an empty microbatch")
        batch: dict[str, Any] = {}
        for key in self.VIDEO_KEYS:
            max_frames = max(int(item[key].shape[0]) for item in converted_items)
            batch[key] = np.ascontiguousarray(
                np.stack([_pad_front_array(np.asarray(item[key]), max_frames) for item in converted_items], axis=0)
            )
        for key in (
            "state.left_joint_pos",
            "state.left_gripper_pos",
            "state.right_joint_pos",
            "state.right_gripper_pos",
        ):
            max_rows = max(int(np.asarray(item[key]).shape[0]) for item in converted_items)
            batch[key] = np.ascontiguousarray(
                np.stack([_pad_front_array(np.asarray(item[key]), max_rows) for item in converted_items], axis=0)
            )
        batch["annotation.task"] = np.asarray([str(item.get("annotation.task", "")) for item in converted_items])
        return batch

    def infer_batch(self, observations: list[dict[str, Any]]) -> list[np.ndarray]:
        converted_items = [self._convert_one(obs) for obs in observations]
        converted_obs = self._merge_converted(converted_items)
        batch_size = len(converted_items)

        # Correctness-first v1: model temporal state is global today, so every
        # mixed-session microbatch runs from explicit per-session frame windows.
        _reset_model_temporal_state(self._policy)
        worker_obs = self._set_reset_flag(converted_obs, True)
        signal_tensor = torch.tensor([CONTINUE_SIGNAL], dtype=torch.int32, device="cpu")
        dist.broadcast(signal_tensor, src=0, group=self._signal_group)
        self._broadcast_payload_to_workers({"obs": worker_obs, "model_kwargs": {}})

        batch = Batch(obs=converted_obs)
        dist.barrier()
        with torch.no_grad():
            result_batch, _video_pred = self._policy.lazy_joint_forward_causal(batch)
        dist.barrier()

        action_dict = _extract_action_dict(result_batch.act)
        action = _normalize_action_batch(self._convert_action(action_dict), batch_size)
        logger.info(
            "microbatch infer batch_size=%d sessions=%s prompts=%s action_shape=%s",
            batch_size,
            [item.get("session_id", "") for item in converted_items],
            [item.get("annotation.task", "") for item in converted_items],
            tuple(action.shape),
        )
        return [np.ascontiguousarray(action[index]) for index in range(batch_size)]

    def flush_pending_video(self) -> None:
        self._sessions.clear()
        _reset_model_temporal_state(self._policy)


class RobotwinMicrobatchWebsocketServer:
    def __init__(
        self,
        *,
        policy: RobotwinMicrobatchPolicy,
        server_config: PolicyServerConfig,
        host: str,
        port: int,
        open_timeout: float | None,
        metadata: dict[str, Any],
        microbatch_wait_ms: float,
        microbatch_max_size: int,
    ) -> None:
        self.policy = policy
        self.server_config = server_config
        self.host = host
        self.port = int(port)
        self.open_timeout = None if open_timeout is None or open_timeout <= 0 else float(open_timeout)
        self.metadata = dict(metadata)
        self.microbatch_wait_s = max(float(microbatch_wait_ms), 0.0) / 1000.0
        self.microbatch_max_size = max(int(microbatch_max_size), 1)
        self.queue: asyncio.Queue[PendingRequest] = asyncio.Queue()
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        scheduler = asyncio.create_task(self._microbatch_loop())
        try:
            async with websockets.asyncio.server.serve(
                self._handler,
                self.host,
                self.port,
                compression=None,
                max_size=None,
                open_timeout=self.open_timeout,
                ping_interval=None,
            ) as server:
                await server.serve_forever()
        finally:
            scheduler.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await scheduler

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection) -> None:
        logger.info("Connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        metadata = dataclasses.asdict(self.server_config)
        metadata.update(self.metadata)
        await websocket.send(packer.pack(metadata))

        while True:
            try:
                obs = msgpack_numpy.unpackb(await websocket.recv())
                endpoint = obs["endpoint"]
                del obs["endpoint"]
                if endpoint == "reset":
                    self.policy.reset(obs)
                    await websocket.send("reset successful")
                    continue
                if endpoint != "infer":
                    raise ValueError(f"Unknown endpoint: {endpoint!r}")
                loop = asyncio.get_running_loop()
                future: asyncio.Future = loop.create_future()
                request = PendingRequest(
                    obs=obs,
                    future=future,
                    enqueued_at=time.perf_counter(),
                    session_id=_session_id_value(obs.get("session_id")),
                )
                await self.queue.put(request)
                response = await future
                await websocket.send(packer.pack(response))
            except websockets.ConnectionClosed:
                logger.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                import traceback

                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    async def _microbatch_loop(self) -> None:
        while True:
            first = await self.queue.get()
            pending = [first]
            deadline = time.perf_counter() + self.microbatch_wait_s
            while len(pending) < self.microbatch_max_size:
                timeout = max(deadline - time.perf_counter(), 0.0)
                if timeout <= 0:
                    break
                try:
                    pending.append(await asyncio.wait_for(self.queue.get(), timeout=timeout))
                except asyncio.TimeoutError:
                    break

            batch_start = time.perf_counter()
            queue_depth = self.queue.qsize()
            try:
                actions = self.policy.infer_batch([item.obs for item in pending])
                policy_infer_time = time.perf_counter() - batch_start
                cuda_allocated = 0.0
                cuda_reserved = 0.0
                if torch.cuda.is_available():
                    cuda_allocated = torch.cuda.memory_allocated() / 1_000_000.0
                    cuda_reserved = torch.cuda.memory_reserved() / 1_000_000.0
                for item, action in zip(pending, actions, strict=True):
                    server_timing = {
                        "policy_infer_time": policy_infer_time,
                        "T_policy_infer": policy_infer_time,
                        "T_microbatch_wait": max(batch_start - item.enqueued_at, 0.0),
                        "microbatch_size": len(pending),
                        "session_queue_depth": queue_depth,
                        "server_vram_allocated_mb": cuda_allocated,
                        "server_vram_reserved_mb": cuda_reserved,
                    }
                    if not item.future.done():
                        item.future.set_result({"actions": action, "server_timing": server_timing})
            except Exception as exc:
                for item in pending:
                    if not item.future.done():
                        item.future.set_exception(exc)


def _build_output_dir(output_root: str | None, model_path: str, index: int) -> str:
    checkpoint_path = Path(model_path.rstrip("/"))
    output_root_path = Path(output_root) if output_root is not None else checkpoint_path.parent
    timestamp = datetime.datetime.now().strftime("%Y%m%d")
    output_dir = output_root_path / f"robotwin_microbatch_server_{timestamp}_{index}" / checkpoint_path.name
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir)


def main(args: Args) -> None:
    process_started_at = time.perf_counter()
    os.environ["ENABLE_DIT_CACHE"] = "true" if args.enable_dit_cache else "false"
    os.environ.setdefault("ATTENTION_BACKEND", "TE")
    if hasattr(torch._dynamo.config, "recompile_limit"):
        torch._dynamo.config.recompile_limit = 800

    logging.basicConfig(level=logging.INFO, force=True)
    if DEFAULT_TORCH_COMPILE_BACKEND is not None:
        logger.info("Using torch.compile backend=%s for this entrypoint", DEFAULT_TORCH_COMPILE_BACKEND)
    if args.microbatch_max_size <= 0:
        raise ValueError("--microbatch-max-size must be positive")

    device_mesh = init_mesh()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    timeout_delta = datetime.timedelta(seconds=args.timeout_seconds)
    signal_group = dist.new_group(backend="gloo", timeout=timeout_delta)
    logger.info("Rank %d/%d initialized signal_group", rank, world_size)

    embodiment_tag = EmbodimentTag.ALOHA_X5LITE_BIMANUAL
    logger.info("Loading DreamZero checkpoint from %s with embodiment=%s", args.model_path, embodiment_tag.value)
    model_load_start = time.perf_counter()
    policy = GrootSimPolicy(
        embodiment_tag=embodiment_tag,
        model_path=args.model_path,
        tokenizer_path_override=args.tokenizer_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        device_mesh=device_mesh,
    )
    load_model_time = time.perf_counter() - model_load_start
    logger.info("Loaded DreamZero checkpoint in %.3fs", load_model_time)

    if args.image_height is not None and args.image_width is not None:
        image_height, image_width = int(args.image_height), int(args.image_width)
        logger.info("Using CLI image resolution %dx%d", image_height, image_width)
    else:
        image_height, image_width = _get_expected_video_resolution(policy)
        logger.info("Using checkpoint image resolution %dx%d", image_height, image_width)

    output_dir = (
        _build_output_dir(args.output_root, args.model_path, args.index)
        if rank == 0 and args.save_server_video
        else None
    )
    if output_dir:
        logger.info("Server video output requested, but microbatch v1 keeps only session-safe model outputs.")

    wrapper_policy = RobotwinMicrobatchPolicy(
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
        server_ready_time = time.perf_counter() - process_started_at
        server_timing = {
            "load_model_time": load_model_time,
            "server_ready_time": server_ready_time,
            "T_load_model": load_model_time,
            "T_server_ready": server_ready_time,
        }
        metadata = {
            "server_timing": server_timing,
            "microbatch": {
                "enabled": True,
                "microbatch_wait_ms": float(args.microbatch_wait_ms),
                "microbatch_max_size": int(args.microbatch_max_size),
                "rtc_enabled": False,
                "cache_strategy": "correctness_first_reset_temporal_state_per_batch",
            },
        }
        logger.info(
            "Serving RoboTwin microbatch websocket server on ws://%s:%d "
            "(T_load_model=%.3fs T_server_ready=%.3fs wait_ms=%.3f max_size=%d)",
            args.host,
            args.port,
            load_model_time,
            server_ready_time,
            args.microbatch_wait_ms,
            args.microbatch_max_size,
        )
        server = RobotwinMicrobatchWebsocketServer(
            policy=wrapper_policy,
            server_config=server_config,
            host=args.host,
            port=args.port,
            open_timeout=args.handshake_timeout_seconds,
            metadata=metadata,
            microbatch_wait_ms=args.microbatch_wait_ms,
            microbatch_max_size=args.microbatch_max_size,
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
