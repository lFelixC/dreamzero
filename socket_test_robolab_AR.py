"""RoboLab / RoboArena DreamZero AR websocket server.

This is the **RoboArena-protocol** AR server used by RoboLab evaluation. It:

* wraps the DreamZero policy in :class:`ARDroidRoboarenaPolicy` (which converts
  ``observation/*`` frames into the DROID temporal cadence),
* on connect sends a full :class:`PolicyServerConfig` (so the RoboLab client /
  ``run_parallel.py`` preflight can read ``supports_parallel_sessions`` /
  ``supports_batching`` / ``n_external_cameras`` / ``image_resolution`` ...),
* receives observations using the RoboArena keys
  (``observation/exterior_image_0_left`` / ``observation/exterior_image_1_left``
  / ``observation/wrist_image_left`` / ``observation/joint_position`` /
  ``observation/gripper_position`` / ``prompt`` / ``session_id``) plus an
  ``endpoint`` field,
* returns ``{"actions": (N, 8)}``.

For the legacy native ``video.*`` / ``state.*`` / ``annotation.*`` protocol, use
``socket_test_optimized_AR.py`` instead. The two servers are NOT wire-compatible
and must not be pointed at the wrong client.

History: ac75342 made the RoboArena server the default inside
``socket_test_optimized_AR.py``; later commits (778c406, f8d3b91, 5cabc25,
a054ecb, f05ff92) layered per-session state, batching, reset ordering, and
RoboLab docs on top of it. This file splits that RoboArena entrypoint back out
into its own script so the legacy native server can be restored without losing
the RoboLab functionality. Shared logic lives in ``eval_utils.server_common``.

Usage:
    torchrun --standalone --nproc_per_node=N socket_test_robolab_AR.py \\
        --host 127.0.0.1 --port 8000 \\
        --model-path /path/to/checkpoint --enable-dit-cache \\
        --batch-max-size 8 --batch-timeout-ms 8.0
"""

import asyncio
import dataclasses
import logging
from typing import Literal

import torch.distributed as dist
import tyro

from eval_utils.policy_server import PolicyServerConfig
from eval_utils.policy_server import WebsocketPolicyServer as RoboarenaServer
from eval_utils.server_common import (
    ARDroidRoboarenaPolicy,
    LoadedPolicy,
    WebsocketPolicyServer,
    load_policy,
)
from eval_utils.torch_compile_backend import configure_torch_compile_backend

DEFAULT_TORCH_COMPILE_BACKEND = configure_torch_compile_backend(default_backend="cudagraphs")

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


def main(args: Args) -> None:
    loaded: LoadedPolicy = load_policy(args)
    rank = dist.get_rank()

    # Wrap the DreamZero policy so it speaks the RoboArena protocol: convert
    # observation/* frames into the DROID video block cadence, keep per-session
    # temporal/cache state, and return (N, 8) action arrays.
    wrapper_policy = ARDroidRoboarenaPolicy(
        groot_policy=loaded.policy,
        signal_group=loaded.signal_group,
        image_height=loaded.image_height,
        image_width=loaded.image_width,
        output_dir=loaded.output_dir,
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

    # RoboArena/RoboLab protocol config sent to every client on connect.
    # n_external_cameras=2 because DreamZero DROID inputs always have two
    # exterior slots; the RoboLab default cam2_source='right' now fills the
    # second slot with the real over-shoulder-right camera instead of a black
    # frame (run.py registers WRIST_LEFT_RIGHT by default).
    server_config = PolicyServerConfig(
        image_resolution=(loaded.image_height, loaded.image_width),
        needs_wrist_camera=True,
        n_external_cameras=2,
        needs_stereo_camera=False,
        needs_session_id=True,  # Track session to reset state for new clients
        action_space="joint_position",
        wam_architecture=loaded.loaded_architecture or loaded.effective_architecture,
        mot_inference_video_mode=loaded.mot_inference_video_mode,
        cache_order_sensitive=loaded.cache_order_sensitive,
        supports_rtc=not loaded.cache_order_sensitive,
        supports_async_prefetch=not loaded.cache_order_sensitive,
        supports_parallel_sessions=True,
        supports_batching=loaded.supports_batching,
        max_batch_size=args.batch_max_size,
        batch_timeout_ms=args.batch_timeout_ms,
    )

    if rank == 0:
        logging.info("Using roboarena/robolab policy server interface")
        logging.info(f"Server config: {server_config}")
        logging.info(
            "Serving DreamZero RoboLab AR %s websocket server on ws://%s:%d "
            "(RoboArena observation/* + session_id protocol)",
            loaded.effective_architecture,
            args.host,
            args.port,
        )
        roboarena_server = RoboarenaServer(
            policy=wrapper_policy,
            server_config=server_config,
            host=args.host,
            port=args.port,
            open_timeout=args.handshake_timeout_seconds,
        )
        roboarena_server.serve_forever()
    else:
        # Non-rank-0 processes run the distributed worker loop. The RoboArena
        # server only runs on rank 0, so reuse the native WebsocketPolicyServer's
        # protocol-agnostic worker loop (it only drives the distributed forward
        # pass + per-session restore/capture on the underlying policy).
        server = WebsocketPolicyServer(
            policy=loaded.policy,
            host=args.host,
            port=args.port,
            metadata={},
            output_dir=loaded.output_dir,
            signal_group=loaded.signal_group,
        )
        asyncio.run(server._worker_loop())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    if DEFAULT_TORCH_COMPILE_BACKEND is not None:
        logger.info("Using torch.compile backend=%s for this entrypoint", DEFAULT_TORCH_COMPILE_BACKEND)
    args = tyro.cli(Args)
    main(args)
