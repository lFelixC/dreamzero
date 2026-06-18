"""Legacy native DreamZero AR websocket server.

This is the **native** AR_droid/DROID wire protocol server. It speaks
DreamZero's original protocol directly:

* on connect it sends a plain ``policy_metadata`` dict
  (``embodiment`` / ``model_name`` / ``model_path`` / ``architecture``) -- NOT a
  RoboArena ``PolicyServerConfig``,
* clients send observations using the native DROID keys:
  ``video.exterior_image_1_left`` / ``video.exterior_image_2_left`` /
  ``video.wrist_image_left`` / ``state.joint_position`` /
  ``state.gripper_position`` / ``annotation.language.action_text``,
* the server returns the ``action.*`` dict straight from the policy
  (no ``{"actions": ...}`` wrapping, no ``endpoint`` field).

It is single-session (no per-session_id state isolation, no server-side
batching). For the RoboArena/RoboLab protocol (``observation/*`` keys,
``session_id``, ``supports_parallel_sessions`` + server-side batching), use
``socket_test_robolab_AR.py`` instead. The two servers are NOT wire-compatible
and must not be pointed at the wrong client.

History: commit 91661d6 had this as the default entrypoint with an optional
``--roboarena-server`` branch; ac75342 flipped the default to the RoboArena
server, overwriting the native semantics. This file restores the native
entrypoint. All shared logic (model loading, distributed worker, architecture
override, RTC, session store, native websocket server) lives in
``eval_utils.server_common`` so the implementation is not duplicated.

Usage:
    torchrun --standalone --nproc_per_node=N socket_test_optimized_AR.py \\
        --host 0.0.0.0 --port 8000 --model-path /path/to/checkpoint \\
        --enable-dit-cache
"""

import asyncio
import dataclasses
import logging
from typing import Literal

import torch.distributed as dist
import tyro

from eval_utils.server_common import (
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
    use_rtc: bool = False
    rtc_execution_horizon: int = 10
    rtc_max_guidance_weight: float = 10.0
    rtc_prefix_attention_schedule: str = "EXP"
    rtc_guidance_max_steps: int = 4
    rtc_guidance_step_stride: int = 1
    # NOTE: no batch_max_size / batch_timeout_ms here -- the native server is
    # single-session and does not aggregate requests. Use the RoboLab server
    # (socket_test_robolab_AR.py) for server-side batching.


def main(args: Args) -> None:
    loaded: LoadedPolicy = load_policy(args)
    rank = dist.get_rank()

    embodiment_tag = "oxe_droid"
    # Native protocol sends only model metadata -- NOT a RoboArena
    # PolicyServerConfig. Clients that expect PolicyServerConfig must talk to
    # socket_test_robolab_AR.py instead.
    policy_metadata = {
        "embodiment": embodiment_tag,
        "model_name": "dreamzero",
        "model_path": args.model_path,
        "architecture": loaded.effective_architecture,
    }

    if rank == 0:
        logging.info(
            "Serving DreamZero native AR %s websocket server on ws://%s:%d "
            "(native video.*/state.*/annotation.* protocol)",
            loaded.effective_architecture,
            args.host,
            args.port,
        )
        server = WebsocketPolicyServer(
            policy=loaded.policy,
            host=args.host,
            port=args.port,
            metadata=policy_metadata,
            output_dir=loaded.output_dir,
            signal_group=loaded.signal_group,
        )
        server.serve_forever()
    else:
        # Non-rank-0 processes run the distributed worker loop. The native
        # WebsocketPolicyServer owns the worker loop and it is protocol-agnostic
        # (it only drives the distributed forward pass), so the RoboLab server
        # reuses the exact same loop.
        server = WebsocketPolicyServer(
            policy=loaded.policy,
            host=args.host,
            port=args.port,
            metadata=policy_metadata,
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
