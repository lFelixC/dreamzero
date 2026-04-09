#!/usr/bin/env python3
"""Thin DreamZero websocket server wrapper for video evaluation."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

from einops import rearrange
import imageio
import torch
from torch.distributed.device_mesh import init_device_mesh

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_utils.policy_server import PolicyServerConfig, WebsocketPolicyServer
from eval_utils.serve_dreamzero_wan22 import (  # noqa: E402
    DreamZeroWan225BPolicy,
    _get_expected_video_resolution,
    _maybe_init_distributed,
)
from groot.vla.data.schema import EmbodimentTag  # noqa: E402
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy  # noqa: E402

from common import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    DEFAULT_PORT,
    VIDEO_RESULTS_ROOT,
    append_run_log,
    ensure_dir,
)


LOGGER = logging.getLogger(__name__)
DEFAULT_DIST_PORT = 29500
DIST_PORT_OFFSET = 10000


def resolve_dist_port(websocket_port: int, dist_port_override: int | None) -> int:
    if dist_port_override is not None:
        return dist_port_override
    candidate = websocket_port + DIST_PORT_OFFSET
    if 0 < candidate <= 65535 and candidate != websocket_port:
        return candidate
    fallback = websocket_port + 1
    if 0 < fallback <= 65535 and fallback != websocket_port:
        return fallback
    return DEFAULT_DIST_PORT


class SessionAwareDreamZeroPolicy(DreamZeroWan225BPolicy):
    """Save predicted videos into session-specific result directories."""

    def __init__(self, *args, results_root: Path, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._results_root = ensure_dir(results_root)

    def _session_run_dir(self) -> Path:
        session_id = self._current_session_id or "unspecified_session"
        return ensure_dir(self._results_root / session_id)

    def _session_log_path(self) -> Path:
        return self._session_run_dir() / "run.log"

    def infer(self, obs: dict) -> torch.Tensor:
        session_id = obs.get("session_id", "unspecified_session")
        ensure_dir(self._results_root / str(session_id))
        append_run_log(
            self._results_root / str(session_id) / "run.log",
            (
                f"[server] infer session_id={session_id} "
                f"prompt={obs.get('prompt', '')!r} "
                f"keys={sorted(obs.keys())}"
            ),
        )
        action = super().infer(obs)
        append_run_log(
            self._results_root / str(session_id) / "run.log",
            f"[server] action_shape={tuple(action.shape)}",
        )
        return action

    def _save_predicted_video(self) -> None:
        if not self._video_pred_latents:
            append_run_log(self._session_log_path(), "[server] no predicted latents to save")
            return

        output_path = self._session_run_dir() / "pred.mp4"
        try:
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
            frames = (
                ((frames.float() + 1) * 127.5)
                .clip(0, 255)
                .cpu()
                .numpy()
                .astype("uint8")
            )
            imageio.mimsave(str(output_path), list(frames), fps=5, codec="libx264")
            append_run_log(
                self._session_log_path(),
                f"[server] saved_pred_video={output_path} frames={len(frames)}",
            )
        except Exception as exc:  # pragma: no cover - best effort logging
            append_run_log(self._session_log_path(), f"[server] failed_to_save_pred: {exc}")
            raise

    def reset(self, reset_info: dict) -> None:
        append_run_log(
            self._session_log_path(),
            f"[server] reset requested keys={sorted(reset_info.keys())}",
        )
        super().reset(reset_info)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Checkpoint path.")
    parser.add_argument("--host", default="0.0.0.0", help="Server host.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server port.")
    parser.add_argument(
        "--dist-port",
        type=int,
        default=None,
        help="Optional torch.distributed master port. Defaults to a value derived from --port.",
    )
    parser.add_argument(
        "--results-root",
        default=str(VIDEO_RESULTS_ROOT),
        help="Root directory for per-session outputs.",
    )
    parser.add_argument(
        "--embodiment-tag",
        default="oxe_droid",
        help="Embodiment tag passed to GrootSimPolicy.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Optional tokenizer override for the checkpoint.",
    )
    parser.add_argument("--image-height", type=int, default=None, help="Optional image height override.")
    parser.add_argument("--image-width", type=int, default=None, help="Optional image width override.")
    parser.add_argument(
        "--handshake-timeout-seconds",
        type=float,
        default=0.0,
        help="Websocket opening-handshake timeout in seconds. Use 0 or a negative value to disable it.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)

    results_root = ensure_dir(Path(args.results_root))
    dist_port = resolve_dist_port(args.port, args.dist_port)
    LOGGER.info("Using torch.distributed master port %d", dist_port)
    _maybe_init_distributed(master_port=dist_port)
    device_mesh = init_device_mesh("cuda", mesh_shape=(1,), mesh_dim_names=("ip",))

    LOGGER.info("Loading DreamZero checkpoint from %s", args.model_path)
    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag(args.embodiment_tag),
        model_path=args.model_path,
        tokenizer_path_override=args.tokenizer_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        device_mesh=device_mesh,
    )
    if args.image_height is not None and args.image_width is not None:
        image_height, image_width = args.image_height, args.image_width
        LOGGER.info("Using CLI image resolution %dx%d", image_height, image_width)
    else:
        image_height, image_width = _get_expected_video_resolution(policy)
        LOGGER.info("Using checkpoint image resolution %dx%d", image_height, image_width)

    wrapper = SessionAwareDreamZeroPolicy(
        groot_policy=policy,
        image_height=image_height,
        image_width=image_width,
        embodiment_tag=args.embodiment_tag,
        save_video_pred=True,
        video_output_dir=str(results_root),
        results_root=results_root,
    )
    server_config = PolicyServerConfig(
        image_resolution=(image_height, image_width),
        needs_wrist_camera=True,
        n_external_cameras=2,
        needs_stereo_camera=False,
        needs_session_id=True,
        action_space="joint_position",
    )
    LOGGER.info("Serving on ws://%s:%s", args.host, args.port)
    server = WebsocketPolicyServer(
        policy=wrapper,
        server_config=server_config,
        host=args.host,
        port=args.port,
        open_timeout=args.handshake_timeout_seconds,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
