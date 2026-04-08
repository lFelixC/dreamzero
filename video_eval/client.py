#!/usr/bin/env python3
"""DreamZero video-eval client for debug-image and manifest-based inference."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_utils.policy_client import WebsocketClientPolicy  # noqa: E402
from eval_utils.policy_server import PolicyServerConfig  # noqa: E402

from common import (  # noqa: E402
    CAMERA_FILES,
    DEFAULT_DEBUG_IMAGE_DIR,
    DEFAULT_DEBUG_PROMPT,
    DEFAULT_MODEL_PATH,
    DEFAULT_PORT,
    MANIFEST_VIDEO_KEYS,
    VIDEO_RESULTS_ROOT,
    append_run_log,
    build_chunk_schedule,
    compose_droid_views,
    ensure_dir,
    load_manifest_entry,
    read_video_frames,
    resize_frames,
    save_video,
    utc_timestamp,
    write_json,
)


LOGGER = logging.getLogger(__name__)
PROXY_ENV_KEYS = (
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "all_proxy",
    "no_proxy",
    "NO_PROXY",
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Server hostname.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server port.")
    parser.add_argument(
        "--input-mode",
        choices=("debug_image", "manifest"),
        default="debug_image",
        help="Input source for inference.",
    )
    parser.add_argument(
        "--debug-image-dir",
        default=str(DEFAULT_DEBUG_IMAGE_DIR),
        help="Directory containing debug_image mp4 files.",
    )
    parser.add_argument(
        "--manifest-path",
        default=None,
        help="Manifest JSONL created by convert_roboset_to_dreamzero.py.",
    )
    parser.add_argument("--sample-id", default=None, help="Specific manifest sample to run.")
    parser.add_argument(
        "--results-root",
        default=str(VIDEO_RESULTS_ROOT),
        help="Root directory for timestamped outputs.",
    )
    parser.add_argument("--run-id", default=None, help="Optional run directory name.")
    parser.add_argument("--prompt", default=None, help="Optional prompt override.")
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=None,
        help="Maximum number of 4-frame chunks to send after the initial frame.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Checkpoint path recorded in metadata.json.",
    )
    parser.add_argument(
        "--wait-timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for pred.mp4 after reset.",
    )
    return parser


def load_debug_camera_frames(debug_image_dir: Path) -> dict[str, np.ndarray]:
    frames: dict[str, np.ndarray] = {}
    for cam_key, filename in CAMERA_FILES.items():
        frames[cam_key] = read_video_frames(debug_image_dir / filename)
    return frames


def resize_camera_frames(
    camera_frames: dict[str, np.ndarray],
    image_resolution: tuple[int, int] | list[int] | None,
) -> dict[str, np.ndarray]:
    if image_resolution is None:
        return camera_frames
    target_height, target_width = image_resolution
    return {
        cam_key: resize_frames(frames, target_height, target_width)
        for cam_key, frames in camera_frames.items()
    }


def load_manifest_camera_frames(entry: dict[str, Any]) -> dict[str, np.ndarray]:
    video_paths = entry.get("video_paths")
    if not isinstance(video_paths, dict):
        raise KeyError(f"Manifest row is missing video_paths: {entry.get('sample_id')}")
    return {
        MANIFEST_VIDEO_KEYS[source_key]: read_video_frames(Path(source_path))
        for source_key, source_path in video_paths.items()
    }


def read_anchor_state(parquet_path: Path, anchor_frame_index: int) -> tuple[np.ndarray, np.ndarray]:
    import pyarrow.parquet as pq

    table = pq.read_table(str(parquet_path), columns=["observation.state"])
    state = np.asarray(table.column("observation.state")[anchor_frame_index].as_py(), dtype=np.float32)
    if state.shape[0] < 14:
        raise ValueError(f"Expected DROID-style observation.state length 14, got {state.shape[0]}")
    gripper = state[6:7]
    joints = state[7:14]
    return joints, gripper


def make_observation(
    camera_frames: dict[str, np.ndarray],
    frame_indices: list[int],
    joint_position: np.ndarray,
    gripper_position: np.ndarray,
    prompt: str,
    session_id: str,
) -> dict[str, Any]:
    obs: dict[str, Any] = {}
    for cam_key, frames in camera_frames.items():
        selected = frames[frame_indices]
        obs[cam_key] = selected[0] if len(frame_indices) == 1 else selected
    obs["observation/joint_position"] = joint_position.astype(np.float32)
    obs["observation/cartesian_position"] = np.zeros(6, dtype=np.float32)
    obs["observation/gripper_position"] = gripper_position.astype(np.float32)
    obs["prompt"] = prompt
    obs["session_id"] = session_id
    return obs


def summarize_action(action: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(action.shape),
        "min": float(action.min()),
        "max": float(action.max()),
    }


def wait_for_file(path: Path, timeout_seconds: float) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if path.exists() and path.stat().st_size > 0:
            return
        time.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for {path}")


def clear_proxy_environment() -> None:
    for key in PROXY_ENV_KEYS:
        os.environ.pop(key, None)


def fetch_server_metadata(host: str, port: int, timeout_seconds: float = 120.0) -> dict[str, Any]:
    clear_proxy_environment()
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    next_status_log = 0.0
    while time.time() < deadline:
        try:
            client = WebsocketClientPolicy(host=host, port=port, log_wait=False)
            try:
                return client.get_server_metadata()
            finally:
                client.close()
        except Exception as exc:  # pragma: no cover - network timing dependent
            last_error = exc
            now = time.time()
            if now >= next_status_log:
                logging.info("server still loading checkpoint: waiting for ws://%s:%s", host, port)
                next_status_log = now + 30.0
            time.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for server metadata from ws://{host}:{port}: {last_error}")


def create_gt_video_from_entry(
    run_dir: Path,
    entry: dict[str, Any],
    camera_frames: dict[str, np.ndarray],
    image_resolution: tuple[int, int] | list[int] | None,
    log_path: Path | None = None,
) -> Path:
    if image_resolution is None:
        raise ValueError("Server did not provide image_resolution; cannot build gt.mp4")
    target_height, target_width = image_resolution
    left = resize_frames(camera_frames["observation/exterior_image_0_left"], target_height, target_width)
    right = resize_frames(camera_frames["observation/exterior_image_1_left"], target_height, target_width)
    wrist = resize_frames(camera_frames["observation/wrist_image_left"], target_height, target_width)
    gt_frames = compose_droid_views(left, right, wrist)
    gt_path = run_dir / "gt.mp4"
    save_video(gt_frames, gt_path, fps=int(entry.get("fps", 15)))
    append_run_log(log_path or (run_dir / "run.log"), f"[client] saved_gt_video={gt_path}")
    return gt_path


def prepare_manifest_request(
    entry: dict[str, Any],
    server_config: PolicyServerConfig,
    prompt_override: str | None = None,
    num_chunks: int | None = None,
) -> tuple[str, str, dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, Any]]:
    prompt_variants = entry.get("prompt_variants") or [DEFAULT_DEBUG_PROMPT]
    prompt = prompt_override or entry.get("prompt") or prompt_variants[0]
    sample_id = str(entry["sample_id"])
    camera_frames = load_manifest_camera_frames(entry)
    joint_position, gripper_position = read_anchor_state(
        Path(entry["parquet_path"]),
        int(entry.get("anchor_frame_index", 23)),
    )
    chunk_frame_indices = entry.get("chunk_frame_indices") or build_chunk_schedule(
        min(frames.shape[0] for frames in camera_frames.values()),
        num_chunks,
    )
    if num_chunks is not None:
        chunk_frame_indices = chunk_frame_indices[:num_chunks]
    request_payload = {
        "input_mode": "manifest",
        "sample_id": sample_id,
        "prompt": prompt,
        "entry": entry,
        "initial_frame_indices": entry.get("initial_frame_indices", [0]),
        "chunk_frame_indices": chunk_frame_indices,
        "server_metadata": {
            "image_resolution": list(server_config.image_resolution),
            "needs_wrist_camera": server_config.needs_wrist_camera,
            "n_external_cameras": server_config.n_external_cameras,
            "needs_stereo_camera": server_config.needs_stereo_camera,
            "needs_session_id": server_config.needs_session_id,
            "action_space": server_config.action_space,
        },
    }
    return prompt, sample_id, camera_frames, joint_position, gripper_position, request_payload


def ensure_gt_video_for_entry(
    run_dir: Path,
    entry: dict[str, Any],
    image_resolution: tuple[int, int] | list[int] | None,
    log_path: Path | None = None,
) -> Path:
    gt_path = run_dir / "gt.mp4"
    if gt_path.exists() and gt_path.stat().st_size > 0:
        return gt_path
    camera_frames = load_manifest_camera_frames(entry)
    return create_gt_video_from_entry(
        run_dir=run_dir,
        entry=entry,
        camera_frames=camera_frames,
        image_resolution=image_resolution,
        log_path=log_path,
    )


def run_inference_session(
    run_dir: Path,
    host: str,
    port: int,
    checkpoint_path: str,
    input_mode: str,
    prompt_override: str | None = None,
    num_chunks: int | None = None,
    wait_timeout: float = 120.0,
    session_id: str | None = None,
    debug_image_dir: Path | None = None,
    manifest_path: Path | None = None,
    manifest_entry: dict[str, Any] | None = None,
    server_metadata: dict[str, Any] | None = None,
    cli_argv: list[str] | None = None,
) -> dict[str, Any]:
    run_dir = ensure_dir(run_dir)
    log_path = run_dir / "run.log"
    if cli_argv is not None:
        append_run_log(log_path, f"[client] argv={json.dumps(cli_argv)}")

    metadata = server_metadata or fetch_server_metadata(host=host, port=port, timeout_seconds=wait_timeout)
    server_config = PolicyServerConfig(**metadata)
    append_run_log(log_path, f"[client] server_metadata={json.dumps(metadata, sort_keys=True)}")

    clear_proxy_environment()
    client = WebsocketClientPolicy(host=host, port=port, log_wait=False)

    request_payload: dict[str, Any]
    prompt: str
    sample_id: str
    gt_path: Path | None = None

    if input_mode == "debug_image":
        debug_dir = debug_image_dir or DEFAULT_DEBUG_IMAGE_DIR
        prompt = prompt_override or DEFAULT_DEBUG_PROMPT
        sample_id = "debug_image"
        joint_position = np.zeros(7, dtype=np.float32)
        gripper_position = np.zeros(1, dtype=np.float32)
        camera_frames = resize_camera_frames(
            load_debug_camera_frames(debug_dir),
            server_config.image_resolution,
        )
        total_frames = min(frames.shape[0] for frames in camera_frames.values())
        chunks = build_chunk_schedule(total_frames, num_chunks)
        request_payload = {
            "input_mode": "debug_image",
            "sample_id": sample_id,
            "prompt": prompt,
            "debug_image_dir": str(debug_dir),
            "initial_frame_indices": [0],
            "chunk_frame_indices": chunks,
            "server_metadata": metadata,
        }
    elif input_mode == "manifest":
        entry = manifest_entry
        if entry is None:
            if manifest_path is None:
                raise ValueError("--manifest-path is required when --input-mode manifest")
            entry = load_manifest_entry(manifest_path, None)
        prompt, sample_id, camera_frames, joint_position, gripper_position, request_payload = prepare_manifest_request(
            entry=entry,
            server_config=server_config,
            prompt_override=prompt_override,
            num_chunks=num_chunks,
        )
        if manifest_path is not None:
            request_payload["manifest_path"] = str(manifest_path)
        gt_path = ensure_gt_video_for_entry(
            run_dir=run_dir,
            entry=entry,
            image_resolution=server_config.image_resolution,
            log_path=log_path,
        )
    else:
        raise ValueError(f"Unsupported input_mode: {input_mode}")

    current_session_id = session_id or run_dir.name
    request_payload["session_id"] = current_session_id
    write_json(run_dir / "request.json", request_payload)

    action_summaries: list[dict[str, Any]] = []
    initial_frame_indices = request_payload["initial_frame_indices"]
    chunk_frame_indices = request_payload["chunk_frame_indices"]

    append_run_log(log_path, f"[client] starting sample_id={sample_id} prompt={prompt!r}")
    all_calls = [initial_frame_indices] + list(chunk_frame_indices)
    for call_index, frame_indices in enumerate(all_calls):
        obs = make_observation(
            camera_frames=camera_frames,
            frame_indices=list(frame_indices),
            joint_position=joint_position,
            gripper_position=gripper_position,
            prompt=prompt,
            session_id=current_session_id,
        )
        start = time.time()
        actions = client.infer(obs)
        elapsed = time.time() - start
        summary = summarize_action(np.asarray(actions))
        summary["elapsed_seconds"] = elapsed
        summary["frame_indices"] = list(frame_indices)
        action_summaries.append(summary)
        append_run_log(
            log_path,
            (
                f"[client] infer_call={call_index} frame_indices={list(frame_indices)} "
                f"action_shape={summary['shape']} elapsed={elapsed:.2f}s "
                f"min={summary['min']:.4f} max={summary['max']:.4f}"
            ),
        )

    append_run_log(log_path, "[client] sending reset")
    client.reset({})

    pred_path = run_dir / "pred.mp4"
    wait_for_file(pred_path, timeout_seconds=wait_timeout)
    append_run_log(log_path, f"[client] detected_pred_video={pred_path}")

    metadata_payload = {
        "run_id": current_session_id,
        "sample_id": sample_id,
        "input_mode": input_mode,
        "checkpoint_path": checkpoint_path,
        "prompt": prompt,
        "task": request_payload.get("entry", {}).get("task"),
        "request_path": str(run_dir / "request.json"),
        "pred_video": str(pred_path),
        "gt_video": str(gt_path) if gt_path is not None else None,
        "server_metadata": metadata,
        "action_summaries": action_summaries,
    }
    write_json(run_dir / "metadata.json", metadata_payload)
    LOGGER.info("Finished run: %s", run_dir)
    return metadata_payload


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)

    run_id = args.run_id or utc_timestamp()
    run_dir = Path(args.results_root) / run_id
    manifest_entry = None
    manifest_path = Path(args.manifest_path) if args.manifest_path else None
    if args.input_mode == "manifest":
        if manifest_path is None:
            raise ValueError("--manifest-path is required when --input-mode manifest")
        manifest_entry = load_manifest_entry(manifest_path, args.sample_id)

    run_inference_session(
        run_dir=run_dir,
        host=args.host,
        port=args.port,
        checkpoint_path=args.checkpoint_path,
        input_mode=args.input_mode,
        prompt_override=args.prompt,
        num_chunks=args.num_chunks,
        wait_timeout=args.wait_timeout,
        session_id=run_id,
        debug_image_dir=Path(args.debug_image_dir),
        manifest_path=manifest_path,
        manifest_entry=manifest_entry,
        cli_argv=sys.argv,
    )


if __name__ == "__main__":
    main()
