#!/usr/bin/env python3
"""
Visualize training-style video latent input -> clean output interpolation.

For each short training window, this script visualizes the video-side training
input and output:

    output_pixels = real 17-frame training window
    input_pixels  = first 9 real frames + repeat(frame 8) for the future frames
    output_latents = VAE.encode(output_pixels)
    input_latents  = VAE.encode(input_pixels)

It then decodes latent-space interpolation:

    latent(alpha) = (1 - alpha) * input_latents + alpha * output_latents

alpha=0 is the training input/prefix-static latent, alpha=1 is the clean VAE
output target latent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.analysis.visualize_residual_alpha_chunks import (  # noqa: E402
    assemble_grid,
    decode_video_latents,
    encode_video_latents,
    latents_to_frames,
    load_model_video,
    load_wan_vae,
    parse_alpha_values,
    torch_dtype,
    write_video,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training-window x_t -> clean latent interpolation visualization"
    )
    parser.add_argument("--vae_path", required=True)
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_height", type=int, default=160)
    parser.add_argument("--image_width", type=int, default=320)
    parser.add_argument("--z_dim", type=int, default=48)
    parser.add_argument("--vae38_dim", type=int, default=160)
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument(
        "--prefix_frames",
        type=int,
        default=9,
        help="Observed prefix frames used to build the static training input.",
    )
    parser.add_argument("--num_frame_per_block", type=int, default=2)
    parser.add_argument("--window_starts", type=str, default="")
    parser.add_argument("--num_windows", type=int, default=6)
    parser.add_argument("--window_stride", type=int, default=80)
    parser.add_argument("--raw_sample_stride", type=int, default=1)
    parser.add_argument("--alphas", type=str, default="0,0.2,0.4,0.6,0.8,1.0")
    parser.add_argument("--grid_cols", type=int, default=3)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_tiled", action="store_true")
    parser.add_argument("--tile_size_height", type=int, default=34)
    parser.add_argument("--tile_size_width", type=int, default=34)
    parser.add_argument("--tile_stride_height", type=int, default=18)
    parser.add_argument("--tile_stride_width", type=int, default=16)
    parser.add_argument("--vae_dtype", choices=("bfloat16", "float32"), default="bfloat16")
    return parser.parse_args()


def parse_window_starts(value: str) -> list[int]:
    if not value.strip():
        return []
    starts = []
    for item in value.split(","):
        item = item.strip()
        if item:
            starts.append(int(item))
    return starts


def choose_window_starts(
    total_frames: int,
    num_frames: int,
    explicit_starts: list[int],
    num_windows: int,
    window_stride: int,
) -> list[int]:
    max_start = total_frames - num_frames
    if max_start < 0:
        raise ValueError(f"Video has {total_frames} frames, shorter than num_frames={num_frames}")
    if explicit_starts:
        starts = explicit_starts
    else:
        starts = [i * window_stride for i in range(num_windows)]
    clean_starts = []
    for start in starts:
        start = max(0, min(int(start), max_start))
        if start not in clean_starts:
            clean_starts.append(start)
    return clean_starts


def build_prefix_static_input_pixels(window: torch.Tensor, prefix_frames: int) -> torch.Tensor:
    if window.ndim != 5:
        raise ValueError(f"Expected [B,C,T,H,W] window, got {tuple(window.shape)}")
    if prefix_frames < 1:
        raise ValueError(f"prefix_frames must be >= 1, got {prefix_frames}")
    if prefix_frames >= window.shape[2]:
        raise ValueError(
            f"prefix_frames must be smaller than num_frames, got "
            f"prefix_frames={prefix_frames}, num_frames={window.shape[2]}"
        )
    if (prefix_frames - 1) % 4 != 0:
        raise ValueError(f"prefix_frames must be 1 + 4k for Wan VAE, got {prefix_frames}")
    future_frames = window.shape[2] - prefix_frames
    if future_frames % 4 != 0:
        raise ValueError(
            f"num_frames - prefix_frames must be divisible by 4, got "
            f"{window.shape[2]} - {prefix_frames} = {future_frames}"
        )
    prefix = window[:, :, :prefix_frames]
    static_future = window[:, :, prefix_frames - 1:prefix_frames].repeat(
        1, 1, future_frames, 1, 1
    )
    return torch.cat([prefix, static_future], dim=2)


def latent_frame_count(pixel_frames: int) -> int:
    if (pixel_frames - 1) % 4 != 0:
        raise ValueError(f"pixel_frames must be 1 + 4k, got {pixel_frames}")
    return 1 + (pixel_frames - 1) // 4


@torch.no_grad()
def main() -> None:
    args = parse_args()
    alphas = parse_alpha_values(args.alphas)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch_dtype(args.vae_dtype)
    tiled = not args.no_tiled
    tile_size = (args.tile_size_height, args.tile_size_width)
    tile_stride = (args.tile_stride_height, args.tile_stride_width)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos, raw_indices = load_model_video(
        args.video_path,
        args.image_height,
        args.image_width,
        args.raw_sample_stride,
        start_frame=0,
        end_frame=-1,
        max_model_frames=0,
        device=device,
        dtype=dtype,
    )
    starts = choose_window_starts(
        total_frames=videos.shape[2],
        num_frames=args.num_frames,
        explicit_starts=parse_window_starts(args.window_starts),
        num_windows=args.num_windows,
        window_stride=args.window_stride,
    )
    print(f"Loaded model video: {tuple(videos.shape)}; window_starts={starts}")

    vae = load_wan_vae(args.vae_path, args.z_dim, args.vae38_dim, device, dtype)

    all_stats = {
        "video_path": args.video_path,
        "input_video_shape_bcthw": list(videos.shape),
        "raw_frame_indices": raw_indices,
        "image_height": args.image_height,
        "image_width": args.image_width,
        "num_frames": args.num_frames,
        "prefix_frames": args.prefix_frames,
        "num_frame_per_block": args.num_frame_per_block,
        "prefix_latent_frames": latent_frame_count(args.prefix_frames),
        "output_latent_frames": latent_frame_count(args.num_frames),
        "alphas": alphas,
        "windows": [],
        "visualization": "decode((1-alpha)*input_latents + alpha*output_latents)",
        "input_pixels": "real prefix frames followed by repeated last prefix frame",
    }

    for window_index, start in enumerate(starts):
        window_dir = output_dir / f"window_{window_index:03d}_start_{start:04d}"
        window_dir.mkdir(parents=True, exist_ok=True)
        window = videos[:, :, start:start + args.num_frames]
        print(f"Window {window_index}: start={start}, tensor={tuple(window.shape)}")

        input_pixels = build_prefix_static_input_pixels(window, args.prefix_frames)
        input_latents = encode_video_latents(vae, input_pixels, tiled, tile_size, tile_stride)
        output_latents = encode_video_latents(vae, window, tiled, tile_size, tile_stride)
        if input_latents.shape != output_latents.shape:
            raise ValueError(
                f"Input/output latent shape mismatch: "
                f"{tuple(input_latents.shape)} vs {tuple(output_latents.shape)}"
            )

        decoded_by_alpha = []
        for alpha in alphas:
            latent = torch.lerp(input_latents, output_latents, alpha)
            decoded = decode_video_latents(vae, latent, tiled, tile_size, tile_stride)
            decoded_by_alpha.append(latents_to_frames(decoded))

        grid = assemble_grid(decoded_by_alpha, alphas, args.grid_cols)
        grid_path = window_dir / "training_input_to_clean_grid.mp4"
        write_video(grid, str(grid_path), args.fps)

        window_stats = {
            "window_index": window_index,
            "start_model_frame": start,
            "end_model_frame_exclusive": start + args.num_frames,
            "start_raw_frame": raw_indices[start],
            "end_raw_frame_exclusive": raw_indices[start + args.num_frames - 1] + 1,
            "window_shape_bcthw": list(window.shape),
            "input_pixels_shape_bcthw": list(input_pixels.shape),
            "input_latent_shape_bcthw": list(input_latents.shape),
            "output_latent_shape_bcthw": list(output_latents.shape),
            "input_to_output_l2_norm": float(torch.linalg.vector_norm((input_latents - output_latents).float()).cpu()),
            "input_to_output_mse": float(((input_latents - output_latents).float().square().mean()).cpu()),
            "output_grid_video": str(grid_path),
            "visualization": "decode((1-alpha)*input_latents + alpha*output_latents)",
        }
        with (window_dir / "stats.json").open("w", encoding="utf-8") as handle:
            json.dump(window_stats, handle, ensure_ascii=False, indent=2)
        all_stats["windows"].append(window_stats)
        print(f"  wrote {grid_path}")

    with (output_dir / "stats.json").open("w", encoding="utf-8") as handle:
        json.dump(all_stats, handle, ensure_ascii=False, indent=2)
    print(f"Done. Wrote summary: {output_dir / 'stats.json'}")


if __name__ == "__main__":
    main()
