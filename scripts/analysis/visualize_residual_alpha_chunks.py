#!/usr/bin/env python3
"""
Chunked DreamZero-aligned residual alpha visualization.

This script treats a full episode video as a sequence of DreamZero video chunks.
For DROID/Wan2.2 each chunk covers a 24-step raw-video window and samples:

    [anchor, anchor + 3, anchor + 6, ..., anchor + 24]

The first sampled frame is the chunk anchor. Wan VAE38 maps these 9 pixel frames
to 3 latent frames; the first latent is the anchor and the next 2 latent frames
are the causal future block. Static init is built by VAE-encoding a pixel-space
static copy of the anchor frame, then using latent frames [1:3].
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))

from scripts.analysis.visualize_residual_alpha import (  # noqa: E402
    encode_video_latents,
    load_wan_vae,
    parse_alpha_values,
    write_video_grid,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="DreamZero chunked residual-alpha visualization for full episode videos"
    )
    p.add_argument("--vae_path", required=True, help="Path to Wan VAE checkpoint")
    p.add_argument("--video_path", required=True, help="Composite episode video path")
    p.add_argument("--output_dir", required=True, help="Output directory")
    p.add_argument("--image_height", type=int, default=320)
    p.add_argument("--image_width", type=int, default=640)
    p.add_argument("--z_dim", type=int, default=48)
    p.add_argument("--alphas", type=str, default="0,0.2,0.4,0.6,0.8,1.0")
    p.add_argument("--grid_cols", type=int, default=3)
    p.add_argument("--fps", type=int, default=5)
    p.add_argument("--chunk_stride", type=int, default=24)
    p.add_argument("--sample_stride", type=int, default=3)
    p.add_argument("--num_frame_per_block", type=int, default=2)
    p.add_argument("--no_tiled", action="store_true", default=False)
    return p.parse_args()


def load_full_video(path: str, height: int, width: int, device: str) -> torch.Tensor:
    try:
        import decord
    except ImportError:
        import decord2 as decord

    vr = decord.VideoReader(path)
    indices = np.arange(len(vr), dtype=np.int64)
    frames = vr.get_batch(indices).asnumpy()  # [T, H, W, C]
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(0)
    frames = frames.permute(0, 2, 1, 3, 4)  # [1, C, T, H, W]
    if frames.shape[-2] != height or frames.shape[-1] != width:
        frames = torch.nn.functional.interpolate(
            frames.flatten(0, 1),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).unflatten(0, frames.shape[:2])
    return (frames.float() / 127.5 - 1.0).to(device)


def build_chunk_sample_indices(
    total_frames: int,
    chunk_stride: int,
    sample_stride: int,
) -> list[list[int]]:
    if total_frames <= 0:
        raise ValueError("Video has no frames")
    if chunk_stride <= 0 or sample_stride <= 0:
        raise ValueError("chunk_stride and sample_stride must be positive")
    if chunk_stride % sample_stride != 0:
        raise ValueError(
            f"chunk_stride must be divisible by sample_stride, got "
            f"{chunk_stride} and {sample_stride}"
        )

    samples_per_chunk = chunk_stride // sample_stride + 1
    starts = list(range(0, max(1, total_frames - 1), chunk_stride))
    chunks = []
    for start in starts:
        indices = [start + sample_stride * i for i in range(samples_per_chunk)]
        indices = [min(idx, total_frames - 1) for idx in indices]
        chunks.append(indices)
    return chunks


def build_static_chunk_latents(
    vae,
    chunk_video: torch.Tensor,
    clean_latents: torch.Tensor,
    tiled: bool,
    num_frame_per_block: int,
) -> torch.Tensor:
    if clean_latents.shape[2] != num_frame_per_block + 1:
        raise ValueError(
            "Expected one anchor latent plus num_frame_per_block future latents, "
            f"got clean_latents.shape={tuple(clean_latents.shape)}"
        )
    anchor = chunk_video[:, :, 0:1]
    static_pixels = anchor.repeat(1, 1, 1 + 4 * num_frame_per_block, 1, 1)
    static_block = encode_video_latents(vae, static_pixels, tiled)
    static_block = static_block[:, :, 1:1 + num_frame_per_block]
    static_init = clean_latents.clone()
    static_init[:, :, 1:1 + num_frame_per_block] = static_block
    return static_init


def latent_to_uint8_frames(video: torch.Tensor) -> np.ndarray:
    frames = video[0].permute(1, 2, 3, 0)
    frames = (frames.clamp(-1, 1) + 1) / 2 * 255
    return frames.cpu().to(torch.uint8).numpy()


def make_label(label_text: str, alpha_val: float, label_h: int, width: int) -> np.ndarray:
    bg = np.zeros((label_h, width, 3), dtype=np.uint8)
    bg[:, :, 0] = int(60 + 160 * alpha_val)
    bg[:, :, 1] = 20
    bg[:, :, 2] = int(200 - 140 * alpha_val)
    try:
        from PIL import Image, ImageDraw, ImageFont

        img = Image.fromarray(bg)
        draw = ImageDraw.Draw(img)
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        font = ImageFont.truetype(font_path, size=max(10, label_h - 8))
        bbox = draw.textbbox((0, 0), label_text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = (width - tw) // 2
        ty = (label_h - th) // 2
        draw.text((tx + 1, ty + 1), label_text, fill=(0, 0, 0), font=font)
        draw.text((tx, ty), label_text, fill=(255, 255, 255), font=font)
        return np.array(img)
    except Exception:
        return bg


def assemble_grid(decoded_by_alpha: list[list[np.ndarray]], alphas: list[float], grid_cols: int) -> np.ndarray:
    videos = [np.concatenate(chunks, axis=0) for chunks in decoded_by_alpha]
    t_out = min(video.shape[0] for video in videos)
    videos = [video[:t_out] for video in videos]
    h_sub, w_sub = videos[0].shape[1], videos[0].shape[2]
    label_h = max(32, h_sub // 12)
    labels = []
    for alpha in alphas:
        suffix = ""
        if alpha == 0.0:
            suffix = " (static)"
        elif alpha == 1.0:
            suffix = " (real)"
        labels.append(make_label(f"alpha={alpha:.2f}{suffix}", alpha, label_h, w_sub))

    grid_cols = max(1, grid_cols)
    grid_rows = int(np.ceil(len(alphas) / grid_cols))
    blank = np.zeros((h_sub + label_h, w_sub, 3), dtype=np.uint8)
    frames = []
    for t in range(t_out):
        rows = []
        for row in range(grid_rows):
            parts = []
            for col in range(grid_cols):
                i = row * grid_cols + col
                if i < len(videos):
                    part = np.concatenate([labels[i], videos[i][t]], axis=0)
                else:
                    part = blank
                parts.append(part)
            rows.append(np.concatenate(parts, axis=1))
        frames.append(np.concatenate(rows, axis=0))
    return np.stack(frames, axis=0)


@torch.no_grad()
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tiled = not args.no_tiled
    alphas = parse_alpha_values(args.alphas)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Loading video: {args.video_path}")
    video = load_full_video(args.video_path, args.image_height, args.image_width, device)
    total_frames = int(video.shape[2])
    chunks = build_chunk_sample_indices(total_frames, args.chunk_stride, args.sample_stride)
    print(f"Loaded video: {tuple(video.shape)}")
    print(f"DreamZero chunks: {len(chunks)} x {len(chunks[0])} sampled frames")
    print(f"First chunk indices: {chunks[0]}")
    print(f"Last chunk indices:  {chunks[-1]}")

    vae = load_wan_vae(args.vae_path, args.z_dim, device)
    decoded_by_alpha: list[list[np.ndarray]] = [[] for _ in alphas]
    residual_norms = []

    for chunk_index, indices in enumerate(chunks):
        print(f"Chunk {chunk_index + 1}/{len(chunks)} indices={indices}")
        chunk_video = video[:, :, indices]
        clean_latents = encode_video_latents(vae, chunk_video, tiled)
        static_init = build_static_chunk_latents(
            vae,
            chunk_video,
            clean_latents,
            tiled,
            args.num_frame_per_block,
        )
        residual = clean_latents - static_init
        residual_norms.append(float(torch.norm(residual[:, :, 1:]).item()))

        for alpha_index, alpha in enumerate(alphas):
            latent = static_init + alpha * residual
            decoded = vae.decode(
                latent,
                tiled=tiled,
                tile_size=(34, 34),
                tile_stride=(18, 16),
            )
            decoded_by_alpha[alpha_index].append(latent_to_uint8_frames(decoded))

    grid = assemble_grid(decoded_by_alpha, alphas, args.grid_cols)
    grid_path = out / "grid_all_alphas.mp4"
    write_video_grid(grid, str(grid_path), args.fps)

    stats = {
        "video_path": args.video_path,
        "input_shape": list(video.shape),
        "image_height": args.image_height,
        "image_width": args.image_width,
        "alphas": alphas,
        "grid_cols": args.grid_cols,
        "fps": args.fps,
        "chunk_stride": args.chunk_stride,
        "sample_stride": args.sample_stride,
        "num_frame_per_block": args.num_frame_per_block,
        "num_chunks": len(chunks),
        "samples_per_chunk": len(chunks[0]),
        "chunk_sample_indices": chunks,
        "residual_norm_per_chunk": residual_norms,
        "output_frames": int(grid.shape[0]),
    }
    with (out / "stats.json").open("w") as f:
        json.dump(stats, f, indent=2)
    print(f"Done. Grid video: {grid_path}")


if __name__ == "__main__":
    main()
