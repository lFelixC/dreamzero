#!/usr/bin/env python3
"""
Visualize prefix-aware static-init latents against clean video latents.

This restores the old residual-alpha idea with the current DreamZero static
video initialization semantics:

    clean_latents = VAE(real sampled video)
    static_init   = VAE(prefix + repeated last-prefix frame), block by block
    decoded(alpha)= VAE.decode(static_init + alpha * (clean_latents - static_init))

The default mode treats the input as one full model video, so each static block is
encoded with all previous prefix frames. This mirrors
WANPolicyHead._encode_static_training_init rather than the older per-chunk anchor
only visualization.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DreamZero prefix-aware residual-alpha latent interpolation"
    )
    parser.add_argument("--vae_path", required=True, help="Path to Wan VAE checkpoint")
    parser.add_argument("--video_path", required=True, help="Input composite video path")
    parser.add_argument("--output_dir", required=True, help="Directory for output videos")
    parser.add_argument("--image_height", type=int, default=160)
    parser.add_argument("--image_width", type=int, default=320)
    parser.add_argument("--z_dim", type=int, default=48, help="48 for Wan2.2, 16 for Wan2.1")
    parser.add_argument("--vae38_dim", type=int, default=160, help="WanVideoVAE38 dim")
    parser.add_argument("--num_frame_per_block", type=int, default=2)
    parser.add_argument(
        "--raw_sample_stride",
        type=int,
        default=3,
        help="Stride in raw input video frames before VAE encoding.",
    )
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument(
        "--end_frame",
        type=int,
        default=-1,
        help="Exclusive raw frame end. -1 means video end.",
    )
    parser.add_argument(
        "--max_model_frames",
        type=int,
        default=0,
        help="Cap sampled model frames before boundary trimming. 0 means no cap.",
    )
    parser.add_argument("--alphas", type=str, default="0,0.2,0.4,0.6,0.8,1.0")
    parser.add_argument("--grid_cols", type=int, default=3)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_tiled", action="store_true")
    parser.add_argument("--tile_size_height", type=int, default=34)
    parser.add_argument("--tile_size_width", type=int, default=34)
    parser.add_argument("--tile_stride_height", type=int, default=18)
    parser.add_argument("--tile_stride_width", type=int, default=16)
    parser.add_argument(
        "--vae_dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
        help="Dtype used for VAE encode/decode.",
    )
    parser.add_argument(
        "--write_individual",
        action="store_true",
        help="Also write one MP4 per alpha in addition to grid_all_alphas.mp4.",
    )
    return parser.parse_args()


def parse_alpha_values(value: str) -> list[float]:
    alphas = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        alpha = float(part)
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError(f"alpha values must be in [0, 1], got {alpha}")
        alphas.append(alpha)
    if not alphas:
        raise ValueError("At least one alpha is required")
    return alphas


def torch_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def load_wan_vae(
    vae_path: str,
    z_dim: int,
    vae38_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    if device.type != "cuda":
        raise RuntimeError("Wan VAE construction in this repo expects CUDA tensors.")

    from groot.vla.model.dreamzero.modules.wan_video_vae import WanVideoVAE, WanVideoVAE38

    if z_dim == 48:
        vae = WanVideoVAE38(z_dim=z_dim, dim=vae38_dim, vae_pretrained_path=vae_path)
    else:
        vae = WanVideoVAE(z_dim=z_dim, vae_pretrained_path=vae_path)

    print(f"Loading VAE weights: {vae_path}")
    try:
        state_dict = torch.load(vae_path, map_location="cpu", mmap=True, weights_only=True)
    except (TypeError, ValueError):
        state_dict = torch.load(vae_path, map_location="cpu")
    incompatible = vae.model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            "Warning: VAE state dict mismatch: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    del state_dict

    vae = vae.to(device=device, dtype=dtype)
    vae.mean = vae.mean.to(device=device)
    vae.std = vae.std.to(device=device)
    vae.scale = [vae.mean, 1.0 / vae.std]
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    return vae


def _read_video_decord(path: str, indices: np.ndarray) -> np.ndarray | None:
    try:
        import decord
    except ImportError:
        try:
            import decord2 as decord
        except ImportError:
            return None

    vr = decord.VideoReader(path)
    if indices.size == 0:
        return np.empty((0, 0, 0, 3), dtype=np.uint8)
    valid_indices = np.clip(indices, 0, len(vr) - 1)
    return vr.get_batch(valid_indices).asnumpy()


def _read_video_cv2(path: str, indices: np.ndarray) -> np.ndarray | None:
    try:
        import cv2
    except ImportError:
        return None

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    frames = []
    wanted = set(int(i) for i in indices.tolist())
    current = 0
    while wanted:
        ok, frame = cap.read()
        if not ok:
            break
        if current in wanted:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            wanted.remove(current)
        current += 1
    cap.release()
    if not frames:
        return None
    return np.stack(frames, axis=0)


def get_video_length(path: str) -> int:
    try:
        import decord

        return len(decord.VideoReader(path))
    except ImportError:
        pass

    try:
        import cv2

        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if length > 0:
                return length
    except ImportError:
        pass

    raise RuntimeError("Could not determine video length; install decord or opencv-python.")


def resize_frames_torch(frames: np.ndarray, height: int, width: int) -> torch.Tensor:
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    if tensor.shape[-2:] != (height, width):
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
    return tensor


def load_model_video(
    video_path: str,
    height: int,
    width: int,
    raw_sample_stride: int,
    start_frame: int,
    end_frame: int,
    max_model_frames: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, list[int]]:
    if raw_sample_stride <= 0:
        raise ValueError(f"raw_sample_stride must be positive, got {raw_sample_stride}")
    total = get_video_length(video_path)
    start = max(0, start_frame)
    end = total if end_frame < 0 else min(end_frame, total)
    if end <= start:
        raise ValueError(f"Invalid frame range: start={start}, end={end}, total={total}")

    indices = np.arange(start, end, raw_sample_stride, dtype=np.int64)
    if max_model_frames > 0:
        indices = indices[:max_model_frames]
    if indices.size == 0:
        raise ValueError("No frames selected")

    frames = _read_video_decord(video_path, indices)
    if frames is None:
        frames = _read_video_cv2(video_path, indices)
    if frames is None:
        raise RuntimeError("Could not read video frames; install decord or opencv-python.")

    tensor = resize_frames_torch(frames, height, width)
    tensor = tensor.permute(1, 0, 2, 3).unsqueeze(0)  # [1, 3, T, H, W]
    tensor = tensor / 127.5 - 1.0
    return tensor.to(device=device, dtype=dtype), indices.tolist()


def valid_model_frame_count(sampled_frames: int, num_frame_per_block: int) -> int:
    if sampled_frames < 1:
        raise ValueError("At least one sampled frame is required")
    if num_frame_per_block <= 0:
        raise ValueError("num_frame_per_block must be positive")

    latent_frames = 1 + (sampled_frames - 1) // 4
    future_latent_frames = latent_frames - 1
    future_latent_frames -= future_latent_frames % num_frame_per_block
    latent_frames = 1 + future_latent_frames
    return 1 + 4 * (latent_frames - 1)


def prefix_pixel_frames_to_latent_frames(pixel_frames: int) -> int:
    if pixel_frames < 1:
        raise ValueError(f"pixel_frames must be >= 1, got {pixel_frames}")
    if (pixel_frames - 1) % 4 != 0:
        raise ValueError(f"pixel_frames must be 1 + 4k, got {pixel_frames}")
    return 1 + (pixel_frames - 1) // 4


def build_static_prediction_pixels_from_prefix(
    prefix_pixels: torch.Tensor,
    latent_frames: int,
) -> torch.Tensor:
    prefix_pixel_frames_to_latent_frames(prefix_pixels.shape[2])
    if latent_frames < 1:
        raise ValueError(f"latent_frames must be >= 1, got {latent_frames}")
    previous_frame = prefix_pixels[:, :, -1:]
    static_future_pixels = previous_frame.repeat(1, 1, 4 * latent_frames, 1, 1)
    return torch.cat([prefix_pixels, static_future_pixels], dim=2)


@torch.no_grad()
def encode_video_latents(
    vae: torch.nn.Module,
    videos: torch.Tensor,
    tiled: bool,
    tile_size: tuple[int, int],
    tile_stride: tuple[int, int],
) -> torch.Tensor:
    return vae.encode(videos, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)


@torch.no_grad()
def decode_video_latents(
    vae: torch.nn.Module,
    latents: torch.Tensor,
    tiled: bool,
    tile_size: tuple[int, int],
    tile_stride: tuple[int, int],
) -> torch.Tensor:
    return vae.decode(latents, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)


@torch.no_grad()
def encode_static_block_latents_from_prefix(
    vae: torch.nn.Module,
    prefix_pixels: torch.Tensor,
    latent_frames: int,
    tiled: bool,
    tile_size: tuple[int, int],
    tile_stride: tuple[int, int],
    dtype: torch.dtype,
) -> torch.Tensor:
    static_pixels = build_static_prediction_pixels_from_prefix(prefix_pixels, latent_frames)
    static_latents = encode_video_latents(vae, static_pixels, tiled, tile_size, tile_stride)
    prefix_latent_frames = prefix_pixel_frames_to_latent_frames(prefix_pixels.shape[2])
    static_latents = static_latents[:, :, prefix_latent_frames:prefix_latent_frames + latent_frames]
    return static_latents.to(dtype=dtype)


@torch.no_grad()
def build_prefix_aware_static_init(
    vae: torch.nn.Module,
    videos: torch.Tensor,
    clean_latents: torch.Tensor,
    num_frame_per_block: int,
    tiled: bool,
    tile_size: tuple[int, int],
    tile_stride: tuple[int, int],
    dtype: torch.dtype,
) -> tuple[torch.Tensor, list[dict]]:
    latent_frames = clean_latents.shape[2]
    future_latent_frames = latent_frames - 1
    if future_latent_frames <= 0:
        return clean_latents.clone(), []
    if future_latent_frames % num_frame_per_block != 0:
        raise ValueError(
            "future latent frames must form complete blocks: "
            f"future={future_latent_frames}, num_frame_per_block={num_frame_per_block}"
        )

    static_init = clean_latents.clone()
    block_stats = []
    for block_start in range(1, latent_frames, num_frame_per_block):
        prefix_pixel_frames = 1 + 4 * (block_start - 1)
        prefix_pixels = videos[:, :, :prefix_pixel_frames]
        block_latents = encode_static_block_latents_from_prefix(
            vae,
            prefix_pixels,
            latent_frames=num_frame_per_block,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
            dtype=dtype,
        )
        block_slice = slice(block_start, block_start + num_frame_per_block)
        if block_latents.shape != static_init[:, :, block_slice].shape:
            raise ValueError(
                "Static block latent shape mismatch: "
                f"got {tuple(block_latents.shape)}, "
                f"expected {tuple(static_init[:, :, block_slice].shape)}"
            )
        static_init[:, :, block_slice] = block_latents
        block_stats.append(
            {
                "block_start_latent": block_start,
                "block_end_latent_exclusive": block_start + num_frame_per_block,
                "prefix_pixel_frames": prefix_pixel_frames,
            }
        )
    return static_init, block_stats


def latents_to_frames(video: torch.Tensor) -> np.ndarray:
    frames = video[0].permute(1, 2, 3, 0)
    frames = ((frames.float().clamp(-1, 1) + 1.0) * 127.5).round()
    return frames.cpu().to(torch.uint8).numpy()


def make_label(text: str, alpha: float, height: int, width: int) -> np.ndarray:
    label = np.zeros((height, width, 3), dtype=np.uint8)
    label[:, :, 0] = int(55 + 155 * alpha)
    label[:, :, 1] = 26
    label[:, :, 2] = int(205 - 135 * alpha)
    try:
        from PIL import Image, ImageDraw, ImageFont

        image = Image.fromarray(label)
        draw = ImageDraw.Draw(image)
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        font_size = max(11, height - 9)
        font = ImageFont.truetype(font_path, size=font_size)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = max(2, (width - text_w) // 2)
        y = max(1, (height - text_h) // 2)
        draw.text((x + 1, y + 1), text, fill=(0, 0, 0), font=font)
        draw.text((x, y), text, fill=(255, 255, 255), font=font)
        return np.asarray(image)
    except Exception:
        return label


def pad_even(frames: np.ndarray) -> np.ndarray:
    _, h, w, _ = frames.shape
    pad_h = h % 2
    pad_w = w % 2
    if pad_h == 0 and pad_w == 0:
        return frames
    return np.pad(frames, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)), mode="edge")


def assemble_grid(videos: list[np.ndarray], alphas: list[float], grid_cols: int) -> np.ndarray:
    if not videos:
        raise ValueError("No videos to assemble")
    frame_count = min(video.shape[0] for video in videos)
    videos = [video[:frame_count] for video in videos]
    h, w = videos[0].shape[1:3]
    label_h = max(24, h // 12)
    labels = []
    for alpha in alphas:
        suffix = ""
        if alpha == 0.0:
            suffix = " static"
        elif alpha == 1.0:
            suffix = " clean"
        labels.append(make_label(f"alpha={alpha:.2f}{suffix}", alpha, label_h, w))

    cols = max(1, grid_cols)
    rows = int(np.ceil(len(videos) / cols))
    blank = np.zeros((h + label_h, w, 3), dtype=np.uint8)
    output = []
    for t in range(frame_count):
        row_frames = []
        for row in range(rows):
            parts = []
            for col in range(cols):
                index = row * cols + col
                if index < len(videos):
                    parts.append(np.concatenate([labels[index], videos[index][t]], axis=0))
                else:
                    parts.append(blank)
            row_frames.append(np.concatenate(parts, axis=1))
        output.append(np.concatenate(row_frames, axis=0))
    return pad_even(np.stack(output, axis=0))


def write_video(frames: np.ndarray, path: str, fps: int) -> None:
    frames = pad_even(frames)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        import cv2

        height, width = frames.shape[1:3]
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {path}")
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        return
    except ImportError:
        pass

    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError("Writing video requires imageio or opencv-python") from exc

    imageio.mimsave(path, frames, fps=fps, codec="libx264")


def write_individual_videos(
    videos: Iterable[np.ndarray],
    alphas: Iterable[float],
    output_dir: Path,
    fps: int,
) -> None:
    for video, alpha in zip(videos, alphas, strict=True):
        name = f"alpha_{alpha:.2f}".replace(".", "p") + ".mp4"
        write_video(video, str(output_dir / name), fps)


def block_residual_stats(
    residual: torch.Tensor,
    num_frame_per_block: int,
    block_stats: list[dict],
) -> list[dict]:
    result = []
    for item in block_stats:
        start = int(item["block_start_latent"])
        block = residual[:, :, start:start + num_frame_per_block]
        block_float = block.float()
        result.append(
            {
                **item,
                "l2_norm": float(torch.linalg.vector_norm(block_float).cpu()),
                "mse": float((block_float.square().mean()).cpu()),
                "mean_abs": float((block_float.abs().mean()).cpu()),
                "max_abs": float((block_float.abs().max()).cpu()),
            }
        )
    return result


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

    torch.manual_seed(args.seed)
    print(f"Device: {device}, dtype: {dtype}, tiled: {tiled}")
    print(f"Loading video: {args.video_path}")
    videos, raw_indices = load_model_video(
        args.video_path,
        args.image_height,
        args.image_width,
        args.raw_sample_stride,
        args.start_frame,
        args.end_frame,
        args.max_model_frames,
        device,
        dtype,
    )

    valid_frames = valid_model_frame_count(videos.shape[2], args.num_frame_per_block)
    if valid_frames != videos.shape[2]:
        print(f"Trimming sampled model frames from {videos.shape[2]} to {valid_frames}")
        videos = videos[:, :, :valid_frames]
        raw_indices = raw_indices[:valid_frames]
    if valid_frames < 1 + 4 * args.num_frame_per_block:
        raise ValueError(
            "Not enough sampled frames for one future block after trimming: "
            f"valid_frames={valid_frames}, num_frame_per_block={args.num_frame_per_block}"
        )

    print(f"Model video tensor: {tuple(videos.shape)} [B,C,T,H,W]")
    print(f"Raw frame range used: first={raw_indices[0]}, last={raw_indices[-1]}, count={len(raw_indices)}")

    vae = load_wan_vae(args.vae_path, args.z_dim, args.vae38_dim, device, dtype)

    print("Encoding clean full video latents...")
    clean_latents = encode_video_latents(vae, videos, tiled, tile_size, tile_stride)
    clean_latents = clean_latents.to(dtype=dtype)
    print(f"Clean latents: {tuple(clean_latents.shape)} [B,C,T,H,W]")

    print("Building prefix-aware static-init latents...")
    static_init, block_stats = build_prefix_aware_static_init(
        vae,
        videos,
        clean_latents,
        args.num_frame_per_block,
        tiled,
        tile_size,
        tile_stride,
        dtype,
    )
    residual = clean_latents - static_init
    residual_stats = block_residual_stats(residual, args.num_frame_per_block, block_stats)

    decoded_by_alpha = []
    for alpha in alphas:
        print(f"Decoding alpha={alpha:.3f}")
        latent = torch.lerp(static_init, clean_latents, alpha)
        decoded = decode_video_latents(vae, latent, tiled, tile_size, tile_stride)
        decoded_by_alpha.append(latents_to_frames(decoded))

    grid = assemble_grid(decoded_by_alpha, alphas, args.grid_cols)
    grid_path = output_dir / "grid_all_alphas.mp4"
    write_video(grid, str(grid_path), args.fps)
    print(f"Wrote grid video: {grid_path}")

    if args.write_individual:
        write_individual_videos(decoded_by_alpha, alphas, output_dir, args.fps)
        print(f"Wrote individual alpha videos to: {output_dir}")

    stats = {
        "video_path": args.video_path,
        "vae_path": args.vae_path,
        "image_height": args.image_height,
        "image_width": args.image_width,
        "z_dim": args.z_dim,
        "vae38_dim": args.vae38_dim,
        "num_frame_per_block": args.num_frame_per_block,
        "raw_sample_stride": args.raw_sample_stride,
        "raw_frame_indices": raw_indices,
        "input_tensor_shape_bcthw": list(videos.shape),
        "clean_latent_shape_bcthw": list(clean_latents.shape),
        "static_latent_shape_bcthw": list(static_init.shape),
        "alphas": alphas,
        "fps": args.fps,
        "grid_cols": args.grid_cols,
        "tiled": tiled,
        "tile_size": list(tile_size),
        "tile_stride": list(tile_stride),
        "residual_l2_norm": float(torch.linalg.vector_norm(residual.float()).cpu()),
        "residual_mse": float((residual.float().square().mean()).cpu()),
        "residual_mean_abs": float(residual.float().abs().mean().cpu()),
        "residual_norm_per_block": residual_stats,
        "output_grid_video": str(grid_path),
        "visualization": "decode(static_init + alpha * (clean_latents - static_init))",
        "static_init": "prefix-aware blockwise VAE encode of prefix plus repeated last prefix frame",
    }
    with (output_dir / "stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)
    print(f"Wrote stats: {output_dir / 'stats.json'}")


if __name__ == "__main__":
    main()
