#!/usr/bin/env python3
"""
Visualization experiment: Compare noise-init vs static-frame-init for video denoising.

This script demonstrates WHY initializing video latent blocks from a static copy of
the previous frame (VAE-encoded) is geometrically better than starting from pure
Gaussian noise in flow-matching training.

Key insight (flow matching):
  Forward:  noisy = (1 - sigma) * clean_latent + sigma * init_latent
  Target:   velocity = init_latent - clean_latent   (what the model must predict)

  - Noise init:  init = ε ~ N(0, I)     → noisy = (1-σ)*clean + σ*ε,   target = ε - clean
  - Static init: init = s (VAE encoded)  → noisy = (1-σ)*clean + σ*s,   target = s - clean

Since s is a valid VAE latent of a real frame (the previous frame), it lives much closer
to the clean latent manifold than pure Gaussian noise does. This means:
  1. ||s - clean|| << ||ε - clean||  →  smaller training target, easier to learn
  2. (1-σ)*clean + σ*s is closer to clean  →  less denoising work needed
  3. The velocity field for static init has lower variance → more stable training

Usage:
  python scripts/analysis/visualize_video_init_latents.py \
    --vae_path /data/checkpoints/dreamzero/Wan2.2-TI2V-5B/Wan2.2_VAE.pth \
    --video_path /path/to/sample_video.mp4 \
    --output_dir ./init_viz_output \
    --num_frames 25 --sigma 0.99
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# --- Add project root to path ---
_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize noise vs static-frame initialization for video denoising"
    )
    parser.add_argument(
        "--vae_path", type=str, default=None,
        help="Path to Wan2.2 VAE checkpoint (.pth file). If not provided, uses synthetic latents."
    )
    parser.add_argument(
        "--video_path", type=str, default=None,
        help="Path to a sample video (.mp4). If not provided, uses synthetic video data."
    )
    parser.add_argument(
        "--synthetic", action="store_true", default=False,
        help="Force synthetic mode: skip VAE entirely, use realistic synthetic latents. "
             "Useful for quick visualization without GPU or VAE checkpoint."
    )
    parser.add_argument(
        "--output_dir", type=str, default="./init_viz_output",
        help="Directory to save visualizations"
    )
    parser.add_argument(
        "--num_frames", type=int, default=25,
        help="Number of video frames to simulate"
    )
    parser.add_argument(
        "--image_height", type=int, default=160,
        help="Video frame height"
    )
    parser.add_argument(
        "--image_width", type=int, default=320,
        help="Video frame width"
    )
    parser.add_argument(
        "--sigma", type=float, default=0.999,
        help="Flow-matching sigma (noise level). Default 0.999 ≈ one-step video timestep."
    )
    parser.add_argument(
        "--num_train_timesteps", type=int, default=1000,
        help="Number of training timesteps in scheduler"
    )
    parser.add_argument(
        "--shift", type=float, default=5.0,
        help="Scheduler shift parameter"
    )
    parser.add_argument(
        "--num_frame_per_block", type=int, default=1,
        help="Number of latent frames per causal block"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--z_dim", type=int, default=48,
        help="VAE latent dimension (48 for Wan2.2, 16 for Wan2.1)"
    )
    parser.add_argument(
        "--spatial_scale", type=int, default=16,
        help="VAE spatial downsampling factor (16 for Wan2.2, 8 for Wan2.1)"
    )
    return parser.parse_args()


def setup_output_dir(output_dir: str) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


# ==============================================================================
# VAE Loading (standalone, minimal)
# ==============================================================================

def load_wan_vae(vae_path: str, z_dim: int, device: str = "cuda") -> torch.nn.Module:
    """Load the Wan2.2 VAE from a checkpoint."""
    from groot.vla.model.dreamzero.modules.wan_video_vae import WanVideoVAE, WanVideoVAE38

    if z_dim == 48:
        vae = WanVideoVAE38(vae_pretrained_path=vae_path)
    else:
        vae = WanVideoVAE(vae_pretrained_path=vae_path)

    # Load checkpoint weights into the underlying model
    print(f"  Loading VAE weights from {vae_path}...")
    try:
        state_dict = torch.load(vae_path, map_location="cpu", mmap=True, weights_only=True)
    except (TypeError, ValueError):
        state_dict = torch.load(vae_path, map_location="cpu")
    incompatible = vae.model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(f"  Warning: missing_keys={incompatible.missing_keys}, unexpected_keys={incompatible.unexpected_keys}")
    del state_dict

    vae = vae.to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    return vae


# ==============================================================================
# Synthetic Data Generation (when VAE is unavailable)
# ==============================================================================

def generate_synthetic_latents(
    batch_size: int,
    z_dim: int,
    num_latent_frames: int,
    latent_h: int,
    latent_w: int,
    seed: int = 42,
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    """
    Generate realistic synthetic VAE latents that mimic real VAE latent statistics.

    Real VAE latents have:
    - Per-channel means near 0 (after scaling), with structured spatial correlations
    - Non-uniform variance across channels
    - Strong spatial smoothness (from the VAE's convolutional structure)
    - Temporal coherence between adjacent frames

    We simulate these properties to produce meaningful visualizations.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Per-channel statistics (matching Wan2.2 VAE's 48 channels)
    # These are approximate - real values depend on the input data
    rng = np.random.default_rng(seed)
    channel_means = torch.from_numpy(rng.normal(0, 0.3, z_dim)).float().to(device)
    channel_stds = torch.from_numpy(rng.uniform(0.3, 1.5, z_dim)).float().to(device)

    # Generate base: smooth spatial structure via Fourier features
    def make_spatial_base(B, C, T, H, W):
        """Generate spatially smooth latent via low-pass filtered noise."""
        # Create noise in frequency domain, apply low-pass filter
        base = torch.randn(B, C, T, H, W, device=device)
        # Apply a simple box blur to create spatial smoothness
        kernel_size = 3
        pad = kernel_size // 2
        # Reshape to [B*C*T, 1, H, W] for conv2d
        base_flat = base.reshape(B * C * T, 1, H, W)
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=device) / (kernel_size ** 2)
        blurred = torch.nn.functional.conv2d(
            torch.nn.functional.pad(base_flat, (pad, pad, pad, pad), mode="reflect"),
            kernel,
        )
        base = blurred.reshape(B, C, T, H, W)
        return base

    # Generate clean latent: smooth spatial structure + per-channel scaling
    clean = make_spatial_base(batch_size, z_dim, num_latent_frames, latent_h, latent_w)
    # Apply per-channel mean/std
    clean = clean * channel_stds.view(1, -1, 1, 1, 1) + channel_means.view(1, -1, 1, 1, 1)

    # Add temporal coherence: smooth transitions between frames
    for t in range(1, num_latent_frames):
        clean[:, :, t] = 0.7 * clean[:, :, t] + 0.3 * clean[:, :, t - 1]

    # Generate static init: similar to clean but using the first frame as reference
    static = clean.clone()
    for t in range(1, num_latent_frames):
        # Static init mixes the previous frame's latent with some perturbation
        prev_frame = clean[:, :, t - 1]
        # Add small perturbation (real VAE encoding of prev frame won't be exact)
        perturbation = 0.1 * torch.randn_like(prev_frame)
        static[:, :, t] = prev_frame + perturbation

    # Generate pure Gaussian noise
    noise = torch.randn(batch_size, z_dim, num_latent_frames, latent_h, latent_w, device=device)

    return {
        "clean": clean,
        "static": static,
        "noise": noise,
        "channel_means": channel_means,
        "channel_stds": channel_stds,
    }


# ==============================================================================
# Data Simulation
# ==============================================================================

def load_sample_video(
    video_path: str | None,
    num_frames: int,
    height: int,
    width: int,
    batch_size: int,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Load or simulate video frames.
    Returns: [B, 3, T, H, W] tensor in [-1, 1] range.
    """
    if video_path is not None and os.path.exists(video_path):
        import decord
        vr = decord.VideoReader(video_path)
        total_frames = len(vr)
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy()  # [T, H, W, 3] uint8
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0)  # [1, T, 3, H, W]
        frames = frames.permute(0, 2, 1, 3, 4)  # [1, 3, T, H, W]
        # Resize if needed
        if frames.shape[-2] != height or frames.shape[-1] != width:
            frames = torch.nn.functional.interpolate(
                frames.flatten(0, 1), size=(height, width), mode="bilinear", align_corners=False
            ).unflatten(0, frames.shape[:2])
        frames = frames.float() / 127.5 - 1.0  # normalize to [-1, 1]
        frames = frames.repeat(batch_size, 1, 1, 1, 1)
    else:
        # Simulate structured video: moving Gaussian blob
        print(f"No video file provided, generating synthetic video ({num_frames} frames)...")
        frames = _generate_synthetic_video(num_frames, height, width, batch_size)
        frames = frames.to(device)
    return frames.to(device)


def _generate_synthetic_video(
    num_frames: int, height: int, width: int, batch_size: int
) -> torch.Tensor:
    """Generate a synthetic video with a moving blob for visualization."""
    torch.manual_seed(42)
    frames = []
    y_grid, x_grid = torch.meshgrid(
        torch.linspace(-1, 1, height),
        torch.linspace(-1, 1, width),
        indexing="ij",
    )
    for t in range(num_frames):
        # Moving Gaussian blob
        cx = 0.3 * torch.sin(2 * np.pi * t / num_frames)
        cy = 0.3 * torch.cos(2 * np.pi * t / num_frames)
        blob = torch.exp(-((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / 0.1)
        # Add some static background
        bg = 0.3 * torch.exp(-((x_grid) ** 2 + (y_grid) ** 2) / 0.5)
        frame = (blob + bg).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        frame = frame.repeat(1, 3, 1, 1)  # RGB
        frames.append(frame)
    video = torch.cat(frames, dim=1)  # [1, T, H, W] per channel → wrong
    video = torch.stack(frames, dim=2).squeeze(1)  # [1, 3, T, H, W]
    video = video.repeat(batch_size, 1, 1, 1, 1)
    return video


# ==============================================================================
# Forward Process Simulation (mirrors flow_match_scheduler.py + wan_flow_matching_action_tf.py)
# ==============================================================================

def compute_target_sigma(num_train_timesteps: int, shift: float) -> float:
    """Compute the one-step video training sigma (mirrors _infer_mot_one_step_video_training_timestep)."""
    base_sigma = 1.0 - 1.0 / num_train_timesteps
    target_sigma = shift * base_sigma / (1.0 + (shift - 1.0) * base_sigma)
    return target_sigma


def add_noise(
    clean: torch.Tensor, init: torch.Tensor, sigma: float
) -> torch.Tensor:
    """Flow-matching forward: noisy = (1 - sigma) * clean + sigma * init."""
    return (1.0 - sigma) * clean + sigma * init


def compute_training_target(
    clean: torch.Tensor, init: torch.Tensor
) -> torch.Tensor:
    """Flow-matching velocity target: target = init - clean."""
    return init - clean


def compute_static_init_latents(
    vae: torch.nn.Module,
    videos: torch.Tensor,
    clean_latents: torch.Tensor,
    num_frame_per_block: int,
    tiled: bool = True,
    tile_size: tuple[int, int] = (34, 34),
    tile_stride: tuple[int, int] = (18, 16),
) -> torch.Tensor:
    """
    Build static-initialization latents by VAE-encoding a copy of the previous frame
    for each causal block (mirrors _encode_static_training_init).
    """
    B, C, T_pixel, H, W = videos.shape
    _, z_dim, T_lat, H_lat, W_lat = clean_latents.shape

    future_frames = T_lat - 1
    if future_frames <= 0 or future_frames % num_frame_per_block != 0:
        return clean_latents.clone()

    block_starts = list(range(1, T_lat, num_frame_per_block))
    static_init = clean_latents.clone()

    for block_start in block_starts:
        prev_pixel_idx = 4 * (block_start - 1)
        prev_frame = videos[:, :, prev_pixel_idx : prev_pixel_idx + 1]  # [B, 3, 1, H, W]

        # Build static pixels: repeat frame to fill 1 + 4*N pixel frames
        static_pixels = prev_frame.repeat(1, 1, 1 + 4 * num_frame_per_block, 1, 1)

        # VAE encode
        with torch.no_grad():
            static_latent_block = vae.encode(
                static_pixels, tiled=tiled,
                tile_size=tile_size, tile_stride=tile_stride,
            )
        # Slice: keep only the latent frames for this block
        static_latent_block = static_latent_block[:, :, 1 : 1 + num_frame_per_block]

        static_init[:, :, block_start : block_start + num_frame_per_block] = static_latent_block

    return static_init


# ==============================================================================
# Metrics
# ==============================================================================

def compute_metrics(
    clean: torch.Tensor,
    noise: torch.Tensor,
    static: torch.Tensor,
    noisy_noise: torch.Tensor,
    noisy_static: torch.Tensor,
    target_noise: torch.Tensor,
    target_static: torch.Tensor,
) -> dict:
    """Compute comprehensive comparison metrics."""
    def mse(a, b):
        return torch.mean((a - b) ** 2).item()

    def cosine_sim(a, b):
        a_flat = a.reshape(-1)
        b_flat = b.reshape(-1)
        return torch.dot(a_flat, b_flat).item() / (
            torch.norm(a_flat).item() * torch.norm(b_flat).item() + 1e-8
        )

    def lp_norm(t, p=2):
        return torch.norm(t.reshape(-1), p=p).item()

    metrics = {
        # Distance from clean latent
        "mse_clean_vs_noise": mse(clean, noise),
        "mse_clean_vs_static": mse(clean, static),
        "mse_clean_vs_noisy_noise": mse(clean, noisy_noise),
        "mse_clean_vs_noisy_static": mse(clean, noisy_static),

        # Cosine similarity to clean latent
        "cosine_clean_vs_noise": cosine_sim(clean, noise),
        "cosine_clean_vs_static": cosine_sim(clean, static),
        "cosine_clean_vs_noisy_noise": cosine_sim(clean, noisy_noise),
        "cosine_clean_vs_noisy_static": cosine_sim(clean, noisy_static),

        # Target (velocity) magnitude
        "target_norm_noise_init": lp_norm(target_noise),
        "target_norm_static_init": lp_norm(target_static),

        # Ratio metrics (how much better is static init?)
        "static_vs_noise_mse_ratio": mse(clean, static) / (mse(clean, noise) + 1e-8),
        "static_vs_noise_target_ratio": lp_norm(target_static) / (lp_norm(target_noise) + 1e-8),
        "noisy_static_vs_noisy_noise_mse_ratio": mse(clean, noisy_static) / (mse(clean, noisy_noise) + 1e-8),
    }
    return metrics


# ==============================================================================
# Visualization Functions
# ==============================================================================

def set_plotting_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.titlesize": 15,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })


def plot_latent_space_2d(
    clean: torch.Tensor,
    noise: torch.Tensor,
    static: torch.Tensor,
    noisy_noise: torch.Tensor,
    noisy_static: torch.Tensor,
    target_noise: torch.Tensor,
    target_static: torch.Tensor,
    output_path: Path,
    method: str = "pca",
):
    """
    Project all latent types into 2D using PCA or t-SNE and visualize.
    We sample latent vectors across spatial+temporal positions to show the geometry.
    """
    # Sample vectors: for each type, take vectors from all (B, T, H, W) positions
    # Shape: [B, C, T, H, W] → sample positions → [num_samples, C]
    def sample_vectors(tensor, n_samples=2000):
        B, C, T, H_l, W_l = tensor.shape
        total = B * T * H_l * W_l
        indices = torch.randperm(total, device=tensor.device)[:n_samples]
        flat = tensor.permute(0, 2, 3, 4, 1).reshape(-1, C)  # [B*T*H*W, C]
        return flat[indices].cpu().float().numpy()

    # Use fewer samples for t-SNE (it's slow)
    n_samples = 1000 if method == "tsne" else 3000

    vectors_clean = sample_vectors(clean, n_samples)
    vectors_noise = sample_vectors(noise, n_samples)
    vectors_static = sample_vectors(static, n_samples)
    vectors_noisy_noise = sample_vectors(noisy_noise, n_samples)
    vectors_noisy_static = sample_vectors(noisy_static, n_samples)
    vectors_target_noise = sample_vectors(target_noise, n_samples)
    vectors_target_static = sample_vectors(target_static, n_samples)

    all_vectors = np.concatenate([
        vectors_clean, vectors_noise, vectors_static,
        vectors_noisy_noise, vectors_noisy_static,
        vectors_target_noise, vectors_target_static,
    ], axis=0)

    # Dimensionality reduction
    if method == "pca":
        reducer = PCA(n_components=2, random_state=42)
    else:
        reducer = TSNE(n_components=2, random_state=42, perplexity=50, max_iter=1000)

    all_2d = reducer.fit_transform(all_vectors)

    # Split back
    n = n_samples
    results = {
        "Clean latent": all_2d[0*n:1*n],
        "Gaussian noise (ε)": all_2d[1*n:2*n],
        "Static init (s)": all_2d[2*n:3*n],
        "Noisy (noise init)": all_2d[3*n:4*n],
        "Noisy (static init)": all_2d[4*n:5*n],
        "Target (ε - clean)": all_2d[5*n:6*n],
        "Target (s - clean)": all_2d[6*n:7*n],
    }

    colors = {
        "Clean latent": "#2ecc71",
        "Gaussian noise (ε)": "#e74c3c",
        "Static init (s)": "#3498db",
        "Noisy (noise init)": "#e67e22",
        "Noisy (static init)": "#9b59b6",
        "Target (ε - clean)": "#95a5a6",
        "Target (s - clean)": "#1abc9c",
    }

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Left: all points
    ax = axes[0]
    for label, vecs in results.items():
        ax.scatter(vecs[:, 0], vecs[:, 1], s=3, alpha=0.6, label=label, color=colors[label], rasterized=True)
    ax.set_title(f"Latent Space Geometry ({method.upper()})")
    ax.set_xlabel(f"{method.upper()} Component 1")
    ax.set_ylabel(f"{method.upper()} Component 2")
    ax.legend(markerscale=5, loc="upper right", framealpha=0.8)

    # Right: zoom on clean + static + noisy-static (the "good" path)
    ax = axes[1]
    key_groups = ["Clean latent", "Static init (s)", "Noisy (static init)", "Target (s - clean)"]
    for label in key_groups:
        vecs = results[label]
        ax.scatter(vecs[:, 0], vecs[:, 1], s=5, alpha=0.7, label=label, color=colors[label], rasterized=True)
    # Draw arrows from clean to static, clean to noisy_static
    clean_arr = results["Clean latent"]
    static_arr = results["Static init (s)"]
    noisy_static_arr = results["Noisy (static init)"]
    target_static_arr = results["Target (s - clean)"]
    if all(len(a) > 0 for a in [clean_arr, static_arr, noisy_static_arr, target_static_arr]):
        clean_center = clean_arr.mean(0)
        static_center = static_arr.mean(0)
        noisy_static_center = noisy_static_arr.mean(0)
        target_static_center = target_static_arr.mean(0)
        ax.annotate("", xy=static_center, xytext=clean_center,
                    arrowprops=dict(arrowstyle="->", color=colors["Static init (s)"], lw=2, alpha=0.8))
        ax.annotate("", xy=noisy_static_center, xytext=clean_center,
                    arrowprops=dict(arrowstyle="->", color=colors["Noisy (static init)"], lw=2, alpha=0.8))
        ax.annotate("", xy=target_static_center, xytext=clean_center,
                    arrowprops=dict(arrowstyle="->", color=colors["Target (s - clean)"], lw=2, alpha=0.8, ls="--"))
    ax.set_title(f"Static Init Path ({method.upper()}) — Closer to Clean")
    ax.set_xlabel(f"{method.upper()} Component 1")
    ax.set_ylabel(f"{method.upper()} Component 2")
    ax.legend(markerscale=5, loc="upper right", framealpha=0.8)

    plt.suptitle(
        f"Video Latent Space: Noise Init vs Static Frame Init\n"
        f"(Static init is geometrically closer to clean, requiring less denoising)",
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  → Saved {method.upper()} plot: {output_path}")


def plot_distance_barchart(metrics: dict, output_path: Path):
    """Bar chart comparing various distances from clean latent."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: MSE from clean
    ax = axes[0]
    mse_items = [
        ("Noise (ε)", metrics["mse_clean_vs_noise"]),
        ("Static (s)", metrics["mse_clean_vs_static"]),
        ("Noisy\n(noise init)", metrics["mse_clean_vs_noisy_noise"]),
        ("Noisy\n(static init)", metrics["mse_clean_vs_noisy_static"]),
    ]
    labels, values = zip(*mse_items)
    colors_mse = ["#e74c3c", "#3498db", "#e67e22", "#9b59b6"]
    bars = ax.bar(labels, values, color=colors_mse, edgecolor="white", linewidth=0.5)
    ax.set_title("MSE from Clean Latent ↓ (lower is better)")
    ax.set_ylabel("Mean Squared Error")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                f"{val:.4f}", ha="center", fontsize=9, fontweight="bold")

    # Panel 2: Cosine similarity to clean
    ax = axes[1]
    cos_items = [
        ("Noise (ε)", metrics["cosine_clean_vs_noise"]),
        ("Static (s)", metrics["cosine_clean_vs_static"]),
        ("Noisy\n(noise init)", metrics["cosine_clean_vs_noisy_noise"]),
        ("Noisy\n(static init)", metrics["cosine_clean_vs_noisy_static"]),
    ]
    labels_cos, values_cos = zip(*cos_items)
    colors_cos = ["#e74c3c", "#3498db", "#e67e22", "#9b59b6"]
    bars = ax.bar(labels_cos, values_cos, color=colors_cos, edgecolor="white", linewidth=0.5)
    ax.set_title("Cosine Similarity to Clean ↑ (higher is better)")
    ax.set_ylabel("Cosine Similarity")
    ax.set_ylim(0, 1.05)
    for bar, val in zip(bars, values_cos):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", fontsize=9, fontweight="bold")

    # Panel 3: Target (velocity) norm
    ax = axes[2]
    target_items = [
        ("Target\n(ε - clean)", metrics["target_norm_noise_init"]),
        ("Target\n(s - clean)", metrics["target_norm_static_init"]),
    ]
    labels_t, values_t = zip(*target_items)
    colors_t = ["#95a5a6", "#1abc9c"]
    bars = ax.bar(labels_t, values_t, color=colors_t, edgecolor="white", linewidth=0.5)
    ax.set_title("Training Target L2 Norm ↓ (easier to learn)")
    ax.set_ylabel("L2 Norm of Velocity Field")
    for bar, val in zip(bars, values_t):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values_t) * 0.01,
                f"{val:.2f}", ha="center", fontsize=10, fontweight="bold")

    reduction = (1 - metrics["static_vs_noise_target_ratio"]) * 100
    fig.suptitle(
        f"Distance Metrics: Static Init Reduces Target Norm by {reduction:.1f}%\n"
        f"MSE Ratio (static/noise): {metrics['static_vs_noise_mse_ratio']:.4f}  |  "
        f"Target Ratio: {metrics['static_vs_noise_target_ratio']:.4f}",
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  → Saved distance bar chart: {output_path}")


def plot_channel_statistics(
    clean: torch.Tensor,
    noise: torch.Tensor,
    static: torch.Tensor,
    output_path: Path,
):
    """Per-channel mean and variance for clean, noise, and static init."""
    def channel_stats(tensor):
        # tensor: [B, C, T, H, W] → stats per channel
        t = tensor.detach().float()
        mean = t.mean(dim=[0, 2, 3, 4]).cpu().numpy()  # [C]
        std = t.std(dim=[0, 2, 3, 4]).cpu().numpy()     # [C]
        return mean, std

    clean_mean, clean_std = channel_stats(clean)
    noise_mean, noise_std = channel_stats(noise)
    static_mean, static_std = channel_stats(static)

    C = len(clean_mean)
    x = np.arange(C)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Top-left: Channel means
    ax = axes[0, 0]
    ax.plot(x, clean_mean, "o-", label="Clean latent", color="#2ecc71", markersize=3, linewidth=1)
    ax.plot(x, noise_mean, "s-", label="Gaussian noise (ε)", color="#e74c3c", markersize=3, linewidth=1)
    ax.plot(x, static_mean, "d-", label="Static init (s)", color="#3498db", markersize=3, linewidth=1)
    ax.set_title("Per-Channel Mean")
    ax.set_xlabel("Channel Index")
    ax.set_ylabel("Mean Value")
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # Top-right: Channel std
    ax = axes[0, 1]
    ax.plot(x, clean_std, "o-", label="Clean latent", color="#2ecc71", markersize=3, linewidth=1)
    ax.plot(x, noise_std, "s-", label="Gaussian noise (ε)", color="#e74c3c", markersize=3, linewidth=1)
    ax.plot(x, static_std, "d-", label="Static init (s)", color="#3498db", markersize=3, linewidth=1)
    ax.set_title("Per-Channel Standard Deviation")
    ax.set_xlabel("Channel Index")
    ax.set_ylabel("Std Dev")
    ax.legend()

    # Bottom-left: Mean error vs clean
    ax = axes[1, 0]
    ax.bar(x - 0.2, noise_mean - clean_mean, 0.4, label="Noise error", color="#e74c3c", alpha=0.7)
    ax.bar(x + 0.2, static_mean - clean_mean, 0.4, label="Static error", color="#3498db", alpha=0.7)
    ax.set_title("Per-Channel Mean Error (vs Clean)")
    ax.set_xlabel("Channel Index")
    ax.set_ylabel("Mean Difference from Clean")
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # Bottom-right: Std error vs clean
    ax = axes[1, 1]
    ax.bar(x - 0.2, noise_std - clean_std, 0.4, label="Noise error", color="#e74c3c", alpha=0.7)
    ax.bar(x + 0.2, static_std - clean_std, 0.4, label="Static error", color="#3498db", alpha=0.7)
    ax.set_title("Per-Channel Std Dev Error (vs Clean)")
    ax.set_xlabel("Channel Index")
    ax.set_ylabel("Std Dev Difference from Clean")
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    mae_mean_noise = np.abs(noise_mean - clean_mean).mean()
    mae_mean_static = np.abs(static_mean - clean_mean).mean()
    fig.suptitle(
        f"Channel Statistics: Static Init Much Closer to Clean\n"
        f"Mean MAE: noise={mae_mean_noise:.4f}, static={mae_mean_static:.4f} "
        f"(static is {mae_mean_noise/mae_mean_static:.1f}x better)",
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  → Saved channel statistics: {output_path}")


def plot_spatial_error_map(
    clean: torch.Tensor,
    noise: torch.Tensor,
    static: torch.Tensor,
    noisy_noise: torch.Tensor,
    noisy_static: torch.Tensor,
    output_path: Path,
    frame_idx: int = 1,
):
    """
    Show spatial error maps (per-latent-position difference from clean)
    for a specific frame.
    """
    # Take a single frame, average over channels
    def frame_error(tensor, clean_t, f_idx):
        # tensor: [B, C, T, H, W] → take frame f_idx, average |diff| over C
        diff = (tensor - clean_t).abs()
        frame_diff = diff[0, :, f_idx, :, :].mean(0).cpu().float().numpy()  # [H, W]
        return frame_diff

    errors = {
        "Noise (ε)": frame_error(noise, clean, frame_idx),
        "Static (s)": frame_error(static, clean, frame_idx),
        "Noisy (noise init)": frame_error(noisy_noise, clean, frame_idx),
        "Noisy (static init)": frame_error(noisy_static, clean, frame_idx),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    vmin = min(e.min() for e in errors.values())
    vmax = max(e.max() for e in errors.values())

    for ax, (label, err_map) in zip(axes.flat, errors.items()):
        im = ax.imshow(err_map, cmap="hot", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(f"|Error|: {label}", fontweight="bold")
        ax.set_xlabel("Latent Width")
        ax.set_ylabel("Latent Height")
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(
        f"Spatial Error Maps (Latent Frame {frame_idx}) — Mean |diff| over Channels\n"
        f"Static init error is concentrated; noise error is uniformly high",
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  → Saved spatial error maps: {output_path}")


def plot_value_distribution(
    clean: torch.Tensor,
    noise: torch.Tensor,
    static: torch.Tensor,
    output_path: Path,
):
    """Histogram/KDE of latent values."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    def get_values(tensor, n=50000):
        vals = tensor.detach().float().cpu().numpy().ravel()
        if len(vals) > n:
            indices = np.random.choice(len(vals), n, replace=False)
            vals = vals[indices]
        return vals

    # Panel 1: Full distribution
    ax = axes[0]
    for label, tensor, color in [
        ("Clean latent", clean, "#2ecc71"),
        ("Gaussian noise (ε)", noise, "#e74c3c"),
        ("Static init (s)", static, "#3498db"),
    ]:
        vals = get_values(tensor)
        ax.hist(vals, bins=80, density=True, alpha=0.4, label=label, color=color)
    ax.set_title("Latent Value Distribution")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()

    # Panel 2: Zoom on clean range
    ax = axes[1]
    clean_vals = get_values(clean)
    lo, hi = np.percentile(clean_vals, [0.5, 99.5])
    for label, tensor, color in [
        ("Clean latent", clean, "#2ecc71"),
        ("Gaussian noise (ε)", noise, "#e74c3c"),
        ("Static init (s)", static, "#3498db"),
    ]:
        vals = get_values(tensor)
        ax.hist(vals, bins=80, density=True, alpha=0.4, label=label, color=color, range=(lo, hi))
    ax.set_title(f"Distribution (Zoomed to Clean Range [{lo:.2f}, {hi:.2f}])")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()

    # Compute overlap metrics
    clean_v = get_values(clean)
    static_v = get_values(static)
    noise_v = get_values(noise)
    from scipy import stats
    ks_static = stats.ks_2samp(clean_v, static_v).statistic
    ks_noise = stats.ks_2samp(clean_v, noise_v).statistic

    fig.suptitle(
        f"Latent Value Distributions: Static Init Much Better Overlap with Clean\n"
        f"KS distance to clean: noise={ks_noise:.4f}, static={ks_static:.4f} "
        f"(static is {ks_noise/ks_static:.1f}x closer)",
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  → Saved value distributions: {output_path}")


def plot_summary_dashboard(
    metrics: dict,
    clean: torch.Tensor,
    noise: torch.Tensor,
    static: torch.Tensor,
    noisy_noise: torch.Tensor,
    noisy_static: torch.Tensor,
    output_path: Path,
    sigma: float,
):
    """One-page summary dashboard with key insights."""
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

    # --- Row 1, Col 1: Key numbers ---
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    reduction_mse = (1 - metrics["static_vs_noise_mse_ratio"]) * 100
    reduction_target = (1 - metrics["static_vs_noise_target_ratio"]) * 100
    summary_text = f"""
    ╔══════════════════════════╗
    ║   STATIC INIT ADVANTAGE  ║
    ╠══════════════════════════╣
    ║                          ║
    ║  MSE to clean:           ║
    ║    Noise:  {metrics['mse_clean_vs_noise']:>10.4f}  ║
    ║    Static: {metrics['mse_clean_vs_static']:>10.4f}  ║
    ║    → {reduction_mse:>6.1f}% reduction     ║
    ║                          ║
    ║  Cosine sim to clean:    ║
    ║    Noise:  {metrics['cosine_clean_vs_noise']:>10.4f}  ║
    ║    Static: {metrics['cosine_clean_vs_static']:>10.4f}  ║
    ║                          ║
    ║  Target L2 norm:         ║
    ║    Noise:  {metrics['target_norm_noise_init']:>10.2f}  ║
    ║    Static: {metrics['target_norm_static_init']:>10.2f}  ║
    ║    → {reduction_target:>6.1f}% reduction     ║
    ║                          ║
    ║  Noisy MSE to clean:     ║
    ║    Noise init:  {metrics['mse_clean_vs_noisy_noise']:>8.4f} ║
    ║    Static init: {metrics['mse_clean_vs_noisy_static']:>8.4f} ║
    ║                          ║
    ║  σ = {sigma:.4f}                ║
    ╚══════════════════════════╝
    """
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#ecf0f1", alpha=0.8))

    # --- Row 1, Col 2-3: All-in-one PCA ---
    ax = fig.add_subplot(gs[0, 1:])
    _quick_pca_plot(ax, clean, noise, static, noisy_noise, noisy_static)

    # --- Row 2, Col 1: MSE bar chart (mini) ---
    ax = fig.add_subplot(gs[1, 0])
    _mini_mse_bars(ax, metrics)

    # --- Row 2, Col 2: Cosine sim bar chart (mini) ---
    ax = fig.add_subplot(gs[1, 1])
    _mini_cosine_bars(ax, metrics)

    # --- Row 2, Col 3: Target norm bar chart (mini) ---
    ax = fig.add_subplot(gs[1, 2])
    _mini_target_bars(ax, metrics)

    # --- Row 3, Col 1-3: Channel mean comparison ---
    ax = fig.add_subplot(gs[2, :])
    _mini_channel_comparison(ax, clean, noise, static)

    fig.suptitle(
        f"Video Denoising Initialization Analysis\n"
        f"Static Frame Init vs Gaussian Noise Init — Flow Matching (σ={sigma:.4f})",
        fontweight="bold", fontsize=16,
    )
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Saved summary dashboard: {output_path}")


def _quick_pca_plot(ax, clean, noise, static, noisy_noise, noisy_static):
    """Inline PCA for dashboard."""
    def sample(t, n=800):
        B, C, T_l, H_l, W_l = t.shape
        flat = t.permute(0, 2, 3, 4, 1).reshape(-1, C).cpu().float().numpy()
        idx = np.random.choice(len(flat), min(n, len(flat)), replace=False)
        return flat[idx]

    all_data = np.concatenate([
        sample(clean), sample(noise), sample(static),
        sample(noisy_noise), sample(noisy_static),
    ])
    pca = PCA(n_components=2, random_state=42)
    all_2d = pca.fit_transform(all_data)
    n = min(800, all_data.shape[0] // 5)
    groups = {
        "Clean": (all_2d[0*n:1*n], "#2ecc71"),
        "Noise (ε)": (all_2d[1*n:2*n], "#e74c3c"),
        "Static (s)": (all_2d[2*n:3*n], "#3498db"),
        "Noisy (noise)": (all_2d[3*n:4*n], "#e67e22"),
        "Noisy (static)": (all_2d[4*n:5*n], "#9b59b6"),
    }
    for label, (vecs, color) in groups.items():
        ax.scatter(vecs[:, 0], vecs[:, 1], s=2, alpha=0.5, label=label, color=color, rasterized=True)
    ax.set_title("PCA: Latent Space Overview", fontweight="bold")
    ax.legend(markerscale=4, fontsize=7, loc="upper right")
    ax.set_xticks([])
    ax.set_yticks([])


def _mini_mse_bars(ax, metrics):
    items = [
        ("Noise ε", metrics["mse_clean_vs_noise"], "#e74c3c"),
        ("Static s", metrics["mse_clean_vs_static"], "#3498db"),
        ("Noisy\n(noise)", metrics["mse_clean_vs_noisy_noise"], "#e67e22"),
        ("Noisy\n(static)", metrics["mse_clean_vs_noisy_static"], "#9b59b6"),
    ]
    labels, values, colors = zip(*items)
    ax.bar(labels, values, color=colors, edgecolor="white")
    ax.set_title("MSE from Clean ↓", fontweight="bold")
    ax.tick_params(axis="x", labelsize=8)


def _mini_cosine_bars(ax, metrics):
    items = [
        ("Noise ε", metrics["cosine_clean_vs_noise"], "#e74c3c"),
        ("Static s", metrics["cosine_clean_vs_static"], "#3498db"),
        ("Noisy\n(noise)", metrics["cosine_clean_vs_noisy_noise"], "#e67e22"),
        ("Noisy\n(static)", metrics["cosine_clean_vs_noisy_static"], "#9b59b6"),
    ]
    labels, values, colors = zip(*items)
    ax.bar(labels, values, color=colors, edgecolor="white")
    ax.set_title("Cosine Sim to Clean ↑", fontweight="bold")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", labelsize=8)


def _mini_target_bars(ax, metrics):
    items = [
        ("ε - clean", metrics["target_norm_noise_init"], "#95a5a6"),
        ("s - clean", metrics["target_norm_static_init"], "#1abc9c"),
    ]
    labels, values, colors = zip(*items)
    ax.bar(labels, values, color=colors, edgecolor="white")
    ax.set_title("Target L2 Norm ↓", fontweight="bold")
    ax.tick_params(axis="x", labelsize=8)


def _mini_channel_comparison(ax, clean, noise, static):
    clean_mean = clean.mean(dim=[0, 2, 3, 4]).cpu().numpy()
    noise_mean = noise.mean(dim=[0, 2, 3, 4]).cpu().numpy()
    static_mean = static.mean(dim=[0, 2, 3, 4]).cpu().numpy()
    C = len(clean_mean)
    x = np.arange(C)
    ax.plot(x, clean_mean, "o-", label="Clean", color="#2ecc71", markersize=2, linewidth=1)
    ax.plot(x, noise_mean, "s-", label="Noise (ε)", color="#e74c3c", markersize=2, linewidth=1)
    ax.plot(x, static_mean, "d-", label="Static (s)", color="#3498db", markersize=2, linewidth=1)
    ax.set_title("Per-Channel Mean Values", fontweight="bold")
    ax.set_xlabel("Channel Index")
    ax.set_ylabel("Mean")
    ax.legend(fontsize=8)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)


# ==============================================================================
# Main Experiment
# ==============================================================================

def plot_sigma_sweep(
    clean: torch.Tensor,
    noise: torch.Tensor,
    static: torch.Tensor,
    output_path: Path,
    num_train_timesteps: int,
    shift: float,
    n_points: int = 50,
):
    """
    Show how the advantage of static init varies across the noise schedule (σ from 0 to 1).

    Key insight: at σ≈0.999 (one-step video training), static init has the maximum advantage.
    """
    sigmas = np.linspace(0.0, 1.0, n_points)

    mse_noise_init = []
    mse_static_init = []
    cos_noise_init = []
    cos_static_init = []
    target_norm_noise = []
    target_norm_static = []

    for sigma in sigmas:
        noisy_n = add_noise(clean, noise, float(sigma))
        noisy_s = add_noise(clean, static, float(sigma))

        mse_noise_init.append(torch.mean((clean - noisy_n) ** 2).item())
        mse_static_init.append(torch.mean((clean - noisy_s) ** 2).item())

        # Cosine similarity
        c_flat = clean.reshape(-1)
        cos_noise_init.append(torch.dot(c_flat, noisy_n.reshape(-1)).item() /
                              (torch.norm(c_flat).item() * torch.norm(noisy_n.reshape(-1)).item() + 1e-8))
        cos_static_init.append(torch.dot(c_flat, noisy_s.reshape(-1)).item() /
                               (torch.norm(c_flat).item() * torch.norm(noisy_s.reshape(-1)).item() + 1e-8))

        # Target norms
        t_n = noise - clean
        t_s = static - clean
        target_norm_noise.append(torch.norm(t_n.reshape(-1)).item())
        target_norm_static.append(torch.norm(t_s.reshape(-1)).item())

    target_sigma = compute_target_sigma(num_train_timesteps, shift)

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # Panel 1: MSE(clean, noisy) vs sigma
    ax = axes[0, 0]
    ax.plot(sigmas, mse_noise_init, label="Noise init", color="#e74c3c", linewidth=2)
    ax.plot(sigmas, mse_static_init, label="Static init", color="#3498db", linewidth=2)
    ax.axvline(x=target_sigma, color="gray", linestyle="--", alpha=0.7, label=f"1-step σ={target_sigma:.4f}")
    ax.set_xlabel("σ (noise level)")
    ax.set_ylabel("MSE from Clean")
    ax.set_title("Noisy Latent Error vs σ")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: MSE ratio (static/noise) vs sigma
    ax = axes[0, 1]
    mse_ratio = [s / (n + 1e-8) for s, n in zip(mse_static_init, mse_noise_init)]
    ax.plot(sigmas, mse_ratio, color="#9b59b6", linewidth=2)
    ax.axvline(x=target_sigma, color="gray", linestyle="--", alpha=0.7, label=f"1-step σ={target_sigma:.4f}")
    ax.axhline(y=1.0, color="black", linestyle=":", alpha=0.5)
    ax.set_xlabel("σ (noise level)")
    ax.set_ylabel("MSE Ratio (static / noise)")
    ax.set_title("Static Advantage vs σ (lower = better)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Cosine similarity vs sigma
    ax = axes[0, 2]
    ax.plot(sigmas, cos_noise_init, label="Noise init → clean", color="#e74c3c", linewidth=2)
    ax.plot(sigmas, cos_static_init, label="Static init → clean", color="#3498db", linewidth=2)
    ax.axvline(x=target_sigma, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("σ (noise level)")
    ax.set_ylabel("Cosine Similarity to Clean")
    ax.set_title("Cosine Similarity vs σ")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: Log-scale MSE
    ax = axes[1, 0]
    ax.semilogy(sigmas, mse_noise_init, label="Noise init", color="#e74c3c", linewidth=2)
    ax.semilogy(sigmas, mse_static_init, label="Static init", color="#3498db", linewidth=2)
    ax.axvline(x=target_sigma, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("σ (noise level)")
    ax.set_ylabel("MSE from Clean (log scale)")
    ax.set_title("Log-Scale: Noisy Latent Error vs σ")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 5: Target norm comparison
    ax = axes[1, 1]
    ax.bar(["Noise init\n(ε-clean)", "Static init\n(s-clean)"],
           [target_norm_noise[0], target_norm_static[0]],
           color=["#e74c3c", "#3498db"], edgecolor="white")
    ax.set_title("Training Target L2 Norm\n(independent of σ)")
    ax.set_ylabel("L2 Norm")

    # Panel 6: Summary text
    ax = axes[1, 2]
    ax.axis("off")
    reduction = (1 - target_norm_static[0] / target_norm_noise[0]) * 100
    mse_red = (1 - mse_static_init[-1] / (mse_noise_init[-1] + 1e-8)) * 100
    summary = f"""
    ╔══════════════════════════╗
    ║    SIGMA SWEEP SUMMARY   ║
    ╠══════════════════════════╣
    ║                          ║
    ║  σ=0:   Both identical   ║
    ║         (no difference)  ║
    ║                          ║
    ║  σ=1:   Max difference   ║
    ║         Noisy = init     ║
    ║                          ║
    ║  1-step training σ:      ║
    ║  {target_sigma:.4f}               ║
    ║                          ║
    ║  Target norm reduction:  ║
    ║  {reduction:>6.1f}%              ║
    ║                          ║
    ║  Noisy MSE reduction:    ║
    ║  {mse_red:>6.1f}%              ║
    ║                          ║
    ║  → Static init advantage ║
    ║    is MAXIMAL at the     ║
    ║    training σ            ║
    ╚══════════════════════════╝
    """
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#ecf0f1", alpha=0.8))

    fig.suptitle(
        "Static Init Advantage Across the Noise Schedule\n"
        "At σ=0, inits are identical (no noise). At σ→1, advantage is maximal.",
        fontweight="bold", fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  → Saved sigma sweep: {output_path}")


def run_experiment(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Setup
    output_dir = setup_output_dir(args.output_dir)
    set_plotting_style()

    # Compute sigma (the noise level)
    target_sigma = compute_target_sigma(args.num_train_timesteps, args.shift)
    sigma = args.sigma if args.sigma is not None else target_sigma
    print(f"\n{'='*60}")
    print(f"Flow Matching Sigma: {sigma:.6f}")
    print(f"Num train timesteps: {args.num_train_timesteps}, Shift: {args.shift}")
    print(f"{'='*60}\n")

    use_synthetic = args.synthetic or args.vae_path is None

    if use_synthetic:
        # --- SYNTHETIC MODE: use realistic simulated latents ---
        H_lat = args.image_height // args.spatial_scale
        W_lat = args.image_width // args.spatial_scale
        T_lat = 1 + (args.num_frames - 1) // 4  # temporal compression

        print(f"Using SYNTHETIC latents (no VAE needed)")
        print(f"  Latent shape: [{args.batch_size}, {args.z_dim}, {T_lat}, {H_lat}, {W_lat}]")

        synth = generate_synthetic_latents(
            batch_size=args.batch_size,
            z_dim=args.z_dim,
            num_latent_frames=T_lat,
            latent_h=H_lat,
            latent_w=W_lat,
            seed=args.seed,
            device=device,
        )
        clean_latents = synth["clean"]
        noise = synth["noise"]
        static_init = synth["static"]
        print(f"  Clean latents:  mean={clean_latents.mean().item():.4f}, std={clean_latents.std().item():.4f}")
        print(f"  Noise:          mean={noise.mean().item():.4f}, std={noise.std().item():.4f}")
        print(f"  Static init:    mean={static_init.mean().item():.4f}, std={static_init.std().item():.4f}")

    else:
        # --- REAL MODE: load VAE and encode real/synthetic video ---
        print("Loading VAE...")
        vae = load_wan_vae(args.vae_path, args.z_dim, device=device)
        print(f"  VAE loaded on {device}")

        print("\nLoading video data...")
        videos = load_sample_video(
            args.video_path, args.num_frames,
            args.image_height, args.image_width,
            args.batch_size, device=device,
        )
        print(f"  Video shape: {tuple(videos.shape)} (B, C, T, H, W)")

        print("\nEncoding video to clean latents...")
        with torch.no_grad():
            clean_latents = vae.encode(videos, tiled=True, tile_size=(34, 34), tile_stride=(18, 16))
        print(f"  Clean latents shape: {tuple(clean_latents.shape)} (B, z_dim, T_lat, H_lat, W_lat)")
        T_lat = clean_latents.shape[2]
        print(f"  Temporal compression: {args.num_frames} pixel frames → {T_lat} latent frames")

        torch.manual_seed(args.seed)
        noise = torch.randn_like(clean_latents)
        print(f"  Noise shape: {tuple(noise.shape)}")

        print("\nBuilding static frame initialization...")
        static_init = compute_static_init_latents(
            vae, videos, clean_latents,
            num_frame_per_block=args.num_frame_per_block,
            tiled=True, tile_size=(34, 34), tile_stride=(18, 16),
        )
        print(f"  Static init shape: {tuple(static_init.shape)}")

    # Simulate forward process at the given sigma
    print(f"\nSimulating flow-matching forward process (σ={sigma:.4f})...")
    noisy_noise_init = add_noise(clean_latents, noise, sigma)
    noisy_static_init = add_noise(clean_latents, static_init, sigma)

    # Compute training targets
    target_noise_init = compute_training_target(clean_latents, noise)
    target_static_init = compute_training_target(clean_latents, static_init)

    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(
        clean_latents, noise, static_init,
        noisy_noise_init, noisy_static_init,
        target_noise_init, target_static_init,
    )

    tl = clean_latents.shape[2]
    print(f"\n{'='*60}")
    print("KEY METRICS:")
    print(f"{'='*60}")
    print(f"  MSE(clean, noise):              {metrics['mse_clean_vs_noise']:.6f}")
    print(f"  MSE(clean, static):             {metrics['mse_clean_vs_static']:.6f}")
    print(f"    → Static is {metrics['mse_clean_vs_noise'] / (metrics['mse_clean_vs_static'] + 1e-8):.1f}x closer")
    print(f"  MSE(clean, noisy_noise):        {metrics['mse_clean_vs_noisy_noise']:.6f}")
    print(f"  MSE(clean, noisy_static):       {metrics['mse_clean_vs_noisy_static']:.6f}")
    print(f"  Cosine(clean, noise):           {metrics['cosine_clean_vs_noise']:.6f}")
    print(f"  Cosine(clean, static):          {metrics['cosine_clean_vs_static']:.6f}")
    print(f"  Target norm (noise init):       {metrics['target_norm_noise_init']:.2f}")
    print(f"  Target norm (static init):      {metrics['target_norm_static_init']:.2f}")
    print(f"    → Target reduction: {(1 - metrics['static_vs_noise_target_ratio']) * 100:.1f}%")
    print(f"  Noisy reduction:                {(1 - metrics['noisy_static_vs_noisy_noise_mse_ratio']) * 100:.1f}%")
    print(f"{'='*60}\n")

    # Generate visualizations
    print("Generating visualizations...")

    # PCA latent space
    plot_latent_space_2d(
        clean_latents, noise, static_init,
        noisy_noise_init, noisy_static_init,
        target_noise_init, target_static_init,
        output_dir / "latent_space_pca.png",
        method="pca",
    )

    # t-SNE latent space (skip if too large)
    if tl * clean_latents.shape[3] * clean_latents.shape[4] < 50000:
        plot_latent_space_2d(
            clean_latents, noise, static_init,
            noisy_noise_init, noisy_static_init,
            target_noise_init, target_static_init,
            output_dir / "latent_space_tsne.png",
            method="tsne",
        )
    else:
        print("  Skipping t-SNE (latent grid too large)")

    # Distance bar charts
    plot_distance_barchart(metrics, output_dir / "distance_metrics.png")

    # Channel statistics
    plot_channel_statistics(
        clean_latents, noise, static_init,
        output_dir / "channel_statistics.png",
    )

    # Spatial error maps (for a future frame)
    future_frame_idx = min(1, tl - 1)
    plot_spatial_error_map(
        clean_latents, noise, static_init,
        noisy_noise_init, noisy_static_init,
        output_dir / "spatial_error_maps.png",
        frame_idx=future_frame_idx,
    )

    # Value distributions
    plot_value_distribution(
        clean_latents, noise, static_init,
        output_dir / "value_distributions.png",
    )

    # Summary dashboard
    plot_summary_dashboard(
        metrics, clean_latents, noise, static_init,
        noisy_noise_init, noisy_static_init,
        output_dir / "summary_dashboard.png",
        sigma=sigma,
    )

    # Sigma sweep: how does the advantage change with noise level?
    plot_sigma_sweep(
        clean_latents, noise, static_init,
        output_dir / "sigma_sweep.png",
        args.num_train_timesteps,
        args.shift,
    )

    # Save metrics as JSON
    import json
    metrics_json = {}
    for k, v in metrics.items():
        if isinstance(v, (np.floating, float)):
            metrics_json[k] = float(v)
        elif isinstance(v, torch.Tensor):
            metrics_json[k] = float(v.detach().cpu().item())
        else:
            metrics_json[k] = v
    metrics_json["sigma"] = sigma
    metrics_json["num_train_timesteps"] = args.num_train_timesteps
    metrics_json["shift"] = args.shift
    metrics_json["num_frame_per_block"] = args.num_frame_per_block
    metrics_json["latent_frames"] = tl
    metrics_json["z_dim"] = args.z_dim
    metrics_json["mode"] = "synthetic" if use_synthetic else "real_vae"
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"\n  → Saved metrics: {output_dir / 'metrics.json'}")

    print(f"\n{'='*60}")
    print(f"All visualizations saved to: {output_dir}")
    print(f"{'='*60}")
    return metrics


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
