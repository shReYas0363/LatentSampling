#!/usr/bin/env python3
"""
visualize_zero_shot_reconstruction.py
--------------------------------------
Loads a trained BetaTCVAE checkpoint, extracts 15 samples from the
zero-shot (held-out) split of the dSprites dataset, encodes and decodes them,
and plots them .
"""

import os
import sys
import argparse
from typing import Tuple, Optional, Dict, Any

import numpy as np
import torch
import matplotlib.pyplot as plt

# Ensure current directory is in sys.path for imports
if "." not in sys.path:
    sys.path.insert(0, ".")

from train_vae import BetaTCVAE


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize BetaTCVAE reconstructions on the zero-shot dSprites set."
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="FinalMods/betatcvae_dim10_beta4_LH_seed2.pth",
        help="Path to the trained VAE checkpoint (.pth file)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz",
        help="Path to dSprites .npz dataset file",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=15,
        help="Number of zero-shot samples to reconstruct (default: 15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample selection reproducibility",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="zero_shot_reconstructions.png",
        help="Output image path for the comparison visualization",
    )
    return parser.parse_args()


def load_vae_model(checkpoint_path: str, device: torch.device) -> BetaTCVAE:
    if not os.path.exists(checkpoint_path):
        candidates = [
            "betatcvae_dim10_beta4_OnlySquareonLeft.pth",
        ]
        for cand in candidates:
            if os.path.exists(cand):
                print(f"[*] Checkpoint '{checkpoint_path}' not found. Using '{cand}' instead.")
                checkpoint_path = cand
                break

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    print(f"[*] Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "args" in checkpoint and hasattr(checkpoint["args"], "latent_dim"):
        latent_dim = checkpoint["args"].latent_dim
    elif "args" in checkpoint and isinstance(checkpoint["args"], dict) and "latent_dim" in checkpoint["args"]:
        latent_dim = checkpoint["args"]["latent_dim"]
    elif "model_state_dict" in checkpoint and "encoder.lin1.weight" in checkpoint["model_state_dict"]:
        latent_dim = checkpoint["model_state_dict"]["encoder.mu_logvar_gen.weight"].shape[0] // 2
    else:
        latent_dim = 10 if "dim10" in checkpoint_path else 12

    print(f"[*] Initializing BetaTCVAE (latent_dim={latent_dim}) on {device}...")
    model = BetaTCVAE(latent_dim=latent_dim).to(device)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def get_zero_shot_samples(
    data_path: str, num_samples: int = 15, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts zero-shot test samples from dSprites.
    Zero-shot set matches train_vae.py exclusions:
    - Shapes: Ellipses (shape=1) and Hearts (shape=2)
    - Position: Left half of screen (pos_x < 16)
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    print(f"[*] Loading dataset from: {data_path}")
    data = np.load(data_path, allow_pickle=True, encoding="bytes")
    raw_images = data["imgs"]              # Shape: [N, 64, 64]
    latents_classes = data["latents_classes"]  # Shape: [N, 6]
    # latents_classes columns: [color(0), shape(1), scale(2), orientation(3), pos_x(4), pos_y(5)]

    shape_classes = latents_classes[:, 1]
    scale_classes = latents_classes[:, 2]
    pos_x_classes = latents_classes[:, 4]

    # Zero-shot mask: Ellipse (1) or Heart (2) located on left side (pos_x < 16)
    zero_shot_mask = (shape_classes == 2 ) & ((scale_classes ==2) | (scale_classes ==3) ) & (pos_x_classes<16)
    zero_shot_indices = np.where(zero_shot_mask)[0]

    print(f"[*] Total zero-shot samples in dataset: {len(zero_shot_indices):,}")

    # Deterministic sampling
    rng = np.random.RandomState(seed)
    chosen_indices = rng.choice(
        zero_shot_indices,
        size=min(num_samples, len(zero_shot_indices)),
        replace=False
    )
    samples = raw_images[chosen_indices]
    sample_classes = latents_classes[chosen_indices]

    return samples, sample_classes


def create_side_by_side_bordered_canvas(
    orig: np.ndarray,
    recon: np.ndarray,
    border_width: int = 2,
    divider_width: int = 3,
    border_val: float = 0.35,
    divider_val: float = 0.85,
) -> np.ndarray:
    """
    Combines an original image and its reconstruction into a single canvas with:
    - Solid outer bounding border (so 64x64 boundaries are clearly visible against dark backgrounds)
    - Distinct central divider line between Original (Left) and Reconstruction (Right)
    """
    H, W = orig.shape  # 64, 64
    b = border_width
    d = divider_width

    canvas_h = H + 2 * b
    canvas_w = 2 * W + 2 * b + d

    # Initialize canvas with outer border color
    canvas = np.full((canvas_h, canvas_w), fill_value=border_val, dtype=np.float32)

    # Insert Original on the left
    canvas[b : b + H, b : b + W] = orig

    # Draw vertical divider
    div_start = b + W
    canvas[:, div_start : div_start + d] = divider_val

    # Insert Reconstruction on the right
    recon_start = div_start + d
    canvas[b : b + H, recon_start : recon_start + W] = recon

    return canvas


def plot_and_save_visualizations(
    orig_imgs: np.ndarray,
    recon_imgs: np.ndarray,
    sample_classes: np.ndarray,
    output_path: str,
):
    """
    Generates a high-quality 3-row x 5-column figure where each panel displays
    a clearly framed side-by-side [Original | Reconstruction] pair.
    """
    shape_names = {0: "Square", 1: "Ellipse", 2: "Heart"}
    n_samples = len(orig_imgs)
    n_cols = 5
    n_rows = int(np.ceil(n_samples / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.4 * n_rows), dpi=180)
    fig.patch.set_facecolor("#121212")

    # Overall header
    fig.suptitle(
        "Beta-TCVAE Zero-Shot Reconstructions\n"
        "[ Left: Original Ground Truth (Zero-Shot)  |  Right: VAE Reconstruction ]",
        fontsize=15,
        fontweight="bold",
        color="#FFFFFF",
        y=0.98,
    )

    for i in range(n_rows * n_cols):
        ax = axes[i // n_cols, i % n_cols]
        if i < n_samples:
            pair_canvas = create_side_by_side_bordered_canvas(
                orig_imgs[i],
                recon_imgs[i],
                border_width=2,
                divider_width=3,
                border_val=0.30,
                divider_val=0.90,
            )

            ax.imshow(pair_canvas, cmap="gray", vmin=0.0, vmax=1.0)

            # Metadata label for each sample
            shape_idx = sample_classes[i, 1]
            scale_idx = sample_classes[i, 2]
            pos_x_idx = sample_classes[i, 4]
            shape_str = shape_names.get(shape_idx, f"Shape {shape_idx}")

            ax.set_title(
                f"Sample #{i+1}: {shape_str}\n(Scale: {scale_idx}, PosX: {pos_x_idx})",
                fontsize=9,
                fontweight="semibold",
                color="#E0E0E0",
                pad=6,
            )

            # Draw outer highlight spine around each subplot
            for spine in ax.spines.values():
                spine.set_color("#4A90E2")
                spine.set_linewidth(1.5)
        else:
            ax.axis("off")

        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])

    # Save figure
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"[✓] Successfully saved comparison visualization to: {output_path}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Execution device: {device}")

    # 1. Load VAE model
    vae = load_vae_model(args.checkpoint_path, device)

    # 2. Extract zero-shot samples
    orig_samples_np, sample_classes = get_zero_shot_samples(
        args.data_path, num_samples=args.num_samples, seed=args.seed
    )

    # 3. Prepare inputs: [N, 1, 64, 64]
    input_tensors = torch.from_numpy(orig_samples_np).float().unsqueeze(1).to(device)

    # 4. Encode & Decode through VAE
    print("[*] Encoding and decoding zero-shot samples...")
    with torch.no_grad():
        mu, logvar = vae.encode(input_tensors)
        recon_logits = vae.decode(mu)
        recon_imgs_np = torch.sigmoid(recon_logits).squeeze(1).cpu().numpy()

    # 5. Plot & Save with clear image borders
    print("[*] Rendering side-by-side comparison with borders...")
    plot_and_save_visualizations(
        orig_imgs=orig_samples_np,
        recon_imgs=recon_imgs_np,
        sample_classes=sample_classes,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
