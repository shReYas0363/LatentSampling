#!/usr/bin/env python3
"""
For each sampling method, decode the last-trial samples through the VAE,
run the shape classifier on every decoded image, and draw a histogram of
predicted shape classes (0-Square, 1-Ellipse, 2-Heart).

Each row shows the sampler name on the left and its histogram on the right,
mirroring the layout of visualize_methods_comparison.py.

Usage:
    python visualize_shape_histogram.py \
        --results-root ./results_2a_MidSquare \
        --combo-name 2_square_mid \
        --vae-checkpoint FinalMods/betatcvae_dim10_beta4_MidScaleSq.pth \
        --latent-dim 10 \
        --title "Shape Distribution – Medium-Scale Squares"
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import importlib


# ── helpers ─────────────────────────────────────────────────────────────────

def load_samples_from_json(json_path: str) -> torch.Tensor:
    """Load the last-trial latent vectors from a *_last_trial_final_samples.json."""
    with open(json_path, "r") as f:
        data = json.load(f)
    z = torch.tensor(data["last_trial_final_samples"], dtype=torch.float32)
    return z


def load_results_json(json_path: str) -> dict:
    with open(json_path, "r") as f:
        return json.load(f)


def load_vae(vae_checkpoint: str, latent_dim: int, device: str):
    train_vae = importlib.import_module("train_vae")
    model = train_vae.BetaTCVAE(latent_dim=latent_dim).to(device)
    ckpt = torch.load(vae_checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.eval()
    return model


def load_shape_classifier(checkpoint_path: str, device: str):
    """Load the ShapeNet classifier (3 classes: square, ellipse, heart)."""
    classifier_module = importlib.import_module("classifier")
    model = classifier_module.ShapeNet(num_classes=3).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    model.eval()
    return model


def decode_to_images(z, decode, device):
    """Decode latent vectors to normalised [0,1] images. Returns (N, 1, H, W) tensor."""
    with torch.inference_mode():
        imgs = torch.sigmoid(decode(z.to(device))).detach().float().cpu()
    # Per-image min-max normalisation (matches sampling_common convention)
    imgs_min = imgs.amin(dim=(1, 2, 3), keepdim=True)
    imgs_max = imgs.amax(dim=(1, 2, 3), keepdim=True)
    imgs = (imgs - imgs_min) / (imgs_max - imgs_min).clamp_min(1e-8)
    return imgs  # (N, 1, H, W)


def classify_shapes(imgs_tensor, shape_classifier, device):
    """Run the shape classifier on decoded images.

    Returns
    -------
    predicted_classes : np.ndarray (N,)  – argmax class per image
    confidences       : np.ndarray (N, 3) – softmax probabilities per class
    """
    with torch.inference_mode():
        logits = shape_classifier(imgs_tensor.to(device)).float()
        probs = F.softmax(logits, dim=1).cpu().numpy()
    predicted = probs.argmax(axis=1)
    return predicted, probs




METHOD_SPECS = [
    {
        "label": "Rejection Sampling",
        "dir_name": "rejection",
        "samples_file": "rejection_last_trial_final_samples.json",
        "results_file": "rejection_results.json",
    },
    {
        "label": "RW-MH",
        "dir_name": "mh",
        "samples_file": "mh_last_trial_final_samples.json",
        "results_file": "mh_results.json",
    },
    {
        "label": "MALA",
        "dir_name": "mala_no_anneal",
        "samples_file": "mala_last_trial_final_samples.json",
        "results_file": "mala_results.json",
    },
    {
        "label": "Annealed MALA",
        "dir_name": "mala_annealed",
        "samples_file": "mala_last_trial_final_samples.json",
        "results_file": "mala_results.json",
    },
]

SHAPE_CLASSES = {0: "Square", 1: "Ellipse", 2: "Heart"}
CLASS_COLORS = {0: "#4C9BE8", 1: "#E87D4C", 2: "#E84C6C"}  



def main():
    parser = argparse.ArgumentParser(
        description="Shape-class histogram per sampler from last-trial decoded samples"
    )
    parser.add_argument("--results-root", required=True,
                        help="Top-level results directory (e.g. ./results_2a_MidSquare)")
    parser.add_argument("--combo-name", required=True,
                        help="Combo subdirectory name (e.g. 2_square_mid)")
    parser.add_argument("--vae-checkpoint", required=True,
                        help="Path to the VAE .pth checkpoint")
    parser.add_argument("--latent-dim", type=int, required=True)
    parser.add_argument("--shape-classifier", type=str, default=None,
                        help="Path to shape classifier .pth  (auto-discovered if omitted)")
    parser.add_argument("--title", type=str, required=True,
                        help="Title displayed at the top of the figure")
    parser.add_argument("--output", type=str, default=None,
                        help="Output image path (default: <results-root>/<combo_name>_shape_hist.png)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device

  
    print(f"Loading VAE from {args.vae_checkpoint} ...")
    vae = load_vae(args.vae_checkpoint, args.latent_dim, device)


    shape_ckpt_path = args.shape_classifier
    if shape_ckpt_path is None:
    
        for spec in METHOD_SPECS:
            rj = os.path.join(args.results_root, spec["dir_name"],
                              args.combo_name, spec["results_file"])
            if os.path.exists(rj):
                res = load_results_json(rj)
                cfg = res.get("experiment_config", {})
                ckpt_map = cfg.get("checkpoints", {}).get("classifiers", {})
                if "shape" in ckpt_map:
                    shape_ckpt_path = ckpt_map["shape"]
                    break

        if shape_ckpt_path is None:
            shape_ckpt_path = "Classifiers/shape.pth"

        
        search_dirs = ["Classifiers", "FinalMods", "CoarseClassifiers", "."]
        resolved = None
        for prefix in search_dirs:
            candidate = os.path.join(prefix, shape_ckpt_path)
            if os.path.exists(candidate):
                resolved = candidate
                break
        if resolved is None and os.path.exists(shape_ckpt_path):
            resolved = shape_ckpt_path
        if resolved is None:
            print(f"ERROR: cannot find shape classifier '{shape_ckpt_path}' "
                  f"in any of {search_dirs}", file=sys.stderr)
            sys.exit(1)
        shape_ckpt_path = resolved

    print(f"Shape classifier: {shape_ckpt_path}")
    shape_clf = load_shape_classifier(shape_ckpt_path, device)


    rows_data = []  

    for spec in METHOD_SPECS:
        samples_path = os.path.join(
            args.results_root, spec["dir_name"], args.combo_name, spec["samples_file"]
        )
        if not os.path.exists(samples_path):
            print(f"WARNING: {samples_path} not found — skipping {spec['label']}")
            continue

        z_all = load_samples_from_json(samples_path)
        imgs = decode_to_images(z_all, vae.decode, device)         # (N, 1, H, W)
        predicted, probs = classify_shapes(imgs, shape_clf, device)  # (N,), (N,3)

        rows_data.append((spec["label"], predicted, probs))
        print(f"  {spec['label']:20s}  N={len(predicted):4d}  "
              f"Sq={np.sum(predicted==0):3d}  "
              f"El={np.sum(predicted==1):3d}  "
              f"He={np.sum(predicted==2):3d}")

    if not rows_data:
        print("ERROR: no method data found at all.", file=sys.stderr)
        sys.exit(1)

  
    n_cols = len(rows_data)
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 3.2, 3.6))
    if n_cols == 1:
        axes = [axes]

    class_names = [SHAPE_CLASSES[i] for i in range(3)]
    x_pos = np.arange(3)
    bar_width = 0.55

    for col_idx, (label, predicted, probs) in enumerate(rows_data):
        ax = axes[col_idx]
        counts = np.array([np.sum(predicted == c) for c in range(3)])
        colors = [CLASS_COLORS[c] for c in range(3)]

        bars = ax.bar(x_pos, counts, width=bar_width, color=colors,
                      edgecolor="white", linewidth=0.8, zorder=3)

        # Annotate each bar: count and mean confidence
        for c_idx, bar in enumerate(bars):
            mask = predicted == c_idx
            n = counts[c_idx]
            if n > 0:
                mean_conf = probs[mask, c_idx].mean()
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{n}\n(conf {mean_conf:.2f})",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, 0.3,
                        "0", ha="center", va="bottom", fontsize=8,
                        fontweight="bold", color="gray")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(class_names, fontsize=9)
        if col_idx == 0:
            ax.set_ylabel("Count", fontsize=9)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(0, max(counts.max() * 1.35, 1))
        ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(args.title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    out_path = args.output or os.path.join(
        args.results_root, f"{args.combo_name}_shape_hist.png"
    )
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved shape histogram to: {out_path}")


if __name__ == "__main__":
    main()
