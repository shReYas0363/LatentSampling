#!/usr/bin/env python3
"""
Visualize n random decoded samples from the last trial of each sampling
method side by side.  Rows (top to bottom):
    1. Rejection Sampling
    2. RW-MH
    3. MALA (no anneal)
    4. Annealed MALA

Each image is annotated with its joint classifier probability.

Usage:
    python visualize_methods_comparison.py \
        --results-root ./results_2a_MidSquare \
        --combo-name 2_square_mid \
        --vae-checkpoint FinalMods/betatcvae_dim10_beta4_MidScaleSq.pth \
        --latent-dim 10 \
        --title "Medium-Scale Squares"
"""

import argparse
import json
import os
import sys
import random

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


def load_classifiers(attribute_names, classifier_checkpoints, device):
    """Load classifiers using the same convention as sampling_common."""
    classifier_module = importlib.import_module("classifier")

    ATTRIBUTE_CLASS_NAME = {
        "shape": "ShapeNet",
        "scale": "ScaleNet",
        "position": "PositionNet",
        "position_y": "PositionNet",
    }
    ATTRIBUTE_NUM_CLASSES = {
        "shape": 3,
        "scale": 3,
        "position": 2,
        "position_y": 2,
    }

    classifiers = {}
    for name, ckpt_path in zip(attribute_names, classifier_checkpoints):
        class_name = ATTRIBUTE_CLASS_NAME[name]
        n_cls = ATTRIBUTE_NUM_CLASSES[name]
        cls = getattr(classifier_module, class_name)
        model = cls(num_classes=n_cls).to(device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
        model.eval()
        classifiers[name] = model
    return classifiers


def compute_joint_prob(z, decode, classifiers, target_classes, device):
    """Compute the joint probability = product of P(target_class | x) for
    each attribute classifier.  Returns a (N,) tensor of probabilities."""
    with torch.inference_mode():
        imgs = torch.sigmoid(decode(z.to(device)))
        log_joint = torch.zeros(z.shape[0], device=device)
        for name, clf in classifiers.items():
            logits = clf(imgs).float()
            log_probs = F.log_softmax(logits, dim=1)
            target = target_classes[name]
            log_joint += log_probs[:, target]
    return log_joint.exp().cpu()


def decode_to_images(z, decode, device):
    """Decode latent vectors to normalized [0,1] images. Returns (N, H, W) numpy."""
    with torch.inference_mode():
        imgs = torch.sigmoid(decode(z.to(device))).detach().float().cpu()
    # Per-image min-max normalization (matches sampling_common convention)
    imgs_min = imgs.amin(dim=(1, 2, 3), keepdim=True)
    imgs_max = imgs.amax(dim=(1, 2, 3), keepdim=True)
    imgs = (imgs - imgs_min) / (imgs_max - imgs_min).clamp_min(1e-8)
    # (N, 1, H, W) -> (N, H, W)
    return imgs.squeeze(1).numpy()



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



def main():
    parser = argparse.ArgumentParser(
        description="4-row comparison grid: Rejection / RW-MH / MALA / Annealed MALA"
    )
    parser.add_argument("--results-root", required=True,
                        help="Top-level results directory (e.g. ./results_2a_MidSquare)")
    parser.add_argument("--combo-name", required=True,
                        help="Combo subdirectory name (e.g. 2_square_mid)")
    parser.add_argument("--vae-checkpoint", required=True,
                        help="Path to the VAE .pth checkpoint")
    parser.add_argument("--latent-dim", type=int, required=True)
    parser.add_argument("--title", type=str, required=True,
                        help="Title displayed at the top of the figure")
    parser.add_argument("--n-samples", type=int, default=8,
                        help="Number of samples per method row (default 8)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None,
                        help="Output image path (default: <results-root>/<combo_name>_comparison.png)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device

  
    print(f"Loading VAE from {args.vae_checkpoint} ...")
    vae = load_vae(args.vae_checkpoint, args.latent_dim, device)


    attribute_names = None
    target_classes = None
    classifier_ckpts = None

    for spec in METHOD_SPECS:
        rj = os.path.join(args.results_root, spec["dir_name"], args.combo_name, spec["results_file"])
        if os.path.exists(rj):
            res = load_results_json(rj)
            cfg = res.get("experiment_config", {})
            attribute_names = cfg.get("attributes")
            target_classes = cfg.get("target_classes")

            ckpt_map = cfg.get("checkpoints", {}).get("classifiers", {})
            classifier_ckpts = [ckpt_map[a] for a in attribute_names]
            break

    if attribute_names is None:
        print("ERROR: could not find any *_results.json to discover attributes / targets.", file=sys.stderr)
        sys.exit(1)

        
    resolved_ckpts = []
    search_dirs = ["Classifiers", "FinalMods", "CoarseClassifiers", "."]
    for ckpt_basename in classifier_ckpts:
        found = False
        for prefix in search_dirs:
            candidate = os.path.join(prefix, ckpt_basename)
            if os.path.exists(candidate):
                resolved_ckpts.append(candidate)
                found = True
                break
        if not found:
            if os.path.exists(ckpt_basename):
                resolved_ckpts.append(ckpt_basename)
            else:
                print(f"ERROR: cannot find classifier checkpoint '{ckpt_basename}' in any of {search_dirs}",
                      file=sys.stderr)
                sys.exit(1)

    print(f"Attributes: {attribute_names}")
    print(f"Target classes: {target_classes}")
    print(f"Classifier checkpoints: {resolved_ckpts}")

    classifiers = load_classifiers(attribute_names, resolved_ckpts, device)

    n = args.n_samples
    rows_data = []  

    for spec in METHOD_SPECS:
        samples_path = os.path.join(
            args.results_root, spec["dir_name"], args.combo_name, spec["samples_file"]
        )
        if not os.path.exists(samples_path):
            print(f"WARNING: {samples_path} not found — skipping {spec['label']}")
            continue

        z_all = load_samples_from_json(samples_path)
        n_available = z_all.shape[0]
        if n_available < n:
            print(f"WARNING: {spec['label']} has only {n_available} samples (requested {n})")
        z_selected = z_all[:min(n, n_available)]   # first N chains
        probs = compute_joint_prob(z_selected, vae.decode, classifiers, target_classes, device)
        imgs = decode_to_images(z_selected, vae.decode, device)

        rows_data.append((spec["label"], imgs, probs.numpy()))

    if not rows_data:
        print("ERROR: no method data found at all.", file=sys.stderr)
        sys.exit(1)

  
    n_rows = len(rows_data)
    n_cols = n

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 1.6, n_rows * 2.2),
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for row_idx, (label, imgs, probs) in enumerate(rows_data):
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            if col_idx < len(imgs):
                ax.imshow(imgs[col_idx], cmap="gray", vmin=0, vmax=1)
                ax.set_title(f"{probs[col_idx]:.3f}", fontsize=8, pad=2)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        
        axes[row_idx, 0].set_ylabel(
            label, fontsize=11, fontweight="bold", rotation=90,
            labelpad=10, va="center", ha="center",
        )

    fig.suptitle(args.title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = args.output or os.path.join(
        args.results_root, f"{args.combo_name}_comparison.png"
    )
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison grid to: {out_path}")


if __name__ == "__main__":
    main()
