"""
sampling_common.py

Shared infrastructure imported by mh_sampler.py, mala_sampler.py, and
rejection_sampler.py.

KEY DESIGN:

1. TRIALS: each trial runs --n-chains independent chains (1000 default)
   for the full n_steps, then reports the MEAN of the FINAL per-chain
   likelihood/entropy across those chains -- a single scalar per metric,
   taken only at the last step. R-hat is the one exception: it needs the
   full measurement-window trace (rank-normalized split R-hat, computed
   per latent dimension -- see rank_normalized_split_rhat_per_dimension).
   Uncertainty is ACROSS TRIALS (--n-trials, 15 default): each trial
   contributes one number per metric; aggregate_trial_metrics() computes
   {mean, variance, std} across those n_trials numbers.


2. METRICS: r_hat_max / r_hat_mean / r_hat_per_dimension,
   avg_raw_likelihood_individual (per attribute), avg_log_likelihood_
   individual (per attribute), avg_raw_likelihood_joint, avg_log_
   likelihood_joint, avg_entropy (per attribute), acceptance_rate_sample
   -- each reported as {mean, variance, std} ACROSS TRIALS. No prior-
   density metric, no prior tempering anywhere in this codebase..

3. VISUALIZATION: save_labeled_sample_grid() -- a 5x5 grid of 25 samples
   from the LAST trial (random chains for MCMC, first 25 accepted
   samples for rejection sampling), each labeled above the image with
   its own joint attribute likelihood probability, titled with the
   sampling method and overall acceptance rate. save_trace_plot() for
   one chain's full per-dimension trajectory in the last trial.

   --> to compare, use visualize_method_comparison.py
  -->  to visualize the shape diversity of samples across samplers, use visualize_shape_histogram.py
"""

import argparse
import importlib
import json
import math
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, List, Optional, Tuple

import torch
import torch.nn.functional as F


ATTRIBUTE_CHOICES = ["shape", "scale", "position", "position_y"]

# Model class per attribute. 'position_y' reuses PositionNet verbatim
# (confirmed from the actual training script's build_model():
# `if name in ('position', 'position_y'): return PositionNet(...)`).
ATTRIBUTE_CLASS_NAME = {
    "shape": "ShapeNet",
    "scale": "ScaleNet",
    "position": "PositionNet",
    "position_y": "PositionNet",
}

# Fixed, hardcoded num_classes per attribute -- no longer a CLI choice.
# Matches the training script's actual GRANULARITIES convention: scale
# is always the 3-class ("fine") variant, position/position_y are always
# the 2-class ("coarse") variant.
ATTRIBUTE_NUM_CLASSES = {
    "shape": 3,
    "scale": 3,
    "position": 2,
    "position_y": 2,
}

# Default number of independent trials. 15 gives 14 degrees of freedom
# for the across-trial variance/std -- a solid number for a real error
# bar without being excessive, given each trial is a full run of
# n_chains chains for n_steps steps.
DEFAULT_N_TRIALS = 15


# =============================================================================
# Rank-normalized split R-hat (Vehtari, Gelman, Simpson, Carpenter,
# Burkner 2021, "Rank-normalization, folding, and localization: An
# improved R-hat for assessing convergence of MCMC") -- computed on the
# RAW LATENT DIMENSIONS of the samples, per dimension.
# =============================================================================

def rank_normalize(x: torch.Tensor) -> torch.Tensor:
    """Ranks are computed over ALL elements of x pooled together (matching
    Vehtari et al.'s procedure: ranks pooled across every chain and every
    draw), then mapped to normal scores via the van der Waerden / Blom
    transform: z_i = Phi^-1((rank_i - 3/8) / (S - 3/4 + 1)). Uses ordinal
    ranking (no explicit tie-averaging) -- validated to match scipy's
    average-rank tie handling exactly for continuous data with no ties,
    which is the standard case for MCMC output in a continuous latent
    space (ties have probability zero)."""
    shape = x.shape
    flat = x.reshape(-1)
    S = flat.numel()
    order = torch.argsort(flat)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(1, S + 1, device=x.device, dtype=order.dtype)
    p = (ranks.to(torch.float64) - 3.0 / 8.0) / (S - 3.0 / 4.0 + 1.0)
    z = torch.erfinv(2 * p - 1) * math.sqrt(2)  # inverse standard normal CDF via erfinv identity
    return z.reshape(shape).to(x.dtype)


def rank_normalized_split_rhat(trace: torch.Tensor) -> float:
    """trace: (n_draws, n_chains) for ONE scalar quantity (here, one
    latent dimension). Splits each chain into first/second half, rank-
    normalizes the WHOLE pooled set (both halves, all chains, together),
    then applies the classical split-R-hat formula to the rank-
    normalized values -- validated numerically against a closed-form
    reference (well-mixed Gaussian chains -> R-hat approx 1.0; chains with
    deliberately offset means -> R-hat clearly > 1) before use here.
    """
    n_draws, n_chains = trace.shape
    half = n_draws // 2
    if half < 2:
        return float("nan")
    first, second = trace[:half], trace[half:2 * half]
    pooled = torch.cat([first, second], dim=0)
    ranked = rank_normalize(pooled)
    ranked_first, ranked_second = ranked[:half], ranked[half:]

    chain_means = torch.cat([ranked_first.mean(dim=0), ranked_second.mean(dim=0)])
    chain_vars = torch.cat([ranked_first.var(dim=0, unbiased=True), ranked_second.var(dim=0, unbiased=True)])
    W = chain_vars.mean()
    grand_mean = chain_means.mean()
    m = 2 * n_chains
    B = half / (m - 1) * torch.sum((chain_means - grand_mean) ** 2)
    var_hat = (half - 1) / half * W + B / half
    if W <= 0:
        return float("nan")
    return torch.sqrt(var_hat / W).item()


def rank_normalized_split_rhat_per_dimension(z_trace: torch.Tensor) -> torch.Tensor:
    """z_trace: (n_draws, n_chains, latent_dim). Returns a (latent_dim,)
    tensor of rank-normalized split R-hat, one per latent dimension
    (looped -- ranking is inherently per-quantity, and latent_dim is
    small enough that a Python loop over dimensions costs nothing next
    to the sampling loop itself)."""
    latent_dim = z_trace.shape[-1]
    return torch.tensor(
        [rank_normalized_split_rhat(z_trace[:, :, d]) for d in range(latent_dim)],
        device=z_trace.device,
    )


class WelfordPerChain:
    """Online per-chain mean/variance (Welford's algorithm). Not used for
    R-hat in this revision -- kept available as a general-purpose
    per-chain accumulator in case it's useful elsewhere."""

    def __init__(self, n_chains: int, device, dtype=torch.float32):
        self.count = 0
        self.mean = torch.zeros(n_chains, device=device, dtype=dtype)
        self.M2 = torch.zeros(n_chains, device=device, dtype=dtype)

    def update(self, x: torch.Tensor):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def variance(self) -> torch.Tensor:
        if self.count < 2:
            return torch.zeros_like(self.mean)
        return self.M2 / (self.count - 1)


# =============================================================================
# Prior -- Normal ONLY. GMM prior/proposal removed entirely (per explicit
# instruction): gmm_log_prior, gmm_logp_and_score, and the GMM branch of
# every sampler's prior_logp closure no longer exist anywhere.
# =============================================================================

def normal_log_prior(z: torch.Tensor, temp) -> torch.Tensor:
    d = z.shape[1]
    if not torch.is_tensor(temp):
        temp = torch.tensor(float(temp), device=z.device)
    sq_term = (z * z).sum(dim=1) / temp
    log_norm_const = d * (torch.log(temp) + math.log(2 * math.pi))
    return -0.5 * (sq_term + log_norm_const)


def normal_logp_and_score(z: torch.Tensor, temp) -> Tuple[torch.Tensor, torch.Tensor]:
    if not torch.is_tensor(temp):
        temp = torch.tensor(float(temp), device=z.device)
    d = z.shape[1]
    z = z.detach()
    sq_term = (z * z).sum(dim=1) / temp
    log_norm_const = d * (torch.log(temp) + math.log(2 * math.pi))
    logp = -0.5 * (sq_term + log_norm_const)
    score = -z / temp
    return logp, score


# =============================================================================
# Attribute (classifier) evaluation -- generalized over an arbitrary set of
# 1-3 attributes, not hardcoded to a fixed pair.
# =============================================================================

@torch.compile
def _eval_attribute_nograd(z, decode, classify, target_class, use_amp, amp_device):
    """No-gradient attribute evaluation, for MH (which never needs
    autograd) and for one-off diagnostic evaluations in the gradient-based
    samplers."""
    with torch.autocast(device_type=amp_device, enabled=use_amp, dtype=torch.bfloat16):
        imgs = torch.sigmoid(decode(z))
        logits = classify(imgs)
    logits = logits.float()
    log_probs = F.log_softmax(logits, dim=1)
    log_likelihood = log_probs[:, target_class]
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=1)
    return log_likelihood, entropy


@torch.compile(backend="aot_eager")
def _attr_forward_grad(z, decode, classify, target_class):
    imgs = torch.sigmoid(decode(z))
    logits = classify(imgs)
    log_probs = F.log_softmax(logits, dim=1)
    ll = log_probs[:, target_class]
    return ll, log_probs


def eval_active_attributes_nograd(z, active_specs: Dict[str, Dict], use_amp: bool, amp_device: str):
    """MH-style: no gradient, returns per-attribute log-likelihood + entropy."""
    ll_dict, ent_dict = {}, {}
    for name, spec in active_specs.items():
        ll, ent = _eval_attribute_nograd(
            z, spec["decode"], spec["classify"], spec["target"], use_amp, amp_device
        )
        ll_dict[name] = ll
        ent_dict[name] = ent
    return ll_dict, ent_dict


def attrs_logp_and_score(z: torch.Tensor, active_specs: Dict[str, Dict]) -> Tuple[Dict, Dict, torch.Tensor]:
    """Gradient-based: runs decode+classify for every ACTIVE attribute,
    sums their log-likelihoods, does ONE backward pass for the combined
    gradient. Works for any number (1-3) of active attributes."""
    z_req = z.detach().requires_grad_(True)
    ll_dict, ent_dict = {}, {}
    total_ll = None
    for name, spec in active_specs.items():
        ll, log_probs = _attr_forward_grad(z_req, spec["decode"], spec["classify"], spec["target"])
        ll_dict[name] = ll.detach()
        probs = log_probs.detach().exp()
        ent_dict[name] = -(probs * log_probs.detach()).sum(dim=1)
        total_ll = ll if total_ll is None else total_ll + ll
    score = torch.autograd.grad(total_ll.sum(), z_req)[0].detach()
    return ll_dict, ent_dict, score


def combined_ll_from_dict(ll_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    total = None
    for ll in ll_dict.values():
        total = ll if total is None else total + ll
    return total


# =============================================================================
# Attribute ordering / active-set schedule (generalized to N attributes)
# =============================================================================

def build_active_from(
    order_mode: str,
    attribute_order: List[str],
    switch_steps: Optional[List[int]],
    burn_in: int,
) -> Tuple[Dict[str, int], int]:
    """Returns (active_from: {attribute_name: step it becomes active},
    measurement_start: step at which ALL chosen attributes are active and
    post-burn-in metric collection should begin)."""
    if order_mode == "joint":
        active_from = {name: 0 for name in attribute_order}
        measurement_start = burn_in
    else:
        assert switch_steps is not None and len(switch_steps) == len(attribute_order) - 1, (
            "sequential order_mode needs exactly len(attribute_order)-1 switch_steps"
        )
        active_from = {attribute_order[0]: 0}
        for name, sw in zip(attribute_order[1:], switch_steps):
            active_from[name] = sw
        prev = burn_in
        for sw in switch_steps:
            assert sw > prev, "switch_steps must be strictly increasing and each > burn_in"
            prev = sw
        measurement_start = switch_steps[-1]
    return active_from, measurement_start


# =============================================================================
# Per-trial metric assembly -- each call produces ONE scalar per metric
# (mean across the n_chains chains' FINAL samples for MCMC, mean across
# the trial's accepted samples for rejection sampling). Uncertainty comes
# from aggregate_trial_metrics() across n_trials of these, not from
# anything computed here.
# =============================================================================

def assemble_mcmc_trial_metrics(
    final_log_lik_joint: torch.Tensor,
    final_raw_lik_joint: torch.Tensor,
    final_log_lik_attr: Dict[str, torch.Tensor],
    final_raw_lik_attr: Dict[str, torch.Tensor],
    final_entropy_attr: Dict[str, torch.Tensor],
    acceptance_rate: float,
    r_hat_per_dim: torch.Tensor,
) -> Dict[str, Any]:
    """final_* arguments are (n_chains,) tensors -- the FINAL step's
    per-chain values, already accept/reject-blended. Each is reduced to
    its mean across chains, giving ONE scalar per metric for this trial.
    acceptance_rate is already a trial-level scalar.

    r_hat_per_dim: (latent_dim,) tensor of rank-normalized split R-hat,
    one per latent dimension. Reported as r_hat_max (the standard
    headline convergence check -- Vehtari et al. recommend using the
    MAXIMUM R-hat across all monitored quantities), r_hat_mean
    (additional context), and r_hat_per_dimension (every dimension
    individually).
    """
    r_hat_dict = {f"z{d}": r_hat_per_dim[d].item() for d in range(r_hat_per_dim.shape[0])}
    return {
        "r_hat_max": r_hat_per_dim.max().item(),
        "r_hat_mean": r_hat_per_dim.mean().item(),
        "r_hat_per_dimension": r_hat_dict,
        "avg_raw_likelihood_individual": {k: v.mean().item() for k, v in final_raw_lik_attr.items()},
        "avg_log_likelihood_individual": {k: v.mean().item() for k, v in final_log_lik_attr.items()},
        "avg_raw_likelihood_joint": final_raw_lik_joint.mean().item(),
        "avg_log_likelihood_joint": final_log_lik_joint.mean().item(),
        "avg_entropy": {k: v.mean().item() for k, v in final_entropy_attr.items()},
        "acceptance_rate_sample": acceptance_rate,
    }


def assemble_rejection_trial_metrics(
    accepted_ll_attr: Dict[str, torch.Tensor],
    accepted_ent_attr: Dict[str, torch.Tensor],
    proposal_acceptance_rate: float,
    n_proposed: int,
    n_accepted: int,
    importance_ess: float,
    importance_ess_fraction: float,
) -> Dict[str, Any]:
    """accepted_ll_attr / accepted_ent_attr hold ALL accepted samples in
    this trial (not just the final one -- rejection sampling has no
    "chain" to take a final state from). Each is reduced to its mean
    across the trial's accepted-sample population, giving one scalar per
    metric for this trial; trial-to-trial variance comes from
    aggregate_trial_metrics() on top.

    proposal_acceptance_rate is rejection sampling's own natural
    statistic -- kept under a distinct name from acceptance_rate_sample
    so it's never confused with the MCMC concept.

    importance_ess / importance_ess_fraction: every proposal's likelihood
    is an importance weight against the same target, accumulated over
    EVERY proposal in the trial (not just accepted ones) via
    ESS = (sum w_i)^2 / sum(w_i^2).
    """
    combined_ll = combined_ll_from_dict(accepted_ll_attr) if accepted_ll_attr else torch.empty(0)
    return {
        "avg_raw_likelihood_individual": {k: (v.exp().mean().item() if v.numel() > 0 else float("nan")) for k, v in accepted_ll_attr.items()},
        "avg_log_likelihood_individual": {k: (v.mean().item() if v.numel() > 0 else float("nan")) for k, v in accepted_ll_attr.items()},
        "avg_raw_likelihood_joint": combined_ll.exp().mean().item() if combined_ll.numel() > 0 else float("nan"),
        "avg_log_likelihood_joint": combined_ll.mean().item() if combined_ll.numel() > 0 else float("nan"),
        "avg_entropy": {k: (v.mean().item() if v.numel() > 0 else float("nan")) for k, v in accepted_ent_attr.items()},
        "proposal_acceptance_rate": proposal_acceptance_rate,
        "n_proposed": n_proposed,
        "n_accepted": n_accepted,
        "importance_ess": importance_ess,
        "importance_ess_fraction": importance_ess_fraction,
    }


# =============================================================================
# Cross-trial aggregation: {mean, variance, std} across n_trials
# =============================================================================

def _variance_std(values: List[float]) -> Tuple[float, float]:
    n = len(values)
    if n < 2:
        return float("nan"), float("nan")
    m = sum(values) / n
    var = sum((v - m) ** 2 for v in values) / (n - 1)  # sample variance, ddof=1
    return var, math.sqrt(var)


def _aggregate_leaf(values: List[Any]) -> Any:
    non_none = [
        v for v in values
        if v is not None and not (isinstance(v, float) and math.isnan(v))
    ]
    if len(non_none) == 0:
        return {"mean": float("nan"), "variance": float("nan"), "std": float("nan"), "n_trials_contributing": 0}
    if all(isinstance(v, (int, float)) for v in non_none):
        mean = sum(non_none) / len(non_none)
        variance, std = _variance_std(non_none)
        result = {"mean": mean, "variance": variance, "std": std}
        if len(non_none) < len(values):
            result["n_trials_contributing"] = len(non_none)
        return result
    return values


def aggregate_trial_metrics(trial_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Walks the (possibly nested, e.g. per-attribute) per-trial metric
    dicts and computes {mean, variance, std} across trials for every
    leaf -- e.g. avg_log_likelihood_joint becomes the mean, variance, and
    std of the n_trials single numbers (each already the mean over that
    trial's n_chains final chain states)."""
    assert len(trial_metrics) > 0
    all_keys = list({k for t in trial_metrics for k in t.keys()})
    out = {}
    for k in all_keys:
        values = [t.get(k) for t in trial_metrics]
        dict_values = [v for v in values if isinstance(v, dict)]
        if dict_values:
            sub_keys = list({sk for d in dict_values for sk in d.keys()})
            out[k] = {
                sk: _aggregate_leaf([v.get(sk, float("nan")) if isinstance(v, dict) else float("nan") for v in values])
                for sk in sub_keys
            }
        else:
            out[k] = _aggregate_leaf(values)
    return out


# =============================================================================
# Saving: images, JSON
# =============================================================================

def save_sample_grid(decode, z, save_dir, filename="samples.png", n_samples=15, rows=3, cols=5):
    """Plain (unlabeled) grid -- still used for pre-switch mixing
    snapshots in sequential order_mode. See save_labeled_sample_grid for
    the final-results visualization with per-image likelihood labels."""
    import matplotlib.pyplot as plt

    n_samples = min(n_samples, z.shape[0])
    rows_cols = rows * cols
    if n_samples < rows_cols:
        cols = max(1, n_samples)
        rows = 1
    os.makedirs(save_dir, exist_ok=True)
    with torch.inference_mode():
        imgs = torch.sigmoid(decode(z[:n_samples])).detach().float().cpu()
    imgs_min = imgs.amin(dim=(1, 2, 3), keepdim=True)
    imgs_max = imgs.amax(dim=(1, 2, 3), keepdim=True)
    imgs = (imgs - imgs_min) / (imgs_max - imgs_min).clamp_min(1e-8)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]
    for i, ax in enumerate(axes_flat):
        if i >= n_samples:
            ax.axis("off")
            continue
        img = imgs[i].permute(1, 2, 0).numpy()
        ax.imshow(img.squeeze(-1), cmap="gray") if img.shape[-1] == 1 else ax.imshow(img)
        ax.axis("off")
    fig.tight_layout()
    out_path = os.path.join(save_dir, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_trace_plot(
    z_trace: torch.Tensor,
    save_dir: str,
    filename: str,
    title: str,
) -> str:
    """z_trace: (n_steps, latent_dim) for ONE chain across the FULL run
    (including burn-in, so burn-in behavior is visible). One subplot per
    latent dimension, stacked vertically (dimension d's row shows its
    value across every step for this single chain), each with its own
    y-axis scale (dimensions can have very different ranges, so a shared
    scale would flatten some of them to invisibility). Figure height
    scales with latent_dim so a 10-dimensional trace doesn't get
    squished into unreadable subplots.
    """
    import matplotlib.pyplot as plt

    n_steps, latent_dim = z_trace.shape
    z_np = z_trace.detach().cpu().numpy()
    steps = range(n_steps)

    fig, axes = plt.subplots(
        latent_dim, 1, figsize=(11, max(2.0, 1.6 * latent_dim)), sharex=True
    )
    if latent_dim == 1:
        axes = [axes]
    for d, ax in enumerate(axes):
        ax.plot(steps, z_np[:, d], linewidth=0.6)
        ax.set_ylabel(f"z{d}", rotation=0, labelpad=20, va="center")
        ax.tick_params(labelsize=8)
    axes[-1].set_xlabel("step")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_labeled_sample_grid(
    decode: Callable,
    z_selected: torch.Tensor,
    joint_probs: torch.Tensor,
    save_dir: str,
    filename: str,
    title: str,
    rows: int = 5,
    cols: int = 5,
) -> str:
    """Grid (up to 5x5 / 25 samples, adapted if fewer samples) with each image's own joint attribute
    likelihood PROBABILITY (raw, not log) printed above it, and an
    overall figure title. z_selected and joint_probs must have matching
    first dimension."""
    import matplotlib.pyplot as plt

    n = z_selected.shape[0]
    if n == 0:
        return ""
    if n < rows * cols:
        cols = min(cols, max(1, n))
        rows = (n + cols - 1) // cols

    os.makedirs(save_dir, exist_ok=True)
    with torch.inference_mode():
        imgs = torch.sigmoid(decode(z_selected)).detach().float().cpu()
    imgs_min = imgs.amin(dim=(1, 2, 3), keepdim=True)
    imgs_max = imgs.amax(dim=(1, 2, 3), keepdim=True)
    imgs = (imgs - imgs_min) / (imgs_max - imgs_min).clamp_min(1e-8)

    probs_list = joint_probs.detach().cpu().tolist() if torch.is_tensor(joint_probs) else list(joint_probs)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.3), squeeze=False)
    axes_flat = axes.flatten()
    for i, ax in enumerate(axes_flat):
        if i < n:
            img = imgs[i].permute(1, 2, 0).numpy()
            ax.imshow(img.squeeze(-1), cmap="gray") if img.shape[-1] == 1 else ax.imshow(img)
            ax.set_title(f"{probs_list[i]:.3f}", fontsize=9)
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(save_dir, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _json_safe(obj):
    import numpy as np

    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def save_results_json(results: Dict[str, Any], save_dir: str, filename: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, filename)
    with open(out_path, "w") as f:
        json.dump(_json_safe(results), f, indent=2)
    return out_path


# =============================================================================
# Trial orchestration
# =============================================================================

def run_trials(
    run_one_trial_fn: Callable[[int], Dict[str, Any]],
    n_trials: int,
    decode: Callable,
    save_dir: str,
    method_name: str,
    sampling_method_label: str,
    select_mode: str,
    base_seed: Optional[int] = None,
    experiment_metadata: Optional[Dict[str, Any]] = None,
    trace_plot_title: Optional[str] = None,
) -> Dict[str, Any]:
    """Runs run_one_trial_fn(trial_idx) n_trials times. Each call must
    return {"final_samples": tensor, "final_joint_prob": tensor,
    "metrics": {...single scalars...}} -- see each sampler's
    assemble_*_trial_metrics for what "metrics" contains per trial.
    Optionally also "z_trace_chain0": (n_steps, latent_dim) tensor, the
    full run's trace for one representative chain -- if present, a trace
    plot is saved from the LAST trial's trace (see save_trace_plot).
    Not applicable to rejection sampling (no chain/trajectory concept),
    which simply omits this key and the trace-plot step is skipped.

    Aggregates metrics across trials via aggregate_trial_metrics
    ({mean, variance, std} for every leaf). Saves a labeled 25-sample
    grid from the LAST trial only (select_mode='random' for MCMC chains,
    'first' for rejection sampling's accepted-sample list).

    Results are saved incrementally after every trial (not just at the
    end), so a crash partway through survives with whatever was gathered
    so far on disk, tagged status='in_progress' vs 'complete'.
    """
    trial_metrics = []
    last_final_samples = None
    last_final_joint_prob = None
    last_acceptance_rate = None
    last_z_trace_chain0 = None

    for trial in range(n_trials):
        if base_seed is not None:
            torch.manual_seed(base_seed + trial)
        print(f"=== {method_name}: trial {trial + 1}/{n_trials} ===")
        result = run_one_trial_fn(trial)
        trial_metrics.append(result["metrics"])
        last_final_samples = result["final_samples"]
        last_final_joint_prob = result["final_joint_prob"]
        last_z_trace_chain0 = result.get("z_trace_chain0")
        last_acceptance_rate = result["metrics"].get(
            "acceptance_rate_sample", result["metrics"].get("proposal_acceptance_rate")
        )

        save_results_json(
            {
                "method": method_name,
                "experiment_config": experiment_metadata or {},
                "n_trials_planned": n_trials,
                "trials_completed": trial + 1,
                "status": "complete" if trial + 1 == n_trials else "in_progress",
                "per_trial_metrics": trial_metrics,
                "aggregated_metrics": aggregate_trial_metrics(trial_metrics),
            },
            save_dir,
            filename=f"{method_name}_results.json",
        )

    os.makedirs(save_dir, exist_ok=True)
    n_available = last_final_samples.shape[0]
    n_show = min(25, n_available)
    if select_mode == "random":
        idx = torch.randperm(n_available, device=last_final_samples.device)[:n_show]
        unit = "Chains"
    else:
        idx = torch.arange(n_show, device=last_final_samples.device)
        unit = "Samples"
    z_selected = last_final_samples[idx]
    probs_selected = last_final_joint_prob[idx]
    title = f"Final Samples from {n_show} {unit} using {sampling_method_label} (acceptance rate: {last_acceptance_rate:.3f})"
    grid_path = save_labeled_sample_grid(
        decode, z_selected, probs_selected, save_dir,
        filename=f"{method_name}_final_samples_grid.png", title=title,
    )

    trace_plot_path = None
    if last_z_trace_chain0 is not None:
        trace_plot_path = save_trace_plot(
            last_z_trace_chain0, save_dir,
            filename=f"{method_name}_trace_plot.png",
            title=trace_plot_title or f"Trace plot - {sampling_method_label}",
        )

    out = {
        "method": method_name,
        "experiment_config": experiment_metadata or {},
        "n_trials_planned": n_trials,
        "trials_completed": n_trials,
        "status": "complete",
        "per_trial_metrics": trial_metrics,
        "aggregated_metrics": aggregate_trial_metrics(trial_metrics),
        "final_samples_grid": grid_path,
        "trace_plot": trace_plot_path,
    }
    save_results_json(out, save_dir, filename=f"{method_name}_results.json")
    save_results_json(
        {"last_trial_final_samples": last_final_samples}, save_dir,
        filename=f"{method_name}_last_trial_final_samples.json",
    )
    return out


# =============================================================================
# Model / classifier loading (generic, convention-based)
# =============================================================================

def load_attribute_classifiers(
    attribute_names: List[str],
    checkpoint_paths: List[str],
    device: str,
) -> Dict[str, torch.nn.Module]:
    """Loads one classifier per chosen attribute. num_classes is no
    longer a caller-supplied argument -- it's looked up from the fixed
    ATTRIBUTE_NUM_CLASSES mapping above (shape=3, scale=3, position=2,
    position_y=2), matching the actual training script's convention
    exactly (scale always 3-class, position/position_y always 2-class).
    'position_y' reuses PositionNet verbatim, same as ATTRIBUTE_CLASS_NAME
    reflects. A checkpoint whose saved state_dict doesn't actually match
    the hardcoded num_classes will fail loudly at load_state_dict with a
    shape-mismatch error -- there's no longer a soft warning step for
    this, since num_classes isn't user-suppliable anymore to create
    ambiguity in the first place.
    """
    classifier_module = importlib.import_module("classifier")
    classifiers = {}
    for name, ckpt in zip(attribute_names, checkpoint_paths):
        if name not in ATTRIBUTE_CLASS_NAME:
            raise ValueError(
                f"no known classifier class for attribute '{name}' -- add it to "
                f"ATTRIBUTE_CLASS_NAME and ATTRIBUTE_NUM_CLASSES in sampling_common.py"
            )
        class_name = ATTRIBUTE_CLASS_NAME[name]
        if not hasattr(classifier_module, class_name):
            raise ValueError(
                f"classifier.py has no '{class_name}' -- required for --attributes including '{name}'"
            )
        n_cls = ATTRIBUTE_NUM_CLASSES[name]
        cls = getattr(classifier_module, class_name)
        model = cls(num_classes=n_cls).to(device)
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
        model.eval()
        classifiers[name] = model
    return classifiers


def load_vae(vae_checkpoint: str, latent_dim: int, device: str):
    """VAE only -- GMM loading removed entirely (no GMM prior/proposal
    anywhere in this codebase anymore)."""
    train_vae = importlib.import_module("train_vae")
    model = train_vae.BetaTCVAE(latent_dim=latent_dim).to(device)
    ckpt = torch.load(vae_checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.eval()
    return model


# =============================================================================
# Shared CLI argument parsing
# =============================================================================

def parse_csv(s: Optional[str]) -> List[str]:
    return [x.strip() for x in s.split(",")] if s else []


def parse_csv_int(s: Optional[str]) -> List[int]:
    return [int(x.strip()) for x in s.split(",")] if s else []


def build_common_arg_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)

    p.add_argument("--vae-checkpoint", required=True, help="Path to the BetaTCVAE .pth checkpoint")
    p.add_argument("--latent-dim", type=int, required=True)

    p.add_argument(
        "--attributes", required=True,
        help="Comma list of 1, 2, or 3 from: shape,scale,position,position_y",
    )
    p.add_argument(
        "--target-classes", required=True,
        help="Comma list of target class indices, matching --attributes order",
    )
    p.add_argument(
        "--classifier-checkpoints", required=True,
        help="Comma list of classifier .pth paths, matching --attributes order",
    )

    p.add_argument("--n-chains", type=int, default=1000)
    p.add_argument("--n-steps", type=int, default=3000)
    p.add_argument("--burn-in", type=int, default=1000)

    p.add_argument("--order-mode", choices=["joint", "sequential"], default="joint")
    p.add_argument(
        "--attribute-order", default=None,
        help="Comma list; only used if --order-mode sequential. Defaults to --attributes order.",
    )
    p.add_argument(
        "--switch-steps", default=None,
        help="Comma list of ints, length = n_attributes-1, required if --order-mode sequential.",
    )

    p.add_argument(
        "--prior-temp", type=float, default=1.0,
        help="Variance of the Normal(0, prior_temp) prior/proposal -- the only prior in this codebase.",
    )

    p.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-dir", required=True)

    p.add_argument("--pre-switch-snapshot", action="store_true")

    return p


def build_experiment_metadata(
    sampler: str,
    attribute_names: List[str],
    targets: Dict[str, int],
    order_mode: str,
    attribute_order: Optional[List[str]],
    switch_steps: Optional[List[int]],
    tempering: Dict[str, Any],
    vae_checkpoint: str,
    classifier_checkpoints: Dict[str, str],
) -> Dict[str, Any]:
    """Builds the metadata dict embedded under 'experiment_config' in the
    results JSON (see run_trials), so every saved result is self-
    describing: which sampler, which attributes/target classes, joint or
    sequential (+ order/switch steps if sequential), tempering setup, and
    which checkpoint files were used. prior_type is hardcoded to "normal"
    -- there's no other option in this codebase. Checkpoint paths are
    reduced to basenames -- full paths are usually machine-specific and
    not useful for cross-referencing runs later.
    """
    return {
        "sampler": sampler,
        "attributes": attribute_names,
        "target_classes": targets,
        "order_mode": order_mode,
        "attribute_order": attribute_order if order_mode == "sequential" else None,
        "switch_steps": switch_steps if order_mode == "sequential" else None,
        "prior_type": "normal",
        "tempering": tempering,
        "checkpoints": {
            "vae": os.path.basename(vae_checkpoint),
            "classifiers": {name: os.path.basename(ckpt) for name, ckpt in classifier_checkpoints.items()},
        },
    }


def resolve_attribute_args(args):
    """Parses and validates the shared attribute-related CLI args. Returns
    (attribute_names, targets_dict, classifiers_dict, attribute_order,
    switch_steps)."""
    attribute_names = parse_csv(args.attributes)
    assert 1 <= len(attribute_names) <= 3, "choose 1, 2, or 3 attributes"
    assert all(a in ATTRIBUTE_CHOICES for a in attribute_names), (
        f"attributes must be from {ATTRIBUTE_CHOICES}"
    )

    target_classes = parse_csv_int(args.target_classes)
    checkpoints = parse_csv(args.classifier_checkpoints)
    assert len(target_classes) == len(attribute_names) == len(checkpoints), (
        "--attributes, --target-classes, --classifier-checkpoints must all have the same "
        "length and matching order"
    )
    targets = dict(zip(attribute_names, target_classes))

    classifiers = load_attribute_classifiers(attribute_names, checkpoints, args.device)

    if args.order_mode == "sequential":
        assert len(attribute_names) >= 2, (
            "order_mode='sequential' needs at least 2 attributes -- there's nothing to sequence "
            "with only 1. Use --order-mode joint for a single-attribute run."
        )
        attribute_order = parse_csv(args.attribute_order) if args.attribute_order else list(attribute_names)
        assert set(attribute_order) == set(attribute_names)
        switch_steps = parse_csv_int(args.switch_steps)
        assert len(switch_steps) == len(attribute_order) - 1, (
            "--switch-steps must have length len(attributes)-1 for sequential order_mode"
        )
    else:
        attribute_order = list(attribute_names)
        switch_steps = None

    return attribute_names, targets, classifiers, attribute_order, switch_steps