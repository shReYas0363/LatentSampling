"""
Rejection sampler over the VAE's latent space: propose z ~ prior(z),
accept with probability = joint classifier likelihood(z) = product of
each active attribute's target-class softmax probability. Since each
per-attribute probability is <= 1, the joint likelihood is a valid
acceptance probability with bound M=1 -- no tuning needed, and every
accepted sample is an EXACT, unbiased draw from the true posterior
prior(z) * likelihood(z).


TRIALS (default --n-trials 15). Within one trial, the reported number
for each likelihood/entropy metric is the mean over ALL accepted samples
in that trial (there's no "final chain state" concept here -- accepted
samples are i.i.d., not chains). Uncertainty is the VARIANCE (and std)
of that per-trial mean ACROSS the n_trials trials, computed the same way
as MH/MALA -- see sampling_common.py's module docstring.


*DISPLAYED, NOT REPORTED: EFFECTIVE SAMPLE SIZE OF IMPORTANCE WEIGHTS*: alongside the accept/reject
statistic, every proposal's likelihood(z) is also treated as an
importance weight Included in code, but not utilized in any way against the same target, and the standard IS
effective-sample-size diagnostic ESS = (sum w_i)^2 / sum(w_i^2) is
computed over EVERY proposal in each trial -- not just accepted ones.

Run directly:

python rejection_sampler.py \
    --vae-checkpoint /path/betatcvae.pth --latent-dim 10 \
    --attributes shape,scale --target-classes 2,1 \
    --classifier-checkpoints /path/shape.pth,/path/scale_fine.pth \
    --n-proposals-per-round 10000 --max-proposals 2000000 --n-accept-target 500 \
    --n-trials 15 --save-dir ./results
"""

from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Optional

import torch
from tqdm import trange

from sampling_common import (
    eval_active_attributes_nograd, combined_ll_from_dict,
    assemble_rejection_trial_metrics, run_trials,
    load_vae, resolve_attribute_args, build_common_arg_parser,
    build_experiment_metadata, parse_csv,
)


@dataclass
class RejectionConfig:
    n_proposals_per_round: int = 10000
    max_proposals: int = 2_000_000
    n_accept_target: int = 500   

    prior_temp: float = 1.0

    use_amp: bool = True
    device: str = "cuda"
    dtype: torch.dtype = torch.float32


def run_rejection_sampling(
    decode: Callable,
    attribute_classifiers: Dict[str, Callable],
    attribute_targets: Dict[str, int],
    latent_dim: int,
    cfg: RejectionConfig,
) -> Dict[str, Any]:
    """Runs ONE trial. Metrics are plain scalars (mean over this trial's
    accepted-sample population); trial-to-trial variance is computed by
    the caller (run_trials)."""
    device = cfg.device
    amp_device = "cuda" if "cuda" in device else "cpu"
    attrs = {
        name: dict(decode=decode, classify=attribute_classifiers[name], target=attribute_targets[name])
        for name in attribute_classifiers
    }

    def prior_sample(n):
        return torch.randn(n, latent_dim, device=device, dtype=cfg.dtype) * cfg.prior_temp ** 0.5

    accepted_z, accepted_ll, accepted_ent = [], {n: [] for n in attrs}, {n: [] for n in attrs}
    n_proposed, n_accepted = 0, 0
    best_ll_seen, best_z_seen = float("-inf"), None
    sum_w, sum_w_sq = 0.0, 0.0  # accumulated over EVERY proposal, for importance-weight ESS

    with torch.inference_mode():
        pbar = trange(0, cfg.max_proposals, cfg.n_proposals_per_round, desc="Rejection sampling")
        for _ in pbar:
            if n_accepted >= cfg.n_accept_target:
                break
            z = prior_sample(cfg.n_proposals_per_round)
            ll_dict, ent_dict = eval_active_attributes_nograd(z, attrs, cfg.use_amp, amp_device)
            combined = combined_ll_from_dict(ll_dict)
            raw_joint = combined.exp()  # valid acceptance probability: product of probabilities, bound M=1

            batch_best_idx = int(combined.argmax().item())
            if combined[batch_best_idx].item() > best_ll_seen:
                best_ll_seen = combined[batch_best_idx].item()
                best_z_seen = z[batch_best_idx:batch_best_idx + 1].clone()

            u = torch.rand(cfg.n_proposals_per_round, device=device)
            accept_mask = u < raw_joint
            n_proposed += cfg.n_proposals_per_round

            sum_w += raw_joint.sum().item()
            sum_w_sq += (raw_joint ** 2).sum().item()

            if accept_mask.any():
                idx = accept_mask.nonzero(as_tuple=True)[0]
                take = idx[: max(0, cfg.n_accept_target - n_accepted)]
                accepted_z.append(z[take])
                for name in attrs:
                    accepted_ll[name].append(ll_dict[name][take])
                    accepted_ent[name].append(ent_dict[name][take])
                n_accepted += take.shape[0]

            running_ess_frac = ((sum_w ** 2) / sum_w_sq / n_proposed) if sum_w_sq > 0 else 0.0
            pbar.set_postfix({
                "accepted": n_accepted,
                "rate": f"{n_accepted / max(n_proposed, 1):.2e}",
                "IS_ESS_frac": f"{running_ess_frac:.2e}",
            })

    proposal_acceptance_rate = n_accepted / max(n_proposed, 1)
    importance_ess = (sum_w ** 2) / sum_w_sq if sum_w_sq > 0 else 0.0
    importance_ess_fraction = importance_ess / max(n_proposed, 1)

    if n_accepted == 0:
        print(
            f"[warn] 0 samples accepted after {n_proposed} proposals (budget {cfg.max_proposals}). "
            f"All likelihood/entropy metrics will be NaN for this trial -- for a held-out combination, "
            f"that absence of acceptances may itself be the finding. Using the single highest-joint-"
            f"likelihood proposal seen (log-lik={best_ll_seen:.3f}) as a visualization-only fallback for "
            f"final_samples; it is NOT a valid posterior sample and is excluded from all metrics."
        )
        final_samples = best_z_seen if best_z_seen is not None else prior_sample(1)
        final_joint_prob = torch.zeros(final_samples.shape[0], device=device)
        accepted_ll_full = {n: torch.empty(0, device=device) for n in attrs}
        accepted_ent_full = {n: torch.empty(0, device=device) for n in attrs}
    else:
        final_samples = torch.cat(accepted_z, dim=0)
        accepted_ll_full = {n: torch.cat(accepted_ll[n]) for n in attrs}
        accepted_ent_full = {n: torch.cat(accepted_ent[n]) for n in attrs}
        final_joint_prob = combined_ll_from_dict(accepted_ll_full).exp().detach()

    metrics = assemble_rejection_trial_metrics(
        accepted_ll_full, accepted_ent_full,
        proposal_acceptance_rate, n_proposed, n_accepted,
        importance_ess, importance_ess_fraction,
    )

    return {"final_samples": final_samples, "final_joint_prob": final_joint_prob, "metrics": metrics}


def main():
    parser = build_common_arg_parser("Rejection sampler (exact posterior draws, no MCMC)")
    parser.add_argument("--n-proposals-per-round", type=int, default=10000)
    parser.add_argument("--max-proposals", type=int, default=2_000_000)
    parser.add_argument("--n-accept-target", type=int, default=500)
    args = parser.parse_args()

    assert args.order_mode == "joint", (
        "rejection sampling has no order_mode='sequential' analogue (i.i.d. proposals have no time "
        "axis to introduce a second attribute partway through) -- use --order-mode joint."
    )
    assert args.n_accept_target >= 25, "--n-accept-target must be >= 25 to fill the final 5x5 sample grid"

    attribute_names, targets, classifiers, attribute_order, switch_steps = resolve_attribute_args(args)
    model = load_vae(args.vae_checkpoint, args.latent_dim, args.device)

    cfg = RejectionConfig(
        n_proposals_per_round=args.n_proposals_per_round, max_proposals=args.max_proposals,
        n_accept_target=args.n_accept_target, prior_temp=args.prior_temp,
        device=args.device,
    )

    def run_one_trial(trial_idx: int) -> Dict[str, Any]:
        return run_rejection_sampling(
            decode=model.decode, attribute_classifiers=classifiers, attribute_targets=targets,
            latent_dim=args.latent_dim, cfg=cfg,
        )

    metadata = build_experiment_metadata(
        sampler="rejection", attribute_names=attribute_names, targets=targets,
        order_mode="joint", attribute_order=None, switch_steps=None,
        tempering={"type": "none"},
        vae_checkpoint=args.vae_checkpoint,
        classifier_checkpoints=dict(zip(attribute_names, parse_csv(args.classifier_checkpoints))),
    )

    result = run_trials(
        run_one_trial_fn=run_one_trial, n_trials=args.n_trials, decode=model.decode,
        save_dir=args.save_dir, method_name="rejection", sampling_method_label="Rejection Sampling",
        select_mode="first", base_seed=args.seed, experiment_metadata=metadata,
    )
    print("\n=== Aggregated metrics (mean, variance, std across trials) ===")
    print(result["aggregated_metrics"])


if __name__ == "__main__":
    main()