"""
Metropolis-Hastings posterior sampler over the VAE's latent space.

Additional details:

TRIALS (default --n-trials 15): within one trial, --n-chains (1000
default) independent chains run for the full n_steps, and each
likelihood/entropy metric is reduced to the MEAN of its FINAL per-chain
value across those chains -- a single scalar per trial, taken only at
the last step. Uncertainty is then the VARIANCE (and std) of that scalar
ACROSS the n_trials trials. R-hat is the one exception: it inherently
needs the trajectory (first-half vs. second-half chain comparison), so
it's still computed from the full measurement-window trace every trial.

The likelihood's inverse-temperature ramps geometrically from --min-beta
(hot) to 1.0 (cold, full strength) across --anneal-fraction (default 0.7)
of burn_in, then holds flat at 1.0 for the remainder of burn_in and the
entire measurement window.

Toggle with --anneal-temperature / --no-anneal-temperature; disabled
means flat beta=1.0 the whole run (plain, untempered MCMC).

Run directly, e.g.:

python mh_sampler.py \
    --vae-checkpoint /path/betatcvae.pth --latent-dim 10 \
    --attributes shape,scale --target-classes 2,1 \
    --classifier-checkpoints /path/shape.pth,/path/scale_fine.pth \
    --n-chains 1000 --n-steps 3000 --burn-in 1000 --min-beta 0.2 \
    --proposal-std 0.752622 --n-trials 15 --save-dir ./results
"""

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, List, Optional, Tuple

import torch
from tqdm import trange

from sampling_common import (
    rank_normalized_split_rhat_per_dimension,
    normal_log_prior,
    eval_active_attributes_nograd, combined_ll_from_dict,
    build_active_from, assemble_mcmc_trial_metrics,
    save_sample_grid, run_trials,
    load_vae, resolve_attribute_args,
    build_common_arg_parser, build_experiment_metadata, parse_csv,
)


@dataclass
class MhConfig:
    n_chains: int = 1000
    n_steps: int = 3000
    burn_in: int = 1000
    proposal_std: float = 0.1   # CONSTANT throughout the run -- no adaptation

    anneal_temperature: bool = True   # False -> flat beta=1.0 (likelihood), no tempering
    min_beta: float = 0.2             # starting (hot) exponent for the likelihood schedule
    anneal_fraction: float = 0.7      # fraction of burn_in spent ramping; rest holds beta=1.0

    prior_temp: float = 1.0           # Normal(0, prior_temp) -- the only prior/proposal here

    order_mode: str = "joint"
    attribute_order: List[str] = field(default_factory=list)
    switch_steps: Optional[List[int]] = None

    use_amp: bool = True
    device: str = "cuda"
    dtype: torch.dtype = torch.float32

    def __post_init__(self):
        assert self.order_mode in ("joint", "sequential")
        if self.n_chains < 20:
            print(f"[warn] n_chains={self.n_chains} is small for a reliable R-hat.")


def run_mh_sampling(
    decode: Callable,
    attribute_classifiers: Dict[str, Callable],
    attribute_targets: Dict[str, int],
    latent_dim: int,
    cfg: MhConfig,
    pre_switch_snapshot_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Runs ONE trial: cfg.n_chains chains for cfg.n_steps steps. Returns
    a metrics dict of PLAIN SCALARS (mean over the final per-chain state,
    for likelihood/entropy; r_hat and acceptance rate as their usual
    trial-level values) -- see sampling_common.assemble_mcmc_trial_metrics.
    Trial-to-trial variance is computed by the caller (run_trials), not
    here.
    """
    device = cfg.device
    amp_device = "cuda" if "cuda" in device else "cpu"
    n_chains = cfg.n_chains

    if cfg.anneal_temperature and cfg.burn_in > 0:
        anneal_steps = max(1, int(cfg.burn_in * cfg.anneal_fraction))
        lik_betas = torch.cat([
            torch.exp(torch.linspace(math.log(cfg.min_beta), math.log(1.0), anneal_steps, device=device)),
            torch.ones(cfg.n_steps - anneal_steps, device=device),
        ])
    else:
        lik_betas = torch.ones(cfg.n_steps, device=device)

    attrs = {
        name: dict(decode=decode, classify=attribute_classifiers[name], target=attribute_targets[name])
        for name in cfg.attribute_order
    }
    active_from, measurement_start = build_active_from(
        cfg.order_mode, cfg.attribute_order, cfg.switch_steps, cfg.burn_in
    )
    measurement_len = cfg.n_steps - measurement_start

    def prior_logp(z):
        return normal_log_prior(z, cfg.prior_temp)

    def eval_active(z, step):
        active_specs = {name: attrs[name] for name in attrs if step >= active_from[name]}
        return eval_active_attributes_nograd(z, active_specs, cfg.use_amp, amp_device)

    current_z = torch.randn(n_chains, latent_dim, device=device, dtype=cfg.dtype) * cfg.prior_temp ** 0.5

    proposal_z = torch.empty_like(current_z)
    noise_buffer = torch.empty_like(current_z)
    uniform_buffer = torch.empty(n_chains, device=device)

    proposal_std = cfg.proposal_std  


    z_trace_measurement = torch.empty(measurement_len, n_chains, latent_dim, device=device)
    z_trace_chain0 = torch.empty(cfg.n_steps, latent_dim, device=device)

    accept_count = 0
    accept_steps = 0

    switch_snapshot_steps = set((sw - 1) for sw in (cfg.switch_steps or []))

    with torch.inference_mode():
        pbar = trange(cfg.n_steps, desc="MH [normal / " + cfg.order_mode + "]")
        for step in pbar:
            temp = lik_betas[step]
            current_prior = prior_logp(current_z)
            current_ll_dict, current_ent_dict = eval_active(current_z, step)
            current_combined_ll = combined_ll_from_dict(current_ll_dict)
            current_log_target = current_prior + temp * current_combined_ll

            noise_buffer.normal_(mean=0.0, std=1.0)
            proposal_z.copy_(current_z + noise_buffer * proposal_std)
            prop_prior = prior_logp(proposal_z)
            prop_ll_dict, prop_ent_dict = eval_active(proposal_z, step)
            prop_combined_ll = combined_ll_from_dict(prop_ll_dict)
            proposal_log_target = prop_prior + temp * prop_combined_ll

            log_acceptance_ratio = proposal_log_target - current_log_target
            uniform_buffer.uniform_(0.0, 1.0)
            accept_mask = torch.log(uniform_buffer) <= log_acceptance_ratio

            current_z = torch.where(accept_mask.unsqueeze(1), proposal_z, current_z)
            current_prior = torch.where(accept_mask, prop_prior, current_prior)
            for name in current_ll_dict:
                current_ll_dict[name] = torch.where(accept_mask, prop_ll_dict[name], current_ll_dict[name])
                current_ent_dict[name] = torch.where(accept_mask, prop_ent_dict[name], current_ent_dict[name])
            current_combined_ll = combined_ll_from_dict(current_ll_dict)

            if step >= measurement_start:
                accept_count += accept_mask.float().sum().item()
                accept_steps += n_chains

            z_trace_chain0[step] = current_z[0]

            if pre_switch_snapshot_dir is not None and step in switch_snapshot_steps:
                snapshot_path = save_sample_grid(
                    decode, current_z, save_dir=pre_switch_snapshot_dir,
                    filename=f"mh_pre_switch_step{step + 1}.png",
                )
                print(f"[pre-switch snapshot] step {step}: saved to {snapshot_path}")

            if step >= measurement_start:
                z_trace_measurement[step - measurement_start] = current_z
                pbar.set_postfix({"accept": f"{accept_count / max(accept_steps, 1):.3f}"})
            else:
                pbar.set_postfix({"phase": "burn-in", "beta_lik": f"{temp.item():.3f}"})

    r_hat_per_dim = rank_normalized_split_rhat_per_dimension(z_trace_measurement)
    acceptance_rate = accept_count / max(accept_steps, 1)

    metrics = assemble_mcmc_trial_metrics(
        current_combined_ll, current_combined_ll.exp(),
        current_ll_dict, {k: v.exp() for k, v in current_ll_dict.items()},
        current_ent_dict,
        acceptance_rate, r_hat_per_dim,
    )
    metrics["anneal_temperature"] = cfg.anneal_temperature
    metrics["anneal_fraction"] = cfg.anneal_fraction if cfg.anneal_temperature else None

    final_joint_prob = current_combined_ll.exp().detach()

    return {
        "final_samples": current_z.detach(),
        "final_joint_prob": final_joint_prob,
        "z_trace_chain0": z_trace_chain0.detach(),
        "metrics": metrics,
    }


def main():
    parser = build_common_arg_parser("MH sampler with optional geometric temperature annealing")
    parser.add_argument("--proposal-std", type=float, default=0.1, help="constant proposal std -- no adaptation")
    parser.add_argument("--anneal-temperature", dest="anneal_temperature", action="store_true", default=True)
    parser.add_argument("--no-anneal-temperature", dest="anneal_temperature", action="store_false")
    parser.add_argument("--min-beta", type=float, default=0.2, help="starting (hot) exponent for the likelihood schedule")
    parser.add_argument(
        "--anneal-fraction", type=float, default=0.7,
        help="fraction of burn_in spent ramping; the rest holds beta=1.0 stationary",
    )
    args = parser.parse_args()

    attribute_names, targets, classifiers, attribute_order, switch_steps = resolve_attribute_args(args)
    model = load_vae(args.vae_checkpoint, args.latent_dim, args.device)

    cfg = MhConfig(
        n_chains=args.n_chains, n_steps=args.n_steps, burn_in=args.burn_in,
        proposal_std=args.proposal_std, anneal_temperature=args.anneal_temperature, min_beta=args.min_beta,
        anneal_fraction=args.anneal_fraction, prior_temp=args.prior_temp,
        order_mode=args.order_mode, attribute_order=attribute_order, switch_steps=switch_steps,
        device=args.device,
    )

    def run_one_trial(trial_idx: int) -> Dict[str, Any]:
        return run_mh_sampling(
            decode=model.decode, attribute_classifiers=classifiers, attribute_targets=targets,
            latent_dim=args.latent_dim, cfg=cfg,
            pre_switch_snapshot_dir=(args.save_dir if args.pre_switch_snapshot else None),
        )

    metadata = build_experiment_metadata(
        sampler="mh", attribute_names=attribute_names, targets=targets,
        order_mode=args.order_mode, attribute_order=attribute_order, switch_steps=switch_steps,
        tempering={"type": "geometric_annealing" if args.anneal_temperature else "none",
                   "min_beta": args.min_beta if args.anneal_temperature else None,
                   "anneal_fraction": args.anneal_fraction if args.anneal_temperature else None,
                   "likelihood_terminal_beta": 1.0 if args.anneal_temperature else None},
        vae_checkpoint=args.vae_checkpoint,
        classifier_checkpoints=dict(zip(attribute_names, parse_csv(args.classifier_checkpoints))),
    )

    result = run_trials(
        run_one_trial_fn=run_one_trial, n_trials=args.n_trials, decode=model.decode,
        save_dir=args.save_dir, method_name="mh", sampling_method_label="MH",
        select_mode="random", base_seed=args.seed, experiment_metadata=metadata,
        trace_plot_title="Trace plot - RWMH",
    )
    print("\n=== Aggregated metrics (mean, variance, std across trials) ===")
    print(result["aggregated_metrics"])


if __name__ == "__main__":
    main()