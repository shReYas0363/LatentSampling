"""
MALA posterior sampler over the VAE's latent space.

Additional details:

TRIALS (default --n-trials 15): within one trial, --n-chains (1000
default) independent chains run for the full n_steps, and each
likelihood/entropy metric is reduced to the MEAN of its FINAL per-chain
value across those chains -- a single scalar per trial, taken only at
the last step. Uncertainty is the VARIANCE (and std) of that scalar
ACROSS the n_trials trials. R-hat is the one exception: it inherently
needs the trajectory, so it's still computed from the full measurement-
window trace every trial.

The likelihood's inverse-temperature ramps geometrically from --min-beta
(hot) to 1.0 (cold, full strength) across the ENTIRE burn_in window (per
explicit instruction: MALA anneals across all of burn_in, unlike MH,
which anneals across only a fraction of it -- see mh_sampler.py).

Single chain population -- no ladder, no swaps. Toggle with
--anneal-temperature / --no-anneal-temperature; disabled means flat
beta=1.0 the whole run (plain, untempered MALA).

--step-size defaults to 0.005.

Run directly, e.g.:

python mala_sampler.py \
    --vae-checkpoint /path/betatcvae.pth --latent-dim 10 \
    --attributes shape,scale,position --target-classes 2,1,0 \
    --classifier-checkpoints /path/shape.pth,/path/scale_fine.pth,/path/position_coarse.pth \
    --n-chains 1000 --n-steps 3000 --burn-in 1000 --min-beta 0.2 \
    --step-size 0.005 --n-trials 15 --save-dir ./results
"""

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, List, Optional, Tuple

import torch
from tqdm import trange

from sampling_common import (
    rank_normalized_split_rhat_per_dimension,
    normal_logp_and_score,
    attrs_logp_and_score, combined_ll_from_dict,
    build_active_from, assemble_mcmc_trial_metrics,
    save_sample_grid, run_trials,
    load_vae, resolve_attribute_args,
    build_common_arg_parser, build_experiment_metadata, parse_csv,
)


@dataclass
class MalaConfig:
    n_chains: int = 1000
    n_steps: int = 3000
    burn_in: int = 1000
    step_size: float = 0.005  

    anneal_temperature: bool = True
    min_beta: float = 0.2  

    prior_temp: float = 1.0        

    order_mode: str = "joint"
    attribute_order: List[str] = field(default_factory=list)
    switch_steps: Optional[List[int]] = None

    device: str = "cuda"
    dtype: torch.dtype = torch.float32

    def __post_init__(self):
        assert self.order_mode in ("joint", "sequential")
        if self.n_chains < 20:
            print(f"[warn] n_chains={self.n_chains} is small for a reliable R-hat.")


def run_mala_sampling(
    decode: Callable,
    attribute_classifiers: Dict[str, Callable],
    attribute_targets: Dict[str, int],
    latent_dim: int,
    cfg: MalaConfig,
    pre_switch_snapshot_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Runs ONE trial. See mh_sampler.run_mh_sampling's docstring for the
    same final-state-mean-per-trial design; identical here.
    """
    device = cfg.device
    n_chains = cfg.n_chains

    if cfg.anneal_temperature and cfg.burn_in > 0:
        betas = torch.cat([
            torch.exp(torch.linspace(math.log(cfg.min_beta), math.log(1.0), cfg.burn_in, device=device)),
            torch.ones(cfg.n_steps - cfg.burn_in, device=device),
        ])
    else:
        betas = torch.ones(cfg.n_steps, device=device)

    attrs = {
        name: dict(decode=decode, classify=attribute_classifiers[name], target=attribute_targets[name])
        for name in cfg.attribute_order
    }
    active_from, measurement_start = build_active_from(
        cfg.order_mode, cfg.attribute_order, cfg.switch_steps, cfg.burn_in
    )

    def active_specs_at(step):
        return {name: attrs[name] for name in attrs if step >= active_from[name]}

    def prior_logp_and_score(z):
        return normal_logp_and_score(z, cfg.prior_temp)

    current_z = torch.randn(n_chains, latent_dim, device=device, dtype=cfg.dtype) * cfg.prior_temp ** 0.5

    noise_buffer = torch.empty(n_chains, latent_dim, device=device, dtype=cfg.dtype)
    proposal_z = torch.empty_like(current_z)
    uniform_buffer = torch.empty(n_chains, device=device)

    step_size = cfg.step_size 

    measurement_len = cfg.n_steps - measurement_start
    z_trace_measurement = torch.empty(measurement_len, n_chains, latent_dim, device=device)
    z_trace_chain0 = torch.empty(cfg.n_steps, latent_dim, device=device)

    accept_count = 0
    accept_steps = 0

    switch_snapshot_steps = set((sw - 1) for sw in (cfg.switch_steps or []))

    pbar = trange(cfg.n_steps, desc="MALA [normal / " + cfg.order_mode + "]")
    for step in pbar:
        temp = betas[step]
        active_specs = active_specs_at(step)

        z_start = current_z
        prior_logp_start, prior_score_start = prior_logp_and_score(z_start)
        ll_dict_start, ent_dict_start, lik_score_start = attrs_logp_and_score(z_start, active_specs)
        combined_ll_start = combined_ll_from_dict(ll_dict_start)
        log_target_start = prior_logp_start + temp * combined_ll_start
        score = prior_score_start + temp * lik_score_start

        noise_std = (2 * step_size) ** 0.5
        noise_buffer.normal_(mean=0.0, std=1.0)
        proposal_z.copy_(z_start + step_size * score + noise_std * noise_buffer)

        prop_prior_logp, prop_prior_score = prior_logp_and_score(proposal_z)
        prop_ll_dict, prop_ent_dict, prop_lik_score = attrs_logp_and_score(proposal_z, active_specs)
        prop_combined_ll = combined_ll_from_dict(prop_ll_dict)
        proposal_log_target = prop_prior_logp + temp * prop_combined_ll
        proposal_score = prop_prior_score + temp * prop_lik_score

        forward_drift = proposal_z - z_start - step_size * score
        log_q_proposal = -0.25 * torch.sum(forward_drift ** 2, dim=1) / step_size
        backward_drift = z_start - proposal_z - step_size * proposal_score
        log_q_current = -0.25 * torch.sum(backward_drift ** 2, dim=1) / step_size
        log_alpha = (proposal_log_target + log_q_current) - (log_target_start + log_q_proposal)

        uniform_buffer.uniform_(0.0, 1.0)
        accept_mask = torch.log(uniform_buffer) <= log_alpha

        with torch.no_grad():
            current_z = torch.where(accept_mask.unsqueeze(1), proposal_z, z_start)
            current_prior = torch.where(accept_mask, prop_prior_logp, prior_logp_start)
            current_ll_dict, current_ent_dict = {}, {}
            for name in ll_dict_start:
                current_ll_dict[name] = torch.where(accept_mask, prop_ll_dict[name], ll_dict_start[name])
                current_ent_dict[name] = torch.where(accept_mask, prop_ent_dict[name], ent_dict_start[name])
            current_combined_ll = combined_ll_from_dict(current_ll_dict)

            if step >= measurement_start:
                accept_count += accept_mask.float().sum().item()
                accept_steps += n_chains

            z_trace_chain0[step] = current_z[0]

            if pre_switch_snapshot_dir is not None and step in switch_snapshot_steps:
                snapshot_path = save_sample_grid(
                    decode, current_z, save_dir=pre_switch_snapshot_dir,
                    filename=f"mala_pre_switch_step{step + 1}.png",
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

    final_joint_prob = current_combined_ll.exp().detach()

    return {
        "final_samples": current_z.detach(),
        "final_joint_prob": final_joint_prob,
        "z_trace_chain0": z_trace_chain0.detach(),
        "metrics": metrics,
    }


def main():
    parser = build_common_arg_parser("MALA sampler with optional geometric temperature annealing")
    parser.add_argument("--step-size", type=float, default=0.005, help="constant step size -- no adaptation")
    parser.add_argument("--anneal-temperature", dest="anneal_temperature", action="store_true", default=True)
    parser.add_argument("--no-anneal-temperature", dest="anneal_temperature", action="store_false")
    parser.add_argument("--min-beta", type=float, default=0.2, help="starting (hot) exponent for the likelihood schedule")
    args = parser.parse_args()

    attribute_names, targets, classifiers, attribute_order, switch_steps = resolve_attribute_args(args)
    model = load_vae(args.vae_checkpoint, args.latent_dim, args.device)

    cfg = MalaConfig(
        n_chains=args.n_chains, n_steps=args.n_steps, burn_in=args.burn_in,
        step_size=args.step_size, anneal_temperature=args.anneal_temperature, min_beta=args.min_beta,
        prior_temp=args.prior_temp,
        order_mode=args.order_mode, attribute_order=attribute_order, switch_steps=switch_steps,
        device=args.device,
    )

    def run_one_trial(trial_idx: int) -> Dict[str, Any]:
        return run_mala_sampling(
            decode=model.decode, attribute_classifiers=classifiers, attribute_targets=targets,
            latent_dim=args.latent_dim, cfg=cfg,
            pre_switch_snapshot_dir=(args.save_dir if args.pre_switch_snapshot else None),
        )

    metadata = build_experiment_metadata(
        sampler="mala", attribute_names=attribute_names, targets=targets,
        order_mode=args.order_mode, attribute_order=attribute_order, switch_steps=switch_steps,
        tempering={"type": "geometric_annealing" if args.anneal_temperature else "none",
                   "min_beta": args.min_beta if args.anneal_temperature else None,
                   "likelihood_terminal_beta": 1.0 if args.anneal_temperature else None},
        vae_checkpoint=args.vae_checkpoint,
        classifier_checkpoints=dict(zip(attribute_names, parse_csv(args.classifier_checkpoints))),
    )

    trace_method_label = "Annealed MALA" if args.anneal_temperature else "MALA"
    result = run_trials(
        run_one_trial_fn=run_one_trial, n_trials=args.n_trials, decode=model.decode,
        save_dir=args.save_dir, method_name="mala", sampling_method_label="MALA",
        select_mode="random", base_seed=args.seed, experiment_metadata=metadata,
        trace_plot_title=f"Trace plot - {trace_method_label}",
    )
    print("\n=== Aggregated metrics (mean, variance, std across trials) ===")
    print(result["aggregated_metrics"])


if __name__ == "__main__":
    main()