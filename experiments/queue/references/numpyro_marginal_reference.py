"""Run the independent exact marginalized NumPyro/JAX queue reference.

Usage:
    uv run python experiments/queue/references/numpyro_marginal_reference.py

For a quick integration check:
    uv run python numpyro_marginal_reference.py \
        --warmup 100 --samples 100 --chains 2 --output /tmp/reference.npz
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import arviz as az
import jax
import numpy as np
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_value

from mnpe_paper.models.queue.config import generate_reference_observation
from mnpe_paper.models.queue.spec import GAMMA_MEDIAN, MODEL_VERSION
from mnpe_paper.models.queue.targets import (
    CONFIGURATIONS,
    batched_configuration_probabilities,
    marginalized_queue_model,
)
from mnpe_paper.paths import model_artifacts_dir

DEFAULT_STARTS = (
    (GAMMA_MEDIAN, 3.5, 3.0),
    (GAMMA_MEDIAN, 5.0, 4.0),
    (GAMMA_MEDIAN, 8.0, 5.0),
    (GAMMA_MEDIAN, 12.0, 10.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=2_000)
    parser.add_argument("--samples", type=int, default=5_000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-accept", type=float, default=0.95)
    parser.add_argument("--max-tree-depth", type=int, default=10)
    parser.add_argument("--save", type=int, default=10_000)
    parser.add_argument("--max-rhat", type=float, default=1.01)
    parser.add_argument(
        "--allow-nonconverged",
        action="store_true",
        help="Write an explicitly flagged diagnostic artifact despite bad R-hat.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=model_artifacts_dir("queue") / "numpyro_marginal_reference.npz",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    observation = generate_reference_observation(seed=42)
    observed = observation["x_dict"]

    print("=" * 72)
    print(f"Independent exact NumPyro reference: {MODEL_VERSION}")
    print("=" * 72)
    print(
        f"JAX {jax.__version__}; configurations={len(CONFIGURATIONS)}; "
        f"chains={args.chains}"
    )

    if args.chains > len(DEFAULT_STARTS):
        raise ValueError(
            f"At most {len(DEFAULT_STARTS)} dispersed chains are configured"
        )
    starts = DEFAULT_STARTS[: args.chains]
    chain_samples: list[dict[str, np.ndarray]] = []
    chain_divergences: list[np.ndarray] = []
    start = time.time()
    for chain_index, (gamma_start, mu1_start, mu2_start) in enumerate(starts):
        initial_values = {
            "gamma": gamma_start,
            "mu1": mu1_start,
            "mu2": mu2_start,
        }
        print(f"\nChain {chain_index + 1} initial values: {initial_values}")
        kernel = NUTS(
            marginalized_queue_model,
            target_accept_prob=args.target_accept,
            max_tree_depth=args.max_tree_depth,
            init_strategy=init_to_value(values=initial_values),
        )
        mcmc = MCMC(
            kernel,
            num_warmup=args.warmup,
            num_samples=args.samples,
            num_chains=1,
            progress_bar=True,
        )
        mcmc.run(
            jax.random.fold_in(jax.random.key(args.seed), chain_index),
            observations=observed,
        )
        chain_samples.append(
            {
                name: np.asarray(values)
                for name, values in mcmc.get_samples().items()
            }
        )
        chain_divergences.append(
            np.asarray(mcmc.get_extra_fields()["diverging"])
        )
    elapsed = time.time() - start

    samples_by_chain = {
        name: np.stack([chain[name] for chain in chain_samples])
        for name in ("gamma", "mu1", "mu2")
    }
    idata = az.from_dict(posterior=samples_by_chain)
    summary = az.summary(idata, var_names=["gamma", "mu1", "mu2"])
    print("\nContinuous diagnostics:")
    print(summary[["mean", "sd", "r_hat", "ess_bulk", "ess_tail"]])
    max_rhat = float(summary["r_hat"].max())
    samples = {
        name: values.reshape(-1)
        for name, values in samples_by_chain.items()
    }
    config_probs = batched_configuration_probabilities(samples, observed)
    discrete_posterior = config_probs.mean(axis=0)

    print("\nRao-Blackwellized joint discrete posterior:")
    order = np.argsort(-discrete_posterior)
    for index in order:
        c1, c2 = CONFIGURATIONS[index]
        print(
            f"  p({c1},{c2} | x) = {discrete_posterior[index]:.6f}"
        )

    rng = np.random.default_rng(args.seed)
    uniforms = rng.random(len(config_probs))
    config_indices = (
        np.cumsum(config_probs, axis=1) < uniforms[:, None]
    ).sum(axis=1)
    mixed_configs = CONFIGURATIONS[config_indices]
    selected = rng.permutation(len(samples["gamma"]))[
        : min(args.save, len(samples["gamma"]))
    ]
    divergences = int(np.concatenate(chain_divergences).sum())
    converged = np.isfinite(max_rhat) and max_rhat <= args.max_rhat
    if not converged and not args.allow_nonconverged:
        raise RuntimeError(
            f"Refusing to write reference: max R-hat {max_rhat:.3f} "
            f"exceeds {args.max_rhat:.3f}. Increase run length or improve "
            "multimodal sampling. Use --allow-nonconverged only for a "
            "diagnostic artifact."
        )

    np.savez(
        args.output,
        model_version=np.array(MODEL_VERSION),
        implementation=np.array("numpyro_exact_logsumexp_v1"),
        gamma=samples["gamma"][selected],
        mu1=samples["mu1"][selected],
        mu2=samples["mu2"][selected],
        c1=mixed_configs[selected, 0],
        c2=mixed_configs[selected, 1],
        configurations=CONFIGURATIONS,
        configuration_probabilities=discrete_posterior,
        divergences=np.array(divergences),
        max_rhat=np.array(max_rhat),
        converged=np.array(converged),
        initial_values=np.asarray(starts),
        elapsed_seconds=np.array(elapsed),
    )
    print(
        f"\nSaved {len(selected):,} samples to {args.output}; "
        f"divergences={divergences}; wall time={elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
