"""Paper-grade MCMC reference: exact marginalization plus SMC, no quadrature.

Marginalizes the 25 server-count configurations by explicit ``logsumexp`` and
samples the resulting three-dimensional target with ``pm.sample_smc()``. NUTS
does not mix on this target; SMC does. Convergence is assessed across
independent runs, since SMC has no within-chain autocorrelation, and the
artifact is not written unless R-hat and the across-run configuration spread
are both within tolerance.

``--verify-against-pymc-marginalize`` checks the explicit logsumexp against
``pymc_extras.marginalize``.

Rationale, including why NUTS fails on this target and why the library's own
marginalization is not used directly, is in ../README.md under "Why the
reference is built this way".

Usage:
    uv run python experiments/queue/references/smc_reference.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import arviz as az
import numpy as np
import pymc as pm

from mnpe_paper.models.queue.config import (
    MODEL_VERSION,
    build_pymc_marginalized_model,
    configuration_probabilities_numpy,
    generate_reference_observation,
)
from mnpe_paper.paths import model_artifacts_dir

ARTIFACTS_DIR = model_artifacts_dir("queue")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=5_000)  # 5 runs x 4 chains x 5,000 = --save
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Independent SMC runs with different seeds. Across-run spread is "
        "the convergence diagnostic; SMC has no within-chain autocorrelation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=int, default=100_000)
    parser.add_argument(
        "--max-rhat",
        type=float,
        default=1.01,
        help="Refuse to write the artifact above this pooled R-hat.",
    )
    parser.add_argument(
        "--max-config-spread",
        type=float,
        default=0.01,
        help="Refuse to write if any configuration probability varies by more "
        "than this across independent runs.",
    )
    parser.add_argument(
        "--verify-against-pymc-marginalize",
        action="store_true",
        help="Check the explicit logsumexp density against pymc_extras.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def verify_against_pymc_marginalize(observations: dict) -> None:
    """Confirm the hand-written logsumexp matches PyMC's own marginalization.

    The stability guard is re-expressed as an observed Bernoulli so that
    ``marginalize`` will accept it. The two densities then differ only by the
    epsilon that trick requires, so they should agree closely wherever the
    constraint is satisfied.
    """
    import pytensor.tensor as pt
    from pymc_extras import marginalize

    from mnpe_paper.models.queue.config import (
        C_VALUES,
        GAMMA_MEDIAN,
        GAMMA_SIGMA,
        MAX_UTILIZATION,
        MU1_MEDIAN,
        MU1_SIGMA,
        MU2_MEDIAN,
        MU2_SIGMA,
        N_C,
        OBSERVATION_NOISE,
        T,
        _compute_E_Q_pytensor,
    )

    epsilon = 1e-300
    with pm.Model() as model:
        c1_idx = pm.Categorical("c1_idx", p=np.ones(N_C) / N_C)
        c2_idx = pm.Categorical("c2_idx", p=np.ones(N_C) / N_C)
        gamma = pm.LogNormal("gamma", mu=np.log(GAMMA_MEDIAN), sigma=GAMMA_SIGMA)
        mu1 = pm.LogNormal("mu1", mu=np.log(MU1_MEDIAN), sigma=MU1_SIGMA)
        mu2 = pm.LogNormal("mu2", mu=np.log(MU2_MEDIAN), sigma=MU2_SIGMA)

        # _compute_E_Q_pytensor needs a concrete server count, so build the
        # five branches and index them with the categorical.
        def per_station(mu):
            expectations, stabilities = [], []
            for c_value in C_VALUES:
                c_int = int(c_value)
                rho = gamma / (c_int * mu)
                is_stable = rho < MAX_UTILIZATION
                safe_rho = pt.switch(is_stable, rho, 0.5)
                expectations.append(
                    _compute_E_Q_pytensor(safe_rho * c_int, safe_rho, c_int)
                )
                stabilities.append(is_stable)
            return pt.stack(expectations), pt.stack(stabilities)

        e_q1_all, stable1_all = per_station(mu1)
        e_q2_all, stable2_all = per_station(mu2)
        e_q1 = e_q1_all[c1_idx]
        e_q2 = e_q2_all[c2_idx]
        stable = pt.and_(stable1_all[c1_idx], stable2_all[c2_idx])

        # The Potential equivalent, expressed as a likelihood term so that
        # marginalize() will accept the dependency on c1_idx and c2_idx.
        pm.Bernoulli(
            "stability",
            p=pt.switch(stable, 1.0 - epsilon, epsilon),
            observed=1,
        )

        pm.Poisson("n_arrivals_obs", mu=gamma * T, observed=observations["n_arrivals"])
        pm.Poisson(
            "n_completions_1_obs", mu=gamma * T, observed=observations["n_completions_1"]
        )
        pm.Poisson(
            "n_completions_2_obs", mu=gamma * T, observed=observations["n_completions_2"]
        )
        pm.TruncatedNormal(
            "queue_length_1_obs",
            mu=e_q1,
            sigma=OBSERVATION_NOISE,
            lower=0,
            observed=observations["queue_length_1"],
        )
        pm.TruncatedNormal(
            "queue_length_2_obs",
            mu=e_q2,
            sigma=OBSERVATION_NOISE,
            lower=0,
            observed=observations["queue_length_2"],
        )

    marginalized = marginalize(model, ["c1_idx", "c2_idx"])
    library_logp = marginalized.compile_logp()

    ours = build_pymc_marginalized_model(observations)
    our_logp = ours.compile_logp()

    print("\n── Verifying explicit logsumexp against pymc_extras.marginalize ──")
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(20):
        point = {
            "gamma_log__": float(np.log(rng.uniform(8.0, 10.0))),
            "mu1_log__": float(np.log(rng.uniform(4.0, 12.0))),
            "mu2_log__": float(np.log(rng.uniform(3.0, 10.0))),
        }
        a, b = float(our_logp(point)), float(library_logp(point))
        if np.isfinite(a) and np.isfinite(b):
            worst = max(worst, abs(a - b))
    print(f"  largest |log-density difference| over 20 points: {worst:.3e}")
    if worst < 1e-6:
        print("  MATCH: the explicit marginalization equals PyMC's own.")
    else:
        print("  WARNING: densities differ by more than 1e-6; investigate.")


def main() -> None:
    args = parse_args()
    observations = generate_reference_observation(seed=42)

    print("=" * 72)
    print(f"SMC MCMC reference (no quadrature): {MODEL_VERSION}")
    print("=" * 72)
    print(f"PyMC {pm.__version__}")
    print(
        f"{args.runs} independent SMC runs x {args.chains} chains x "
        f"{args.draws:,} draws"
    )

    if args.verify_against_pymc_marginalize:
        verify_against_pymc_marginalize(observations)

    model = build_pymc_marginalized_model(observations)

    per_run_configs = []
    gammas, mu1s, mu2s = [], [], []
    per_run_idata = []
    start = time.time()

    for run in range(args.runs):
        with model:
            idata = pm.sample_smc(
                draws=args.draws,
                chains=args.chains,
                random_seed=args.seed + run,
                progressbar=False,
            )
        gamma = idata.posterior["gamma"].values.reshape(-1)
        mu1 = idata.posterior["mu1"].values.reshape(-1)
        mu2 = idata.posterior["mu2"].values.reshape(-1)
        configs, probabilities = configuration_probabilities_numpy(
            gamma, mu1, mu2, observations
        )
        per_run_configs.append(probabilities.mean(axis=0))
        gammas.append(gamma)
        mu1s.append(mu1)
        mu2s.append(mu2)
        per_run_idata.append(idata)
        summary = az.summary(idata, var_names=["gamma", "mu1", "mu2"])
        print(
            f"  run {run}: max R-hat {summary['r_hat'].max():.4f}, "
            f"min ESS {summary['ess_bulk'].min():.0f}, "
            f"({time.time() - start:.0f}s cumulative)",
            flush=True,
        )

    elapsed = time.time() - start
    config_matrix = np.stack(per_run_configs)
    discrete_posterior = config_matrix.mean(axis=0)
    config_spread = config_matrix.std(axis=0, ddof=1)

    pooled = az.concat(per_run_idata, dim="chain")
    pooled_summary = az.summary(pooled, var_names=["gamma", "mu1", "mu2"])
    max_rhat = float(pooled_summary["r_hat"].max())

    print("\n── Pooled continuous diagnostics across all runs ──")
    print(pooled_summary[["mean", "sd", "r_hat", "ess_bulk"]])

    print("\n── Joint discrete posterior (mean over runs, +/- across-run sd) ──")
    order = np.argsort(-discrete_posterior)
    for index in order[:10]:
        c1, c2 = configs[index]
        print(
            f"  p({c1},{c2} | x) = {discrete_posterior[index]:.6f} "
            f"+/- {config_spread[index]:.6f}"
        )

    worst_spread = float(config_spread.max())
    print(f"\nLargest across-run configuration spread: {worst_spread:.6f}")
    print(f"Pooled max R-hat: {max_rhat:.4f}")
    print(f"Total wall time: {elapsed:.0f}s for {args.runs} runs")

    converged = max_rhat <= args.max_rhat
    stable_configs = worst_spread <= args.max_config_spread
    if not (converged and stable_configs):
        raise SystemExit(
            f"Refusing to write reference: R-hat {max_rhat:.4f} "
            f"(limit {args.max_rhat}), configuration spread {worst_spread:.6f} "
            f"(limit {args.max_config_spread})."
        )

    gamma_all = np.concatenate(gammas)
    mu1_all = np.concatenate(mu1s)
    mu2_all = np.concatenate(mu2s)
    rng = np.random.default_rng(args.seed)

    # Assign a configuration label per draw so the artifact carries mixed
    # posterior samples, matching the deterministic reference's schema.
    _, per_draw = configuration_probabilities_numpy(
        gamma_all, mu1_all, mu2_all, observations
    )
    uniforms = rng.random(len(per_draw))
    indices = (np.cumsum(per_draw, axis=1) < uniforms[:, None]).sum(axis=1)
    config_array = np.asarray(configs)
    selected = rng.permutation(len(gamma_all))[: min(args.save, len(gamma_all))]

    output = args.out or ARTIFACTS_DIR / "smc_reference.npz"
    np.savez(
        output,
        model_version=np.array(MODEL_VERSION),
        implementation=np.array("pymc_exact_logsumexp_smc_v1"),
        gamma=gamma_all[selected],
        mu1=mu1_all[selected],
        mu2=mu2_all[selected],
        c1=config_array[indices, 0][selected],
        c2=config_array[indices, 1][selected],
        configurations=config_array,
        configuration_probabilities=discrete_posterior,
        configuration_spread=config_spread,
        max_rhat=np.array(max_rhat),
        n_runs=np.array(args.runs),
        elapsed_seconds=np.array(elapsed),
        converged=np.array(True),
    )
    print(f"\nSaved {len(selected):,} samples to {output.name}")


if __name__ == "__main__":
    main()
