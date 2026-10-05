"""Exact PyMC reference with both server counts marginalized by logsumexp.

The mixed target is the base product prior conditioned jointly on queue
stability. The global stability normalizer cancels from the posterior. For
every continuous parameter vector, this script sums the 25 stable
configuration likelihood terms exactly and samples only the resulting
three-dimensional continuous target with NUTS.

Discrete joint probabilities are recovered by Rao-Blackwellizing
``p(c1,c2 | gamma,mu1,mu2,x)`` over the continuous NUTS draws. Joint
configuration labels are then sampled once per draw for mixed-posterior C2ST.

Usage:
    uv run python experiments/queue/references/pymc_automarginal.py
"""

import time

import arviz as az
import numpy as np
import pymc as pm

from mnpe_paper.models.queue.config import (
    C1_TRUE,
    C2_TRUE,
    GAMMA_MEDIAN,
    GAMMA_TRUE,
    MODEL_VERSION,
    MU1_TRUE,
    MU2_TRUE,
    N_C,
    build_pymc_marginalized_model,
    configuration_probabilities_numpy,
    generate_reference_observation,
)
from mnpe_paper.paths import model_artifacts_dir

DRAWS = 5_000
TUNE = 2_000
CHAINS = 4
RANDOM_SEED = 42
TARGET_ACCEPT = 0.95
N_SAVE = 10_000
MAX_RHAT = 1.01

obs_data = generate_reference_observation(seed=42)
print("=" * 72)
print(f"Exact marginalized PyMC reference: {MODEL_VERSION}")
print("=" * 72)
print(f"PyMC {pm.__version__}; configurations={N_C**2}")
print(
    f"Observed counts: {obs_data['n_arrivals']}, "
    f"{obs_data['n_completions_1']}, {obs_data['n_completions_2']}"
)
print(
    f"Observed queues: {obs_data['queue_length_1']:.4f}, "
    f"{obs_data['queue_length_2']:.4f}"
)

model = build_pymc_marginalized_model(obs_data)
print(f"Free RVs: {[rv.name for rv in model.free_RVs]}")

# Dispersed starts help diagnose mode coverage rather than initializing every
# chain at the same point.
initvals = [
    {"gamma": GAMMA_MEDIAN, "mu1": mu1, "mu2": mu2}
    for mu1, mu2 in [(6.0, 4.0), (8.0, 5.0), (10.0, 7.0), (12.0, 10.0)]
][:CHAINS]

print(
    f"\nSampling {CHAINS} chains x ({TUNE} tune + {DRAWS} draws), "
    f"target_accept={TARGET_ACCEPT}"
)
t0 = time.time()
with model:
    idata = pm.sample(
        draws=DRAWS,
        tune=TUNE,
        chains=CHAINS,
        cores=CHAINS,
        random_seed=RANDOM_SEED,
        target_accept=TARGET_ACCEPT,
        init="adapt_diag",
        initvals=initvals,
        progressbar=True,
    )
elapsed = time.time() - t0

summary = az.summary(idata, var_names=["gamma", "mu1", "mu2"])
print("\nContinuous diagnostics:")
print(summary[["mean", "sd", "hdi_3%", "hdi_97%", "r_hat", "ess_bulk"]])

gamma = idata.posterior["gamma"].values.reshape(-1)
mu1 = idata.posterior["mu1"].values.reshape(-1)
mu2 = idata.posterior["mu2"].values.reshape(-1)
configs, config_probs = configuration_probabilities_numpy(
    gamma, mu1, mu2, obs_data
)
discrete_probs = config_probs.mean(axis=0)

print("\nRao-Blackwellized joint discrete posterior:")
for (c1, c2), probability in sorted(
    zip(configs, discrete_probs), key=lambda item: -item[1]
):
    marker = " <- TRUE" if (c1, c2) == (C1_TRUE, C2_TRUE) else ""
    print(f"  p({c1},{c2} | x) = {probability:.6f}{marker}")

rng = np.random.default_rng(RANDOM_SEED)
uniforms = rng.random(len(config_probs))
config_indices = (
    np.cumsum(config_probs, axis=1) < uniforms[:, None]
).sum(axis=1)
config_array = np.asarray(configs)
c1 = config_array[config_indices, 0]
c2 = config_array[config_indices, 1]

rng_indices = rng.permutation(len(gamma))[: min(N_SAVE, len(gamma))]

# NUTS does not mix on this target -- marginalizing the configurations leaves
# well-separated modes in (mu1, mu2) that gradient proposals cannot cross. Tag
# non-converged output so it can never be mistaken for a reference.
max_rhat = float(summary["r_hat"].max())
divergences = int(idata.sample_stats["diverging"].values.sum())
converged = np.isfinite(max_rhat) and max_rhat <= MAX_RHAT

suffix = "" if converged else "_NONCONVERGED"
reference_output = (
    model_artifacts_dir("queue") / f"pymc_marginal_reference{suffix}.npz"
)
np.savez(
    reference_output,
    model_version=np.array(MODEL_VERSION),
    gamma=gamma[rng_indices],
    mu1=mu1[rng_indices],
    mu2=mu2[rng_indices],
    c1=c1[rng_indices],
    c2=c2[rng_indices],
    max_rhat=np.array(max_rhat),
    divergences=np.array(divergences),
    converged=np.array(converged),
    discrete_posterior=np.array(
        [
            (c1_value, c2_value, probability)
            for (c1_value, c2_value), probability
            in zip(configs, discrete_probs)
        ],
        dtype=[("c1", int), ("c2", int), ("prob", float)],
    ),
)
print(f"\nSaved {len(rng_indices):,} mixed samples to {reference_output}")
print(f"Wall time: {elapsed:.1f}s")

if not converged:
    print(
        f"\n{'!' * 72}\n"
        f"NOT CONVERGED: max R-hat {max_rhat:.3f} exceeds {MAX_RHAT}, "
        f"{divergences} divergences.\n"
        "This artifact is a sampler diagnostic, NOT a reference posterior. Its\n"
        "configuration probabilities reflect where individual chains became\n"
        "trapped. Use deterministic_reference_*.npz as ground truth.\n"
        f"{'!' * 72}"
    )

print("\nTruth coverage:")
for name, truth in [
    ("gamma", GAMMA_TRUE),
    ("mu1", MU1_TRUE),
    ("mu2", MU2_TRUE),
]:
    hdi = az.hdi(idata, var_names=[name], hdi_prob=0.95)[name].values
    print(
        f"  {name}: true={truth:.3f}, HDI=[{hdi[0]:.3f}, {hdi[1]:.3f}], "
        f"covered={bool(hdi[0] <= truth <= hdi[1])}"
    )
