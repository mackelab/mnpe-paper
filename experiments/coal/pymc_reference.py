"""PyMC reference posterior for the coal-mining-disaster changepoint model.

Uses ``pymc_extras.marginalize()`` to integrate out the discrete switchpoint,
following the approach in the PyMC documentation:
https://www.pymc.io/projects/examples/en/latest/howto/marginalizing-models.html

Strategy:
1. Define the standard model with a ``DiscreteUniform`` switchpoint.
2. Marginalize the switchpoint via ``pmx.marginalize()`` (logsumexp internally).
3. Sample the continuous parameters (early_rate, late_rate) with NUTS.
4. Recover the discrete switchpoint via ``pmx.recover_marginals()``.

Marginalising rather than sampling the discrete parameter directly is what makes
this reference trustworthy: a mixed-space sampler would have to move between 111
switchpoint values, and its mixing would then be the thing under suspicion when
MNPE and MCMC disagreed.

Usage:
    uv run python experiments/coal/pymc_reference.py

Outputs:
    artifacts/coal/pymc_reference_samples.npz
        - early_rate: shape (N,)
        - late_rate: shape (N,)
        - switchpoint: shape (N,) — 0-indexed (0 to N_YEARS-1)
"""

import time

import arviz as az
import numpy as np
import pymc as pm
import pymc_extras as pmx

from mnpe_paper.models.coal.config import (
    DISASTER_DATA,
    EARLY_RATE_TRUE,
    LATE_RATE_TRUE,
    N_YEARS,
    RATE_PRIOR_SCALE,
    YEAR_IDX,
    YEARS,
    idx_to_year,
)
from mnpe_paper.paths import model_artifacts_dir

# ── Settings ────────────────────────────────────────────────────────────────
DRAWS = 5000
TUNE = 2000
CHAINS = 4
RANDOM_SEED = 42
TARGET_ACCEPT = 0.95

OUTPUT_FILE = model_artifacts_dir("coal") / "pymc_reference_samples.npz"

# ── Print setup ─────────────────────────────────────────────────────────────
print("=" * 72)
print("PyMC Reference Posterior — Coal Mining Disaster Changepoint")
print(f"  {N_YEARS} switchpoints ({YEARS[0]}-{YEARS[-1]})")
print(f"  NUTS: {DRAWS} draws x {CHAINS} chains, tune={TUNE}")
print("=" * 72)

# ── Build model ─────────────────────────────────────────────────────────────
# Following the PyMC docs convention exactly:
#   switchpoint uses 0-indexed values (0 to N_YEARS-1)
#   rate = early_rate if year_idx < switchpoint else late_rate
# This matches the SBI simulator in mnpe_paper.models.coal.config.
with pm.Model() as disaster_model:
    switchpoint = pm.DiscreteUniform("switchpoint", lower=0, upper=N_YEARS - 1)
    early_rate = pm.Exponential("early_rate", RATE_PRIOR_SCALE)
    late_rate = pm.Exponential("late_rate", RATE_PRIOR_SCALE)

    rate = pm.math.switch(switchpoint > YEAR_IDX, early_rate, late_rate)
    pm.Poisson("disasters", rate, observed=DISASTER_DATA)

# ── Marginalize and sample ──────────────────────────────────────────────────
disaster_model_marg = pmx.marginalize(disaster_model, ["switchpoint"])

print("\nSampling with NUTS...")
t0 = time.time()

with disaster_model_marg:
    idata = pm.sample(
        draws=DRAWS,
        tune=TUNE,
        chains=CHAINS,
        random_seed=RANDOM_SEED,
        return_inferencedata=True,
        progressbar=True,
        nuts_sampler_kwargs={"target_accept": TARGET_ACCEPT},
    )

elapsed = time.time() - t0
print(f"\nSampling took {elapsed:.1f}s")

# ── Diagnostics ─────────────────────────────────────────────────────────────
summary = az.summary(idata, var_names=["early_rate", "late_rate"])
n_div = int(idata.sample_stats["diverging"].values.sum())
r_hat_max = summary["r_hat"].max()
ess_min = summary["ess_bulk"].min()

print("\nDiagnostics:")
print(f"  r_hat max: {r_hat_max:.3f}")
print(f"  ESS min:   {ess_min:.0f}")
print(f"  Divergences: {n_div}")
print(summary)

# ── Recover marginals ──────────────────────────────────────────────────────
idata = pmx.recover_marginals(idata, model=disaster_model_marg, random_seed=RANDOM_SEED)

early_samples = idata.posterior["early_rate"].values.flatten()
late_samples = idata.posterior["late_rate"].values.flatten()
switchpoint_samples = idata.posterior["switchpoint"].values.flatten().astype(int)

n_total = len(early_samples)
print(f"\n  Total samples: {n_total}")

# Switchpoint PMF
sp_counts = np.bincount(switchpoint_samples, minlength=N_YEARS) / n_total
map_idx = np.argmax(sp_counts)

print(f"  MAP switchpoint: {idx_to_year(map_idx)} (index {map_idx})")
print(f"  p(MAP): {sp_counts[map_idx]:.4f}")

# Top switchpoint years
sorted_idx = np.argsort(-sp_counts)
print("\n  Top switchpoint years:")
for i in range(min(10, N_YEARS)):
    idx = sorted_idx[i]
    if sp_counts[idx] < 0.001:
        break
    print(f"    {idx_to_year(idx)} (idx {idx}): p = {sp_counts[idx]:.4f}")

# Continuous parameter summaries
for param, samples, true_val in [
    ("early_rate", early_samples, EARLY_RATE_TRUE),
    ("late_rate", late_samples, LATE_RATE_TRUE),
]:
    lo, hi = np.percentile(samples, [2.5, 97.5])
    in_ci = lo <= true_val <= hi
    print(f"  {param:12s}: mean={samples.mean():.3f} +/- {samples.std():.3f}, "
          f"95% CI=[{lo:.3f}, {hi:.3f}] (true={true_val}) "
          f"{'OK' if in_ci else 'MISS'}")

# ── Save ────────────────────────────────────────────────────────────────────
np.savez(
    OUTPUT_FILE,
    early_rate=early_samples,
    late_rate=late_samples,
    switchpoint=switchpoint_samples,  # 0-indexed
)
print(f"\nSaved {n_total} samples to {OUTPUT_FILE}")
