"""MNPE budget convergence study for the tractable Gaussian benchmark.

Trains MNPE at a range of simulation budgets with repeated runs, scores each
against the analytic posterior (C2ST, Wasserstein-1, total variation on the
discrete marginal), and saves the trained posterior from the largest budget.

Because the posterior here is available in closed form, this is the one
convergence study measured against a known answer rather than against another
sampler, which is what makes it the reference point for the other experiments.

Usage:
    uv run python experiments/toy/budget_study.py

Outputs (under ``artifacts/toy/``):
    budget_study_results.pkl   — convergence metrics per budget
    mnpe_posterior.pt          — posterior object from the largest budget

Both are pipeline artifacts rather than tracked results: the pickle carries
metrics the figure needs, but ``figures/make_toy_figure.py`` reads them through
the compact export written by ``scripts/export_figure_inputs.py``.
"""

import pickle
import warnings

import numpy as np
import torch
from sbi.inference import MNPE
from sbi.neural_nets import posterior_nn
from sbi.utils.metrics import c2st
from scipy.stats import wasserstein_distance

from mnpe_paper.models.toy.config import (
    p_z_given_x,
    sample_gt_posterior,
    sample_prior,
    simulator,
)
from mnpe_paper.paths import model_artifacts_dir

# ── Settings ────────────────────────────────────────────────────────────────
BUDGETS = [100, 200, 500, 1_000, 2_000, 5_000, 10_000]
N_REPEATS = 5
N_EVAL_SAMPLES = 1_000
N_TEST_OBS = 10
TEST_SEED = 789

# MNPE architecture (lightweight for this simple problem).
# `num_bins=10` is pinned so the architecture is visible here rather than
# inherited. Ten is what the published runs used: this experiment builds through
# `posterior_nn`, whose own default is 10 and which forwards it to the builder
# explicitly, so the low-level `_build_mixed_density_estimator` default (5 in the
# pinned sbi) never applies here. Pinning still earns its place: it is the factory
# default that is load-bearing here, and that could drift.
# `hidden_layers` has no effect: build_nsf takes it via **kwargs and discards it.
# `num_components` likewise never reaches the continuous head, because build_nsf
# has no such parameter and this project always uses flow_model="nsf".
HPARAMS = dict(
    hidden_features=32,
    num_transforms=3,
    num_bins=10,
    hidden_layers=3,
    num_blocks=4,
)

# A larger flow for the final model, whose draws the paper figure shows.
FINAL_HPARAMS = dict(
    hidden_features=32,
    num_transforms=5,
    num_bins=10,
    hidden_layers=3,
    num_blocks=4,
)

ARTIFACTS_DIR = model_artifacts_dir("toy")
OUTPUT_FILE = ARTIFACTS_DIR / "budget_study_results.pkl"
POSTERIOR_FILE = ARTIFACTS_DIR / "mnpe_posterior.pt"

# ── Fixed test observations (shared across all budgets) ─────────────────────
print("=" * 60)
print("MNPE Budget Study — Tractable Gaussian Benchmark")
print(f"Budgets: {BUDGETS}")
print(f"Repeats per budget: {N_REPEATS}")
print(f"Test observations: {N_TEST_OBS}")
print("=" * 60)

np.random.seed(TEST_SEED)
test_thetas = np.random.randn(N_TEST_OBS)
test_zs = np.random.randint(0, 2, N_TEST_OBS)

torch.manual_seed(TEST_SEED)
test_x_obs = []
for t, z in zip(test_thetas, test_zs):
    params = torch.tensor([[t, float(z)]])
    test_x_obs.append(simulator(params))

# ── Budget sweep ────────────────────────────────────────────────────────────
all_c2st = {n: [] for n in BUDGETS}
all_wass = {n: [] for n in BUDGETS}
all_tv = {n: [] for n in BUDGETS}

for n_sims in BUDGETS:
    for rep in range(N_REPEATS):
        seed = 1000 * rep + n_sims
        torch.manual_seed(seed)
        np.random.seed(seed)

        theta_train = sample_prior(n_sims)
        x_train = simulator(theta_train)

        de = posterior_nn(model="mnpe", **HPARAMS)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tr = MNPE(density_estimator=de, show_progress_bars=False)
            tr.append_simulations(theta_train, x_train).train(show_train_summary=False)
            post = tr.build_posterior()

        c2st_scores, w_dists, tv_dists = [], [], []
        for x_obs in test_x_obs:
            x_val = x_obs[0, 0].item()

            gt_theta, gt_z = sample_gt_posterior(x_val, N_EVAL_SAMPLES)
            gt_samples = torch.tensor(
                np.column_stack([gt_theta, gt_z]), dtype=torch.float32
            )

            mnpe_samples = post.sample(
                (N_EVAL_SAMPLES,), x=x_obs, show_progress_bars=False
            )

            c2st_scores.append(c2st(gt_samples, mnpe_samples, classifier="mlp").item())
            w_dists.append(wasserstein_distance(gt_theta, mnpe_samples[:, 0].numpy()))
            gt_pz1 = p_z_given_x(x_val, z=1)
            mnpe_pz1 = mnpe_samples[:, 1].numpy().mean()
            tv_dists.append(abs(gt_pz1 - mnpe_pz1))

        all_c2st[n_sims].append(np.mean(c2st_scores))
        all_wass[n_sims].append(np.mean(w_dists))
        all_tv[n_sims].append(np.mean(tv_dists))

        print(
            f"N={n_sims:>6d}  rep={rep}  "
            f"C2ST={np.mean(c2st_scores):.3f}  "
            f"Wass={np.mean(w_dists):.4f}  "
            f"TV={np.mean(tv_dists):.4f}"
        )

# ── Train the final model at the largest budget ─────────────────────────────
max_budget = max(BUDGETS)
print(f"\n{'=' * 60}")
print(f"Training final model at {max_budget:,} simulations")
print(f"Architecture: {FINAL_HPARAMS}")
print("=" * 60)

torch.manual_seed(42)
np.random.seed(42)

theta_final = sample_prior(max_budget)
x_final = simulator(theta_final)

de_final = posterior_nn(model="mnpe", **FINAL_HPARAMS)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    trainer_final = MNPE(density_estimator=de_final, show_progress_bars=True)
    trainer_final.append_simulations(theta_final, x_final).train(show_train_summary=True)
    posterior_final = trainer_final.build_posterior()

torch.save(posterior_final, POSTERIOR_FILE)
print(f"Saved posterior to {POSTERIOR_FILE}")

# ── Save results ────────────────────────────────────────────────────────────
output = {
    "budgets": BUDGETS,
    "hparams": HPARAMS,
    "final_hparams": FINAL_HPARAMS,
    "n_repeats": N_REPEATS,
    "n_test_obs": N_TEST_OBS,
    "all_c2st": all_c2st,
    "all_wass": all_wass,
    "all_tv": all_tv,
}

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(output, f)

print(f"\nSaved convergence results to {OUTPUT_FILE}")
print("Done.")
