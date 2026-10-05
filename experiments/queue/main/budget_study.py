"""
MNPE budget convergence study — 5-category discrete setup (c ∈ {2,3,4,5,6}).

Trains MNPE at multiple simulation budgets with multiple repetitions (varying
NN training seed) and saves all results to a single pickle file for plotting.

Uses the deterministic quadrature/rejection posterior as the reference.

Observation preprocessing: log1p is applied to queue length dimensions (q1, q2)
before training and at inference time. This reduces the extreme right skew in queue
lengths (median ~0.1, max ~10) that otherwise causes z-scoring to compress the
informative low-queue range into a tiny sliver.

Usage:
    uv run python experiments/queue/main/budget_study.py

Outputs:
    budget_study_results.pkl
"""

import os
import pickle

import numpy as np
import torch
from sbi.inference import MNPE
from sbi.inference.posteriors.posterior_parameters import DirectPosteriorParameters

from mnpe_paper.models.queue.config import (
    C_VALUES,
    MODEL_VERSION,
    build_sbi_prior,
    generate_reference_observation,
    generate_training_data,
    simulator,
    transform_x,
)
from mnpe_paper.models.queue.eval import (
    build_tuned_mnpe,
    evaluate_posterior,
    format_metrics,
    summarize_c2st,
)
from mnpe_paper.models.queue.spec import (
    DISCRETE_OFFSET,
    SBC_TEST_SEED,
    SIMULATION_SEED,
    TUNED_HPARAMS,
)
from mnpe_paper.paths import model_artifacts_dir

# ── Settings ────────────────────────────────────────────────────────────────
BUDGETS = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000]
N_REPS = 5

N_SBC_TEST = 500
N_PRED = 500  # posterior predictive samples per method
N_PRIOR_PRED = 500  # prior predictive samples

ARTIFACTS_DIR = model_artifacts_dir("queue")
OUTPUT_FILE = str(ARTIFACTS_DIR / "budget_study_results.pkl")
POSTERIORS_DIR = ARTIFACTS_DIR / "posteriors"
REF_FILE = str(
    ARTIFACTS_DIR / "deterministic_reference.npz"
)

# ── Setup ───────────────────────────────────────────────────────────────────
print("=" * 72)
print("MNPE Budget Study (5-category discrete)")
print(f"  Budgets: {BUDGETS}, {N_REPS} reps each")
print("=" * 72)

prior = build_sbi_prior()
POSTERIORS_DIR.mkdir(exist_ok=True)
obs_data = generate_reference_observation(seed=42)
x_o_raw = torch.tensor(obs_data["x_array"], dtype=torch.float32)
x_o = transform_x(x_o_raw.unsqueeze(0)).squeeze(0)
x_o_np = obs_data["x_array"]

# Load and shuffle independent rejection-reference draws before slicing.
ref_data = np.load(REF_FILE)
ref_samples = np.column_stack(
    [
        ref_data["gamma"],
        ref_data["mu1"],
        ref_data["mu2"],
        ref_data["c1"],
        ref_data["c2"],
    ]
)
rng = np.random.default_rng(0)
rng.shuffle(ref_samples)

# Pre-generate largest budget; slice for smaller ones
print(f"\nGenerating {max(BUDGETS):,} training samples...")
theta_all, x_all_raw = generate_training_data(max(BUDGETS), seed=SIMULATION_SEED)
x_all = transform_x(x_all_raw)

# SBC test set (shared across budgets)
theta_test, x_test_raw = generate_training_data(N_SBC_TEST, seed=SBC_TEST_SEED)
x_test = transform_x(x_test_raw)

# Prior predictive (for panel a)
prior_theta = prior.sample((N_PRIOR_PRED,)).numpy()
prior_pred_obs = simulator(prior_theta)
prior_pred_valid = prior_pred_obs[~np.isnan(prior_pred_obs).any(axis=1)]

# ── Train and evaluate ──────────────────────────────────────────────────────
budget_results = {}

for budget in BUDGETS:
    print(f"\n{'=' * 72}")
    print(f"BUDGET: {budget:,}  ({N_REPS} reps)")
    print("=" * 72)

    theta_train = theta_all[:budget]
    x_train = x_all[:budget]

    reps = []
    for rep in range(N_REPS):
        rep_seed = rep * 1000  # deterministic but different per rep
        print(f"\n  -- Rep {rep + 1}/{N_REPS} (seed={rep_seed}) --")

        torch.manual_seed(rep_seed)
        trainer = MNPE(prior=prior, density_estimator=build_tuned_mnpe())
        trainer.append_simulations(theta_train, x_train)
        trainer.train(
            training_batch_size=TUNED_HPARAMS["training_batch_size"],
            learning_rate=TUNED_HPARAMS["learning_rate"],
        )
        post = trainer.build_posterior(
            posterior_parameters=DirectPosteriorParameters(enable_transform=False)
        )

        post_file = POSTERIORS_DIR / f"posterior_{budget}_rep{rep}.pkl"
        with open(post_file, "wb") as f:
            pickle.dump(post, f)

        metrics = evaluate_posterior(
            post, x_o, ref_samples, theta_test, x_test, prior
        )
        samples = metrics["samples"]

        # Posterior predictive (only for first rep to save space)
        if rep == 0:
            pred_theta = samples[:N_PRED].copy()
            pred_theta[:, 3:5] -= DISCRETE_OFFSET  # back to 0-indexed for simulator
            pred_obs = simulator(pred_theta)
        else:
            pred_obs = None

        reps.append({**metrics, "pred_obs": pred_obs})
        print(format_metrics(metrics))

    # Summary across reps
    print(f"\n  Summary ({N_REPS} reps):")
    print(summarize_c2st(reps, N_REPS))

    budget_results[budget] = {"reps": reps}

# ── Reference predictive ────────────────────────────────────────────────────
ref_pred_theta = ref_samples[:N_PRED].copy()
ref_pred_theta[:, 3:5] -= DISCRETE_OFFSET  # back to 0-indexed for simulator
ref_pred_obs = simulator(ref_pred_theta)

# ── Save ────────────────────────────────────────────────────────────────────
output = {
    "model_version": MODEL_VERSION,
    "budgets": BUDGETS,
    "n_reps": N_REPS,
    "hparams": TUNED_HPARAMS,
    "x_o": x_o_np,
    "ref_samples": ref_samples,
    "ref_pred_obs": ref_pred_obs,
    "prior_pred_obs": prior_pred_valid,
    "budget_results": budget_results,
    "reference_file": os.path.basename(REF_FILE),
    "discrete_values": C_VALUES.tolist(),
    "observation_transform": "log1p on dims [3, 4] (queue lengths)",
}

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(output, f)
print(f"\nSaved results to {OUTPUT_FILE}")
