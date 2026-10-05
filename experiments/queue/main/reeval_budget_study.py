"""
Re-evaluate the MNPE budget study with a larger SBC/ECE test set.

Loads saved versioned posteriors (never overwrites them) and re-runs
all evaluation metrics (C2ST, SBC, ECE) with 1,000 test cases.

Saves a separately versioned 1,000-case result file; the training-run result
file is not modified.

Usage:
    uv run python experiments/queue/main/reeval_budget_study.py
"""

import os
import pickle

import numpy as np
import torch

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
    evaluate_posterior,
    format_metrics,
    summarize_c2st,
)
from mnpe_paper.models.queue.spec import DISCRETE_OFFSET, SBC_TEST_SEED
from mnpe_paper.paths import model_artifacts_dir

# ── Settings ────────────────────────────────────────────────────────────────
BUDGETS = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000]
N_REPS = 5

N_SBC_TEST = 1_000
N_PRED = 500
N_PRIOR_PRED = 500

ARTIFACTS_DIR = model_artifacts_dir("queue")
POSTERIORS_DIR = ARTIFACTS_DIR / "posteriors"
REF_FILE = str(
    ARTIFACTS_DIR / "deterministic_reference.npz"
)
OUTPUT_FILE = str(
    ARTIFACTS_DIR / "budget_study_results_n1000.pkl"
)

# Safety: do not overwrite original results
ORIGINAL_FILE = ARTIFACTS_DIR / "budget_study_results.pkl"
assert ORIGINAL_FILE.exists(), "Original budget_study_results.pkl not found"
assert str(OUTPUT_FILE) != str(ORIGINAL_FILE), "Would overwrite original file!"

# ── Setup ───────────────────────────────────────────────────────────────────
print("=" * 72)
print("MNPE Budget Study — Re-evaluation with larger test set")
print(f"  Budgets: {BUDGETS}, {N_REPS} reps each")
print(f"  Test set: {N_SBC_TEST} stable prior-predictive cases")
print(f"  Output: {OUTPUT_FILE}")
print("=" * 72)

prior = build_sbi_prior()
obs_data = generate_reference_observation(seed=42)
x_o_raw = torch.tensor(obs_data["x_array"], dtype=torch.float32)
x_o = transform_x(x_o_raw.unsqueeze(0)).squeeze(0)
x_o_np = obs_data["x_array"]

# Load and shuffle independent rejection-reference draws.
ref_data = np.load(REF_FILE)
ref_samples = np.column_stack(
    [ref_data["gamma"], ref_data["mu1"], ref_data["mu2"],
     ref_data["c1"], ref_data["c2"]]
)
rng = np.random.default_rng(0)
rng.shuffle(ref_samples)

# SBC/ECE test set (larger than original)
print(f"\nGenerating {N_SBC_TEST} test samples (seed={SBC_TEST_SEED})...")
theta_test, x_test_raw = generate_training_data(N_SBC_TEST, seed=SBC_TEST_SEED)
x_test = transform_x(x_test_raw)
N_ACTUAL = len(theta_test)
print(f"  Using {N_ACTUAL} stable test cases")

# Prior predictive (for panel a — recompute to be self-contained)
prior_theta = prior.sample((N_PRIOR_PRED,)).numpy()
prior_pred_obs = simulator(prior_theta)
prior_pred_valid = prior_pred_obs[~np.isnan(prior_pred_obs).any(axis=1)]

# ── Load posteriors and evaluate ────────────────────────────────────────────
budget_results = {}

for budget in BUDGETS:
    print(f"\n{'=' * 72}")
    print(f"BUDGET: {budget:,}  ({N_REPS} reps) — eval only")
    print("=" * 72)

    reps = []
    for rep in range(N_REPS):
        post_file = POSTERIORS_DIR / f"posterior_{budget}_rep{rep}.pkl"
        assert post_file.exists(), f"Missing posterior: {post_file}"

        print(f"\n  -- Rep {rep + 1}/{N_REPS} (loading {post_file.name}) --")
        with open(post_file, "rb") as f:
            post = pickle.load(f)

        torch.manual_seed(rep)  # a loaded posterior samples from the global torch RNG
        metrics = evaluate_posterior(
            post, x_o, ref_samples, theta_test, x_test, prior
        )
        samples = metrics["samples"]

        # Posterior predictive (only first rep)
        if rep == 0:
            pred_theta = samples[:N_PRED].copy()
            pred_theta[:, 3:5] -= DISCRETE_OFFSET  # back to 0-indexed for simulator
            pred_obs = simulator(pred_theta)
        else:
            pred_obs = None

        reps.append({**metrics, "pred_obs": pred_obs})
        print(format_metrics(metrics))

    print(f"\n  Summary ({N_REPS} reps):")
    print(summarize_c2st(reps, N_REPS))

    budget_results[budget] = {"reps": reps}

# ── Reference predictive ────────────────────────────────────────────────────
ref_pred_theta = ref_samples[:N_PRED].copy()
ref_pred_theta[:, 3:5] -= DISCRETE_OFFSET
ref_pred_obs = simulator(ref_pred_theta)

# ── Save ────────────────────────────────────────────────────────────────────
# Load original to copy hparams
with open(ORIGINAL_FILE, "rb") as f:
    original = pickle.load(f)

output = {
    "model_version": MODEL_VERSION,
    "budgets": BUDGETS,
    "n_reps": N_REPS,
    "hparams": original["hparams"],
    "x_o": x_o_np,
    "ref_samples": ref_samples,
    "ref_pred_obs": ref_pred_obs,
    "prior_pred_obs": prior_pred_valid,
    "budget_results": budget_results,
    "reference_file": os.path.basename(REF_FILE),
    "discrete_values": C_VALUES.tolist(),
    "observation_transform": "log1p on dims [3, 4] (queue lengths)",
    "n_sbc_test_actual": N_ACTUAL,
}

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(output, f)
print(f"\nSaved results to {OUTPUT_FILE}")
print(f"  N_test = {N_ACTUAL}")
