"""
Evaluate the 1M-budget queue posteriors with the budget-study metric suite.

`train_queue_1M.py` saves only the posterior and a single joint C2ST, so the 1M
runs cannot be compared with the budget-study points without this step. Loads the
three saved 1M posteriors (never overwrites them) and scores them through the
same `eval.evaluate_posterior` as the 1k..100k points.

Writes `queue_1M_eval.pkl` holding a rep list with the same keys as a
`budget_results[budget]["reps"]` entry. The name deliberately avoids the
protected `budget_study_results*.pkl` pattern.

Usage:
    uv run python experiments/queue/main/eval_queue_1M.py
"""

import pickle

import numpy as np
import torch

from mnpe_paper.models.queue.config import (
    MODEL_VERSION,
    build_sbi_prior,
    generate_reference_observation,
    generate_training_data,
    transform_x,
)
from mnpe_paper.models.queue.eval import evaluate_posterior, format_metrics
from mnpe_paper.models.queue.spec import SBC_TEST_SEED
from mnpe_paper.paths import model_artifacts_dir

# ── Settings ────────────────────────────────────────────────────────────────
BUDGET = 1_000_000
N_REPS = 3
N_SBC_TEST = 1_000

ARTIFACTS_DIR = model_artifacts_dir("queue")
REF_FILE = ARTIFACTS_DIR / "deterministic_reference.npz"
OUTPUT_FILE = ARTIFACTS_DIR / "queue_1M_eval.pkl"

print("=" * 72)
print(f"1M-budget evaluation ({N_REPS} reps), test set {N_SBC_TEST} cases")
print("=" * 72)

prior = build_sbi_prior()
obs_data = generate_reference_observation(seed=42)
x_o = transform_x(
    torch.tensor(obs_data["x_array"], dtype=torch.float32).unsqueeze(0)
).squeeze(0)

ref_data = np.load(REF_FILE)
ref_samples = np.column_stack(
    [ref_data["gamma"], ref_data["mu1"], ref_data["mu2"],
     ref_data["c1"], ref_data["c2"]]
)
rng = np.random.default_rng(0)
rng.shuffle(ref_samples)

print(f"\nGenerating {N_SBC_TEST} test samples (seed={SBC_TEST_SEED})...")
theta_test, x_test_raw = generate_training_data(N_SBC_TEST, seed=SBC_TEST_SEED)
x_test = transform_x(x_test_raw)
N_ACTUAL = len(theta_test)
print(f"  Using {N_ACTUAL} stable test cases")

reps = []
for rep in range(N_REPS):
    post_file = ARTIFACTS_DIR / f"posterior_queue_1M_rep{rep}.pkl"
    assert post_file.exists(), f"Missing posterior: {post_file}"

    print(f"\n  -- Rep {rep + 1}/{N_REPS} (loading {post_file.name}) --")
    with open(post_file, "rb") as f:
        post = pickle.load(f)

    torch.manual_seed(rep)  # a loaded posterior samples from the global torch RNG
    metrics = evaluate_posterior(
        post, x_o, ref_samples, theta_test, x_test, prior
    )
    reps.append({**metrics, "pred_obs": None})
    print(format_metrics(metrics))

c2st_vals = [r["c2st_all"] for r in reps]
print(f"\nSummary ({N_REPS} reps): C2ST joint {np.mean(c2st_vals):.4f} "
      f"+/- {np.std(c2st_vals):.4f} (sd)")

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(
        {
            "model_version": MODEL_VERSION,
            "budget": BUDGET,
            "n_reps": N_REPS,
            "reps": reps,
            "reference_file": REF_FILE.name,
            "n_sbc_test_actual": N_ACTUAL,
        },
        f,
    )
print(f"Saved -> {OUTPUT_FILE.name}")
