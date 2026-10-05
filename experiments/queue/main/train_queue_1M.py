"""Train the queueing MNPE posterior at a 1M simulation budget.

Supplies the 1M-budget C2ST quoted in Section 4.2.

Pipeline is a faithful single-rep copy of budget_study.py (same TUNED_HPARAMS,
SIMULATION_SEED, build_mnpe + Identity embedding, DirectPosteriorParameters).
Writes to a separate file `artifacts/queue/posterior_queue_1M_rep{rep}.pkl` — it
does NOT touch the protected posteriors/ dir or budget_study_results*.pkl.

Run:  uv run python experiments/queue/main/train_queue_1M.py [--smoke]
"""

import argparse
import pickle

import numpy as np
import torch
from sbi.inference import MNPE
from sbi.inference.posteriors.posterior_parameters import DirectPosteriorParameters
from sbi.utils.metrics import c2st

from mnpe_paper.models.queue.config import (
    C_VALUES,
    build_sbi_prior,
    generate_reference_observation,
    generate_training_data,
    transform_x,
)
from mnpe_paper.models.queue.eval import build_tuned_mnpe
from mnpe_paper.models.queue.spec import SIMULATION_SEED, TUNED_HPARAMS
from mnpe_paper.paths import model_artifacts_dir, model_results_dir

ARTIFACTS_DIR = model_artifacts_dir("queue")
RESULTS_DIR = model_results_dir("queue")
REF_FILE = ARTIFACTS_DIR / "deterministic_reference.npz"

parser = argparse.ArgumentParser()
parser.add_argument("--budget", type=int, default=1_000_000)
parser.add_argument("--rep", type=int, default=0)
parser.add_argument("--smoke", action="store_true", help="tiny budget to validate the pipeline")
args = parser.parse_args()
budget = 2_000 if args.smoke else args.budget
rep_seed = args.rep * 1000
out_file = ARTIFACTS_DIR / (
    f"posterior_queue_1M_rep{args.rep}.pkl"
    if not args.smoke
    else "posterior_queue_1M_SMOKE.pkl"
)

print(f"Training queue MNPE: budget={budget:,}, rep={args.rep} (seed={rep_seed})")
prior = build_sbi_prior()

print(f"Generating {budget:,} prior-predictive training samples (seed={SIMULATION_SEED}) ...")
theta_all, x_all_raw = generate_training_data(budget, seed=SIMULATION_SEED)
x_all = transform_x(x_all_raw)
assert not torch.isnan(x_all_raw).any(), "Stable prior generated NaN observations"
print(f"  {len(theta_all):,} stable samples")

torch.manual_seed(rep_seed)
trainer = MNPE(prior=prior, density_estimator=build_tuned_mnpe())
trainer.append_simulations(theta_all, x_all)
trainer.train(
    training_batch_size=TUNED_HPARAMS["training_batch_size"],
    learning_rate=TUNED_HPARAMS["learning_rate"],
)
post = trainer.build_posterior(
    posterior_parameters=DirectPosteriorParameters(enable_transform=False)
)

with open(out_file, "wb") as f:
    pickle.dump(post, f)
print(f"Saved posterior -> {out_file.name}")

# ── Sanity: joint C2ST vs the shuffled deterministic reference at x_o ───────
obs = generate_reference_observation(seed=42)
x_o = transform_x(torch.tensor(obs["x_array"], dtype=torch.float32).unsqueeze(0)).squeeze(0)
ref = np.load(REF_FILE)
ref_samples = np.column_stack([ref["gamma"], ref["mu1"], ref["mu2"], ref["c1"], ref["c2"]])
rng = np.random.default_rng(0)
rng.shuffle(ref_samples)
# reference is in physical units (c 2..6); convert MNPE 0-indexed discretes to match
samples = post.sample((10_000,), x=x_o).numpy()
for col in (3, 4):
    samples[:, col] = np.array([C_VALUES[int(v)] for v in samples[:, col]])
n = min(len(samples), len(ref_samples), 10_000)
score = float(c2st(torch.tensor(samples[:n], dtype=torch.float32),
                   torch.tensor(ref_samples[:n], dtype=torch.float32)))
# Read the floor from file rather than restating it here.
floor_file = RESULTS_DIR / "c2st_floor.csv"
if floor_file.exists():
    import pandas as pd

    floor_row = pd.read_csv(floor_file).iloc[0]
    floor_text = f"floor {floor_row['mean']:.4f} +/- {floor_row['std']:.4f}"
else:
    floor_text = "floor unavailable; run posterior_metrics.py"
print(
    f"joint C2ST vs deterministic reference at x_o: {score:.3f} ({floor_text})"
)
