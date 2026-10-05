"""Dequantization baseline on the tractable toy model (Appendix B.3, Table B-1).

Standard NPE on (theta, z + u) with u ~ U[0,1), evaluated against the ANALYTIC
posterior — the cleanest setting for the dequantization question, because every
error is exactly measurable, including the coupling:

  - discrete posterior error |q(z=1|x) - p(z=1|x)| (mass of samples in [1,2))
  - joint C2ST vs analytic posterior samples (floored z)
  - per-z conditional moments vs the exact N((x - a z)/tau, v)

The trained toy MNPE posterior is scored with the same metrics for a
side-by-side comparison, so the two columns of Table B-1 differ only in how the
discrete parameter is represented.

Requires ``artifacts/toy/mnpe_posterior.pt`` from ``budget_study.py``.

Usage:
    uv run python experiments/toy/dequantization_baseline_toy.py

Outputs:
    results/toy/dequantization_baseline_toy.csv — per (method, seed, x_o) metrics
"""

import numpy as np
import pandas as pd
import torch
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from sbi.utils import MultipleIndependent
from torch.distributions import Normal, Uniform

from mnpe_paper.models.toy.config import sample_prior, simulator
from mnpe_paper.models.toy.eval import N_SAMPLES, X_GRID, eval_samples
from mnpe_paper.paths import model_artifacts_dir, model_results_dir

ARTIFACTS_DIR = model_artifacts_dir("toy")
RESULTS_DIR = model_results_dir("toy")
POSTERIOR_FILE = ARTIFACTS_DIR / "mnpe_posterior.pt"

BUDGET = 10_000
N_SEEDS = 3

if not POSTERIOR_FILE.exists():
    raise SystemExit(
        f"{POSTERIOR_FILE} is missing — the MNPE column of this comparison is the "
        "trained estimator itself. Run experiments/toy/budget_study.py first."
    )

rows = []

# ── Dequantized NPE, several seeds ────────────────────────────────────────
prior = MultipleIndependent([
    Normal(torch.zeros(1), torch.ones(1)),
    Uniform(torch.zeros(1), torch.full((1,), 2.0)),
])

for seed in range(N_SEEDS):
    print(f"── dequant NPE seed {seed} ──")
    torch.manual_seed(seed * 1000)
    rng = np.random.default_rng(seed)

    params = sample_prior(BUDGET)  # (N, 2): [theta, z]
    x = simulator(params)
    params_deq = params.clone()
    params_deq[:, 1] = params[:, 1] + torch.from_numpy(rng.uniform(size=BUDGET)).float()

    # Match the toy MNPE checkpoint's actual continuous flow: nflows NSF with
    # FINAL_HPARAMS hidden_features=32, num_transforms=5, num_blocks=4
    # (hidden_layers is a dead kwarg in build_mnpe, so it is not matched here).
    build_fn = posterior_nn(
        model="nsf", num_transforms=5, hidden_features=32, num_blocks=4
    )
    trainer = NPE(prior=prior, density_estimator=build_fn)
    trainer.append_simulations(params_deq, x)
    trainer.train()
    post = trainer.build_posterior()

    for x_val in X_GRID:
        s = post.sample(
            (N_SAMPLES,), x=torch.tensor([[x_val]]), show_progress_bars=False
        ).numpy()
        z_s = np.clip(np.floor(s[:, 1]), 0, 1)
        rows.append(eval_samples(s[:, 0], z_s, x_val, "dequant_npe", seed))
        print(f"  x={x_val:+.1f}: c2st={rows[-1]['c2st_joint']:.3f}, "
              f"p(z=1) err={rows[-1]['p_z1_abs_err']:.4f}")

# ── Trained MNPE (existing artifact) ──────────────────────────────────────
print(f"── trained MNPE ({POSTERIOR_FILE.name}) ──")
post_mnpe = torch.load(POSTERIOR_FILE, weights_only=False)
torch.manual_seed(0)
for x_val in X_GRID:
    s = post_mnpe.sample(
        (N_SAMPLES,), x=torch.tensor([[x_val]]), show_progress_bars=False
    ).numpy()
    rows.append(eval_samples(s[:, 0], s[:, 1], x_val, "mnpe", 0))
    print(f"  x={x_val:+.1f}: c2st={rows[-1]['c2st_joint']:.3f}, "
          f"p(z=1) err={rows[-1]['p_z1_abs_err']:.4f}")

df = pd.DataFrame(rows)
df.to_csv(RESULTS_DIR / "dequantization_baseline_toy.csv", index=False)

print("\n── Mean over x grid ──")
summary = df.groupby("method")[
    ["c2st_joint", "p_z1_abs_err", "cond_mean_abs_err_z0", "cond_mean_abs_err_z1"]
].mean().round(4)
print(summary.to_string())
print("Saved dequantization_baseline_toy.csv")
