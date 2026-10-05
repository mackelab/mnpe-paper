"""Observed ECE vs. finite-sample baseline as a function of the bin count B.

Backs the closing paragraph of Appendix A.1 and Figure A-1. Both the observed
ECE and the noise-floor baseline depend on the bin count B, so the point is that
they move in lockstep: the calibration conclusion — observed close to baseline
under good calibration — is stable across B rather than an artifact of B = 10.

Uses ALL reps at the largest budget from the N=1000 re-evaluation, reported as
mean +/- sd across reps. Selecting a single rep would bias the observed-to-floor
distance: C2ST and ECE rank together here, so a C2ST-chosen rep is also the
ECE-best one (0.0259 against a 5-rep mean of 0.0310, ~1.8 sd low).

Observed ECE is top-label ECE, matching the paper.

Outputs:
  ece_observed_vs_baseline_vary_b.csv     — observed ECE + three baselines vs B
  ece_observed_vs_baseline_vary_b.pdf/png — one-panel figure

Run:  uv run python experiments/queue/diagnostics/ece_observed_vs_baseline_vary_b.py
"""

import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from mnpe_paper.calibration import (
    compute_ece,
    ece_baseline_analytical,
    ece_baseline_empirical,
    ece_baseline_equal_mass,
)
from mnpe_paper.paths import model_artifacts_dir, model_results_dir
from mnpe_paper.plotting.plot_style import COLORS, despine

ARTIFACTS_DIR = model_artifacts_dir("queue")
RESULTS_DIR = model_results_dir("queue")

RESULTS_FILE = ARTIFACTS_DIR / "budget_study_results_n1000.pkl"
with open(RESULTS_FILE, "rb") as f:
    study = pickle.load(f)

BUDGETS = study["budgets"]
budget_results = study["budget_results"]
K = len(study.get("discrete_values", [2, 3, 4, 5, 6]))

best_budget = max(BUDGETS)
reps = budget_results[best_budget]["reps"]
rep_probs = [torch.as_tensor(np.asarray(r["ece_probs"]), dtype=torch.float32) for r in reps]
rep_true = [torch.as_tensor(np.asarray(r["ece_true_vals"]), dtype=torch.long) for r in reps]
N, num_discrete, _ = rep_probs[0].shape
print(f"Budget={best_budget}, {len(reps)} reps, N={N}, K={K}")

B_values = [5, 10, 15, 20, 30, 50]

rows = []
for B in B_values:
    bin_edges = np.linspace(0, 1, B + 1)
    obs_per_rep, emp_per_rep, eqm_per_rep = [], [], []
    for probs, true_vals in zip(rep_probs, rep_true):
        obs_per_rep.append(
            compute_ece(true_vals, probs, num_bins=B, method="top_label").numpy()
        )
        emp_j, eqm_j = [], []
        for j in range(num_discrete):
            top_conf = probs[:, j, :].max(dim=-1).values.numpy()
            counts, _ = np.histogram(top_conf, bins=bin_edges)
            emp_j.append(ece_baseline_empirical(counts, B=B, K=K))
            eqm_j.append(ece_baseline_equal_mass(top_conf, B=B))
        emp_per_rep.append(emp_j)
        eqm_per_rep.append(eqm_j)

    # (n_reps, num_discrete) -> average the two discrete parameters within a
    # rep, then take mean/sd across reps, so the sd is across-rep variation.
    obs = np.asarray(obs_per_rep)
    emp = np.asarray(emp_per_rep)
    eqm = np.asarray(eqm_per_rep)
    obs_mean = float(obs.mean(axis=1).mean())
    baseline_emp = float(emp.mean(axis=1).mean())
    rows.append({
        "B": B,
        "n_reps": len(reps),
        "observed_c1": float(obs[:, 0].mean()),
        "observed_c2": float(obs[:, 1].mean()),
        "observed_mean": obs_mean,
        "observed_sd": float(obs.mean(axis=1).std()),
        "analytical": ece_baseline_analytical(N, B=B, K=K),
        "empirical": baseline_emp,
        "empirical_sd": float(emp.mean(axis=1).std()),
        "empirical_c1": float(emp[:, 0].mean()),
        "empirical_c2": float(emp[:, 1].mean()),
        "equal_mass": float(eqm.mean(axis=1).mean()),
        "observed_over_empirical": obs_mean / baseline_emp,
        "observed_over_empirical_c1": float(obs[:, 0].mean()) / float(emp[:, 0].mean()),
        "observed_over_empirical_c2": float(obs[:, 1].mean()) / float(emp[:, 1].mean()),
    })

df = pd.DataFrame(rows)
df.to_csv(RESULTS_DIR / "ece_observed_vs_baseline_vary_b.csv", index=False)
print(df.to_string(index=False))

# ── Figure (appendix single panel, paper style: rc fonts, no overrides) ──
fig, ax = plt.subplots(figsize=(2.8, 2.3))
ax.fill_between(df["B"], df["observed_mean"] - df["observed_sd"],
                df["observed_mean"] + df["observed_sd"],
                color=COLORS["MNPE"], alpha=0.20, lw=0)
ax.plot(df["B"], df["observed_mean"], "D-", color=COLORS["MNPE"],
        label=f"Observed (MNPE, {int(df['n_reps'].iloc[0])} reps)")
ax.plot(df["B"], df["analytical"], "o-", color="0.3", label="Baseline (analytical)")
ax.plot(df["B"], df["empirical"], "s--", color="C0", label="Baseline (empirical)")
ax.axvline(10, color="0.5", ls=":", lw=0.8, zorder=0)
ax.text(10, ax.get_ylim()[1] * 0.98, "  B=10", color="0.5", va="top")
ax.set_xlabel("B (ECE bins)")
ax.set_ylabel("ECE")
# Lower right: the curves all rise to the top-right corner, leaving that region
# free, and the B=10 annotation needs the top-left.
ax.legend(loc="lower right", frameon=False, handlelength=1.4, fontsize="x-small")
despine(ax, offset=3)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(ARTIFACTS_DIR / f"ece_observed_vs_baseline_vary_b.{ext}")
plt.close(fig)
print("Saved figure to ece_observed_vs_baseline_vary_b.pdf/png")
