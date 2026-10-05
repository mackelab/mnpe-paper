"""ECE noise floor as a function of samples per bin, N/B (Appendix A.1).

Companion to `ece_observed_vs_baseline_vary_b.py`. That script showed observed
ECE and the noise floor move together as the bin count B changes. This script
makes the *governing quantity* explicit: for a perfectly calibrated model the
floor scales as

    E[ECE_floor]  ~  sqrt(2/pi) * sqrt(pbar (1-pbar)) * sqrt(B / N)   ~  1/sqrt(N/B),

so what matters is the number of samples per bin, n_bin = N/B, not B or N alone.
Increasing B at fixed N and decreasing N at fixed B are the same move in n_bin,
and both trace the same sqrt(B/N) floor.

Observed ECE at each N is obtained by subsampling the stored per-case
predictions (no inference, no retraining): the budget-study pickle carries
`ece_probs` (N,2,K) and `ece_true_vals` (N,2) per rep. The pickle is read-only;
no protected file is modified.

All reps at the largest budget are used. Taking the single lowest-C2ST rep
would be selection on the reported metric in disguise: C2ST and ECE rank
together here, so the C2ST-best rep is also the ECE-best one. Two spreads are
distinguished — `observed_std` is the subsampling spread at
that N (the band in panel B, which is what this figure is about), while
`observed_sd_across_reps` is the run-to-run spread.

Outputs:
  ece_observed_vs_baseline_vary_n.csv  — observed ECE + floors vs N (fixed B)
  ece_samples_per_bin.pdf/png          — 2 panels: (A) vs B at fixed N,
                                         (B) observed + floor vs N/B (log-log)

Run:  uv run python experiments/queue/diagnostics/ece_samples_per_bin.py
"""

import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from mnpe_paper.calibration import compute_ece, ece_baseline_analytical, ece_baseline_empirical
from mnpe_paper.paths import model_artifacts_dir, model_results_dir
from mnpe_paper.plotting.plot_style import COLORS, despine

ARTIFACTS_DIR = model_artifacts_dir("queue")
RESULTS_DIR = model_results_dir("queue")
B_FIXED = 10  # bin count held fixed for the N-sweep (paper default)
N_MC = 1000  # bootstrap subsamples per N for the observed-ECE band
SEED = 42

# ── Load the largest-budget per-case discrete predictions (read-only) ───────
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
N_stored, num_discrete, _ = rep_probs[0].shape
print(f"Budget={best_budget}, {len(reps)} reps, N_stored={N_stored}, K={K}")

# ── Observed ECE and floor vs N at fixed B (subsample without replacement) ──
rng = np.random.default_rng(SEED)
N_values = [n for n in [50, 100, 200, 300, 500, 750, 1000] if n <= N_stored]
bin_edges = np.linspace(0, 1, B_FIXED + 1)

top_conf_full = [
    [probs[:, j, :].max(dim=-1).values.numpy() for j in range(num_discrete)]
    for probs in rep_probs
]

rows = []
for N in N_values:
    obs_rep_means, obs_rep_boot_sd, emp_rep_means = [], [], []
    for r, (probs, true_vals) in enumerate(zip(rep_probs, rep_true)):
        obs_trials = np.empty(N_MC)
        emp_trials = np.empty(N_MC)  # empirical floor on the SAME N-sized occupancy
        for t in range(N_MC):
            idx = rng.choice(N_stored, size=N, replace=False)
            obs = compute_ece(
                true_vals[idx], probs[idx], num_bins=B_FIXED, method="top_label"
            )
            obs_trials[t] = float(obs.mean())
            emp_j = []
            for j in range(num_discrete):
                counts, _ = np.histogram(top_conf_full[r][j][idx], bins=bin_edges)
                emp_j.append(ece_baseline_empirical(counts, B=B_FIXED, K=K))
            emp_trials[t] = float(np.mean(emp_j))
        obs_rep_means.append(float(obs_trials.mean()))
        obs_rep_boot_sd.append(float(obs_trials.std()))
        emp_rep_means.append(float(emp_trials.mean()))

    rows.append({
        "N": N,
        "samples_per_bin": N / B_FIXED,
        "n_reps": len(reps),
        "observed_mean": float(np.mean(obs_rep_means)),
        # Subsampling spread at this N, averaged over reps -- the quantity the
        # panel-B band is about. Run-to-run spread is reported separately.
        "observed_std": float(np.mean(obs_rep_boot_sd)),
        "observed_sd_across_reps": float(np.std(obs_rep_means)),
        "analytical": ece_baseline_analytical(N, B=B_FIXED, K=K),
        "empirical": float(np.mean(emp_rep_means)),
    })

df_n = pd.DataFrame(rows)
df_n.to_csv(RESULTS_DIR / "ece_observed_vs_baseline_vary_n.csv", index=False)
print(df_n.to_string(index=False))

# ── Load the fixed-N, varying-B companion for panel A + the B-side of N/B ───
df_b = pd.read_csv(RESULTS_DIR / "ece_observed_vs_baseline_vary_b.csv")
N_B_SWEEP = 1000  # the vary-B sweep was run at N=1000 (analytical floors match)
df_b = df_b.assign(samples_per_bin=N_B_SWEEP / df_b["B"])

# ── Figure ──────────────────────────────────────────────────────────────────
fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.0, 2.5))

# Panel A: observed + floor vs B at fixed N (the lockstep view)
axA.fill_between(df_b["B"], df_b["observed_mean"] - df_b["observed_sd"],
                 df_b["observed_mean"] + df_b["observed_sd"],
                 color=COLORS["MNPE"], alpha=0.20, lw=0)
axA.plot(df_b["B"], df_b["observed_mean"], "D-", ms=3, color=COLORS["MNPE"],
         label="observed (MNPE)")
axA.plot(df_b["B"], df_b["analytical"], "o-", ms=3, color="0.3",
         label="floor (analytical)")
axA.plot(df_b["B"], df_b["empirical"], "s--", ms=3, color="C0",
         label="floor (empirical)")
axA.set_xlabel("$B$ (ECE bins), $N=1000$")
axA.set_ylabel("ECE")
axA.legend(frameon=False, handlelength=1.4, fontsize="x-small")

# Panel B: observed + floor vs samples-per-bin N/B (log-log), both sweeps
axB.plot(df_n["samples_per_bin"], df_n["analytical"], "o-", ms=3, color="0.3",
         label=r"floor, vary $N$ ($B{=}10$)")
axB.plot(df_b["samples_per_bin"], df_b["analytical"], "^--", ms=3, color="0.55",
         label=r"floor, vary $B$ ($N{=}1000$)")
axB.plot(df_n["samples_per_bin"], df_n["observed_mean"], "D-", ms=3,
         color=COLORS["MNPE"], label=r"observed, vary $N$")
axB.fill_between(df_n["samples_per_bin"],
                 df_n["observed_mean"] - df_n["observed_std"],
                 df_n["observed_mean"] + df_n["observed_std"],
                 color=COLORS["MNPE"], alpha=0.20, lw=0)
# sqrt(B/N) reference line anchored to the analytical floor at N/B=100
anchor = df_n.loc[df_n["samples_per_bin"] == 100, "analytical"]
c = float(anchor.iloc[0]) * np.sqrt(100.0) if len(anchor) else float(
    df_n["analytical"].iloc[-1]) * np.sqrt(df_n["samples_per_bin"].iloc[-1])
spb = np.array(sorted(set(df_n["samples_per_bin"]) | set(df_b["samples_per_bin"])))
axB.plot(spb, c / np.sqrt(spb), ":", color="k", lw=1.0,
         label=r"$\propto (N/B)^{-1/2}$")
axB.set_xscale("log")
axB.set_yscale("log")
axB.set_xlabel(r"samples per bin $N/B$")
axB.set_ylabel("ECE")
axB.legend(frameon=False, handlelength=1.6, fontsize="x-small")

for ax in (axA, axB):
    despine(ax, offset=3)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(ARTIFACTS_DIR / f"ece_samples_per_bin.{ext}")
plt.close(fig)
print("Saved ece_observed_vs_baseline_vary_n.csv and ece_samples_per_bin.pdf/png")
