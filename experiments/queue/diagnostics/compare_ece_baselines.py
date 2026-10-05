"""Compare three ECE baselines under perfect calibration.

For a K-class problem with N SBC test cases and B ECE bins, the expected
ECE of a perfectly calibrated model is nonzero due to finite-sample noise.

Three baselines:
  1. Analytical — equal-width bins, equal occupancy (N / B_active per bin)
  2. Empirical  — equal-width bins, empirical occupancy from model bin counts
  3. Equal-mass — adaptive quantile bins, per-sample confidence scores

Outputs:
  ece_baseline_across_budgets.csv  — three baselines for each training budget
  ece_baseline_vary_n.csv          — three baselines as a function of N
  ece_baseline_vary_k.csv          — three baselines vs K
  ece_baseline_vary_b.csv          — three baselines vs B (bin count)
  ece_baseline_sensitivity.pdf/png — 3-panel figure
  ece_baseline_vary_b.pdf/png      — 1-panel vary-B figure

Run:  uv run python experiments/queue/diagnostics/compare_ece_baselines.py
"""

import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mnpe_paper.calibration import (
    ece_baseline_analytical,
    ece_baseline_empirical,
    ece_baseline_equal_mass,
)
from mnpe_paper.paths import model_artifacts_dir, model_results_dir
from mnpe_paper.plotting.plot_style import FS_LABEL, FS_LEGEND, despine

ARTIFACTS_DIR = model_artifacts_dir("queue")
RESULTS_DIR = model_results_dir("queue")

# ── Load budget study ─────────────────────────────────────────────────
# Prefer the n1000 results if available, fall back to original.
RESULTS_FILE = ARTIFACTS_DIR / "budget_study_results_n1000.pkl"
if not RESULTS_FILE.exists():
    RESULTS_FILE = ARTIFACTS_DIR / "budget_study_results.pkl"
    print(f"Note: n1000 results not found, using {RESULTS_FILE.name}")

with open(RESULTS_FILE, "rb") as f:
    study = pickle.load(f)

print(f"Loaded: {RESULTS_FILE.name}")

BUDGETS = study["budgets"]
budget_results = study["budget_results"]
K = len(study.get("discrete_values", [2, 3, 4, 5, 6]))
B = 10

# All reps, not the lowest-C2ST one. The empirical and equal-mass baselines
# depend on the model's confidence histogram, so picking a single rep would make
# these baselines depend on an arbitrary (and quality-correlated) choice of run.
best_budget = max(BUDGETS)
all_reps = budget_results[best_budget]["reps"]


# ═══════════════════════════════════════════════════════════════════════
# 1. Three baselines across training budgets
# ═══════════════════════════════════════════════════════════════════════

rows_budget = []
for b in BUDGETS:
    reps = budget_results[b]["reps"]
    N_sbc = reps[0]["ece_probs"].shape[0]
    val_analytical = ece_baseline_analytical(N_sbc, B=B, K=K)

    bin_edges_budget = np.linspace(0, 1, B + 1)
    emp_per_rep, eqm_per_rep = [], []
    for rep in reps:
        emp_vals, eqm_vals = [], []
        for j in range(2):
            top_conf = rep["ece_probs"][:, j, :].max(axis=1)
            bin_counts_j, _ = np.histogram(top_conf, bins=bin_edges_budget)
            emp_vals.append(ece_baseline_empirical(bin_counts_j, B=B, K=K))
            eqm_vals.append(ece_baseline_equal_mass(top_conf, B=B))
        emp_per_rep.append(np.mean(emp_vals))
        eqm_per_rep.append(np.mean(eqm_vals))

    rows_budget.append({
        "budget": b,
        "N_sbc": N_sbc,
        "n_reps": len(reps),
        "analytical": val_analytical,
        "empirical": float(np.mean(emp_per_rep)),
        "empirical_sd": float(np.std(emp_per_rep)),
        "equal_mass": float(np.mean(eqm_per_rep)),
        "equal_mass_sd": float(np.std(eqm_per_rep)),
    })

df_budget = pd.DataFrame(rows_budget)
df_budget.to_csv(RESULTS_DIR / "ece_baseline_across_budgets.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════
# 2. Three baselines as a function of N (subsampled from the largest budget)
# ═══════════════════════════════════════════════════════════════════════
# Use per-sample confidences at the largest budget. For each N, draw N
# samples without replacement, compute all three baselines on each draw,
# and average over MC trials. N is capped at N_stored.

# Pool the confidences of all reps at the largest budget, so the sampled
# baselines describe a typical run rather than one selected run.
ref_confidences = []
for j in range(2):
    ref_confidences.append(
        np.concatenate([r["ece_probs"][:, j, :].max(axis=1) for r in all_reps])
    )

N_stored = all_reps[0]["ece_probs"].shape[0]
print(f"N_stored = {N_stored} per rep, pooled over {len(all_reps)} reps "
      f"at budget {best_budget}")

rng = np.random.default_rng(42)
n_mc = 2000
N_values = [10, 20, 50, 100, 200, 300, 500, 750, 1000]
N_values = [n for n in N_values if n <= N_stored]

bin_edges_sub = np.linspace(0, 1, B + 1)

rows_vary_n = []
for N in N_values:
    val_analytical = ece_baseline_analytical(N, B=B, K=K)

    emp_trials = []
    eqm_trials = []
    for j in range(2):
        conf_ref = ref_confidences[j]
        emp_j = []
        eqm_j = []
        for _ in range(n_mc):
            draw = rng.choice(conf_ref, size=N, replace=False)
            counts, _ = np.histogram(draw, bins=bin_edges_sub)
            emp_j.append(ece_baseline_empirical(counts, B=B, K=K))
            eqm_j.append(ece_baseline_equal_mass(draw, B=B))
        emp_trials.append(float(np.mean(emp_j)))
        eqm_trials.append(float(np.mean(eqm_j)))

    rows_vary_n.append({
        "N": N,
        "analytical": val_analytical,
        "empirical": np.mean(emp_trials),
        "equal_mass": np.mean(eqm_trials),
    })

df_vary_n = pd.DataFrame(rows_vary_n)
df_vary_n.to_csv(RESULTS_DIR / "ece_baseline_vary_n.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════
# 3. Three baselines: vary K (number of classes)
# ═══════════════════════════════════════════════════════════════════════
# K affects which bins are "active" (upper edge > 1/K) for analytical and
# empirical. Equal-mass is K-independent — bins adapt to the data.

K_values = [2, 3, 4, 5, 6, 8, 10, 20]

rows_vary_k = []
for K_val in K_values:
    val_analytical = ece_baseline_analytical(N_stored, B=B, K=K_val)

    emp_vals = []
    eqm_vals = []
    for j in range(2):
        conf_j = ref_confidences[j]
        bin_edges_k = np.linspace(0, 1, B + 1)
        counts_j, _ = np.histogram(conf_j, bins=bin_edges_k)
        emp_vals.append(ece_baseline_empirical(counts_j, B=B, K=K_val))
        eqm_vals.append(ece_baseline_equal_mass(conf_j, B=B))

    rows_vary_k.append({
        "K": K_val,
        "analytical": val_analytical,
        "empirical": np.mean(emp_vals),
        "equal_mass": np.mean(eqm_vals),
    })

df_vary_k = pd.DataFrame(rows_vary_k)
df_vary_k.to_csv(RESULTS_DIR / "ece_baseline_vary_k.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════
# 4. Three baselines: vary B (number of bins)
# ═══════════════════════════════════════════════════════════════════════

B_values = list(range(5, 51))

rows_vary_b = []
for B_val in B_values:
    val_analytical = ece_baseline_analytical(N_stored, B=B_val, K=K)

    emp_vals = []
    eqm_vals = []
    for j in range(2):
        conf_j = ref_confidences[j]
        bin_edges_b = np.linspace(0, 1, B_val + 1)
        counts_j, _ = np.histogram(conf_j, bins=bin_edges_b)
        emp_vals.append(ece_baseline_empirical(counts_j, B=B_val, K=K))
        eqm_vals.append(ece_baseline_equal_mass(conf_j, B=B_val))

    rows_vary_b.append({
        "B": B_val,
        "analytical": val_analytical,
        "empirical": np.mean(emp_vals),
        "equal_mass": np.mean(eqm_vals),
    })

df_vary_b = pd.DataFrame(rows_vary_b)
df_vary_b.to_csv(RESULTS_DIR / "ece_baseline_vary_b.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════
# 5. Figures: 3-panel paper figure + standalone vary-B figure
# ═══════════════════════════════════════════════════════════════════════

BASELINE_STYLES = {
    "analytical": ("o-", "0.3", "Analytical"),
    "empirical": ("s--", "C0", "Empirical"),
    "equal_mass": ("^:", "C1", "Equal-mass"),
}

fig, axes = plt.subplots(1, 3, figsize=(11, 3))

# ── Panel 1: Vary N ──────────────────────────────────────────────────
ax = axes[0]
for key, (fmt, color, label) in BASELINE_STYLES.items():
    ax.plot(df_vary_n["N"], df_vary_n[key], fmt, ms=4, label=label, color=color)
ax.axvline(N_stored, color="0.5", ls=":", lw=0.8, zorder=0)
ax.text(
    N_stored, ax.get_ylim()[1] * 0.95, f"  N={N_stored}",
    fontsize=FS_LEGEND, color="0.5", va="top",
)
ax.set_xscale("log")
ax.set_xlabel("N (test cases)", fontsize=FS_LABEL)
ax.set_ylabel("ECE baseline", fontsize=FS_LABEL)
ax.legend(fontsize=FS_LEGEND)
ax.set_title(f"Vary N (B={B}, K={K})", fontsize=FS_LABEL)
despine(ax)

# ── Panel 2: Vary K ─────────────────────────────────────────────────
ax = axes[1]
for key, (fmt, color, label) in BASELINE_STYLES.items():
    ax.plot(df_vary_k["K"], df_vary_k[key], fmt, ms=4, label=label, color=color)
ax.set_xlabel("K (classes)", fontsize=FS_LABEL)
ax.set_ylabel("ECE baseline", fontsize=FS_LABEL)
ax.legend(fontsize=FS_LEGEND)
ax.set_title(f"Vary K (B={B}, N={N_stored})", fontsize=FS_LABEL)
despine(ax)

# ── Panel 3: Across budgets ─────────────────────────────────────────
ax = axes[2]
for key, (fmt, color, label) in BASELINE_STYLES.items():
    ax.plot(df_budget["budget"], df_budget[key], fmt, ms=4, label=label, color=color)
ax.set_xscale("log")
ax.set_xlabel("Training budget", fontsize=FS_LABEL)
ax.set_ylabel("ECE baseline", fontsize=FS_LABEL)
ax.legend(fontsize=FS_LEGEND)
ax.set_title(f"Across budgets (B={B}, K={K})", fontsize=FS_LABEL)
despine(ax)

# ── Save ──────────────────────────────────────────────────────────────
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(ARTIFACTS_DIR / f"ece_baseline_sensitivity.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved 3-panel figure to ece_baseline_sensitivity.pdf/png")

# ── Standalone vary-B figure ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(4, 3))
for key, (fmt, color, label) in BASELINE_STYLES.items():
    ax.plot(df_vary_b["B"], df_vary_b[key], fmt, ms=4, label=label, color=color)
ax.axvline(B, color="0.5", ls=":", lw=0.8, zorder=0)
ax.text(B, ax.get_ylim()[1] * 0.95, f"  B={B}", fontsize=FS_LEGEND, color="0.5", va="top")
ax.set_xlabel("B (ECE bins)", fontsize=FS_LABEL)
ax.set_ylabel("ECE baseline", fontsize=FS_LABEL)
ax.legend(fontsize=FS_LEGEND)
ax.set_title(f"Vary B (K={K}, N={N_stored})", fontsize=FS_LABEL)
despine(ax)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(ARTIFACTS_DIR / f"ece_baseline_vary_b.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved vary-B figure to ece_baseline_vary_b.pdf/png")
