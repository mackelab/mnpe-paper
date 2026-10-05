"""Figure for the scaling study: MNPE accuracy vs number of discrete dimensions.

Three panels over D (one curve per budget, mean over seeds, min-max band):
  (a) per-dim discrete TV vs the exact posterior, with the M-sample MC
      measurement floor E|q_hat - p| = mean sqrt(2 p (1-p) / (pi M)) — measured
      TV cannot fall below this even for a perfect posterior;
  (b) per-dim discrete *marginal* ECE with the analytical finite-sample
      baseline;
  (c) joint C2ST vs exact posterior samples (0.5 = indistinguishable).

All three metrics run the full range to D=128. Only 16 discrete variables can
ever be marginalized exactly (a 32768-state budget buys the first 16 whatever D
is), so beyond D=32 most of the ECE comes from sampled marginals — but at 65536
ancestral samples those are precise enough to report. The evidence is direct: a
16x increase in sample count moved the ECE by at most 1.4e-4 across 28
recomputed rows, against a marginal-vs-factor gap of up to 1.3e-3 and a
smallest floor margin of 2.0e-3, i.e. an order of magnitude finer than the
effects being read off the panel.

``ece_marginal_mc_spread`` looks less reassuring (1e-3 to 2e-3 at high D) and is
recorded per row, but it is a poor precision measure here: it is a maximum over
dimensions of a *binned* statistic, so a test point sitting near a bin edge
flips bins under vanishing noise and moves the ECE discretely. It stops falling
like 1/sqrt(N) well before the estimate itself stops moving.

This figure builds from a tracked CSV rather than through
:mod:`mnpe_paper.figure_io`, because the sweep's merged grid *is* the compact
result — there is no heavier artifact for it to prefer.

Usage:
    uv run python figures/scaling_figure.py [--strict]

Outputs:
    figures/scaling_figure.pdf
    figures/scaling_figure.png
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mnpe_paper.figure_io import save_figure
from mnpe_paper.models.scaling.exact_posterior import p_z_given_x
from mnpe_paper.models.scaling.sweep_common import (
    M_TV_SAMPLES,
    N_TV_TEST,
    check_seed_completeness,
    make_test_set,
)
from mnpe_paper.paths import model_results_dir
from mnpe_paper.plotting.plot_style import COLORS, despine

# The sampling floor below is only the right curve for the sample count and test
# set the sweep actually used, so both come from the sweep's own definition
# rather than being restated here where they could drift out of step with it.
M_SAMPLES = M_TV_SAMPLES
N_TEST = N_TV_TEST

parser = argparse.ArgumentParser()
parser.add_argument("--results", default="scaling_study_results.csv")
parser.add_argument(
    "--strict",
    action="store_true",
    help="fail instead of warn when a (D, budget) cell has fewer than 3 seeds",
)
args = parser.parse_args()


CSV_PATH = model_results_dir("scaling") / args.results

if not CSV_PATH.exists():
    raise SystemExit(
        f"{CSV_PATH} is missing. It is tracked in this repository; if it has been "
        "removed, regenerate it with experiments/scaling/scaling_study.py "
        "(see experiments/scaling/README.md for the grid it was run over)."
    )

df = pd.read_csv(CSV_PATH)
check_seed_completeness(df, "D", label=args.results, strict=args.strict)

budgets = [int(b) for b in sorted(df["budget"].unique())]
dims = [int(d) for d in sorted(df["D"].unique())]
# lowest budget lightest grey -> darker; highest budget = MNPE green
_greys = np.linspace(0.7, 0.4, max(len(budgets) - 1, 1))
budget_colors = {b: f"{_greys[i]:.2f}" for i, b in enumerate(budgets[:-1])}
budget_colors[budgets[-1]] = COLORS["MNPE"]


def tv_mc_floor(D):
    """Expected |q_hat - p| under a perfect posterior, from M binomial samples,
    averaged over the same fixed test set the study uses."""
    _, x = make_test_set(N_TEST, D)
    p = p_z_given_x(x.numpy())
    return np.sqrt(2 * p * (1 - p) / (np.pi * M_SAMPLES)).mean()


floors = {D: tv_mc_floor(D) for D in dims}

fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.1))
panels = [
    ("tv_disc_mean", "discrete TV (per dim)"),
    ("ece_marginal_mean", "discrete ECE (per dim)"),
    ("c2st_joint_mean", "joint C2ST"),
]
for ax, (col, ylabel) in zip(axes, panels):
    shown = dims
    for budget in budgets:
        g = df[df["budget"] == budget].groupby("D")[col]
        label = (f"{budget // 1_000_000}M sims" if budget >= 1_000_000
                 else f"{budget // 1000}k sims")
        ax.plot(g.mean().index, g.mean(), "o-", ms=2.5, lw=1,
                color=budget_colors[budget], label=label)
        ax.fill_between(g.mean().index, g.min(), g.max(),
                        color=budget_colors[budget], alpha=0.2, lw=0)

    ax.set_xlabel("discrete dimensions $D$")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log", base=2)
    ticks = [d for d in shown if d & (d - 1) == 0] if max(shown) > 12 else shown
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)
    ax.minorticks_off()
    if col.startswith("c2st"):
        # full C2ST range: high-D values reach ~0.98, a tighter cap would clip
        ax.set_ylim(0.49, 1.0)
    else:
        ax.set_ylim(bottom=0)
    despine(ax, offset=3)

axes[0].plot(dims, [floors[D] for D in dims], ls="--", color="0.4", lw=0.8,
             label="sampling floor")
ece_baseline = df["ece_baseline"].iloc[0]
axes[1].axhline(ece_baseline, ls="--", color="0.4", lw=0.8, label="calibration floor")
axes[2].axhline(0.5, ls="--", color="0.4", lw=0.8)
axes[0].legend(frameon=False, handlelength=1.4, loc="upper left")

# Report which ECE points are sampled rather than exactly enumerated, and at
# what particle count, so the panel's precision claim is checkable from a run.
if "ece_marginal_exact" in df.columns:
    inexact = df[~df["ece_marginal_exact"].astype(bool)]
    sampled = [int(d) for d in sorted(inexact["D"].unique())]
    if sampled:
        counts = sorted({int(n) for n in inexact["ece_marginal_samples"].dropna()})
        exact_dims = sorted({int(n) for n in inexact["ece_marginal_exact_dims"].dropna()})
        print(f"ECE panel: D={sampled} marginalized by Monte Carlo at {counts} samples")
        print(f"           dims enumerated exactly per row: {exact_dims}")

fig.tight_layout()
save_figure(fig, "scaling_figure")
plt.close(fig)
print(f"Saved scaling_figure.pdf/png ({len(df)} runs, D={dims}, budgets={budgets})")
