"""Figure for the K-sweep: MNPE accuracy vs number of classes per discrete dim.

Three panels over K (one curve per budget, mean over 3 seeds, min-max band):
  (a) per-dim discrete TV vs the exact posterior (exact marginalization, no
      sampling floor);
  (b) per-dim discrete *marginal* ECE with the analytical finite-sample
      baseline (K-aware);
  (c) joint C2ST vs exact posterior samples (0.5 = indistinguishable).

Every K-sweep point enumerates its prefixes exactly, so unlike the D-sweep no
panel here carries Monte Carlo marginalization noise.

The main curve is the default width of 32. The width probe behind the paper's
capacity claim is a separate study under ``results/scaling/`` and is filtered
out here, so this figure isolates the class count from capacity exactly as its
caption states.

Usage:
    uv run python figures/k_sweep_figure.py [--strict]

Outputs:
    figures/k_sweep_figure.pdf
    figures/k_sweep_figure.png
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mnpe_paper.figure_io import save_figure
from mnpe_paper.models.scaling.sweep_common import check_seed_completeness
from mnpe_paper.paths import model_results_dir
from mnpe_paper.plotting.plot_style import COLORS, despine

DEFAULT_WIDTH = 32  # the capacity probe's wider networks are a separate study

parser = argparse.ArgumentParser()
parser.add_argument("--results", default="k_sweep_results.csv")
parser.add_argument(
    "--strict",
    action="store_true",
    help="fail instead of warn when a (K, budget) cell has fewer than 3 seeds",
)
args = parser.parse_args()

CSV_PATH = model_results_dir("scaling") / args.results

if not CSV_PATH.exists():
    raise SystemExit(
        f"{CSV_PATH} is missing. It is tracked in this repository; if it has been "
        "removed, regenerate it with experiments/scaling/k_sweep.py "
        "(see experiments/scaling/README.md for the grid it was run over)."
    )

df = pd.read_csv(CSV_PATH)
if "hidden_features" in df.columns:
    df = df[df["hidden_features"] == DEFAULT_WIDTH]
check_seed_completeness(df, "K", label=args.results, strict=args.strict)
budgets = [int(b) for b in sorted(df["budget"].unique())]
ks = [int(k) for k in sorted(df["K"].unique())]
# lowest budget lightest grey -> darker; highest budget = MNPE green
_greys = np.linspace(0.7, 0.4, max(len(budgets) - 1, 1))
budget_colors = {b: f"{_greys[i]:.2f}" for i, b in enumerate(budgets[:-1])}
budget_colors[budgets[-1]] = COLORS["MNPE"]

fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.1))
panels = [
    ("tv_disc_mean", "discrete TV (per dim)"),
    ("ece_marginal_mean", "discrete ECE (per dim)"),
    ("c2st_joint_mean", "joint C2ST"),
]
for ax, (col, ylabel) in zip(axes, panels):
    for budget in budgets:
        g = df[df["budget"] == budget].groupby("K")[col]
        label = (f"{budget // 1_000_000}M sims" if budget >= 1_000_000
                 else f"{budget // 1000}k sims")
        ax.plot(g.mean().index, g.mean(), "o-", ms=2.5, lw=1,
                color=budget_colors[budget], label=label)
        ax.fill_between(g.mean().index, g.min(), g.max(),
                        color=budget_colors[budget], alpha=0.2, lw=0)
    ax.set_xlabel("classes per discrete dim $K$")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels(ks)
    ax.minorticks_off()
    ax.set_ylim(bottom=0.49 if col.startswith("c2st") else 0)
    despine(ax, offset=3)

# K-aware ECE floor (varies only slightly with K) and C2ST chance line
ece_floor = df.groupby("K")["ece_baseline"].first()
axes[1].plot(ece_floor.index, ece_floor.values, ls="--", color="0.4", lw=0.8,
             label="calibration floor")
axes[2].axhline(0.5, ls="--", color="0.4", lw=0.8)
axes[0].legend(frameon=False, handlelength=1.4, loc="upper left")

fig.tight_layout()
save_figure(fig, "k_sweep_figure")
plt.close(fig)
print(f"Saved k_sweep_figure.pdf/png ({len(df)} runs, K={ks})")
