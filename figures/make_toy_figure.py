"""Paper figure for the tractable Gaussian benchmark: joint posterior + discrete insets.

Layout (1x4, two panel labels)::

    a (3 subpanels): joint posterior histogram (purple) with GT overlay (black),
      small inset bar charts for the discrete P(theta_d | x_o) (orange)
    b: C2ST convergence

The three observations span three regimes of the discrete posterior — at
x_o = -0.5 the analytic p(z=1|x) is small, at 2.5 it is close to one, and at 1.0
the posterior is genuinely bimodal in theta_c. That last panel is the point of
the example: the continuous marginal has two well-separated modes whose weights
are set by the discrete posterior, so a method that got the discrete part wrong
could not place them correctly.

**On reproducing the published PDF.** The published posterior panels came from
unseeded draws. The tracked export seeds the draws (seed 42, see
:mod:`mnpe_paper.models.toy.figure_inputs`), so it will not match the published
panels pixel for pixel. Panel b is read from the budget study and does match.
Expect the three histograms and the inset bars to differ slightly; expect nothing
else to.

Usage:
    uv run python figures/make_toy_figure.py

Outputs:
    figures/toy_paper_figure.pdf
    figures/toy_paper_figure.png
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from mnpe_paper.figure_io import load_figure_inputs, save_figure
from mnpe_paper.models.toy import figure_inputs
from mnpe_paper.models.toy.config import p_theta_given_x, p_z_given_x
from mnpe_paper.plotting.plot_style import (
    CLR_C1,
    CLR_GT,
    CLR_MU1,
    COLORS,
    FS_LABEL,
    FS_LEGEND,
    FS_PANEL,
    FS_TICK,
    despine,
)

FS_TITLE = 9
THETA_GRID = np.linspace(-3, 3, 300)

inputs = load_figure_inputs(
    figure_inputs.MODEL,
    artifacts=figure_inputs.ARTIFACTS,
    assemble=figure_inputs.assemble,
)

BUDGETS = inputs["budgets"]
N_REPEATS = inputs["n_repeats"]
all_c2st = inputs["all_c2st"]
posterior_samples = inputs["posterior_samples"]
# Back to Python floats: the panel titles interpolate these, and a numpy scalar
# would render differently.
x_obs_values = [float(value) for value in inputs["x_obs_values"]]

print(f"Budgets: {BUDGETS}")
print(f"Reps per budget: {N_REPEATS}")
print(
    f"C2ST at {max(BUDGETS):,}: {np.mean(all_c2st[max(BUDGETS)]):.3f} "
    f"± {np.std(all_c2st[max(BUDGETS)]) / np.sqrt(N_REPEATS):.3f} (SEM)"
)

fig, axes = plt.subplots(1, 4, figsize=(7, 1.5))

# ── Panel a: three posterior subpanels ──
for col, x_val in enumerate(x_obs_values):
    ax = axes[col]

    mnpe_samp = posterior_samples[col]
    theta_m = mnpe_samp[:, 0]
    z_m = mnpe_samp[:, 1].astype(int)

    # Joint histogram of theta_c (purple, the continuous parameter family)
    ax.hist(
        theta_m,
        bins=40,
        density=True,
        alpha=0.5,
        color=CLR_MU1,
        edgecolor="none",
    )

    # GT analytic density
    ax.plot(THETA_GRID, p_theta_given_x(THETA_GRID, x_val), color=CLR_GT, lw=1.5)

    ax.set_xlabel(r"$\theta_c$", fontsize=FS_LABEL)
    ax.set_xlim(-3, 4)
    ax.set_title(f"$x_o = {x_val}$", fontsize=FS_TITLE)
    if col == 0:
        ax.set_ylabel("Density", fontsize=FS_LABEL)

    # ── Discrete inset: P(theta_d | x_o) ──
    inset = ax.inset_axes([0.72, 0.5, 0.27, 0.45])

    gt_probs = [p_z_given_x(x_val, z=0), p_z_given_x(x_val, z=1)]
    mnpe_probs = [(z_m == 0).mean(), (z_m == 1).mean()]

    bar_w = 0.35
    x_pos = np.array([0, 1])
    inset.bar(
        x_pos - bar_w / 2,
        gt_probs,
        width=bar_w,
        fill=False,
        edgecolor=CLR_GT,
        linewidth=1.0,
    )
    inset.bar(
        x_pos + bar_w / 2,
        mnpe_probs,
        width=bar_w,
        color=CLR_C1,
        alpha=0.7,
        edgecolor="none",
    )

    inset.set_xticks([0, 1])
    inset.set_xticklabels(["0", "1"], fontsize=FS_TICK)
    inset.set_xlim(-0.5, 1.5)
    inset.set_ylim(0, 1.0)
    inset.set_yticks([0, 1])
    inset.set_xlabel(r"$\theta_d$", fontsize=FS_TICK, labelpad=1)
    inset.spines["top"].set_visible(False)
    inset.spines["right"].set_visible(False)
    inset.spines["left"].set_bounds(0, 1)
    inset.spines["bottom"].set_bounds(-0.5, 1.5)

# Sync the y-axis across the three posterior panels
ymax = 1
for col in range(3):
    axes[col].set_ylim(0, ymax)
    axes[col].set_yticks([0, ymax])
    despine(axes[col], offset=3)

# ── Panel b: C2ST convergence (mean ± SEM) + global legend ──
ax = axes[3]
means = [np.mean(all_c2st[n]) for n in BUDGETS]
sems = [np.std(all_c2st[n]) / np.sqrt(N_REPEATS) for n in BUDGETS]

ax.errorbar(
    BUDGETS,
    means,
    yerr=sems,
    fmt="o-",
    capsize=3,
    color=COLORS["MNPE"],
    markersize=4,
    lw=1.2,
)
ax.set_xscale("log")
ax.set_xlabel("Simulations", fontsize=FS_LABEL)
ax.set_ylabel("C2ST", fontsize=FS_LABEL)
ax.set_ylim(0.5, 1.0)
ax.set_yticks([0.5, 1.0])
ax.set_xlim(70, 15000)
despine(ax, offset=3)

# Global legend, in the white space of the C2ST panel
legend_handles = [
    Line2D([0], [0], color=CLR_GT, lw=1.5, label=r"$\theta_c$ GT"),
    Patch(fc=CLR_MU1, alpha=0.5, label=r"$cont. \theta_c$"),
    Patch(fc=CLR_C1, alpha=0.7, label=r"$discr. \theta_d$"),
    Line2D([0], [0], color=COLORS["MNPE"], lw=1.5, label=r"MNPE"),
]
ax.legend(
    handles=legend_handles,
    fontsize=FS_LEGEND,
    frameon=False,
    loc="center right",
    bbox_to_anchor=(1.1, 0.7),
    handlelength=1.2,
)

# ── Panel labels ──
# Read the positions before tight_layout: the labels are placed against the
# default subplot grid and the axes then move under them, which is the published
# layout. Reordering these two steps shifts both labels.
pos_a = axes[0].get_position()
pos_b = axes[3].get_position()
fig.text(pos_a.x0 - 0.12, pos_a.y1, "a", fontsize=FS_PANEL, fontweight="bold", va="bottom")
fig.text(pos_b.x0 + 0.01, pos_b.y1, "b", fontsize=FS_PANEL, fontweight="bold", va="bottom")

plt.tight_layout()

written = save_figure(fig, "toy_paper_figure", bbox_inches="tight")
plt.close(fig)
print(f"Saved {written.name} (and .png) to {written.parent}")
