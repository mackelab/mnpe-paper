"""Paper figure for the coal-mining-disaster changepoint model.

Layout::

    ┌──── Left (row 1: 2 panels) ──┐┌────── Right 3x3 pairplot ──────┐
    │ (a) Cumul.  │ (b) C2ST       ││                                │
    │ disasters   │ vs sims        ││  early_rate x late_rate x       │
    │─────────────┼────────┬───────││  switchpoint (year)             │
    │ (d) SBC     │ ECE    │ EoD / ││                                │
    │ eCDF (100k) │ (100k) │ ECE   ││                                │
    └─────────────┴────────┴───────┘└────────────────────────────────┘

Panels:
  (a) cumulative disasters, with the historical switchpoint marked
  (b) C2ST convergence against the MCMC reference, per budget
  (c) 3x3 pairplot, MNPE against MCMC, at the largest budget
  (d) SBC eCDF, ECE reliability, and the EoD/ECE convergence curves

Setup: 111 possible switchpoints (1851-1961), Exponential(1) priors on the rates.

**One example replicate, aggregated numbers.** The single-model visual panels —
the pairplot, the SBC eCDF and the ECE reliability histogram — show one
replicate, chosen as the lowest joint C2ST at the largest budget. Showing a good
run in a figure is fine; quoting its metrics as typical is not, so every *number*
here is aggregated over all replicates: panels (b) and (d3) plot mean ± SEM
across them. The chosen replicate is printed below and belongs in the caption.
The selection rule lives in :func:`mnpe_paper.models.coal.figure_inputs.example_rep_index`,
so the export and this script cannot disagree about which replicate is meant.

The pairplot is sbi's own, not the fork under :mod:`mnpe_paper.plotting`; the
published panel was drawn with sbi's.

Usage:
    uv run python figures/make_coal_figure.py

Outputs:
    figures/coal_paper_figure.pdf
    figures/coal_paper_figure.png
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sbi.analysis import pairplot
from scipy.stats import binom, expon

from mnpe_paper.calibration import (
    ece_baseline_empirical,
    eod_expected_under_uniformity,
    eod_from_ranks,
)
from mnpe_paper.figure_io import load_figure_inputs, save_figure
from mnpe_paper.models.coal import figure_inputs
from mnpe_paper.models.coal.config import (
    DISASTER_DATA,
    EARLY_RATE_TRUE,
    LATE_RATE_TRUE,
    SWITCHPOINT_TRUE_YEAR,
    YEARS,
)
from mnpe_paper.plotting.plot_style import (
    CLR_C1,
    CLR_EARLY,
    CLR_GAMMA,
    CLR_GT,
    CLR_LATE,
    CLR_SWITCH,
    COLORS,
    FS_LABEL,
    FS_LEGEND,
    FS_PANEL,
    FS_TICK,
    despine,
)

# Stronger colours for the pairplot than the shared palette's
CLR_MNPE_STRONG = "#1a8f1a"  # saturated green
CLR_MCMC_STRONG = "#0b5ea2"  # saturated blue

N_SBC_POSTERIOR = 1000
N_CAL_SAMPLES = 250
N_ECE_BINS = 10
SPINE_OFFSET = 2

PARAM_NAMES_DISPLAY = [
    r"$\lambda_{\mathrm{early}}$",
    r"$\lambda_{\mathrm{late}}$",
    r"$s$",
]


def sbc_rank_ecdf(
    ax,
    ranks,
    num_bins=None,
    colors=None,
    labels=None,
    alpha=0.8,
    uniform_alpha=0.2,
):
    """SBC rank eCDF with a 95% uniform confidence band."""
    num_sbc_runs, num_params = ranks.shape
    if num_bins is None:
        num_bins = max(num_sbc_runs // 4, 10)
    num_repeats = 50

    if colors is None:
        colors = [f"C{i}" for i in range(num_params)]
    if labels is None:
        labels = [f"dim {i}" for i in range(num_params)]

    uni_bins = binom.ppf(0.5, num_sbc_runs, 1 / num_bins) * np.ones(num_bins)
    uni_bins_cdf = uni_bins.cumsum() / uni_bins.sum()
    uni_bins_cdf[-1] -= 1e-9
    lower = np.array([binom.ppf(0.025, num_sbc_runs, p) for p in uni_bins_cdf])
    upper = np.array([binom.ppf(0.975, num_sbc_runs, p) for p in uni_bins_cdf])

    ax.fill_between(
        np.linspace(0, num_bins, num_repeats * num_bins),
        np.repeat(lower / lower.max(), num_repeats),
        np.repeat(upper / upper.max(), num_repeats),
        color="grey",
        alpha=uniform_alpha,
        label="uniform",
    )

    for i in range(num_params):
        hist, _ = np.histogram(ranks[:, i], bins=num_bins)
        histcs = hist.cumsum()
        ax.plot(
            np.linspace(0, num_bins, num_repeats * num_bins),
            np.repeat(histcs / histcs.max(), num_repeats),
            lw=1.2,
            color=colors[i],
            label=labels[i],
            alpha=alpha,
        )

    ax.set_xlabel("Posterior rank", fontsize=FS_LABEL)
    ax.set_ylabel("eCDF", fontsize=FS_LABEL)
    ax.set_xlim(0, num_bins)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, num_bins // 2, num_bins])
    ax.set_yticks([0, 0.5, 1])
    despine(ax)


# ═══════════════════════════════════════════════════════════════════════════
# Load
# ═══════════════════════════════════════════════════════════════════════════
inputs = load_figure_inputs(
    figure_inputs.MODEL,
    artifacts=figure_inputs.ARTIFACTS,
    assemble=figure_inputs.assemble,
)

# The budget cap the figure stops at is applied when the inputs are assembled,
# on both the artifact and the export path.
BUDGETS = inputs["budgets"]
N_REPS = inputs["n_reps"]
x_o_np = inputs["x_o"]
ref_samples = inputs["ref_samples"]
ref_pred_obs = inputs["ref_pred_obs"]
prior_pred_obs = inputs["prior_pred_obs"]
budget_results = inputs["budget_results"]
DISCRETE_VALUES = inputs["discrete_values"]

largest_budget = max(BUDGETS)
reps_at_largest = budget_results[largest_budget]["reps"]
example_idx = figure_inputs.example_rep_index(reps_at_largest)
res_example = reps_at_largest[example_idx]
mnpe_samples = res_example["samples"]
# pred_obs is only saved for rep 0 — always take it from there.
mnpe_pred_obs = reps_at_largest[figure_inputs.PRED_OBS_REP]["pred_obs"]

# The reference draws arrive already shuffled and sliced (see
# figure_export.subsample_reference); shuffling again here would reorder them
# away from the predictive draws computed alongside them.

TRUE_PARAMS = [EARLY_RATE_TRUE, LATE_RATE_TRUE, float(SWITCHPOINT_TRUE_YEAR)]
budgets_arr = np.array(BUDGETS)
n_params = 3

print(f"Budgets: {BUDGETS}")
print(f"Reps per budget: {N_REPS}")
_c2st_reps = [r["c2st_all"] for r in reps_at_largest]
print(f"Largest budget: {largest_budget:,}")
print(f"  joint C2ST per rep: {[round(v, 4) for v in _c2st_reps]}")
print(f"  mean {np.mean(_c2st_reps):.4f} +/- {np.std(_c2st_reps):.4f} (sd over reps)")
print(f"  VISUAL panels show rep {example_idx} (C2ST {_c2st_reps[example_idx]:.4f}) "
      f"-- an example, not a typical run; state this in the caption")
print(f"MNPE samples: {mnpe_samples.shape}")
print(f"Reference samples: {ref_samples.shape}")
print(f"Prior predictive: {prior_pred_obs.shape}")
print(f"x_o: {x_o_np.shape}")

ranks_example = np.asarray(res_example["sbc_ranks"])
K = len(DISCRETE_VALUES)  # 111 switchpoints

# Pairplot data
N_PP = min(len(mnpe_samples), len(ref_samples))
mnpe_pp = mnpe_samples[:N_PP]
ref_pp = ref_samples[:N_PP]

rate_xlim = (0, expon.ppf(0.995, scale=1 / 1.0))  # Exponential(1) prior
pp_limits = [list(rate_xlim), list(rate_xlim), [float(YEARS[0]), float(YEARS[-1])]]

# ── Aggregate across reps (mean ± SEM) ──
c2st_joint_mean, c2st_joint_sem = [], []
c2st_early_mean, c2st_early_sem = [], []
c2st_late_mean, c2st_late_sem = [], []
avg_eod_mean, avg_eod_sem = [], []
avg_discrete_ece_mean, avg_discrete_ece_sem = [], []
avg_ece_baseline_mean = []
bin_edges_bl = np.linspace(0, 1, N_ECE_BINS + 1)

for b in BUDGETS:
    reps = budget_results[b]["reps"]
    n = len(reps)

    joint = [r["c2st_all"] for r in reps]
    early = [r["c2st_marginal"]["early_rate"] for r in reps]
    late = [r["c2st_marginal"]["late_rate"] for r in reps]
    c2st_joint_mean.append(np.mean(joint))
    c2st_joint_sem.append(np.std(joint) / np.sqrt(n))
    c2st_early_mean.append(np.mean(early))
    c2st_early_sem.append(np.std(early) / np.sqrt(n))
    c2st_late_mean.append(np.mean(late))
    c2st_late_sem.append(np.std(late) / np.sqrt(n))

    # EoD (continuous calibration)
    eod_vals = [
        eod_from_ranks(
            np.asarray(r["sbc_ranks"])[:N_CAL_SAMPLES, :2], N_SBC_POSTERIOR
        ).mean()
        for r in reps
    ]
    avg_eod_mean.append(np.mean(eod_vals))
    avg_eod_sem.append(np.std(eod_vals) / np.sqrt(n))

    # Discrete ECE
    ece_vals = [np.mean(r["ece"][:N_CAL_SAMPLES]) for r in reps]
    avg_discrete_ece_mean.append(np.mean(ece_vals))
    avg_discrete_ece_sem.append(np.std(ece_vals) / np.sqrt(n))

    # Per-rep empirical ECE baseline (one discrete parameter: the switchpoint)
    bl_vals = []
    for r in reps:
        top_conf = r["ece_probs"][:N_CAL_SAMPLES, 0, :].max(axis=1)
        counts_j, _ = np.histogram(top_conf, bins=bin_edges_bl)
        bl_vals.append(ece_baseline_empirical(counts_j, B=N_ECE_BINS, K=K))
    avg_ece_baseline_mean.append(np.mean(bl_vals))

c2st_joint_mean = np.array(c2st_joint_mean)
c2st_joint_sem = np.array(c2st_joint_sem)
c2st_early_mean = np.array(c2st_early_mean)
c2st_early_sem = np.array(c2st_early_sem)
c2st_late_mean = np.array(c2st_late_mean)
c2st_late_sem = np.array(c2st_late_sem)
avg_eod_mean = np.array(avg_eod_mean)
avg_eod_sem = np.array(avg_eod_sem)
avg_discrete_ece_mean = np.array(avg_discrete_ece_mean)
avg_discrete_ece_sem = np.array(avg_discrete_ece_sem)
avg_ece_baseline_mean = np.array(avg_ece_baseline_mean)

eod_expected = eod_expected_under_uniformity(num_sbc_runs=N_CAL_SAMPLES)
print(f"EoD baseline (expected under uniformity): {eod_expected:.4f}")
print(f"Discrete ECE baseline (empirical, best budget): {avg_ece_baseline_mean[-1]:.4f}")

# ── Cumulative predictive curves for panel (a) ──
prior_cum = np.cumsum(prior_pred_obs, axis=1)
ref_cum = np.cumsum(ref_pred_obs, axis=1)
mnpe_cum = np.cumsum(mnpe_pred_obs, axis=1)
obs_cum = np.cumsum(DISASTER_DATA)

# ═══════════════════════════════════════════════════════════════════════════
# Figure
# ═══════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(7, 2.75))
gs_top = GridSpec(1, 2, figure=fig, width_ratios=[5, 3], wspace=0.1)

# Left: 2 rows — row 1 has 2 panels, row 2 has 3 panels
gs_left = gs_top[0, 0].subgridspec(2, 1, hspace=0.75)
gs_row1 = gs_left[0, 0].subgridspec(1, 2, wspace=0.6)
gs_row2 = gs_left[1, 0].subgridspec(1, 3, wspace=0.7)

# Right: 3×3 pairplot
gs_right = gs_top[0, 1].subgridspec(3, 3, hspace=0.1, wspace=0.1)

# ── Panel (a): posterior predictive, cumulative disaster curves ──
ax_a = fig.add_subplot(gs_row1[0, 0])

ax_a.fill_between(
    YEARS,
    np.percentile(prior_cum, 5, axis=0),
    np.percentile(prior_cum, 95, axis=0),
    color="grey",
    alpha=0.2,
)
ax_a.fill_between(
    YEARS,
    np.percentile(ref_cum, 5, axis=0),
    np.percentile(ref_cum, 95, axis=0),
    color=COLORS["Marginalized"],
    alpha=0.3,
)
ax_a.fill_between(
    YEARS,
    np.percentile(mnpe_cum, 5, axis=0),
    np.percentile(mnpe_cum, 95, axis=0),
    color=COLORS["MNPE"],
    alpha=0.3,
)

ax_a.plot(YEARS, obs_cum, color=CLR_GT, lw=1.5)
ax_a.axvline(SWITCHPOINT_TRUE_YEAR, color=CLR_GT, ls="--", lw=1.0, alpha=0.7)

ax_a.set_xlabel("Year", fontsize=FS_LABEL)
ax_a.set_ylabel("Cumulative disasters", fontsize=FS_LABEL)
ax_a.tick_params(labelsize=FS_TICK)
ax_a.set_xlim(1851, 1962)
ax_a.set_xticks([1851, 1905, 1962])
ax_a.set_yticks([0, 300])
ax_a.set_ylim(0, 300)
despine(ax_a, offset=SPINE_OFFSET)

# ── Panel (b): C2ST vs budget (mean ± SEM) ──
ax_b = fig.add_subplot(gs_row1[0, 1])

ax_b.errorbar(
    budgets_arr,
    c2st_joint_mean,
    yerr=c2st_joint_sem,
    fmt="o-",
    ms=4,
    color="gray",
    capsize=2,
    capthick=0.8,
    lw=1.2,
    label="Joint",
)
for pname, mean, sem, clr, marker in [
    (PARAM_NAMES_DISPLAY[0], c2st_early_mean, c2st_early_sem, CLR_EARLY, "s"),
    (PARAM_NAMES_DISPLAY[1], c2st_late_mean, c2st_late_sem, CLR_LATE, "^"),
]:
    ax_b.errorbar(
        budgets_arr,
        mean,
        yerr=sem,
        fmt=f"{marker}--",
        ms=3,
        color=clr,
        capsize=2,
        capthick=0.8,
        lw=1.0,
        label=pname,
    )

ax_b.set_xscale("log")
ax_b.set_xlabel("Simulations", fontsize=FS_LABEL)
ax_b.set_ylabel("C2ST (vs. MCMC)", fontsize=FS_LABEL)
ax_b.set_xlim(1e3 - 100, 1e5 + 2000)
ax_b.set_ylim(0.5, 1.0)
ax_b.set_yticks([0.5, 1.0])
ax_b.tick_params(labelsize=FS_TICK)
despine(ax_b, offset=SPINE_OFFSET)

# ── Panel (d1): SBC eCDF, example rep of the largest budget ──
ax_sbc = fig.add_subplot(gs_row2[0, 0])
sbc_rank_ecdf(
    ax_sbc,
    ranks_example[:, :2],  # 2 continuous params
    colors=[CLR_EARLY, CLR_LATE],
    labels=[PARAM_NAMES_DISPLAY[0], PARAM_NAMES_DISPLAY[1]],
)
ax_sbc.tick_params(labelsize=FS_TICK)

# ── Panel (d2): ECE reliability, example rep of the largest budget ──
ax_ece = fig.add_subplot(gs_row2[0, 1])
# Single discrete parameter (the switchpoint) — bin from the stored probabilities
probs_d = res_example["ece_probs"][:, 0]  # (N, K)
true_d = res_example["ece_true_vals"][:, 0]  # (N,)
confidence = probs_d.max(axis=-1)
predicted = probs_d.argmax(axis=-1)
correct = (predicted == true_d).astype(float)
ece_val = res_example["ece"][0]
bin_edges = np.linspace(0, 1, N_ECE_BINS + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_idx = np.clip(np.digitize(confidence, bin_edges[:-1]) - 1, 0, N_ECE_BINS - 1)
acc = np.array([
    correct[bin_idx == b].mean() if (bin_idx == b).any() else np.nan
    for b in range(N_ECE_BINS)
])
mask = ~np.isnan(acc)
ax_ece.bar(
    bin_centers[mask],
    acc[mask],
    width=0.07,
    alpha=0.7,
    color=CLR_SWITCH,
    edgecolor="white",
    lw=0.5,
    label=f"Switchpoint\n(ECE={ece_val:.3f})",
)
ax_ece.plot([0, 1], [0, 1], "--", color="gray", lw=0.8, alpha=0.7)
ax_ece.set_xlabel("Confidence", fontsize=FS_LABEL)
ax_ece.set_ylabel("Accuracy", fontsize=FS_LABEL)
ax_ece.set_xlim(0, 1)
ax_ece.set_ylim(0, 1)
ax_ece.set_xticks([0, 1])
ax_ece.set_yticks([0, 1])
ax_ece.tick_params(labelsize=FS_TICK)
despine(ax_ece)

# ── Panel (d3): EoD / ECE convergence vs budget ──
ax_conv = fig.add_subplot(gs_row2[0, 2])
plot_ymax = 0.05

ax_conv.errorbar(
    budgets_arr,
    avg_eod_mean,
    yerr=avg_eod_sem,
    fmt="-",
    ms=4,
    color=CLR_EARLY,
    markerfacecolor=CLR_EARLY,
    markeredgecolor="white",
    markeredgewidth=0.5,
    capsize=2,
    capthick=0.8,
)
ax_conv.axhline(eod_expected, color=CLR_EARLY, ls="--", lw=1.0, alpha=0.7)

ax_conv.errorbar(
    budgets_arr,
    avg_discrete_ece_mean,
    yerr=avg_discrete_ece_sem,
    ms=4,
    color=CLR_SWITCH,
    markerfacecolor=CLR_SWITCH,
    markeredgecolor="white",
    markeredgewidth=0.5,
    alpha=0.7,
    capsize=2,
    capthick=0.8,
    lw=1.0,
)
# The empirical ECE baseline moves with the budget, so it is a curve rather than
# a level: judging an observed ECE against a single number would be wrong here.
ax_conv.plot(
    budgets_arr,
    avg_ece_baseline_mean,
    ls="--",
    lw=1.0,
    color=CLR_SWITCH,
    alpha=0.7,
)

ax_conv.set_xscale("log")
ax_conv.set_xlabel("Simulations", fontsize=FS_LABEL)
ax_conv.set_ylabel("Avg. EoD / ECE", fontsize=FS_LABEL)
ax_conv.set_xlim(1e3 - 100, 1e5 + 10000)
ax_conv.set_ylim(0, plot_ymax)
ax_conv.set_yticks([0.0, plot_ymax], [0.0, plot_ymax])
despine(ax_conv, offset=SPINE_OFFSET)

# ── Right: 3×3 pairplot ──
pair_axes = np.array(
    [[fig.add_subplot(gs_right[r, c]) for c in range(n_params)] for r in range(n_params)]
)

pairplot(
    samples=[
        torch.tensor(ref_pp, dtype=torch.float32),
        torch.tensor(mnpe_pp, dtype=torch.float32),
    ],
    points=torch.tensor([TRUE_PARAMS]),
    limits=pp_limits,
    labels=PARAM_NAMES_DISPLAY,
    upper="scatter",
    lower=None,
    diag="hist",
    fig=fig,
    axes=pair_axes,
    fig_kwargs=dict(
        samples_colors=[CLR_MCMC_STRONG, CLR_MNPE_STRONG],
        points_colors=[CLR_GT],
        points_labels=[r"$\theta_o$"],
        points_offdiag={"markersize": 3},
        despine={"offset": 5},
        fig_subplots_adjust={},
    ),
    diag_kwargs=[
        dict(mpl_kwargs=dict(alpha=0.8, bins=50, label="MCMC")),
        dict(mpl_kwargs=dict(alpha=0.8, bins=50, label="MNPE")),
    ],
    upper_kwargs=[
        dict(mpl_kwargs=dict(s=4, alpha=0.8, rasterized=True)),
        dict(mpl_kwargs=dict(s=4, alpha=0.8, rasterized=True)),
    ],
)

# Relocate the pairplot legend from the diagonal to the empty lower-left cell
legend_handles = [
    Patch(facecolor="grey", alpha=0.3, label="Prior"),
    Patch(facecolor=COLORS["MNPE"], alpha=0.7, label="MNPE"),
    Patch(facecolor=COLORS["Marginalized"], alpha=0.7, label="MCMC"),
    Line2D([], [], color=CLR_GAMMA, lw=1.5, label=r"Cont. ($\lambda_i$)"),
    Line2D([], [], color=CLR_C1, lw=1.5, label=r"Discr. ($s$)"),
    Line2D([], [], color="gray", ls="--", lw=1.0, label="Baseline"),
]

pair_axes[2, 0].legend(
    handles=legend_handles,
    loc="center",
    fontsize=FS_LEGEND,
    frameon=False,
    handlelength=1.2,
    handletextpad=0.5,
    labelspacing=0.4,
    columnspacing=0.8,
    bbox_to_anchor=(0.5, 0.5),
)

# ── Panel labels ──
fig.text(0.05, 0.95, "a", fontsize=FS_PANEL, fontweight="bold")
fig.text(0.32, 0.95, "b", fontsize=FS_PANEL, fontweight="bold")
fig.text(0.6, 0.95, "c", fontsize=FS_PANEL, fontweight="bold")
fig.text(0.05, 0.45, "d", fontsize=FS_PANEL, fontweight="bold")

written = save_figure(fig, "coal_paper_figure", bbox_inches="tight")
plt.close(fig)
print(f"Saved {written.name} (and .png) to {written.parent}")
