"""Paper figure for the tandem M/M/c queueing model, the main experiment.

Layout::

    ┌──────── Left (2x3) ────────┐┌──── Right 5x5 pairplot ────┐
    │ (a1) pred.  │ (a2) │ (b)   ││                            │
    │ n_arrivals  │ q1   │ C2ST  ││  gamma x mu1 x mu2 x        │
    │─────────────┼──────┼───────││  c1 x c2, MNPE vs reference │
    │ (d1) SBC    │ (d2) │ (d3)  ││                            │
    │ rank eCDF   │ ECE  │ EoD / ││                            │
    │             │ rel. │ ECE   ││                            │
    └─────────────┴──────┴───────┘└────────────────────────────┘

Panels:
  (a) posterior predictive against prior predictive, at the observation
  (b) joint and per-parameter C2ST against the reference, versus budget
  (c) 5x5 pairplot, MNPE against the reference, at the display budget
  (d) SBC rank eCDF, ECE reliability, and the EoD/ECE convergence curves

Setup: c in {2, 3, 4, 5, 6}, sigma_prior = 0.3, sigma_obs = 0.1, T = 100, with
the joint prior conditioned on rho_1, rho_2 < 0.999. Architecture is the Optuna
winner — 4 transforms, 256 hidden units, 4 layers, 9 spline bins, lr 1.1e-4 —
with log1p preprocessing on the two queue-length observations.

Four choices in this figure are deliberate.

1. **The level in panel (b) is the measured split floor**, about 0.502, read
   from ``c2st_floor.csv``: disjoint 10,000-draw splits of the reference against
   itself under the same C2ST protocol. It is not the disagreement between two
   approximate references sharing one evidence approximation, which would
   measure reference error rather than sampling noise. C2ST is never
   floor-subtracted or normalised.
2. **Every single-model panel shows replicate 0 at the 100k budget.** Selecting
   the lowest-C2ST replicate would condition on the same reference the figure
   scores against, and replicate 0 is already the one whose posterior predictive
   panel (a) shows, so one model appears throughout. The budget is 100k rather
   than the largest available because that is the model the text describes; the
   1M runs have three replicates and no stored posterior predictive.
3. **The ECE reference in (d3) is a simulated null band**, not an analytical
   baseline: labels are resampled from each model's own PMF with the confidences
   held fixed, so the model is calibrated by construction and the spread is the
   actual finite-sample floor. It is the null for an *individual* replicate,
   pooled across the five — they share one SBC test set, so their label noise is
   common rather than independent, and a replicate-averaged null would understate
   the floor by roughly sqrt(n_reps).
4. **Error bars are the standard deviation across replicates, not the SEM.**
   Every reference level drawn here describes where a *single* run should land,
   so the observed spread is reported on the same footing.

Two numbers are computed and printed but deliberately not drawn: the reference
gap (SMC against quadrature, about 0.515) and the split floor's own spread. On a
linear (0.5, 1.0) axis they sit a few pixels apart and both hug the bottom spine,
so the panel carries the convergence shape and the text carries the comparison.

The 1M evaluation is not drawn: extending the budget axis by a decade for one
experiment would put this figure on a different scale from every other one. The
1M result is quoted in the text; when the 1M run is present, the export carries its
metrics in ``one_m``.

Usage:
    uv run python figures/make_queue_figure.py

Outputs:
    figures/queue_paper_figure.pdf
    figures/queue_paper_figure.png
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sbi.analysis import pairplot
from scipy.stats import binom

from mnpe_paper.calibration import eod_expected_under_uniformity, eod_from_ranks
from mnpe_paper.figure_io import load_figure_inputs, save_figure
from mnpe_paper.models.queue import figure_inputs
from mnpe_paper.models.queue.spec import MODEL_VERSION
from mnpe_paper.paths import model_results_dir
from mnpe_paper.plotting.plot_style import (
    CLR_C1,
    CLR_C2,
    CLR_GAMMA,
    CLR_GT,
    CLR_MU1,
    CLR_MU2,
    COLORS,
    FS_LABEL,
    FS_LEGEND,
    FS_PANEL,
    FS_TICK,
    despine,
)

# ── Load ────────────────────────────────────────────────────────────────────
inputs = load_figure_inputs(
    "queue",
    artifacts=figure_inputs.ARTIFACTS,
    assemble=figure_inputs.assemble,
)

# The filenames carry no version tag, so a stale artifact cannot be spotted by
# name — regenerating one reuses the name. The version travels inside the
# payload instead, and this is the assertion that catches it.
assert inputs["model_version"] == MODEL_VERSION, "stale budget study artifact"

BUDGETS = list(inputs["budgets"])
N_REPS = inputs.get("n_reps", 1)
x_o_np = inputs["x_o"]
ref_samples = inputs["ref_samples"]
ref_pred_obs = inputs["ref_pred_obs"]
prior_pred_obs = inputs["prior_pred_obs"]
budget_results = inputs["budget_results"]
C_VALUES = inputs.get("discrete_values", [2, 3, 4, 5, 6])

FIG_REP = figure_inputs.FIGURE_REP
FIG_BUDGET = figure_inputs.FIGURE_BUDGET
res_fig = budget_results[FIG_BUDGET]["reps"][FIG_REP]
mnpe_samples = res_fig["samples"]
mnpe_pred_obs = res_fig["pred_obs"]

# The reference draws arrive already shuffled -- both load paths shuffle with
# figure_inputs.REFERENCE_SHUFFLE_SEED before subsampling. Shuffling again here
# would reorder them a second time and, on the export path, permute a set that
# was already reduced. The shuffle matters because the reference .npz is sorted
# by (c1, c2), so slicing an unshuffled array yields only the dominant
# configuration and misses every other mode.

# ── Constants ───────────────────────────────────────────────────────────────
TRUE_PARAMS = [9.0, 8.0, 5.0, 2, 3]
PARAM_NAMES = [r"$\gamma$", r"$\mu_1$", r"$\mu_2$", r"$c_1$", r"$c_2$"]
OBS_NAMES = [
    r"$n_{\mathrm{arr}}$",
    r"$n_{\mathrm{comp},1}$",
    r"$n_{\mathrm{comp},2}$",
    r"$q_1$",
    r"$q_2$",
]
budgets_arr = np.array(BUDGETS)
n_params = 5

RESULTS = model_results_dir("queue")

# C2ST floor: disjoint splits of the reference against itself, same protocol as
# the MNPE rows. This is sampling noise, not reference disagreement.
_floor = pd.read_csv(RESULTS / "c2st_floor.csv").iloc[0]
c2st_floor_mean = float(_floor["mean"])
c2st_floor_sd = float(_floor["std"])

# Reported, not drawn: on a linear (0.5, 1.0) axis this level and the split
# floor are a few pixels apart and both hug the bottom spine, so panel (b) shows
# the convergence shape and the numbers below carry the comparison.
#
# Second yardstick: how far apart two legitimate methods land on the *same*
# posterior -- exact marginalization plus SMC against deterministic quadrature.
# This is not circular. MNPE is scored against the quadrature reference; SMC is
# an independent algorithm targeting the same thing, so its disagreement with
# quadrature bounds how much of an elevated C2ST is attributable to
# reference-quality methods differing rather than to MNPE error.
_yard = pd.read_csv(RESULTS / "reference_yardstick_smc.csv")
ref_gap_c2st = float(
    _yard.loc[_yard["source"].str.contains("reference gap"), "joint_c2st"].iloc[0]
)

# Dequantized NPE at the same 100k budget, the ablation for the discrete head.
# Its C2ST uses the same reference and sample count but an unseeded reference
# subsample and no explicit C2ST seed, so it is comparable to within C2ST noise
# (about 0.01) rather than exactly matched to the MNPE rows.
_deq = pd.read_csv(RESULTS / "dequantization_baseline_results.csv")
DEQ_BUDGET = 100_000
deq_c2st_mean = float(_deq["c2st_all"].mean())
deq_c2st_sd = float(_deq["c2st_all"].std(ddof=0))

c2st_at_fig = [r["c2st_all"] for r in budget_results[FIG_BUDGET]["reps"]]

print(f"Model version: {inputs['model_version']}")
# The reference name recorded when the study ran. It carries a version tag that
# the reference file written by this repository does not, so it is labelled
# rather than printed as if it were a path a reader could open. Nothing loads
# from it.
print(f"Reference (name recorded at run time): {inputs['reference_file']}")
print(f"Budgets: {BUDGETS}")
print(f"Reps per budget: {N_REPS}")
print(f"C values: {C_VALUES}")
print(f"Single-model panels: budget {FIG_BUDGET:,}, rep {FIG_REP}")
print(f"MNPE samples: {mnpe_samples.shape}")
print(f"Reference samples: {ref_samples.shape}")
print(f"Prior predictive: {prior_pred_obs.shape}")
print(f"x_o: {x_o_np}")
print(f"Observation transform: {inputs.get('observation_transform', 'none')}")
print(f"Reference gap (SMC vs quadrature): {ref_gap_c2st:.4f}")
print(
    f"Dequantized NPE at {DEQ_BUDGET:,}: "
    f"{deq_c2st_mean:.4f}, sd {deq_c2st_sd:.4f} ({len(_deq)} seeds)"
)
print(
    f"\nC2ST split floor: {c2st_floor_mean:.4f}, sd {c2st_floor_sd:.4f} "
    f"({int(_floor['n_splits'])} splits of {int(_floor['n_samples']):,})"
)
print(
    f"MNPE joint C2ST at {FIG_BUDGET:,}: "
    f"{np.mean(c2st_at_fig):.4f}, sd {np.std(c2st_at_fig):.4f} "
    f"(rep {FIG_REP}: {c2st_at_fig[FIG_REP]:.4f})"
)


def sbc_rank_ecdf(
    ax,
    ranks,
    num_bins=None,
    colors=None,
    alpha=0.8,
    uniform_alpha=0.2,
    uniform_color="0.45",
):
    """SBC rank eCDF with 95% uniform confidence band."""
    num_sbc_runs, num_params = ranks.shape
    if num_bins is None:
        num_bins = max(num_sbc_runs // 4, 10)
    num_repeats = 50

    if colors is None:
        colors = [f"C{i}" for i in range(num_params)]

    uni_bins = binom.ppf(0.5, num_sbc_runs, 1 / num_bins) * np.ones(num_bins)
    uni_bins_cdf = uni_bins.cumsum() / uni_bins.sum()
    uni_bins_cdf[-1] -= 1e-9
    lower = np.array([binom.ppf(0.025, num_sbc_runs, p) for p in uni_bins_cdf])
    upper = np.array([binom.ppf(0.975, num_sbc_runs, p) for p in uni_bins_cdf])

    ax.fill_between(
        np.linspace(0, num_bins, num_repeats * num_bins),
        np.repeat(lower / lower.max(), num_repeats),
        np.repeat(upper / upper.max(), num_repeats),
        color=uniform_color,
        alpha=uniform_alpha,
    )

    for i in range(num_params):
        hist, _ = np.histogram(ranks[:, i], bins=num_bins)
        histcs = hist.cumsum()
        ax.plot(
            np.linspace(0, num_bins, num_repeats * num_bins),
            np.repeat(histcs / histcs.max(), num_repeats),
            lw=1.2,
            color=colors[i],
            alpha=alpha,
        )

    ax.set_xlabel("Rank", fontsize=FS_LABEL)
    ax.set_ylabel("eCDF", fontsize=FS_LABEL)
    ax.set_xlim(0, num_bins)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, num_bins])
    ax.set_xticklabels(["0", "1"])
    ax.set_yticks([0, 1])
    despine(ax)


def simulate_ece_null(probs, num_bins=10, n_replicates=200, seed=0):
    """ECE draws under perfect calibration, conditional on the confidences.

    Labels are resampled from the model's own predicted PMF, so the model is
    calibrated by construction and the spread of the recomputed ECE is the
    finite-sample noise floor for exactly these confidences. Mirrors
    ``mnpe_paper.models.queue.metrics.ece_null_interval`` and matches the top-label,
    10-equal-width-bin definition used by ``run_discrete_calibration``.

    Returns an array of length ``n_replicates`` holding the ECE averaged over
    the two discrete parameters, matching how the observed ECE is aggregated.
    """
    probs = np.asarray(probs, dtype=float)
    rng = np.random.default_rng(seed)
    n_cases, n_vars, _ = probs.shape
    edges = np.linspace(0, 1, num_bins + 1)

    cdf = np.cumsum(probs, axis=-1)
    u = rng.random((n_replicates, n_cases, n_vars, 1))
    labels = (u > cdf[None, ...]).sum(axis=-1)

    ece = np.zeros((n_replicates, n_vars))
    for j in range(n_vars):
        conf = probs[:, j, :].max(axis=1)
        pred = probs[:, j, :].argmax(axis=1)
        bin_idx = np.clip(np.digitize(conf, edges[1:-1]), 0, num_bins - 1)
        correct = (labels[:, :, j] == pred[None, :]).astype(float)
        for b in range(num_bins):
            mask = bin_idx == b
            if not mask.any():
                continue
            ece[:, j] += (mask.sum() / n_cases) * np.abs(
                conf[mask].mean() - correct[:, mask].mean(axis=1)
            )
    return ece.mean(axis=1)


# ── Data for convergence panels ──
ranks_fig = np.asarray(res_fig["sbc_ranks"])
n_sbc_post = 1000
N_SBC_TEST = ranks_fig.shape[0]

# ── Aggregate across reps ──
# Error bars are the standard deviation across repetitions, not the standard
# error of their mean. Every reference level drawn in this figure -- the C2ST
# split floor, the EoD expectation, the ECE null band -- describes where a
# *single* run should land, so the observed spread has to be reported on the
# same footing.
c2st_joint_mean, c2st_joint_sd = [], []
c2st_marginal_mean = {k: [] for k in ["gamma", "mu1", "mu2"]}
c2st_marginal_sd = {k: [] for k in ["gamma", "mu1", "mu2"]}
avg_eod_mean, avg_eod_sd = [], []
avg_discrete_ece_mean, avg_discrete_ece_sd = [], []
ece_null_lo, ece_null_hi = [], []
n_ece_bins_bl = 10
N_ECE_NULL_REPLICATES = 200
n_rep_inside, n_rep_total = 0, 0

for b in BUDGETS:
    reps = budget_results[b]["reps"]

    joint = [r["c2st_all"] for r in reps]
    c2st_joint_mean.append(np.mean(joint))
    c2st_joint_sd.append(np.std(joint))

    for key in ["gamma", "mu1", "mu2"]:
        vals = [r["c2st_marginal"][key] for r in reps]
        c2st_marginal_mean[key].append(np.mean(vals))
        c2st_marginal_sd[key].append(np.std(vals))

    eod_vals = [eod_from_ranks(np.asarray(r["sbc_ranks"])[:, :3], n_sbc_post).mean() for r in reps]
    avg_eod_mean.append(np.mean(eod_vals))
    avg_eod_sd.append(np.std(eod_vals))

    ece_vals = [np.mean(r["ece"]) for r in reps]
    avg_discrete_ece_mean.append(np.mean(ece_vals))
    avg_discrete_ece_sd.append(np.std(ece_vals))

    # Null for an individual rep, pooled over the five reps at this budget.
    # It must not be narrowed by averaging: the reps share one SBC test set, so
    # their label noise is common rather than independent, and a rep-averaged
    # null would understate the floor by roughly sqrt(n_reps).
    per_rep_draws = [
        simulate_ece_null(
            r["ece_probs"],
            num_bins=n_ece_bins_bl,
            n_replicates=N_ECE_NULL_REPLICATES,
            seed=1000 * BUDGETS.index(b) + rep_idx,
        )
        for rep_idx, r in enumerate(reps)
    ]
    for observed, draws in zip(ece_vals, per_rep_draws):
        lo, hi = np.quantile(draws, [0.025, 0.975])
        n_rep_inside += int(lo <= observed <= hi)
        n_rep_total += 1

    pooled = np.concatenate(per_rep_draws)
    ece_null_lo.append(np.quantile(pooled, 0.025))
    ece_null_hi.append(np.quantile(pooled, 0.975))

c2st_joint_mean = np.array(c2st_joint_mean)
c2st_joint_sd = np.array(c2st_joint_sd)
for key in c2st_marginal_mean:
    c2st_marginal_mean[key] = np.array(c2st_marginal_mean[key])
    c2st_marginal_sd[key] = np.array(c2st_marginal_sd[key])
avg_eod_mean = np.array(avg_eod_mean)
avg_eod_sd = np.array(avg_eod_sd)
avg_discrete_ece_mean = np.array(avg_discrete_ece_mean)
avg_discrete_ece_sd = np.array(avg_discrete_ece_sd)
ece_null_lo = np.array(ece_null_lo)
ece_null_hi = np.array(ece_null_hi)

print(f"Individual reps inside their own 95% ECE null: {n_rep_inside}/{n_rep_total}")
for b, obs, lo, hi in zip(BUDGETS, avg_discrete_ece_mean, ece_null_lo, ece_null_hi):
    flag = "" if lo <= obs <= hi else "   <-- mean outside"
    print(f"  {b:>7,}: mean {obs:.4f}, null [{lo:.4f}, {hi:.4f}]{flag}")

# Predictive panel indices: 0=n_arr (count), 3=q1 (queue length)
PRED_INDICES = [0, 3]
PRED_NAMES = [OBS_NAMES[i] for i in PRED_INDICES]
PRED_XLIMS = [1500, 2]

# Pairplot samples
N_PP = min(len(mnpe_samples), len(ref_samples))
mnpe_pp = mnpe_samples[:N_PP]
ref_pp = ref_samples[:N_PP]
all_pp_samples = np.vstack([mnpe_pp, ref_pp])
pp_limits = []
for i in range(n_params):
    lo, hi = all_pp_samples[:, i].min(), all_pp_samples[:, i].max()
    if i >= 3:
        lo, hi = 1.5, 6.5
    else:
        margin = (hi - lo) * 0.05
        lo -= margin
        hi += margin
    pp_limits.append([lo, hi])

print(f"Pairplot: {N_PP} samples per method")

CLR_MNPE_STRONG = "#1a8f1a"
CLR_MCMC_STRONG = "#0b5ea2"
SPINE_OFFSET = 2

# ── One meaning per visual token, figure-wide ───────────────────────────────
# Colour says *what* is measured:
#   purple family  -> continuous parameters, darkest (CLR_GAMMA) first
#   orange family  -> discrete parameters, darkest (CLR_C1) first
#   CLR_JOINT      -> MNPE joint over all five parameters, darkest of the ramp
#   CLR_DEQUANT    -> the dequantized-NPE baseline, neutral against the family
#   green / blue   -> MNPE vs reference posterior
#   light grey     -> prior
# Style says *what kind* of quantity, independently of colour:
#   solid + markers -> measured value
#   dashed line     -> the level a perfect result would sit at
#   shaded band     -> the spread around that level
# A null keeps the colour of whatever it is the null *for*. Where it applies to
# every curve in the panel it is drawn neutral -- the C2ST floor in (b), the
# uniform band in (d1), the identity line in (d2). Where a panel holds two
# quantities with two different nulls, as (d3) does, each null takes its own
# family colour; drawing both neutral there would make the grey ECE band look
# like it bounds the purple EoD curve as well.
# In (d3) the two curves are averages over a family and take that family's
# darkest shade, which the "Avg." in the axis label signals.
BUDGET_TICKS = [1e3, 1e4, 1e5]
BUDGET_TICKLABELS = ["$10^3$", "$10^4$", "$10^5$"]
# The joint curve is MNPE aggregated over all five parameters, and the violets
# below it are the same model's per-parameter marginals. Green would read as a
# different method; this is the same family, one shade darker than gamma so the
# aggregate sits at the top of the ramp.
CLR_JOINT = "#2a1155"
CLR_DEQUANT = "#4a4a4a"
CLR_NULL_LINE = "#111111"
CLR_NULL = "0.45"
CLR_PRIOR = "0.55"

# ═══════════════════════════════════════════════════════════════════════════
# Figure
# ═══════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(8, 2.65))
gs_top = GridSpec(1, 2, figure=fig, width_ratios=[5, 4], wspace=0.15)
gs_left = gs_top[0, 0].subgridspec(2, 3, hspace=0.6, wspace=0.7)
gs_right = gs_top[0, 1].subgridspec(5, 5, hspace=0.03, wspace=0.03)

# ═══════════════════════════════════════════════════════════════════════════
# Left Row 1: Predictives + C2ST
# ═══════════════════════════════════════════════════════════════════════════

# ── (a1) Predictive: n_arrivals ──
ax = fig.add_subplot(gs_left[0, 0])
idx = PRED_INDICES[0]
ax.hist(prior_pred_obs[:, idx], bins="auto", alpha=0.5, density=True, color=CLR_PRIOR)
ax.hist(ref_pred_obs[:, idx], bins=30, alpha=0.5, density=True, color=COLORS["Marginalized"])
ax.hist(mnpe_pred_obs[:, idx], bins=30, alpha=0.5, density=True, color=COLORS["MNPE"])
ax.axvline(x_o_np[idx], color=CLR_GT, lw=1.5, ls="--", alpha=0.6)
ax.set_xlabel(PRED_NAMES[0], fontsize=FS_LABEL)
ax.set_ylabel("Density", fontsize=FS_LABEL)
ax.set_xlim(right=PRED_XLIMS[0])
ax.set_ylim(bottom=0, top=0.015)
ax.set_yticks([0, 0.015], [])
ax.tick_params(labelsize=FS_TICK)
despine(ax, offset=SPINE_OFFSET)

# ── (a2) Predictive: q1 ──
ax = fig.add_subplot(gs_left[0, 1])
idx = PRED_INDICES[1]
ax.hist(prior_pred_obs[:, idx], bins=100, alpha=0.5, density=True, color=CLR_PRIOR)
ax.hist(mnpe_pred_obs[:, idx], bins=30, alpha=0.5, density=True, color=COLORS["MNPE"])
ax.hist(ref_pred_obs[:, idx], bins=30, alpha=0.5, density=True, color=COLORS["Marginalized"])
ax.axvline(x_o_np[idx], color=CLR_GT, lw=1.5, ls="--", alpha=0.6)
ax.set_xlabel(PRED_NAMES[1], fontsize=FS_LABEL)
ax.set_xlim(left=0, right=PRED_XLIMS[1])
ax.set_xticks(np.linspace(0, PRED_XLIMS[1], 3))
ax.set_ylim(bottom=0, top=6)
ax.set_yticks([0, 6], [])
ax.tick_params(labelsize=FS_TICK)
despine(ax)

# ── (b) C2ST vs training budget ──
ax = fig.add_subplot(gs_left[0, 2])
ax.errorbar(
    budgets_arr,
    c2st_joint_mean,
    yerr=c2st_joint_sd,
    fmt="o-",
    ms=2,
    color=CLR_JOINT,
    capsize=2,
    capthick=0.8,
    lw=1.2,
)
for key, clr, marker in [
    ("gamma", CLR_GAMMA, "s"),
    ("mu1", CLR_MU1, "^"),
    ("mu2", CLR_MU2, "v"),
]:
    ax.errorbar(
        budgets_arr,
        c2st_marginal_mean[key],
        yerr=c2st_marginal_sd[key],
        fmt=f"{marker}-",
        ms=2,
        color=clr,
        capsize=2,
        capthick=0.8,
        lw=1.0,
    )
# Measured C2ST floor: disjoint splits of the reference against itself. On this
# linear axis the band is only a few pixels tall and sits close to the bottom
# spine; the quantitative comparison lives in the yardstick table, and the panel
# only has to show that the curves converge toward it without reaching it.
ax.axhspan(
    max(c2st_floor_mean - c2st_floor_sd, 0.5),
    c2st_floor_mean + c2st_floor_sd,
    color=CLR_NULL,
    alpha=0.25,
    lw=0,
)
ax.axhline(c2st_floor_mean, color=CLR_NULL_LINE, ls="--", lw=0.9)
# Dequantized NPE: same quantity as the joint curve, different method, so the
# joint colour with a hollow marker.
ax.errorbar(
    [DEQ_BUDGET],
    [deq_c2st_mean],
    yerr=[deq_c2st_sd],
    fmt="o",
    ms=2.5,
    mew=0.8,
    mfc="white",
    mec=CLR_DEQUANT,
    ecolor=CLR_DEQUANT,
    capsize=2,
    capthick=0.8,
    lw=0,
)
ax.set_xscale("log")
ax.set_ylabel("C2ST", fontsize=FS_LABEL)
ax.set_ylim(0.5, 1.0)
ax.set_yticks([0.5, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.5", "0.6", "0.8", "1.0"])
ax.set_xlim(1e3 - 100, 1.3e5)
ax.set_xticks(BUDGET_TICKS)
ax.set_xticklabels(BUDGET_TICKLABELS)
ax.tick_params(labelsize=FS_TICK)
# The joint curve is the one token that exists in this panel and nowhere else,
# so it is labelled here rather than in the shared legend.
ax.legend(
    handles=[
        Line2D([], [], color=CLR_JOINT, lw=1.2, marker="o", ms=2, label="MNPE (joint)"),
        Line2D(
            [],
            [],
            color=CLR_DEQUANT,
            lw=0,
            marker="o",
            ms=2.5,
            mew=0.8,
            mfc="white",
            mec=CLR_DEQUANT,
            label="NPE baseline",
        ),
    ],
    # Nudged past the axes corner so the handles clear the 1k data points.
    loc="upper right",
    bbox_to_anchor=(1.13, 1.03),
    fontsize=FS_LEGEND,
    frameon=False,
    handlelength=0.8,
    handletextpad=0.3,
    labelspacing=0.35,
    borderpad=0.0,
    borderaxespad=0.0,
)
despine(ax, offset=SPINE_OFFSET)

# ═══════════════════════════════════════════════════════════════════════════
# Left Row 2: SBC eCDF + ECE reliability + EoD / ECE convergence
# ═══════════════════════════════════════════════════════════════════════════

# ── (d1) SBC rank eCDF ──
ax_sbc = fig.add_subplot(gs_left[1, 0])
sbc_rank_ecdf(
    ax_sbc,
    ranks_fig[:, :3],
    colors=[CLR_GAMMA, CLR_MU1, CLR_MU2],
)
ax_sbc.tick_params(labelsize=FS_TICK)

# ── (d2) ECE reliability diagram ──
ax = fig.add_subplot(gs_left[1, 1])
n_ece_bins = 10
bin_edges_rel = np.linspace(0, 1, n_ece_bins + 1)
bin_centers = (np.arange(n_ece_bins) + 0.5) / n_ece_bins
bar_width = 0.04
offsets = [-bar_width / 2, bar_width / 2]
for j, (offset, clr) in enumerate(zip(offsets, [CLR_C1, CLR_C2])):
    top_conf = res_fig["ece_probs"][:, j, :].max(axis=1)
    true_cat = res_fig["ece_true_vals"][:, j].astype(int)
    pred_cat = res_fig["ece_probs"][:, j, :].argmax(axis=1)
    correct = (pred_cat == true_cat).astype(float)
    bin_idx = np.digitize(top_conf, bin_edges_rel) - 1
    bin_idx = np.clip(bin_idx, 0, n_ece_bins - 1)
    counts = np.array([np.sum(bin_idx == b) for b in range(n_ece_bins)])
    acc = np.array(
        [correct[bin_idx == b].mean() if counts[b] > 0 else 0.0 for b in range(n_ece_bins)]
    )
    mask = counts > 0
    ax.bar(
        bin_centers[mask] + offset,
        acc[mask],
        width=bar_width,
        alpha=0.7,
        color=clr,
        edgecolor="white",
        lw=0.5,
    )
ax.plot([0, 1], [0, 1], "--", color=CLR_NULL_LINE, lw=0.8, alpha=0.7)
ax.set_xlabel("Confidence", fontsize=FS_LABEL)
ax.set_ylabel("Accuracy", fontsize=FS_LABEL)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.tick_params(labelsize=FS_TICK)
despine(ax)

# ── (d3) Calibration convergence: avg EoD + avg ECE ──
ax_cont = fig.add_subplot(gs_left[1, 2])
eod_expected = eod_expected_under_uniformity(num_sbc_runs=N_SBC_TEST)
plot_ymax = 0.05

ax_cont.errorbar(
    budgets_arr,
    avg_eod_mean,
    yerr=avg_eod_sd,
    fmt="-",
    ms=4,
    color=CLR_GAMMA,
    markerfacecolor=CLR_GAMMA,
    markeredgecolor="white",
    markeredgewidth=0.5,
    capsize=2,
    capthick=0.8,
)
ax_cont.axhline(eod_expected, color=CLR_GAMMA, ls="--", lw=1.0, alpha=0.9)
ax_cont.set_xscale("log")
ax_cont.set_xlabel("Simulations", fontsize=FS_LABEL)
ax_cont.set_ylabel("Avg. EoD / ECE", fontsize=FS_LABEL)
ax_cont.tick_params(axis="y", labelsize=FS_TICK)
ax_cont.tick_params(axis="x", labelsize=FS_TICK)
ax_cont.set_xlim(1e3 - 100, 1.3e5)
ax_cont.set_xticks(BUDGET_TICKS)
ax_cont.set_xticklabels(BUDGET_TICKLABELS)
ax_cont.set_ylim(0, plot_ymax)
ax_cont.set_yticks([0.0, plot_ymax])
despine(ax_cont, offset=SPINE_OFFSET)

ax_disc = ax_cont
# 95% null for the rep-averaged ECE, simulated from each model's own PMF.
ax_disc.fill_between(
    budgets_arr,
    ece_null_lo,
    ece_null_hi,
    color=CLR_C1,
    alpha=0.2,
    lw=0,
)
ax_disc.errorbar(
    budgets_arr,
    avg_discrete_ece_mean,
    yerr=avg_discrete_ece_sd,
    ms=4,
    color=CLR_C1,
    markerfacecolor=CLR_C1,
    markeredgecolor="white",
    markeredgewidth=0.5,
    alpha=0.7,
    capsize=2,
    capthick=0.8,
    lw=1.0,
)

# ═══════════════════════════════════════════════════════════════════════════
# Right: 5×5 Pairplot
# ═══════════════════════════════════════════════════════════════════════════
pair_axes = np.array(
    [[fig.add_subplot(gs_right[r, c]) for c in range(n_params)] for r in range(n_params)]
)

pairplot(
    samples=[ref_pp, mnpe_pp],
    points=np.array(TRUE_PARAMS).reshape(1, -1),
    limits=pp_limits,
    labels=PARAM_NAMES,
    upper="scatter",
    lower=None,
    diag="hist",
    fig=fig,
    axes=pair_axes,
    fig_kwargs=dict(
        legend=False,
        samples_labels=["Reference", "MNPE"],
        samples_colors=[CLR_MCMC_STRONG, CLR_MNPE_STRONG],
        points_colors=[CLR_GT],
        points_labels=[r"$\theta_o$"],
        points_offdiag={"markersize": 3},
        despine={"offset": 5},
        fig_subplots_adjust={},
    ),
    # Reference as an outline so it is not buried under the filled MNPE
    # histogram; with both filled, only the second-drawn method is visible.
    diag_kwargs=[
        dict(mpl_kwargs=dict(bins=50, histtype="step", lw=0.9)),
        dict(mpl_kwargs=dict(alpha=0.7, bins=50)),
    ],
    upper_kwargs=[
        dict(mpl_kwargs=dict(s=4, alpha=0.8, rasterized=True)),
        dict(mpl_kwargs=dict(s=4, alpha=0.8, rasterized=True)),
    ],
)

for r in range(n_params):
    for c in range(n_params):
        pair_axes[r, c].tick_params(labelsize=FS_TICK)

# A 50-bin histogram of a 5-category variable renders as arbitrary thin spikes
# and hides the reference. The discrete weights are where MNPE's whole residual
# error sits, so these two diagonals are redrawn as side-by-side PMF bars.
discrete_cats = np.array(C_VALUES, dtype=float)
for k in (3, 4):
    ax_diag = pair_axes[k, k]
    xlabel = ax_diag.get_xlabel()
    ax_diag.clear()
    ref_pmf = np.array([(np.rint(ref_pp[:, k]) == c).mean() for c in discrete_cats])
    mnpe_pmf = np.array([(np.rint(mnpe_pp[:, k]) == c).mean() for c in discrete_cats])
    bar_w = 0.36
    ax_diag.bar(discrete_cats - bar_w / 2, ref_pmf, width=bar_w, color=CLR_MCMC_STRONG, lw=0)
    ax_diag.bar(discrete_cats + bar_w / 2, mnpe_pmf, width=bar_w, color=CLR_MNPE_STRONG, lw=0)
    ax_diag.axvline(TRUE_PARAMS[k], color=CLR_GT, lw=1.0, zorder=0)
    ax_diag.set_xlim(*pp_limits[k])
    ax_diag.set_ylim(0, max(ref_pmf.max(), mnpe_pmf.max()) * 1.15)
    ax_diag.set_xticks([2, 4, 6])
    ax_diag.set_yticks([])
    ax_diag.set_xlabel(xlabel, fontsize=FS_LABEL)
    ax_diag.tick_params(labelsize=FS_TICK)
    for side in ("top", "right", "left"):
        ax_diag.spines[side].set_visible(False)
    ax_diag.spines["bottom"].set_position(("outward", 5))

    tv = 0.5 * np.abs(ref_pmf - mnpe_pmf).sum()
    print(f"  {PARAM_NAMES[k]} marginal TV (reference vs MNPE): {tv:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# Consolidated legend in pairplot lower-left triangle
# ═══════════════════════════════════════════════════════════════════════════
# Two columns, two kinds of entry. The left column names *what is plotted*, the
# right column names *which parameter a colour stands for*. Splitting them is
# what makes the legend readable across panels: otherwise one grey swatch would
# have to cover the prior, the joint C2ST curve and three different null bands.
# matplotlib fills ncol=2 column-major, so the handles are ordered accordingly.
# Nulls are not listed: dashed means "the level a perfect result sits at" and a
# shaded band means "the spread around it", which the caption states once. The
# two black entries are distinct objects -- (a) marks the observation x_o with a
# dashed line, (c) marks the true parameters theta_o with a solid line and dot.
legend_handles = [
    Patch(facecolor=COLORS["MNPE"], alpha=0.7, label="MNPE"),
    Patch(facecolor=COLORS["Marginalized"], alpha=0.7, label="Reference"),
    Patch(facecolor=CLR_PRIOR, alpha=0.5, label="Prior"),
    Line2D([], [], color=CLR_GT, ls="--", lw=1.2, alpha=0.6, label=r"$x_o$"),
    Line2D([], [], color=CLR_GT, lw=1.2, marker="o", ms=3, label=r"$\theta_o$"),
    Line2D([], [], color=CLR_GAMMA, lw=1.5, label=r"$\gamma$"),
    Line2D([], [], color=CLR_MU1, lw=1.5, label=r"$\mu_1$"),
    Line2D([], [], color=CLR_MU2, lw=1.5, label=r"$\mu_2$"),
    Line2D([], [], color=CLR_C1, lw=1.5, label=r"$c_1$"),
    Line2D([], [], color=CLR_C2, lw=1.5, label=r"$c_2$"),
]

ax_leg = pair_axes[2, 0]
ax_leg.legend(
    handles=legend_handles,
    loc="center",
    fontsize=FS_LEGEND,
    frameon=False,
    ncol=2,
    handlelength=1.0,
    handletextpad=0.5,
    labelspacing=0.4,
    columnspacing=0.8,
    bbox_to_anchor=(1.2, -1.5),
)

# ── Panel labels ──
fig.text(0.08, 0.93, "a", fontsize=FS_PANEL, fontweight="bold")
fig.text(0.37, 0.93, "b", fontsize=FS_PANEL, fontweight="bold")
fig.text(0.56, 0.93, "c", fontsize=FS_PANEL, fontweight="bold")
fig.text(0.08, 0.45, "d", fontsize=FS_PANEL, fontweight="bold")

written = save_figure(fig, "queue_paper_figure", bbox_inches="tight")
plt.close(fig)
print(f"Saved {written.name} (and .png) to {written.parent}")
