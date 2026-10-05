"""Conditional-SBC deviation as a function of training budget.

Companion to conditional_diagnostics.py, backing Appendix A.3. That diagnostic
shows a small upward shift in the mu1/mu2 conditionals of the lowest-C2ST
100k-budget posterior. The question this answers: does the deviation shrink with
simulation budget — a finite-budget artifact — or persist, which would make it a
structural coupling failure?

Runs the same conditional rank loop (rank theta_c* under
q(theta_c | theta_d = theta_d*, x), M=1000) for EVERY rep at EVERY budget,
on the same seed-9999 test set, and reports the KS statistic and p-value per
continuous parameter.

Run with --all-reps-at BUDGET to instead evaluate ALL reps at one budget
(seed-to-seed spread at fixed budget — distinguishes training variability from
structural miscalibration).

Outputs:
  conditional_sbc_budget_trend.csv     — budget, rep, KS stat + p per parameter
  conditional_sbc_budget_trend.pdf/png — KS statistic vs budget
  conditional_sbc_reps_<BUDGET>.csv    — with --all-reps-at

Run:  uv run python experiments/queue/diagnostics/conditional_sbc_budget_trend.py
"""

import argparse
import pickle
import sys

import matplotlib.pyplot as plt
import pandas as pd
import torch
from scipy import stats

from mnpe_paper.models.queue.conditional import conditional_ranks
from mnpe_paper.models.queue.config import generate_training_data, transform_x
from mnpe_paper.models.queue.spec import SBC_TEST_SEED
from mnpe_paper.paths import model_artifacts_dir, model_results_dir
from mnpe_paper.plotting.plot_style import CLR_GAMMA, CLR_MU1, CLR_MU2, despine

ARTIFACTS_DIR = model_artifacts_dir("queue")
RESULTS_DIR = model_results_dir("queue")
N_SBC_TEST = 1_000
N_CAP = 1_000
M_COND = 1_000
SEED = 0
CONT_NAMES = ["gamma", "mu1", "mu2"]
CONT_COLORS = [CLR_GAMMA, CLR_MU1, CLR_MU2]

with open(
    ARTIFACTS_DIR / "budget_study_results_n1000.pkl", "rb"
) as f:
    study = pickle.load(f)

theta_test, x_test_raw = generate_training_data(N_SBC_TEST, seed=SBC_TEST_SEED)
theta_test, x_test_raw = theta_test[:N_CAP], x_test_raw[:N_CAP]
x_test = transform_x(x_test_raw)
n_test = len(theta_test)
print(f"{n_test} valid test cases (seed={SBC_TEST_SEED})")


def ks_row(budget, idx):
    with open(
        ARTIFACTS_DIR / "posteriors"
        / f"posterior_{budget}_rep{idx}.pkl",
        "rb",
    ) as f:
        post = pickle.load(f)
    torch.manual_seed(SEED)
    ranks = conditional_ranks(post.posterior_estimator, theta_test, x_test, M_COND)
    row = {"budget": budget, "rep": idx}
    for k, name in enumerate(CONT_NAMES):
        ks = stats.kstest(ranks[:, k].numpy(), stats.uniform(loc=0, scale=M_COND).cdf)
        row[f"ks_stat_{name}"] = float(ks.statistic)
        row[f"ks_pval_{name}"] = float(ks.pvalue)
    print(
        f"  budget {budget:>7,} (rep {idx}): "
        + ", ".join(
            f"{n} D={row[f'ks_stat_{n}']:.3f} p={row[f'ks_pval_{n}']:.3f}"
            for n in CONT_NAMES
        )
    )
    return row


parser = argparse.ArgumentParser()
parser.add_argument(
    "--all-reps-at", type=int, metavar="BUDGET",
    help="evaluate all reps at one budget (seed spread) instead of the budget trend",
)
args = parser.parse_args()

if args.all_reps_at:
    budget = args.all_reps_at
    n_reps = len(study["budget_results"][budget]["reps"])
    df = pd.DataFrame([ks_row(budget, r) for r in range(n_reps)])
    df.to_csv(
        RESULTS_DIR / f"conditional_sbc_reps_{budget}.csv",
        index=False,
    )
    print(f"Saved conditional_sbc_reps_{budget}.csv")
    sys.exit(0)

# Evaluate every repetition at every budget. Selecting the C2ST-best rep per
# budget picks a different model at each point, so the resulting curve confounds
# the budget effect with selection noise. Averaging over all
# reps and reporting the spread gives a controlled estimate instead.
rows = []
for budget in study["budgets"]:
    n_reps = len(study["budget_results"][budget]["reps"])
    for idx in range(n_reps):
        rows.append(ks_row(budget, idx))

df = pd.DataFrame(rows)
df.to_csv(
    RESULTS_DIR / "conditional_sbc_budget_trend.csv", index=False
)

# ── Figure: KS statistic vs budget, mean over reps with spread ──
summary = df.groupby("budget").agg(["mean", "std"])
summary.to_csv(
    RESULTS_DIR / "conditional_sbc_budget_trend_summary.csv"
)
budgets = summary.index.to_numpy()

fig, ax = plt.subplots(figsize=(3.2, 2.4))
for name, color in zip(CONT_NAMES, CONT_COLORS):
    mean = summary[(f"ks_stat_{name}", "mean")].to_numpy()
    spread = summary[(f"ks_stat_{name}", "std")].to_numpy()
    ax.errorbar(
        budgets, mean, yerr=spread, fmt="o-", color=color, label=name,
        capsize=2, lw=1.0, markersize=3,
    )
    # Individual repetitions, so the spread is visible rather than only summarized.
    ax.plot(
        df["budget"], df[f"ks_stat_{name}"], ".", color=color,
        alpha=0.25, markersize=3, zorder=0,
    )
crit = stats.ksone.ppf(1 - 0.025, n_test)  # two-sided 5% critical value
ax.axhline(crit, color="0.5", ls=":", lw=0.8)
ax.text(df["budget"].min(), crit, "  KS crit. (5%)", color="0.5", va="bottom")
ax.set_xscale("log")
ax.set_xlabel("Training budget")
ax.set_ylabel("KS statistic (cond. ranks)")
ax.legend()
despine(ax, offset=3)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(
        ARTIFACTS_DIR / f"conditional_sbc_budget_trend.{ext}"
    )
plt.close(fig)
print("Saved conditional_sbc_budget_trend.csv/.pdf/.png")
