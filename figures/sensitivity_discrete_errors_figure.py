"""Error-propagation figure for the toy example (paper Appendix B.5).

Reads ``results/toy/sensitivity_discrete_errors.csv`` and plots the discrete and
continuous total-variation errors against the temperature T applied to the
discrete head. Both are V-shaped with a minimum near T=1; the continuous error
floors at the (T-independent) conditional-head error while the discrete error
dips to ~0, and that gap is the irreducible conditional error rather than
propagation.

Also prints the propagation factor quoted in the paper: the slope of the *excess*
continuous error against the *excess* discrete error, both relative to their T=1
values so that the fit passes through the origin. A raw ratio TV_c/TV_d is not
used, because near T=1 the discrete error goes to 0 while the continuous error
floors, so the ratio diverges on the nonzero intercept rather than on any
amplification.

This figure builds from a tracked CSV rather than through
:mod:`mnpe_paper.figure_io`, because the sweep's own output *is* the compact
result — there is no heavy artifact for it to prefer.

Usage:
    uv run python figures/sensitivity_discrete_errors_figure.py

Outputs:
    figures/sensitivity_discrete_errors_redesign.pdf
    figures/sensitivity_discrete_errors_redesign.png
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mnpe_paper.figure_io import save_figure
from mnpe_paper.paths import model_results_dir
from mnpe_paper.plotting.plot_style import CLR_CONTINOUS, CLR_DISCRETE, despine

CSV_PATH = model_results_dir("toy") / "sensitivity_discrete_errors.csv"

if not CSV_PATH.exists():
    raise SystemExit(
        f"{CSV_PATH} is missing. It is tracked in this repository; if it has been "
        "removed, regenerate it with experiments/toy/sensitivity_discrete_errors.py "
        "(which needs artifacts/toy/mnpe_posterior.pt)."
    )

df = pd.read_csv(CSV_PATH)
TEMPS = sorted(df["T"].unique())

# ── Propagation factor ─────────────────────────────────────────────────────
base = df[df["T"] == 1.0].set_index("x_o")
d = df.copy()
d["ex_disc"] = d.apply(lambda r: r["tv_discrete"] - base.loc[r["x_o"], "tv_discrete"], axis=1)
d["ex_cont"] = d.apply(lambda r: r["tv_continuous"] - base.loc[r["x_o"], "tv_continuous"], axis=1)
d = d[d["T"] != 1.0]
pooled_slope = float(np.sum(d["ex_disc"] * d["ex_cont"]) / np.sum(d["ex_disc"] ** 2))
per_x_slopes = [
    float(np.sum(g["ex_disc"] * g["ex_cont"]) / np.sum(g["ex_disc"] ** 2))
    for _, g in d.groupby("x_o")
]
print(f"propagation slope (excess-error, pooled): {pooled_slope:.3f}")
print(f"per-x_o slopes: {[round(s, 3) for s in per_x_slopes]} "
      f"(range {min(per_x_slopes):.3f}-{max(per_x_slopes):.3f})")

# ── Figure ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(3.4, 2.3))

for _, g in df.groupby("x_o"):
    g = g.sort_values("T")
    ax.plot(g["T"], g["tv_discrete"], "-", lw=0.7, alpha=0.30, color=CLR_DISCRETE)
    ax.plot(g["T"], g["tv_continuous"], "-", lw=0.7, alpha=0.30, color=CLR_CONTINOUS)
mean_by_T = df.groupby("T")[["tv_discrete", "tv_continuous"]].mean().sort_index()
ax.plot(mean_by_T.index, mean_by_T["tv_discrete"], "o-", ms=3, lw=1.6,
        color=CLR_DISCRETE, label=r"discrete TV$(q_T(z),\,p(z))$")
ax.plot(mean_by_T.index, mean_by_T["tv_continuous"], "o-", ms=3, lw=1.6,
        color=CLR_CONTINOUS, label=r"continuous TV$(q_T(\theta),\,p(\theta))$")
ax.axvline(1.0, ls=":", color="0.5", lw=0.8, zorder=0)
ax.set_xscale("log", base=2)
ax.set_xticks(TEMPS)
ax.set_xticklabels([f"{t:g}" for t in TEMPS])
ax.minorticks_off()
ax.set_xlabel(r"temperature $T$")
ax.set_ylabel("total variation error")
ax.legend(frameon=False, handlelength=1.4, loc="upper center")

despine(ax, offset=3)
fig.tight_layout()
written = save_figure(fig, "sensitivity_discrete_errors_redesign")
plt.close(fig)
print(f"Saved {written.name} (and .png) to {written.parent}")
