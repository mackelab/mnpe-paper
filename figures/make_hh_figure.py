"""Main Hodgkin-Huxley figure (Fig. 4): MNPE on a biophysical neuron model.

Layout::

    +-------+-------+-----------+
    |   a   |   b   |           |
    +-------+-------+     e     |
    |   c   |   d   |           |
    +-------+-------+-----------+

Panels:
  (a) observed voltage trace with two posterior-predictive traces
  (b) median summary-statistic MSE against the training budget (mean +- std, 3 seeds)
  (c) SBC rank eCDF of the continuous marginals, and discrete reliability diagram
  (d) marginal EoD and discrete ECE against the training budget, with baselines
  (e) posterior pairplot over g_Na, g_K and the two channel switches

Panels (a), (c) and (e) show one run, 100k simulations at seed 2, and one
observation (284); panels (b) and (d) aggregate all runs. The pairplot's jitter on
the discrete axes is random, so it is seeded here to make rebuilds byte-identical.

Usage:
    uv run python figures/make_hh_figure.py

Outputs:
    figures/figureHH.pdf
    figures/figureHH.png
"""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sbi.analysis.plot import sbc_rank_plot

from mnpe_paper.calibration import (
    discrete_calibration_overlay_plot,
    eod_expected_under_uniformity,
)
from mnpe_paper.figure_io import load_figure_inputs, save_figure
from mnpe_paper.models.hh import figure_inputs
from mnpe_paper.plotting.pairplot import pairplot
from mnpe_paper.plotting.plot_style import (
    CLR_CA,
    CLR_CONTINOUS,
    CLR_DISCRETE,
    CLR_EL,
    CLR_GK,
    CLR_GL,
    CLR_GNA,
    CLR_GT,
    CLR_KM,
)
from mnpe_paper.plotting.plot_style import (
    COLORS as PS_COLORS,
)

CLR_MNPE = PS_COLORS["MNPE"]
CLR_MNPE_LIGHT = PS_COLORS["MNPE_light"]
# Per-parameter colors for the continuous SBC rank plot in panel c
HH_PARAM_COLORS = [CLR_GNA, CLR_GK, CLR_GL, CLR_EL]

JITTER_SEED = 0


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.3, y: float = 1.55) -> None:
    """Add a bold panel label (a, b, …) to the top-left of an axis."""
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
        ha="left",
    )


def plot_traces(ax: plt.Axes, obs_data: dict, n_traces: int = 3) -> None:
    """Panel a: observed trace overlaid with posterior predictive samples.

    Parameters
    ----------
    obs_data : dict
        Dict with keys 'x_raw_o' (observed trace) and 'x_post' (simulated
        traces, shape (n_samples, T)).
    n_traces : int
        Number of posterior predictive traces to overlay.
    """
    dt = 0.02  # ms per time step
    x_raw_o = obs_data["x_raw_o"]
    x_post = obs_data["x_post"]
    t = np.arange(len(x_raw_o)) * dt

    _trace_colors = [CLR_MNPE, CLR_MNPE_LIGHT]
    for j in range(min(n_traces, len(x_post))):
        ax.plot(t, x_post[j], color=_trace_colors[j % len(_trace_colors)], linewidth=0.8, alpha=1)
    ax.plot(t, x_raw_o, color=CLR_GT, linewidth=0.8, label="Observation")

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Voltage (mV)")
    ax.set_ylim(-90, 60)
    ax.tick_params(direction="out")
    ax.spines["left"].set_position(("outward", 3))
    ax.spines["bottom"].set_position(("outward", 3))


def plot_mse(ax: plt.Axes, agg: pd.DataFrame) -> None:
    """Panel b: median MSE vs number of training samples."""
    ax.errorbar(
        agg["n_train"],
        agg["mse_p50_avg"],
        yerr=agg["mse_p50_std"],
        fmt="o-",
        capsize=3,
        color=CLR_MNPE,
        linewidth=1.0,
        markersize=3,
    )
    ax.set_ylabel("MSE")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([100, 1000, 10000, 100000])
    ax.set_yticks([0.1, 1, 10])
    ax.tick_params(direction="out")
    ax.spines["left"].set_position(("outward", 3))
    ax.spines["bottom"].set_position(("outward", 3))


PARAM_LABELS = ["$g_{Na}$", "$g_K$", "$g_L$", "$E_L$", "$K_M$", "$Ca$"]


def plot_panel_c(ax: plt.Axes, sbc_data: dict, n_sbc_continuous: int = 1000) -> tuple[plt.Axes, plt.Axes]:
    """Panel c: c1 = SBC rank plot (marginals), c2 = discrete calibration overlay.

    Parameters
    ----------
    ax : plt.Axes
        Parent axes (hidden); c1 and c2 are inset axes inside it.
    sbc_data : dict
        SBC ranks and discrete calibration of the displayed run.
    n_sbc_continuous : int
        Number of continuous parameters in the SBC data to use
    """
    fig = ax.get_figure()
    pos = ax.get_position()
    ax.set_axis_off()
    ax.patch.set_visible(False)

    # Use fig.add_axes (absolute figure fractions) so plt.sca() works inside sbc_rank_plot
    w = pos.width * 0.40
    ax_left = fig.add_axes([pos.x0, pos.y0, w, pos.height])
    ax_right = fig.add_axes([pos.x0 + pos.width * 0.60, pos.y0, w, pos.height])

    # --- c1: SBC rank plot ---
    ranks = sbc_data["ranks_marginals"][:n_sbc_continuous,:4]
    num_posterior_samples = sbc_data["num_posterior_samples"]
    sbc_rank_plot(
        ranks,
        num_posterior_samples,
        plot_type="cdf",
        parameter_labels=PARAM_LABELS,
        colors=HH_PARAM_COLORS,
        fig=ax_left.get_figure(),
        ax=ax_left,
        num_bins=20,

    )
    ax_left.set_xlabel("Rank")
    ax_left.set_ylabel("CDF")
    xlim = ax_left.get_xlim()
    ylim = ax_left.get_ylim()
    ax_left.set_xticks([xlim[0], xlim[1]])
    ax_left.set_yticks([ylim[0], ylim[1]])
    ax_left.set_xticklabels(["0", "1"])
    ax_left.set_yticklabels(["0", "1"])

    if ax_left.get_legend():
        ax_left.get_legend().remove()

    # --- c2: discrete calibration overlay ---
    import matplotlib as _mpl
    _old_cycle = _mpl.rcParams["axes.prop_cycle"]
    _mpl.rcParams["axes.prop_cycle"] = _mpl.cycler(color=[CLR_KM, CLR_CA])
    discrete_calibration_overlay_plot(
        true_vals=sbc_data["true_vals_discrete"],
        probs=sbc_data["probs_discrete"],
        ece=sbc_data["ece_discrete"],
        param_names=[f"theta[{i}]" for i in sbc_data["discrete_indices"]],
        num_bins=sbc_data["num_discrete_bins"],
        exclude_low_populated=True,
        min_samples=10,
        ax=ax_right,
        show_legend=False,
        grouped=True,
        show_chance_region=False,
    )
    _mpl.rcParams["axes.prop_cycle"] = _old_cycle
    ax_right.set_ylabel("Accuracy", labelpad=0)
    ax_right.set_yticks([0, 1])
    ax_right.set_yticklabels([])

    return ax_left, ax_right


def plot_calibration(
    ax: plt.Axes,
    agg: pd.DataFrame,
    ece_expected: float,
) -> None:
    """Panel d: marginal (left axis) and discrete (right axis) ECE vs training samples."""
    fig = ax.get_figure()
    pos = ax.get_position()

    # Use fig.add_axes (same as plot_panel_c) so both panels use identical
    # axis-placement mechanism and stay aligned under bbox_inches='tight'.
    ax2 = fig.add_axes([pos.x0, pos.y0, pos.width, pos.height], label="cal_right")
    ax2.sharex(ax)
    ax2.patch.set_visible(False)
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.xaxis.set_visible(False)
    for spine in ["top", "left", "bottom"]:
        ax2.spines[spine].set_visible(False)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(CLR_DISCRETE)
    ax2.spines["right"].set_position(("outward", 3))

    ax.axhline(ece_expected, color=CLR_CONTINOUS, linestyle="--", zorder=0, alpha=0.7)

    ax.errorbar(
        agg["n_train"],
        agg["sbc_ece_marginals_avg"],
        yerr=agg["sbc_ece_marginals_std"],
        fmt="o-",
        capsize=3,
        color=CLR_CONTINOUS,
        linewidth=1.0,
        markersize=3,
    )
    ax2.errorbar(
        agg["n_train"],
        agg["ece_discrete_avg"],
        yerr=agg["ece_discrete_std"],
        fmt="o-",
        capsize=3,
        color=CLR_DISCRETE,
        linewidth=1.0,
        markersize=3,
    )
    if "ece_discrete_baseline_avg" in agg.columns:
        ax2.plot(
            agg["n_train"],
            agg["ece_discrete_baseline_avg"],
            ls="--",
            lw=1.0,
            color=CLR_DISCRETE,
            alpha=0.7,
            zorder=0,
        )

    ax.set_xlabel("Training samples")
    ax.set_ylabel("EoD (marg.)", color=CLR_CONTINOUS)
    ax.tick_params(axis="y", colors=CLR_CONTINOUS, direction="out")
    ax.spines["left"].set_color(CLR_CONTINOUS)
    ax.spines["left"].set_position(("outward", 3))
    ax.spines["bottom"].set_position(("outward", 3))
    ax.set_xscale("log")
    ax.set_xticks([100, 1000, 5000, 10000, 100000])
    ax.tick_params(axis="x", direction="out")

    ax2.set_ylabel("ECE (discr.)", color=CLR_DISCRETE)
    ax2.tick_params(axis="y", colors=CLR_DISCRETE, direction="out")

    ax.set_yticks([0, 0.1])
    ax2.set_yticks([0, 0.1])
    ax.set_ylim(0, 0.1)
    ax2.set_ylim(0, 0.1)


def plot_posterior(ax: plt.Axes, obs_data: dict) -> None:
    """Panel e: pairplot of posterior samples embedded into ax.

    Parameters
    ----------
    obs_data : dict
        Dict with keys 'samples' and 'theta_o' (torch.Tensors).
    """
    samples = obs_data["samples"]
    theta_o = obs_data["theta_o"]

    fig = ax.get_figure()
    bbox = ax.get_position()   # already shifted by row1_shift

    ax.set_axis_off()
    ax.patch.set_visible(False)

    gs_pp = gridspec.GridSpec(
        4, 4,
        left=bbox.x0, right=bbox.x1,
        bottom=bbox.y0, top=bbox.y1,
        hspace=0.2, wspace=0.2,
    )
    axes_pp = np.array(
        [[fig.add_subplot(gs_pp[i, j]) for j in range(4)] for i in range(4)]
    )

    # Snapshot positions of all axes (including axes_pp) so pairplot's
    # subplots_adjust doesn't corrupt the main layout or move axes_pp away
    # from the bbox we carefully placed them in.
    saved_positions = {a: a.get_position() for a in fig.axes}

    pairplot(
        samples,
        points=theta_o,
        subset=[0, 1, 4, 5],
        limits=[[-4, 4]] * 4 + [[-0.2, 1.2], [-0.2, 2.2]],
        ticks=[[-3, 0, 2], [-2, 0, 2], [0,1], [0,1],[0,1], [0,1],[0,1]],
        labels=["$g_{Na}$", "$g_K$", "$g_L$", "$E_L$", "$K_M$", "$Ca$"],
        figsize=(2.2, 2.2),        # ignored; axes are pre-supplied
        diag="hist",
        discrete_indices=[4, 5],
        discrete_offdiag="scatter",
        offdiag="scatter",
        diag_kwargs={"mpl_kwargs": {"color": CLR_MNPE}},
        upper_kwargs={"mpl_kwargs": {"s": 2, "alpha": 0.1, "edgecolor": "none", "color": CLR_MNPE}},
        fig_kwargs={
            "points_colors": [CLR_GT],
            "points_offdiag": {"marker": ".", "markersize": 3, "alpha": 0.8},
            "points_diag": {"linewidth": 1.5},
            "fig_subplots_adjust": {},
            "square_subplots": False,
        },
        fig=fig,
        axes=axes_pp,
    )

    # Restore all positions disturbed by pairplot's internal subplots_adjust call.
    for a, pos in saved_positions.items():
        a.set_position(pos)

    GAP_x = 0.03 * bbox.width
    GAP_y = 0.03 * bbox.height

    for r in range(4):
        for c in range(2, 4):
            pos = axes_pp[r, c].get_position()
            axes_pp[r, c].set_position([pos.x0 + GAP_x, pos.y0, pos.width, pos.height])

    for r in range(2, 4):
        for c in range(4):
            pos = axes_pp[r, c].get_position()
            axes_pp[r, c].set_position([pos.x0, pos.y0 - GAP_y, pos.width, pos.height])

    # Override ticks: pairplot only supports 2 per dim, so set manually.
    col_ticks = [[-3, 0, 3], [-3, 0, 3], [0, 1], [0, 1, 2]]
    col_labels = [None, None, ["off", "on"], ["off", "L", "T"]]

    for r in range(4):
        for c in range(4):
            axes_pp[r, c].set_xticks(col_ticks[c])
            if col_labels[c] is not None:
                axes_pp[r, c].set_xticklabels(col_labels[c])
            if r != c:
                axes_pp[r, c].set_yticks(col_ticks[r])
                if col_labels[r] is not None:
                    axes_pp[r, c].set_yticklabels(col_labels[r])

def make_figure(
    agg: pd.DataFrame,
    obs_data: dict,
    sbc_data: dict,
    n_traces: int = 2,
    n_sbc_continuous: int = 1000,
) -> plt.Figure:
    """Create the main paper figure.

    Parameters
    ----------
    agg : pd.DataFrame
        Budget-sweep metrics for one dropout rate, one row per training budget.
    obs_data : dict
        One observation: 'x_raw_o', 'x_post', 'samples', 'theta_o'.
    sbc_data : dict
        SBC ranks and discrete calibration of the displayed run.
    n_traces : int
        Number of posterior predictive traces to show in panel a.
    """
    ece_expected = eod_expected_under_uniformity(
        num_sbc_runs=sbc_data["ranks_marginals"].shape[0],
    )
    print(f"Expected EoD under uniformity: {ece_expected:.4f}")

    np.random.seed(JITTER_SEED)

    fig = plt.figure(figsize=(6.75, 1.5))

    gs = gridspec.GridSpec(
        2, 3,
        width_ratios=[1, 1, 1.2],
        hspace=0.75,
        wspace=0.45,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[:, 2])

    # Shift bottom row (c, d) down to increase the gap below a/b.
    # Also extend ax_e downward to keep it spanning both rows visually.
    row1_shift = 0.2  # figure fractions; increase to move c/d further down
    for ax in [ax_c, ax_d]:
        p = ax.get_position()
        ax.set_position([p.x0, p.y0 - row1_shift, p.width, p.height])
    p = ax_e.get_position()
    ax_e.set_position([p.x0, p.y0 - row1_shift, p.width, p.height + row1_shift])

    # --- panel a: observed + posterior predictive traces ---
    plot_traces(ax_a, obs_data, n_traces=n_traces)
    add_panel_label(ax_a, "a")

    # --- panel b: MSE vs training size ---
    plot_mse(ax_b, agg)
    add_panel_label(ax_b, "b")

    # --- panel c: SBC rank plot (c1) + discrete calibration (c2) ---
    plot_panel_c(ax_c, sbc_data, n_sbc_continuous=n_sbc_continuous)
    add_panel_label(ax_c, "c")

    # --- panel d: calibration error ---
    plot_calibration(ax_d, agg, ece_expected=ece_expected)
    add_panel_label(ax_d, "d")

    # --- panel e: posterior pairplot ---
    plot_posterior(ax_e, obs_data)
    # ax_e spans 2 rows; place label in figure coords so it sits at the same
    # absolute y as labels a/b rather than 18% above the taller ax_e.
    pos_a = ax_a.get_position()
    pos_e = ax_e.get_position()
    fig.text(
        pos_e.x0 - 0.22 * pos_e.width,
        pos_a.y1 + 0.30 * pos_a.height,
        "e",
        fontsize=8,
        fontweight="bold",
        va="top",
        ha="left",
    )
    return fig


def _with_tensors(data: dict, keys: tuple[str, ...]) -> dict:
    """Hand the plotting code tensors where the evaluation pipeline stored tensors."""
    return {k: torch.from_numpy(v) if k in keys else v for k, v in data.items()}


if __name__ == "__main__":
    inputs = load_figure_inputs(
        figure_inputs.MODEL,
        artifacts=figure_inputs.ARTIFACTS,
        assemble=figure_inputs.assemble,
    )

    agg = pd.DataFrame(inputs["aggregated_results"])
    agg = agg[agg["dropout_rate"] == figure_inputs.DROPOUT_RATE]
    obs_data = _with_tensors(
        inputs["observations"][figure_inputs.FIGURE_OBS], ("samples", "theta_o")
    )
    sbc_data = _with_tensors(
        inputs["calibration"],
        ("ranks_marginals", "ece_discrete", "probs_discrete", "true_vals_discrete"),
    )

    fig = make_figure(agg, obs_data, sbc_data)
    written = save_figure(fig, "figureHH", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {written.name} (and .png) to {written.parent}")
