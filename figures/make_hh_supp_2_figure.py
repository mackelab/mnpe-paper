"""Hodgkin-Huxley supplementary figure (Fig. C-3): observations and posterior predictives.

One row per observation: the observed trace (red), then three posterior-predictive
traces (green) simulated from MNPE posterior samples of the 100k-simulation run at
seed 2. Row order is ``figure_inputs.TRACE_GRID_OBS``, as published.

Usage:
    uv run python figures/make_hh_supp_2_figure.py

Outputs:
    figures/figureHH_supp_2.pdf
    figures/figureHH_supp_2.png
"""

import matplotlib.pyplot as plt
import numpy as np

from mnpe_paper.figure_io import load_figure_inputs, save_figure
from mnpe_paper.models.hh import figure_inputs
from mnpe_paper.plotting.plot_style import COLORS as PS_COLORS

CLR_MNPE = PS_COLORS["MNPE"]
# The supplementary figures draw the ground truth in red rather than the shared
# black CLR_GT; their captions say red.
CLR_GT = "#d62728"

DT = 0.02          # ms per time step
YLIM = (-90, 60)   # mV


def _plot_trace(ax: plt.Axes, t: np.ndarray, trace: np.ndarray, color: str) -> None:
    ax.plot(t, trace, color=color, linewidth=0.5)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(*YLIM)
    ax.set_axis_off()


def _add_scale_bars(
    ax: plt.Axes,
    t: np.ndarray,
    x_bar_ms: float = 1000.0,
    y_bar_mv: float = 50.0,
    pad_x: float = 0.0,
    pad_y: float = 0.0,
) -> None:
    """Draw x and y scale bars in the bottom-right corner of ax.

    Parameters
    ----------
    x_bar_ms : float
        Length of the horizontal scale bar in ms.
    y_bar_mv : float
        Height of the vertical scale bar in mV.
    pad_x, pad_y : float
        Padding from the right/bottom edges in axes fraction units.
    """
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    x_range = x_max - x_min
    y_range = y_max - y_min

    # Anchor point (bottom-right, with padding)
    anchor_x = x_max - pad_x * x_range
    anchor_y = y_min + pad_y * y_range

    # Horizontal bar (time)
    ax.plot(
        [anchor_x - x_bar_ms, anchor_x],
        [anchor_y, anchor_y],
        color="black", linewidth=1.2, solid_capstyle="butt",
        clip_on=False,
    )
    ax.text(
        anchor_x - x_bar_ms / 2,
        anchor_y - 0.06 * y_range,
        f"{int(x_bar_ms)} ms",
        ha="center", va="top", fontsize=5,
    )

    # Vertical bar (voltage) — offset right so it doesn't touch the x-bar
    v_x = anchor_x + 0.06 * x_range
    ax.plot(
        [v_x, v_x],
        [anchor_y, anchor_y + y_bar_mv],
        color="black", linewidth=1.2, solid_capstyle="butt",
        clip_on=False,
    )
    ax.text(
        v_x + 0.02 * x_range,
        anchor_y + y_bar_mv / 2,
        f"{int(y_bar_mv)} mV",
        ha="left", va="center", fontsize=5,
    )



def make_figure(
    observations: list[dict],
    n_samples: int = 3,
    show_ids: bool = False,
) -> plt.Figure:
    """Create supplementary figure 2: grid of traces.

    Layout (one row per observation):
        +--------+----------+----------+----------+
        |   GT   | sample 1 | sample 2 | sample 3 |
        +--------+----------+----------+----------+
        |  ...   |   ...    |   ...    |   ...    |

    Scale bars (1000 ms, 50 mV) are drawn only in the bottom-right panel.
    All axes are hidden.

    Parameters
    ----------
    observations : list of dict
        One observation per row. Each must contain 'x_raw_o' and 'x_post'.
    n_samples : int
        Number of posterior samples to show (columns after GT).
    """
    n_rows = len(observations)
    n_cols = 1 + n_samples  # GT + samples

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6.75, n_rows * 0.7),
        squeeze=False,
    )
    plt.subplots_adjust(hspace=0.08, wspace=0.04)

    for row, obs_data in enumerate(observations):
        x_raw_o = obs_data["x_raw_o"]
        x_post = obs_data["x_post"]
        t = np.arange(len(x_raw_o)) * DT

        # Column 0: ground truth
        _plot_trace(axes[row, 0], t, x_raw_o, CLR_GT)
        if show_ids:
            axes[row, 0].set_title(f"ID {obs_data['obs_id']}", fontsize=5, pad=2)

        # Columns 1..n_samples: posterior samples
        for col in range(n_samples):
            _plot_trace(axes[row, 1 + col], t, x_post[col], CLR_MNPE)

        # Scale bars only in bottom-right panel
        if row == n_rows - 1:
            _add_scale_bars(axes[row, -1], t)

    return fig


if __name__ == "__main__":
    inputs = load_figure_inputs(
        figure_inputs.MODEL,
        artifacts=figure_inputs.ARTIFACTS,
        assemble=figure_inputs.assemble,
    )
    observations = [
        {**inputs["observations"][obs_id], "obs_id": obs_id}
        for obs_id in figure_inputs.TRACE_GRID_OBS
    ]

    fig = make_figure(observations, show_ids=False)
    written = save_figure(fig, "figureHH_supp_2", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {written.name} (and .png) to {written.parent}")
