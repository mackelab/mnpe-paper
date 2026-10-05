"""Hodgkin-Huxley supplementary figure (Fig. C-2): full 6x6 posterior pairplot.

All four continuous and both discrete parameters for observation 284, the
observation of the main figure, from the 100k-simulation run at seed 2. The
pairplot's jitter on the discrete axes is random, so it is seeded here to make
rebuilds byte-identical.

Usage:
    uv run python figures/make_hh_supp_1_figure.py

Outputs:
    figures/figureHH_supp_1.pdf
    figures/figureHH_supp_1.png
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from mnpe_paper.figure_io import load_figure_inputs, save_figure
from mnpe_paper.models.hh import figure_inputs
from mnpe_paper.plotting.pairplot import pairplot
from mnpe_paper.plotting.plot_style import COLORS as PS_COLORS

CLR_MNPE = PS_COLORS["MNPE"]
# The supplementary figures draw the ground truth in red rather than the shared
# black CLR_GT; their captions say red.
CLR_GT = "#d62728"

JITTER_SEED = 0


def make_figure(obs_data: dict) -> plt.Figure:
    """Create supplementary figure 1: full 6×6 posterior pairplot.

    Parameters
    ----------
    obs_data : dict
        Must contain keys 'samples' and 'theta_o'.
    """
    samples = obs_data["samples"]
    theta_o = obs_data["theta_o"]

    np.random.seed(JITTER_SEED)
    fig, axes_pp = pairplot(
        samples,
        points=theta_o,
        limits=[[-4, 4]] * 4 + [[-0.2, 1.2], [-0.2, 2.2]],
        ticks=[[-3, 0, 3], [-3, 0, 3], [-3, 0, 3], [-3, 0, 3], [0, 1], [0, 1, 2]],
        labels=["$g_{Na}$", "$g_K$", "$g_L$", "$E_L$", "$K_M$", "$Ca$"],
        figsize=(3.25, 3.25),
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
            "square_subplots": False,
        },
    )

    # Override ticks manually (same approach as the main figure).
    col_ticks = [[-3, 0, 3], [-3, 0, 3], [-3, 0, 3], [-3, 0, 3], [0, 1], [0, 1, 2]]
    col_labels = [None, None, None, None, ["off", "on"], ["off", "L", "T"]]

    for r in range(6):
        for c in range(6):
            axes_pp[r, c].set_xticks(col_ticks[c])
            if col_labels[c] is not None:
                axes_pp[r, c].set_xticklabels(col_labels[c])
            if r != c:
                axes_pp[r, c].set_yticks(col_ticks[r])
                if col_labels[r] is not None:
                    axes_pp[r, c].set_yticklabels(col_labels[r])
    return fig


if __name__ == "__main__":
    inputs = load_figure_inputs(
        figure_inputs.MODEL,
        artifacts=figure_inputs.ARTIFACTS,
        assemble=figure_inputs.assemble,
    )
    obs = inputs["observations"][figure_inputs.FIGURE_OBS]
    obs_data = {k: torch.from_numpy(obs[k]) for k in ("samples", "theta_o")}

    fig = make_figure(obs_data)
    written = save_figure(fig, "figureHH_supp_1", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {written.name} (and .png) to {written.parent}")
