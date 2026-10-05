"""Shared plot style for paper figures.

Loads mackelab.matplotlibrc for all matplotlib defaults (fonts, ticks, spines,
line widths, etc.). This file only adds colors and the despine helper.

Font sizes are derived from rcParams so mackelab.matplotlibrc is the single
source of truth. FS_PANEL has no rcParam equivalent and is set here.
"""

from pathlib import Path

import matplotlib.pyplot as plt

# ── Load mackelab configuration (serif 7.5pt, spines, ticks, line widths) ──
_RC_PATH = Path(__file__).resolve().parent / "mackelab.matplotlibrc"
plt.style.use(str(_RC_PATH))

# ── Font-size constants derived from rcParams (single source of truth) ──
FS_LABEL = plt.rcParams["axes.labelsize"]
FS_TICK = plt.rcParams["xtick.labelsize"]
FS_LEGEND = plt.rcParams["legend.fontsize"]
# Panel labels ("a", "b", ...), as in the HH figure
FS_PANEL = 8

# ── Colors ──
COLORS = {
    "MNPE": "#2ca02c",       # green
    "MNPE_light": "#61c44e", # light green (second posterior predictive sample)
    "Marginalized": "#1f77b4",  # blue (MCMC reference)
}
CLR_GT = "black"  # ground truth

# Parameter color families
CLR_GAMMA = "#5e35b1"  # deep purple
CLR_MU1 = "#9575cd"  # medium purple
CLR_MU2 = "#b39ddb"  # light purple
CLR_C1 = "#d84315"  # deep orange
CLR_C2 = "#ff8a65"  # light orange

# Coal mining parameter colors
CLR_EARLY = "#5e35b1"  # deep purple (same family as CLR_GAMMA)
CLR_LATE = "#9575cd"  # medium purple (same family as CLR_MU1)
CLR_SWITCH = "#d84315"  # deep orange (same family as CLR_C1)

# Hodgkin-Huxley parameter colors
# continuous
CLR_CONTINOUS = "#3b197e"
CLR_GNA = CLR_GAMMA  # deep purple
CLR_GK = CLR_MU1    # medium purple
CLR_GL = CLR_MU2    # light purple
CLR_EL = "#c1b3dc"
# discrete
CLR_DISCRETE = "#ef6400"
CLR_KM = CLR_C1  # deep orange
CLR_CA = CLR_C2  # light orange


def despine(ax, offset=0):
    """Remove top and right spines.

    If offset > 0, the left spine is pushed outward to create a small gap at
    the corner where the axes meet.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if offset:
        ax.spines["left"].set_position(("outward", offset))
