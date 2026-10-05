"""Panels for Figure 1, the inference method overview.

The figure itself is assembled by hand in Inkscape: the panels below were pasted
into ``figures/fig1.svg``, arranged, and joined with the boxes, arrows, shaded
network backgrounds and labels drawn there, and ``figures/fig1.pdf`` is that
file's PDF export. So this script regenerates the seven matplotlib panels, but
rebuilding the PDF from them is a manual step: the composite holds copies of the
panels, not links to these files.

Panels:
  prior                mixed prior: categorical theta_1, Gaussian (theta_2, theta_3)
  data                 simulations from three discrete simulator families
  condition            one observation
  simulator_schematic  the simulator branching on the discrete parameter
  network1, network2   the discrete and continuous subnetworks
  posterior            mixed posterior

Usage:
    uv run python figures/make_fig1_panels.py

Outputs:
    figures/fig1_<panel>.svg for each panel above
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, FancyArrowPatch
from mpl_toolkits.axisartist.axislines import Subplot
from scipy.stats import multivariate_normal

from mnpe_paper.figure_io import save_figure
from mnpe_paper.plotting.plot_style import CLR_C1, CLR_C2, CLR_GAMMA, CLR_MU1, CLR_MU2

# tight_layout lays the data and condition panels out differently under the macOS
# backend, so pin the one the published panels were drawn with.
plt.switch_backend("agg")

CONT_C1 = CLR_GAMMA  # deep purple
CONT_C2 = CLR_MU1  # medium purple
CONT_C3 = CLR_MU2  # light purple
DISC_C1 = CLR_C1  # deep orange
DISC_C2 = "#ff7043"  # medium orange
DISC_C3 = CLR_C2  # light orange

CONT_CMAP = ListedColormap([CONT_C3, CONT_C2, CONT_C1])
DISC_COLORS = {"A": DISC_C3, "B": DISC_C2, "C": DISC_C1}


def save_panel(fig, name, **savefig_kwargs):
    save_figure(fig, f"fig1_{name}", formats=("svg",), transparent=True, **savefig_kwargs)
    plt.close(fig)


def setup_arrow_axes(ax, xlabel, ylabel=""):
    """Bare axes drawn as two arrows, with no ticks."""
    for direction in ["bottom", "left"]:
        ax.axis[direction].set_axisline_style("-|>")
        ax.axis[direction].set_visible(True)
        ax.axis[direction].major_ticks.set_tick_out(False)
        ax.axis[direction].major_ticks.set_visible(False)
        ax.axis[direction].line.set_color("black")
        ax.axis[direction].line.set_mutation_scale(25)
    for direction in ["right", "top"]:
        ax.axis[direction].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis["bottom"].label.set_text(xlabel)
    ax.axis["bottom"].label.set_visible(True)
    ax.axis["bottom"].label.set_pad(0)
    ax.axis["left"].label.set_text(ylabel)
    ax.axis["left"].label.set_visible(True)
    ax.axis["left"].label.set_pad(0)


def plot_mixed_distribution(title, probabilities, density, bar_alpha=None):
    """Categorical bars for theta_1 beside filled density contours for (theta_2, theta_3)."""
    fig = plt.figure(figsize=(2.2, 0.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 3], wspace=0.5)
    fig.suptitle(title, y=1.2, x=0.46)
    ax1 = Subplot(fig, gs[0, 0])
    ax2 = Subplot(fig, gs[0, 1])
    fig.add_subplot(ax1)
    fig.add_subplot(ax2)

    ax1.bar([0, 1, 2], probabilities, color=[DISC_C3, DISC_C2, DISC_C1],
            edgecolor="black", width=0.7, alpha=bar_alpha)
    setup_arrow_axes(ax1, r"$\theta_1$", "probability")
    ax1.set_xlim(-0.8, 2.8)
    ax1.set_ylim(0, 1.0)

    v, a = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
    Z = density(np.dstack((v, a)))
    z_min, z_max = Z.min(), Z.max()
    # Drop the faintest 10% of the density range so the outer contour stays compact.
    levels = np.linspace(z_min + 0.1 * (z_max - z_min), z_max, 4)
    contour = ax2.contourf(v, a, Z, levels=levels, cmap=CONT_CMAP)
    setup_arrow_axes(ax2, r"$\theta_2$", r"$\theta_3$")

    cbar = fig.colorbar(contour, ax=ax2, shrink=0.7, pad=0.15, aspect=10)
    cbar.set_label("probability", labelpad=2)
    cbar.set_ticklabels([])
    return fig


# ── Mixed prior ──
fig = plot_mixed_distribution(
    "Mixed Prior",
    [0.33, 0.33, 0.33],
    multivariate_normal([0, 0], [[3.5, 0.0], [0.0, 3.5]]).pdf,
)
save_panel(fig, "prior")


# ── Simulations and the observation ──
rng = np.random.default_rng(0)

x = np.linspace(0, 10, 200)
obs_noise_std = 0.02
n_samples = 5  # (theta_2, theta_3) draws per simulator family
theta2_bounds = (1.5, 3.0)  # arrival rate
theta3_bounds = (0.6, 1.4)  # service/disruption scale


def simulator(k, theta2, theta3, x, seed):
    """A queue whose service capacity depends on the discrete family k.

    k      : discrete, simulator family "A", "B" or "C"
    theta2 : continuous, arrival rate
    theta3 : continuous, service/disruption scale
    x      : time grid

    The observation noise is drawn from the module-level ``rng``, not from
    ``seed``, which is how the published panels were drawn.
    """
    sim_rng = np.random.default_rng(seed)
    dt = x[1] - x[0]
    queue = 0
    queue_hist = []
    for i in range(len(x)):
        arrivals = sim_rng.poisson(theta2 * dt)
        if k == "A":
            capacity = sim_rng.poisson((2.5 / theta3) * dt)
        elif k == "B":
            if sim_rng.random() < 0.1:
                capacity = sim_rng.poisson((1.2 / theta3) * dt)
            else:
                capacity = sim_rng.poisson((2.5 / theta3) * dt)
        elif k == "C":
            if i % 20 < 4:
                capacity = sim_rng.poisson((1.4 / theta3) * dt)
            else:
                capacity = sim_rng.poisson((2.8 / theta3) * dt)
        else:
            raise ValueError(f"Unknown simulator family: {k}")
        queue += arrivals
        queue -= min(queue, capacity)
        queue_hist.append(queue)
    y = np.array(queue_hist, dtype=float)
    y += rng.normal(0, obs_noise_std, size=y.shape)
    return y


fig = plt.figure(figsize=(1.0, 0.9))
ax = Subplot(fig, fig.add_gridspec(1, 1)[0, 0])
fig.add_subplot(ax)
for family, color in DISC_COLORS.items():
    theta2_samples = rng.uniform(*theta2_bounds, size=n_samples)
    theta3_samples = rng.uniform(*theta3_bounds, size=n_samples)
    for theta2, theta3 in zip(theta2_samples, theta3_samples):
        seed = int(rng.integers(0, 2**31))
        y = simulator(family, theta2, theta3, x, seed=seed)
        ax.plot(x, y, color=color, linewidth=1.2, alpha=0.7)
setup_arrow_axes(ax, "", "")
fig.tight_layout()
save_panel(fig, "data")

fig = plt.figure(figsize=(1.0, 0.9))
ax = Subplot(fig, fig.add_gridspec(1, 1)[0, 0])
fig.add_subplot(ax)
theta2 = rng.uniform(*theta2_bounds, size=1)[0]
theta3 = rng.uniform(*theta3_bounds, size=1)[0]
seed = int(rng.integers(0, 2**31))
y = simulator("C", theta2, theta3, x, seed=seed)
ax.plot(x, y, color="k", linewidth=1.2, alpha=0.7)
setup_arrow_axes(ax, "", "")
fig.tight_layout()
save_panel(fig, "condition")


# ── Simulator schematic ──
def draw_simulator_schematic(ax, rng):
    """A cluster of states branching into one output pattern per family."""
    ax.set_xlim(0, 4.4)
    ax.set_ylim(0, 3.0)
    ax.set_aspect("equal")
    ax.axis("off")

    dot_r = 0.055
    inner_r = 0.045
    circ_r = 0.20
    cluster = np.array([1.2, 1.5]) + rng.normal(scale=[0.22, 0.32], size=(10, 2))
    branch_start = (1.6, 1.5)
    targets = {"A": (3.0, 2.10), "B": (3.0, 1.50), "C": (3.0, 0.90)}
    patterns = {
        "A": [(3.55, 2.10), (3.85, 2.10), (4.15, 2.10)],
        "B": [(3.50, 1.50), (3.70, 1.50), (4.20, 1.50)],
        "C": [(3.45, 0.90), (3.60, 0.90), (3.75, 0.90), (4.20, 0.90)],
    }

    for cx, cy in cluster:
        ax.add_patch(Circle((cx, cy), dot_r, color="black"))
    for tx, ty in targets.values():
        ax.add_patch(FancyArrowPatch(branch_start, (tx - circ_r, ty), arrowstyle="-|>",
                                     mutation_scale=6, color="black"))
    for key, (tx, ty) in targets.items():
        ax.add_patch(Circle((tx, ty), circ_r, fill=False, ec="black"))
        ax.add_patch(Circle((tx - 0.06, ty), inner_r, color=DISC_COLORS[key]))
        ax.add_patch(Circle((tx + 0.06, ty), inner_r, color=DISC_COLORS[key]))
    for key, pattern in patterns.items():
        for px, py in pattern:
            ax.add_patch(Circle((px, py), dot_r, color=DISC_COLORS[key]))


fig, ax = plt.subplots(figsize=(1.6, 0.8))
draw_simulator_schematic(ax, np.random.default_rng(0))
save_panel(fig, "simulator_schematic", bbox_inches="tight", pad_inches=0.02)


# ── Subnetworks ──
def plot_network(ax, layer_sizes, node_color, x_spacing=1.0, y_spacing=0.5,
                 neuron_radius=0.2, y_center=1.5):
    """A fully connected network schematic, one column of nodes per layer."""
    x_locs = np.linspace(0, x_spacing * (len(layer_sizes) - 1), len(layer_sizes))

    def layer_ys(size):
        return [y_center - ((size - 1) * y_spacing) / 2 + i * y_spacing for i in range(size)]

    for layer_idx in range(len(layer_sizes) - 1):
        x1, x2 = x_locs[layer_idx], x_locs[layer_idx + 1]
        for y1 in layer_ys(layer_sizes[layer_idx]):
            for y2 in layer_ys(layer_sizes[layer_idx + 1]):
                ax.plot([x1 + neuron_radius, x2 - neuron_radius], [y1, y2],
                        "k-", alpha=0.9, zorder=1)
    for x_loc, size in zip(x_locs, layer_sizes):
        for y in layer_ys(size):
            ax.add_patch(Circle((x_loc, y), neuron_radius, facecolor=node_color,
                                edgecolor="black", zorder=2, alpha=0.75))


for name, layer_sizes in [("network1", [2, 4, 4, 1]), ("network2", [3, 4, 4, 2])]:
    fig, ax = plt.subplots(figsize=(1.6, 0.9))
    ax.axis("off")
    plot_network(ax, layer_sizes, plt.cm.Grays(0.3), x_spacing=0.9, y_spacing=0.5)
    save_panel(fig, name)


# ── Mixed posterior ──
means = [(-1.0, 0.0), (1.0, -2.0), (1.0, 2.0)]
covs = [
    [[2, 0.3], [0.3, 0.8]],
    [[2, -0.8], [-0.8, 2]],
    [[2, -0.2], [-0.2, 0.4]],
]
fig = plot_mixed_distribution(
    "Mixed Posterior",
    [0.1, 0.65, 0.25],
    lambda pos: sum(multivariate_normal(m, c).pdf(pos) for m, c in zip(means, covs)),
    bar_alpha=0.75,
)
save_panel(fig, "posterior")

print("Saved figures/fig1_*.svg (the composite fig1.svg/.pdf is assembled by hand)")
