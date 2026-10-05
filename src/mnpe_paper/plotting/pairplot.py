# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import copy
from typing import (
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
    get_args,
)
from warnings import warn

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import FigureBase
from sbi.analysis.plotting_classes import (
    DiagOptions,
    FigOptions,
    OffDiagOptions,
    get_default_diag_kwargs,
    get_default_offdiag_kwargs,
)
from sbi.utils.plotting_helpers import (
    convert_to_list_of_numpy,
    ensure_numpy,
    handle_nan_infs,
    to_list_kwargs,
    to_list_string,
    update,
)
from scipy.stats import gaussian_kde, iqr
from torch import Tensor

UpperLiteral = Literal["hist", "scatter", "contour", "kde"]
LowerLiteral = Literal["hist", "scatter", "contour", "kde"]
DiagLiteral = Literal["hist", "scatter", "kde"]
K = TypeVar("K")
KwargsType = Union[List[Optional[Union[Dict, K]]], Dict, K, None]


def pairplot(
    samples: Union[List[np.ndarray], List[torch.Tensor], np.ndarray, torch.Tensor],
    points: Optional[
        Union[List[np.ndarray], List[torch.Tensor], np.ndarray, torch.Tensor]
    ] = None,
    limits: Optional[Union[List, torch.Tensor]] = None,
    subset: Optional[List[int]] = None,
    upper: Optional[Union[List[Optional[UpperLiteral]], UpperLiteral]] = "hist",
    lower: Optional[Union[List[Optional[LowerLiteral]], LowerLiteral]] = None,
    diag: Optional[Union[List[Optional[DiagLiteral]], DiagLiteral]] = "hist",
    figsize: Tuple = (10, 10),
    labels: Optional[List[str]] = None,
    ticks: Optional[Union[List, torch.Tensor]] = None,
    offdiag: Optional[Union[List[Optional[str]], str]] = None,
    diag_kwargs: KwargsType[DiagOptions] = None,
    upper_kwargs: KwargsType[OffDiagOptions] = None,
    lower_kwargs: KwargsType[OffDiagOptions] = None,
    fig_kwargs: Optional[Union[Dict, FigOptions]] = None,
    fig: Optional[FigureBase] = None,
    axes: Optional[Axes] = None,
    discrete_indices: Optional[List[int]] = None,
    discrete_offdiag: str = "scatter",
) -> Tuple[FigureBase, Axes]:
    """
    Plot samples in a 2D grid showing marginals and pairwise marginals.

    Each of the diagonal plots can be interpreted as a 1D-marginal of the distribution
    that the samples were drawn from. Each upper-diagonal plot can be interpreted as a
    2D-marginal of the distribution.

    Args:
        samples: Samples used to build the histogram.
        points: List of additional points to scatter.
        limits: Array containing the plot xlim for each parameter dimension. If None,
            just use the min and max of the passed samples
        subset: List containing the dimensions to plot. E.g. subset=[1,3] will plot
            plot only the 1st and 3rd dimension but will discard the 0th and 2nd (and,
            if they exist, the 4th, 5th and so on).
        upper: Plotting style for upper diagonal, {hist, scatter, contour, kde,
            None}.
        lower: Plotting style for upper diagonal, {hist, scatter, contour, kde,
            None}.
        diag: Plotting style for diagonal, {hist, scatter, kde}.
        figsize: Size of the entire figure.
        labels: List of strings specifying the names of the parameters.
        ticks: Position of the ticks.
        offdiag: deprecated, use upper instead.
        diag_kwargs: Additional arguments to adjust the diagonal plot,
            see the source code in `KdeDiagOptions`, `HistDiagOptions` or
            `ScatterDiagOptions`.
        upper_kwargs: Additional arguments to adjust the upper diagonal plot,
            see the source code in `KdeOffDiagOptions`, `HistOffDiagOptions`,
            `ScatterOffDiagOptions`, `ContourOffDiagOptions` or `PlotOffDiagOptions`.
        lower_kwargs: Additional arguments to adjust the lower diagonal plot,
            see the source code in `KdeOffDiagOptions`, `HistOffDiagOptions`,
            `ScatterOffDiagOptions`, `ContourOffDiagOptions` or `PlotOffDiagOptions`.
        fig_kwargs: Additional arguments to adjust the overall figure,
            see the source code in `FigOptions`
        fig: matplotlib figure to plot on.
        axes: matplotlib axes corresponding to fig.
        discrete_indices: List of parameter indices that are discrete (e.g. [4, 5]).
            Discrete dimensions use bar charts on the diagonal.
        discrete_offdiag: Plot style for off-diagonal subplots involving discrete
            dimensions. Options: "scatter" (jittered scatter, default) or "heatmap"
            (colored grid showing co-occurrence probabilities). For mixed
            continuous/discrete pairs, scatter with jitter is always used regardless
            of this setting.

    Returns: figure and axis of posterior distribution plot
    """

    upper = _prepare_upper(offdiag, upper)  # type: ignore

    plotting_styles = [
        (diag, DiagLiteral, "diag"),
        (upper, UpperLiteral, "upper"),
        (lower, LowerLiteral, "lower"),
    ]

    for plotting_style, literal, argument_name in plotting_styles:
        _validate_plotting_style(plotting_style, literal, argument_name)

    samples, dim, limits = prepare_for_plot(samples, limits, points)

    # prepare figure kwargs
    fig_kwargs_filled = _prepare_fig_kwargs(fig_kwargs, samples)

    # Prepare diag
    diag_kwargs_filled, diag_func = _prepare_kwargs(
        plot=diag,  # type: ignore
        samples=samples,
        get_plot_funcs=get_diag_funcs,
        get_default_kwargs=get_default_diag_kwargs,
        plot_kwargs=diag_kwargs,
    )

    # Prepare upper
    upper_kwargs_filled, upper_func = _prepare_kwargs(
        plot=upper,  # type: ignore
        samples=samples,
        get_plot_funcs=get_offdiag_funcs,
        get_default_kwargs=get_default_offdiag_kwargs,
        plot_kwargs=upper_kwargs,
    )

    # Prepare lower
    lower_kwargs_filled, lower_func = _prepare_kwargs(
        plot=lower,  # type: ignore
        samples=samples,
        get_plot_funcs=get_offdiag_funcs,
        get_default_kwargs=get_default_offdiag_kwargs,
        plot_kwargs=lower_kwargs,
    )

    return _arrange_grid(
        diag_func,
        upper_func,
        lower_func,
        diag_kwargs_filled,
        upper_kwargs_filled,
        lower_kwargs_filled,
        samples,
        points,
        limits,
        subset,
        figsize,
        labels,
        ticks,
        fig,
        axes,
        fig_kwargs_filled,
        discrete_indices=discrete_indices,
        discrete_offdiag=discrete_offdiag,
    )


# Plotting functions
def plt_bar_1d(
    ax: Axes,
    samples: np.ndarray,
    limits: torch.Tensor,
    diag_kwargs: Dict,
) -> None:
    """Bar chart for discrete 1D marginal showing class probabilities."""
    unique_vals, counts = np.unique(samples, return_counts=True)
    probs = counts / counts.sum()
    bar_width = 0.4
    mpl_kwargs = diag_kwargs.get("mpl_kwargs", {})
    bar_kw = {}
    if "color" in mpl_kwargs:
        bar_kw["color"] = mpl_kwargs["color"]
    if "alpha" in mpl_kwargs:
        bar_kw["alpha"] = mpl_kwargs["alpha"]
    if "edgecolor" in mpl_kwargs:
        bar_kw["edgecolor"] = mpl_kwargs["edgecolor"]
    if "linewidth" in mpl_kwargs:
        bar_kw["linewidth"] = mpl_kwargs["linewidth"]
    ax.bar(unique_vals, probs, width=bar_width, **bar_kw)
    ax.set_ylabel("")


def plt_heatmap_2d(
    ax: Axes,
    samples_col: np.ndarray,
    samples_row: np.ndarray,
    limits_col: Union[torch.Tensor, List],
    limits_row: Union[torch.Tensor, List],
    offdiag_kwargs: Dict,
) -> None:
    """2D heatmap for discrete x discrete dimensions."""
    unique_col = np.unique(samples_col)
    unique_row = np.unique(samples_row)

    # Build count matrix
    counts = np.zeros((len(unique_row), len(unique_col)))
    for i, rv in enumerate(unique_row):
        for j, cv in enumerate(unique_col):
            counts[i, j] = np.sum((samples_row == rv) & (samples_col == cv))
    probs = counts / counts.sum()

    mpl_kwargs = offdiag_kwargs.get("mpl_kwargs", {})
    cmap = mpl_kwargs.get("cmap", "Blues")
    alpha = mpl_kwargs.get("alpha", None)
    imshow_kw = {"cmap": cmap, "interpolation": "nearest"}
    if alpha is not None:
        imshow_kw["alpha"] = alpha
    ax.imshow(
        probs,
        origin="lower",
        aspect="auto",
        extent=(
            unique_col[0] - 0.5,
            unique_col[-1] + 0.5,
            unique_row[0] - 0.5,
            unique_row[-1] + 0.5,
        ),
        **imshow_kw,
    )


def plt_scatter_mixed_2d(
    ax: Axes,
    samples_col: np.ndarray,
    samples_row: np.ndarray,
    limits_col: Union[torch.Tensor, List],
    limits_row: Union[torch.Tensor, List],
    offdiag_kwargs: Dict,
    discrete_col: bool = False,
    discrete_row: bool = False,
) -> None:
    """Scatter plot with jitter on discrete axis for mixed continuous/discrete pairs."""
    jitter_amount = 0.15
    col_plot = samples_col.copy()
    row_plot = samples_row.copy()
    if discrete_col:
        col_plot = col_plot + np.random.uniform(
            -jitter_amount, jitter_amount, size=len(col_plot)
        )
    if discrete_row:
        row_plot = row_plot + np.random.uniform(
            -jitter_amount, jitter_amount, size=len(row_plot)
        )
    mpl_kwargs = offdiag_kwargs.get("mpl_kwargs", {})
    # Build scatter kwargs from the offdiag style, filtering to valid scatter keys
    scatter_kw = {}
    valid_scatter_keys = {
        "s", "c", "color", "marker", "alpha", "linewidths",
        "edgecolors", "edgecolor", "facecolors", "zorder", "label",
        "rasterized",
    }
    for k, v in mpl_kwargs.items():
        if k in valid_scatter_keys:
            scatter_kw[k] = v
    ax.scatter(col_plot, row_plot, **scatter_kw)


def plt_hist_1d(
    ax: Axes,
    samples: np.ndarray,
    limits: torch.Tensor,
    diag_kwargs: Dict,
) -> None:
    """Plot 1D histogram."""
    hist_kwargs = copy.deepcopy(diag_kwargs["mpl_kwargs"])
    if "bins" not in hist_kwargs or hist_kwargs["bins"] is None:
        if diag_kwargs["bin_heuristic"] == "Freedman-Diaconis":
            # The Freedman-Diaconis heuristic
            binsize = 2 * iqr(samples) * len(samples) ** (-1 / 3)
            hist_kwargs["bins"] = np.arange(limits[0], limits[1] + binsize, binsize)
        else:
            # TODO: add more bin heuristics
            pass
    if isinstance(hist_kwargs["bins"], int):
        hist_kwargs["bins"] = np.linspace(limits[0], limits[1], hist_kwargs["bins"])
    ax.hist(samples, **hist_kwargs)


def plt_kde_1d(
    ax: Axes,
    samples: np.ndarray,
    limits: torch.Tensor,
    diag_kwargs: Dict,
) -> None:
    """Run 1D kernel density estimation on samples and plot it on a given axes."""
    density = gaussian_kde(samples, bw_method=diag_kwargs["bw_method"])
    xs = np.linspace(limits[0], limits[1], diag_kwargs["bins"])
    ys = density(xs)
    ax.plot(xs, ys, **diag_kwargs["mpl_kwargs"])


def plt_scatter_1d(
    ax: Axes,
    samples: np.ndarray,
    limits: torch.Tensor,
    diag_kwargs: Dict,
) -> None:
    """Plot vertical lines for each sample. Note: limits are not used."""
    for single_sample in samples:
        ax.axvline(single_sample, **diag_kwargs["mpl_kwargs"])


def plt_hist_2d(
    ax: Axes,
    samples_col: np.ndarray,
    samples_row: np.ndarray,
    limits_col: Union[torch.Tensor, List],
    limits_row: Union[torch.Tensor, List],
    offdiag_kwargs: Dict,
):
    """Plot 2D histogram."""
    hist_kwargs = copy.deepcopy(offdiag_kwargs)

    if isinstance(limits_col, torch.Tensor):
        limits_col = limits_col.tolist()
    if isinstance(limits_row, torch.Tensor):
        limits_row = limits_row.tolist()

    if (
        "bins" not in hist_kwargs["np_hist_kwargs"]
        or hist_kwargs["np_hist_kwargs"]["bins"] is None
    ):
        if hist_kwargs["bin_heuristic"] == "Freedman-Diaconis":
            # The Freedman-Diaconis heuristic applied to each direction
            binsize_col = 2 * iqr(samples_col) * len(samples_col) ** (-1 / 3)
            n_bins_col = int((limits_col[1] - limits_col[0]) / binsize_col)
            binsize_row = 2 * iqr(samples_row) * len(samples_row) ** (-1 / 3)
            n_bins_row = int((limits_row[1] - limits_row[0]) / binsize_row)
            hist_kwargs["np_hist_kwargs"]["bins"] = [n_bins_col, n_bins_row]
        else:
            # TODO: add more bin heuristics
            pass
    hist, xedges, yedges = np.histogram2d(
        samples_col,
        samples_row,
        range=[
            [limits_col[0], limits_col[1]],
            [limits_row[0], limits_row[1]],
        ],
        **hist_kwargs["np_hist_kwargs"],
    )
    ax.imshow(
        hist.T,
        extent=(
            xedges[0],
            xedges[-1],
            yedges[0],
            yedges[-1],
        ),
        **hist_kwargs["mpl_kwargs"],
    )


def plt_kde_2d(
    ax: Axes,
    samples_col: np.ndarray,
    samples_row: np.ndarray,
    limits_col: torch.Tensor,
    limits_row: torch.Tensor,
    offdiag_kwargs: Dict,
) -> None:
    """Run 2D Kernel Density Estimation and plot it on given axis."""
    X, Y, Z = get_kde(samples_col, samples_row, limits_col, limits_row, offdiag_kwargs)

    ax.imshow(
        Z,
        extent=(
            limits_col[0].item(),
            limits_col[1].item(),
            limits_row[0].item(),
            limits_row[1].item(),
        ),
        **offdiag_kwargs["mpl_kwargs"],
    )


def plt_contour_2d(
    ax: Axes,
    samples_col: np.ndarray,
    samples_row: np.ndarray,
    limits_col: torch.Tensor,
    limits_row: torch.Tensor,
    offdiag_kwargs: Dict,
) -> None:
    """2D Contour based on Kernel Density Estimation."""

    X, Y, Z = get_kde(samples_col, samples_row, limits_col, limits_row, offdiag_kwargs)

    ax.contour(
        X,
        Y,
        Z,
        extent=(
            limits_col[0],
            limits_col[1],
            limits_row[0],
            limits_row[1],
        ),
        levels=offdiag_kwargs["levels"],
        **offdiag_kwargs["mpl_kwargs"],
    )


def plt_scatter_2d(
    ax: Axes,
    samples_col: np.ndarray,
    samples_row: np.ndarray,
    limits_col: torch.Tensor,
    limits_row: torch.Tensor,
    offdiag_kwargs: Dict,
) -> None:
    """Scatter plot 2D. Note: limits are not used"""
    ax.scatter(
        samples_col,
        samples_row,
        **offdiag_kwargs["mpl_kwargs"],
    )


def plt_plot_2d(
    ax: Axes,
    samples_col: np.ndarray,
    samples_row: np.ndarray,
    limits_col: torch.Tensor,
    limits_row: torch.Tensor,
    offdiag_kwargs: Dict,
) -> None:
    """Plot 2D trajectory. Note: limits are not used."""

    ax.plot(
        samples_col,
        samples_row,
        **offdiag_kwargs["mpl_kwargs"],
    )


def get_kde(
    samples_col: np.ndarray,
    samples_row: np.ndarray,
    limits_col: torch.Tensor,
    limits_row: torch.Tensor,
    offdiag_kwargs: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2D Kernel Density Estimation."""

    density = gaussian_kde(
        np.array([samples_col, samples_row]),
        bw_method=offdiag_kwargs["bw_method"],
    )
    X, Y = np.meshgrid(
        np.linspace(
            limits_col[0],
            limits_col[1],
            offdiag_kwargs["bins"],
        ),
        np.linspace(
            limits_row[0],
            limits_row[1],
            offdiag_kwargs["bins"],
        ),
    )
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(density(positions).T, X.shape)
    if "percentile" in offdiag_kwargs and "levels" in offdiag_kwargs:
        Z = probs2contours(Z, offdiag_kwargs["levels"])
    else:
        Z = (Z - Z.min()) / (Z.max() - Z.min())
    return X, Y, Z


def get_diag_funcs(
    diag_list: List[Optional[str]],
) -> List[
    Union[
        Callable[
            [
                Axes,
                np.ndarray,
                torch.Tensor,
                Dict,
            ],
            None,
        ],
        None,
    ]
]:
    """make a list of the functions for the diagonal plots."""
    diag_funcs = []
    for diag in diag_list:
        if diag == "hist":
            diag_funcs.append(plt_hist_1d)
        elif diag == "kde":
            diag_funcs.append(plt_kde_1d)
        elif diag == "scatter":
            diag_funcs.append(plt_scatter_1d)
        else:
            diag_funcs.append(None)

    return diag_funcs


def get_offdiag_funcs(
    off_diag_list: List[Optional[str]],
) -> List[
    Union[
        Callable[
            [
                Axes,
                np.ndarray,
                torch.Tensor,
                Dict,
            ],
            None,
        ],
        None,
    ]
]:
    """make a list of the functions for the off-diagonal plots."""
    offdiag_funcs = []
    for offdiag in off_diag_list:
        if offdiag == "hist" or offdiag == "hist2d":
            offdiag_funcs.append(plt_hist_2d)
        elif offdiag == "kde" or offdiag == "kde2d":
            offdiag_funcs.append(plt_kde_2d)
        elif offdiag == "contour" or offdiag == "contourf":
            offdiag_funcs.append(plt_contour_2d)
        elif offdiag == "scatter":
            offdiag_funcs.append(plt_scatter_2d)
        elif offdiag == "plot":
            offdiag_funcs.append(plt_plot_2d)
        else:
            offdiag_funcs.append(None)
    return offdiag_funcs


def _format_subplot(
    ax: Axes,
    current: str,
    limits: Union[List[List[float]], torch.Tensor],
    ticks: Optional[Union[List, torch.Tensor]],
    labels_dim: List[str],
    fig_kwargs: FigOptions,
    row: int,
    col: int,
    dim: int,
    flat: bool,
    excl_lower: bool,
) -> None:
    """
    Format subplot according to fig_kwargs and other arguments
    Args:
        ax: matplotlib axis
        current: str, 'diag','upper' or 'lower'
        limits: list of lists, limits for each dimension
        ticks: list of lists, ticks for each dimension
        labels_dim: list of strings, labels for each dimension
        fig_kwargs: FigOptions dataclass
        row: int, row index
        col: int, column index
        dim: int, number of dimensions
        flat: bool, whether the plot is flat (1 row)
        excl_lower: bool, whether lower triangle is empty

    """

    # Background color
    if (
        current in fig_kwargs.fig_bg_colors
        and fig_kwargs.fig_bg_colors[current] is not None
    ):
        ax.set_facecolor(fig_kwargs.fig_bg_colors[current])
    # Limits
    if isinstance(limits, Tensor):
        assert limits.dim() == 2, "Limits should be a 2D tensor."
        limits = limits.tolist()
    if current == "diag":
        eps = fig_kwargs.x_lim_add_eps
        ax.set_xlim((limits[col][0] - eps, limits[col][1] + eps))
    else:
        ax.set_xlim((limits[col][0], limits[col][1]))

    if current != "diag":
        ax.set_ylim((limits[row][0], limits[row][1]))

    # Ticks
    if ticks is not None:
        ax.set_xticks((ticks[col][0], ticks[col][1]))  # pyright: ignore[reportCallIssue]
        if current != "diag":
            ax.set_yticks((ticks[row][0], ticks[row][1]))  # pyright: ignore[reportCallIssue]

    # make square
    if fig_kwargs.square_subplots:
        ax.set_box_aspect(1)
    # Despine
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_position(("outward", fig_kwargs.despine["offset"]))

    # Formatting axes
    if current == "diag":  # diagonals
        if excl_lower or col == dim - 1 or flat:
            _format_axis(
                ax,
                xhide=False,
                xlabel=labels_dim[col],
                yhide=True,
                tickformatter=fig_kwargs.tickformatter,
            )
        else:
            _format_axis(ax, xhide=True, yhide=True)
    else:  # off-diagonals
        if row == dim - 1:
            _format_axis(
                ax,
                xhide=False,
                xlabel=labels_dim[col],
                yhide=True,
                tickformatter=fig_kwargs.tickformatter,
            )
        else:
            _format_axis(ax, xhide=True, yhide=True)
    if fig_kwargs.tick_labels is not None:
        ax.set_xticklabels((  # pyright: ignore[reportCallIssue]
            str(fig_kwargs.tick_labels[col][0]),
            str(fig_kwargs.tick_labels[col][1]),
        ))


def _format_axis(
    ax: Axes,
    xhide: bool = True,
    yhide: bool = True,
    xlabel: str = "",
    ylabel: str = "",
    tickformatter=None,
) -> Axes:
    """Format axis spines and ticks."""
    for loc in ["right", "top", "left", "bottom"]:
        ax.spines[loc].set_visible(False)
    if xhide:
        ax.set_xlabel("")
        ax.xaxis.set_ticks_position("none")
        ax.xaxis.set_tick_params(labelbottom=False)
    if yhide:
        ax.set_ylabel("")
        ax.yaxis.set_ticks_position("none")
        ax.yaxis.set_tick_params(labelleft=False)
    if not xhide:
        ax.set_xlabel(xlabel)
        ax.xaxis.set_ticks_position("bottom")
        ax.xaxis.set_tick_params(labelbottom=True)
        if tickformatter is not None:
            ax.xaxis.set_major_formatter(tickformatter)
        ax.spines["bottom"].set_visible(True)  # pyright: ignore[reportCallIssue]
    if not yhide:
        ax.set_ylabel(ylabel)
        ax.yaxis.set_ticks_position("left")
        ax.yaxis.set_tick_params(labelleft=True)
        if tickformatter is not None:
            ax.yaxis.set_major_formatter(tickformatter)
        ax.spines["left"].set_visible(True)
    return ax


def probs2contours(
    probs: np.ndarray,
    levels: Union[List, torch.Tensor, np.ndarray],
) -> np.ndarray:
    """Takes an array of probabilities and produces an array of contours at specified
    percentile levels.
    Args:
        probs: Probability array. doesn't have to sum to 1, but it is assumed it
            contains all the mass
        levels: Percentile levels, have to be in [0.0, 1.0]. Specifies contour levels
            that include a given proportion of samples, i.e., 0.1 specifies where the
            top 10% of the density is.
    Returns:
        contours: Array of same shape as probs with percentile labels. Values in output
        array denote labels which percentile bin the probability mass belongs to.

    Example: for levels = [0.1, 0.5], output array will take on values [1.0, 0.5, 0.1],
    where elements labeled "0.1" correspond to the top 10% of the density, "0.5"
    corresponds to between top 50% to 10%, etc.
    """
    # make sure all contour levels are in [0.0, 1.0]
    levels = np.asarray(levels)
    assert np.all(levels <= 1.0) and np.all(levels >= 0.0)

    # flatten probability array
    shape = probs.shape
    probs = probs.flatten()

    # sort probabilities in descending order
    idx_sort = probs.argsort()[::-1]
    idx_unsort = idx_sort.argsort()
    probs = probs[idx_sort]

    # cumulative probabilities
    cum_probs = probs.cumsum()
    cum_probs /= cum_probs[-1]

    # create contours at levels
    contours = np.ones_like(cum_probs)
    levels = np.sort(levels)[::-1]
    for level in levels:
        contours[cum_probs <= level] = level

    # make sure contours have the order and the shape of the original
    # probability array
    contours = np.reshape(contours[idx_unsort], shape)

    return contours


def infer_limits(
    samples: List[np.ndarray],
    dim: int,
    points: Optional[List[np.ndarray]] = None,
    eps: float = 0.1,
) -> List[List[float]]:
    """Infer limits for the plot.

    Args:
        samples: List of set of samples.
        dim: Dimension of the samples.
        points: List of points.
        eps: Relative margin for the limits.
    """
    limits = []
    for d in range(dim):
        # get min and max across all sets of samples
        min_val = min(np.min(sample[:, d]) for sample in samples)
        max_val = max(np.max(sample[:, d]) for sample in samples)
        # include points in the limits
        if points is not None:
            min_val = min(min_val, min(np.min(point[:, d]) for point in points))
            max_val = max(max_val, max(np.max(point[:, d]) for point in points))
        # add margin
        max_min_range = max_val - min_val
        epsilon_range = eps * max_min_range
        limits.append([min_val - epsilon_range, max_val + epsilon_range])
    return limits


def prepare_for_plot(
    samples: Union[List[np.ndarray], List[torch.Tensor], np.ndarray, torch.Tensor],
    limits: Optional[Union[List, torch.Tensor, np.ndarray]] = None,
    points: Optional[
        Union[List[np.ndarray], List[torch.Tensor], np.ndarray, torch.Tensor]
    ] = None,
) -> Tuple[List[np.ndarray], int, torch.Tensor]:
    """
    Ensures correct formatting for samples and limits, and returns dimension
    of the samples.
    """

    samples = convert_to_list_of_numpy(samples)
    if points is not None:
        points = convert_to_list_of_numpy(points)

    samples = handle_nan_infs(samples)

    dim = samples[0].shape[1]

    if limits is None or limits == []:
        limits = infer_limits(samples, dim, points)
    else:
        limits = [limits[0] for _ in range(dim)] if len(limits) == 1 else limits

    limits = torch.as_tensor(limits)
    return samples, dim, limits


def _validate_plotting_style(
    plotting_style: Union[List[Optional[str]], str, None],
    style_literal: str,
    argument_name: str,
):
    """
    This method validates if the plotting style that is passed is None or among the
    allowed plotting styles defined in style_literal.

    Args:
        plotting_style: The plotting style that is selected.
        style_literal: Allowed plotting styles.
        argument_name: The name of the argument being validated.
    """

    allowed_styles = get_args(style_literal)

    if isinstance(plotting_style, list):
        for i, val in enumerate(plotting_style):
            if val is not None and val not in allowed_styles:
                raise ValueError(
                    f"Expected {argument_name} at index {i} to be None or one",
                    f" of {allowed_styles} but got {val}.",
                )
    elif plotting_style is not None and plotting_style not in allowed_styles:
        raise ValueError(
            f"Expected {argument_name} to be None, a list or one",
            f" of {allowed_styles} but got {plotting_style}.",
        )


def _prepare_kwargs(
    plot: Optional[Union[List[Optional[str]], str]],
    samples: Union[List[np.ndarray], List[torch.Tensor], np.ndarray, torch.Tensor],
    get_plot_funcs: Callable,
    get_default_kwargs: Callable,
    plot_kwargs: KwargsType,
) -> Tuple[List[Optional[Dict]], List[Optional[Callable]]]:
    """
    Prepares a list of plotting functions and their corresponding kwargs per sample.

    Args:
        plot: Plot type(s) to use (e.g., 'scatter', 'kde', etc.).
        samples: Input samples to be plotted.
        get_plot_funcs: Function that maps plot names to callable plot functions.
        get_default_kwargs: Function that generates default kwargs for each plot type.
        plot_kwargs: User-specified overrides for the plot kwargs.

    Returns:
        - List of merged keyword arguments for each sample.
        - List of corresponding plot functions.
    """

    plot_list = to_list_string(plot, len(samples))
    plot_kwargs_list = to_list_kwargs(plot_kwargs, len(samples))
    plot_func = get_plot_funcs(plot_list)
    plot_kwargs_filled = []
    for i, (plot_i, plot_kwargs_i) in enumerate(
        zip(plot_list, plot_kwargs_list, strict=False)
    ):
        plot_kwarg_filled_i = get_default_kwargs(plot_i, i)
        # update the defaults dictionary with user provided values
        plot_kwarg_filled_i = update(plot_kwarg_filled_i, plot_kwargs_i)
        plot_kwargs_filled.append(plot_kwarg_filled_i)

    return plot_kwargs_filled, plot_func


def _prepare_fig_kwargs(
    fig_kwargs: Optional[Union[Dict, FigOptions]],
    samples: Union[List[np.ndarray], List[torch.Tensor], np.ndarray, torch.Tensor],
) -> FigOptions:
    """
    Converts user-provided figure keyword arguments into a FigOptions dataclass.

    This function ensures that figure configuration is consistently represented as a
    ``FigOptions`` dataclass, regardless of whether the user supplies no options,
    a dictionary of keyword arguments, or an existing ``FigOptions`` instance.

    Args:
        fig_kwargs: User-provided figure keyword arguments.
        samples: Input samples to be plotted.

    Raises:
        ValueError: If the number of sample labels is less than
            the number of samples when a legend is specified.

    Returns:
        FigOptions dataclass.
    """

    if fig_kwargs is None:
        fig_kwargs = FigOptions()
    elif isinstance(fig_kwargs, dict):
        fig_kwargs = FigOptions(**fig_kwargs)

    if fig_kwargs.legend and len(fig_kwargs.samples_labels) < len(samples):
        raise ValueError("Provide at least as many labels as samples.")

    return fig_kwargs


def _prepare_upper(
    offdiag: Optional[Union[List[Optional[str]], str]],
    upper: Optional[Union[List[Optional[str]], str]],
) -> Optional[Union[List[Optional[str]], str]]:
    """
    Handles deprecated 'offdiag' argument by falling back to 'upper'.

    If 'offdiag' is provided, a deprecation warning is issued and its value is used.
    Otherwise, 'upper' is returned.

    Args:
        offdiag: Deprecated off-diagonal plot specifier.
        upper: New argument to specify upper triangle plot behavior.

    Returns:
        Resolved plot specification.
    """

    if offdiag is not None:
        if upper != "hist" and offdiag != upper:
            raise ValueError(
                "You have passed mismatching values to both upper and offdiag"
                " arguments. offdiag is deprecated, use upper instead."
            )
        warn("offdiag is deprecated, use upper or lower instead.", stacklevel=2)
    return offdiag or upper


def _arrange_grid(
    diag_funcs: List[Optional[Callable]],
    upper_funcs: List[Optional[Callable]],
    lower_funcs: List[Optional[Callable]],
    diag_kwargs: List[Optional[Dict]],
    upper_kwargs: List[Optional[Dict]],
    lower_kwargs: List[Optional[Dict]],
    samples: List[np.ndarray],
    points: Optional[
        Union[List[np.ndarray], List[torch.Tensor], np.ndarray, torch.Tensor]
    ],
    limits: torch.Tensor,
    subset: Optional[List[int]],
    figsize: Optional[Tuple],
    labels: Optional[List[str]],
    ticks: Optional[Union[List, torch.Tensor]],
    fig: Optional[FigureBase],
    axes: Optional[Axes],
    fig_kwargs: FigOptions,
    discrete_indices: Optional[List[int]] = None,
    discrete_offdiag: str = "scatter",
) -> Tuple[FigureBase, Axes]:
    """
    Arranges the plots for any function that plots parameters either in a row of 1D
    marginals or a pairplot setting.

    Args:
        diag_funcs: List of plotting function that will be executed for the diagonal
            elements of the plot (or the columns of a row of 1D marginals).
        upper_funcs: List of plotting function that will be executed for the
            upper-diagonal elements of the plot. None if we are in a 1D setting.
        lower_funcs: List of plotting function that will be executed for the
            lower-diagonal elements of the plot. None if we are in a 1D setting.
        diag_kwargs: Additional arguments to adjust the diagonal plot,
            see the source code in `_get_default_diag_kwarg()`
        upper_kwargs: Additional arguments to adjust the upper diagonal plot,
            see the source code in `_get_default_offdiag_kwarg()`
        lower_kwargs: Additional arguments to adjust the lower diagonal plot,
            see the source code in `_get_default_offdiag_kwarg()`
        samples: List of samples given to the plotting functions
        points: List of additional points to scatter.
        limits: Limits for each dimension / axis.
        subset: List containing the dimensions to plot. E.g. subset=[1,3] will plot
            plot only the 1st and 3rd dimension
        figsize: Size of the entire figure.
        labels: List of strings specifying the names of the parameters.
        ticks: Position of the ticks.
        fig: matplotlib figure to plot on.
        axes: matplotlib axes corresponding to fig.
        fig_kwargs: Additional arguments to adjust the overall figure,
            see the source code in `_get_default_fig_kwargs()`

    Returns:
        Fig: matplotlib figure
        Axes: matplotlib axes
    """
    dim = samples[0].shape[1]
    # Prepare points
    if points is None:
        points = []
    if not isinstance(points, list):
        points = ensure_numpy(points)  # type: ignore
        points = [points]
    points = [np.atleast_2d(p) for p in points]
    points = [np.atleast_2d(ensure_numpy(p)) for p in points]
    # TODO: add asserts checking compatibility of dimensions

    # Prepare labels
    if labels == [] or labels is None:
        labels = ["dim {}".format(i + 1) for i in range(dim)]

    # Prepare ticks
    if ticks is not None:
        if len(ticks) == 1:
            ticks = [ticks[0] for _ in range(dim)]
        elif ticks == []:
            ticks = None

    # Figure out if we subset the plot
    if subset is None:
        rows = cols = dim
        subset = [i for i in range(dim)]
    else:
        if isinstance(subset, int):
            subset = [subset]
        elif isinstance(subset, list):
            pass
        else:
            raise NotImplementedError
        rows = cols = len(subset)

    # check which subplots are empty
    excl_lower = all(v is None for v in lower_funcs)
    excl_upper = all(v is None for v in upper_funcs)
    excl_diag = all(v is None for v in diag_funcs)
    flat = excl_lower and excl_upper
    one_dim = dim == 1
    # select the subset of rows and cols to be plotted
    if flat:
        rows = 1
        subset_rows = [1]
    else:
        subset_rows = subset
    subset_cols = subset

    # Create fig and axes if they were not passed.
    if fig is None or axes is None:
        fig, axes = plt.subplots(rows, cols, figsize=figsize, **fig_kwargs.subplots)  # pyright: ignore reportAssignmenttype
    else:
        assert axes.shape == (  # pyright: ignore reportAttributeAccessIssue
            rows,
            cols,
        ), f"Passed axes must match subplot shape: {rows, cols}."

    # Style figure
    fig.subplots_adjust(**fig_kwargs.fig_subplots_adjust)
    fig.suptitle(fig_kwargs.title or "", **fig_kwargs.title_format)

    # Main Loop through all subplots, style and create the figures
    for row_idx, row in enumerate(subset_rows):
        for col_idx, col in enumerate(subset_cols):
            if flat or row == col:
                current = "diag"
            elif row < col:
                current = "upper"
            else:
                current = "lower"

            if one_dim:
                ax = axes  # pyright: ignore reportIndexIssue
            elif flat:
                ax = axes[col_idx]  # pyright: ignore reportIndexIssue
            else:
                ax = axes[row_idx, col_idx]  # pyright: ignore reportIndexIssue
            # Diagonals
            _format_subplot(
                ax,  # pyright: ignore reportArgumentType
                current,
                limits,
                ticks,
                labels,
                fig_kwargs,
                row,
                col,
                dim,
                flat,
                excl_lower,
            )
            _disc = set(discrete_indices) if discrete_indices else set()
            if current == "diag":
                if excl_diag:
                    ax.axis("off")  # pyright: ignore reportOptionalMemberAccess
                else:
                    for sample_ind, sample in enumerate(samples):
                        if row in _disc:
                            plt_bar_1d(
                                ax, sample[:, row], limits[row], diag_kwargs[sample_ind]
                            )
                        else:
                            diag_f = diag_funcs[sample_ind]
                            if callable(diag_f):  # is callable:
                                diag_f(
                                    ax, sample[:, row], limits[row], diag_kwargs[sample_ind]
                                )

                if len(points) > 0:
                    extent = ax.get_ylim()  # pyright: ignore reportOptionalMemberAccess
                    for n, v in enumerate(points):
                        ax.plot(  # pyright: ignore reportOptionalMemberAccess
                            [v[:, col], v[:, col]],
                            extent,
                            color=fig_kwargs.points_colors[n],
                            **fig_kwargs.points_diag,
                            label=fig_kwargs.points_labels[n],
                        )
                if fig_kwargs.legend and col == 0:
                    ax.legend(**fig_kwargs.legend_kwargs)  # pyright: ignore reportOptionalMemberAccess

            # Off-diagonals

            # upper
            elif current == "upper":
                if excl_upper:
                    ax.axis("off")  # pyright: ignore reportOptionalMemberAccess
                else:
                    col_disc = col in _disc
                    row_disc = row in _disc
                    for sample_ind, sample in enumerate(samples):
                        if col_disc and row_disc:
                            if discrete_offdiag == "heatmap":
                                plt_heatmap_2d(
                                    ax,
                                    sample[:, col],
                                    sample[:, row],
                                    limits[col],
                                    limits[row],
                                    upper_kwargs[sample_ind],
                                )
                            else:
                                plt_scatter_mixed_2d(
                                    ax,
                                    sample[:, col],
                                    sample[:, row],
                                    limits[col],
                                    limits[row],
                                    upper_kwargs[sample_ind],
                                    discrete_col=True,
                                    discrete_row=True,
                                )
                        elif col_disc or row_disc:
                            plt_scatter_mixed_2d(
                                ax,
                                sample[:, col],
                                sample[:, row],
                                limits[col],
                                limits[row],
                                upper_kwargs[sample_ind],
                                discrete_col=col_disc,
                                discrete_row=row_disc,
                            )
                        else:
                            upper_f = upper_funcs[sample_ind]
                            if callable(upper_f):
                                upper_f(
                                    ax,
                                    sample[:, col],
                                    sample[:, row],
                                    limits[col],
                                    limits[row],
                                    upper_kwargs[sample_ind],
                                )
                    if len(points) > 0:
                        for n, v in enumerate(points):
                            ax.plot(  # pyright: ignore reportOptionalMemberAccess
                                v[:, col],
                                v[:, row],
                                color=fig_kwargs.points_colors[n],
                                **fig_kwargs.points_offdiag,
                            )
            # lower
            elif current == "lower":
                if excl_lower:
                    ax.axis("off")  # pyright: ignore reportOptionalMemberAccess
                else:
                    col_disc = col in _disc
                    row_disc = row in _disc
                    for sample_ind, sample in enumerate(samples):
                        if col_disc and row_disc:
                            if discrete_offdiag == "heatmap":
                                plt_heatmap_2d(
                                    ax,
                                    sample[:, row],
                                    sample[:, col],
                                    limits[row],
                                    limits[col],
                                    lower_kwargs[sample_ind],
                                )
                            else:
                                plt_scatter_mixed_2d(
                                    ax,
                                    sample[:, row],
                                    sample[:, col],
                                    limits[row],
                                    limits[col],
                                    lower_kwargs[sample_ind],
                                    discrete_col=True,
                                    discrete_row=True,
                                )
                        elif col_disc or row_disc:
                            plt_scatter_mixed_2d(
                                ax,
                                sample[:, row],
                                sample[:, col],
                                limits[row],
                                limits[col],
                                lower_kwargs[sample_ind],
                                discrete_col=row_disc,
                                discrete_row=col_disc,
                            )
                        else:
                            lower_f = lower_funcs[sample_ind]
                            if callable(lower_f):
                                lower_f(
                                    ax,
                                    sample[:, row],
                                    sample[:, col],
                                    limits[row],
                                    limits[col],
                                    lower_kwargs[sample_ind],
                                )
                    if len(points) > 0:
                        for n, v in enumerate(points):
                            ax.plot(  # pyright: ignore reportOptionalMemberAccess
                                v[:, col],
                                v[:, row],
                                color=fig_kwargs.points_colors[n],
                                **fig_kwargs.points_offdiag,
                            )
    # Add dots if we subset
    if len(subset) < dim:
        if flat:
            ax = axes[len(subset) - 1]  # pyright: ignore[reportIndexIssue, reportOptionalSubscript]
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            text_kwargs = {"fontsize": plt.rcParams["font.size"] * 2.0}  # pyright: ignore[reportOptionalOperand]
            ax.text(x1 + (x1 - x0) / 8.0, (y0 + y1) / 2.0, "...", **text_kwargs)
        else:
            for row in range(len(subset)):
                ax = axes[row, len(subset) - 1]  # pyright: ignore[reportIndexIssue, reportOptionalSubscript]
                x0, x1 = ax.get_xlim()
                y0, y1 = ax.get_ylim()
                text_kwargs = {"fontsize": plt.rcParams["font.size"] * 2.0}  # pyright: ignore[reportOptionalOperand]
                ax.text(x1 + (x1 - x0) / 8.0, (y0 + y1) / 2.0, "...", **text_kwargs)
                if row == len(subset) - 1:
                    ax.text(
                        x1 + (x1 - x0) / 12.0,
                        y0 - (y1 - y0) / 1.5,
                        "...",
                        rotation=-45,
                        **text_kwargs,
                    )

    return fig, axes  # pyright: ignore[reportReturnType]














# Diagnostics for hypothesis tests






# Customized plotting functions for LC2ST






# TO BE DEPRECATED
# ----------------








