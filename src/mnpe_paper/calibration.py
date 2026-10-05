"""Calibration utilities for MNPE discrete parameters."""

from typing import List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from torch import Tensor


def run_discrete_calibration(
    thetas: Tensor,
    xs: Tensor,
    posterior: NeuralPosterior,
    discrete_indices: List[int],
    num_bins: int = 10,
    method: str = "top_label",
    probability_type: Literal["marginal", "factor"] = "marginal",
    max_exact_states: int = 4096,
    num_marginal_samples: int = 4096,
    marginal_seed: int = 0,
    marginal_batch_size: int = 128,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute calibration data for discrete parameters directly from the MNPE network.

    Extracts probabilities from the CategoricalMADE sub-network, then computes
    the Expected Calibration Error (ECE) per discrete parameter.

    The underlying network is a MADE (Masked Autoregressive Density Estimator), so the
    joint P(theta_disc | x) factorises autoregressively:

        P(theta_0, theta_1, ... | x) = P(theta_0 | x) * P(theta_1 | theta_0, x) * ...

    By default, this function returns genuine one-dimensional marginals
    ``P(theta_i | x)``, summing the autoregressive factors over the preceding
    variables. Marginalizing variable ``i`` costs ``prod(num_categories[:i])``
    prefixes, so the exact budget is spent by the leading variables and the
    choice of strategy is made **per variable**: variables whose prefix count
    fits ``max_exact_states`` are summed exactly, and the remaining tail uses
    Rao-Blackwellized ancestral Monte Carlo seeded with exact draws from the
    enumerated prefix distribution. The leading marginals therefore carry no
    sampling error even when the joint is far too large to enumerate.

    Set ``probability_type="factor"`` for the teacher-forced
    diagnostic. In that case, the returned probabilities are ``P(theta_0 | x)``
    for the first variable and ``P(theta_i | theta_{<i,true}, x)`` thereafter.

    Args:
        thetas: Ground-truth parameters, shape (N, D).
        xs: Observations corresponding to thetas, shape (N, obs_dim).
        posterior: A posterior obtained from MNPE.
        discrete_indices: Indices of the discrete parameters in theta.
        num_bins: Number of bins for the calibration curve.
        method: ``"top_label"`` or ``"classwise"``. See ``compute_ece`` for details.
        probability_type: ``"marginal"`` for per-variable marginal posterior
            probabilities or ``"factor"`` for teacher-forced autoregressive
            factors.
        max_exact_states: Maximum number of preceding discrete configurations
            to enumerate for any single variable. Applied per variable, so a
            budget of 32768 buys exact marginals for the first 16 binary
            variables regardless of how many follow them.
        num_marginal_samples: Number of ancestral prefix samples used for the
            variables whose prefix count exceeds ``max_exact_states``.
        marginal_seed: Local random seed for Monte Carlo marginalization.
        marginal_batch_size: Number of observations marginalized together.
            Limits memory use when enumerating or sampling many prefixes.

    Returns:
        ece: ECE for each discrete parameter, shape (num_discrete,).
        probs: Predicted probabilities, shape (N, num_discrete, num_categories).
        true_vals: Ground-truth discrete values as 0-indexed category indices,
            shape (N, num_discrete). Mapped from original labels if needed.
    """
    net = posterior.potential_fn.posterior_estimator
    categorical_made = net.discrete_net.net

    true_vals = thetas[:, discrete_indices]
    true_disc = true_vals.float()
    if categorical_made._needs_mapping:
        true_disc = categorical_made._map_values_to_indices(true_disc)

    with torch.no_grad():
        if probability_type == "factor":
            logits = categorical_made.forward(true_disc, condition=xs)
            logits = logits.reshape(
                xs.shape[0], len(discrete_indices), categorical_made.max_num_categories
            )
            probs = torch.softmax(logits, dim=-1)
        elif probability_type == "marginal":
            probs = _marginal_discrete_probs(
                categorical_made,
                xs,
                max_exact_states=max_exact_states,
                num_samples=num_marginal_samples,
                seed=marginal_seed,
                batch_size=marginal_batch_size,
            )
        else:
            raise ValueError(
                f"Unknown probability_type '{probability_type}', "
                "use 'marginal' or 'factor'"
            )

    true_disc = true_disc.long()
    ece = compute_ece(true_disc, probs, num_bins, method=method)
    return ece, probs, true_disc


def exact_marginal_variable_count(
    num_categories: List[int], max_exact_states: int
) -> int:
    """How many leading variables exact enumeration can afford.

    Marginalizing variable ``i`` enumerates ``prod(num_categories[:i])``
    prefixes, so the cost is wildly uneven across variables: variable 0 needs a
    single forward pass and variable 31 needs 2**31. The exact budget is
    therefore spent by the *leading* variables, which is why the switch to
    sampling belongs per-variable rather than all-or-nothing.
    """
    states = 1
    count = 0
    for c in num_categories:
        if states > max_exact_states:
            break
        count += 1
        states *= c
    return count


def _marginal_discrete_probs(
    categorical_made,
    xs: Tensor,
    max_exact_states: int,
    num_samples: int,
    seed: int,
    batch_size: int,
) -> Tensor:
    """Return ``q(theta_i | x)`` for every variable in a CategoricalMADE."""
    if batch_size < 1:
        raise ValueError("marginal_batch_size must be positive")

    max_categories = categorical_made.max_num_categories
    num_categories = [
        int(c) for c in categorical_made.num_categories_per_var.detach().cpu()
    ]

    max_prefix_states = 1
    for c in num_categories[:-1]:
        max_prefix_states *= c

    num_exact = exact_marginal_variable_count(num_categories, max_exact_states)

    chunks = []
    for start in range(0, xs.shape[0], batch_size):
        xs_chunk = xs[start : start + batch_size]
        if max_prefix_states <= max_exact_states:
            probs = _marginal_discrete_probs_exact(
                categorical_made, xs_chunk, num_categories, max_categories
            )
        else:
            probs = _marginal_discrete_probs_hybrid(
                categorical_made,
                xs_chunk,
                num_categories,
                max_categories,
                num_exact=num_exact,
                num_samples=num_samples,
                seed=seed + start,
            )
        chunks.append(probs)
    return torch.cat(chunks, dim=0)


def _marginal_discrete_probs_exact(
    categorical_made,
    xs: Tensor,
    num_categories: List[int],
    max_categories: int,
) -> Tensor:
    """Exactly sum over all autoregressive prefixes."""
    batch_size = xs.shape[0]
    num_variables = len(num_categories)
    dtype = xs.dtype
    device = xs.device
    marginals = torch.zeros(
        batch_size, num_variables, max_categories, dtype=dtype, device=device
    )

    prefixes = torch.zeros(
        1, batch_size, num_variables, dtype=dtype, device=device
    )
    prefix_weights = torch.ones(1, batch_size, dtype=dtype, device=device)

    for i, n_categories_i in enumerate(num_categories):
        num_states = prefixes.shape[0]
        flat_prefixes = prefixes.reshape(num_states * batch_size, num_variables)
        condition = xs.repeat(num_states, *([1] * (xs.ndim - 1)))
        logits = categorical_made.forward(flat_prefixes, condition=condition)
        conditional = torch.softmax(
            logits.reshape(
                num_states,
                batch_size,
                num_variables,
                max_categories,
            )[:, :, i, :],
            dim=-1,
        )
        marginals[:, i, :] = (prefix_weights.unsqueeze(-1) * conditional).sum(0)

        if i == num_variables - 1:
            continue

        prefixes = (
            prefixes.unsqueeze(1)
            .expand(num_states, n_categories_i, batch_size, num_variables)
            .reshape(num_states * n_categories_i, batch_size, num_variables)
            .clone()
        )
        category_values = (
            torch.arange(n_categories_i, dtype=dtype, device=device)
            .repeat(num_states)
            .unsqueeze(1)
        )
        prefixes[:, :, i] = category_values
        prefix_weights = (
            prefix_weights.unsqueeze(2)
            * conditional[:, :, :n_categories_i]
        ).permute(0, 2, 1).reshape(num_states * n_categories_i, batch_size)

    return marginals


def _marginal_discrete_probs_hybrid(
    categorical_made,
    xs: Tensor,
    num_categories: List[int],
    max_categories: int,
    num_exact: int,
    num_samples: int,
    seed: int,
) -> Tensor:
    """Exact marginals for the leading variables, ancestral MC for the tail.

    Switching exact-versus-sampled on the *joint* prefix count sends every
    variable to Monte Carlo the moment the joint exceeds the budget, including
    variable 0 — whose marginal is a single forward pass and needs no
    marginalization at all. At D=32 that would make all 32 marginals noisy when
    16 of them are exactly affordable under the same cap, and cost more compute
    than the exact route for those 16.

    This walks the enumeration as far as the cap allows, draws ``num_samples``
    prefixes from that exact distribution, and continues ancestrally. Variables
    below ``num_exact`` are then noise-free, and the tail estimator stays
    unbiased because the particles are exact draws from the true prefix
    marginal rather than an approximation of it.

    The enumeration loop below deliberately restates
    ``_marginal_discrete_probs_exact`` instead of calling it, so that routine
    stays independent of the mixed case; the price is that a fix to the loop has
    to be applied in both.
    """
    if not 1 <= num_exact < len(num_categories):
        raise ValueError(
            f"num_exact must be in [1, {len(num_categories)}), got {num_exact}; "
            "max_exact_states must be at least 1"
        )

    batch_size = xs.shape[0]
    num_variables = len(num_categories)
    dtype = xs.dtype
    device = xs.device
    marginals = torch.zeros(
        batch_size, num_variables, max_categories, dtype=dtype, device=device
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    # --- exact enumeration over variables [0, num_exact) ---
    prefixes = torch.zeros(1, batch_size, num_variables, dtype=dtype, device=device)
    prefix_weights = torch.ones(1, batch_size, dtype=dtype, device=device)
    conditional = None
    for i in range(num_exact):
        num_states = prefixes.shape[0]
        logits = categorical_made.forward(
            prefixes.reshape(num_states * batch_size, num_variables),
            condition=xs.repeat(num_states, *([1] * (xs.ndim - 1))),
        )
        conditional = torch.softmax(
            logits.reshape(
                num_states, batch_size, num_variables, max_categories
            )[:, :, i, :],
            dim=-1,
        )
        marginals[:, i, :] = (prefix_weights.unsqueeze(-1) * conditional).sum(0)

        # The final exact variable is not expanded: its conditional feeds the
        # transition draw below instead of another enumeration level.
        if i == num_exact - 1:
            break
        n_categories_i = num_categories[i]
        prefixes = (
            prefixes.unsqueeze(1)
            .expand(num_states, n_categories_i, batch_size, num_variables)
            .reshape(num_states * n_categories_i, batch_size, num_variables)
            .clone()
        )
        category_values = (
            torch.arange(n_categories_i, dtype=dtype, device=device)
            .repeat(num_states)
            .unsqueeze(1)
        )
        prefixes[:, :, i] = category_values
        prefix_weights = (
            prefix_weights.unsqueeze(2) * conditional[:, :, :n_categories_i]
        ).permute(0, 2, 1).reshape(num_states * n_categories_i, batch_size)

    # --- transition: exact draws from the enumerated prefix distribution ---
    last_exact = num_exact - 1
    observations = torch.arange(batch_size, device=device).unsqueeze(0)
    state_index = torch.multinomial(
        prefix_weights.transpose(0, 1),
        num_samples,
        replacement=True,
        generator=generator,
    ).transpose(0, 1)
    particles = prefixes[state_index, observations, :].clone()
    tail_probs = conditional[state_index, observations, :][
        :, :, : num_categories[last_exact]
    ]
    draws = torch.multinomial(
        tail_probs.reshape(num_samples * batch_size, num_categories[last_exact]),
        num_samples=1,
        generator=generator,
    )
    particles[:, :, last_exact] = draws.reshape(num_samples, batch_size).to(dtype)

    # --- ancestral Monte Carlo over the remaining variables ---
    condition = xs.repeat(num_samples, *([1] * (xs.ndim - 1)))
    for i in range(num_exact, num_variables):
        logits = categorical_made.forward(
            particles.reshape(num_samples * batch_size, num_variables),
            condition=condition,
        )
        conditional_i = torch.softmax(
            logits.reshape(
                num_samples, batch_size, num_variables, max_categories
            )[:, :, i, :],
            dim=-1,
        )
        marginals[:, i, :] = conditional_i.mean(0)
        if i == num_variables - 1:
            break
        draws = torch.multinomial(
            conditional_i[:, :, : num_categories[i]].reshape(
                num_samples * batch_size, num_categories[i]
            ),
            num_samples=1,
            generator=generator,
        )
        particles[:, :, i] = draws.reshape(num_samples, batch_size).to(dtype)

    return marginals


def compute_ece(
    true_vals: Tensor,
    probs: Tensor,
    num_bins: int,
    method: str = "top_label",
) -> Tensor:
    """Compute ECE for each discrete parameter.

    Public because experiment scripts score probabilities they produced
    themselves, rather than going through :func:`run_discrete_calibration`, which
    wants a posterior to draw from. The two ``_compute_ece_*`` implementations
    below stay private; this is the entry point.

    Args:
        true_vals: Ground-truth 0-indexed category indices, shape (N, num_discrete).
        probs: Predicted probabilities, shape (N, num_discrete, num_categories).
        num_bins: Number of calibration bins.
        method: ``"top_label"`` — confidence = p(mode), accuracy = 1{mode == true}.
            ``"classwise"`` — evaluates calibration of the full PMF: for every
            (sample, class) pair, confidence = p(class | x), correct = 1{class == true}.

    Returns:
        ECE per discrete parameter, shape (num_discrete,).
    """
    if method == "top_label":
        return _compute_ece_top_label(true_vals, probs, num_bins)
    elif method == "classwise":
        return _compute_ece_classwise(true_vals, probs, num_bins)
    else:
        raise ValueError(f"Unknown method '{method}', use 'top_label' or 'classwise'")


def _compute_ece_top_label(true_vals: Tensor, probs: Tensor, num_bins: int) -> Tensor:
    num_samples, num_discrete, _ = probs.shape
    ece = torch.zeros(num_discrete, device=probs.device)

    for i in range(num_discrete):
        confidence, predicted = probs[:, i].max(dim=-1)
        correct = (predicted == true_vals[:, i].long()).float()

        bin_edges = torch.linspace(0, 1, num_bins + 1, device=probs.device)
        bin_idx = torch.bucketize(confidence, bin_edges[:-1], right=False) - 1
        bin_idx = bin_idx.clamp(0, num_bins - 1)

        for b in range(num_bins):
            mask = bin_idx == b
            if mask.sum() > 0:
                ece[i] += (mask.sum() / num_samples) * (
                    confidence[mask].mean() - correct[mask].mean()
                ).abs()

    return ece


def _compute_ece_classwise(true_vals: Tensor, probs: Tensor, num_bins: int) -> Tensor:
    """Classwise ECE: evaluate calibration of the full PMF, not just the mode.

    For each test sample and each class k, confidence = p(k | x) and
    correct = 1{k == true value}. All (sample, class) pairs are pooled and
    binned by confidence.

    Note: assumes all discrete variables share the same number of categories
    (i.e. ``probs`` has a single last dimension K). If variables have mixed
    cardinalities, padding slots will silently contribute near-zero confidence
    entries and inflate the denominator.
    """
    num_samples, num_discrete, num_categories = probs.shape
    ece = torch.zeros(num_discrete, device=probs.device)

    bin_edges = torch.linspace(0, 1, num_bins + 1, device=probs.device)

    for i in range(num_discrete):
        confidence = probs[:, i].reshape(-1)  # (N * K,)
        true_idx = true_vals[:, i].long()  # (N,)
        # For each sample n, class k: correct = 1{k == true_idx[n]}
        class_indices = (
            torch.arange(num_categories, device=probs.device)
            .unsqueeze(0)
            .expand(num_samples, -1)
        )
        correct = (class_indices == true_idx.unsqueeze(1)).float().reshape(-1)  # (N * K,)

        n_total = confidence.numel()
        bin_idx = torch.bucketize(confidence, bin_edges[:-1], right=False) - 1
        bin_idx = bin_idx.clamp(0, num_bins - 1)

        for b in range(num_bins):
            mask = bin_idx == b
            if mask.sum() > 0:
                ece[i] += (mask.sum() / n_total) * (
                    confidence[mask].mean() - correct[mask].mean()
                ).abs()

    return ece


def _plot_param_bars(
    ax,
    true_vals_i: Tensor,
    probs_i: Tensor,
    color: str,
    label: str,
    num_bins: int,
    bin_edges: np.ndarray,
    bin_centers: np.ndarray,
    bar_width: float,
    alpha: float,
    exclude_low_populated: bool,
    min_samples: int,
) -> float:
    """Plot calibration bars for one discrete parameter onto ax.

    Returns chance_level (1 / num_categories) for the parameter.
    """
    confidence, predicted = probs_i.max(dim=-1)
    confidence = confidence.detach().numpy()
    predicted = predicted.detach().numpy()
    correct = (predicted == true_vals_i.long().numpy()).astype(float)

    bin_idx = np.digitize(confidence, bin_edges[:-1]) - 1
    bin_idx = np.clip(bin_idx, 0, num_bins - 1)

    accuracies = np.full(num_bins, np.nan)
    counts = np.zeros(num_bins, dtype=int)
    for b in range(num_bins):
        mask = bin_idx == b
        counts[b] = mask.sum()
        if counts[b] > 0:
            accuracies[b] = correct[mask].mean()

    n_categories = int((probs_i.sum(dim=0) > 0).sum())
    chance_level = 1 / n_categories

    populated = ~np.isnan(accuracies)
    shown = populated & (counts >= min_samples) if exclude_low_populated else populated
    ax.bar(bin_centers[shown], accuracies[shown], width=bar_width,
           color=color, alpha=alpha, label=label)

    return chance_level


def discrete_calibration_overlay_plot(
    true_vals: Tensor,
    probs: Tensor,
    ece: Optional[Tensor] = None,
    param_names: Optional[List[str]] = None,
    num_bins: int = 10,
    figsize: Optional[Tuple[float, float]] = None,
    exclude_low_populated: bool = False,
    min_samples: int = 20,
    ax: Optional[plt.Axes] = None,
    show_legend: bool = True,
    grouped: bool = False,
    show_chance_region: bool = True,
) -> Tuple[Figure, plt.Axes]:
    """Plot reliability diagrams for all discrete parameters on one axes.

    Args:
        true_vals: Ground-truth discrete values, shape (N, num_discrete).
        probs: Predicted probabilities, shape (N, num_discrete, num_categories).
        ece: Optional ECE per parameter, shape (num_discrete,).
        param_names: Optional names for each discrete parameter.
        num_bins: Number of bins for the calibration curve.
        figsize: Figure size; defaults to (5, 5).
        exclude_low_populated: If True, omit bins with fewer than min_samples.
        min_samples: Minimum number of samples per bin.
        grouped: If True, plot bars side-by-side within each bin instead of
            overlaying them. If False (default), bars are overlaid with alpha.
        show_chance_region: If True (default), shade the below-chance region in gray.

    Returns:
        fig: The matplotlib figure.
        ax: The single axes.
    """
    num_discrete = probs.shape[1]

    if param_names is None:
        param_names = [f"param_{i}" for i in range(num_discrete)]
    if ax is None:
        if figsize is None:
            figsize = (5, 5)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Grouped mode: split each bin into num_discrete sub-bars side by side.
    # Overlay mode: all bars share the same bin centers with alpha blending.
    bin_width = 1.0 / num_bins
    if grouped:
        sub_width = bin_width / num_discrete * 0.85
        alpha = 1.0
    else:
        sub_width = bin_width
        alpha = 0.5

    chance_levels = []
    for i in range(num_discrete):
        label = param_names[i]
        if ece is not None:
            label += f" (ECE={ece[i].item():.3f})"
        if grouped:
            offset = (i - (num_discrete - 1) / 2) * sub_width
            centers_i = bin_centers + offset
        else:
            centers_i = bin_centers
        chance_level = _plot_param_bars(
            ax, true_vals[:, i], probs[:, i], f"C{i}", label,
            num_bins, bin_edges, centers_i,
            sub_width, alpha,
            exclude_low_populated, min_samples,
        )
        chance_levels.append(chance_level)

    gray_end = np.floor(min(chance_levels) * num_bins) / num_bins
    if show_chance_region:
        ax.axvspan(0, gray_end, color="gray", alpha=0.2, zorder=0)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    if show_legend:
        ax.legend(fontsize=8)
    fig.tight_layout()

    return fig, ax


def eod_from_ranks(ranks, num_posterior_samples):
    """Compute the Error over Diagonal (EoD) from SBC ranks.

    EoD measures calibration of continuous parameters by comparing the empirical
    CDF of SBC ranks to the uniform CDF:

        EoD_d = mean_i | sorted_rank_i / K - (i+0.5)/n |

    Args:
        ranks: Rank values from SBC, shape (num_sbc_runs, num_dims).
        num_posterior_samples: Number of posterior samples used in the SBC procedure.

    Returns:
        EoD per dimension, shape (num_dims,).
    """
    n = ranks.shape[0]
    uniform_quantiles = (np.arange(n) + 0.5) / n
    return np.mean(
        np.abs(np.sort(ranks, axis=0) / num_posterior_samples - uniform_quantiles[:, None]),
        axis=0,
    )


def eod_expected_under_uniformity(
    num_sbc_runs: int,
    n_mc: int = 1000,
    seed: int = 0,
) -> float:
    """Expected EoD under the null hypothesis of uniform ranks via MC simulation.

    Under perfect calibration, SBC ranks are uniform. This computes the expected
    EoD due to finite-sample noise alone, serving as a baseline for convergence
    plots.

    Args:
        num_sbc_runs: Number of SBC test points.
        n_mc: Number of MC replicates.
        seed: Random seed for reproducibility.

    Returns:
        Expected EoD under uniformity.
    """
    rng = np.random.default_rng(seed)
    uniform_quantiles = (np.arange(num_sbc_runs) + 0.5) / num_sbc_runs  # (n,)
    samples = np.sort(rng.uniform(0, 1, size=(n_mc, num_sbc_runs)), axis=1)
    eod_samples = np.mean(np.abs(samples - uniform_quantiles[None, :]), axis=1)
    return float(np.mean(eod_samples))


# ── Discrete ECE baselines under perfect calibration ──────────────────


def ece_baseline_analytical(N, B=10, K=5):
    """Expected ECE of a perfectly calibrated K-class model.

    Assumes equal occupancy across active bins (maximum-ignorance assumption).

    Args:
        N: Number of SBC test cases.
        B: Number of equal-width ECE bins on [0, 1].
        K: Number of discrete classes.

    Returns:
        ECE baseline value (float).
    """
    bin_edges = np.linspace(0, 1, B + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    min_confidence = 1.0 / K

    # Active bins: upper edge > 1/K (can receive samples)
    active_mask = bin_edges[1:] > min_confidence
    active_centers = bin_centers[active_mask]
    B_active = int(active_mask.sum())
    n_per_bin = N / B_active

    # Per-bin noise: sqrt(2/pi) * sqrt(p*(1-p) / n)
    per_bin_noise = np.sqrt(2 / np.pi) * np.sqrt(
        active_centers * (1 - active_centers) / n_per_bin
    )

    # ECE = weighted average; with equal occupancy, weight = 1/B_active
    return float(np.mean(per_bin_noise))


def ece_baseline_empirical(bin_counts, B=10, K=5):
    """Expected ECE using analytical bin centers but empirical bin occupancy.

    Uses the simplified formula:
        E[ECE] = (1/N) * sqrt(2/pi) * sum_active sqrt(n_b * p_b * (1 - p_b))

    Args:
        bin_counts: Number of samples per bin (length B).
        B: Number of equal-width bins (used to compute bin centers).
        K: Number of discrete classes (determines active bin threshold 1/K).

    Returns:
        ECE baseline value (float).
    """
    counts = np.asarray(bin_counts, dtype=int)
    bin_edges = np.linspace(0, 1, B + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    n_total = counts.sum()

    # Active bins: upper edge > 1/K (can receive samples)
    min_confidence = 1.0 / K
    active_mask = bin_edges[1:] > min_confidence

    p_b = bin_centers[active_mask]
    n_b = counts[active_mask]

    # Empty active bins (n_b=0) contribute 0 naturally.
    return float(np.sqrt(2 / np.pi) / n_total * np.sum(np.sqrt(n_b * p_b * (1 - p_b))))


def ece_baseline_equal_mass(confidences, B=10):
    """Expected ECE for a perfectly calibrated model using equal-mass (adaptive) binning.

    Sorts predictions by confidence and splits into B quantile bins. The
    expected accuracy in each bin equals its mean confidence (perfect
    calibration assumption). No 1/K filtering needed — bins adapt to the
    actual confidence distribution.

    Uses the same simplified formula as the empirical baseline:
        E[ECE] = (1/N) * sqrt(2/pi) * sum_b sqrt(n_b * p_b * (1 - p_b))

    Args:
        confidences: 1D array of top-label confidence scores (length N).
        B: Number of equal-mass bins.

    Returns:
        ECE baseline value (float).
    """
    conf = np.sort(np.asarray(confidences))
    N = len(conf)
    bins = np.array_split(conf, B)

    total = 0.0
    for bin_conf in bins:
        n_b = len(bin_conf)
        if n_b == 0:
            continue
        p_b = bin_conf.mean()
        total += np.sqrt(n_b * p_b * (1 - p_b))

    return float(np.sqrt(2 / np.pi) / N * total)
