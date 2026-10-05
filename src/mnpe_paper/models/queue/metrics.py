"""Metrics recomputed from saved budget-study posteriors.

Nothing here retrains. Each function reads an existing posterior or reference
and derives a number from it, so a metric definition can change without the
pipeline being rerun.

    M1  ``c2st_split_floor`` — the C2ST noise floor, from disjoint splits of the
        reference against itself. This is what an identical-distribution
        comparison actually scores under the study's protocol -- not the
        disagreement between two approximate references, which would measure
        reference error.
    M2  ``exact_joint_discrete`` and ``discrete_agreement`` — total variation
        against the reference, reading q(c1, c2 | x_o) out of the CategoricalMADE
        by enumeration rather than by sampling.
    M4  ``ece_null_interval`` — the ECE null simulated from the model's own PMF,
        rather than the equal-occupancy analytical approximation.

M3, marginal versus factor ECE, needs no function here: it is
``calibration.run_discrete_calibration`` called twice with different
``probability_type``. The distinction matters — the factor evaluates each
autoregressive term at the true preceding value, which is unavailable at
inference time — so the two are always reported side by side.

The driver that walks every saved posterior and writes the tables is
``experiments/queue/diagnostics/posterior_metrics.py``. These functions live here
because ``reference_yardstick.py`` uses three of them for a different purpose.
"""

from __future__ import annotations

import numpy as np
import torch
from sbi.utils.metrics import c2st

from mnpe_paper.calibration import compute_ece
from mnpe_paper.models.queue.config import C_VALUES

__all__ = [
    "c2st_split_floor",
    "discrete_agreement",
    "ece_null_interval",
    "exact_joint_discrete",
]


# ── M1 ──────────────────────────────────────────────────────────────────────


def c2st_split_floor(
    reference: np.ndarray, n: int, n_splits: int, reserved: int
) -> dict:
    """C2ST between disjoint reference subsets, none overlapping the first
    ``reserved`` rows used for the MNPE comparisons."""
    available = len(reference) - reserved
    needed = 2 * n * n_splits
    if available < needed:
        raise SystemExit(
            f"Need {needed:,} unused reference draws for {n_splits} disjoint "
            f"splits of {n:,}, but only {available:,} remain after reserving "
            f"{reserved:,}. Reduce --n-floor-splits."
        )
    scores = []
    for split in range(n_splits):
        base = reserved + 2 * n * split
        first = reference[base : base + n]
        second = reference[base + n : base + 2 * n]
        scores.append(
            c2st(
                torch.tensor(first, dtype=torch.float32),
                torch.tensor(second, dtype=torch.float32),
                seed=1,
            ).item()
        )
    return {
        "scores": scores,
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores, ddof=1)),
    }


# ── M2 ──────────────────────────────────────────────────────────────────────


def exact_joint_discrete(posterior, x_o: torch.Tensor) -> np.ndarray:
    """Read ``q(c1, c2 | x_o)`` exactly out of the CategoricalMADE.

    With two variables of five categories there are only 25 configurations, so
    the autoregressive factors can be enumerated rather than sampled. The
    result carries no Monte Carlo noise, which matters because the quadrature
    side of the comparison carries none either.
    """
    net = posterior.potential_fn.posterior_estimator
    made = net.discrete_net.net
    n_categories = len(C_VALUES)

    grid = torch.tensor(
        [[c1, c2] for c1 in range(n_categories) for c2 in range(n_categories)],
        dtype=torch.float32,
    )
    condition = x_o.unsqueeze(0).expand(len(grid), -1)

    with torch.no_grad():
        logits = made.forward(grid, condition=condition)
    logits = logits.reshape(len(grid), 2, made.max_num_categories)
    factors = torch.softmax(logits, dim=-1)

    index = grid.long()
    # factors[i, 0, c1] is q(c1 | x); factors[i, 1, c2] is q(c2 | c1, x).
    joint = (
        factors[torch.arange(len(grid)), 0, index[:, 0]]
        * factors[torch.arange(len(grid)), 1, index[:, 1]]
    )
    return joint.numpy()


def discrete_agreement(model_joint: np.ndarray, quadrature: np.ndarray) -> dict:
    n_categories = len(C_VALUES)
    model_grid = model_joint.reshape(n_categories, n_categories)
    quad_grid = quadrature.reshape(n_categories, n_categories)
    return {
        "joint_tv": 0.5 * float(np.abs(model_joint - quadrature).sum()),
        "max_cell_error": float(np.abs(model_joint - quadrature).max()),
        "c1_tv": 0.5 * float(np.abs(model_grid.sum(1) - quad_grid.sum(1)).sum()),
        "c2_tv": 0.5 * float(np.abs(model_grid.sum(0) - quad_grid.sum(0)).sum()),
        "model_sum": float(model_joint.sum()),
    }


# ── M4 ──────────────────────────────────────────────────────────────────────


def ece_null_interval(
    probs: torch.Tensor,
    num_bins: int,
    method: str,
    n_replicates: int,
    seed: int,
) -> dict:
    """Simulate ECE under perfect calibration, conditional on the confidences.

    Labels are drawn from the model's own predicted PMF. Under that draw the
    model is calibrated by construction, so the spread of the recomputed ECE is
    exactly the finite-sample noise floor for these confidences -- with no
    equal-occupancy or bin-centre approximation.
    """
    generator = torch.Generator().manual_seed(seed)
    n_cases, n_variables, n_categories = probs.shape
    flat = probs.reshape(-1, n_categories)

    values = []
    for _ in range(n_replicates):
        drawn = torch.multinomial(flat, num_samples=1, generator=generator)
        simulated = drawn.reshape(n_cases, n_variables)
        values.append(compute_ece(simulated, probs, num_bins, method=method).numpy())

    stacked = np.stack(values)
    return {
        "mean": stacked.mean(axis=0),
        "q025": np.quantile(stacked, 0.025, axis=0),
        "q975": np.quantile(stacked, 0.975, axis=0),
    }
