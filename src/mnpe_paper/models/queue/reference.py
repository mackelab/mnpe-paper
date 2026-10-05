"""Deterministic evidence and rejection sampling for the queue posterior.

The model factorizes conditional on the arrival rate.  This reduces every
configuration evidence to nested one-dimensional integrals and avoids asking
MCMC to estimate the relative masses of separated server-count modes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid, quad, quad_vec
from scipy.optimize import minimize_scalar
from scipy.special import gammaln, log_ndtr

from mnpe_paper.models.queue.spec import (
    C_VALUES,
    GAMMA_MEDIAN,
    GAMMA_SIGMA,
    MAX_UTILIZATION,
    MU1_MEDIAN,
    MU1_SIGMA,
    MU2_MEDIAN,
    MU2_SIGMA,
    OBSERVATION_NOISE,
    T,
)

CONFIGURATIONS = np.asarray(
    [(int(c1), int(c2)) for c1 in C_VALUES for c2 in C_VALUES],
    dtype=np.int64,
)
_LOG_TWO_PI = np.log(2.0 * np.pi)
@dataclass(frozen=True)
class EvidenceResult:
    """All configuration evidences and their normalized probabilities."""

    configurations: np.ndarray
    evidences: np.ndarray
    probabilities: np.ndarray
    absolute_error: float


def _standard_normal_logpdf(z: float | np.ndarray) -> float | np.ndarray:
    return -0.5 * np.asarray(z) ** 2 - 0.5 * _LOG_TWO_PI


def expected_queue_numpy(
    c: int | np.ndarray,
    gamma: float | np.ndarray,
    mu: float | np.ndarray,
) -> np.ndarray:
    """Independently implemented stationary M/M/c expected queue length."""
    c, gamma, mu = np.broadcast_arrays(c, gamma, mu)
    c = c.astype(int)
    r = gamma / mu
    rho = gamma / (c * mu)
    finite_sum = np.zeros_like(r, dtype=float)
    for n in range(int(C_VALUES.max())):
        finite_sum += np.where(c > n, r**n / np.exp(gammaln(n + 1)), 0.0)
    factorial_c = np.exp(gammaln(c + 1))
    denominator = 1.0 - rho + 1e-10
    geometric = (r**c / factorial_c) / denominator
    pi_zero = 1.0 / (finite_sum + geometric)
    return (
        (r**c * rho)
        / (factorial_c * denominator**2)
        * pi_zero
    )


def queue_loglik_numpy(
    c: int | np.ndarray,
    gamma: float | np.ndarray,
    mu: float | np.ndarray,
    observed_queue: float,
) -> np.ndarray:
    """Lower-truncated Gaussian queue-observation log likelihood."""
    c, gamma, mu = np.broadcast_arrays(c, gamma, mu)
    rho = gamma / (c * mu)
    stable = rho < MAX_UTILIZATION
    expected = expected_queue_numpy(c, gamma, mu)
    standardized = (observed_queue - expected) / OBSERVATION_NOISE
    log_density = (
        -0.5 * standardized**2
        - np.log(OBSERVATION_NOISE)
        - 0.5 * _LOG_TWO_PI
        - log_ndtr(expected / OBSERVATION_NOISE)
    )
    return np.where(stable & (observed_queue >= 0.0), log_density, -np.inf)


def count_loglik_numpy(
    gamma: float | np.ndarray,
    observations: Mapping[str, float],
) -> np.ndarray:
    """Log likelihood of the three conditionally independent counts."""
    gamma = np.asarray(gamma)
    rate = gamma * T
    counts = np.asarray(
        [
            observations["n_arrivals"],
            observations["n_completions_1"],
            observations["n_completions_2"],
        ]
    )
    return np.sum(
        counts * np.log(rate[..., None])
        - rate[..., None]
        - gammaln(counts + 1.0),
        axis=-1,
    )


def _station_integral(
    gamma: float,
    c: int,
    observed_queue: float,
    mu_median: float,
    mu_sigma: float,
    *,
    epsabs: float,
    epsrel: float,
) -> float:
    lower = (
        np.log(gamma / (c * MAX_UTILIZATION)) - np.log(mu_median)
    ) / mu_sigma

    def integrand(z: float) -> float:
        mu = mu_median * np.exp(mu_sigma * z)
        log_value = (
            _standard_normal_logpdf(z)
            + queue_loglik_numpy(c, gamma, mu, observed_queue)
        )
        return float(np.exp(log_value))

    if lower >= 12.0:
        return 0.0
    value, _ = quad(
        integrand,
        lower,
        12.0,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=200,
    )
    return value


def configuration_evidences(
    observations: Mapping[str, float],
    *,
    epsabs: float = 1e-12,
    epsrel: float = 1e-9,
) -> EvidenceResult:
    """Calculate all 25 configuration evidences by adaptive quadrature."""
    q1 = float(observations["queue_length_1"])
    q2 = float(observations["queue_length_2"])
    inner_epsabs = min(epsabs, 1e-12)

    def negative_count_log_density(z_gamma: float) -> float:
        gamma = GAMMA_MEDIAN * np.exp(GAMMA_SIGMA * z_gamma)
        return float(
            -_standard_normal_logpdf(z_gamma)
            - count_loglik_numpy(gamma, observations)
        )

    count_mode = minimize_scalar(
        negative_count_log_density,
        bounds=(-10.0, 10.0),
        method="bounded",
    )
    log_scale = -float(count_mode.fun)

    def outer_integrand(z_gamma: float) -> np.ndarray:
        gamma = GAMMA_MEDIAN * np.exp(GAMMA_SIGMA * z_gamma)
        common = np.exp(
            _standard_normal_logpdf(z_gamma)
            + count_loglik_numpy(gamma, observations)
            - log_scale
        )
        station1 = np.asarray(
            [
                _station_integral(
                    gamma,
                    int(c),
                    q1,
                    MU1_MEDIAN,
                    MU1_SIGMA,
                    epsabs=inner_epsabs,
                    epsrel=epsrel,
                )
                for c in C_VALUES
            ]
        )
        station2 = np.asarray(
            [
                _station_integral(
                    gamma,
                    int(c),
                    q2,
                    MU2_MEDIAN,
                    MU2_SIGMA,
                    epsabs=inner_epsabs,
                    epsrel=epsrel,
                )
                for c in C_VALUES
            ]
        )
        return common * np.outer(station1, station2).reshape(-1)

    evidences, error = quad_vec(
        outer_integrand,
        -10.0,
        10.0,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=300,
    )
    scale = np.exp(log_scale) / len(CONFIGURATIONS)
    evidences = np.asarray(evidences) * scale
    probabilities = evidences / evidences.sum()
    return EvidenceResult(
        configurations=CONFIGURATIONS.copy(),
        evidences=evidences,
        probabilities=probabilities,
        absolute_error=float(error) * scale,
    )


def _gamma_count_inverse_cdf(
    observations: Mapping[str, float],
    *,
    grid_size: int = 50_001,
) -> tuple[np.ndarray, np.ndarray]:
    """Tabulate the one-dimensional gamma-given-counts inverse CDF."""

    def negative_log_density(z: float) -> float:
        gamma = GAMMA_MEDIAN * np.exp(GAMMA_SIGMA * z)
        return float(
            -_standard_normal_logpdf(z)
            - count_loglik_numpy(gamma, observations)
        )

    optimum = minimize_scalar(
        negative_log_density,
        bounds=(-10.0, 10.0),
        method="bounded",
        options={"xatol": 1e-13},
    )
    mode = float(optimum.x)
    curvature_step = 1e-4
    curvature = (
        negative_log_density(mode + curvature_step)
        - 2.0 * negative_log_density(mode)
        + negative_log_density(mode - curvature_step)
    ) / curvature_step**2
    scale = 1.0 / np.sqrt(curvature)
    grid = np.linspace(mode - 12.0 * scale, mode + 12.0 * scale, grid_size)
    log_density = np.asarray(
        [
            -negative_log_density(float(z))
            for z in grid
        ]
    )
    density = np.exp(log_density - log_density.max())
    cdf = np.concatenate(
        ([0.0], cumulative_trapezoid(density, grid))
    )
    cdf /= cdf[-1]
    return grid, cdf


def _sample_gamma_given_counts(
    inverse_cdf: tuple[np.ndarray, np.ndarray],
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    grid, cdf = inverse_cdf
    z_samples = np.interp(rng.random(size), cdf, grid)
    return GAMMA_MEDIAN * np.exp(GAMMA_SIGMA * z_samples)


def _queue_density_bound(observed_queue: float) -> float:
    """Analytic upper bound for all nonnegative truncated-normal locations."""
    base = 1.0 / (OBSERVATION_NOISE * np.sqrt(2.0 * np.pi))
    midpoint = observed_queue / 2.0
    upper_half_bound = base / np.exp(
        log_ndtr(midpoint / OBSERVATION_NOISE)
    )
    lower_half_bound = 2.0 * base * np.exp(
        -(observed_queue**2) / (8.0 * OBSERVATION_NOISE**2)
    )
    return float(max(upper_half_bound, lower_half_bound))


def sample_posterior_rejection(
    observations: Mapping[str, float],
    num_samples: int,
    *,
    seed: int = 42,
    batch_size: int = 100_000,
) -> tuple[dict[str, np.ndarray], float]:
    """Draw independent mixed-posterior samples by bounded rejection.

    The proposal absorbs the exact count-only posterior for ``gamma``. Server
    counts and service rates come from their base priors. Stability is enforced
    before accepting with the two queue likelihoods.
    """
    rng = np.random.default_rng(seed)
    accepted: dict[str, list[np.ndarray]] = {
        name: [] for name in ("gamma", "mu1", "mu2", "c1", "c2")
    }
    total_accepted = 0
    total_proposed = 0
    gamma_inverse_cdf = _gamma_count_inverse_cdf(observations)
    bound1 = _queue_density_bound(
        float(observations["queue_length_1"])
    )
    bound2 = _queue_density_bound(
        float(observations["queue_length_2"])
    )
    log_bound = np.log(bound1) + np.log(bound2)

    while total_accepted < num_samples:
        current_batch = max(batch_size, num_samples - total_accepted)
        gamma = _sample_gamma_given_counts(
            gamma_inverse_cdf, current_batch, rng
        )
        c1 = rng.choice(C_VALUES, size=current_batch)
        c2 = rng.choice(C_VALUES, size=current_batch)
        mu1 = rng.lognormal(
            mean=np.log(MU1_MEDIAN),
            sigma=MU1_SIGMA,
            size=current_batch,
        )
        mu2 = rng.lognormal(
            mean=np.log(MU2_MEDIAN),
            sigma=MU2_SIGMA,
            size=current_batch,
        )
        log_acceptance = (
            queue_loglik_numpy(
                c1,
                gamma,
                mu1,
                float(observations["queue_length_1"]),
            )
            + queue_loglik_numpy(
                c2,
                gamma,
                mu2,
                float(observations["queue_length_2"]),
            )
            - log_bound
        )
        if np.nanmax(log_acceptance) > 1e-8:
            raise FloatingPointError(
                "Rejection likelihood exceeded its certified density bound"
            )
        keep = np.log(rng.random(current_batch)) < log_acceptance
        kept = int(keep.sum())
        if kept:
            for name, values in [
                ("gamma", gamma),
                ("mu1", mu1),
                ("mu2", mu2),
                ("c1", c1),
                ("c2", c2),
            ]:
                accepted[name].append(values[keep])
            total_accepted += kept
        total_proposed += current_batch

    samples = {
        name: np.concatenate(chunks)[:num_samples]
        for name, chunks in accepted.items()
    }
    return samples, total_accepted / total_proposed
