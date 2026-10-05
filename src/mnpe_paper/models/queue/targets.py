"""Independent exact NumPyro/JAX target for the queue experiment.

This module deliberately reimplements the queue likelihood in JAX. It shares only
the scientific constants in :mod:`mnpe_paper.models.queue.spec` with the PyMC
target; it does not call the PyMC/PyTensor or NumPy likelihood implementations,
which is what makes agreement between the two an independent check rather than a
restatement of one implementation.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import numpyro
import numpyro.distributions as dist

from mnpe_paper.models.queue.spec import (
    C_VALUES,
    GAMMA_MEDIAN,
    GAMMA_SIGMA,
    MAX_UTILIZATION,
    MU1_MEDIAN,
    MU1_SIGMA,
    MU2_MEDIAN,
    MU2_SIGMA,
    N_C,
    OBSERVATION_NOISE,
    T,
)

jax.config.update("jax_enable_x64", True)

CONFIGURATIONS = np.asarray(
    [(int(c1), int(c2)) for c1 in C_VALUES for c2 in C_VALUES],
    dtype=np.int64,
)
_C1 = jnp.asarray(CONFIGURATIONS[:, 0], dtype=jnp.float64)
_C2 = jnp.asarray(CONFIGURATIONS[:, 1], dtype=jnp.float64)
_N = jnp.arange(int(C_VALUES.max()), dtype=jnp.float64)
_LOG_FACTORIAL_N = jsp.special.gammaln(_N + 1.0)


def expected_queue_jax(
    c: jax.Array, gamma: jax.Array, mu: jax.Array
) -> jax.Array:
    """Expected number waiting in a stationary M/M/c queue."""
    c = jnp.asarray(c, dtype=jnp.float64)
    gamma = jnp.asarray(gamma, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    r = gamma / mu
    rho = gamma / (c * mu)

    terms = jnp.exp(
        _N[:, None] * jnp.log(r)
        - _LOG_FACTORIAL_N[:, None]
    )
    finite_sum = jnp.sum(
        jnp.where(_N[:, None] < c[None, :], terms, 0.0), axis=0
    )
    factorial_c = jnp.exp(jsp.special.gammaln(c + 1.0))
    denominator = 1.0 - rho + 1e-10
    geometric_term = (r**c / factorial_c) / denominator
    pi_zero = 1.0 / (finite_sum + geometric_term)
    return (
        (r**c * rho)
        / (factorial_c * denominator**2)
        * pi_zero
    )


def _lower_truncated_normal_logpdf(
    value: float | jax.Array,
    loc: jax.Array,
    scale: float = OBSERVATION_NOISE,
) -> jax.Array:
    """Log density of Normal(loc, scale) truncated to [0, infinity)."""
    value = jnp.asarray(value, dtype=jnp.float64)
    scale = jnp.asarray(scale, dtype=jnp.float64)
    untruncated = dist.Normal(loc, scale).log_prob(value)
    log_normalizer = jsp.special.log_ndtr(loc / scale)
    return jnp.where(value >= 0.0, untruncated - log_normalizer, -jnp.inf)


def configuration_log_weights_jax(
    gamma: jax.Array,
    mu1: jax.Array,
    mu2: jax.Array,
    observations: Mapping[str, float],
) -> jax.Array:
    """Unnormalized log p(c1,c2, queue observations | continuous rates)."""
    gamma = jnp.asarray(gamma, dtype=jnp.float64)
    mu1 = jnp.asarray(mu1, dtype=jnp.float64)
    mu2 = jnp.asarray(mu2, dtype=jnp.float64)

    rho1 = gamma / (_C1 * mu1)
    rho2 = gamma / (_C2 * mu2)
    stable = (rho1 < MAX_UTILIZATION) & (rho2 < MAX_UTILIZATION)
    eq1 = expected_queue_jax(_C1, gamma, mu1)
    eq2 = expected_queue_jax(_C2, gamma, mu2)
    log_q1 = _lower_truncated_normal_logpdf(
        observations["queue_length_1"], eq1
    )
    log_q2 = _lower_truncated_normal_logpdf(
        observations["queue_length_2"], eq2
    )
    log_config_prior = -2.0 * jnp.log(jnp.asarray(N_C, dtype=jnp.float64))
    return jnp.where(stable, log_config_prior + log_q1 + log_q2, -jnp.inf)


def marginalized_queue_loglik_jax(
    gamma: jax.Array,
    mu1: jax.Array,
    mu2: jax.Array,
    observations: Mapping[str, float],
) -> jax.Array:
    """Queue likelihood after exact summation over all server counts."""
    return jsp.special.logsumexp(
        configuration_log_weights_jax(
            gamma, mu1, mu2, observations
        )
    )


def configuration_probabilities_jax(
    gamma: jax.Array,
    mu1: jax.Array,
    mu2: jax.Array,
    observations: Mapping[str, float],
) -> jax.Array:
    """Conditional joint probabilities for all 25 server configurations."""
    log_weights = configuration_log_weights_jax(
        gamma, mu1, mu2, observations
    )
    return jax.nn.softmax(log_weights)


def marginalized_queue_model(observations: Mapping[str, float]) -> None:
    """NumPyro model with both server counts exactly marginalized."""
    gamma = numpyro.sample(
        "gamma",
        dist.LogNormal(jnp.log(GAMMA_MEDIAN), GAMMA_SIGMA),
    )
    mu1 = numpyro.sample(
        "mu1",
        dist.LogNormal(jnp.log(MU1_MEDIAN), MU1_SIGMA),
    )
    mu2 = numpyro.sample(
        "mu2",
        dist.LogNormal(jnp.log(MU2_MEDIAN), MU2_SIGMA),
    )

    numpyro.sample(
        "n_arrivals_obs",
        dist.Poisson(gamma * T),
        obs=jnp.asarray(observations["n_arrivals"]),
    )
    numpyro.sample(
        "n_completions_1_obs",
        dist.Poisson(gamma * T),
        obs=jnp.asarray(observations["n_completions_1"]),
    )
    numpyro.sample(
        "n_completions_2_obs",
        dist.Poisson(gamma * T),
        obs=jnp.asarray(observations["n_completions_2"]),
    )
    numpyro.factor(
        "marginal_queue_likelihood",
        marginalized_queue_loglik_jax(gamma, mu1, mu2, observations),
    )


def batched_configuration_probabilities(
    samples: Mapping[str, np.ndarray],
    observations: Mapping[str, float],
) -> np.ndarray:
    """Evaluate configuration probabilities for a NumPyro sample dictionary."""
    probability_fn = jax.jit(
        jax.vmap(
            lambda gamma, mu1, mu2: configuration_probabilities_jax(
                gamma, mu1, mu2, observations
            )
        )
    )
    return np.asarray(
        probability_fn(samples["gamma"], samples["mu1"], samples["mu2"])
    )
