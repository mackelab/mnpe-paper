"""Exact posterior for the scalable Gaussian benchmark (independent version).

The model factorizes over dimensions, so every posterior quantity is a product
(or concatenation) of per-dimension closed forms — no 2^D enumeration needed,
valid at any D:

    p(z_i = 1 | x) = sigmoid((A x_i - A^2/2) / TAU)      (toy logistic formula)
    theta_i | z_i, x ~ N((x_i - A z_i) / TAU, V)

Discrete error is reported as per-dimension TV (binary: |q_i - p_i|), matching
how the queueing model reports c1/c2 per parameter and avoiding the 2^D
blowup of a joint-configuration TV. Function names mirror
:mod:`mnpe_paper.models.toy.config`, so the D=1 case can be checked against the
toy model's own analytics (see ``tests/test_exact_posterior.py``).
"""

import numpy as np
from scipy.special import expit

from mnpe_paper.models.scaling.config import TAU, A, V


def p_z_given_x(x):
    """Per-dim P(z_i = 1 | x). x: array-like (..., D); returns the same shape."""
    return expit((A * np.asarray(x, dtype=float) - 0.5 * A**2) / TAU)


def posterior_mean_given_z(x, z):
    """Conditional posterior mean (x_i - A z_i) / TAU, elementwise."""
    return (np.asarray(x, dtype=float) - A * np.asarray(z, dtype=float)) / TAU


def sample_gt_posterior(x, n_samples, rng=None):
    """Exact joint posterior samples at one observation x of shape (D,).

    Returns:
        samples: (n_samples, 2D) — theta in columns 0..D-1, z in columns D..2D-1.
    """
    rng = np.random.default_rng() if rng is None else rng
    x = np.asarray(x, dtype=float).reshape(-1)
    z = rng.binomial(1, p_z_given_x(x), size=(n_samples, x.size))
    theta = rng.normal(posterior_mean_given_z(x, z), np.sqrt(V))
    return np.concatenate([theta, z.astype(float)], axis=1)


def tv_discrete_per_dim(q_z1, x):
    """Per-dim binary TV |q(z_i=1|x) - p(z_i=1|x)| for estimated probs q_z1 (D,)."""
    return np.abs(np.asarray(q_z1, dtype=float) - p_z_given_x(x))


# ── K-way categorical extension (K-sweep; binary functions above unchanged) ──
# Class k has mean A*k, so marginalizing theta gives x_i | z_i=k ~ N(A*k, TAU).
# With a uniform class prior, p(z_i=k | x) = softmax_k(-(x_i - A*k)^2 / (2 TAU)),
# and theta_i | z_i=k, x ~ N((x_i - A*k)/TAU, V) — the same conditional as the
# binary case with A*z generalized to A*k. K=2 reproduces the binary posterior.


def p_z_categorical(x, K):
    """Per-dim categorical posterior p(z_i | x). x: (..., D) -> (..., D, K)."""
    x = np.asarray(x, dtype=float)
    means = A * np.arange(K)  # (K,)
    logits = -((x[..., None] - means) ** 2) / (2 * TAU)  # (..., D, K)
    logits -= logits.max(axis=-1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=-1, keepdims=True)


def sample_gt_posterior_k(x, n_samples, K, rng=None):
    """Exact joint posterior samples at one observation x of shape (D,), K classes.

    Returns (n_samples, 2D): theta in cols 0..D-1, z in cols D..2D-1.
    """
    rng = np.random.default_rng() if rng is None else rng
    x = np.asarray(x, dtype=float).reshape(-1)
    probs = p_z_categorical(x, K)  # (D, K)
    D = x.size
    z = np.empty((n_samples, D))
    for d in range(D):
        z[:, d] = rng.choice(K, size=n_samples, p=probs[d])
    theta = rng.normal(posterior_mean_given_z(x, z), np.sqrt(V))
    return np.concatenate([theta, z], axis=1)


def tv_categorical_per_dim(q_probs, x, K):
    """Per-dim categorical TV 0.5*sum_k |q_k - p_k|.

    q_probs: estimated per-dim class probabilities, shape (D, K).
    """
    p = p_z_categorical(x, K)  # (D, K)
    return 0.5 * np.abs(np.asarray(q_probs, dtype=float) - p).sum(axis=-1)
