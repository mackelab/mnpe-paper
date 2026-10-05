"""Tractable Gaussian benchmark — model config, simulator, and analytical posteriors.

Model:
    θ ~ N(0, 1),  z ~ Bernoulli(1/2)
    x | θ, z ~ N(θ + a·z, σ²)

The posterior is analytically tractable:
    p(z=1 | x) = sigmoid((a·x - a²/2) / (1 + σ²))
    p(θ | x)   = mixture of two Gaussians (one per z)
"""

import numpy as np
import torch
from scipy import stats
from scipy.special import expit

# ── Model hyperparameters ──────────────────────────────────────────────────
a = 2.0  # shift magnitude for z=1
sigma = 0.5  # observation noise std

# Derived quantities
sigma2 = sigma**2
tau = 1 + sigma2  # marginal variance
v = sigma2 / tau  # posterior variance for θ|x,z


# ── Simulator ──────────────────────────────────────────────────────────────
def simulator(params):
    """Simulate x | θ, z ~ N(θ + a·z, σ²).

    Args:
        params: tensor of shape (n, 2) — column 0 is θ, column 1 is z.

    Returns:
        x: tensor of shape (n, 1).
    """
    theta = params[:, 0]
    z = params[:, 1]
    mean = theta + a * z
    noise = torch.randn_like(theta) * sigma
    return (mean + noise).unsqueeze(1)


# ── Analytical ground truth ────────────────────────────────────────────────
def p_z_given_x(x, z=1):
    """p(z | x) using the logistic form."""
    logit = (a * x - 0.5 * a**2) / tau
    p_z1 = expit(logit)
    return p_z1 if z == 1 else 1 - p_z1


def posterior_mean_given_z(x, z):
    """Conditional posterior mean m_z = (x - a·z) / (1 + σ²)."""
    return (x - a * z) / tau


def p_theta_given_x(theta, x):
    """Marginal posterior p(θ | x) — two-component Gaussian mixture."""
    p_z0 = p_z_given_x(x, z=0)
    p_z1 = p_z_given_x(x, z=1)
    m0 = posterior_mean_given_z(x, 0)
    m1 = posterior_mean_given_z(x, 1)
    return (
        p_z0 * stats.norm.pdf(theta, loc=m0, scale=np.sqrt(v))
        + p_z1 * stats.norm.pdf(theta, loc=m1, scale=np.sqrt(v))
    )


def sample_gt_posterior(x, n_samples):
    """Sample from the ground truth posterior p(θ, z | x).

    Returns:
        theta: (n_samples,) continuous samples
        z: (n_samples,) discrete samples (0 or 1)
    """
    p_z1 = p_z_given_x(x, z=1)
    z = np.random.binomial(1, p_z1, size=n_samples)
    m_z = posterior_mean_given_z(x, z)
    theta = np.random.normal(m_z, np.sqrt(v))
    return theta, z


def sample_prior(n):
    """Sample (θ, z) from the prior. Returns tensor of shape (n, 2)."""
    theta = torch.randn(n, 1)
    z = torch.randint(0, 2, (n, 1)).float()
    return torch.cat([theta, z], dim=1)
