"""Scalable Gaussian benchmark: D independent copies of the tractable toy model.

Model (per dimension i = 1..D):
    theta_i ~ N(0, 1),  z_i ~ Bernoulli(1/2)
    x_i | theta_i, z_i ~ N(theta_i + A * z_i, SIGMA^2)

Identical per-dimension structure to :mod:`mnpe_paper.models.toy.config` (same
A, SIGMA), so the exact posterior factorizes over dimensions and stays
computable at any D — see :mod:`mnpe_paper.models.scaling.exact_posterior`.

Column convention: parameters are (n, 2D) with continuous theta in columns
0..D-1 and discrete z in columns D..2D-1 (continuous first, discrete last).

Deviation from the project convention of generating training data via
pm.sample_prior_predictive(): there is no PyMC likelihood model in this
experiment that a separate simulator could mismatch — the pure-Gaussian torch
simulator below IS the model, exactly — so direct sampling is used.
"""

import torch

A = 2.0  # class offset for z_i = 1 (matches models.toy.config.a)
SIGMA = 0.5  # observation noise std (matches models.toy.config.sigma)
SIGMA2 = SIGMA**2
TAU = 1 + SIGMA2  # marginal variance of x_i | z_i
V = SIGMA2 / TAU  # posterior variance of theta_i | x_i, z_i


def sample_prior(n, D, K=2):
    """Sample (theta, z) from the prior. Returns tensor of shape (n, 2D).

    K is the number of classes per discrete dim (K=2 = the binary D-sweep
    model). Class means are equally spaced at A*k (k=0..K-1): the per-class
    separation is held fixed as K grows, isolating "scaling in the number of
    classes" from per-class difficulty (mirrors how the D-sweep holds per-dim
    difficulty fixed while scaling D).
    """
    theta = torch.randn(n, D)
    z = torch.randint(0, K, (n, D)).float()
    return torch.cat([theta, z], dim=1)


def simulator(params, D):
    """Simulate x_i = theta_i + A * z_i + eps_i. Returns tensor of shape (n, D)."""
    theta, z = params[:, :D], params[:, D:]
    signal = theta + A * z
    return signal + SIGMA * torch.randn_like(signal)


def generate_training_data(n, D, seed, K=2):
    """Prior-predictive training set (theta, x); fork_rng keeps global RNG state.

    simulator needs no K: x_i = theta_i + A*z_i already places class k at mean
    A*k for any integer z_i, so the binary path is unchanged when K=2.
    """
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        theta = sample_prior(n, D, K)
        x = simulator(theta, D)
    return theta, x
