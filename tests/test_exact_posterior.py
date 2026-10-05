"""The exact posterior the scaling sweeps are scored against.

Both sweeps measure MNPE against this closed form rather than against another
sampler, so a bug here would not raise anywhere — it would silently move every
number in Appendix B.1. These are therefore gating tests, and they check the
closed form against something that does not assume it:

  1. D=1 reproduces the toy model's own analytics (same A, sigma), which are
     derived independently in ``experiments/toy/tractable-example-derivation.md``.
  2. D=2 matches brute-force numerical integration on a 2-D grid, with NO use
     of the factorization assumption on the brute-force side.
  3. Simulator prior-predictive moments match theory; dimensions uncorrelated.
  4. The K-way categorical extension reduces to the binary case at K=2 and
     matches numerical integration at K=3.
  5. Ground-truth sampler self-consistency (frequencies, shapes, column layout).
"""

import numpy as np

from mnpe_paper.models.scaling.config import SIGMA, TAU, A, V, generate_training_data
from mnpe_paper.models.scaling.exact_posterior import (
    p_z_categorical,
    p_z_given_x,
    posterior_mean_given_z,
    sample_gt_posterior,
    sample_gt_posterior_k,
    tv_categorical_per_dim,
    tv_discrete_per_dim,
)
from mnpe_paper.models.toy import config as toy


def test_d1_matches_toy_analytics():
    assert A == toy.a and SIGMA == toy.sigma
    assert TAU == toy.tau and V == toy.v
    x = np.linspace(-4, 6, 101)
    np.testing.assert_allclose(p_z_given_x(x), toy.p_z_given_x(x, z=1), rtol=1e-12)
    for z in (0, 1):
        np.testing.assert_allclose(
            posterior_mean_given_z(x, z), toy.posterior_mean_given_z(x, z), rtol=1e-12
        )


def test_d1_gt_sampler_matches_analytic_mixture():
    x_val = 1.3
    rng = np.random.default_rng(0)
    s = sample_gt_posterior([x_val], 400_000, rng=rng)
    p1 = p_z_given_x(x_val)
    mix_mean = (1 - p1) * posterior_mean_given_z(x_val, 0) + p1 * posterior_mean_given_z(x_val, 1)
    mix_var = V + p1 * (1 - p1) * (A / TAU) ** 2
    assert abs(s[:, 1].mean() - p1) < 4 * np.sqrt(p1 * (1 - p1) / len(s))
    assert abs(s[:, 0].mean() - mix_mean) < 4 * np.sqrt(mix_var / len(s))
    assert abs(s[:, 0].var() - mix_var) < 0.01


def test_d2_matches_brute_force_grid():
    """Brute-force p(z|x) and E[theta|z,x] on a 2-D grid, no factorization assumed."""
    x = np.array([0.3, -1.2])
    t = np.linspace(-8, 8, 1201)
    t1, t2 = np.meshgrid(t, t, indexing="ij")

    def gauss(u, mean, var):
        return np.exp(-0.5 * (u - mean) ** 2 / var) / np.sqrt(2 * np.pi * var)

    prior_2d = gauss(t1, 0, 1) * gauss(t2, 0, 1)
    post_config, mean_t1 = {}, {}
    for z1 in (0, 1):
        for z2 in (0, 1):
            lik = gauss(x[0], t1 + A * z1, SIGMA**2) * gauss(x[1], t2 + A * z2, SIGMA**2)
            joint = prior_2d * lik * 0.25  # uniform prior over the 4 configs
            post_config[(z1, z2)] = np.trapezoid(np.trapezoid(joint, t, axis=1), t)
            mean_t1[(z1, z2)] = (
                np.trapezoid(np.trapezoid(joint * t1, t, axis=1), t) / post_config[(z1, z2)]
            )
    total = sum(post_config.values())

    p1 = p_z_given_x(x)  # analytic per-dim marginals
    bf_p_z1_dim0 = (post_config[(1, 0)] + post_config[(1, 1)]) / total
    bf_p_z1_dim1 = (post_config[(0, 1)] + post_config[(1, 1)]) / total
    np.testing.assert_allclose([bf_p_z1_dim0, bf_p_z1_dim1], p1, rtol=1e-6)

    # analytic factorized config probabilities vs brute force
    for (z1, z2), bf in post_config.items():
        analytic = (p1[0] if z1 else 1 - p1[0]) * (p1[1] if z2 else 1 - p1[1])
        np.testing.assert_allclose(bf / total, analytic, rtol=1e-6)

    # conditional continuous mean per config vs closed form
    for (z1, z2), bf_mean in mean_t1.items():
        np.testing.assert_allclose(bf_mean, posterior_mean_given_z(x[0], z1), rtol=1e-5)


def test_simulator_prior_predictive_moments():
    D = 4
    _, x = generate_training_data(300_000, D, seed=0)
    x = x.numpy()
    np.testing.assert_allclose(x.mean(0), A / 2, atol=0.02)
    np.testing.assert_allclose(x.var(0), 1 + A**2 / 4 + SIGMA**2, atol=0.03)
    off_diag = np.cov(x.T) - np.diag(np.diag(np.cov(x.T)))
    assert np.abs(off_diag).max() < 0.02  # dims independent


def test_categorical_k2_reduces_to_binary():
    x = np.linspace(-4, 6, 101).reshape(-1, 1)  # (N, D=1)
    p = p_z_categorical(x, 2)  # (N, 1, 2)
    np.testing.assert_allclose(p[..., 1], p_z_given_x(x), rtol=1e-12)
    np.testing.assert_allclose(p[..., 0], 1 - p_z_given_x(x), rtol=1e-12)
    np.testing.assert_allclose(p.sum(-1), 1.0, rtol=1e-12)


def test_categorical_matches_brute_force_k3():
    """p(z=k|x) via numerical integration over theta, no closed form assumed."""
    K, x = 3, np.array([[1.4]])  # (N=1, D=1)
    t = np.linspace(-12, 12, 20001)
    prior = np.exp(-0.5 * t**2) / np.sqrt(2 * np.pi)
    unn = []
    for k in range(K):
        lik = np.exp(-0.5 * (x[0, 0] - (t + A * k)) ** 2 / SIGMA**2) / np.sqrt(
            2 * np.pi * SIGMA**2
        )
        unn.append((1 / K) * np.trapezoid(prior * lik, t))
    bf = np.array(unn) / sum(unn)
    np.testing.assert_allclose(p_z_categorical(x, K)[0, 0], bf, rtol=1e-5)


def test_tv_categorical_k2_matches_binary():
    x = np.array([0.5, -1.0])  # (D=2,)
    q_z1 = np.array([0.3, 0.8])
    q_probs = np.stack([1 - q_z1, q_z1], axis=-1)  # (D, 2)
    np.testing.assert_allclose(
        tv_categorical_per_dim(q_probs, x, 2), tv_discrete_per_dim(q_z1, x), rtol=1e-12
    )


def test_categorical_gt_sampler_freqs():
    K, D = 25, 2
    x = np.array([6.0, 20.0])  # near class 3 and class 10 (means A*k = 0,2,4,...)
    s = sample_gt_posterior_k(x, 200_000, K, rng=np.random.default_rng(0))
    assert s.shape == (200_000, 2 * D)
    p = p_z_categorical(x, K)  # (D, K)
    for d in range(D):
        freq = np.bincount(s[:, D + d].astype(int), minlength=K) / len(s)
        assert np.abs(freq - p[d]).max() < 0.01


def test_gt_sampler_shapes_and_layout():
    D = 7
    x = np.linspace(-1, 3, D)
    s = sample_gt_posterior(x, 1000, rng=np.random.default_rng(1))
    assert s.shape == (1000, 2 * D)
    z = s[:, D:]
    assert set(np.unique(z)).issubset({0.0, 1.0})  # discrete columns last
    assert len(np.unique(s[:, :D])) > 900  # continuous columns first
    np.testing.assert_allclose(tv_discrete_per_dim(p_z_given_x(x), x), 0, atol=1e-15)
