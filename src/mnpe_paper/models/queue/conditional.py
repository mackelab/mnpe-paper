"""Sampling q(theta_c | theta_d, x) directly from a MixedDensityEstimator.

`MixedDensityEstimator.sample` draws theta_d and theta_c jointly. The
conditional diagnostics need theta_d held fixed, which means reaching into the
continuous block with the discrete values supplied rather than sampled.
`verify_conditional_sampler` checks that this reproduces the estimator's own
sampling path.
"""

import torch
from sbi.utils.metrics import c2st


def sample_continuous_given_discrete(estimator, theta_d, x, num_samples):
    """Draw `num_samples` from q(theta_c | theta_d, x).

    theta_d: (num_discrete,) raw 0-indexed values; x: (1, dim_x) transformed.
    Returns (num_samples, num_continuous).
    """
    with torch.no_grad():
        embedded = estimator.condition_embedding(x.reshape(1, -1))
        disc = torch.as_tensor(theta_d, dtype=torch.float32).reshape(1, -1)
        combined = torch.cat(
            (disc.repeat(num_samples, 1), embedded.repeat(num_samples, 1)), dim=-1
        )
        cont = estimator.continuous_net.sample(torch.Size([]), condition=combined)
    if estimator.log_transform_input:
        cont = cont.exp()
    return cont


def conditional_ranks(estimator, theta_test, x_test, num_samples, progress_every=None):
    """Rank each test case's true theta_c within q(theta_c | theta_d, x)."""
    n = len(theta_test)
    ranks = torch.zeros(n, 3)
    for i in range(n):
        theta_star = theta_test[i]
        cont = sample_continuous_given_discrete(
            estimator, theta_star[3:5], x_test[i], num_samples
        )
        ranks[i] = (cont < theta_star[:3]).sum(dim=0).float()
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  {i + 1}/{n}")
    return ranks


def verify_conditional_sampler(post, x_o, seed, tol=0.03, verbose=False):
    """Check the helper against joint samples filtered by their drawn theta_d.

    That filtering is q(theta_c | theta_d, x) by construction, so the two must
    be indistinguishable (C2ST ~ 0.5). Tests the plumbing, not the calibration.
    """
    estimator = post.posterior_estimator
    torch.manual_seed(seed)
    joint = post.sample((100_000,), x=x_o, show_progress_bars=False)
    configs, counts = torch.stack([joint[:, 3], joint[:, 4]]).T.unique(
        dim=0, return_counts=True
    )
    top = configs[counts.argmax()]
    filtered = joint[(joint[:, 3] == top[0]) & (joint[:, 4] == top[1])][:, :3]
    cond = sample_continuous_given_discrete(estimator, top, x_o, 20_000)
    n = min(len(filtered), 5_000)
    score = float(c2st(filtered[:n], cond[:n]))

    print(f"verify-helper: dominant config {top.tolist()} (0-indexed), n={n}")
    if verbose:
        print(f"  joint-filtered moments: {filtered.mean(0).numpy().round(4)}"
              f" / {filtered.std(0).numpy().round(4)}")
        print(f"  conditional moments:    {cond.mean(0).numpy().round(4)}"
              f" / {cond.std(0).numpy().round(4)}")
    print(f"  C2ST(joint-filtered, conditional) = {score:.3f}  (expect ~0.5)")
    assert abs(score - 0.5) < tol, "helper does not replicate the joint sampling path"
    print("  OK")
    return score
