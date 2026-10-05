"""Shared evaluation grid and metrics for the toy factorization studies.

The dequantization baseline, the three-way factorization study and the
error-propagation sweep all score against the same analytic posterior, so they
share one grid and one metric function. Keeping them here is what makes their
result tables comparable: a grid or a sample count changed in one script only
would silently move the numbers it is compared against.
"""

import numpy as np
import torch
from sbi.utils.metrics import c2st

from mnpe_paper.models.toy.config import (
    p_z_given_x,
    posterior_mean_given_z,
    sample_gt_posterior,
    v,
)

__all__ = ["N_SAMPLES", "X_GRID", "eval_samples"]

X_GRID = [-1.0, 0.0, 1.0, 2.0, 3.0]  # p(z=1|x) from ~0.01 to ~0.96
N_SAMPLES = 10_000


def eval_samples(theta_s, z_s, x_val, method, seed, n_samples=N_SAMPLES):
    """Metrics for joint samples (theta_s, z_s in {0,1}) at observation x_val."""
    gt_theta, gt_z = sample_gt_posterior(x_val, n_samples)
    joint = torch.tensor(np.stack([theta_s, z_s], axis=1), dtype=torch.float32)
    gt = torch.tensor(
        np.stack([gt_theta, gt_z.astype(float)], axis=1), dtype=torch.float32
    )

    row = {
        "method": method,
        "seed": seed,
        "x_o": x_val,
        "p_z1_true": float(p_z_given_x(x_val, z=1)),
        "p_z1_est": float((z_s == 1).mean()),
        "c2st_joint": float(c2st(joint, gt)),
    }
    row["p_z1_abs_err"] = abs(row["p_z1_est"] - row["p_z1_true"])

    # coupling: conditional moments per z vs exact N(m_z, sqrt(v))
    for z_val in (0, 1):
        mask = z_s == z_val
        m_exact = posterior_mean_given_z(x_val, z_val)
        if mask.sum() >= 50:
            row[f"cond_mean_abs_err_z{z_val}"] = abs(float(theta_s[mask].mean()) - m_exact)
            row[f"cond_std_ratio_z{z_val}"] = float(theta_s[mask].std()) / np.sqrt(v)
        else:
            row[f"cond_mean_abs_err_z{z_val}"] = np.nan
            row[f"cond_std_ratio_z{z_val}"] = np.nan
    return row
