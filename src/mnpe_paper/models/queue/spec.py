"""Canonical scientific specification for the tandem M/M/c experiment.

The prior is the original independent product prior conditioned on both
stations lying in the stationary region ``rho < MAX_UTILIZATION``.
"""

import numpy as np

MODEL_VERSION = "stable_joint_rho0999_v1"

# Independent base prior before conditioning on stability.
GAMMA_MEDIAN = 9.0
GAMMA_SIGMA = 0.3
MU1_MEDIAN = 8.0
MU1_SIGMA = 0.3
MU2_MEDIAN = 5.0
MU2_SIGMA = 0.3
C_VALUES = np.array([2, 3, 4, 5, 6])
N_C = len(C_VALUES)

# Observation model.
T = 100.0
OBSERVATION_NOISE = 0.1

# A small margin from rho=1 avoids the singular stationary queue formula.
MAX_UTILIZATION = 0.999

# Reference parameters.
C1_TRUE = 2
C2_TRUE = 3
GAMMA_TRUE = 9.0
MU1_TRUE = 8.0
MU2_TRUE = 5.0

# Training draw and the held-out test set used for SBC and ECE.
SIMULATION_SEED = 42
SBC_TEST_SEED = 9999

# Optuna winner from experiments/queue/main/mnpe_hparam_tuning.py (NLTP = 4.847).
#
# `hidden_layers` is swallowed by build_nsf's **kwargs and has no effect; the
# flows use num_blocks=2. It is kept because the result pickles record it, so
# dropping it would change what the saved runs report they were trained with.
#
# `num_components` is deliberately absent for the same reason: build_mnpe passes
# it on to the continuous head, but build_nsf has no such parameter and discards
# it, and this project always uses flow_model="nsf". Its default differs between
# sbi versions (5 in the pinned one, 10 later), which cannot affect these networks.
# `num_transforms` and `num_bins` ARE consumed, which is why both are pinned here.
TUNED_HPARAMS = dict(
    num_transforms=4,
    hidden_features=256,
    hidden_layers=4,
    num_bins=9,
    training_batch_size=256,
    learning_rate=1.1e-4,
)

# For the plain-NPE baselines, whose posterior_nn signature has no hidden_layers.
TUNED_FLOW_HPARAMS = {k: v for k, v in TUNED_HPARAMS.items() if k != "hidden_layers"}

# Discrete parameters are 0-indexed inside SBI; C_VALUES[i] is the server count.
DISCRETE_OFFSET = int(C_VALUES[0])


def to_physical(idx):
    """Map 0-indexed server categories to physical server counts."""
    return C_VALUES[np.asarray(idx).astype(int)]


def to_index(c):
    """Map physical server counts back to 0-indexed categories."""
    return np.asarray(c).astype(int) - DISCRETE_OFFSET


def stability_mask_numpy(theta: np.ndarray) -> np.ndarray:
    """Check support of the stability-conditioned prior.

    ``theta`` uses SBI ordering and zero-indexed server categories:
    ``[gamma, mu1, mu2, c1_idx, c2_idx]``.
    """
    theta = np.asarray(theta)
    theta_2d = np.atleast_2d(theta)
    c1_idx = theta_2d[:, 3].astype(int)
    c2_idx = theta_2d[:, 4].astype(int)
    valid_indices = (
        (c1_idx >= 0)
        & (c1_idx < N_C)
        & (c2_idx >= 0)
        & (c2_idx < N_C)
        & (theta_2d[:, 3] == c1_idx)
        & (theta_2d[:, 4] == c2_idx)
    )

    safe_c1_idx = np.clip(c1_idx, 0, N_C - 1)
    safe_c2_idx = np.clip(c2_idx, 0, N_C - 1)
    gamma, mu1, mu2 = theta_2d[:, 0], theta_2d[:, 1], theta_2d[:, 2]
    positive = (gamma > 0) & (mu1 > 0) & (mu2 > 0)
    rho1 = gamma / (C_VALUES[safe_c1_idx] * mu1)
    rho2 = gamma / (C_VALUES[safe_c2_idx] * mu2)
    stable = (
        valid_indices
        & positive
        & (rho1 < MAX_UTILIZATION)
        & (rho2 < MAX_UTILIZATION)
    )
    return stable if theta.ndim > 1 else stable[0]
