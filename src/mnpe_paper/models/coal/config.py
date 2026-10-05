"""Coal-mining-disaster changepoint model: data, priors, simulator, training data.

The classic Bayesian changepoint example, in which a discrete switchpoint
parameter marks the year the disaster rate changed:

    switchpoint  ~ DiscreteUniform(0, N_YEARS - 1)
    early_rate   ~ Exponential(1)
    late_rate    ~ Exponential(1)
    disasters[t] ~ Poisson(early_rate if t < switchpoint else late_rate)

The switchpoint is carried as a 0-based year index rather than a year throughout
inference, and converted for display only; :func:`idx_to_year` and
:func:`year_to_idx` are the single place that offset is written down.

References:
    PyMC, "Marginalizing models":
    https://www.pymc.io/projects/examples/en/latest/howto/marginalizing-models.html
    Raftery & Akman (1986); counts from Jarrett (1979).
"""

import numpy as np
import torch

__all__ = [
    "DISASTER_DATA",
    "EARLY_RATE_TRUE",
    "FIRST_YEAR",
    "LATE_RATE_TRUE",
    "N_SWITCHPOINTS",
    "N_YEARS",
    "RATE_PRIOR_SCALE",
    "SWITCHPOINT_MAX",
    "SWITCHPOINT_TRUE_IDX",
    "SWITCHPOINT_TRUE_YEAR",
    "YEARS",
    "YEAR_IDX",
    "build_sbi_prior",
    "compute_rate_vector",
    "generate_training_data",
    "idx_to_year",
    "simulator",
    "transform_x",
    "year_to_idx",
]

# ==================== DATA ====================

# Number of coal mining disasters per year in the UK (1851-1962).
# Source: Jarrett (1979), "A note on the intervals between coal-mining disasters".
# The two NaNs of the original series are replaced with 2, as the PyMC example does.
DISASTER_DATA = np.array([
    4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
    3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
    2, 2, 3, 4, 2, 1, 3, 2, 2, 1, 1, 1, 1, 3, 0, 0,
    1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
    0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
    3, 3, 1, 2, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
    0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1
], dtype=np.float32)

FIRST_YEAR = 1851

YEARS = np.arange(FIRST_YEAR, FIRST_YEAR + len(DISASTER_DATA))
N_YEARS = len(YEARS)

# Year indices (0-indexed for modelling)
YEAR_IDX = np.arange(N_YEARS)

# ==================== PRIOR SPECIFICATION ====================

# Discrete: switchpoint as year index (0 to N_YEARS-1)
SWITCHPOINT_MAX = N_YEARS - 1
N_SWITCHPOINTS = N_YEARS  # 111 possible values

# Continuous: Exponential(1) priors on both rates
RATE_PRIOR_SCALE = 1.0

# ==================== TRUE PARAMETERS (HISTORICAL ESTIMATE) ====================

# Historical analysis places the switchpoint near 1890, when mining safety
# regulation tightened. These are reference values for the figures, not the
# ground truth of a simulation study.
SWITCHPOINT_TRUE_YEAR = 1890
SWITCHPOINT_TRUE_IDX = SWITCHPOINT_TRUE_YEAR - FIRST_YEAR  # = 39

EARLY_RATE_TRUE = 3.0  # ~3 disasters/year before 1890
LATE_RATE_TRUE = 1.0  # ~1 disaster/year after 1890


# ==================== HELPER FUNCTIONS ====================

def year_to_idx(year):
    """Convert a year to its 0-based index."""
    return year - FIRST_YEAR


def idx_to_year(idx):
    """Convert a 0-based index to its year."""
    return idx + FIRST_YEAR


def compute_rate_vector(switchpoint_idx, early_rate, late_rate):
    """Return the Poisson rate for each year, given the switchpoint.

    Args:
        switchpoint_idx: Index of the switchpoint (0 to N_YEARS-1).
        early_rate: Rate before the switchpoint.
        late_rate: Rate at and after the switchpoint.

    Returns:
        Array of shape ``(N_YEARS,)``.
    """
    return np.where(YEAR_IDX < switchpoint_idx, early_rate, late_rate)


# ==================== OBSERVATION TRANSFORM ====================

def transform_x(x):
    """Variance-stabilising transform for the Poisson count observations.

    Applies a square root to all ``N_YEARS`` count dimensions. For
    ``X ~ Poisson(lambda)``, ``Var(sqrt(X))`` is approximately 1/4 whatever
    lambda is, which makes the data far friendlier to z-scoring and stops the
    IQR-based outlier check in sbi's training pipeline from firing.

    Args:
        x: Raw Poisson counts, shape ``(..., N_YEARS)``.

    Returns:
        Tensor of the same shape, square-root transformed.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    return torch.sqrt(x)


# ==================== SBI PRIOR ====================

def build_sbi_prior():
    """Build the joint prior for MNPE.

    Returns:
        A ``MultipleIndependent`` over ``[early_rate, late_rate,
        switchpoint_idx]`` — continuous parameters first, the discrete one last,
        which is the order MNPE expects.
    """
    from sbi.utils import MultipleIndependent
    from torch.distributions import Categorical, Exponential

    prior_early = Exponential(torch.tensor([RATE_PRIOR_SCALE]))
    prior_late = Exponential(torch.tensor([RATE_PRIOR_SCALE]))

    probs = torch.ones(N_SWITCHPOINTS) / N_SWITCHPOINTS
    prior_switchpoint = Categorical(probs=probs.unsqueeze(0))

    return MultipleIndependent([prior_early, prior_late, prior_switchpoint])


# ==================== TRAINING DATA VIA PYMC ====================

def generate_training_data(n_samples, seed=42):
    """Generate ``(theta, x)`` training pairs by PyMC prior predictive sampling.

    Drawing through ``pm.sample_prior_predictive`` on the generative model rather
    than through a standalone simulator guarantees the training distribution
    matches the PyMC likelihood the MCMC reference targets. A separate simulator
    that differs even slightly produces a well-calibrated posterior for the wrong
    model, which is not visible in any diagnostic run against itself.

    Args:
        n_samples: Number of training pairs.
        seed: Seed for reproducibility.

    Returns:
        ``theta`` of shape ``(n_samples, 3)`` holding
        ``[early_rate, late_rate, switchpoint_idx]``, and ``x`` of shape
        ``(n_samples, N_YEARS)`` holding disaster counts per year.
    """
    import pymc as pm

    with pm.Model():
        switchpoint = pm.DiscreteUniform("switchpoint", lower=0, upper=N_YEARS - 1)
        early_rate = pm.Exponential("early_rate", RATE_PRIOR_SCALE)
        late_rate = pm.Exponential("late_rate", RATE_PRIOR_SCALE)

        rate = pm.math.switch(switchpoint > YEAR_IDX, early_rate, late_rate)
        pm.Poisson("disasters", rate, shape=N_YEARS)

        prior_pred = pm.sample_prior_predictive(samples=n_samples, random_seed=seed)

    early_vals = prior_pred.prior["early_rate"].values.flatten()
    late_vals = prior_pred.prior["late_rate"].values.flatten()
    sp_vals = prior_pred.prior["switchpoint"].values.flatten()

    theta_np = np.column_stack([early_vals, late_vals, sp_vals])
    x_np = prior_pred.prior["disasters"].values.reshape(n_samples, N_YEARS)

    return (
        torch.tensor(theta_np, dtype=torch.float32),
        torch.tensor(x_np, dtype=torch.float32),
    )


# ==================== SIMULATOR (PREDICTIVE CHECKS ONLY) ====================

def simulator(theta, seed=None):
    """Forward-simulate disaster counts from parameters.

    For training data use :func:`generate_training_data`, which samples through
    the PyMC prior predictive and so cannot drift from the inference model. This
    function exists for prior and posterior predictive checks, where its speed
    matters and an exact likelihood match does not.

    Args:
        theta: Parameters ``[early_rate, late_rate, switchpoint_idx]``, shape
            ``(3,)`` or ``(n_samples, 3)``.
        seed: Optional seed.

    Returns:
        Simulated counts of shape ``(N_YEARS,)`` or ``(n_samples, N_YEARS)``.
    """
    single_sample = theta.ndim == 1
    if single_sample:
        theta = theta.reshape(1, -1)

    n_samples = theta.shape[0]
    x = np.zeros((n_samples, N_YEARS))
    rng = np.random.default_rng(seed)

    for i in range(n_samples):
        early_rate, late_rate, switchpoint_idx = theta[i]
        rates = compute_rate_vector(int(switchpoint_idx), early_rate, late_rate)
        x[i] = rng.poisson(rates)

    return x.reshape(-1) if single_sample else x
