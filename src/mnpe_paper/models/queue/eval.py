"""Shared network builder and evaluation for the queue budget studies.

`budget_study.py`, `reeval_budget_study.py` and `eval_queue_1M.py` differ only
in where the posterior comes from; scoring them through one function keeps the
reference, test set, C2ST seed and bin count identical across the three.

`post.sample`, `run_sbc` and `prior.sample` all draw from the global torch RNG,
so the order of operations below is part of what a seed reproduces.
"""

from functools import partial

import numpy as np
import torch
from sbi.diagnostics.sbc import check_sbc, run_sbc
from sbi.neural_nets.net_builders.mixed_nets import build_mnpe
from sbi.utils.metrics import c2st

from mnpe_paper.calibration import run_discrete_calibration
from mnpe_paper.models.queue.spec import TUNED_HPARAMS, to_physical

PARAM_NAMES = ["gamma", "mu1", "mu2", "c1", "c2"]
DISCRETE_COLS = [3, 4]

N_POSTERIOR_SAMPLES = 10_000
N_SBC_POSTERIOR = 1000
N_ECE_BINS = 10
C2ST_SEED = 1


def build_tuned_mnpe(**overrides):
    """`build_mnpe` fixed to the tuned architecture with an identity embedding."""
    kwargs = dict(
        num_transforms=TUNED_HPARAMS["num_transforms"],
        hidden_features=TUNED_HPARAMS["hidden_features"],
        hidden_layers=TUNED_HPARAMS["hidden_layers"],
        num_bins=TUNED_HPARAMS["num_bins"],
        embedding_net=torch.nn.Identity(),
    )
    kwargs.update(overrides)
    return partial(build_mnpe, **kwargs)


def evaluate_posterior(
    post,
    x_o,
    ref_samples,
    theta_test,
    x_test,
    prior,
    num_posterior_samples=N_POSTERIOR_SAMPLES,
    num_sbc_posterior=N_SBC_POSTERIOR,
    num_ece_bins=N_ECE_BINS,
):
    """Score one posterior against the reference and the held-out test set.

    Returns the metric dict stored per rep: posterior samples with the discrete
    columns mapped to physical server counts, per-marginal and joint C2ST,
    marginal and joint SBC, and top-label and classwise ECE.
    """
    samples = post.sample((num_posterior_samples,), x=x_o).numpy()
    for col in DISCRETE_COLS:
        samples[:, col] = to_physical(samples[:, col])

    # C2ST vs reference (per-marginal + joint)
    n = min(len(samples), len(ref_samples))
    c2st_marginal = {}
    for i, name in enumerate(PARAM_NAMES):
        Xi = torch.tensor(samples[:n, i : i + 1], dtype=torch.float32)
        Yi = torch.tensor(ref_samples[:n, i : i + 1], dtype=torch.float32)
        c2st_marginal[name] = c2st(Xi, Yi, seed=C2ST_SEED).item()
    c2st_all = c2st(
        torch.tensor(samples[:n], dtype=torch.float32),
        torch.tensor(ref_samples[:n], dtype=torch.float32),
        seed=C2ST_SEED,
    ).item()

    # SBC (continuous)
    ranks, dap_samples = run_sbc(
        theta_test,
        x_test,
        post,
        num_posterior_samples=num_sbc_posterior,
        reduce_fns="marginals",
        num_workers=1,
        show_progress_bar=True,
    )
    theta_for_check = prior.sample((len(theta_test),))
    sbc_results = check_sbc(
        ranks,
        theta_for_check,
        dap_samples,
        num_posterior_samples=num_sbc_posterior,
    )
    sbc_ks_pvals = [sbc_results["ks_pvals"][i].item() for i in range(3)]

    # Joint SBC via log_prob
    estimator = post.posterior_estimator

    def joint_log_prob(theta, x, _est=estimator):
        theta_3d = theta.unsqueeze(0)
        x_2d = x.unsqueeze(0).expand(theta.shape[0], -1)
        return _est.log_prob(theta_3d, condition=x_2d).squeeze(0)

    joint_ranks, joint_dap = run_sbc(
        theta_test,
        x_test,
        post,
        num_posterior_samples=num_sbc_posterior,
        reduce_fns=joint_log_prob,
        num_workers=1,
        show_progress_bar=True,
    )
    joint_sbc_results = check_sbc(
        joint_ranks,
        theta_for_check,
        joint_dap,
        num_posterior_samples=num_sbc_posterior,
    )
    joint_sbc_ks_pval = joint_sbc_results["ks_pvals"][0].item()

    # ECE (discrete) — top-label and classwise
    ece, probs, true_disc = run_discrete_calibration(
        theta_test,
        x_test,
        post,
        discrete_indices=DISCRETE_COLS,
        num_bins=num_ece_bins,
        method="top_label",
    )
    ece_cw, _, _ = run_discrete_calibration(
        theta_test,
        x_test,
        post,
        discrete_indices=DISCRETE_COLS,
        num_bins=num_ece_bins,
        method="classwise",
    )

    return {
        "samples": samples,
        "c2st_marginal": c2st_marginal,
        "c2st_all": c2st_all,
        "sbc_ks_pvals": sbc_ks_pvals,
        "sbc_ranks": ranks.numpy(),
        "joint_sbc_ks_pval": joint_sbc_ks_pval,
        "joint_sbc_ranks": joint_ranks.numpy(),
        "ece": ece.tolist(),
        "ece_classwise": ece_cw.tolist(),
        "ece_probs": probs.numpy(),
        "ece_true_vals": true_disc.numpy(),
    }


def format_metrics(m):
    """One-line per-rep summary, shared by the three studies."""
    return (
        f"    C2ST joint: {m['c2st_all']:.3f}  "
        f"ECE c1={m['ece'][0]:.3f} c2={m['ece'][1]:.3f}  "
        f"ECE-cw c1={m['ece_classwise'][0]:.4f} c2={m['ece_classwise'][1]:.4f}"
    )


def summarize_c2st(reps, n_reps):
    """Across-rep C2ST summary line."""
    c2st_vals = [r["c2st_all"] for r in reps]
    return (
        f"    C2ST joint: {np.mean(c2st_vals):.3f} "
        f"+/- {np.std(c2st_vals) / np.sqrt(n_reps):.3f} (SEM)"
    )
