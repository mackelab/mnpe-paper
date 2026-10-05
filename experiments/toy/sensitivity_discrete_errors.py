"""Sensitivity of the continuous posterior to discrete-head errors (Appendix B.5).

Empirical check of the propagation bound

    TV(q(theta|x), p(theta|x)) <= TV(q(z|x), p(z|x)) + max_z TV(q(theta|z,x), p(theta|z,x))

on the tractable toy model, by injecting controlled distortion into the trained
MNPE's discrete head via temperature scaling, q_T(z|x) prop q(z|x)^(1/T), while
holding the continuous head fixed, which isolates the discrete propagation
channel. T < 1 makes the head overconfident, T > 1 underconfident, T = 1 is as
trained.

Per (x, T) the script measures:
  - discrete error:   TV(q_T(z|x), p(z|x))  (binary TV = |q_T(z=1|x) - p(z=1|x)|)
  - continuous error: TV of the continuous marginal (numerical integration of the
    q_T-weighted mixture of learned conditionals against the analytic marginal),
    Wasserstein-1 and C2ST on 10k continuous marginal samples vs analytic draws
  - the bound above:  TV_disc + max_z TV(learned conditional, exact conditional)
plus, per T, the discrete ECE over a prior-predictive test set (temperature-scaled
head probabilities against realised z*).

Requires ``artifacts/toy/mnpe_posterior.pt`` from ``budget_study.py``.

Usage:
    uv run python experiments/toy/sensitivity_discrete_errors.py

Outputs:
    results/toy/sensitivity_discrete_errors.csv     — per (x, T) metrics + per-T ECE
    artifacts/toy/sensitivity_discrete_errors.pdf/png — continuous vs discrete error

The published Appendix B.5 figure is drawn from the CSV by
``figures/sensitivity_discrete_errors_figure.py``; the plot written here is the
diagnostic view of the same sweep.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sbi.utils.metrics import c2st
from scipy.stats import norm, wasserstein_distance

from mnpe_paper.calibration import compute_ece
from mnpe_paper.models.toy.config import (
    p_theta_given_x,
    p_z_given_x,
    posterior_mean_given_z,
    sample_gt_posterior,
    sample_prior,
    simulator,
    v,
)
from mnpe_paper.models.toy.eval import N_SAMPLES, X_GRID
from mnpe_paper.paths import model_artifacts_dir, model_results_dir
from mnpe_paper.plotting.plot_style import despine

ARTIFACTS_DIR = model_artifacts_dir("toy")
RESULTS_DIR = model_results_dir("toy")
POSTERIOR_FILE = ARTIFACTS_DIR / "mnpe_posterior.pt"

TEMPERATURES = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
N_TEST_ECE = 2_000
THETA_GRID = np.linspace(-6.0, 6.0, 4001)
SEED = 0

torch.manual_seed(SEED)

if not POSTERIOR_FILE.exists():
    raise SystemExit(
        f"{POSTERIOR_FILE} is missing. This sweep distorts a trained discrete head, "
        "so it needs the estimator itself rather than the tracked figure export. "
        "Run experiments/toy/budget_study.py first."
    )

post = torch.load(POSTERIOR_FILE, weights_only=False)
est = post.posterior_estimator
made = est.discrete_net.net
assert made.num_variables == 1, "toy model has a single discrete variable"


def q_z_probs(x):
    """MNPE discrete-head probabilities q(z|x), shape (n, 2).

    With a single discrete variable the MADE forward pass returns the exact
    marginal; the input values are masked out and do not matter (see
    ``run_discrete_calibration`` in mnpe_paper.calibration).
    """
    with torch.no_grad():
        logits = made.forward(torch.zeros(x.shape[0], 1), condition=x)
    return torch.softmax(logits.reshape(x.shape[0], 1, -1), dim=-1)[:, 0, :]


def temperature_scale(probs, T):
    """q_T prop q^(1/T), computed in log space."""
    return torch.softmax(torch.log(probs) / T, dim=-1)


def sample_continuous_given_z(z, x_val, num_samples):
    """theta ~ q(theta | z, x) from the fixed continuous head."""
    with torch.no_grad():
        emb = est.condition_embedding(torch.tensor([[x_val]], dtype=torch.float32))
        disc = z.reshape(num_samples, 1).float()
        comb = torch.cat((disc, emb.repeat(num_samples, 1)), dim=-1)
        cont = est.continuous_net.sample(torch.Size([]), condition=comb)
    if est.log_transform_input:
        cont = cont.exp()
    return cont[:, 0].numpy()


def learned_conditional_density(x_val, z_val):
    """q(theta | z, x) of the fixed continuous head, evaluated on THETA_GRID."""
    n = len(THETA_GRID)
    with torch.no_grad():
        emb = est.condition_embedding(torch.tensor([[x_val]], dtype=torch.float32))
        disc = torch.full((n, 1), float(z_val))
        comb = torch.cat((disc, emb.repeat(n, 1)), dim=-1)
        theta = torch.tensor(THETA_GRID, dtype=torch.float32).reshape(1, n, 1)
        log_p = est.continuous_net.log_prob(theta, condition=comb)
    return np.exp(log_p.numpy().reshape(-1))


def tv_on_grid(p, q):
    return 0.5 * np.trapezoid(np.abs(p - q), THETA_GRID)


# ── Per-(x, T) propagation metrics ─────────────────────────────────────────
rows = []
for x_val in X_GRID:
    q_probs = q_z_probs(torch.tensor([[x_val]], dtype=torch.float32))[0]
    p_z1_true = float(p_z_given_x(x_val, z=1))

    # T-independent pieces: learned conditionals and their error vs exact
    cond_dens = {z: learned_conditional_density(x_val, z) for z in (0, 1)}
    tv_cond = {
        z: tv_on_grid(
            cond_dens[z],
            norm.pdf(THETA_GRID, posterior_mean_given_z(x_val, z), np.sqrt(v)),
        )
        for z in (0, 1)
    }
    p_marg_true = p_theta_given_x(THETA_GRID, x_val)
    gt_theta, _ = sample_gt_posterior(x_val, N_SAMPLES)

    for T in TEMPERATURES:
        qT = temperature_scale(q_probs.unsqueeze(0), T)[0]
        tv_disc = abs(float(qT[1]) - p_z1_true)

        # numerical TV of the continuous marginal: q_T-weighted learned mixture
        q_marg = float(qT[0]) * cond_dens[0] + float(qT[1]) * cond_dens[1]
        tv_cont = tv_on_grid(q_marg, p_marg_true)

        # sample-based metrics on the continuous marginal
        z_s = torch.bernoulli(torch.full((N_SAMPLES,), float(qT[1]))).to(torch.float32)
        theta_s = sample_continuous_given_z(z_s, x_val, N_SAMPLES)
        w1 = wasserstein_distance(theta_s, gt_theta)
        c2st_cont = float(
            c2st(
                torch.tensor(theta_s, dtype=torch.float32).reshape(-1, 1),
                torch.tensor(gt_theta, dtype=torch.float32).reshape(-1, 1),
            )
        )

        rows.append({
            "x_o": x_val,
            "T": T,
            "q_z1_trained": float(q_probs[1]),
            "q_z1_T": float(qT[1]),
            "p_z1_true": p_z1_true,
            "tv_discrete": tv_disc,
            "tv_continuous": tv_cont,
            "tv_cond_max": max(tv_cond.values()),
            "bound": tv_disc + max(tv_cond.values()),
            "w1_continuous": w1,
            "c2st_continuous": c2st_cont,
        })
        print(
            f"x={x_val:+.1f} T={T:<4} TV_d={tv_disc:.3f} TV_c={tv_cont:.3f} "
            f"bound={rows[-1]['bound']:.3f} W1={w1:.3f} C2ST={c2st_cont:.3f}"
        )

df = pd.DataFrame(rows)

# ── Discrete ECE over a prior-predictive test set, per T ──────────────────
params = sample_prior(N_TEST_ECE)
x_test = simulator(params)
z_true = params[:, 1:2]
probs_test = q_z_probs(x_test)
ece_rows = []
for T in TEMPERATURES:
    probs_T = temperature_scale(probs_test, T).unsqueeze(1)  # (N, 1, 2)
    ece = compute_ece(z_true.long(), probs_T, num_bins=10, method="top_label")
    ece_rows.append({"T": T, "ece_discrete": float(ece[0])})
    print(f"T={T:<4} ECE={float(ece[0]):.4f}")

df = df.merge(pd.DataFrame(ece_rows), on="T")
df.to_csv(RESULTS_DIR / "sensitivity_discrete_errors.csv", index=False)

# ── Figure: continuous error vs discrete error, one curve per x ───────────
fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.3))
cmap = plt.get_cmap("viridis")
xcolors = {x_val: cmap(i / (len(X_GRID) - 1)) for i, x_val in enumerate(X_GRID)}

lims = [0, df["tv_discrete"].max() * 1.05]
axes[0].plot(lims, lims, ls="--", color="0.5", lw=0.8, zorder=0, label="TV$_c$ = TV$_d$")
for x_val, g in df.groupby("x_o"):
    g = g.sort_values("tv_discrete")
    axes[0].plot(g["tv_discrete"], g["tv_continuous"], "o-", ms=2.5, lw=1,
                 color=xcolors[x_val], label=f"$x_o={x_val:g}$")
    axes[1].plot(g["tv_discrete"], g["c2st_continuous"], "o-", ms=2.5, lw=1,
                 color=xcolors[x_val])
g1 = df[df["T"] == 1.0]
axes[0].scatter(g1["tv_discrete"], g1["tv_continuous"], s=28, facecolors="none",
                edgecolors="k", lw=0.8, zorder=3, label="$T=1$ (as trained)")
axes[1].scatter(g1["tv_discrete"], g1["c2st_continuous"], s=28, facecolors="none",
                edgecolors="k", lw=0.8, zorder=3)
axes[1].axhline(0.5, ls="--", color="0.5", lw=0.8, zorder=0)
axes[1].set_ylim(0.49, 0.6)
axes[0].set_xlabel(r"discrete error TV$(q_T(z\mid x),\, p(z\mid x))$")
axes[0].set_ylabel(r"continuous error TV$(q_T(\theta\mid x),\, p(\theta\mid x))$")
axes[1].set_xlabel(r"discrete error TV$(q_T(z\mid x),\, p(z\mid x))$")
axes[1].set_ylabel(r"C2ST continuous marginal")
axes[0].legend(frameon=False, handlelength=1.4)
for ax in axes:
    despine(ax, offset=3)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(ARTIFACTS_DIR / f"sensitivity_discrete_errors.{ext}")
plt.close(fig)
print("Saved sensitivity_discrete_errors.csv and the diagnostic pdf/png")
