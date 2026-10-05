"""Three-way factorization study on the tractable toy (Appendix B.4, Table B-2).

Compares three ways to factor the joint posterior p(theta, z | x):
  forward      q(z|x) q(theta|z,x)   — the trained MNPE; the continuous head only
                                       ever fits UNIMODAL per-z conditionals.
  independent  q(z|x) q(theta|x)     — drop the conditioning (tests whether
                                       conditioning matters); cannot represent
                                       the theta-z coupling even with perfect
                                       marginals.
  reversed     q(theta|x) q(z|theta,x) — flip the order (tests which direction);
                                       a VALID factorization, but the first-stage
                                       flow must now fit the MULTIMODAL continuous
                                       marginal q(theta|x) (bimodal at x=1, modes
                                       +/-0.8, ~3.6 posterior SDs apart).

Independent and reversed SHARE one continuous marginal flow q(theta|x) (standard
NPE on the continuous column), so they differ only in the discrete stage — the
clean isolation of the two questions. Discrete stages are small MLP classifiers
trained by cross-entropy on simulator (theta, x) -> z pairs (NOT on flow samples,
so stage-1 error does not compound into stage-2 training).

Matched capacity: the continuous flow uses the toy MNPE's actual continuous
hyperparameters (nflows NSF, hidden 32 / transforms 5 / blocks 4).

Requires ``artifacts/toy/mnpe_posterior.pt`` from ``budget_study.py``.

Usage:
    uv run python experiments/toy/reverse_factorization_toy.py [--smoke]

Outputs:
    results/toy/reverse_factorization_toy.csv — per (method, seed, x_o): joint
        C2ST, discrete probability error, per-z conditional moments
    artifacts/toy/reverse_factorization_toy.pdf/png — the continuous marginal at
        x=1 (analytic bimodal vs the learned reverse/independent flow)
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from torch.distributions import Normal

from mnpe_paper.models.toy.config import (
    p_theta_given_x,
    posterior_mean_given_z,
    sample_prior,
    simulator,
)
from mnpe_paper.models.toy.eval import N_SAMPLES, X_GRID, eval_samples
from mnpe_paper.paths import model_artifacts_dir, model_results_dir
from mnpe_paper.plotting.plot_style import despine

ARTIFACTS_DIR = model_artifacts_dir("toy")
RESULTS_DIR = model_results_dir("toy")
POSTERIOR_FILE = ARTIFACTS_DIR / "mnpe_posterior.pt"

parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true")
args = parser.parse_args()
BUDGET = 1_000 if args.smoke else 10_000
N_SEEDS = 1 if args.smoke else 3
CLF_EPOCHS = 60 if args.smoke else 300
OUT_PREFIX = "reverse_factorization_toy_SMOKE" if args.smoke else "reverse_factorization_toy"
np.random.seed(0)  # global RNG backs sample_gt_posterior -> reproducible C2ST

if not POSTERIOR_FILE.exists():
    raise SystemExit(
        f"{POSTERIOR_FILE} is missing — the forward column of this comparison is the "
        "trained estimator itself. Run experiments/toy/budget_study.py first."
    )


def train_classifier(features, labels, epochs):
    """Small MLP q(z|features), binary cross-entropy. features:(N,d), labels:(N,)."""
    d = features.shape[1]
    net = torch.nn.Sequential(
        torch.nn.Linear(d, 32), torch.nn.ReLU(),
        torch.nn.Linear(32, 32), torch.nn.ReLU(),
        torch.nn.Linear(32, 1),
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    y = labels.reshape(-1, 1).float()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(net(features), y)
        loss.backward()
        opt.step()
    return net


def predict_p1(net, features):
    with torch.no_grad():
        return torch.sigmoid(net(features)).reshape(-1)


rows = []
theta_flow_seed0 = None  # kept for the marginal figure

for seed in range(N_SEEDS):
    print(f"── seed {seed} ──")
    torch.manual_seed(seed * 1000)

    params = sample_prior(BUDGET)  # (N, 2): [theta, z]
    x = simulator(params)
    theta_col = params[:, 0:1]
    z_col = params[:, 1]

    # Continuous marginal flow q(theta | x): standard NPE on the continuous column
    # (matched to the toy MNPE continuous head). Shared by independent + reversed.
    build_fn = posterior_nn(model="nsf", num_transforms=5, hidden_features=32, num_blocks=4)
    flow = NPE(prior=Normal(torch.zeros(1), torch.ones(1)), density_estimator=build_fn)
    flow.append_simulations(theta_col, x)
    flow.train()
    q_theta = flow.build_posterior()

    # Discrete stages: q(z|x) for independent, q(z|theta,x) for reversed.
    clf_zx = train_classifier(x, z_col, CLF_EPOCHS)
    clf_ztx = train_classifier(torch.cat([theta_col, x], dim=1), z_col, CLF_EPOCHS)

    if seed == 0:
        theta_flow_seed0 = q_theta

    for x_val in X_GRID:
        xt = torch.tensor([[x_val]], dtype=torch.float32)
        theta_s = q_theta.sample((N_SAMPLES,), x=xt, show_progress_bars=False)[:, 0]

        # independent: z ~ q(z|x), drawn independently of theta
        p1_marg = float(predict_p1(clf_zx, xt))
        z_ind = (torch.rand(N_SAMPLES) < p1_marg).float().numpy()
        rows.append(eval_samples(theta_s.numpy(), z_ind, x_val, "independent", seed))

        # reversed: z ~ q(z|theta,x), conditioned on the sampled theta
        feats = torch.cat([theta_s.reshape(-1, 1), xt.repeat(N_SAMPLES, 1)], dim=1)
        p1_cond = predict_p1(clf_ztx, feats)
        z_rev = (torch.rand(N_SAMPLES) < p1_cond).float().numpy()
        rows.append(eval_samples(theta_s.numpy(), z_rev, x_val, "reversed", seed))

        print(f"  x={x_val:+.1f}: indep c2st={rows[-2]['c2st_joint']:.3f}, "
              f"reversed c2st={rows[-1]['c2st_joint']:.3f}")

# ── Forward = trained MNPE checkpoint (evaluated once) ──────────────────────
print(f"── forward ({POSTERIOR_FILE.name}) ──")
post_mnpe = torch.load(POSTERIOR_FILE, weights_only=False)
torch.manual_seed(0)
for x_val in X_GRID:
    s = post_mnpe.sample((N_SAMPLES,), x=torch.tensor([[x_val]]), show_progress_bars=False).numpy()
    rows.append(eval_samples(s[:, 0], s[:, 1], x_val, "forward", 0))
    print(f"  x={x_val:+.1f}: forward c2st={rows[-1]['c2st_joint']:.3f}")

df = pd.DataFrame(rows)
df.to_csv(RESULTS_DIR / f"{OUT_PREFIX}.csv", index=False)
print("\n── Mean over x grid (joint C2ST; lower = closer to truth) ──")
print(df.groupby("method")[["c2st_joint", "p_z1_abs_err",
                            "cond_mean_abs_err_z0", "cond_mean_abs_err_z1"]].mean().round(4).to_string())

# ── Continuous marginal at x=1 (bimodal), analytic vs flow ──────────────────
x_fig = 1.0
grid = np.linspace(-3, 3, 400)
analytic = p_theta_given_x(grid, x_fig)
with torch.no_grad():
    lp = theta_flow_seed0.log_prob(
        torch.tensor(grid, dtype=torch.float32).reshape(-1, 1),
        x=torch.tensor([[x_fig]], dtype=torch.float32),
    )
flow_dens = np.exp(lp.numpy().reshape(-1))
fig, ax = plt.subplots(figsize=(3.2, 2.4))
ax.plot(grid, analytic, color="0.2", lw=1.5, label="analytic $p(\\theta\\mid x)$")
ax.plot(grid, flow_dens, color="#d84315", lw=1.5, ls="--",
        label="learned flow $q_\\psi(\\theta\\mid x)$")
for z_val in (0, 1):
    ax.axvline(posterior_mean_given_z(x_fig, z_val), color="0.7", lw=0.7, ls=":")
ax.set_xlabel(r"$\theta$")
ax.set_ylabel("density")
ax.set_title(f"continuous marginal at $x={x_fig:g}$", fontsize="small")
ax.legend(frameon=False, fontsize="x-small")
despine(ax, offset=3)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(ARTIFACTS_DIR / f"{OUT_PREFIX}.{ext}")
plt.close(fig)
print(f"Saved {OUT_PREFIX}.csv and the diagnostic pdf/png")
