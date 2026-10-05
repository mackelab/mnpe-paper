"""MNPE budget convergence study — coal-mining-disaster changepoint model.

Trains MNPE at a range of simulation budgets with several repetitions (varying
the network training seed) and saves every result to one pickle, scored against
the Rao-Blackwell-pooled PyMC reference from ``pymc_reference.py``.

The distinguishing feature of this experiment is the size of the discrete space:
one discrete parameter with 111 categories, against the queueing model's two
with five each. Observations are 111-dimensional too (one count per year), so an
``FCEmbedding`` compresses them before the flow.

Usage:
    uv run python experiments/coal/budget_study.py

Requires ``artifacts/coal/pymc_reference_samples.npz`` from ``pymc_reference.py``.

Outputs:
    artifacts/coal/budget_study_results.pkl
    artifacts/coal/posteriors/posterior_<budget>_rep<n>.pkl

The pickle is a pipeline artifact, not a tracked result: it carries full posterior
sample arrays where the figure needs metrics. ``figures/make_coal_figure.py``
reads it through the compact export written by ``scripts/export_figure_inputs.py``.
"""

import pickle
from functools import partial

import numpy as np
import torch
from sbi.diagnostics.sbc import check_sbc, run_sbc
from sbi.inference import MNPE
from sbi.inference.posteriors.posterior_parameters import DirectPosteriorParameters
from sbi.neural_nets.embedding_nets import FCEmbedding
from sbi.neural_nets.net_builders.mixed_nets import build_mnpe
from sbi.utils.metrics import c2st

from mnpe_paper.calibration import run_discrete_calibration
from mnpe_paper.models.coal.config import (
    DISASTER_DATA,
    N_YEARS,
    YEARS,
    build_sbi_prior,
    generate_training_data,
    idx_to_year,
    simulator,
    transform_x,
)
from mnpe_paper.paths import model_artifacts_dir

# ── Settings ────────────────────────────────────────────────────────────────
BUDGETS = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000]
N_REPS = 5
SIMULATION_SEED = 42

# Architecture: Optuna-tuned (50 trials, NLTP objective, 100k sims). Every
# architectural argument is named here rather than inherited from an sbi default,
# so the published network survives an upstream default moving; see
# tests/test_architecture_pinning.py.
HPARAMS = dict(
    num_transforms=2,
    hidden_features=64,
    hidden_layers=1,
    num_bins=10,
    training_batch_size=256,
    learning_rate=4.4e-4,
    embedding_output_dim=32,
    embedding_num_hiddens=64,
    embedding_num_layers=1,
)

N_POSTERIOR_SAMPLES = 10_000
N_SBC_TEST = 500
N_SBC_POSTERIOR = 1000
SBC_TEST_SEED = 9999
N_PRED = 500  # posterior predictive samples per method
N_PRIOR_PRED = 500  # prior predictive samples

ARTIFACTS_DIR = model_artifacts_dir("coal")
OUTPUT_FILE = ARTIFACTS_DIR / "budget_study_results.pkl"
POSTERIORS_DIR = ARTIFACTS_DIR / "posteriors"
REF_FILE = ARTIFACTS_DIR / "pymc_reference_samples.npz"

# ── Setup ───────────────────────────────────────────────────────────────────
print("=" * 72)
print("MNPE Budget Study (Coal Mining Disaster)")
print(f"  Budgets: {BUDGETS}, {N_REPS} reps each")
print("=" * 72)

if not REF_FILE.exists():
    raise SystemExit(
        f"{REF_FILE} is missing. Every C2ST here is measured against the MCMC "
        "reference; run experiments/coal/pymc_reference.py first."
    )

prior = build_sbi_prior()
x_o_raw = torch.tensor(DISASTER_DATA, dtype=torch.float32)
x_o = transform_x(x_o_raw)
POSTERIORS_DIR.mkdir(parents=True, exist_ok=True)

# Load the reference (0-indexed switchpoints) and shuffle, since npz draws arrive
# ordered by chain.
ref_data = np.load(REF_FILE)
ref_switchpoint_idx = ref_data["switchpoint"]  # 0-indexed
ref_samples = np.column_stack([
    ref_data["early_rate"], ref_data["late_rate"],
    np.array([idx_to_year(int(s)) for s in ref_switchpoint_idx]),
])
rng = np.random.default_rng(0)
rng.shuffle(ref_samples)

# Pre-generate the largest budget via PyMC prior predictive; slice for smaller ones
print(f"\nGenerating {max(BUDGETS):,} training samples via PyMC...")
theta_all, x_all_raw = generate_training_data(max(BUDGETS), seed=SIMULATION_SEED)
x_all = transform_x(x_all_raw)

# SBC test set (shared across budgets) — also via PyMC for consistency
theta_test, x_test_raw = generate_training_data(N_SBC_TEST, seed=SBC_TEST_SEED)
x_test = transform_x(x_test_raw)

# Prior predictive (raw counts for plotting) — the simulator is used for speed
prior_theta = prior.sample((N_PRIOR_PRED,)).numpy()
prior_pred_obs = simulator(prior_theta)

# ── Train and evaluate ──────────────────────────────────────────────────────
budget_results = {}

for budget in BUDGETS:
    print(f"\n{'=' * 72}")
    print(f"BUDGET: {budget:,}  ({N_REPS} reps)")
    print("=" * 72)

    theta_train = theta_all[:budget]
    x_train = x_all[:budget]

    reps = []
    for rep in range(N_REPS):
        rep_seed = rep * 1000  # deterministic but different per rep
        print(f"\n  -- Rep {rep + 1}/{N_REPS} (seed={rep_seed}) --")

        # Embedding net for the 111-dim observations
        torch.manual_seed(rep_seed)
        embedding_net = FCEmbedding(
            input_dim=N_YEARS,
            output_dim=HPARAMS["embedding_output_dim"],
            num_layers=HPARAMS["embedding_num_layers"],
            num_hiddens=HPARAMS["embedding_num_hiddens"],
        )

        density_estimator_build_fn = partial(
            build_mnpe,
            num_transforms=HPARAMS["num_transforms"],
            hidden_features=HPARAMS["hidden_features"],
            hidden_layers=HPARAMS["hidden_layers"],
            num_bins=HPARAMS["num_bins"],
            embedding_net=embedding_net,
        )
        trainer = MNPE(prior=prior, density_estimator=density_estimator_build_fn)
        trainer.append_simulations(theta_train, x_train)
        trainer.train(
            training_batch_size=HPARAMS["training_batch_size"],
            learning_rate=HPARAMS["learning_rate"],
        )
        post = trainer.build_posterior(
            posterior_parameters=DirectPosteriorParameters(enable_transform=False)
        )

        post_file = POSTERIORS_DIR / f"posterior_{budget}_rep{rep}.pkl"
        with open(post_file, "wb") as f:
            pickle.dump(post, f)

        samples = post.sample((N_POSTERIOR_SAMPLES,), x=x_o).numpy()
        samples[:, 2] = np.array([idx_to_year(int(val)) for val in samples[:, 2]])

        # C2ST vs reference (per-marginal + joint continuous)
        n = min(len(samples), len(ref_samples))
        c2st_marginal = {}
        for i, name in enumerate(["early_rate", "late_rate"]):
            Xi = torch.tensor(samples[:n, i:i + 1], dtype=torch.float32)
            Yi = torch.tensor(ref_samples[:n, i:i + 1], dtype=torch.float32)
            c2st_marginal[name] = c2st(Xi, Yi, seed=1).item()
        c2st_all = c2st(
            torch.tensor(samples[:n, :2], dtype=torch.float32),
            torch.tensor(ref_samples[:n, :2], dtype=torch.float32),
            seed=1,
        ).item()

        # SBC (continuous: early_rate, late_rate)
        ranks, dap_samples = run_sbc(
            theta_test, x_test, post,
            num_posterior_samples=N_SBC_POSTERIOR,
            reduce_fns="marginals",
            num_workers=1, show_progress_bar=True,
        )
        theta_for_check = prior.sample((len(theta_test),))
        sbc_results = check_sbc(
            ranks, theta_for_check, dap_samples,
            num_posterior_samples=N_SBC_POSTERIOR,
        )
        sbc_ks_pvals = [sbc_results["ks_pvals"][i].item() for i in range(2)]

        # Joint SBC via log_prob
        estimator = post.posterior_estimator

        def joint_log_prob(theta, x, _est=estimator):
            theta_3d = theta.unsqueeze(0)
            x_2d = x.unsqueeze(0).expand(theta.shape[0], -1)
            return _est.log_prob(theta_3d, condition=x_2d).squeeze(0)

        joint_ranks, joint_dap = run_sbc(
            theta_test, x_test, post,
            num_posterior_samples=N_SBC_POSTERIOR,
            reduce_fns=joint_log_prob,
            num_workers=1, show_progress_bar=True,
        )
        joint_sbc_results = check_sbc(
            joint_ranks, theta_for_check, joint_dap,
            num_posterior_samples=N_SBC_POSTERIOR,
        )
        joint_sbc_ks_pval = joint_sbc_results["ks_pvals"][0].item()

        # ECE (top-label only; classwise is uninformative across 111 classes)
        ece, probs, true_disc = run_discrete_calibration(
            theta_test, x_test, post,
            discrete_indices=[2],
            num_bins=10, method="top_label",
        )

        # Posterior predictive (first rep only, to save space)
        if rep == 0:
            pred_theta = samples[:N_PRED].copy()
            pred_theta[:, 2] -= YEARS[0]
            pred_obs = simulator(pred_theta)
        else:
            pred_obs = None

        reps.append({
            "samples": samples,
            "c2st_marginal": c2st_marginal,
            "c2st_all": c2st_all,
            "sbc_ks_pvals": sbc_ks_pvals,
            "sbc_ranks": ranks.numpy(),
            "joint_sbc_ks_pval": joint_sbc_ks_pval,
            "joint_sbc_ranks": joint_ranks.numpy(),
            "ece": ece.tolist(),
            "ece_probs": probs.numpy(),
            "ece_true_vals": true_disc.numpy(),
            "pred_obs": pred_obs,
        })

        print(f"    C2ST joint: {c2st_all:.3f}  "
              f"early={c2st_marginal['early_rate']:.3f}  "
              f"late={c2st_marginal['late_rate']:.3f}  "
              f"ECE={ece[0]:.3f}")

    # Summary across reps
    c2st_vals = [r["c2st_all"] for r in reps]
    print(f"\n  Summary ({N_REPS} reps):")
    print(f"    C2ST joint: {np.mean(c2st_vals):.3f} +/- "
          f"{np.std(c2st_vals) / np.sqrt(N_REPS):.3f} (SEM)")

    budget_results[budget] = {"reps": reps}

# ── Reference predictive ────────────────────────────────────────────────────
ref_pred_theta = ref_samples[:N_PRED].copy()
ref_pred_theta[:, 2] -= YEARS[0]  # year to 0-indexed for the simulator
ref_pred_obs = simulator(ref_pred_theta)

# ── Save ────────────────────────────────────────────────────────────────────
output = {
    "budgets": BUDGETS,
    "n_reps": N_REPS,
    "hparams": HPARAMS,
    "x_o": DISASTER_DATA,
    "observation_transform": "sqrt",
    "ref_samples": ref_samples,
    "ref_pred_obs": ref_pred_obs,
    "prior_pred_obs": prior_pred_obs,
    "budget_results": budget_results,
    "reference_file": REF_FILE.name,
    "discrete_values": list(range(YEARS[0], YEARS[-1] + 1)),
}

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(output, f)
print(f"\nSaved results to {OUTPUT_FILE}")
