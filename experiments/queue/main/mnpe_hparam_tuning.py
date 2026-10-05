"""
Optuna hyperparameter tuning for MNPE — 5-category discrete setup (c ∈ {2,3,4,5,6}).

Objective: Negative Log Test Posterior (NLTP) on a held-out (theta, x) set,
following the approach from Lueckmann et al. 2021 and the sbi docs
(https://sbi.readthedocs.io/en/latest/how_to_guide/21_hyperparameter_tuning.html).

The NLTP evaluates how well the learned density estimator q_phi(theta|x) assigns
probability to ground-truth (theta, x) pairs. Lower NLTP = better fit.

Observation preprocessing: log1p is applied to queue length dimensions (q1, q2)
before training to reduce skewness. Without this, z-scoring compresses the
informative low-queue range into a tiny sliver (50% of q1 < 0.10, but max ~10).

Setup:
  - 100k sims from generate_training_data() (PyMC prior predictive)
  - 90k train / 10k test split
  - sbi handles its own internal train/val split (for early stopping) within
    the 90k training set

Usage:
    uv run python experiments/queue/main/mnpe_hparam_tuning.py

Outputs:
    hparam_study.db         (Optuna SQLite, resumable)
    hparam_best_params.json  (best hyperparameters)
"""

import json
from functools import partial

import optuna
import torch
from sbi.inference import MNPE
from sbi.inference.posteriors.posterior_parameters import DirectPosteriorParameters
from sbi.neural_nets.embedding_nets import FCEmbedding
from sbi.neural_nets.net_builders.mixed_nets import build_mnpe

from mnpe_paper.models.queue.config import (
    build_sbi_prior,
    generate_training_data,
    transform_x,
)
from mnpe_paper.models.queue.spec import SIMULATION_SEED
from mnpe_paper.paths import model_artifacts_dir

# ── Settings ────────────────────────────────────────────────────────────────
N_SIMULATIONS = 100_000
TEST_FRACTION = 0.1  # 10% held out for NLTP evaluation
N_TRIALS = 50
OPTUNA_SEED = 42
NLTP_BATCH = 2_000  # subsample of test set for fast log_prob evaluation

ARTIFACTS_DIR = model_artifacts_dir("queue")
DB_FILE = str(ARTIFACTS_DIR / "hparam_study.db")
BEST_PARAMS_FILE = str(
    ARTIFACTS_DIR / "hparam_best_params.json"
)


# ── Prepare data ────────────────────────────────────────────────────────────
print("=" * 72)
print("MNPE Hyperparameter Tuning (Optuna, NLTP objective)")
print(f"  {N_SIMULATIONS:,} sims, {N_TRIALS} trials")
print("=" * 72)

prior = build_sbi_prior()

print(f"\nGenerating {N_SIMULATIONS:,} training samples...")
theta_all, x_all = generate_training_data(N_SIMULATIONS, seed=SIMULATION_SEED)
n_total = len(theta_all)
x_all = transform_x(x_all)

# Train / test split (test set = held-out for NLTP, never seen during training)
n_test = int(n_total * TEST_FRACTION)
n_train = n_total - n_test
theta_train, theta_test = theta_all[:n_train], theta_all[n_train:]
x_train, x_test = x_all[:n_train], x_all[n_train:]
print(f"  Train: {n_train:,} | Test (NLTP): {n_test:,}")

# Subsample test set for faster evaluation
n_eval = min(NLTP_BATCH, n_test)
theta_eval = theta_test[:n_eval]
x_eval = x_test[:n_eval]


# ── Objective ───────────────────────────────────────────────────────────────

def objective(trial: optuna.Trial) -> float:
    # Architecture
    num_transforms = trial.suggest_int("num_transforms", 2, 10)
    hidden_features = trial.suggest_categorical(
        "hidden_features", [32, 64, 128, 256]
    )
    hidden_layers = trial.suggest_int("hidden_layers", 1, 5)
    num_bins = trial.suggest_int("num_bins", 3, 12)

    # Training
    training_batch_size = trial.suggest_categorical(
        "training_batch_size", [128, 256, 512, 1024]
    )
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)

    # Embedding net for the condition (x)
    use_embedding = trial.suggest_categorical("use_embedding", [False, True])

    if use_embedding:
        embedding_dim = trial.suggest_categorical("embedding_dim", [16, 32, 64])
        embedding_layers = trial.suggest_int("embedding_layers", 1, 3)
        embedding_hiddens = trial.suggest_categorical(
            "embedding_hiddens", [32, 64, 128]
        )
        embedding_net = FCEmbedding(
            input_dim=x_train.shape[1],
            output_dim=embedding_dim,
            num_layers=embedding_layers,
            num_hiddens=embedding_hiddens,
        )
    else:
        embedding_net = torch.nn.Identity()

    density_estimator_build_fn = partial(
        build_mnpe,
        num_transforms=num_transforms,
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        num_bins=num_bins,
        embedding_net=embedding_net,
    )

    trainer = MNPE(prior=prior, density_estimator=density_estimator_build_fn)
    trainer.append_simulations(theta_train, x_train)

    try:
        trainer.train(
            training_batch_size=training_batch_size,
            learning_rate=learning_rate,
            show_train_summary=False,
        )
    except Exception as e:
        print(f"  Trial {trial.number} FAILED during training: {e}")
        return float("inf")

    # Record sbi's internal best validation loss
    sbi_val_loss = trainer._summary["best_validation_loss"][-1]
    trial.set_user_attr("sbi_val_loss", sbi_val_loss)

    posterior = trainer.build_posterior(
        posterior_parameters=DirectPosteriorParameters(enable_transform=False)
    )

    # NLTP: negative log probability on the held-out test set
    # theta must be (sample, batch, event) for log_prob_batched to match
    # x's batch dimension: (1, 2000, 5) paired with (2000, 5).
    try:
        with torch.no_grad():
            log_probs = posterior.log_prob_batched(
                theta_eval.unsqueeze(0),  # (batch, event) → (1, batch, event)
                x=x_eval,
                norm_posterior=False,
            ).squeeze(0)  # (1, batch) → (batch,)
    except Exception as e:
        print(f"  Trial {trial.number} FAILED during log_prob: {e}")
        return float("inf")

    # Filter out -inf (prior support violations, should be rare)
    finite_mask = torch.isfinite(log_probs)
    n_finite = finite_mask.sum().item()
    if n_finite < n_eval * 0.5:
        print(
            f"  Trial {trial.number}: {n_eval - n_finite}/{n_eval} "
            f"non-finite log_probs, skipping."
        )
        return float("inf")

    nltp = -log_probs[finite_mask].mean().item()
    trial.set_user_attr("n_finite", n_finite)

    emb_str = (
        f"FC({embedding_dim}x{embedding_layers})"
        if use_embedding else "Identity"
    )
    print(
        f"  Trial {trial.number}: NLTP={nltp:.4f} | "
        f"sbi_val={sbi_val_loss:.4f} | "
        f"transforms={num_transforms}, hidden={hidden_features}x{hidden_layers}, "
        f"bins={num_bins}, lr={learning_rate:.1e}, batch={training_batch_size}, "
        f"emb={emb_str}"
    )

    return nltp


# ── Run optimization ────────────────────────────────────────────────────────
print(f"\n── Running Optuna ({N_TRIALS} trials, SQLite: {DB_FILE}) ──")

sampler = optuna.samplers.TPESampler(seed=OPTUNA_SEED, n_startup_trials=10)
study = optuna.create_study(
    study_name="mnpe_hparam_nltp",
    direction="minimize",
    sampler=sampler,
    storage=f"sqlite:///{DB_FILE}",
    load_if_exists=True,  # resumable
)

# Enqueue a known-good configuration as a baseline trial
study.enqueue_trial(
    {
        "num_transforms": 6,
        "hidden_features": 64,
        "hidden_layers": 3,
        "num_bins": 5,
        "training_batch_size": 256,
        "learning_rate": 5e-4,
        "use_embedding": False,
    },
    skip_if_exists=True,
)

# Also enqueue sbi defaults (with hidden_features rounded to nearest in search space)
study.enqueue_trial(
    {
        "num_transforms": 2,
        "hidden_features": 64,
        "hidden_layers": 2,
        "num_bins": 5,
        "training_batch_size": 256,
        "learning_rate": 5e-4,
        "use_embedding": False,
    },
    skip_if_exists=True,
)

n_remaining = max(0, N_TRIALS - len(study.trials))
if n_remaining > 0:
    study.optimize(objective, n_trials=n_remaining, show_progress_bar=True)
else:
    print(
        f"  Study already has {len(study.trials)} trials, skipping optimization."
    )

# ── Report ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("OPTIMIZATION RESULTS")
print("=" * 72)

print(f"\nBest trial: #{study.best_trial.number}")
print(f"Best NLTP: {study.best_value:.4f}")
print("Best hyperparameters:")
for key, val in study.best_params.items():
    print(f"  {key}: {val}")

best_attrs = study.best_trial.user_attrs
if "sbi_val_loss" in best_attrs:
    print(f"  sbi_val_loss: {best_attrs['sbi_val_loss']:.4f}")

# Top 10
print("\nTop 10 trials:")
print(
    f"{'#':>4s}  {'NLTP':>8s}  {'sbi_val':>8s}  {'xfm':>3s}  "
    f"{'hidden':>6s}  {'lay':>3s}  {'bins':>4s}  {'lr':>8s}  "
    f"{'batch':>5s}  {'embedding':>12s}"
)
print("-" * 85)

completed = [
    t for t in study.trials
    if t.state == optuna.trial.TrialState.COMPLETE
]
trials_sorted = sorted(completed, key=lambda t: t.value)
for t in trials_sorted[:10]:
    p = t.params
    emb = "Identity"
    if p.get("use_embedding", False):
        emb = f"FC({p.get('embedding_dim', '?')})"
    vl = t.user_attrs.get("sbi_val_loss", float("nan"))
    print(
        f"{t.number:>4d}  {t.value:>8.4f}  {vl:>8.4f}  {p['num_transforms']:>3d}  "
        f"{p['hidden_features']:>6d}  {p['hidden_layers']:>3d}  "
        f"{p['num_bins']:>4d}  {p['learning_rate']:>8.1e}  "
        f"{p['training_batch_size']:>5d}  {emb:>12s}"
    )

# Save best params
best_output = {
    "best_params": study.best_params,
    "best_nltp": study.best_value,
    "best_sbi_val_loss": best_attrs.get("sbi_val_loss"),
    "best_trial_number": study.best_trial.number,
    "n_trials": len(completed),
    "settings": {
        "n_simulations": N_SIMULATIONS,
        "test_fraction": TEST_FRACTION,
        "nltp_batch": NLTP_BATCH,
        "n_train": n_train,
        "n_test": n_test,
        "observation_transform": "log1p on dims [3, 4] (queue lengths)",
    },
}
with open(BEST_PARAMS_FILE, "w") as f:
    json.dump(best_output, f, indent=2)
print(f"\nSaved best params to {BEST_PARAMS_FILE}")
