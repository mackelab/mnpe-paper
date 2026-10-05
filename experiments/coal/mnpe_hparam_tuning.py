"""Optuna hyperparameter tuning for MNPE — coal-mining-disaster changepoint model.

Objective: negative log test posterior (NLTP) on a held-out (theta, x) set,
following Lueckmann et al. (2021) and the sbi hyperparameter-tuning guide.

Because the observations are 111-dimensional (one count per year), an
``FCEmbedding`` is always used to compress them before the flow, and the
embedding's own shape is part of the search space.

Setup:
  - 100k sims from the prior predictive (so there is no model mismatch)
  - 90k train / 10k test split
  - sbi handles its own internal train/validation split for early stopping within
    the 90k training set

This script documents where the architecture in ``budget_study.py`` came from; it
does not need to be rerun to reproduce any published number, since the selected
values are named there and pinned by ``tests/test_architecture_pinning.py``.

Usage:
    uv run python experiments/coal/mnpe_hparam_tuning.py

Outputs:
    artifacts/coal/hparam_study.db        (Optuna SQLite, resumable)
    artifacts/coal/hparam_best_params.json (best hyperparameters)
"""

import json
from functools import partial

import optuna
import torch
from sbi.inference import MNPE
from sbi.inference.posteriors.posterior_parameters import DirectPosteriorParameters
from sbi.neural_nets.embedding_nets import FCEmbedding
from sbi.neural_nets.net_builders.mixed_nets import build_mnpe

from mnpe_paper.models.coal.config import (
    N_YEARS,
    build_sbi_prior,
    generate_training_data,
    transform_x,
)
from mnpe_paper.paths import model_artifacts_dir

# ── Settings ────────────────────────────────────────────────────────────────
N_SIMULATIONS = 100_000
SIMULATION_SEED = 42
TEST_FRACTION = 0.1  # 10% held out for NLTP evaluation
N_TRIALS = 50
OPTUNA_SEED = 42
NLTP_BATCH = 2_000  # subsample of the test set for fast log_prob evaluation

ARTIFACTS_DIR = model_artifacts_dir("coal")
DB_FILE = ARTIFACTS_DIR / "hparam_study.db"
BEST_PARAMS_FILE = ARTIFACTS_DIR / "hparam_best_params.json"

# ── Prepare data ────────────────────────────────────────────────────────────
print("=" * 72)
print("MNPE Hyperparameter Tuning — Coal Mining Disaster (Optuna, NLTP)")
print(f"Budget: {N_SIMULATIONS:,} sims | Trials: {N_TRIALS}")
print(f"Observation dim: {N_YEARS} | Transform: sqrt")
print("=" * 72)

prior = build_sbi_prior()

print(f"\nGenerating {N_SIMULATIONS:,} training samples via PyMC...")
theta_all, x_all_raw = generate_training_data(N_SIMULATIONS, seed=SIMULATION_SEED)
x_all = transform_x(x_all_raw)
n_total = len(theta_all)

# Train / test split
n_test = int(n_total * TEST_FRACTION)
n_train = n_total - n_test
theta_train, theta_test = theta_all[:n_train], theta_all[n_train:]
x_train, x_test = x_all[:n_train], x_all[n_train:]
print(f"  Train: {n_train:,} | Test (NLTP): {n_test:,}")

# Subsample the test set for faster evaluation
n_eval = min(NLTP_BATCH, n_test)
theta_eval = theta_test[:n_eval]
x_eval = x_test[:n_eval]


# ── Objective ───────────────────────────────────────────────────────────────

def objective(trial: optuna.Trial) -> float:
    # Flow architecture
    num_transforms = trial.suggest_int("num_transforms", 2, 10)
    hidden_features = trial.suggest_categorical("hidden_features", [32, 64, 128, 256])
    hidden_layers = trial.suggest_int("hidden_layers", 1, 5)
    num_bins = trial.suggest_int("num_bins", 3, 12)

    # Training
    training_batch_size = trial.suggest_categorical(
        "training_batch_size", [128, 256, 512, 1024]
    )
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)

    # Embedding net (always used — 111-dim observations need compression)
    embedding_dim = trial.suggest_categorical("embedding_dim", [8, 16, 32, 64])
    embedding_layers = trial.suggest_int("embedding_layers", 1, 3)
    embedding_hiddens = trial.suggest_categorical("embedding_hiddens", [32, 64, 128])
    embedding_net = FCEmbedding(
        input_dim=N_YEARS,
        output_dim=embedding_dim,
        num_layers=embedding_layers,
        num_hiddens=embedding_hiddens,
    )

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
    except Exception as e:  # noqa: BLE001 — a failed trial must not stop the study
        print(f"  Trial {trial.number} FAILED during training: {e}")
        return float("inf")

    # Record sbi's internal best validation loss
    sbi_val_loss = trainer._summary["best_validation_loss"][-1]
    trial.set_user_attr("sbi_val_loss", sbi_val_loss)

    posterior = trainer.build_posterior(
        posterior_parameters=DirectPosteriorParameters(enable_transform=False)
    )

    # NLTP: negative log probability on the held-out test set. theta must be
    # (sample, batch, event) for log_prob_batched to match x's batch dimension:
    # (1, 2000, 3) paired with (2000, 111).
    try:
        with torch.no_grad():
            log_probs = posterior.log_prob_batched(
                theta_eval.unsqueeze(0),  # (batch, event) → (1, batch, event)
                x=x_eval,
                norm_posterior=False,
            ).squeeze(0)  # (1, batch) → (batch,)
    except Exception as e:  # noqa: BLE001 — a failed trial must not stop the study
        print(f"  Trial {trial.number} FAILED during log_prob: {e}")
        return float("inf")

    # Filter out -inf (prior support violations, which should be rare)
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

    print(
        f"  Trial {trial.number}: NLTP={nltp:.4f} | "
        f"sbi_val={sbi_val_loss:.4f} | "
        f"transforms={num_transforms}, hidden={hidden_features}x{hidden_layers}, "
        f"bins={num_bins}, lr={learning_rate:.1e}, batch={training_batch_size}, "
        f"emb=FC({embedding_dim}x{embedding_layers}, h={embedding_hiddens})"
    )

    return nltp


# ── Run optimization ────────────────────────────────────────────────────────
print(f"\n── Running Optuna ({N_TRIALS} trials, SQLite: {DB_FILE}) ──")

sampler = optuna.samplers.TPESampler(seed=OPTUNA_SEED, n_startup_trials=10)
study = optuna.create_study(
    study_name="coal_hparam_nltp",
    direction="minimize",
    sampler=sampler,
    storage=f"sqlite:///{DB_FILE}",
    load_if_exists=True,  # resumable
)

# Enqueue a hand-chosen architecture as a baseline
study.enqueue_trial(
    {
        "num_transforms": 4,
        "hidden_features": 128,
        "hidden_layers": 3,
        "num_bins": 5,
        "training_batch_size": 256,
        "learning_rate": 5e-4,
        "embedding_dim": 16,
        "embedding_layers": 2,
        "embedding_hiddens": 64,
    },
    skip_if_exists=True,
)

# Also enqueue a smaller baseline
study.enqueue_trial(
    {
        "num_transforms": 2,
        "hidden_features": 64,
        "hidden_layers": 2,
        "num_bins": 5,
        "training_batch_size": 256,
        "learning_rate": 5e-4,
        "embedding_dim": 16,
        "embedding_layers": 2,
        "embedding_hiddens": 64,
    },
    skip_if_exists=True,
)

n_remaining = max(0, N_TRIALS - len(study.trials))
if n_remaining > 0:
    study.optimize(objective, n_trials=n_remaining, show_progress_bar=True)
else:
    print(f"  Study already has {len(study.trials)} trials, skipping optimization.")

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
    f"{'batch':>5s}  {'emb_dim':>7s}  {'emb_lay':>7s}  {'emb_h':>5s}"
)
print("-" * 95)

completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
trials_sorted = sorted(completed, key=lambda t: t.value)
for t in trials_sorted[:10]:
    p = t.params
    vl = t.user_attrs.get("sbi_val_loss", float("nan"))
    print(
        f"{t.number:>4d}  {t.value:>8.4f}  {vl:>8.4f}  {p['num_transforms']:>3d}  "
        f"{p['hidden_features']:>6d}  {p['hidden_layers']:>3d}  "
        f"{p['num_bins']:>4d}  {p['learning_rate']:>8.1e}  "
        f"{p['training_batch_size']:>5d}  "
        f"{p['embedding_dim']:>7d}  {p['embedding_layers']:>7d}  "
        f"{p['embedding_hiddens']:>5d}"
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
        "observation_transform": "sqrt",
        "observation_dim": N_YEARS,
    },
}
with open(BEST_PARAMS_FILE, "w") as f:
    json.dump(best_output, f, indent=2)
print(f"\nSaved best params to {BEST_PARAMS_FILE}")
