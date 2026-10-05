"""Dequantization baseline: standard NPE on uniformly dequantized discrete params.

Backs Appendix B.3 and Table B-1, and the grey open circle in Figure 3b: how
standard NPE performs on dequantized discrete parameters, as a baseline.

Baseline: make the 0-indexed server counts continuous via theta_d + u with
u ~ U[0,1) (uniform dequantization), train completely standard NPE (zuko_nsf,
same tuned architecture and budget as MNPE), recover discrete values at
inference by flooring, and categorical probabilities as the fraction of
posterior samples per unit interval.

Evaluated with the same suite as MNPE (same test set, same reference):
  - joint C2ST vs the deterministic reference at x_o
  - discrete ECE (B=10) vs the finite-sample baseline, N_ECE test cases
  - continuous SBC KS p-values
  - raw flow leakage outside [0, K), measured on the estimator BEFORE
    DirectPosterior's rejection correction (post-rejection it is 0 by
    construction)

Outputs:
  dequantization_baseline_results.csv — one row per seed + MNPE comparison row
  dequantization_baseline.pkl         — samples and per-case probs per seed

Run:  uv run python experiments/queue/ablations/dequantization_baseline.py
"""

import pickle
import time

import numpy as np
import pandas as pd
import torch
from sbi.diagnostics.sbc import check_uniformity_frequentist
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from sbi.utils import MultipleIndependent
from sbi.utils.metrics import c2st
from torch.distributions import LogNormal, Uniform

from mnpe_paper.calibration import compute_ece, ece_baseline_analytical
from mnpe_paper.models.queue.config import (
    GAMMA_MEDIAN,
    GAMMA_SIGMA,
    MODEL_VERSION,
    MU1_MEDIAN,
    MU1_SIGMA,
    MU2_MEDIAN,
    MU2_SIGMA,
    N_C,
    generate_reference_observation,
    generate_training_data,
    transform_x,
)
from mnpe_paper.models.queue.spec import SBC_TEST_SEED
from mnpe_paper.models.queue.spec import TUNED_FLOW_HPARAMS as TUNED
from mnpe_paper.paths import model_artifacts_dir, model_results_dir

ARTIFACTS_DIR = model_artifacts_dir("queue")
RESULTS_DIR = model_results_dir("queue")
BUDGET = 100_000
N_SEEDS = 3
N_POSTERIOR_SAMPLES = 10_000
N_ECE_TEST = 300  # per-case posterior sampling is the expensive part
M_ECE = 1_000  # samples per test case for probs/ranks
B_ECE = 10


def build_dequant_prior():
    """Continuous priors as in config, U[0, N_C) boxes for dequantized discrete."""
    return MultipleIndependent([
        LogNormal(torch.log(torch.tensor([GAMMA_MEDIAN])), torch.tensor([GAMMA_SIGMA])),
        LogNormal(torch.log(torch.tensor([MU1_MEDIAN])), torch.tensor([MU1_SIGMA])),
        LogNormal(torch.log(torch.tensor([MU2_MEDIAN])), torch.tensor([MU2_SIGMA])),
        Uniform(torch.zeros(1), torch.full((1,), float(N_C))),
        Uniform(torch.zeros(1), torch.full((1,), float(N_C))),
    ])


def dequantize(theta, rng):
    """theta (N,5) with 0-indexed discrete in cols 3,4 -> continuous."""
    out = theta.clone()
    noise = torch.from_numpy(rng.uniform(size=(len(theta), 2))).float()
    out[:, 3:5] = theta[:, 3:5] + noise
    return out


def floor_discrete(samples):
    """Floor dequantized cols back to categories; report out-of-range mass."""
    disc = torch.floor(samples[:, 3:5])
    oor = float(((disc < 0) | (disc >= N_C)).float().mean())
    disc = disc.clamp(0, N_C - 1)
    return torch.cat([samples[:, :3], disc], dim=1), oor


# ── Shared data (identical seeds to the MNPE pipeline) ───────────────────
print(f"Generating training data (budget={BUDGET:,}, seed=42)...")
theta_all, x_all_raw = generate_training_data(BUDGET, seed=42)
x_all = transform_x(x_all_raw)

theta_test, x_test_raw = generate_training_data(N_ECE_TEST, seed=SBC_TEST_SEED)
x_test = transform_x(x_test_raw)

obs = generate_reference_observation(seed=42)
x_o = transform_x(torch.tensor(obs["x_array"], dtype=torch.float32).unsqueeze(0))

ref = np.load(ARTIFACTS_DIR / "deterministic_reference.npz")
ref_joint = torch.tensor(
    np.stack(
        [ref["gamma"], ref["mu1"], ref["mu2"], ref["c1"] - 2.0, ref["c2"] - 2.0], axis=1
    ),
    dtype=torch.float32,
)

prior = build_dequant_prior()

rows = []
store = {}
for seed in range(N_SEEDS):
    print(f"\n── Dequantized NPE, seed {seed} ──")
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed * 1000)

    theta_deq = dequantize(theta_all, rng)
    build_fn = posterior_nn(
        model="nsf",
        num_transforms=TUNED["num_transforms"],
        hidden_features=TUNED["hidden_features"],
        num_bins=TUNED["num_bins"],
    )
    trainer = NPE(prior=prior, density_estimator=build_fn)
    trainer.append_simulations(theta_deq, x_all)
    t0 = time.perf_counter()
    trainer.train(
        training_batch_size=TUNED["training_batch_size"],
        learning_rate=TUNED["learning_rate"],
    )
    train_time = time.perf_counter() - t0
    post = trainer.build_posterior()
    print(f"  trained in {train_time:.0f}s")

    # Raw flow leakage outside [0, N_C): measured on the estimator directly,
    # BEFORE DirectPosterior's rejection correction (which forces it to 0).
    with torch.no_grad():
        raw = trainer._neural_net.sample(
            torch.Size([N_POSTERIOR_SAMPLES]), condition=x_o
        ).reshape(N_POSTERIOR_SAMPLES, -1)
    raw_disc = raw[:, 3:5]
    oor = float(((raw_disc < 0) | (raw_disc >= N_C)).any(dim=1).float().mean())

    # Joint C2ST at x_o (floored discrete; rejection-corrected samples)
    samples = post.sample((N_POSTERIOR_SAMPLES,), x=x_o, show_progress_bars=False)
    samples_floored, _ = floor_discrete(samples)
    idx = torch.randperm(len(ref_joint))[:N_POSTERIOR_SAMPLES]
    c2st_all = float(c2st(samples_floored, ref_joint[idx]))
    print(f"  joint C2ST vs reference: {c2st_all:.4f} (raw flow leakage {oor:.4f})")

    # Per-test-case probs (ECE) and continuous ranks (SBC)
    probs = torch.zeros(N_ECE_TEST, 2, N_C)
    ranks = torch.zeros(N_ECE_TEST, 3)
    for i in range(N_ECE_TEST):
        s = post.sample((M_ECE,), x=x_test[i], show_progress_bars=False)
        s_floored, _ = floor_discrete(s)
        for j in range(2):
            for k in range(N_C):
                probs[i, j, k] = (s_floored[:, 3 + j] == k).float().mean()
        ranks[i] = (s[:, :3] < theta_test[i, :3]).sum(dim=0).float()
        if (i + 1) % 100 == 0:
            print(f"  test cases {i + 1}/{N_ECE_TEST}")

    ece = compute_ece(theta_test[:, 3:5].long(), probs, num_bins=B_ECE)
    sbc_pvals = np.atleast_1d(
        np.asarray(check_uniformity_frequentist(ranks, M_ECE), dtype=float)
    )
    baseline = ece_baseline_analytical(N_ECE_TEST, B=B_ECE, K=N_C)

    rows.append({
        "method": "dequant_npe",
        "seed": seed,
        "train_time_s": train_time,
        "c2st_all": c2st_all,
        "raw_leakage": oor,
        "ece_c1": float(ece[0]),
        "ece_c2": float(ece[1]),
        "ece_baseline_analytical": baseline,
        "sbc_pval_gamma": float(sbc_pvals[0]),
        "sbc_pval_mu1": float(sbc_pvals[1]),
        "sbc_pval_mu2": float(sbc_pvals[2]),
    })
    store[seed] = {"samples_x_o": samples_floored.numpy(), "probs": probs.numpy()}
    print(
        f"  ECE c1={ece[0]:.4f}, c2={ece[1]:.4f} (baseline {baseline:.4f}); "
        f"SBC p: {sbc_pvals.round(3)}"
    )

df = pd.DataFrame(rows)
df.to_csv(
    RESULTS_DIR / "dequantization_baseline_results.csv",
    index=False,
)
with open(ARTIFACTS_DIR / "dequantization_baseline.pkl", "wb") as f:
    pickle.dump(store, f)

print("\n── Summary (dequantized NPE, mean over seeds) ──")
print(df[["c2st_all", "raw_leakage", "ece_c1", "ece_c2"]].mean().round(4).to_string())

# Read from the results file rather than restated here. Both arms report
# the mean across runs; quoting a selected best rep against a competitor mean
# is the asymmetry that would bias the comparison.
mnpe_results_file = (
    ARTIFACTS_DIR / "budget_study_results_n1000.pkl"
)
if not mnpe_results_file.exists():
    mnpe_results_file = ARTIFACTS_DIR / "budget_study_results.pkl"

print("\n── MNPE comparison ──")
if mnpe_results_file.exists():
    with open(mnpe_results_file, "rb") as f:
        mnpe_study = pickle.load(f)
    largest = max(mnpe_study["budgets"])
    mnpe_c2st = np.array(
        [r["c2st_all"] for r in mnpe_study["budget_results"][largest]["reps"]]
    )
    dequant_c2st = df["c2st_all"].to_numpy()
    print(f"  source: {mnpe_results_file.name}, budget {largest:,}")
    print(
        f"  MNPE     joint C2ST {mnpe_c2st.mean():.4f} +/- "
        f"{mnpe_c2st.std(ddof=1):.4f}  (n={len(mnpe_c2st)} reps, "
        f"min {mnpe_c2st.min():.4f})"
    )
    print(
        f"  dequant  joint C2ST {dequant_c2st.mean():.4f} +/- "
        f"{dequant_c2st.std(ddof=1):.4f}  (n={len(dequant_c2st)} seeds, "
        f"min {dequant_c2st.min():.4f})"
    )
    print(f"  gap (dequant - MNPE): {dequant_c2st.mean() - mnpe_c2st.mean():+.4f}")
else:
    print(f"  [skip] {mnpe_results_file.name} not found")

print(
    "\nNote: the ECE columns above are NOT matched to the MNPE ECE analyses.\n"
    "Dequant uses 300 test cases and estimates class probabilities from\n"
    "posterior samples, while MNPE uses 1,000 cases and reads probabilities\n"
    "from the categorical head directly, so dequant ECE carries extra Monte\n"
    "Carlo noise. Compare joint C2ST, not ECE, until those are matched."
)
print(f"Saved dequantization baseline outputs for {MODEL_VERSION}")
