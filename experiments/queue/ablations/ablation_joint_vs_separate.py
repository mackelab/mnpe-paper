"""Joint versus separate training of the two MNPE heads.

Backs Appendix B.2 and a sentence in Section 3.1: whether the two subnetworks
could be trained independently, and what changes if they are. The MNPE loss
decomposes exactly,

    -log q(theta_d, theta_c | x) = -log q(theta_d | x) - log q(theta_c | theta_d, x)

so joint and separate training differ only if the heads share parameters. In
this project they do not: `eval.build_tuned_mnpe`, which budget_study.py and
train_queue_1M.py use, passes `embedding_net=Identity()`, the continuous head conditions on the raw
0-indexed theta_d, and `MixedDensityEstimator.log_prob` returns the plain sum
`disc_log_prob + cont_log_prob`. Part 1 of this script verifies that claim
numerically rather than asserting it: disjoint parameter sets, a parameter-free
condition embedding, and gradients of the joint loss that equal the gradients of
each head's own term.

Two couplings survive the decomposition, and they are why the ablation is still
worth running empirically. Neither is a gradient coupling -- the optimizer simply
does not apply raw gradients:

  (a) gradient-norm clipping is global. `clip_grad_norm_(net.parameters(), 5.0)`
      (sbi/inference/trainers/base.py:1089) takes the norm over BOTH heads, so
      the discrete head's updates get rescaled by a factor driven mostly by the
      continuous head. Measured at seed 0: total norm ~21 at convergence
      (clipped on 30/30 batches) while the discrete head alone is ~0.9 and would
      never clip. Adam's sqrt(v) normalization absorbs much of a uniform
      rescale, so treat this as a real channel of unquantified size.
  (b) checkpoint selection is shared. `_converged` (base.py:1178-1188) snapshots
      the whole state_dict at the epoch minimizing the SUMMED validation loss,
      so both heads come back from one epoch chosen by the sum.

Part 2 measures the effect of removing both: the separate arm trains each head
under its own loss term -- so clipping and early stopping see only that head --
composes the two trained heads into one estimator, and runs the standard
evaluation suite against the joint arm.

Both arms use the same budget, the same tuned hyperparameters, and the same
per-rep seed. All three trainings are seeded before the trainer is built and the
masking only changes a function called later, so they consume the RNG in the
same order and share the train/validation split (base.py:449), the initial
weights, and the batch ordering. The only differences are (a) and (b) above.

Evaluation mirrors reeval_budget_study.py exactly (same reference, same 1,000-case
seed-9999 test set, same C2ST seed), so the joint arm here is directly comparable
to the 100k point of the budget study. Nothing protected is overwritten: outputs
go to `ablation_joint_vs_separate.{pkl,csv}` and no posterior is written into
`posteriors/`.

Run:  uv run python experiments/queue/ablations/ablation_joint_vs_separate.py [--smoke]
"""

import argparse
import copy
import pickle
from functools import partial

import numpy as np
import pandas as pd
import torch
from sbi.inference import MNPE
from sbi.inference.posteriors.posterior_parameters import DirectPosteriorParameters

from mnpe_paper.models.queue.config import (
    MODEL_VERSION,
    build_sbi_prior,
    generate_reference_observation,
    generate_training_data,
    transform_x,
)
from mnpe_paper.models.queue.eval import build_tuned_mnpe, evaluate_posterior
from mnpe_paper.models.queue.nets import HeadMaskedMDE  # also loads old __main__ part files
from mnpe_paper.models.queue.spec import SBC_TEST_SEED, SIMULATION_SEED, TUNED_HPARAMS
from mnpe_paper.paths import model_artifacts_dir, model_results_dir

# ── Settings ────────────────────────────────────────────────────────────────
# Evaluation settings: must mirror reeval_budget_study.py / eval_queue_1M.py.
N_POSTERIOR_SAMPLES = 10_000
N_SBC_TEST = 1_000
N_SBC_POSTERIOR = 1000

parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true")
args = parser.parse_args()
BUDGET = 2_000 if args.smoke else 100_000
N_SEEDS = 1 if args.smoke else 3
if args.smoke:
    N_POSTERIOR_SAMPLES, N_SBC_TEST, N_SBC_POSTERIOR = 1_000, 100, 200

ARTIFACTS_DIR = model_artifacts_dir("queue")
RESULTS_DIR = model_results_dir("queue")
REF_FILE = ARTIFACTS_DIR / "deterministic_reference.npz"
OUT_PREFIX = "ablation_joint_vs_separate" + ("_SMOKE" if args.smoke else "")

# Each seed writes its own part file and is skipped if that file exists, so an
# interrupted run resumes instead of restarting. Every seed re-seeds torch
# before building anything, so per-process and single-process runs agree.
PARTS_DIR = ARTIFACTS_DIR / f"{OUT_PREFIX}_parts"
PARTS_DIR.mkdir(exist_ok=True)


def build_masked_mnpe(batch_theta, batch_x, train_part="both", freeze=None):
    """build_mnpe with the tuned architecture, then re-typed as HeadMaskedMDE."""
    de = build_tuned_mnpe()(batch_theta, batch_x)
    de.__class__ = HeadMaskedMDE
    de._train_part = train_part
    if freeze == "discrete":
        for p in de.discrete_net.parameters():
            p.requires_grad_(False)
    elif freeze == "continuous":
        for p in de.continuous_net.parameters():
            p.requires_grad_(False)
    return de


# ── Part 1: verify the decomposition numerically ────────────────────────────
def verify_decomposition(theta, x, seed=0):
    """Check the three facts that make joint == separate, and return them."""
    torch.manual_seed(seed)
    de = build_masked_mnpe(theta[:200], x[:200])
    tb, xb = theta[:256], x[:256]

    disc_ids = {id(p) for p in de.discrete_net.parameters()}
    cont_ids = {id(p) for p in de.continuous_net.parameters()}
    n_embed = sum(1 for _ in de.condition_embedding.parameters())
    shared = disc_ids & cont_ids

    def grads(part):
        de._train_part = part
        de.zero_grad(set_to_none=True)
        de.loss(tb, xb).mean().backward()
        return (
            {n: (p.grad.clone() if p.grad is not None else None)
             for n, p in de.discrete_net.named_parameters()},
            {n: (p.grad.clone() if p.grad is not None else None)
             for n, p in de.continuous_net.named_parameters()},
        )

    g_disc_joint, g_cont_joint = grads("both")
    g_disc_only, g_cont_from_disc = grads("discrete")
    g_disc_from_cont, g_cont_only = grads("continuous")
    de._train_part = "both"

    def max_diff(a, b):
        vals = [
            float((a[n] - b[n]).abs().max())
            for n in a
            if a[n] is not None and b[n] is not None
        ]
        return max(vals) if vals else 0.0

    def max_abs(d):
        vals = [float(v.abs().max()) for v in d.values() if v is not None]
        return max(vals) if vals else 0.0

    # Loss values: the joint loss must equal the sum of the two head losses.
    with torch.no_grad():
        de._train_part = "both"
        lj = float(de.loss(tb, xb).mean())
        de._train_part = "discrete"
        ld = float(de.loss(tb, xb).mean())
        de._train_part = "continuous"
        lc = float(de.loss(tb, xb).mean())
        de._train_part = "both"

    facts = {
        "n_shared_parameter_tensors": len(shared),
        "n_condition_embedding_parameters": n_embed,
        "loss_joint": lj,
        "loss_discrete_plus_continuous": ld + lc,
        "loss_decomposition_abs_error": abs(lj - (ld + lc)),
        "grad_discrete_joint_vs_discrete_only": max_diff(g_disc_joint, g_disc_only),
        "grad_continuous_joint_vs_continuous_only": max_diff(g_cont_joint, g_cont_only),
        "grad_continuous_leak_from_discrete_loss": max_abs(g_cont_from_disc),
        "grad_discrete_leak_from_continuous_loss": max_abs(g_disc_from_cont),
    }

    print("\n" + "=" * 72)
    print("Part 1 — architecture and objective decomposition")
    print("=" * 72)
    print(f"  shared parameter tensors between heads : {facts['n_shared_parameter_tensors']}")
    print(f"  parameters in the condition embedding  : "
          f"{facts['n_condition_embedding_parameters']}")
    print(f"  joint loss                             : {lj:.6f}")
    print(f"  discrete loss + continuous loss        : {ld + lc:.6f}  "
          f"(|diff| {facts['loss_decomposition_abs_error']:.2e})")
    print(f"  max |grad_disc(joint) - grad_disc(disc-only)| : "
          f"{facts['grad_discrete_joint_vs_discrete_only']:.2e}")
    print(f"  max |grad_cont(joint) - grad_cont(cont-only)| : "
          f"{facts['grad_continuous_joint_vs_continuous_only']:.2e}")
    print(f"  max |grad_cont| from the discrete-only loss   : "
          f"{facts['grad_continuous_leak_from_discrete_loss']:.2e}")
    print(f"  max |grad_disc| from the continuous-only loss : "
          f"{facts['grad_discrete_leak_from_continuous_loss']:.2e}")

    assert facts["n_shared_parameter_tensors"] == 0
    assert facts["n_condition_embedding_parameters"] == 0
    assert facts["grad_continuous_leak_from_discrete_loss"] == 0.0
    assert facts["grad_discrete_leak_from_continuous_loss"] < 1e-5
    assert facts["grad_discrete_joint_vs_discrete_only"] == 0.0
    # The continuous term is recovered by subtraction, so it is exact only to
    # float32 rounding of a sum whose parts are O(1).
    assert facts["grad_continuous_joint_vs_continuous_only"] < 1e-4
    assert facts["loss_decomposition_abs_error"] < 1e-4
    return facts


# ── Setup ───────────────────────────────────────────────────────────────────
print("=" * 72)
print(f"Joint vs separate MNPE training — budget {BUDGET:,}, {N_SEEDS} seeds")
print("=" * 72)

prior = build_sbi_prior()
obs_data = generate_reference_observation(seed=42)
x_o = transform_x(
    torch.tensor(obs_data["x_array"], dtype=torch.float32).unsqueeze(0)
).squeeze(0)

ref_data = np.load(REF_FILE)
ref_samples = np.column_stack(
    [ref_data["gamma"], ref_data["mu1"], ref_data["mu2"], ref_data["c1"], ref_data["c2"]]
)
np.random.default_rng(0).shuffle(ref_samples)

# Cached so a resumed run does not pay for prior-predictive generation again.
data_cache = PARTS_DIR / "data.pt"
if data_cache.exists():
    print(f"\nLoading cached training/test data ({data_cache.name})")
    _d = torch.load(data_cache, weights_only=False)
    theta_train, x_train, theta_test, x_test = (
        _d["theta_train"], _d["x_train"], _d["theta_test"], _d["x_test"]
    )
else:
    print(f"\nGenerating {BUDGET:,} training samples (seed={SIMULATION_SEED})...")
    theta_train, x_train_raw = generate_training_data(BUDGET, seed=SIMULATION_SEED)
    x_train = transform_x(x_train_raw)
    print(f"Generating {N_SBC_TEST} test samples (seed={SBC_TEST_SEED})...")
    theta_test, x_test_raw = generate_training_data(N_SBC_TEST, seed=SBC_TEST_SEED)
    x_test = transform_x(x_test_raw)
    torch.save(
        {"theta_train": theta_train, "x_train": x_train,
         "theta_test": theta_test, "x_test": x_test},
        data_cache,
    )
N_ACTUAL = len(theta_test)
print(f"  {len(theta_train)} stable training cases, {N_ACTUAL} stable test cases")

facts_cache = PARTS_DIR / "decomposition_facts.pkl"
if facts_cache.exists():
    with open(facts_cache, "rb") as f:
        facts = pickle.load(f)
    print(f"\nPart 1 — already verified ({facts_cache.name})")
else:
    facts = verify_decomposition(theta_train, x_train)
    with open(facts_cache, "wb") as f:
        pickle.dump(facts, f)

# ── Part 2: train both arms ─────────────────────────────────────────────────
print("\n" + "=" * 72)
print("Part 2 — joint vs separate training")
print("=" * 72)

train_kwargs = dict(
    training_batch_size=TUNED_HPARAMS["training_batch_size"],
    learning_rate=TUNED_HPARAMS["learning_rate"],
)
def train_one(seed_i, stage):
    """Train a single head-masking arm and cache it. Each stage is cached on its
    own so an interrupted run loses at most one training, not all three."""
    cache = PARTS_DIR / f"trained_seed{seed_i}_{stage}.pkl"
    if cache.exists():
        print(f"    [{stage}] already cached")
        with open(cache, "rb") as f:
            return pickle.load(f)

    part, freeze, label = {
        "joint": ("both", None, "single loop, combined loss and combined early stopping"),
        "discrete": ("discrete", "continuous", "discrete head alone"),
        "continuous": ("continuous", "discrete", "continuous head alone"),
    }[stage]
    print(f"    [{stage}] {label}")

    torch.manual_seed(seed_i * 1000)  # same seeds as budget_study.py reps 0..2
    trainer = MNPE(
        prior=prior,
        density_estimator=partial(build_masked_mnpe, train_part=part, freeze=freeze),
    )
    trainer.append_simulations(theta_train, x_train)
    trainer.train(**train_kwargs)
    epochs = trainer.summary["epochs_trained"][-1]
    post = trainer.build_posterior(
        posterior_parameters=DirectPosteriorParameters(enable_transform=False)
    )
    out = {"post": post, "epochs": epochs}
    with open(cache, "wb") as f:
        pickle.dump(out, f)
    print(f"    [{stage}] {epochs} epochs")
    return out


def train_seed(seed_i):
    """Both arms for one seed: the joint path, and the two heads composed."""
    joint = train_one(seed_i, "joint")
    disc = train_one(seed_i, "discrete")
    cont = train_one(seed_i, "continuous")

    # Compose the two independently trained heads into one estimator. The
    # continuous arm's discrete head sat at its initialization throughout, so it
    # is simply replaced by the separately trained one; the continuous loss
    # never depended on it.
    post_sep = copy.deepcopy(cont["post"])
    de_sep = post_sep.posterior_estimator
    de_sep.discrete_net.load_state_dict(
        disc["post"].posterior_estimator.discrete_net.state_dict()
    )
    for p in de_sep.parameters():
        p.requires_grad_(True)
    de_sep._train_part = "both"

    print(f"  epochs — joint {joint['epochs']}, discrete {disc['epochs']}, "
          f"continuous {cont['epochs']}")
    return {
        "post_joint": joint["post"],
        "post_separate": post_sep,
        "ep_joint": joint["epochs"],
        "ep_disc": disc["epochs"],
        "ep_cont": cont["epochs"],
    }


for seed_i in range(N_SEEDS):
    eval_files = {a: PARTS_DIR / f"eval_seed{seed_i}_{a}.pkl" for a in ("joint", "separate")}
    if all(p.exists() for p in eval_files.values()):
        print(f"\n  Seed {seed_i} — already evaluated, skipping")
        continue
    print(f"\n{'─' * 72}\n  Seed {seed_i} (torch seed {seed_i * 1000})\n{'─' * 72}")
    trained = train_seed(seed_i)
    ep = trained["ep_joint"], trained["ep_disc"], trained["ep_cont"]

    for arm in ("joint", "separate"):
        if eval_files[arm].exists():
            print(f"    {arm:>8}: already evaluated")
            continue
        torch.manual_seed(seed_i * 1000)
        res = evaluate_posterior(
            trained[f"post_{arm}"], x_o, ref_samples, theta_test, x_test, prior,
            num_posterior_samples=N_POSTERIOR_SAMPLES,
            num_sbc_posterior=N_SBC_POSTERIOR,
        )
        res.update(arm=arm, seed=seed_i)
        row = {
            "arm": arm,
            "seed": seed_i,
            # Epoch counts are the mechanism under test: the joint arm stops
            # both heads at once on the combined validation loss, the separate
            # arm stops each on its own.
            "epochs": {"joint": ep[0], "separate": max(ep[1], ep[2])}[arm],
            "epochs_discrete": {"joint": ep[0], "separate": ep[1]}[arm],
            "epochs_continuous": {"joint": ep[0], "separate": ep[2]}[arm],
            "c2st_all": res["c2st_all"],
            **{f"c2st_{k}": v for k, v in res["c2st_marginal"].items()},
            "sbc_ks_pval_min": min(res["sbc_ks_pvals"]),
            "joint_sbc_ks_pval": res["joint_sbc_ks_pval"],
            "ece_c1": res["ece"][0],
            "ece_c2": res["ece"][1],
            "ece_mean": float(np.mean(res["ece"])),
        }
        with open(eval_files[arm], "wb") as f:
            pickle.dump({"row": row, "rep": res}, f)
        print(f"    {arm:>8}: C2ST {res['c2st_all']:.4f}  "
              f"ECE c1={res['ece'][0]:.4f} c2={res['ece'][1]:.4f}  "
              f"joint-SBC p={res['joint_sbc_ks_pval']:.3f}")

# ── Report ──────────────────────────────────────────────────────────────────
part_files = [
    PARTS_DIR / f"eval_seed{i}_{a}.pkl"
    for i in range(N_SEEDS)
    for a in ("joint", "separate")
]
missing = [p.name for p in part_files if not p.exists()]
if missing:
    raise SystemExit(f"Parts still missing: {missing}. Re-run to continue.")
rows, reps = [], []
for p in part_files:
    with open(p, "rb") as f:
        part = pickle.load(f)
    rows.append(part["row"])
    reps.append(part["rep"])

df = pd.DataFrame(rows)
df.to_csv(RESULTS_DIR / f"{OUT_PREFIX}.csv", index=False)

metrics = ["c2st_all", "c2st_gamma", "c2st_mu1", "c2st_mu2", "c2st_c1", "c2st_c2",
           "ece_mean", "joint_sbc_ks_pval", "epochs_discrete", "epochs_continuous"]
summary = df.groupby("arm")[metrics].agg(["mean", "std"])
print("\n" + "=" * 72)
print(f"Summary over {N_SEEDS} seeds (mean, sd)")
print("=" * 72)
print(summary.round(4).to_string())

j = df[df.arm == "joint"]["c2st_all"].to_numpy()
s = df[df.arm == "separate"]["c2st_all"].to_numpy()
pooled_sd = float(np.sqrt((j.var(ddof=1) + s.var(ddof=1)) / 2)) if len(j) > 1 else float("nan")
print(f"\njoint C2ST {j.mean():.4f}  separate C2ST {s.mean():.4f}  "
      f"difference {s.mean() - j.mean():+.4f}  (pooled across-seed sd {pooled_sd:.4f})")

with open(ARTIFACTS_DIR / f"{OUT_PREFIX}.pkl", "wb") as f:
    pickle.dump(
        {
            "model_version": MODEL_VERSION,
            "budget": BUDGET,
            "n_seeds": N_SEEDS,
            "hparams": TUNED_HPARAMS,
            "decomposition_facts": facts,
            "rows": rows,
            "reps": reps,
            "reference_file": REF_FILE.name,
            "n_sbc_test_actual": N_ACTUAL,
        },
        f,
    )
print(f"\nSaved {OUT_PREFIX}.csv / .pkl")
