"""Sensitivity to the ordering of the discrete variables.

Backs Appendix B.6 and Table B-3. Asks whether MNPE is sensitive to the order
in which the discrete parameters enter the autoregressive factorization of the
discrete head:

    forward   q(c1 | x) q(c2 | c1, x)     <- what MNPE learns
    swapped   q(c2 | x) q(c1 | c2, x)

This is *not* the theta_d / theta_c factorization direction (that is
`reverse_factorization_queue.py`) and not the independence ablation.

Over two variables both orderings are exact factorizations of the same joint, so
they define the same model class: with unlimited capacity they are
indistinguishable and any measured difference comes from finite capacity and
optimization, not from expressiveness. The plausible mechanism is a capacity
asymmetry inside the MADE -- the slot-0 conditional sees only x, the slot-1
conditional also sees slot 0 -- interacting with a likelihood that is not
symmetric in c1 and c2 (c1 pairs with mu1, c2 with mu2, and the stations have
different service rates). A null result is therefore the expected outcome, and
it is reported with a scale attached rather than as a failure to reject.

Design:

  * The two arms are paired. They share the training data, the initialization,
    the batch order, and -- exactly -- the continuous head, which is reused from
    the parts directory of the joint-versus-separate study (Appendix B.2,
    `ablation_joint_vs_separate.py`). In its `continuous` stage the discrete
    head was frozen at initialization and never entered the loss, so the
    continuous head does not depend on either discrete arm.
  * Primary metric: the paired held-out discrete log-likelihood over the cached
    1,000-case test set. A strictly proper scoring rule for the object the
    ordering acts on, paired per case, so n = 1,000 rather than n = 3.
  * C2ST and discrete TV are reported for comparability with the other ablation
    tables, but at 100k their across-rep sd is 0.012 and 0.022, so with three
    seeds they can only detect a large effect.
  * SBC is deliberately dropped: it dominates the joint-versus-separate study's
    runtime and cannot speak to discrete ordering, since the continuous head is
    shared exactly and the discrete dimensions' calibration is measured directly
    by ECE.

The permutation is applied at the discrete head's boundary, not to theta: the
prior pairs c1 with mu1 for the rho < 0.999 support check, and the continuous
head conditions on the discrete sub-vector in physical order. Every path that
reads the MADE directly is made permutation-aware here, because three of them
(`metrics.exact_joint_discrete`, and both branches of
`calibration.run_discrete_calibration`) label slot 0 as c1 unconditionally and
would otherwise manufacture an ordering "effect" out of a transposition.

Run:  uv run python experiments/queue/ablations/ablation_discrete_ordering.py [--smoke]
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
from sbi.utils.metrics import c2st

from mnpe_paper.calibration import _marginal_discrete_probs, run_discrete_calibration
from mnpe_paper.models.queue.config import (
    C_VALUES,
    MODEL_VERSION,
    build_sbi_prior,
    generate_reference_observation,
    transform_x,
)
from mnpe_paper.models.queue.eval import build_tuned_mnpe
from mnpe_paper.models.queue.metrics import (
    discrete_agreement,
    ece_null_interval,
    exact_joint_discrete,
)
from mnpe_paper.models.queue.nets import (  # noqa: F401  (HeadMaskedMDE: pickle resolution)
    HeadMaskedMDE,
    apply_permutation,
    autoregressive_structure,
    enumerate_discrete_joint,
)
from mnpe_paper.models.queue.spec import TUNED_HPARAMS
from mnpe_paper.paths import model_artifacts_dir, model_results_dir

# ── Settings ────────────────────────────────────────────────────────────────
PARAM_NAMES = ["gamma", "mu1", "mu2", "c1", "c2"]
DISCRETE_INDICES = [3, 4]
N_CATEGORIES = len(C_VALUES)

# perm[i] is the physical variable occupying MADE slot i.
ARM_PERMUTATIONS = {"forward": [0, 1], "swapped": [1, 0]}

N_POSTERIOR_SAMPLES = 10_000
N_SBC_TEST = 1_000
NUM_BINS = 10
N_NULL_REPLICATES = 1_000
NULL_SEED = 0

parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true")
args = parser.parse_args()
BUDGET = 2_000 if args.smoke else 100_000
N_SEEDS = 1 if args.smoke else 3
if args.smoke:
    N_POSTERIOR_SAMPLES, N_SBC_TEST, N_NULL_REPLICATES = 1_000, 100, 100

ARTIFACTS_DIR = model_artifacts_dir("queue")
RESULTS_DIR = model_results_dir("queue")
REF_FILE = ARTIFACTS_DIR / "deterministic_reference.npz"
SUFFIX = "_SMOKE" if args.smoke else ""
OUT_PREFIX = f"ablation_discrete_ordering{SUFFIX}"
PARTS_DIR = ARTIFACTS_DIR / f"{OUT_PREFIX}_parts"
PARTS_DIR.mkdir(exist_ok=True)

# Reused from the joint-versus-separate study of Appendix B.2: the data, the
# continuous heads, and the forward-order discrete heads. Nothing here is
# written back into that directory.
JVS_PARTS = ARTIFACTS_DIR / f"ablation_joint_vs_separate{SUFFIX}_parts"
if not (JVS_PARTS / "data.pt").exists():
    raise SystemExit(
        f"Missing {JVS_PARTS.name}/data.pt — run ablation_joint_vs_separate.py"
        f"{' --smoke' if args.smoke else ''} first."
    )


# ── Estimator construction ──────────────────────────────────────────────────
def build_discrete_arm(batch_theta, batch_x, perm=None):
    """build_mnpe with the tuned architecture, masked to the discrete head.

    Mirrors `ablation_joint_vs_separate.build_masked_mnpe` with
    `train_part="discrete", freeze="continuous"`, so the RNG is consumed in the
    same order and the initialization matches the joint-versus-separate study's
    cached forward-order heads.
    The permutation is attached *after* construction and changes no parameter
    shapes, which is what makes that match possible.
    """
    de = build_tuned_mnpe()(batch_theta, batch_x)
    de.__class__ = HeadMaskedMDE
    de._train_part = "discrete"
    for p in de.continuous_net.parameters():
        p.requires_grad_(False)
    if perm is not None:
        apply_permutation(de.discrete_net.net, perm)
    return de


# ── Step 1: the checks that must pass before any training ───────────────────
def check_wrapper(theta, x, x_o, seed=0):
    """Identity check and autoregressive-structure probe on an untrained net.

    The identity check is plumbing: perm = [0, 1] must be a no-op. The structure
    probe is semantics: it reads off which *physical* variable carries the first,
    unconditioned factor, which is the thing the permutation is supposed to
    change and the thing a bitwise identity check cannot establish.
    """
    print("\n" + "=" * 72)
    print("Step 1 — permutation wrapper checks")
    print("=" * 72)

    torch.manual_seed(seed)
    # 2,000 rows, not 200: build_mnpe infers the number of categories from the
    # unique values in the batch, so the batch has to contain all five.
    plain = build_discrete_arm(theta[:2000], x[:2000])
    made = plain.discrete_net.net
    assert int(made.num_variables) == len(DISCRETE_INDICES)
    tb, xb = theta[:256], x[:256]

    with torch.no_grad():
        lp_plain = made.log_prob(tb[:, DISCRETE_INDICES], condition=xb)
    struct_plain = autoregressive_structure(made, N_CATEGORIES, x_o)

    apply_permutation(made, [0, 1])
    with torch.no_grad():
        lp_identity = made.log_prob(tb[:, DISCRETE_INDICES], condition=xb)
    identity_max_diff = float((lp_plain - lp_identity).abs().max())

    apply_permutation(made, [1, 0])
    with torch.no_grad():
        lp_swapped = made.log_prob(tb[:, DISCRETE_INDICES], condition=xb)
    swapped_max_diff = float((lp_plain - lp_swapped).abs().max())
    struct_swapped = autoregressive_structure(made, N_CATEGORIES, x_o)

    facts = {
        "identity_log_prob_max_diff": identity_max_diff,
        "swapped_log_prob_max_diff": swapped_max_diff,
        "forward_structure": struct_plain,
        "swapped_structure": struct_swapped,
    }
    print(f"  perm=[0,1] reproduces log_prob      : max |diff| {identity_max_diff:.2e}")
    print(f"  perm=[1,0] changes log_prob         : max |diff| {swapped_max_diff:.3e}")
    for name, s in (("forward", struct_plain), ("swapped", struct_swapped)):
        print(
            f"  {name:>8}: slot-0 logits vary by {s['slot0_spread_when_slot1_varies']:.2e} "
            f"when slot 1 changes; slot-1 logits vary by "
            f"{s['slot1_spread_when_slot0_varies']:.3e} when slot 0 changes; "
            f"first factor is over c{s['physical_variable_in_slot0'] + 1}"
        )

    # Plumbing: the identity permutation must be exactly a no-op.
    assert identity_max_diff == 0.0, identity_max_diff
    # And a real permutation must not be one (an untrained net is asymmetric).
    assert swapped_max_diff > 1e-6, swapped_max_diff
    # Semantics: slot 0 is unconditioned, slot 1 is conditioned on slot 0. The
    # masked weights are exactly zero but still take part in the residual
    # blocks' float32 accumulation, so slot-0 invariance holds to rounding
    # (~1e-7) rather than bitwise. The contrast is what carries the argument:
    # slot 1 moves by ~0.6, seven orders of magnitude more.
    for s in (struct_plain, struct_swapped):
        assert s["slot0_spread_when_slot1_varies"] < 1e-6, s
        assert s["slot1_spread_when_slot0_varies"] > 1e4 * s[
            "slot0_spread_when_slot1_varies"
        ], s
    assert struct_plain["physical_variable_in_slot0"] == 0
    assert struct_swapped["physical_variable_in_slot0"] == 1
    return facts


# ── Step 2: evaluation, made permutation-aware ──────────────────────────────
def heldout_discrete_logprob(made, theta_test, x_test):
    """Per-case log q(c_true | x) from the discrete head alone.

    Goes through the wrapper's `log_prob`, so the truth stays in physical order
    and the permutation is handled internally. This is the primary metric.
    """
    with torch.no_grad():
        return made.log_prob(theta_test[:, DISCRETE_INDICES], condition=x_test).numpy()


def permuted_calibration(post, made, theta_test, x_test):
    """ECE and marginal probabilities, relabelled from slot order to physical.

    `run_discrete_calibration` enumerates prefixes in slot order and returns
    `probs[:, i]` for slot `i`, but pairs them with `thetas[:, discrete_indices]`
    in physical order. Feeding it slot-ordered truth and reordering the outputs
    is what keeps c1's ground truth matched to c1's predicted marginal.
    """
    perm, inv = made.perm, made.inv_perm
    theta_slots = theta_test.clone()
    theta_slots[:, DISCRETE_INDICES] = theta_test[:, DISCRETE_INDICES][:, perm]

    ece, probs, _ = run_discrete_calibration(
        theta_slots,
        x_test,
        post,
        discrete_indices=DISCRETE_INDICES,
        num_bins=NUM_BINS,
        method="top_label",
        probability_type="marginal",
    )
    return ece[inv], probs[:, inv, :]


def consistency_guards(post, made, x_o, joint, samples_disc):
    """Three cross-checks that a slot/label transposition cannot survive.

    (a) On the forward arm the log_prob enumeration must agree with
        `metrics.exact_joint_discrete`, which reads the factors
        straight off `forward`.
    (b) The marginals implied by the enumerated 5x5 joint must equal the
        marginals the calibration path returns, after relabelling.
    (c) The empirical discrete frequencies of the drawn posterior samples must
        match the enumerated PMF, and must match it *better* than they match its
        transpose -- the reference joint is strongly asymmetric, so a swapped
        labelling shows up immediately.
    """
    grid = joint.reshape(N_CATEGORIES, N_CATEGORIES)
    out = {}

    if int(made.perm[0]) == 0:
        out["vs_exact_joint_discrete"] = float(
            np.abs(joint - exact_joint_discrete(post, x_o)).max()
        )

    with torch.no_grad():
        probs_xo = _marginal_discrete_probs(
            made, x_o.unsqueeze(0), max_exact_states=4096, num_samples=4096,
            seed=0, batch_size=1,
        )
    probs_xo = probs_xo[0][made.inv_perm].numpy()
    out["marginal_vs_enumeration"] = float(
        max(
            np.abs(probs_xo[0] - grid.sum(1)).max(),
            np.abs(probs_xo[1] - grid.sum(0)).max(),
        )
    )

    hist = np.zeros((N_CATEGORIES, N_CATEGORIES))
    for c1, c2 in samples_disc.astype(int):
        hist[c1, c2] += 1
    hist /= hist.sum()
    out["samples_vs_enumeration"] = float(np.abs(hist - grid).max())
    out["samples_vs_transposed_enumeration"] = float(np.abs(hist - grid.T).max())
    return out


def evaluate(post, x_o, ref_samples, quadrature, theta_test, x_test):
    made = post.posterior_estimator.discrete_net.net
    assert post.potential_fn.posterior_estimator is post.posterior_estimator

    samples = post.sample((N_POSTERIOR_SAMPLES,), x=x_o).numpy()
    samples_disc = samples[:, DISCRETE_INDICES].copy()
    for col in DISCRETE_INDICES:
        samples[:, col] = np.array([C_VALUES[int(v)] for v in samples[:, col]])

    n = min(len(samples), len(ref_samples))
    c2st_marginal = {}
    for i, name in enumerate(PARAM_NAMES):
        c2st_marginal[name] = c2st(
            torch.tensor(samples[:n, i : i + 1], dtype=torch.float32),
            torch.tensor(ref_samples[:n, i : i + 1], dtype=torch.float32),
            seed=1,
        ).item()
    c2st_all = c2st(
        torch.tensor(samples[:n], dtype=torch.float32),
        torch.tensor(ref_samples[:n], dtype=torch.float32),
        seed=1,
    ).item()

    logprob_per_case = heldout_discrete_logprob(made, theta_test, x_test)
    joint = enumerate_discrete_joint(made, x_o, N_CATEGORIES)
    agreement = discrete_agreement(joint, quadrature)
    guards = consistency_guards(post, made, x_o, joint, samples_disc)
    ece, probs = permuted_calibration(post, made, theta_test, x_test)
    null = ece_null_interval(probs, NUM_BINS, "top_label", N_NULL_REPLICATES, NULL_SEED)

    return {
        "c2st_all": c2st_all,
        "c2st_marginal": c2st_marginal,
        "logprob_per_case": logprob_per_case,
        "discrete_joint": joint,
        "agreement": agreement,
        "guards": guards,
        "ece": ece.numpy(),
        "ece_null": null,
        "samples": samples,
    }


# ── Setup ───────────────────────────────────────────────────────────────────
print("=" * 72)
print(f"Discrete ordering ablation — budget {BUDGET:,}, {N_SEEDS} seeds")
print("=" * 72)

prior = build_sbi_prior()
obs_data = generate_reference_observation(seed=42)
x_o = transform_x(
    torch.tensor(obs_data["x_array"], dtype=torch.float32).unsqueeze(0)
).squeeze(0)

ref_data = np.load(REF_FILE)
quadrature = ref_data["configuration_probabilities"]
ref_samples = np.column_stack([ref_data[name] for name in PARAM_NAMES])
np.random.default_rng(0).shuffle(ref_samples)

print(f"\nLoading cached training/test data ({JVS_PARTS.name}/data.pt)")
_d = torch.load(JVS_PARTS / "data.pt", weights_only=False)
theta_train, x_train = _d["theta_train"], _d["x_train"]
theta_test, x_test = _d["theta_test"][:N_SBC_TEST], _d["x_test"][:N_SBC_TEST]
# Loose check that this is the right parts directory: generate_training_data
# discards unstable draws, so the count need not equal BUDGET exactly.
assert abs(len(theta_train) - BUDGET) <= 0.2 * BUDGET, (
    f"cached data has {len(theta_train)} cases, expected about {BUDGET}"
)
print(f"  {len(theta_train)} stable training cases, {len(theta_test)} stable test cases")

checks_cache = PARTS_DIR / "wrapper_checks.pkl"
if checks_cache.exists():
    with open(checks_cache, "rb") as f:
        wrapper_facts = pickle.load(f)
    print(f"\nStep 1 — already verified ({checks_cache.name})")
else:
    wrapper_facts = check_wrapper(theta_train, x_train, x_o)
    with open(checks_cache, "wb") as f:
        pickle.dump(wrapper_facts, f)


# ── Step 3: train the swapped heads, verify the reused forward heads ────────
print("\n" + "=" * 72)
print("Step 3 — discrete-head training")
print("=" * 72)

train_kwargs = dict(
    training_batch_size=TUNED_HPARAMS["training_batch_size"],
    learning_rate=TUNED_HPARAMS["learning_rate"],
)


def train_discrete(seed_i, arm, cache_name):
    """Train one discrete head under the given slot ordering, and cache it."""
    cache = PARTS_DIR / cache_name
    if cache.exists():
        print(f"    [{arm}] already cached ({cache.name})")
        with open(cache, "rb") as f:
            return pickle.load(f)

    print(f"    [{arm}] training discrete head, perm={ARM_PERMUTATIONS[arm]}")
    torch.manual_seed(seed_i * 1000)  # same seeds as budget_study.py reps 0..2
    trainer = MNPE(
        prior=prior,
        density_estimator=partial(build_discrete_arm, perm=ARM_PERMUTATIONS[arm]),
    )
    trainer.append_simulations(theta_train, x_train)
    trainer.train(**train_kwargs)
    epochs = trainer.summary["epochs_trained"][-1]
    post = trainer.build_posterior(
        posterior_parameters=DirectPosteriorParameters(enable_transform=False)
    )
    out = {
        "state_dict": copy.deepcopy(
            post.posterior_estimator.discrete_net.state_dict()
        ),
        "epochs": epochs,
    }
    with open(cache, "wb") as f:
        pickle.dump(out, f)
    print(f"    [{arm}] {epochs} epochs")
    return out


def load_jvs_discrete(seed_i):
    with open(JVS_PARTS / f"trained_seed{seed_i}_discrete.pkl", "rb") as f:
        part = pickle.load(f)
    return {
        "state_dict": part["post"].posterior_estimator.discrete_net.state_dict(),
        "epochs": part["epochs"],
    }


# The forward arm is reused from the joint-versus-separate study. That is only
# valid if this script consumes the RNG in that study's order, so retrain seed 0
# here and require a bitwise match rather than assuming it.
reuse_cache = PARTS_DIR / "forward_reuse_check.pkl"
if reuse_cache.exists():
    with open(reuse_cache, "rb") as f:
        reuse = pickle.load(f)
    print(f"\n  Forward-arm reuse — already verified ({reuse_cache.name})")
else:
    print("\n  Verifying that the joint-vs-separate forward-order head is reproducible here")
    fresh = train_discrete(0, "forward", "trained_seed0_forward_check.pkl")
    cached = load_jvs_discrete(0)
    diffs = {
        k: float((fresh["state_dict"][k] - cached["state_dict"][k]).abs().max())
        for k in cached["state_dict"]
    }
    reuse = {
        "max_state_dict_diff": max(diffs.values()),
        "epochs_fresh": fresh["epochs"],
        "epochs_cached": cached["epochs"],
    }
    with open(reuse_cache, "wb") as f:
        pickle.dump(reuse, f)
print(
    f"    max |state_dict diff| {reuse['max_state_dict_diff']:.2e}  "
    f"(epochs: fresh {reuse['epochs_fresh']}, cached {reuse['epochs_cached']})"
)
REUSE_OK = reuse["max_state_dict_diff"] == 0.0 and (
    reuse["epochs_fresh"] == reuse["epochs_cached"]
)
if REUSE_OK:
    print("    reuse verified — seeds 1 and 2 are taken from the joint-vs-separate parts directory")
else:
    print("    reuse NOT verified — retraining all forward-order heads here")


def continuous_post(seed_i):
    with open(JVS_PARTS / f"trained_seed{seed_i}_continuous.pkl", "rb") as f:
        return pickle.load(f)["post"]


def compose(seed_i, arm):
    """Shared continuous head + this arm's discrete head, in one estimator."""
    if arm == "forward" and REUSE_OK:
        head = load_jvs_discrete(seed_i)
    else:
        head = train_discrete(seed_i, arm, f"trained_seed{seed_i}_{arm}.pkl")

    post = copy.deepcopy(continuous_post(seed_i))
    de = post.posterior_estimator
    de.discrete_net.load_state_dict(head["state_dict"])
    for p in de.parameters():
        p.requires_grad_(True)
    de._train_part = "both"
    apply_permutation(de.discrete_net.net, ARM_PERMUTATIONS[arm])
    return post, head["epochs"]


# ── Step 4: evaluate ────────────────────────────────────────────────────────
for seed_i in range(N_SEEDS):
    eval_files = {a: PARTS_DIR / f"eval_seed{seed_i}_{a}.pkl" for a in ARM_PERMUTATIONS}
    if all(p.exists() for p in eval_files.values()):
        print(f"\n  Seed {seed_i} — already evaluated, skipping")
        continue
    print(f"\n{'─' * 72}\n  Seed {seed_i} (torch seed {seed_i * 1000})\n{'─' * 72}")

    for arm in ARM_PERMUTATIONS:
        if eval_files[arm].exists():
            print(f"    {arm:>8}: already evaluated")
            continue
        post, epochs = compose(seed_i, arm)
        torch.manual_seed(seed_i * 1000)
        res = evaluate(post, x_o, ref_samples, quadrature, theta_test, x_test)
        row = {
            "arm": arm,
            "seed": seed_i,
            "epochs": epochs,
            "heldout_logprob_mean": float(res["logprob_per_case"].mean()),
            "heldout_logprob_sem": float(
                res["logprob_per_case"].std(ddof=1) / np.sqrt(len(res["logprob_per_case"]))
            ),
            "c2st_all": res["c2st_all"],
            **{f"c2st_{k}": v for k, v in res["c2st_marginal"].items()},
            "discrete_tv": res["agreement"]["joint_tv"],
            "discrete_tv_c1": res["agreement"]["c1_tv"],
            "discrete_tv_c2": res["agreement"]["c2_tv"],
            "max_cell_error": res["agreement"]["max_cell_error"],
            "ece_c1": float(res["ece"][0]),
            "ece_c2": float(res["ece"][1]),
            **{
                f"ece_null_{bound}_{name}": float(res["ece_null"][key][i])
                for bound, key in (("lo", "q025"), ("hi", "q975"))
                for i, name in enumerate(("c1", "c2"))
            },
        }
        with open(eval_files[arm], "wb") as f:
            pickle.dump({"row": row, "rep": res}, f)

        guards = res["guards"]
        print(
            f"    {arm:>8}: held-out log q {row['heldout_logprob_mean']:.4f} "
            f"± {row['heldout_logprob_sem']:.4f}   C2ST {row['c2st_all']:.4f}   "
            f"TV {row['discrete_tv']:.4f}   ECE c1={row['ece_c1']:.4f} "
            f"c2={row['ece_c2']:.4f}"
        )
        print(
            "              guards: "
            + "  ".join(f"{k} {v:.2e}" for k, v in guards.items())
        )
        # (b) is exact arithmetic on the same softmax outputs; (c) is Monte Carlo
        # at N_POSTERIOR_SAMPLES, so its tolerance has to scale as 1/sqrt(N)
        # (2/sqrt(N) is ~4 binomial sd at the largest cell). The sharp test is
        # the relative one against the transpose, which a mislabelling of the
        # two slots cannot pass: the reference joint is strongly asymmetric.
        assert guards["marginal_vs_enumeration"] < 1e-5, guards
        assert guards["samples_vs_enumeration"] < 2.0 / np.sqrt(N_POSTERIOR_SAMPLES), guards
        assert (
            guards["samples_vs_enumeration"] < guards["samples_vs_transposed_enumeration"]
        ), guards
        if "vs_exact_joint_discrete" in guards:
            assert guards["vs_exact_joint_discrete"] < 1e-6, guards

# ── Report ──────────────────────────────────────────────────────────────────
part_files = [
    PARTS_DIR / f"eval_seed{i}_{a}.pkl" for i in range(N_SEEDS) for a in ARM_PERMUTATIONS
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

metrics = [
    "heldout_logprob_mean", "c2st_all", "c2st_c1", "c2st_c2",
    "discrete_tv", "ece_c1", "ece_c2", "epochs",
]
print("\n" + "=" * 72)
print(f"Summary over {N_SEEDS} seeds (mean, sd)")
print("=" * 72)
print(df.groupby("arm")[metrics].agg(["mean", "std"]).round(4).to_string())

# ── Primary metric: the paired held-out discrete log-likelihood ─────────────
print("\n" + "=" * 72)
print("Primary metric — paired held-out discrete log-likelihood")
print("=" * 72)
by_key = {(row["seed"], row["arm"]): rep for row, rep in zip(rows, reps)}

paired = []
for seed_i in range(N_SEEDS):
    fwd = by_key[(seed_i, "forward")]["logprob_per_case"]
    swp = by_key[(seed_i, "swapped")]["logprob_per_case"]
    delta = fwd - swp
    sem = float(delta.std(ddof=1) / np.sqrt(len(delta)))
    tv_arm = 0.5 * float(
        np.abs(
            by_key[(seed_i, "forward")]["discrete_joint"]
            - by_key[(seed_i, "swapped")]["discrete_joint"]
        ).sum()
    )
    paired.append({
        "seed": seed_i,
        "delta_mean": float(delta.mean()),
        "delta_sem": sem,
        "delta_z": float(delta.mean() / sem) if sem > 0 else float("nan"),
        "arm_vs_arm_tv": tv_arm,
    })
    print(
        f"  seed {seed_i}: forward − swapped = {delta.mean():+.5f} nats "
        f"± {sem:.5f} (paired s.e., n={len(delta)})   z = {paired[-1]['delta_z']:+.2f}"
        f"   arm-vs-arm discrete TV {tv_arm:.5f}"
    )

paired_df = pd.DataFrame(paired)
paired_df.to_csv(RESULTS_DIR / f"{OUT_PREFIX}_paired.csv", index=False)

fwd_rows = df[df.arm == "forward"]
swp_rows = df[df.arm == "swapped"]
print("\n  Question 1 — is any effect detectable?")
print(
    f"    mean paired Δ across seeds {paired_df.delta_mean.mean():+.5f} nats; "
    f"per-seed |z| = {', '.join(f'{abs(z):.1f}' for z in paired_df.delta_z)}"
)
print(
    "    All seeds share the same 1,000-case test set, so a consistent sign is\n"
    "    one observation, not N_SEEDS independent confirmations: a test-set\n"
    "    idiosyncrasy would reproduce across seeds. Read the per-seed z, not the\n"
    "    agreement in sign."
)

print("\n  Question 2 — how large is it, at most?")
# The primary metric is paired, so its yardstick is a ceiling in its own units
# rather than the across-seed sd of the arm mean: pairing removes exactly the
# variance that sd contains, so comparing the two would flatter the effect.
delta = paired_df.delta_mean.mean()
delta_spread = float(paired_df.delta_mean.std(ddof=1)) if N_SEEDS > 1 else float("nan")
scale = abs(float(fwd_rows["heldout_logprob_mean"].mean()))
print(
    f"    {'held-out log q (paired)':>26}: Δ {delta:+.5f} nats, across-seed spread "
    f"{delta_spread:.5f}, single-seed paired s.e. {paired_df.delta_sem.mean():.5f}\n"
    f"    {'':>26}  = {100 * abs(delta) / scale:.2f}% of the {scale:.3f}-nat mean"
)
# C2ST and TV are unpaired cross-arm comparisons against a shared reference, so
# there the seed-to-seed spread of one arm is the right practical yardstick.
for metric in ("c2st_all", "discrete_tv"):
    fwd_sd = float(fwd_rows[metric].std(ddof=1)) if N_SEEDS > 1 else float("nan")
    diff = float(swp_rows[metric].mean() - fwd_rows[metric].mean())
    verdict = (
        "within" if (N_SEEDS > 1 and abs(diff) < fwd_sd) else
        "EXCEEDS" if N_SEEDS > 1 else "n/a"
    )
    print(
        f"    {metric:>26}: swapped − forward {diff:+.5f}   "
        f"forward seed sd {fwd_sd:.5f}   → {verdict}"
    )

# Epochs to convergence are the one place the two orderings separate cleanly.
print("\n  Epochs to convergence (early stopping on the discrete loss alone)")
for arm, rows_arm in (("forward", fwd_rows), ("swapped", swp_rows)):
    print(
        f"    {arm:>8}: {', '.join(str(int(e)) for e in rows_arm.epochs)}  "
        f"(mean {rows_arm.epochs.mean():.1f})"
    )

with open(ARTIFACTS_DIR / f"{OUT_PREFIX}.pkl", "wb") as f:
    pickle.dump(
        {
            "model_version": MODEL_VERSION,
            "budget": BUDGET,
            "n_seeds": N_SEEDS,
            "hparams": TUNED_HPARAMS,
            "permutations": ARM_PERMUTATIONS,
            "wrapper_checks": wrapper_facts,
            "forward_reuse_check": reuse,
            "rows": rows,
            "paired": paired,
            "logprob_per_case": {
                (r["seed"], r["arm"]): rep["logprob_per_case"]
                for r, rep in zip(rows, reps)
            },
            "discrete_joints": {
                (r["seed"], r["arm"]): rep["discrete_joint"]
                for r, rep in zip(rows, reps)
            },
            "guards": {(r["seed"], r["arm"]): rep["guards"] for r, rep in zip(rows, reps)},
            "reference_file": REF_FILE.name,
            "n_test_cases": len(theta_test),
        },
        f,
    )
print(f"\nSaved {OUT_PREFIX}.csv / _paired.csv / .pkl")
