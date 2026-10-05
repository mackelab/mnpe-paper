"""Three-way factorization study on the queueing model.

Backs Appendix B.4 and Table B-2.

Queue analogue of reverse_factorization_toy.py — the evidential case (realistic
server-count/rate compensation, deterministic reference). Compares:
  forward      q(c|x) q(theta_c|c,x)   — the trained MNPE (all 5 reps, 100k budget)
  independent  q(c|x) q(theta_c|x)     — drop conditioning (tests "does it matter")
  reversed     q(theta_c|x) q(c|theta_c,x) — flip order (tests "which direction")

All three arms are reported as mean +/- sd over replicates. The forward arm must
not be the *best* MNPE rep: picking it by argmin over the same joint C2ST the
table reports would select on the reported metric, while the
two competing arms get fresh seeds, biasing the comparison toward MNPE.

theta_c = (gamma, mu1, mu2); c = (c1, c2), each 0-indexed over N_C=5 configs.
Independent and reversed SHARE one continuous marginal flow q(theta_c|x)
(3D standard NPE, matched to the MNPE continuous head: nflows NSF, 256/4, bins 9),
so they differ only in the discrete stage. The discrete stage is autoregressive
q(c1|.) q(c2|c1,.) via two MLP classifiers (cross-entropy), matching the forward
MNPE's autoregressive discrete head; trained on simulator (theta, x) -> c pairs
(NOT on flow samples, so stage-1 error does not compound into stage-2 training).

Evaluated like the other queue diagnostics: joint C2ST vs the reference at
x_o (0-indexed discrete), with the reference-vs-reference floor for scale, plus
the discrete-marginal error vs the reference discrete_posterior.

Run:  uv run python experiments/queue/ablations/reverse_factorization_queue.py [--smoke]
"""

import argparse
import pickle

import numpy as np
import pandas as pd
import torch
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from sbi.utils import MultipleIndependent
from sbi.utils.metrics import c2st
from torch.distributions import LogNormal

from mnpe_paper.models.queue.config import (
    GAMMA_MEDIAN,
    GAMMA_SIGMA,
    MU1_MEDIAN,
    MU1_SIGMA,
    MU2_MEDIAN,
    MU2_SIGMA,
    N_C,
    generate_reference_observation,
    generate_training_data,
    transform_x,
)
from mnpe_paper.models.queue.spec import SIMULATION_SEED as SIM_SEED
from mnpe_paper.models.queue.spec import TUNED_FLOW_HPARAMS as TUNED
from mnpe_paper.paths import model_artifacts_dir, model_results_dir

ARTIFACTS_DIR = model_artifacts_dir("queue")
RESULTS_DIR = model_results_dir("queue")

parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true")
args = parser.parse_args()
BUDGET = 3_000 if args.smoke else 100_000
N_SEEDS = 1 if args.smoke else 3
N_SAMPLES = 2_000 if args.smoke else 10_000
CLF_EPOCHS = 10 if args.smoke else 100
OUT_PREFIX = (
    "reverse_factorization_queue"
    + ("_SMOKE" if args.smoke else "")
)
np.random.seed(0)

# Each seed and the forward block write their own part file and are skipped if
# it already exists, so an interrupted run resumes instead of restarting. Every
# unit re-seeds torch before it does anything random, so splitting the work
# across processes gives bitwise the same numbers as one long run.
PARTS_DIR = ARTIFACTS_DIR / f"{OUT_PREFIX}_parts"
PARTS_DIR.mkdir(exist_ok=True)
if not (ARTIFACTS_DIR / "budget_study_results_n1000.pkl").exists():
    raise SystemExit("Missing budget_study_results_n1000.pkl; run main/reeval_budget_study.py first.")


def part_path(name):
    return PARTS_DIR / f"{name}.csv"


def save_part(name, part_rows):
    pd.DataFrame(part_rows).to_csv(part_path(name), index=False)


def continuous_prior():
    return MultipleIndependent([
        LogNormal(torch.log(torch.tensor([GAMMA_MEDIAN])), torch.tensor([GAMMA_SIGMA])),
        LogNormal(torch.log(torch.tensor([MU1_MEDIAN])), torch.tensor([MU1_SIGMA])),
        LogNormal(torch.log(torch.tensor([MU2_MEDIAN])), torch.tensor([MU2_SIGMA])),
    ])


def train_cat(features, labels, epochs):
    """MLP q(label|features), K=N_C softmax, cross-entropy. labels: long (N,).

    MINI-BATCH training (batch 256): full-batch GD gave only `epochs` optimizer
    steps total, far too few to learn the queue x->config mapping — the head
    collapsed to the prior (~uniform).
    """
    net = torch.nn.Sequential(
        torch.nn.Linear(features.shape[1], 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, N_C),
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(features, labels), batch_size=256, shuffle=True)
    for _ in range(epochs):
        for fb, lb in loader:
            opt.zero_grad()
            loss_fn(net(fb), lb).backward()
            opt.step()
    return net


def onehot(idx):
    return torch.nn.functional.one_hot(idx.long(), N_C).float()


def sample_cat(net, features):
    with torch.no_grad():
        probs = torch.softmax(net(features), dim=-1)
    return torch.multinomial(probs, 1).squeeze(-1)


def eval_joint(theta_c, c1, c2, ref_joint, method, seed):
    """Joint C2ST vs the deterministic reference (0-indexed discrete). Also a
    continuous-only C2ST (theta_c vs ref theta_c) to localise any discrepancy
    between the shared flow and the discrete assignment."""
    joint = torch.cat([theta_c, c1.reshape(-1, 1).float(), c2.reshape(-1, 1).float()], dim=1)
    n = min(len(joint), len(ref_joint))
    idx = torch.randperm(len(ref_joint))[:n]
    return {
        "method": method, "seed": seed,
        "c2st_joint": float(c2st(joint[:n], ref_joint[idx])),
        "c2st_cont": float(c2st(theta_c[:n], ref_joint[idx][:, :3])),
        "c2st_floor": REF_FLOOR_MEAN,
        "c2st_floor_sd": REF_FLOOR_SD,
    }


# ── Shared data + reference (identical seeds to the MNPE pipeline) ──────────
print(f"Generating training data (budget={BUDGET:,}, seed={SIM_SEED}) ...")
theta_all, x_all_raw = generate_training_data(BUDGET, seed=SIM_SEED)
x_all = transform_x(x_all_raw)
theta_c_all = theta_all[:, :3]
c1_all, c2_all = theta_all[:, 3].long(), theta_all[:, 4].long()

obs = generate_reference_observation(seed=42)
x_o = transform_x(torch.tensor(obs["x_array"], dtype=torch.float32).unsqueeze(0))
ref = np.load(ARTIFACTS_DIR / "deterministic_reference.npz")
ref_joint = torch.tensor(
    np.stack([ref["gamma"], ref["mu1"], ref["mu2"], ref["c1"] - 2.0, ref["c2"] - 2.0], axis=1),
    dtype=torch.float32,
)
# C2ST split floor. The canonical value is the 4-disjoint-split estimate from
# posterior_metrics.py, which carries an sd; the single split computed here is
# kept only as a same-protocol sanity check and is not what the table reports.
_floor = pd.read_csv(RESULTS_DIR / "c2st_floor.csv").iloc[0]
REF_FLOOR_MEAN = float(_floor["mean"])
REF_FLOOR_SD = float(_floor["std"])
half = len(ref_joint) // 2
nfl = min(half, N_SAMPLES)
# Seeded, so this split -- and hence the number printed below -- is the same
# from run to run. Every unit below re-seeds torch before it does anything
# random, so this seed affects nothing else.
torch.manual_seed(12345)
perm = ref_joint[torch.randperm(len(ref_joint))]
ref_floor_split = float(c2st(perm[:nfl], perm[half:half + nfl]))
print(f"C2ST split floor: {REF_FLOOR_MEAN:.4f} +/- {REF_FLOOR_SD:.4f} "
      f"(canonical, {int(_floor['n_splits'])} splits); "
      f"single-split check here: {ref_floor_split:.4f}")

# z-score stats for the classifier inputs (NPE z-scores x internally; the MLP
# classifiers need the same, else they collapse toward the prior). Continuous
# features only (x, theta_c) — onehot(c1) is appended AFTER z-scoring.
x_mean, x_std = x_all.mean(0, keepdim=True), x_all.std(0, keepdim=True) + 1e-6
tc_mean, tc_std = theta_c_all.mean(0, keepdim=True), theta_c_all.std(0, keepdim=True) + 1e-6
x_z = (x_all - x_mean) / x_std
tc_z = (theta_c_all - tc_mean) / tc_std
xo_z = (x_o - x_mean) / x_std

for seed in range(N_SEEDS):
    if part_path(f"seed{seed}").exists():
        print(f"\n── seed {seed} ── already done, skipping")
        continue
    print(f"\n── seed {seed} ──")
    rows = []
    torch.manual_seed(seed * 1000)

    # Continuous marginal flow q(theta_c | x): 3D NPE, matched to MNPE cont head
    build_fn = posterior_nn(model="nsf", num_transforms=TUNED["num_transforms"],
                            hidden_features=TUNED["hidden_features"], num_bins=TUNED["num_bins"])
    flow = NPE(prior=continuous_prior(), density_estimator=build_fn)
    flow.append_simulations(theta_c_all, x_all)
    flow.train(training_batch_size=TUNED["training_batch_size"], learning_rate=TUNED["learning_rate"])
    q_theta = flow.build_posterior()

    # Autoregressive discrete classifiers (z-scored continuous inputs).
    # independent conditions on x only; reversed on [theta_c, x]. c2 always
    # conditions on c1 (teacher forced with the true c1 during training).
    clf_c1_x = train_cat(x_z, c1_all, CLF_EPOCHS)
    clf_c2_x = train_cat(torch.cat([x_z, onehot(c1_all)], dim=1), c2_all, CLF_EPOCHS)
    clf_c1_tx = train_cat(torch.cat([tc_z, x_z], dim=1), c1_all, CLF_EPOCHS)
    clf_c2_tx = train_cat(torch.cat([tc_z, x_z, onehot(c1_all)], dim=1), c2_all, CLF_EPOCHS)

    theta_s = q_theta.sample((N_SAMPLES,), x=x_o, show_progress_bars=False)  # raw units for eval
    theta_s_z = (theta_s - tc_mean) / tc_std
    xo_z_rep = xo_z.repeat(N_SAMPLES, 1)

    # independent: c ~ q(c|x) (autoregressive), independent of theta_c
    c1_ind = sample_cat(clf_c1_x, xo_z_rep)
    c2_ind = sample_cat(clf_c2_x, torch.cat([xo_z_rep, onehot(c1_ind)], dim=1))
    rows.append(eval_joint(theta_s, c1_ind, c2_ind, ref_joint, "independent", seed))

    # reversed: c ~ q(c|theta_c,x), conditioned on the sampled theta_c
    feats = torch.cat([theta_s_z, xo_z_rep], dim=1)
    c1_rev = sample_cat(clf_c1_tx, feats)
    c2_rev = sample_cat(clf_c2_tx, torch.cat([feats, onehot(c1_rev)], dim=1))
    rows.append(eval_joint(theta_s, c1_rev, c2_rev, ref_joint, "reversed", seed))

    print(f"  independent c2st={rows[-2]['c2st_joint']:.3f}  "
          f"reversed c2st={rows[-1]['c2st_joint']:.3f}  (floor {REF_FLOOR_MEAN:.3f})")
    save_part(f"seed{seed}", rows)

# ── Forward = trained MNPE, every rep at the largest budget ─────────────────
# Every rep, not the best one: the reps are this arm's replicates, exactly as
# the seeds are for the other two. Each is re-seeded identically before
# sampling so the only thing varying across rows is the trained model.
with open(
    ARTIFACTS_DIR / "budget_study_results_n1000.pkl", "rb"
) as f:
    study = pickle.load(f)
fwd_budget = max(study["budgets"])
n_fwd_reps = len(study["budget_results"][fwd_budget]["reps"])
if part_path("forward").exists():
    print("\n── forward (MNPE, all reps) ── already done, skipping")
else:
    print("\n── forward (MNPE, all reps) ──")
    rows = []
    for rep in range(n_fwd_reps):
        post_file = (
            ARTIFACTS_DIR / "posteriors" / f"posterior_{fwd_budget}_rep{rep}.pkl"
        )
        with open(post_file, "rb") as f:
            post_mnpe = pickle.load(f)
        torch.manual_seed(0)
        s = post_mnpe.sample((N_SAMPLES,), x=x_o.squeeze(0), show_progress_bars=False)
        rows.append(eval_joint(s[:, :3], s[:, 3], s[:, 4], ref_joint, "forward", rep))
        print(f"  rep {rep}: c2st={rows[-1]['c2st_joint']:.4f}")
    save_part("forward", rows)

expected = [f"seed{s}" for s in range(N_SEEDS)] + ["forward"]
missing = [n for n in expected if not part_path(n).exists()]
if missing:
    raise SystemExit(f"Parts still missing: {missing}. Re-run to continue.")
df = pd.concat([pd.read_csv(part_path(n)) for n in expected], ignore_index=True)
df.to_csv(RESULTS_DIR / f"{OUT_PREFIX}.csv", index=False)
summary = df.groupby("method")[["c2st_joint", "c2st_cont"]].agg(["mean", "std", "count"])
print(f"\n── joint C2ST by method (budget {fwd_budget:,}; split floor "
      f"{REF_FLOOR_MEAN:.4f} +/- {REF_FLOOR_SD:.4f}) ──")
print(summary.round(4).to_string())
print(f"Saved {OUT_PREFIX}.csv")
