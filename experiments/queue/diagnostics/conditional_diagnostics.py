"""Conditional coupling diagnostics for the MNPE queueing posterior.

Backs Appendix A.3. The rest of the suite checks continuous marginals (SBC) and
discrete marginals (ECE) but never the learned coupling q(theta_c | theta_d, x),
which is the part of the joint that a mixed method is uniquely able to get
wrong. Two diagnostics close that gap:

D1 — per-config C2ST against the deterministic reference conditional at x_o.
     Split a large joint MNPE sample by its discrete draw and compare the
     continuous block per configuration against the reference draws carrying
     the same labels. Yardstick: reference-vs-reference split C2ST per config.

D2 — conditional SBC over the prior-predictive test set (amortized).
     For each test case rank theta_c* under q(theta_c | theta_d = theta_d*, x)
     — the coupling-sensitive test quantity (Talts 2018; Modrak 2023). Ranks
     are checked for uniformity pooled and grouped by discrete configuration;
     grouping is where a coupling failure surfaces even if marginal SBC passes.

The conditional q(theta_c | theta_d, x) is sampled directly from the
MixedDensityEstimator's continuous flow. Run with --verify-helper to check the
plumbing: conditional samples must be indistinguishable (C2ST ~ 0.5) from
joint-filtered samples at the dominant config, since filtering a joint
ancestral sample by its drawn theta_d IS the conditional by construction.
sbi's run_sbc cannot inject a per-test-point theta_d, hence the bespoke rank
loop.

Outputs (NEW files only):
  conditional_diagnostics_d1.csv        — per-config C2ST, floor, masses, N
  conditional_diagnostics_d2_ks.csv     — KS p-values pooled + grouped
  conditional_diagnostics_d1.pdf/png    — 5x5 C2ST heatmap
  conditional_diagnostics_d2.pdf/png    — pooled conditional-SBC rank hists

Run:  uv run python experiments/queue/diagnostics/conditional_diagnostics.py
"""

import argparse
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sbi.diagnostics.sbc import check_uniformity_frequentist
from sbi.utils.metrics import c2st

from mnpe_paper.models.queue.conditional import conditional_ranks, verify_conditional_sampler
from mnpe_paper.models.queue.config import (
    C_VALUES,
    generate_reference_observation,
    generate_training_data,
    transform_x,
)
from mnpe_paper.models.queue.spec import SBC_TEST_SEED
from mnpe_paper.paths import model_artifacts_dir, model_results_dir
from mnpe_paper.plotting.plot_style import despine

ARTIFACTS_DIR = model_artifacts_dir("queue")
RESULTS_DIR = model_results_dir("queue")
DISCRETE_OFFSET = 2  # c_physical = c_sbi + 2
N_JOINT = 200_000  # joint MNPE samples at x_o for D1
MIN_N_PER_CONFIG = 200  # skip configs with fewer samples on either side
C2ST_CAP = 5_000  # cap per-config sample count fed to C2ST
N_SBC_TEST = 1_000
N_CAP = 1_000
M_COND = 1_000  # conditional samples per test case (matches N_SBC_POSTERIOR)
SEED = 0

CONT_NAMES = ["gamma", "mu1", "mu2"]

# ── Load ALL posteriors at the largest budget ───────────────────────────────
# Every reported number is over all reps: pooled conditional-SBC p-values span
# orders of magnitude across reps, so picking one on a quality metric would be
# selection bias. The figures show a single labelled example rep, named in the
# titles and in the printed output.

with open(
    ARTIFACTS_DIR / "budget_study_results_n1000.pkl", "rb"
) as f:
    study = pickle.load(f)
best_budget = max(study["budgets"])
c2st_per_rep = [r["c2st_all"] for r in study["budget_results"][best_budget]["reps"]]
N_REPS = len(c2st_per_rep)
FIG_REP = int(np.argmin(c2st_per_rep))  # example shown in the figures, labelled


def load_posterior(rep):
    with open(
        ARTIFACTS_DIR / "posteriors" / f"posterior_{best_budget}_rep{rep}.pkl",
        "rb",
    ) as f:
        return pickle.load(f)


post = load_posterior(FIG_REP)
estimator = post.posterior_estimator
print(f"Budget={best_budget}, {N_REPS} reps; figures use rep {FIG_REP} "
      f"(joint C2ST {c2st_per_rep[FIG_REP]:.4f}, "
      f"rep mean {np.mean(c2st_per_rep):.4f})")
assert not estimator.log_transform_input, "helper assumes linear-space continuous block"

PARTS_DIR = ARTIFACTS_DIR / "conditional_diagnostics_parts"
PARTS_DIR.mkdir(exist_ok=True)

obs = generate_reference_observation(seed=42)
x_o = transform_x(torch.tensor(obs["x_array"], dtype=torch.float32).unsqueeze(0))


parser = argparse.ArgumentParser()
parser.add_argument(
    "--verify-helper", action="store_true",
    help="only run the conditional-sampling plumbing check, then exit",
)
args = parser.parse_args()
if args.verify_helper:
    verify_conditional_sampler(post, x_o, SEED, verbose=True)
    sys.exit(0)


# ═══════════════════════════════════════════════════════════════════════
# D1 — per-config C2ST against the deterministic reference conditional at x_o
# ═══════════════════════════════════════════════════════════════════════

print(f"\n── D1: per-config C2ST at x_o ({N_JOINT:,} joint MNPE samples) ──")

ref = np.load(ARTIFACTS_DIR / "deterministic_reference.npz")
ref_cont = np.stack([ref["gamma"], ref["mu1"], ref["mu2"]], axis=1)
ref_c1, ref_c2 = ref["c1"], ref["c2"]  # physical values (2..6)
ref_mass = {(int(r["c1"]), int(r["c2"])): float(r["prob"]) for r in ref["discrete_posterior"]}


def compute_d1(post_rep, rep, verbose=False):
    """Per-config C2ST at x_o for one rep. Seeded identically per rep, so the
    only thing varying across rows is the trained model."""
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    joint = post_rep.sample((N_JOINT,), x=x_o, show_progress_bars=False).numpy()

    rows = []
    for i, c1_phys in enumerate(map(int, C_VALUES)):
        for j, c2_phys in enumerate(map(int, C_VALUES)):
            mnpe_mask = (joint[:, 3] == i) & (joint[:, 4] == j)
            ref_mask = (ref_c1 == c1_phys) & (ref_c2 == c2_phys)
            n_mnpe, n_ref = int(mnpe_mask.sum()), int(ref_mask.sum())
            row = {
                "rep": rep,
                "c1_phys": c1_phys,
                "c2_phys": c2_phys,
                "ref_mass": ref_mass.get((c1_phys, c2_phys), 0.0),
                "mnpe_mass": n_mnpe / N_JOINT,
                "n_mnpe": n_mnpe,
                "n_ref": n_ref,
                "c2st": np.nan,
                "c2st_floor": np.nan,
            }
            if n_mnpe >= MIN_N_PER_CONFIG and n_ref >= MIN_N_PER_CONFIG:
                n = min(n_mnpe, n_ref, C2ST_CAP)
                mnpe_sub = rng.permutation(joint[mnpe_mask][:, :3])[:n]
                ref_sub = rng.permutation(ref_cont[ref_mask])[:n]
                row["c2st"] = float(c2st(torch.tensor(mnpe_sub), torch.tensor(ref_sub)))
                # floor: reference split against itself, same per-side n where possible
                ref_perm = rng.permutation(ref_cont[ref_mask])
                half = len(ref_perm) // 2
                m = min(half, n)
                row["c2st_floor"] = float(
                    c2st(torch.tensor(ref_perm[:m]), torch.tensor(ref_perm[half : half + m]))
                )
                if verbose:
                    print(
                        f"  config ({c1_phys},{c2_phys}): mass ref {row['ref_mass']:.3f} / "
                        f"mnpe {row['mnpe_mass']:.3f}, n={n}, "
                        f"C2ST={row['c2st']:.3f} (floor {row['c2st_floor']:.3f})"
                    )
            rows.append(row)
    return rows


# ═══════════════════════════════════════════════════════════════════════
# D2 — conditional SBC (bespoke rank loop; run_sbc cannot inject theta_d)
# ═══════════════════════════════════════════════════════════════════════

theta_test, x_test_raw = generate_training_data(N_SBC_TEST, seed=SBC_TEST_SEED)
theta_test, x_test_raw = theta_test[:N_CAP], x_test_raw[:N_CAP]
x_test = transform_x(x_test_raw)
n_test = len(theta_test)


def compute_d2_ranks(post_rep):
    """Rank of theta*_c under q(theta_c | theta_d = theta*_d, x), per test case."""
    torch.manual_seed(SEED)
    return conditional_ranks(
        post_rep.posterior_estimator, theta_test, x_test, M_COND
    )


def ks_row(ranks_arr, grouping, group, mask, rep):
    r = ranks_arr[mask]
    pvals = check_uniformity_frequentist(r, M_COND)
    pvals = np.atleast_1d(np.asarray(pvals, dtype=float))
    return {
        "rep": rep,
        "grouping": grouping,
        "group": group,
        "n": int(mask.sum()),
        **{f"ks_pval_{name}": float(p) for name, p in zip(CONT_NAMES, pvals)},
    }


def compute_d2_rows(ranks_arr, rep):
    rows = [ks_row(ranks_arr, "pooled", "all", torch.ones(n_test, dtype=torch.bool), rep)]
    for j, label in [(3, "c1"), (4, "c2")]:
        for v in range(5):
            rows.append(
                ks_row(ranks_arr, f"by_{label}", str(v + DISCRETE_OFFSET),
                       theta_test[:, j] == v, rep)
            )
    for v1 in range(5):
        for v2 in range(5):
            mask = (theta_test[:, 3] == v1) & (theta_test[:, 4] == v2)
            if mask.sum() >= 20:
                rows.append(
                    ks_row(ranks_arr, "by_config",
                           f"({v1 + DISCRETE_OFFSET},{v2 + DISCRETE_OFFSET})", mask, rep)
                )
    return rows


# ── Run both diagnostics for every rep (resumable per rep) ──────────────────
print(f"\n── D2: conditional SBC ({n_test} test cases x {M_COND} conditional "
      f"samples), {N_REPS} reps ──")
all_d1, all_d2, ranks_by_rep = [], [], {}
for rep in range(N_REPS):
    part = PARTS_DIR / f"rep{rep}.pkl"
    if part.exists():
        with open(part, "rb") as f:
            cached = pickle.load(f)
        print(f"  rep {rep}: cached")
    else:
        print(f"  rep {rep}: computing ...")
        post_rep = post if rep == FIG_REP else load_posterior(rep)
        d1_rows = compute_d1(post_rep, rep, verbose=(rep == FIG_REP))
        ranks_rep = compute_d2_ranks(post_rep)
        cached = {
            "d1": d1_rows,
            "d2": compute_d2_rows(ranks_rep, rep),
            "ranks": ranks_rep.numpy(),
        }
        with open(part, "wb") as f:
            pickle.dump(cached, f)
    all_d1.extend(cached["d1"])
    all_d2.extend(cached["d2"])
    ranks_by_rep[rep] = torch.as_tensor(cached["ranks"])
    pooled = [r for r in cached["d2"] if r["grouping"] == "pooled"][0]
    print("    pooled KS p: " + "  ".join(
        f"{n}={pooled[f'ks_pval_{n}']:.4f}" for n in CONT_NAMES))

df_d1_all = pd.DataFrame(all_d1)
df_d1_all.to_csv(RESULTS_DIR / "conditional_diagnostics_d1.csv", index=False)
df_d1 = df_d1_all[df_d1_all["rep"] == FIG_REP].reset_index(drop=True)
ranks = ranks_by_rep[FIG_REP]

# Across-rep summary of the per-config C2ST -- the D1 headline.
d1_ok = df_d1_all[df_d1_all["c2st"].notna()]
per_rep_mean = d1_ok.groupby("rep")["c2st"].mean()
print(f"\n  D1 per-config C2ST, mean over populated configs: "
      f"{per_rep_mean.mean():.4f} +/- {per_rep_mean.std(ddof=0):.4f} across {N_REPS} reps "
      f"(range {d1_ok['c2st'].min():.3f}-{d1_ok['c2st'].max():.3f} over all cells/reps)")
n_eval = int(df_d1["c2st"].notna().sum())
skipped_mass = df_d1.loc[df_d1["c2st"].isna(), "ref_mass"].sum()
print(f"  rep {FIG_REP}: evaluated {n_eval}/25 configs; "
      f"skipped configs carry {skipped_mass:.4f} ref mass")

# ── D1 heatmap (figure rep only, labelled) ──
fig, ax = plt.subplots(figsize=(3.0, 2.6))
grid = df_d1["c2st"].to_numpy().reshape(5, 5)
im = ax.imshow(grid, vmin=0.5, vmax=max(0.65, np.nanmax(grid)), cmap="viridis")
for r in range(5):
    for c in range(5):
        row = df_d1.iloc[r * 5 + c]
        if np.isfinite(row["c2st"]):
            ax.text(
                c, r, f"{row['c2st']:.2f}\n({row['c2st_floor']:.2f})",
                ha="center", va="center", color="white",
            )
        else:
            ax.text(c, r, "—", ha="center", va="center", color="0.6")
ax.set_xticks(range(5), [str(int(c)) for c in C_VALUES])
ax.set_yticks(range(5), [str(int(c)) for c in C_VALUES])
ax.set_xlabel("c2 (servers)")
ax.set_ylabel("c1 (servers)")
fig.colorbar(im, ax=ax, shrink=0.8, label="C2ST (floor)")
ax.set_title(f"rep {FIG_REP} of {N_REPS}", fontsize="small", color="0.4")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(ARTIFACTS_DIR / f"conditional_diagnostics_d1.{ext}")
plt.close(fig)


# ── D2 reporting: all reps ──────────────────────────────────────────────────
df_d2_all = pd.DataFrame(all_d2)
df_d2_all.to_csv(
    RESULTS_DIR / "conditional_diagnostics_d2_ks.csv", index=False
)
df_d2 = df_d2_all[df_d2_all["rep"] == FIG_REP].reset_index(drop=True)

pooled_all = df_d2_all[df_d2_all["grouping"] == "pooled"]
print(f"\n  Pooled conditional-SBC KS p-values, all {N_REPS} reps:")
print(pooled_all[["rep"] + [f"ks_pval_{n}" for n in CONT_NAMES]].to_string(index=False))
print("\n  Across-rep summary (report THIS, not a single rep -- these p-values "
      "swing by orders of magnitude between reps):")
for name in CONT_NAMES:
    v = pooled_all[f"ks_pval_{name}"].to_numpy()
    print(f"    {name:>5}: median {np.median(v):.4f}  range {v.min():.4f}-{v.max():.4f}  "
          f"reps with p<0.05: {(v < 0.05).sum()}/{N_REPS}")

print(f"\n  Pooled + per-c1/per-c2 KS p-values (figure rep {FIG_REP}):")
print(df_d2[df_d2["grouping"] != "by_config"].to_string(index=False))
n_cfg = (df_d2["grouping"] == "by_config").sum()
pcols = [f"ks_pval_{n}" for n in CONT_NAMES]
n_low = int((df_d2.loc[df_d2["grouping"] == "by_config", pcols] < 0.05).sum().sum())
n_tests = int(n_cfg) * 3
print(f"  per-config groups: {n_cfg} (n>=20), {n_low}/{n_tests} KS tests with p<0.05")

# ── D2 pooled rank histograms ──
fig, axes = plt.subplots(1, 3, figsize=(6.0, 2.0), sharey=True)
n_bins = 20
expected = n_test / n_bins
band = 2 * np.sqrt(expected)  # ~95% Poisson band around uniform
for k, (ax, name) in enumerate(zip(axes, CONT_NAMES)):
    ax.hist(ranks[:, k].numpy(), bins=n_bins, range=(0, M_COND), color="C0", alpha=0.8)
    ax.axhline(expected, color="0.3", lw=0.8)
    ax.axhspan(expected - band, expected + band, color="0.3", alpha=0.15, zorder=0)
    ax.set_title(name)
    ax.set_xlabel("conditional rank")
    if k == 0:
        ax.set_ylabel("count")
    despine(ax)
fig.suptitle(f"rep {FIG_REP} of {N_REPS}", fontsize="small", color="0.4", y=1.02)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(ARTIFACTS_DIR / f"conditional_diagnostics_d2.{ext}")
plt.close(fig)

print("\nSaved: conditional_diagnostics_d1.csv/.pdf/.png, "
      "conditional_diagnostics_d2_ks.csv, conditional_diagnostics_d2.pdf/.png")
