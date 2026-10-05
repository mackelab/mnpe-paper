"""Derived metrics over every saved budget-study posterior (M1-M4).

Walks the posteriors the budget study saved and recomputes four quantities that
do not require retraining, writing two tables. The metric definitions live in
``mnpe_paper.models.queue.metrics``; this script is the driver.

    M1  C2ST noise floor from disjoint splits of the reference against itself.
    M2  Exact discrete total variation against the quadrature reference.
    M3  Marginal versus factor ECE, side by side.
    M4  The ECE null interval simulated from each model's own PMF.

M3 is reported as a pair on purpose. The factor form evaluates each
autoregressive term at the *true* preceding value, which is not available at
inference time, so it is systematically the easier number; the marginal form is
the one the paper's calibration claims rest on. M4 is an interval rather than a
point baseline, because a single expected value is not the yardstick a noisy
finite-sample statistic should be judged against.

Usage:
    uv run python experiments/queue/diagnostics/posterior_metrics.py
    uv run python experiments/queue/diagnostics/posterior_metrics.py --results-suffix _n1000

Reads (under ``artifacts/queue/``):
    budget_study_results{suffix}.pkl, deterministic_reference.npz, posteriors/

Writes (under ``results/queue/``):
    posterior_metrics{suffix}.csv  — M2/M3/M4 per budget and repetition
    c2st_floor.csv                 — M1, the level panel (b) of the figure draws
"""

from __future__ import annotations

import argparse
import pickle
import sys

import numpy as np
import pandas as pd
import torch

from mnpe_paper.calibration import run_discrete_calibration
from mnpe_paper.models.queue.config import (
    MODEL_VERSION,
    generate_reference_observation,
    generate_training_data,
    transform_x,
)
from mnpe_paper.models.queue.metrics import (
    c2st_split_floor,
    discrete_agreement,
    ece_null_interval,
    exact_joint_discrete,
)
from mnpe_paper.paths import model_artifacts_dir, model_results_dir

PARAM_NAMES = ("gamma", "mu1", "mu2", "c1", "c2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-suffix",
        default="",
        help="Use '_n1000' to read the reeval_budget_study.py output.",
    )
    parser.add_argument("--n-c2st", type=int, default=10_000)
    # Four splits consume 80,000 of the 90,000 reference draws left after
    # reserving the 10,000 used for the MNPE comparisons. Five would not fit.
    parser.add_argument("--n-floor-splits", type=int, default=4)
    parser.add_argument("--n-sbc-test", type=int, default=1_000)
    parser.add_argument("--sbc-test-seed", type=int, default=9999)
    parser.add_argument("--n-null-replicates", type=int, default=1_000)
    parser.add_argument("--num-bins", type=int, default=10)
    parser.add_argument("--null-seed", type=int, default=0)
    parser.add_argument(
        "--skip-ece",
        action="store_true",
        help="Skip M3/M4, which require loading every saved posterior.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    artifacts = model_artifacts_dir("queue")
    results_file = artifacts / f"budget_study_results{args.results_suffix}.pkl"
    reference_file = artifacts / "deterministic_reference.npz"
    posteriors_dir = artifacts / "posteriors"
    for path in (results_file, reference_file):
        if not path.exists():
            raise SystemExit(
                f"Missing {path}. Run the budget study and "
                "references/deterministic_reference_run.py first."
            )

    reference = np.load(reference_file)
    quadrature = reference["configuration_probabilities"]
    ref_samples = np.column_stack(
        [reference[name] for name in PARAM_NAMES]
    )
    # budget_study.py shuffles with default_rng(0) and then slices the first
    # n rows. Reproduce that exactly so the reserved block is the same one.
    np.random.default_rng(0).shuffle(ref_samples)

    with open(results_file, "rb") as handle:
        results = pickle.load(handle)
    budgets = results["budgets"]
    n_reps = results["n_reps"]

    observation = generate_reference_observation(seed=42)
    x_o = transform_x(
        torch.tensor(observation["x_array"], dtype=torch.float32).unsqueeze(0)
    ).squeeze(0)

    print("=" * 72)
    print(f"Derived posterior metrics: {MODEL_VERSION}{args.results_suffix}")
    print("=" * 72)

    # ── M1 ──
    print("\n--- M1: C2ST noise floor from disjoint reference splits ---")
    floor = c2st_split_floor(
        ref_samples, args.n_c2st, args.n_floor_splits, reserved=args.n_c2st
    )
    print(f"  splits: {', '.join(f'{s:.4f}' for s in floor['scores'])}")
    print(f"  floor : {floor['mean']:.4f} +/- {floor['std']:.4f}")
    print(
        "  This is a same-distribution comparison, so it is the floor the "
        "study's\n  C2ST protocol actually reports. A disagreement between two "
        "approximate\n  references measures reference error instead, and is "
        "not this quantity."
    )

    # ── M2/M3/M4 ──
    rows = []
    if not args.skip_ece:
        print(
            f"\n--- M3/M4: regenerating the {args.n_sbc_test}-case test set ---"
        )
        theta_test, x_test_raw = generate_training_data(
            args.n_sbc_test, seed=args.sbc_test_seed
        )
        x_test = transform_x(x_test_raw)

    print("\n--- M2: exact discrete TV against quadrature ---")
    print(
        f"{'budget':>8} {'rep':>4} {'joint TV':>10} {'max cell':>10} "
        f"{'c1 TV':>8} {'c2 TV':>8}"
        + ("" if args.skip_ece else f" {'ECE marg':>9} {'ECE fact':>9}")
    )

    for budget in budgets:
        for rep in range(n_reps):
            posterior_file = posteriors_dir / f"posterior_{budget}_rep{rep}.pkl"
            if not posterior_file.exists():
                print(f"  [skip] {posterior_file.name} not found")
                continue
            with open(posterior_file, "rb") as handle:
                posterior = pickle.load(handle)

            model_joint = exact_joint_discrete(posterior, x_o)
            agreement = discrete_agreement(model_joint, quadrature)

            row = {"budget": budget, "rep": rep, **agreement}

            if not args.skip_ece:
                ece_marginal, probs_marginal, _ = run_discrete_calibration(
                    theta_test,
                    x_test,
                    posterior,
                    discrete_indices=[3, 4],
                    num_bins=args.num_bins,
                    method="top_label",
                    probability_type="marginal",
                )
                ece_factor, _, _ = run_discrete_calibration(
                    theta_test,
                    x_test,
                    posterior,
                    discrete_indices=[3, 4],
                    num_bins=args.num_bins,
                    method="top_label",
                    probability_type="factor",
                )
                null = ece_null_interval(
                    probs_marginal,
                    args.num_bins,
                    "top_label",
                    args.n_null_replicates,
                    args.null_seed,
                )
                for index, name in enumerate(("c1", "c2")):
                    row[f"ece_marginal_{name}"] = float(ece_marginal[index])
                    row[f"ece_factor_{name}"] = float(ece_factor[index])
                    row[f"ece_null_mean_{name}"] = float(null["mean"][index])
                    row[f"ece_null_lo_{name}"] = float(null["q025"][index])
                    row[f"ece_null_hi_{name}"] = float(null["q975"][index])
                    row[f"ece_within_null_{name}"] = bool(
                        null["q025"][index]
                        <= float(ece_marginal[index])
                        <= null["q975"][index]
                    )

            rows.append(row)
            line = (
                f"{budget:>8,} {rep:>4} {agreement['joint_tv']:10.5f} "
                f"{agreement['max_cell_error']:10.5f} "
                f"{agreement['c1_tv']:8.5f} {agreement['c2_tv']:8.5f}"
            )
            if not args.skip_ece:
                line += (
                    f" {float(ece_marginal.mean()):9.5f}"
                    f" {float(ece_factor.mean()):9.5f}"
                )
            print(line, flush=True)

    if not rows:
        raise SystemExit(
            "No saved posteriors found. Run budget_study.py first."
        )

    frame = pd.DataFrame(rows)
    results_dir = model_results_dir("queue")
    output = results_dir / f"posterior_metrics{args.results_suffix}.csv"
    frame.to_csv(output, index=False)

    print("\n--- Summary across repetitions ---")
    for budget in budgets:
        if budget not in frame["budget"].values:
            continue
        subset = frame[frame["budget"] == budget]
        line = (
            f"  {budget:>7,}  joint TV {subset['joint_tv'].mean():.5f} "
            f"+/- {subset['joint_tv'].std():.5f}"
        )
        if not args.skip_ece:
            within = subset[["ece_within_null_c1", "ece_within_null_c2"]]
            line += (
                f"  ECE inside null interval: "
                f"{int(within.values.sum())}/{within.size}"
            )
        print(line)

    floor_frame = pd.DataFrame(
        [
            {
                "metric": "c2st_split_floor",
                "mean": floor["mean"],
                "std": floor["std"],
                "n_splits": args.n_floor_splits,
                "n_samples": args.n_c2st,
            }
        ]
    )
    floor_output = results_dir / "c2st_floor.csv"
    floor_frame.to_csv(floor_output, index=False)

    print(f"\nWrote {output.name} and {floor_output.name} to {results_dir}")


if __name__ == "__main__":
    sys.exit(main())
