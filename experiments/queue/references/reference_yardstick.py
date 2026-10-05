"""Place MNPE's residual on the scale of reference-method disagreement.

Compares MNPE at each budget against a chosen reference, alongside the gap
between the two independent reference methods (exact marginalization plus SMC,
and deterministic quadrature). That gap is the realistic accuracy limit for
this problem. ``--target`` selects which reference everything is scored
against; running both and getting the same conclusions is the point.

All comparisons use ``budget_study.py``'s protocol: 10,000 samples, seed 1.

The yardstick rationale, and why the per-configuration NUTS references are not
used, are in ../README.md under "Why the reference is built this way".

Usage:
    uv run python experiments/queue/references/reference_yardstick.py --target smc
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from sbi.utils.metrics import c2st

from mnpe_paper.models.queue.metrics import c2st_split_floor
from mnpe_paper.paths import model_artifacts_dir, model_results_dir

ARTIFACTS_DIR = model_artifacts_dir("queue")
RESULTS_DIR = model_results_dir("queue")
PARAM_NAMES = ("gamma", "mu1", "mu2", "c1", "c2")
N_C2ST = 10_000
N_FLOOR_SPLITS = 4

TARGETS = {
    # The best-practice MCMC route: exact marginalization verified against
    # pymc_extras.marginalize, sampled with SMC because NUTS cannot cross the
    # (mu1, mu2) basins. No quadrature anywhere.
    "smc": "smc_reference.npz",
    # Deterministic quadrature plus rejection sampling. Used to validate the
    # MCMC route, not as the headline comparator.
    "quadrature": "deterministic_reference.npz",
}


def stack(data) -> np.ndarray:
    return np.column_stack([data[name] for name in PARAM_NAMES])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        choices=sorted(TARGETS),
        default="smc",
        help="Which reference everything is compared against. Running both "
        "and getting the same conclusions is the point.",
    )
    args = parser.parse_args()

    target_path = ARTIFACTS_DIR / TARGETS[args.target]
    if not target_path.exists():
        raise SystemExit(
            f"Missing {target_path.name}. For 'smc' run smc_reference.py; "
            "for 'quadrature' run deterministic_reference_run.py."
        )
    reference = np.load(target_path)
    quadrature = reference["configuration_probabilities"]
    deterministic = stack(reference)
    np.random.default_rng(0).shuffle(deterministic)
    target = torch.tensor(deterministic[:N_C2ST], dtype=torch.float32)

    print("=" * 72)
    print(f"Reference disagreement yardstick (target: {args.target})")
    print("=" * 72)
    print(
        f"All comparisons: {N_C2ST:,} samples, seed=1, against the same "
        f"shuffled\nslice of {target_path.name}.\n"
    )

    # Cross-check: if the two references agree, no conclusion depends on which
    # is treated as ground truth.
    other = "quadrature" if args.target == "smc" else "smc"
    other_path = ARTIFACTS_DIR / TARGETS[other]
    if other_path.exists():
        other_data = np.load(other_path)
        other_samples = stack(other_data)
        np.random.default_rng(0).shuffle(other_samples)
        n = min(N_C2ST, len(other_samples))
        cross = c2st(
            torch.tensor(other_samples[:n], dtype=torch.float32),
            target[:n],
            seed=1,
        ).item()
        cross_tv = 0.5 * float(
            np.abs(other_data["configuration_probabilities"] - quadrature).sum()
        )
        print(
            f"Cross-check, {other} vs {args.target}: "
            f"C2ST {cross:.4f}, discrete TV {cross_tv:.5f}\n"
        )

    rows = []

    # The floor: disjoint slices of the deterministic reference itself.
    floor = c2st_split_floor(
        deterministic, n=N_C2ST, n_splits=N_FLOOR_SPLITS, reserved=N_C2ST
    )
    rows.append(
        {
            "source": "deterministic split (floor)",
            "joint_c2st": floor["mean"],
            "discrete_tv": 0.0,
        }
    )

    # Reference-disagreement scale: two independent converged methods. The
    # per-configuration references are excluded on purpose -- they share one
    # Laplace evidence approximation, so their agreement is not independence.
    if other_path.exists():
        rows.append(
            {
                "source": f"{other} vs {args.target} (reference gap)",
                "joint_c2st": cross,
                "discrete_tv": cross_tv,
            }
        )

    # Recomputed against the selected target, not carried over from
    # budget_study.py, which always used quadrature: the two references are
    # distinguishable at this sample size, so mixing them would matter.
    from mnpe_paper.models.queue.config import generate_reference_observation, transform_x
    from mnpe_paper.models.queue.metrics import discrete_agreement, exact_joint_discrete

    observation = generate_reference_observation(seed=42)
    x_o = transform_x(
        torch.tensor(observation["x_array"], dtype=torch.float32).unsqueeze(0)
    ).squeeze(0)

    def score_posteriors(paths: list[Path]) -> tuple[float, float] | None:
        scores, tvs = [], []
        for path in paths:
            with open(path, "rb") as handle:
                posterior = pickle.load(handle)
            # Seed the draw. Unseeded, a fixed posterior's C2ST moves by about
            # +/-0.01 between runs, the same size as the differences compared.
            torch.manual_seed(0)
            samples = posterior.sample((N_C2ST,), x=x_o).numpy()
            samples[:, 3] += 2
            samples[:, 4] += 2
            scores.append(
                c2st(
                    torch.tensor(samples, dtype=torch.float32), target, seed=1
                ).item()
            )
            tvs.append(
                discrete_agreement(
                    exact_joint_discrete(posterior, x_o), quadrature
                )["joint_tv"]
            )
        if not scores:
            return None
        return float(np.mean(scores)), float(np.mean(tvs))

    posteriors_dir = ARTIFACTS_DIR / "posteriors"
    results_file = ARTIFACTS_DIR / "budget_study_results.pkl"
    if posteriors_dir.exists() and results_file.exists():
        with open(results_file, "rb") as handle:
            study = pickle.load(handle)
        for budget in study["budgets"]:
            paths = sorted(posteriors_dir.glob(f"posterior_{budget}_rep*.pkl"))
            scored = score_posteriors(paths)
            if scored is None:
                continue
            joint_c2st, discrete_tv = scored
            rows.append(
                {
                    "source": f"MNPE, budget {budget:,} ({len(paths)} reps)",
                    "joint_c2st": joint_c2st,
                    "discrete_tv": discrete_tv,
                }
            )
            print(f"  scored budget {budget:,} ({len(paths)} reps)", flush=True)

    million = sorted(ARTIFACTS_DIR.glob("posterior_queue_1M_rep*.pkl"))
    scored = score_posteriors(million) if million else None
    if scored is not None:
        rows.append(
            {
                "source": f"MNPE, budget 1,000,000 ({len(million)} reps)",
                "joint_c2st": scored[0],
                "discrete_tv": scored[1],
            }
        )

    output = RESULTS_DIR / f"reference_yardstick_{args.target}.csv"
    if not any(row["source"].startswith("MNPE") for row in rows):
        raise SystemExit(
            f"No MNPE posteriors found; run main/budget_study.py first. {output.name} is unchanged."
        )

    # 0.5 + TV/2 is the Bayes bound implied by the discrete weights alone; the
    # excess is roughly the continuous contribution. The bound is attainable only
    # by a Bayes-optimal classifier, so a finite-sample random forest can fall
    # short and the excess is really (continuous contribution) minus (classifier
    # suboptimality). It can go slightly negative once the continuous term
    # vanishes, which is why the trend is the readable part, not the endpoint.
    print(
        f"\n{'source':>34} {'joint C2ST':>11} {'discrete TV':>12} "
        f"{'0.5+TV/2':>9} {'excess':>8}"
    )
    for row in rows:
        if row["discrete_tv"] > 0:
            bound = 0.5 + row["discrete_tv"] / 2
            print(
                f"{row['source']:>34} {row['joint_c2st']:11.4f} "
                f"{row['discrete_tv']:12.5f} {bound:9.4f} "
                f"{row['joint_c2st'] - bound:8.4f}"
            )
        else:
            print(
                f"{row['source']:>34} {row['joint_c2st']:11.4f} "
                f"{'exact':>12} {'--':>9} {'--':>8}"
            )

    import pandas as pd

    for row in rows:
        row["c2st_lower_bound_from_tv"] = (
            0.5 + row["discrete_tv"] / 2 if row["discrete_tv"] > 0 else np.nan
        )
        row["excess_over_discrete_bound"] = (
            row["joint_c2st"] - row["c2st_lower_bound_from_tv"]
            if row["discrete_tv"] > 0
            else np.nan
        )

    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"\nWrote {output.name}")


if __name__ == "__main__":
    main()
