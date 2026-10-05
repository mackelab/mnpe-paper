"""Create the high-accuracy deterministic/rejection queue reference artifact.

Usage:
    uv run python experiments/queue/references/deterministic_reference_run.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from mnpe_paper.models.queue.config import generate_reference_observation
from mnpe_paper.models.queue.reference import (
    CONFIGURATIONS,
    configuration_evidences,
    sample_posterior_rejection,
)
from mnpe_paper.models.queue.spec import MODEL_VERSION
from mnpe_paper.paths import model_artifacts_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epsrel", type=float, default=1e-9)
    parser.add_argument(
        "--verify-epsrel",
        type=float,
        default=1e-7,
        help="Second, looser calculation used to verify numerical stability.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=model_artifacts_dir("queue") / "deterministic_reference.npz",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    observations = generate_reference_observation(seed=42)
    print("=" * 72)
    print(f"Deterministic queue reference: {MODEL_VERSION}")
    print("=" * 72)

    start = time.time()
    result = configuration_evidences(
        observations,
        epsabs=1e-10,
        epsrel=args.epsrel,
    )
    verification = configuration_evidences(
        observations,
        epsabs=1e-10,
        epsrel=args.verify_epsrel,
    )
    max_probability_difference = float(
        np.max(np.abs(result.probabilities - verification.probabilities))
    )
    relative_quadrature_error = (
        result.absolute_error / result.evidences.sum()
    )

    print("\nConfiguration posterior from adaptive quadrature:")
    order = np.argsort(-result.probabilities)
    for index in order:
        c1, c2 = result.configurations[index]
        print(
            f"  p({c1},{c2} | x) = {result.probabilities[index]:.10f}"
        )
    print(
        "\nQuadrature checks: "
        f"reported relative error={relative_quadrature_error:.3e}, "
        f"max cross-tolerance probability change="
        f"{max_probability_difference:.3e}"
    )

    samples, acceptance_rate = sample_posterior_rejection(
        observations,
        args.samples,
        seed=args.seed,
    )
    sample_configurations = np.stack(
        [samples["c1"], samples["c2"]], axis=1
    )
    empirical_probabilities = np.asarray(
        [
            np.mean(np.all(sample_configurations == config, axis=1))
            for config in CONFIGURATIONS
        ]
    )
    standard_errors = np.sqrt(
        result.probabilities
        * (1.0 - result.probabilities)
        / args.samples
    )
    safe_standard_errors = np.maximum(standard_errors, 1.0 / args.samples)
    max_sampling_z = float(
        np.max(
            np.abs(empirical_probabilities - result.probabilities)
            / safe_standard_errors
        )
    )

    elapsed = time.time() - start
    np.savez(
        args.output,
        model_version=np.array(MODEL_VERSION),
        implementation=np.array(
            "adaptive_factorized_quadrature_plus_bounded_rejection_v1"
        ),
        gamma=samples["gamma"],
        mu1=samples["mu1"],
        mu2=samples["mu2"],
        c1=samples["c1"],
        c2=samples["c2"],
        configurations=result.configurations,
        configuration_evidences=result.evidences,
        configuration_probabilities=result.probabilities,
        discrete_posterior=np.array(
            [
                (c1, c2, probability)
                for (c1, c2), probability in zip(
                    result.configurations,
                    result.probabilities,
                )
            ],
            dtype=[("c1", int), ("c2", int), ("prob", float)],
        ),
        empirical_configuration_probabilities=empirical_probabilities,
        quadrature_absolute_error=np.array(result.absolute_error),
        quadrature_relative_error=np.array(relative_quadrature_error),
        cross_tolerance_max_probability_difference=np.array(
            max_probability_difference
        ),
        rejection_acceptance_rate=np.array(acceptance_rate),
        empirical_max_standardized_error=np.array(max_sampling_z),
        elapsed_seconds=np.array(elapsed),
    )
    print(
        f"\nSaved {args.samples:,} independent posterior draws to "
        f"{args.output}"
    )
    print(
        f"Rejection acceptance={acceptance_rate:.4%}; "
        f"max empirical discrepancy={max_sampling_z:.2f} standard errors; "
        f"wall time={elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
