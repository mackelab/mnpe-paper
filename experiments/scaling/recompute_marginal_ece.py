"""Re-derive the marginal discrete ECE of finished D-sweep runs, without retraining.

This produced the calibration column of the D >= 32 rows in
``results/scaling/scaling_study_results.csv``: 27 of the 90, the ones whose
``ece_recompute_device`` is set. Reproducing those rows needs this script as
well as ``scaling_study.py``, which is why it ships.

It is possible because the sweeps pickle every trained posterior (``--posterior-dir``)
and the calibration set is a deterministic function of D at a fixed seed (see
``sweep_common.make_test_set``). Raising the particle count is then a
marginalization cost rather than a training cost, and it is the only lever on
the Monte Carlo noise that the sampled marginalization path carries at D >= 32.

The marginalizer is a stack of dense MADE evaluations over
``num_samples x observations`` rows, which suits a GPU far better than the
batch-200 training loop does -- measured ~22x on a V100 against 8 CPU cores.
CPU and GPU marginals differ only through the RNG stream (their generators
diverge from the same seed), by well under the estimator's own noise.

Only the ``ece_*`` columns are rewritten; training-derived columns (timings,
memory, parameter counts) are carried through untouched, so a recomputed row
stays comparable to the rest of the grid on everything except the metric that
was deliberately re-estimated.

Usage:
    uv run python experiments/scaling/recompute_marginal_ece.py \
        --row-dir rows_scaling --posterior-dir posteriors_scaling --dims 32
"""

import argparse
import pickle
import time
from pathlib import Path

import pandas as pd
import torch

from mnpe_paper.models.scaling.sweep_common import (
    DEFAULT_MARGINAL_MAX_ELEMENTS,
    DEFAULT_MAX_EXACT_STATES,
    N_CALIBRATION,
    dual_discrete_ece,
    make_test_set,
    resolve_run_dir,
    run_key,
    write_row,
)


def move_posterior_to(posterior, device: str):
    """Put the discrete MADE on ``device``; the flow is never evaluated here."""
    estimator = posterior.potential_fn.posterior_estimator
    estimator.discrete_net.net.to(device)
    return posterior


def recompute_one(
    row: pd.Series,
    posterior_path: Path,
    device: str,
    ece_opts: dict,
) -> dict:
    D = int(row["D"])
    with open(posterior_path, "rb") as f:
        posterior = pickle.load(f)
    move_posterior_to(posterior, device)

    params_ece, x_ece = make_test_set(N_CALIBRATION, D)
    params_ece = params_ece.to(device)
    x_ece = x_ece.to(device)

    t0 = time.perf_counter()
    ece = dual_discrete_ece(
        params_ece,
        x_ece,
        posterior,
        discrete_indices=list(range(D, 2 * D)),
        **ece_opts,
    )
    elapsed = time.perf_counter() - t0

    updated = row.to_dict()
    updated.update({k: v for k, v in ece.items()})
    updated["ece_recompute_device"] = device
    updated["ece_recompute_wall_s"] = round(elapsed, 1)
    return updated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--row-dir", required=True, help="directory of per-run row CSVs")
    parser.add_argument("--posterior-dir", required=True, help="directory of pickles")
    parser.add_argument("--dims", type=int, nargs="+", required=True, help="D values to redo")
    parser.add_argument("--budgets", type=int, nargs="+", help="restrict budgets")
    parser.add_argument("--seeds", type=int, nargs="+", help="restrict seeds")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--marginal-samples", type=int, default=262144)
    parser.add_argument("--marginal-batch", type=int, default=128)
    parser.add_argument("--max-exact-states", type=int, default=DEFAULT_MAX_EXACT_STATES)
    parser.add_argument(
        "--marginal-max-elements", type=int, default=DEFAULT_MARGINAL_MAX_ELEMENTS
    )
    parser.add_argument("--ece-noise-check", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true", help="report what would be redone and exit"
    )
    args = parser.parse_args()

    row_dir = resolve_run_dir(args.row_dir)
    posterior_dir = resolve_run_dir(args.posterior_dir)
    ece_opts = dict(
        max_exact_states=args.max_exact_states,
        num_marginal_samples=args.marginal_samples,
        marginal_batch=args.marginal_batch,
        marginal_max_elements=args.marginal_max_elements,
        noise_check=args.ece_noise_check,
    )

    targets = []
    for path in sorted(row_dir.glob("D*.csv")):
        row = pd.read_csv(path).iloc[0]
        if int(row["D"]) not in args.dims:
            continue
        if args.budgets and int(row["budget"]) not in args.budgets:
            continue
        if args.seeds and int(row["seed"]) not in args.seeds:
            continue
        key = run_key(
            "D",
            int(row["D"]),
            int(row["budget"]),
            int(row["seed"]),
            int(row["hidden_features"]),
        )
        pkl = posterior_dir / f"{key}.pkl"
        if not pkl.exists():
            print(f"SKIP {key}: no saved posterior at {pkl}")
            continue
        targets.append((path, row, key, pkl))

    print(
        f"{len(targets)} row(s) to recompute on {args.device} "
        f"at {args.marginal_samples} samples"
        + (", noise-check on" if args.ece_noise_check else "")
    )
    if args.dry_run:
        for _, row, key, _ in targets:
            print(f"  {key}: ece_marginal_mean {row['ece_marginal_mean']:.6f}")
        return

    for i, (path, row, key, pkl) in enumerate(targets, 1):
        before = float(row["ece_marginal_mean"])
        updated = recompute_one(row, pkl, args.device, ece_opts)
        write_row(row_dir, key, updated)
        spread = updated.get("ece_marginal_mc_spread", float("nan"))
        print(
            f"[{i}/{len(targets)}] {key}: "
            f"ece_marginal_mean {before:.6f} -> {updated['ece_marginal_mean']:.6f}  "
            f"exact_dims {updated['ece_marginal_exact_dims']}  "
            f"mc_spread {spread:.6f}  "
            f"({updated['ece_recompute_wall_s']}s)"
        )

    print("Done. Re-run the sweep's --merge to refresh the aggregate CSV.")


if __name__ == "__main__":
    main()
