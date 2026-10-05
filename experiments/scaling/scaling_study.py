"""MNPE accuracy against the number of discrete dimensions D (paper Appendix B.1).

Trains MNPE on D independent copies of the tractable toy model and scores it
against the exact posterior, which stays in closed form at every D because the
model factorizes. Architecture: the toy's nflows NSF held fixed up to D=12
(capacity deliberately not re-tuned per D); for D >= 16 the width scales as 4*D
capped at 256, for both heads (see ``hparams_for``). From D >= 17, 2^D exceeds
the training budget, so most configurations are never seen during training —
the large-D runs test generalization across configurations rather than
per-configuration coverage. Metrics per run:

  - per-dim discrete TV (exact posterior, MNPE marginals from samples)
  - per-dim discrete ECE under both probability definitions (direct MADE query)
    plus the analytical finite-sample baseline
  - joint C2ST against exact posterior samples (the toy budget-study protocol)
  - marginal SBC on the continuous dims (KS p per dim)
  - conditional SBC at D in {2, 8, 32} (rank theta* under q(theta | z*, x))
  - training wall-clock, peak memory, and parameter counts per head

The grid below is 10 values of D x 2 budgets x 3 seeds. The published third
tier is the same grid at ``--budgets 1000000``; see ``experiments/scaling/README.md``
for the commands the tracked results came from.

Results append to ``results/scaling/scaling_study_results.csv`` after every run
(crash-safe); completed (D, budget, seed, width) rows are skipped on restart.
Prefer ``--row-dir`` for anything parallel or long-running: it writes one file
per run through an atomic rename, and ``--merge`` assembles them.

Usage:
    uv run python experiments/scaling/scaling_study.py [--smoke]
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sbi.inference import MNPE
from sbi.neural_nets import posterior_nn
from sbi.utils.metrics import c2st
from scipy import stats

from mnpe_paper.calibration import ece_baseline_analytical
from mnpe_paper.models.scaling.config import generate_training_data
from mnpe_paper.models.scaling.exact_posterior import (
    sample_gt_posterior,
    tv_discrete_per_dim,
)
from mnpe_paper.models.scaling.sweep_common import (
    M_TV_SAMPLES,
    N_CALIBRATION,
    N_TV_TEST,
    add_marginal_ece_args,
    append_row_checked,
    begin_run,
    dual_discrete_ece,
    machine_speed_probe,
    make_test_set,
    merge_row_dir,
    parameter_counts,
    resolve_run_dir,
    row_exists,
    rss_baseline_mb,
    rss_peak_mb,
    run_key,
    save_posterior,
    tensor_mb,
    write_row,
)
from mnpe_paper.paths import model_results_dir

OUT_FILE = model_results_dir("scaling") / "scaling_study_results.csv"
D_GRID = [2, 4, 6, 8, 10, 12, 16, 32, 64, 128]
BUDGETS = [10_000, 100_000]
SEEDS = [0, 1, 2]
# conditional-SBC hook: low-D anchor, per-config data-scarcity stress (D=8),
# and the unseen-configuration regime (D=32: 2^32 configs >> budget)
COND_SBC_DIMS = (2, 8, 32)


WIDTH_OVERRIDE = None  # set via --width for one-off capacity experiments


def hparams_for(D):
    """Fixed toy architecture up to D=12; above that, width scales with D
    (4*D, capped at 256) for both the flow and the MADE discrete head.

    NOTE: no `hidden_layers` — build_nsf has no such parameter and silently
    drops it via **kwargs (same dead-kwarg finding as in the queueing model).
    """
    h = WIDTH_OVERRIDE or (32 if D <= 12 else min(4 * D, 256))
    # `num_bins=10` is pinned so the architecture is visible here rather than
    # inherited. Ten is what the published sweep used: this builds through
    # `posterior_nn`, whose default is 10 and which forwards it explicitly, so
    # the low-level builder's default never applies. Measured rather than read
    # off a signature — see tests/test_architecture_pinning.py.
    return dict(
        hidden_features=h,
        discrete_hidden_features=h,
        num_transforms=5,
        num_bins=10,
        num_blocks=4,
    )


# Shared with the figure, which draws the sampling floor these two determine.
N_TEST = N_TV_TEST  # test observations for TV + SBC
M_SAMPLES = M_TV_SAMPLES  # posterior samples per test observation
N_C2ST_OBS = 10
N_C2ST_SAMPLES = 10_000  # project-wide C2ST protocol (cf. dequantization baseline)
M_COND = 1_000  # conditional-SBC samples per test point


def ks_uniform(ranks, m):
    return float(stats.kstest(ranks, stats.uniform(loc=0, scale=m).cdf).pvalue)


def conditional_sbc(post, params, x, D):
    """KS p per (continuous dim, z*_d value) for ranks of theta* under q(theta | z*, x).

    Grouped by the conditioning value z*_d of the same dim: opposite rank tilts
    for z*_d = 0 vs 1 cancel when pooled, so pooled-only KS can hide coupling
    failures.
    Full-configuration grouping is infeasible at D = 8 (2^8 groups for 300 test
    points); per-dim grouping targets exactly the cancellation mechanism.
    """
    est = post.posterior_estimator
    n = params.shape[0]
    ranks = np.zeros((n, D))
    with torch.no_grad():
        for i in range(n):
            emb = est.condition_embedding(x[i].reshape(1, -1))
            disc = params[i, D:].reshape(1, -1).repeat(M_COND, 1)
            comb = torch.cat((disc, emb.repeat(M_COND, 1)), dim=-1)
            cont = est.continuous_net.sample(torch.Size([]), condition=comb)
            if est.log_transform_input:
                cont = cont.exp()
            ranks[i] = (cont.numpy() < params[i, :D].numpy()).sum(0)
    z_star = params[:, D:].numpy()
    return [
        ks_uniform(ranks[z_star[:, d] == z_val, d], M_COND)
        for d in range(D)
        for z_val in (0.0, 1.0)
    ]


def run_one(D, budget, seed, ece_opts=None, posterior_dir=None):
    ece_opts = ece_opts or {}
    first_in_process = begin_run()
    probe_time = machine_speed_probe()
    theta_train, x_train = generate_training_data(budget, D, seed)
    torch.manual_seed(seed)
    hparams = hparams_for(D)
    de = posterior_nn(model="mnpe", **hparams)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer = MNPE(density_estimator=de, show_progress_bars=False)
        trainer.append_simulations(theta_train, x_train).train(show_train_summary=False)
        post = trainer.build_posterior()
    train_time = time.perf_counter() - t0
    rss_after_train = rss_peak_mb()

    params_test, x_test = make_test_set(N_TEST, D)
    rng = np.random.default_rng(seed)

    # TV per dim + marginal SBC from the same posterior samples
    tv = np.zeros((N_TEST, D))
    sbc_ranks = np.zeros((N_TEST, D))
    for i in range(N_TEST):
        s = post.sample((M_SAMPLES,), x=x_test[i].reshape(1, -1), show_progress_bars=False)
        tv[i] = tv_discrete_per_dim(s[:, D:].mean(0).numpy(), x_test[i].numpy())
        sbc_ranks[i] = (s[:, :D].numpy() < params_test[i, :D].numpy()).sum(0)
    sbc_p = [ks_uniform(sbc_ranks[:, d], M_SAMPLES) for d in range(D)]

    # joint C2ST vs exact posterior samples
    c2st_scores = []
    for i in range(N_C2ST_OBS):
        gt = torch.tensor(
            sample_gt_posterior(x_test[i].numpy(), N_C2ST_SAMPLES, rng=rng),
            dtype=torch.float32,
        )
        s = post.sample((N_C2ST_SAMPLES,), x=x_test[i].reshape(1, -1), show_progress_bars=False)
        c2st_scores.append(float(c2st(gt, s)))

    # per-dim discrete ECE under BOTH probability definitions + analytical baseline
    params_ece, x_ece = make_test_set(N_CALIBRATION, D)
    ece = dual_discrete_ece(
        params_ece, x_ece, post, discrete_indices=list(range(D, 2 * D)), **ece_opts
    )

    if posterior_dir is not None:
        save_posterior(
            post,
            posterior_dir / f"{run_key('D', D, budget, seed, hparams['hidden_features'])}.pkl",
        )

    row = {
        "D": D,
        "budget": budget,
        "seed": seed,
        "hidden_features": hparams["hidden_features"],
        "train_time_s": round(train_time, 1),
        "probe_time_s": round(probe_time, 3),
        "tv_disc_mean": tv.mean(),
        "tv_disc_max_dim": tv.mean(0).max(),
        **ece,
        "ece_baseline": ece_baseline_analytical(N=N_CALIBRATION, B=10, K=2),
        **parameter_counts(post),
        "train_tensor_mb": round(tensor_mb(theta_train, x_train), 2),
        "rss_baseline_mb": round(rss_baseline_mb(), 1),
        "rss_peak_after_train_mb": round(rss_after_train, 1),
        "rss_peak_total_mb": round(rss_peak_mb(), 1),
        "rss_first_run_in_process": first_in_process,
        "c2st_joint_mean": np.mean(c2st_scores),
        "c2st_joint_std": np.std(c2st_scores),
        "sbc_ks_p_mean": np.mean(sbc_p),
        "sbc_ks_p_min": np.min(sbc_p),
        "cond_sbc_p_mean": np.nan,
        "cond_sbc_p_min": np.nan,
    }
    if D in COND_SBC_DIMS:
        cond_p = conditional_sbc(post, params_test, x_test, D)
        row["cond_sbc_p_mean"] = np.mean(cond_p)
        row["cond_sbc_p_min"] = np.min(cond_p)
    return row


def main():
    global N_TEST, M_SAMPLES, N_CALIBRATION, D_GRID, BUDGETS, SEEDS, WIDTH_OVERRIDE
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="tiny run to validate pipeline")
    parser.add_argument("--dims", type=int, nargs="+", help="restrict D grid")
    parser.add_argument("--budgets", type=int, nargs="+", help="override budgets")
    parser.add_argument("--seeds", type=int, nargs="+", help="override seeds")
    parser.add_argument("--width", type=int, help="override hidden width (both heads)")
    parser.add_argument(
        "--out", type=str, help="override output CSV (keeps one-off runs out of the main grid)"
    )
    add_marginal_ece_args(parser)
    parser.add_argument(
        "--merge", action="store_true", help="merge --row-dir into --out and exit (no training)"
    )
    args = parser.parse_args()

    if args.smoke:
        D_GRID, BUDGETS, SEEDS = [2], [1_000], [0]
        N_TEST, M_SAMPLES, N_CALIBRATION = 30, 500, 100
    D_GRID = args.dims or D_GRID
    BUDGETS = args.budgets or BUDGETS
    SEEDS = args.seeds or SEEDS
    WIDTH_OVERRIDE = args.width

    if args.out:
        out_file = Path(args.out)
    elif args.smoke:
        # Matches the ``*_SMOKE.csv`` name .gitignore excludes, so a pipeline
        # check never shows up as an untracked result table.
        out_file = OUT_FILE.with_name("scaling_study_results_SMOKE.csv")
    else:
        out_file = OUT_FILE

    if args.merge:
        if not args.row_dir:
            raise SystemExit("--merge requires --row-dir")
        n = merge_row_dir(
            resolve_run_dir(args.row_dir),
            out_file,
            sort_by=["budget", "D", "seed"],
            prefix="D",
            key_col="D",
        )
        print(f"Merged {n} rows -> {out_file}")
        return

    ece_opts = dict(
        max_exact_states=args.max_exact_states,
        num_marginal_samples=args.marginal_samples,
        marginal_batch=args.marginal_batch,
        marginal_max_elements=args.marginal_max_elements,
        noise_check=args.ece_noise_check,
    )
    row_dir = resolve_run_dir(args.row_dir) if args.row_dir else None
    posterior_dir = resolve_run_dir(args.posterior_dir) if args.posterior_dir else None

    # Single-file mode appends and resumes. The dedup key includes the width, so
    # a width override is not skipped as already done.
    done = set()
    if row_dir is None and out_file.exists() and not args.smoke:
        prev = pd.read_csv(out_file)
        hf = prev["hidden_features"] if "hidden_features" in prev.columns else 32
        done = set(
            zip(prev["D"], prev["budget"], prev["seed"], pd.Series(hf, index=prev.index))
        )
        print(f"Resuming: {len(done)} runs already in {out_file.name}")

    grid = [(D, b, s) for b in BUDGETS for D in D_GRID for s in SEEDS]
    for k, (D, budget, seed) in enumerate(grid):
        width = WIDTH_OVERRIDE or hparams_for(D)["hidden_features"]
        key = run_key("D", D, budget, seed, width)
        if row_dir is not None:
            if row_exists(row_dir, key):
                continue
        elif (D, budget, seed, width) in done:
            continue
        if args.settle_seconds:
            time.sleep(args.settle_seconds)
        print(f"[{k + 1}/{len(grid)}] D={D} budget={budget} seed={seed} ...", flush=True)
        row = run_one(D, budget, seed, ece_opts=ece_opts, posterior_dir=posterior_dir)
        if row_dir is not None:
            write_row(row_dir, key, row)
        else:
            append_row_checked(out_file, row)
        print(
            f"    train {row['train_time_s']}s  TV {row['tv_disc_mean']:.4f}  "
            f"ECE marg {row['ece_marginal_mean']:.4f} / fact {row['ece_factor_mean']:.4f} "
            f"(floor {row['ece_baseline']:.4f}, "
            f"{'exact' if row['ece_marginal_exact'] else 'MC'} {row['ece_marginal_time_s']}s)  "
            f"C2ST {row['c2st_joint_mean']:.3f}  SBC p_min {row['sbc_ks_p_min']:.3f}",
            flush=True,
        )
    print("Done.")


if __name__ == "__main__":
    main()
