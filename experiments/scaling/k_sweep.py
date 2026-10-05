"""MNPE accuracy against the number of classes per discrete variable (Appendix B.1).

Companion to ``scaling_study.py``, which sweeps the number of discrete
dimensions D; here D=2 is fixed and K, the classes per dimension, is swept, so
the two sweeps separate the size of the discrete space from its shape.

Model: D=2 independent copies of the toy with K equally-spaced classes (mean A*k
for class k) — per-class separation held fixed, so this isolates "more classes"
from "harder classes". Adjacent classes are A/sqrt(TAU) ~ 1.8 marginal std apart
(moderate overlap, not near-deterministic); non-adjacent classes separate
further as K grows, so the total label-space size is what varies. The exact
posterior per dim is the softmax p(z_i=k|x) = softmax_k(-(x_i - A*k)^2 / 2 TAU)
(``exact_posterior.p_z_categorical``).

Per (K, budget, seed):
  - per-dim categorical TV vs the exact posterior, from the MADE's exact class
    probabilities (no sampling noise; see ``model_marginal_discrete_probs``)
  - per-dim discrete ECE under both probability definitions, plus the
    analytical finite-sample baseline (K-aware)
  - joint C2ST vs exact posterior samples (10 obs x 10k samples)
  - marginal SBC on the continuous dims

The category count is passed to MNPE explicitly, because sbi otherwise infers it
from the distinct values present in the training data, which is fragile at large K.

The grid below is 4 values of K x 2 budgets x 3 seeds. The published third tier
is the same grid at ``--budgets 1000000``, and the width probe behind the
paper's capacity claim is ``--width 128``; see ``experiments/scaling/README.md``.
Crash-safe append and resume by (K, budget, seed, width).

Usage:
    uv run python experiments/scaling/k_sweep.py [--smoke]
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
    p_z_categorical,
    sample_gt_posterior_k,
)
from mnpe_paper.models.scaling.sweep_common import (
    N_CALIBRATION,
    TEST_SEED,
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

OUT_FILE = model_results_dir("scaling") / "k_sweep_results.csv"
D = 2
K_GRID = [5, 10, 25, 50]
BUDGETS = [10_000, 100_000]
SEEDS = [0, 1, 2]
# `num_bins=10` is pinned so the architecture is visible here rather than
# inherited from a library default. Ten is what the published sweep used: this
# builds through `posterior_nn`, whose default is 10 and which forwards it
# explicitly, so the low-level builder's default never applies. Measured rather
# than read off a signature — see tests/test_architecture_pinning.py.
HPARAMS = dict(hidden_features=32, num_transforms=5, num_bins=10, num_blocks=4)
WIDTH_OVERRIDE = None  # set via --width for the capacity check (scales both heads)


def build_hparams():
    hp = dict(HPARAMS)
    if WIDTH_OVERRIDE is not None:
        hp["hidden_features"] = WIDTH_OVERRIDE
        hp["discrete_hidden_features"] = WIDTH_OVERRIDE
    return hp


N_SBC = 300  # test points for marginal SBC (needs sampling)
M_SBC = 2_000  # posterior samples per SBC test point
N_C2ST_OBS = 10
N_C2ST_SAMPLES = 10_000


def ks_uniform(ranks, m):
    return float(stats.kstest(ranks, stats.uniform(loc=0, scale=m).cdf).pvalue)


def model_marginal_discrete_probs(post, x, K):
    """Model's EXACT marginal q(z_d | x) per dim, no sampling noise (D=2).

    run_discrete_calibration returns the MADE's autoregressive factors, i.e.
    dim-1 is q(z_1 | z_0=true, x), not the marginal. To compare per-dim TV
    against the true marginal (and to match the D-sweep's marginal TV), we
    marginalize the AR factors over the preceding dim explicitly:
    q(z_1|x) = sum_{z0} q(z_0|x) q(z_1|z_0, x). Dim 0's factor already ignores
    the discrete input (autoregressive mask), so it is the exact marginal.
    """
    assert D == 2, "marginalization helper implemented for the K-sweep's D=2"
    made = post.posterior_estimator.discrete_net.net
    n = x.shape[0]

    def forward_probs(disc_values):
        d = disc_values.float()
        if made._needs_mapping:
            d = made._map_values_to_indices(d)
        logits = made.forward(d, condition=x).reshape(n, D, K)
        return torch.softmax(logits, dim=-1)

    with torch.no_grad():
        q0 = forward_probs(torch.zeros(n, D))[:, 0, :]  # exact q(z_0|x)
        q1 = torch.zeros(n, K)
        for z0 in range(K):
            disc = torch.zeros(n, D)
            disc[:, 0] = z0
            q1 += q0[:, z0:z0 + 1] * forward_probs(disc)[:, 1, :]
    return torch.stack([q0, q1], dim=1).numpy()  # (n, D, K)


def run_one(K, budget, seed, ece_opts=None, posterior_dir=None):
    ece_opts = ece_opts or {}
    first_in_process = begin_run()
    probe_time = machine_speed_probe()
    theta_train, x_train = generate_training_data(budget, D, seed, K)
    # fail-fast: MNPE's categorical head requires every class present in the
    # data (else a loud crash mid-eval). Negligible risk at this grid, but a
    # future budget cut / K bump should fail here with a clear message.
    for d in range(D):
        n_unique = theta_train[:, D + d].unique().numel()
        assert n_unique == K, f"dim {d}: only {n_unique}/{K} classes in training data; raise budget"
    torch.manual_seed(seed)
    hparams = build_hparams()
    de = posterior_nn(
        model="mnpe",
        num_categories_per_variable=torch.tensor([K] * D),
        **hparams,
    )
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer = MNPE(density_estimator=de, show_progress_bars=False)
        trainer.append_simulations(theta_train, x_train).train(show_train_summary=False)
        post = trainer.build_posterior()
    train_time = time.perf_counter() - t0
    rss_after_train = rss_peak_mb()

    # ECE under BOTH definitions: the marginal q(z_i|x) the paper defines, and the
    # teacher-forced factor q(z_1|z_0=true,x), recorded as a diagnostic.
    # With D=2 the prefix space is just K states, so this is always exact here.
    params_eval, x_eval = make_test_set(N_CALIBRATION, D, K)
    ece = dual_discrete_ece(
        params_eval,
        x_eval,
        post,
        discrete_indices=list(range(D, 2 * D)),
        num_bins=10,
        **ece_opts,
    )
    # categorical TV per dim: model's EXACT MARGINAL vs the true marginal
    # (marginalized AR factors; matches the D-sweep's marginal TV, no sampling
    # noise). Using the raw AR factors would measure q(z_1|z_0=true,x) for dim 1,
    # which conflates any learned spurious coupling with calibration error.
    p_exact = p_z_categorical(x_eval.numpy(), K)  # (N, D, K)
    q = model_marginal_discrete_probs(post, x_eval, K)  # (N, D, K)
    tv = 0.5 * np.abs(q - p_exact).sum(-1)  # (N, D)

    # joint C2ST vs exact posterior samples
    rng = np.random.default_rng(seed)
    c2st_scores = []
    for i in range(N_C2ST_OBS):
        gt = torch.tensor(
            sample_gt_posterior_k(x_eval[i].numpy(), N_C2ST_SAMPLES, K, rng=rng),
            dtype=torch.float32,
        )
        s = post.sample((N_C2ST_SAMPLES,), x=x_eval[i].reshape(1, -1), show_progress_bars=False)
        c2st_scores.append(float(c2st(gt, s)))

    # marginal SBC on continuous dims (disjoint test set from the eval set)
    params_sbc, x_sbc = make_test_set(N_SBC, D, K, seed=TEST_SEED + 1)
    sbc_ranks = np.zeros((N_SBC, D))
    for i in range(N_SBC):
        s = post.sample((M_SBC,), x=x_sbc[i].reshape(1, -1), show_progress_bars=False)
        sbc_ranks[i] = (s[:, :D].numpy() < params_sbc[i, :D].numpy()).sum(0)
    sbc_p = [ks_uniform(sbc_ranks[:, d], M_SBC) for d in range(D)]

    if posterior_dir is not None:
        save_posterior(
            post,
            posterior_dir / f"{run_key('K', K, budget, seed, hparams['hidden_features'])}.pkl",
        )

    return {
        "K": K,
        "budget": budget,
        "seed": seed,
        "hidden_features": hparams["hidden_features"],
        "train_time_s": round(train_time, 1),
        "probe_time_s": round(probe_time, 3),
        "tv_disc_mean": tv.mean(),
        "tv_disc_max_dim": tv.mean(0).max(),
        **ece,
        "ece_baseline": ece_baseline_analytical(N=N_CALIBRATION, B=10, K=K),
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
    }


def main():
    global K_GRID, BUDGETS, SEEDS, N_CALIBRATION, N_SBC, M_SBC, N_C2ST_SAMPLES, WIDTH_OVERRIDE
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--k", type=int, nargs="+", help="restrict K grid")
    parser.add_argument("--budgets", type=int, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--width", type=int, help="override hidden width (both heads)")
    parser.add_argument(
        "--out", type=str, help="override output CSV (keeps one-offs out of the main grid)"
    )
    add_marginal_ece_args(parser)
    parser.add_argument(
        "--merge", action="store_true", help="merge --row-dir into --out and exit (no training)"
    )
    args = parser.parse_args()
    if args.smoke:
        K_GRID, BUDGETS, SEEDS = [5], [2_000], [0]
        N_CALIBRATION, N_SBC, M_SBC, N_C2ST_SAMPLES = 100, 30, 500, 1_000
    K_GRID = args.k or K_GRID
    BUDGETS = args.budgets or BUDGETS
    SEEDS = args.seeds or SEEDS
    WIDTH_OVERRIDE = args.width

    if args.out:
        out_file = Path(args.out)
    elif args.smoke:
        # Matches the ``*_SMOKE.csv`` name .gitignore excludes, so a pipeline
        # check never shows up as an untracked result table.
        out_file = OUT_FILE.with_name("k_sweep_results_SMOKE.csv")
    else:
        out_file = OUT_FILE
    if args.merge:
        if not args.row_dir:
            raise SystemExit("--merge requires --row-dir")
        n = merge_row_dir(
            resolve_run_dir(args.row_dir),
            out_file,
            sort_by=["budget", "K", "seed"],
            prefix="K",
            key_col="K",
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

    current_hf = WIDTH_OVERRIDE if WIDTH_OVERRIDE is not None else HPARAMS["hidden_features"]
    done = set()
    if row_dir is None and out_file.exists() and not args.smoke:
        prev = pd.read_csv(out_file)
        # dedup on width too; a file without the column is all width 32.
        hf = prev["hidden_features"] if "hidden_features" in prev.columns else 32
        done = set(zip(prev["K"], prev["budget"], prev["seed"], pd.Series(hf, index=prev.index)))
        print(f"Resuming: {len(done)} runs already in {out_file.name}")

    grid = [(K, b, s) for b in BUDGETS for K in K_GRID for s in SEEDS]
    for k, (K, budget, seed) in enumerate(grid):
        key = run_key("K", K, budget, seed, current_hf)
        if row_dir is not None:
            if row_exists(row_dir, key):
                continue
        elif (K, budget, seed, current_hf) in done:
            continue
        if args.settle_seconds:
            time.sleep(args.settle_seconds)
        print(f"[{k + 1}/{len(grid)}] K={K} budget={budget} seed={seed} ...", flush=True)
        row = run_one(K, budget, seed, ece_opts=ece_opts, posterior_dir=posterior_dir)
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
