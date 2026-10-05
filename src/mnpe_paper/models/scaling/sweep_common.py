"""Shared plumbing for the two scaling sweeps.

Used by ``experiments/scaling/scaling_study.py`` and
``experiments/scaling/k_sweep.py``, which differ only in what they sweep.
Three concerns:

1. **Dual discrete ECE.** :func:`mnpe_paper.calibration.run_discrete_calibration`
   scores either teacher-forced autoregressive factors ``q(d_i | d_<i,true, x)``
   or genuine marginals ``q(d_i | x)``. The two coincide only for the first
   discrete variable, so every sweep row carries both, under unambiguous column
   names. A single ``ece_mean`` column is deliberately not written: the whole
   reason for this module is that one such column silently meant two different
   things at different D.

2. **Bounded memory during marginalization.** Exact marginalization enumerates
   ``prod(K_i)`` prefixes and the Monte Carlo fallback carries
   ``num_samples`` particles; either is multiplied by the observation batch
   before hitting the MADE. At D=128 the naive batch of 128 allocates on the
   order of a gigabyte, which is fine serially and fatal when eight grid
   points run concurrently. ``auto_marginal_batch`` caps the row count instead.

3. **Race-free parallel output.** The append-to-one-CSV pattern assumes a
   single writer. Run the grid points concurrently and the appends interleave
   and corrupt rows, so each run writes its own one-row CSV into a directory and
   ``merge_row_dir`` concatenates them afterwards. Resume is then just "does
   this run's file already exist", which also makes partial grids visible.
"""

import resource
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch

from mnpe_paper.calibration import (
    exact_marginal_variable_count,
    run_discrete_calibration,
)
from mnpe_paper.models.scaling.config import sample_prior, simulator
from mnpe_paper.paths import model_artifacts_dir

# The evaluation set is a deterministic function of (seed, n, D, K), so every run
# in a grid scores against the same observations, and a metric can be re-derived
# later from a saved posterior without keeping the data. ``recompute_marginal_ece``
# depends on exactly that.
TEST_SEED = 789

# Observations behind every discrete-ECE number in both sweeps, and the size
# ``recompute_marginal_ece`` must reuse to stay comparable with the row it
# rewrites. **No function in this module may read it.** Both sweeps rebind the
# name in their own namespace under ``--smoke``, which shadows this binding
# rather than changing it, so a reader here would see 1000 during a smoke run
# while every other scaled-down quantity was 100 — and nothing would say so.
N_CALIBRATION = 1_000

# Shared with figures/scaling_figure.py, which draws the sampling floor
# sqrt(2p(1-p)/(pi*M)) that measured TV cannot fall below. That floor is only
# the right curve for the M and the test set the sweep actually used, so the two
# read one definition rather than restating it either side of the pipeline.
N_TV_TEST = 300
M_TV_SAMPLES = 4_000


def make_test_set(n: int, D: int, K: int = 2, seed: int = TEST_SEED):
    """Prior-predictive evaluation set, fixed across the runs at one (D, K)."""
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        params = sample_prior(n, D, K)
        x = simulator(params, D)
    return params, x


def run_key(sweep: str, value: int, budget: int, seed: int, width: int) -> str:
    """Filename key for one grid point.

    ``sweep`` is the merge prefix, "D" or "K" — the same letter ``merge_row_dir``
    globs on, which is what keeps two sweeps sharing a row directory from
    splicing into each other.
    """
    return f"{sweep}{value}_b{budget}_s{seed}_w{width}"


def resolve_run_dir(path: str) -> Path:
    """Resolve a ``--row-dir`` or ``--posterior-dir`` to somewhere gitignored.

    These hold per-run rows and pickled posteriors: intermediates, regenerable,
    and large. A bare relative name on the command line would otherwise land
    beside whatever the reader's working directory happens to be, so relative
    paths resolve under ``artifacts/scaling/`` and only an absolute path escapes.
    """
    candidate = Path(path)
    return candidate if candidate.is_absolute() else model_artifacts_dir("scaling") / candidate

# Exact enumeration stays affordable well past the calibration module's
# conservative 4096: for the
# binary D-sweep, 2**15 prefixes at D=16 costs about the same as that D's Monte
# Carlo path (~8.4M forward rows either way) and is noiseless. D=32 and beyond
# remain out of reach, so they still fall back to Monte Carlo.
DEFAULT_MAX_EXACT_STATES = 32_768
DEFAULT_MARGINAL_SAMPLES = 4_096
# Peak allocation is counted in ELEMENTS, not rows: a row-based cap would never
# engage at D=128, because the Monte Carlo row count
# (num_samples * batch) is independent of D while the tensors are not.
#
# The cap is deliberately optimistic by a factor of ``max_num_categories``, and
# knowing which way is what makes it usable. ``auto_marginal_batch`` counts
# ``states * batch * num_variables``, which sizes the particle and prefix
# tensors exactly; the MADE's logits are ``max_num_categories`` times wider,
# since ``CategoricalMADE`` sets ``output_multiplier`` to that. So 64M elements
# is ~256 MB for the particle tensors and ~512 MB for the binary sweeps' logits,
# plus a softmax copy of the latter — budget on the order of a gigabyte per
# worker at D >= 32, not 256 MB.
#
# The arithmetic is left as the published runs used it rather than corrected,
# because the batch it produces is not only a memory knob: on the sampled path it
# selects the RNG chunking (see ``auto_marginal_batch``), and every Monte Carlo
# row records the batch it used — 7 at D=32 and D=128, 15 at D=64 — as the
# provenance needed to reproduce it. Tightening the cap would silently move those
# rows. Lower ``--marginal-max-elements`` to fit a smaller machine, keeping it
# constant across any grid meant to be compared.
DEFAULT_MARGINAL_MAX_ELEMENTS = 64_000_000


def _categorical_made(posterior):
    """The CategoricalMADE, via whichever attribute path the posterior exposes."""
    estimator = getattr(posterior, "posterior_estimator", None)
    if estimator is None:
        estimator = posterior.potential_fn.posterior_estimator
    return estimator.discrete_net.net


def prefix_state_count(posterior) -> int:
    """Number of autoregressive prefixes an exact marginalization would enumerate."""
    made = _categorical_made(posterior)
    counts = [int(c) for c in made.num_categories_per_var.detach().cpu()]
    states = 1
    for c in counts[:-1]:
        states *= c
    return states


def num_discrete_variables(posterior) -> int:
    return int(_categorical_made(posterior).num_variables)


def auto_marginal_batch(
    posterior,
    max_exact_states: int,
    num_marginal_samples: int,
    max_elements: int,
    requested_batch: int,
) -> int:
    """Largest observation batch keeping marginalization under ``max_elements``.

    The marginalizer expands each observation into either ``prefix_states``
    (exact) or ``num_marginal_samples`` (Monte Carlo) rows, and every such row
    is ``num_variables`` wide — the particle tensor, the repeated condition,
    and the reshaped logits all carry that factor. Peak allocation therefore
    scales with ``states * batch * num_variables``, and bounding that product
    is what keeps memory roughly flat in D and makes concurrent grid points on
    one machine safe.

    CAVEAT. On the exact path the batch is purely a memory
    knob: D=12 returns bit-identical ECE at batches 200/64/17. On the Monte
    Carlo path it is not, because
    :func:`mnpe_paper.calibration._marginal_discrete_probs` seeds each chunk with
    ``seed + start``, so chunk boundaries select a different random stream. Measured drift at D=32 was 7.8e-4 across those batches — about 15%
    of that estimator's own noise (5.1e-3), i.e. re-randomization rather than
    bias, and deterministic for fixed flags since this function is pure. Two
    consequences: keep ``marginal_max_elements`` fixed across a grid you intend
    to compare, and treat the recorded ``ece_marginal_batch`` as provenance
    required to reproduce a Monte Carlo row exactly. ``merge_row_dir``
    enforces the first of these rather than trusting the operator.
    """
    states = prefix_state_count(posterior)
    if states > max_exact_states:
        # Mixed marginalization peaks at whichever phase is wider: the exact
        # enumeration is bounded by the per-variable cap, the sampled tail by
        # the particle count.
        states = max(max_exact_states, num_marginal_samples)
    per_observation = max(1, states * num_discrete_variables(posterior))
    return max(1, min(requested_batch, max_elements // per_observation))


def dual_discrete_ece(
    params: torch.Tensor,
    xs: torch.Tensor,
    posterior,
    discrete_indices: List[int],
    num_bins: int = 10,
    max_exact_states: int = DEFAULT_MAX_EXACT_STATES,
    num_marginal_samples: int = DEFAULT_MARGINAL_SAMPLES,
    marginal_batch: int = 128,
    marginal_max_elements: int = DEFAULT_MARGINAL_MAX_ELEMENTS,
    noise_check: bool = False,
) -> Dict[str, float]:
    """Top-label ECE under both probability definitions, plus provenance.

    ``ece_marginal_*`` is the quantity the paper defines; ``ece_factor_*`` is
    the teacher-forced diagnostic, which the result files with a single
    ``ece_mean`` column carry. Both are emitted so the grids remain comparable.

    When marginalization falls back to Monte Carlo the estimate carries error of
    order ``0.5 / sqrt(num_marginal_samples)`` per probability, which at the
    default 4096 samples is ~0.008 against ECE values around 0.02-0.03. Setting
    ``noise_check`` re-estimates under a second RNG seed and records the largest
    per-dimension discrepancy, so that noise can be compared against the
    marginal-versus-factor gap rather than assumed negligible.
    """
    exact = prefix_state_count(posterior) <= max_exact_states
    made = _categorical_made(posterior)
    exact_dims = exact_marginal_variable_count(
        [int(c) for c in made.num_categories_per_var.detach().cpu()], max_exact_states
    )
    batch = auto_marginal_batch(
        posterior,
        max_exact_states,
        num_marginal_samples,
        marginal_max_elements,
        marginal_batch,
    )

    t0 = time.perf_counter()
    ece_marginal, _, _ = run_discrete_calibration(
        params,
        xs,
        posterior,
        discrete_indices=discrete_indices,
        num_bins=num_bins,
        probability_type="marginal",
        max_exact_states=max_exact_states,
        num_marginal_samples=num_marginal_samples,
        marginal_batch_size=batch,
    )
    marginal_time = time.perf_counter() - t0

    ece_factor, _, _ = run_discrete_calibration(
        params,
        xs,
        posterior,
        discrete_indices=discrete_indices,
        num_bins=num_bins,
        probability_type="factor",
    )

    out = {
        "ece_marginal_mean": float(ece_marginal.mean()),
        "ece_marginal_max_dim": float(ece_marginal.max()),
        "ece_factor_mean": float(ece_factor.mean()),
        "ece_factor_max_dim": float(ece_factor.max()),
        "ece_marginal_exact": bool(exact),
        "ece_marginal_exact_dims": exact_dims,
        "ece_marginal_batch": batch,
        "ece_marginal_samples": None if exact else num_marginal_samples,
        "ece_marginal_time_s": round(marginal_time, 1),
        "ece_marginal_mc_spread": float("nan"),
    }

    # A repeat is only informative for the sampled estimator; exact
    # marginalization is deterministic and would trivially return zero.
    if noise_check and not exact:
        ece_repeat, _, _ = run_discrete_calibration(
            params,
            xs,
            posterior,
            discrete_indices=discrete_indices,
            num_bins=num_bins,
            probability_type="marginal",
            max_exact_states=max_exact_states,
            num_marginal_samples=num_marginal_samples,
            marginal_batch_size=batch,
            marginal_seed=12345,
        )
        out["ece_marginal_mc_spread"] = float((ece_repeat - ece_marginal).abs().max())
    return out


_RUNS_THIS_PROCESS = 0
_RSS_BASELINE_MB = None


def begin_run() -> bool:
    """Mark the start of a run; True if it is the first in this process.

    ``ru_maxrss`` is a high-water mark that never decreases, so a second run in
    the same process inherits the first one's peak and its memory columns are
    meaningless. Recording the flag lets analysis filter rather than trust.

    Also latches the pre-training baseline, because the interpreter plus torch
    alone is several hundred MB — far more than the networks at low D. Without
    the baseline the absolute peak is nearly constant in D and the scaling
    behaviour being claimed is invisible.
    """
    global _RUNS_THIS_PROCESS, _RSS_BASELINE_MB
    _RUNS_THIS_PROCESS += 1
    if _RSS_BASELINE_MB is None:
        _RSS_BASELINE_MB = rss_peak_mb()
    return _RUNS_THIS_PROCESS == 1


def rss_baseline_mb() -> float:
    """Peak RSS latched at the first ``begin_run`` — interpreter plus imports."""
    return _RSS_BASELINE_MB if _RSS_BASELINE_MB is not None else rss_peak_mb()


def machine_speed_probe(
    size: int = 1024, repeats: int = 50, chunks: int = 3, seed: int = 0
) -> float:
    """Wall-clock of a fixed workload — a per-run reference for machine speed.

    Serial execution is what makes ``train_time_s`` usable for a runtime-vs-D
    claim, but a grid walked in D order over many hours confounds D with
    whatever the machine is doing: thermal throttling, a noisy neighbour, a
    frequency governor. This probe is identical in every run regardless of D,
    budget or seed, so ``train_time_s / probe_time_s`` is a throttle-corrected
    relative timing, and the probe series alone shows whether the machine
    drifted at all.

    Deliberately BLAS-bound and roughly half a second: the same resource
    training competes for, at negligible cost. Returns the FASTEST of
    ``chunks`` timed blocks rather than the total — a minimum is the standard
    robust estimator for machine speed, since interference can only ever slow a
    block down. A single short block is useless here: at 512x512x20 the probe
    took 8 ms and varied 75% between consecutive runs, which would have added
    more noise to the normalized timing than it removed.
    """
    generator = torch.Generator().manual_seed(seed)
    a = torch.randn(size, size, generator=generator)
    b = torch.randn(size, size, generator=generator)
    for _ in range(3):  # warm the allocator and BLAS threadpool before timing
        _ = a @ b
    best = float("inf")
    for _ in range(chunks):
        start = time.perf_counter()
        for _ in range(repeats):
            _ = a @ b
        best = min(best, time.perf_counter() - start)
    return best


def rss_peak_mb() -> float:
    """Peak resident set size of this process, in MB.

    ``ru_maxrss`` is bytes on macOS/BSD and kibibytes on Linux — a silent
    1000x discrepancy if the grid is split across platforms.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1e6 if sys.platform == "darwin" else raw / 1e3


def parameter_counts(posterior) -> Dict[str, int]:
    """Trainable parameters, split by head.

    Exact, deterministic, and machine-independent — the structural half of
    "memory usage with increasing dimensionality", where peak RSS is the
    empirical half and is sensitive to allocator behaviour and co-tenancy.
    """
    estimator = getattr(posterior, "posterior_estimator", None)
    if estimator is None:
        estimator = posterior.potential_fn.posterior_estimator
    count = lambda net: sum(p.numel() for p in net.parameters())  # noqa: E731
    return {
        "n_params_total": count(estimator),
        "n_params_discrete": count(estimator.discrete_net),
        "n_params_continuous": count(estimator.continuous_net),
    }


def tensor_mb(*tensors) -> float:
    """Footprint of the training tensors in MB.

    Worth recording separately: at a 1M budget the training set is
    ``budget * 2D * 4`` bytes — about 1 GB at D=128 — which dominates the
    network's own footprint by orders of magnitude. Any honest statement about
    MNPE's memory scaling has to say which term is doing the scaling.
    """
    return sum(t.numel() * t.element_size() for t in tensors) / 1e6


def save_posterior(posterior, path: Path) -> None:
    """Pickle a posterior, matching ``experiments/queue/main/train_queue_1M.py``.

    Needed by ``recompute_marginal_ece.py``, which re-derives the calibration
    metric of a finished run without retraining it.
    """
    import pickle

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(posterior, f)


def write_row(row_dir: Path, key: str, row: dict) -> Path:
    """Write one run's result as its own CSV, so parallel writers never collide."""
    row_dir.mkdir(parents=True, exist_ok=True)
    path = row_dir / f"{key}.csv"
    tmp = path.with_suffix(".csv.tmp")
    pd.DataFrame([row]).to_csv(tmp, index=False)
    tmp.replace(path)  # atomic: a reader never sees a half-written row
    return path


def row_exists(row_dir: Path, key: str) -> bool:
    return (row_dir / f"{key}.csv").exists()


def append_row_checked(out_file: Path, row: dict) -> None:
    """Append to a single CSV, refusing to mix schemas.

    Appending to a file whose header does not match produces a ragged table, and
    every ragged table here means the same thing: two runs of different code
    merged into one grid, with a column that is present for some rows and not
    others. That is precisely the failure this module exists to undo, so it
    fails loudly instead.

    **None of the tracked tables under ``results/scaling/`` can be appended to,
    and that is expected.** The three width and convergence probes carry a
    single ``ece_mean`` rather than separate marginal and factor columns. The two
    grids each differ from what this code emits for their own reason:
    ``k_sweep_results.csv`` lacks ``ece_marginal_exact_dims``, and
    ``scaling_study_results.csv`` carries the two ``ece_recompute_*`` columns that
    only ``recompute_marginal_ece.py`` writes. Extend a grid with ``--row-dir``,
    which merges rather than appends, or write somewhere new with ``--out``.
    """
    if out_file.exists():
        existing = list(pd.read_csv(out_file, nrows=0).columns)
        incoming = list(row.keys())
        if existing != incoming:
            added = [c for c in incoming if c not in existing]
            dropped = [c for c in existing if c not in incoming]
            if added or dropped:
                detail = f"this run adds {added} and lacks {dropped}"
            else:
                # Same columns in a different order. Worth naming separately: the
                # set-based report above would print two empty lists and read as a
                # message about nothing.
                detail = "the same columns in a different order"
            raise SystemExit(
                f"refusing to append to {out_file.name}: schema mismatch — {detail}. "
                "Appending would leave a table whose columns mean different things "
                "in different rows. Use --row-dir to extend a grid, or --out to "
                "write elsewhere."
            )
    pd.DataFrame([row]).to_csv(out_file, mode="a", header=not out_file.exists(), index=False)


def merge_row_dir(
    row_dir: Path,
    out_file: Path,
    sort_by: Optional[List[str]] = None,
    prefix: str = "",
    key_col: str = "",
) -> int:
    """Concatenate per-run row files into one CSV. Returns the row count.

    ``prefix`` restricts the glob to one sweep's keys ("D" or "K"), so pointing
    both sweeps at a shared row directory cannot silently splice a K-sweep row
    into a D-sweep grid. The glob is ``{prefix}*.csv``, which also excludes the
    ``.csv.tmp`` staging files ``write_row`` renames from.

    ``pd.concat`` unions columns and fills the gaps with NaN, so rows written by
    two revisions of this code merge without complaint — which is how the tracked
    ``scaling_study_results.csv`` came to have ``ece_marginal_exact_dims`` on only
    29 of its 90 rows. Harmless there, since nothing reads that column for the
    rows that lack it, but it is the same class of silent mixing that
    ``append_row_checked`` refuses outright, so a ragged merge is at least named.

    The merge replaces ``out_file``, so it refuses when ``out_file`` holds a run
    (identified by the ``sort_by`` columns) that the row directory lacks: merging
    a partial row directory into a tracked grid would otherwise shrink the grid.
    """
    paths = sorted(row_dir.glob(f"{prefix}*.csv"))
    if not paths:
        raise SystemExit(f"no '{prefix}*.csv' row files in {row_dir}")
    frames = [pd.read_csv(p) for p in paths]
    _warn_on_ragged_rows(frames, paths)
    df = pd.concat(frames, ignore_index=True)
    missing = [c for c in (sort_by or []) if c not in df.columns]
    if missing:
        raise SystemExit(
            f"merged frame lacks {missing} — {row_dir} likely mixes sweeps; "
            "write each sweep to its own row directory"
        )
    if key_col:
        _assert_comparable_mc_rows(df, key_col)
    if sort_by and out_file.exists():
        existing = pd.read_csv(out_file)
        runs = [c for c in sort_by if c in existing.columns]
        lost = set(existing[runs].itertuples(index=False)) - set(df[runs].itertuples(index=False))
        if lost:
            example = ", ".join(f"{c}={v}" for c, v in zip(runs, sorted(lost)[0]))
            raise SystemExit(
                f"refusing to merge: {out_file.name} has {len(lost)} run(s) that {row_dir} "
                f"lacks (e.g. {example}), and merging would drop them. Add those rows to "
                "the row directory, or merge into a new --out."
            )
    if sort_by:
        df = df.sort_values(sort_by).reset_index(drop=True)
    df.to_csv(out_file, index=False)
    return len(df)


def _warn_on_ragged_rows(frames: List[pd.DataFrame], paths: List[Path]) -> None:
    """Report per-run rows that do not all carry the same columns.

    A warning rather than a refusal: a grid assembled over weeks legitimately
    spans a code change, and the alternative — discarding finished rows over a
    column nothing reads — is worse than saying so.
    """
    columns = [set(frame.columns) for frame in frames]
    shared = set.intersection(*columns)
    union = set.union(*columns)
    if shared == union:
        return
    uneven = sorted(union - shared)
    print(f"WARNING  {len(paths)} row files do not share a schema: {uneven}")
    print("         missing entries merge as empty cells. This means the rows were")
    print("         written by different revisions of this code.")


def _assert_comparable_mc_rows(df: pd.DataFrame, key_col: str) -> None:
    """Refuse to merge Monte Carlo rows computed under different chunkings.

    On the sampled path the chunk boundary selects the RNG stream (see
    ``auto_marginal_batch``), so two rows at the same D/K but different
    ``ece_marginal_batch`` are not strictly comparable. Different keys are
    expected to differ — the batch is chosen per problem size — so the
    invariant is "constant within a key", not "constant overall". Silently
    splicing such rows is exactly the class of error this rerun exists to undo.
    """
    needed = {"ece_marginal_exact", "ece_marginal_batch", key_col}
    if not needed.issubset(df.columns):
        return
    mc = df[~df["ece_marginal_exact"].astype(bool)]
    if mc.empty:
        return
    offenders = {
        key: sorted(set(group))
        for key, group in mc.groupby(key_col)["ece_marginal_batch"]
        if len(set(group)) > 1
    }
    if offenders:
        detail = "; ".join(f"{key_col}={k}: batches {v}" for k, v in offenders.items())
        raise SystemExit(
            "refusing to merge: Monte Carlo rows at the same "
            f"{key_col} used different marginalization batches ({detail}). "
            "These were computed under different RNG chunkings and are not "
            "comparable. Re-run the odd ones out with a single "
            "--marginal-max-elements, or drop them."
        )


def check_seed_completeness(
    frame: pd.DataFrame,
    key_col: str,
    label: str = "results",
    expected_seeds: int = 3,
    strict: bool = False,
) -> List[str]:
    """Report ``(key, budget)`` cells holding fewer than ``expected_seeds`` seeds.

    A min-max band over two seeds is visually indistinguishable from one over
    three, so a partially-filled grid renders as a confident multi-seed result.
    The figure scripts call this so gaps are named rather than assumed absent.
    Returns the human-readable descriptions of any short cells.

    An empty frame is checked first, because it is the extreme case of the very
    thing this guards and it passes every test below vacuously: there are no
    short cells when there are no cells. Left alone it printed "complete, 3 seeds
    in every cell" and the caller then died indexing an empty budget list.
    """
    if frame.empty:
        raise SystemExit(
            f"{label}: no rows to plot. Either the file is empty or a filter "
            "removed everything — the width filter in k_sweep_figure.py does "
            "exactly that when pointed at one of the capacity probes."
        )
    counts = frame.groupby([key_col, "budget"])["seed"].nunique()
    # A cell with no rows at all is absent from the groupby, so count every
    # combination of the keys and budgets present, with absent cells at zero.
    cells = pd.MultiIndex.from_product(
        [sorted(frame[key_col].unique()), sorted(frame["budget"].unique())]
    )
    counts = counts.reindex(cells, fill_value=0)
    short = counts[counts < expected_seeds]
    if short.empty:
        print(f"{label}: complete, {expected_seeds} seeds in every cell")
        return []
    lines = [f"{key_col}={k}, budget={b}: {n} seed(s)" for (k, b), n in short.items()]
    message = f"{label}: INCOMPLETE — " + "; ".join(lines)
    if strict:
        raise SystemExit(message)
    print(f"WARNING  {message}")
    print("         bands over these cells span fewer seeds than the rest.")
    return lines


def add_marginal_ece_args(parser) -> None:
    """CLI flags shared by both sweeps."""
    parser.add_argument(
        "--row-dir",
        type=str,
        help="write one CSV per run into this directory (required for parallel runs)",
    )
    parser.add_argument(
        "--posterior-dir", type=str, help="pickle each trained posterior into this directory"
    )
    parser.add_argument(
        "--marginal-samples",
        type=int,
        default=DEFAULT_MARGINAL_SAMPLES,
        help="ancestral samples for Monte Carlo marginalization (ignored when exact)",
    )
    parser.add_argument(
        "--marginal-batch", type=int, default=128, help="observations per marginalization chunk"
    )
    parser.add_argument(
        "--marginal-max-elements",
        type=int,
        default=DEFAULT_MARGINAL_MAX_ELEMENTS,
        help=(
            "cap on states*batch*num_variables elements per MADE call; lowers peak memory "
            "under parallelism. Keep identical across a grid: on the Monte Carlo path it "
            "changes the RNG chunking and so perturbs the result"
        ),
    )
    parser.add_argument(
        "--max-exact-states",
        type=int,
        default=DEFAULT_MAX_EXACT_STATES,
        help="enumerate prefixes exactly up to this many states, else Monte Carlo",
    )
    parser.add_argument(
        "--ece-noise-check",
        action="store_true",
        help="re-estimate Monte Carlo marginals under a second seed and record the spread",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=0.0,
        help=(
            "idle before each run so the machine cools between grid points. Only "
            "applies within one process; with one process per point, sleep in the "
            "shell loop instead"
        ),
    )
