# Scaling in the size of the discrete space

How MNPE behaves as the discrete space grows. The model is `D` independent copies of
the tractable toy model, so it scales without changing per-dimension difficulty:

```
theta_i ~ N(0, 1),  z_i ~ Categorical(K),  x_i | theta_i, z_i ~ N(theta_i + A·z_i, sigma^2)
```

Because it factorizes, the exact posterior stays in closed form at every size —
`p(z_i = k | x) = softmax_k(-(x_i - A·k)^2 / 2·tau)` per dimension — so this is
scored against a known answer rather than against another sampler, at every point
of both sweeps. That is the whole reason for building the scaling probe on this
model rather than on the queueing one.

Two sweeps separate the two ways a discrete space can grow:

| Sweep | Varies | Fixed | Question |
|-------|--------|-------|----------|
| `scaling_study.py` | `D` = 2 … 128 binary dimensions | `K` = 2 | more dimensions |
| `k_sweep.py` | `K` = 5 … 50 classes | `D` = 2 | more classes per dimension |

Neither enumerates the discrete space, which is the point: MNPE's cost is
independent of the number of configurations, and from `D >= 17` most of the 2^D
configurations are never seen during training at all.

## Scripts

Run from the repository root.

| Script | Produces | Paper |
|--------|----------|-------|
| `scaling_study.py` | `results/scaling/scaling_study_results.csv` | Appendix B.1, Figure B-1 |
| `k_sweep.py` | `results/scaling/k_sweep_results.csv` | Appendix B.1, Figure B-2 |
| `recompute_marginal_ece.py` | the `ece_*` columns of the `D >= 32` rows above | Appendix B.1, Figure B-1b |
| `figures/scaling_figure.py` | `figures/scaling_figure.pdf` | Figure B-1 |
| `figures/k_sweep_figure.py` | `figures/k_sweep_figure.pdf` | Figure B-2 |

Both sweeps are long: roughly eleven hours each for the full grid on eight CPU
cores. Both take `--smoke` for a one-point pipeline check that takes about a
minute, mostly start-up, and writes to a git-ignored `*_SMOKE.csv`.

```bash
uv run python experiments/scaling/scaling_study.py --smoke   # checks the pipeline
uv run python figures/scaling_figure.py                      # rebuilds from the tracked CSV
```

## The commands the tracked grids came from

Each sweep's default grid is its two smaller budgets; the published third tier is
the same grid run again at `--budgets 1000000`. `--row-dir` gives every grid point
its own one-row CSV written through an atomic rename, which is what makes the runs
safe to parallelize and resumable — rerun the same command and completed points
are skipped. Nothing is merged into the result table until `--merge`.

```bash
# D-sweep: 10 values of D x 3 budgets x 3 seeds = 90 rows
uv run python experiments/scaling/scaling_study.py --row-dir rows_scaling
uv run python experiments/scaling/scaling_study.py --row-dir rows_scaling --budgets 1000000
uv run python experiments/scaling/scaling_study.py --row-dir rows_scaling --merge

# K-sweep: 4 values of K x 3 budgets x 3 seeds = 36 rows
uv run python experiments/scaling/k_sweep.py --row-dir rows_k
uv run python experiments/scaling/k_sweep.py --row-dir rows_k --budgets 1000000
uv run python experiments/scaling/k_sweep.py --row-dir rows_k --merge
```

Relative `--row-dir` and `--posterior-dir` paths resolve under `artifacts/scaling/`,
which is git-ignored, so per-run rows and pickled posteriors never land in the
tracked tree. Give an absolute path to put them elsewhere.

**Use `--row-dir`, not the default append mode, to extend either grid.** Without
it a sweep appends to the tracked CSV directly, and it will refuse: neither
tracked grid has the exact column list this code emits — `k_sweep_results.csv`
lacks `ece_marginal_exact_dims`, and `scaling_study.py` never writes the two
`ece_recompute_*` columns that `scaling_study_results.csv` carries. That refusal
is deliberate. Appending would leave a table whose columns mean different things
in different rows, which is the failure the whole marginal/factor split exists to
prevent. `--merge` unions instead, and says so when the rows disagree. It replaces
the result table, so it refuses when the table holds a run the row directory
lacks; to merge a partial row directory, such as the 27 rows below on their own,
pass `--out`.

Give each sweep its own row directory. If you do share one, `--merge` still
separates them by key prefix (`D*.csv` against `K*.csv`) and refuses a merge whose
sort columns are missing, so a K-sweep row cannot be spliced silently into a
D-sweep grid.

### The `D >= 32` calibration column was recomputed, not swept

27 of the 90 D-sweep rows — every seed at `D` in {32, 64, 128} — carry an
`ece_recompute_device` value. Their calibration column was re-derived from the
saved posterior by `recompute_marginal_ece.py` at a much higher particle count
(65,536, and 262,144 at `D = 32`) on a GPU, rather than at the sweep's default
4,096 on CPU. Reproducing those rows therefore takes two steps:

```bash
# Train all 27, pickling each posterior. Two passes, because the 10^6 tier is
# not in the default budget list.
uv run python experiments/scaling/scaling_study.py \
    --row-dir rows_scaling --posterior-dir posteriors_scaling --dims 32 64 128
uv run python experiments/scaling/scaling_study.py \
    --row-dir rows_scaling --posterior-dir posteriors_scaling --dims 32 64 128 \
    --budgets 1000000

# D=32 was recomputed at four times the particle count of D=64 and D=128, so
# the two are separate commands rather than one over all three dimensions.
uv run python experiments/scaling/recompute_marginal_ece.py \
    --row-dir rows_scaling --posterior-dir posteriors_scaling \
    --dims 32 --marginal-samples 262144
uv run python experiments/scaling/recompute_marginal_ece.py \
    --row-dir rows_scaling --posterior-dir posteriors_scaling \
    --dims 64 128 --marginal-samples 65536

uv run python experiments/scaling/scaling_study.py --row-dir rows_scaling --merge
```

Every row's own `ece_marginal_samples` records which count it was computed at, so
the split above is checkable against the tracked table rather than taken on trust.

This is possible because the evaluation set is a deterministic function of `D` at
a fixed seed, so the metric can be re-estimated without retraining;
`tests/test_scaling_evaluation_set.py` pins that determinism. Only the `ece_*`
columns are rewritten, so timings, memory and parameter counts stay the ones the
training run measured.

## Two ECE definitions, and why both are recorded

`calibration.run_discrete_calibration` can score either the genuine marginals
`q(z_i | x)` or the teacher-forced autoregressive factors `q(z_i | z_<i true, x)`.
The two coincide only for the first discrete variable, so a single `ece_*` column
silently means different things at different `D`. Appendix A.1 defines the ECE as
the marginal and explicitly rules out evaluating the factors at the true preceding
values, so both are written, under names that cannot be confused:

| Column | Quantity |
|--------|----------|
| `ece_marginal_mean`, `ece_marginal_max_dim` | `q(z_i \| x)` — what the paper plots and defines |
| `ece_factor_mean`, `ece_factor_max_dim` | `q(z_i \| z_<i true, x)` — the teacher-forced diagnostic |
| `ece_marginal_exact` | whether the row's marginals were enumerated or sampled |
| `ece_marginal_exact_dims` | how many leading variables were enumerated exactly |
| `ece_marginal_samples`, `ece_marginal_batch` | provenance needed to reproduce a sampled row |

One column is unevenly populated: `ece_marginal_exact_dims` is blank in 61 of the
D-sweep rows and absent from the K-sweep file entirely. The code fills it in every
row it writes, and it is read only for the sampled rows — which all carry it — so
a rerun will simply populate it everywhere. Nothing downstream depends on the
blanks.

Marginalization is exact for as many leading variables as a 32,768-state budget
buys — 16 of them, at any `D` — and Rao-Blackwellized ancestral Monte Carlo for
the tail. So the exact share *shrinks* as `D` grows: half the dimensions at
`D = 32`, an eighth at `D = 128`.

On the sampled path the marginalization batch selects the RNG stream, so two rows
at the same `D` computed under different `--marginal-max-elements` are not
strictly comparable. `--merge` refuses such a mix rather than trusting the
operator.

In the event the two definitions differ very little here: across all 90 D-sweep
rows the mean absolute difference is 0.0007, and the marginal is the larger of the
two in only 38 of them.

## The paper's Appendix B.1 used an earlier grid

Figures B-1 and B-2 and the numbers Appendix B.1 quotes were drawn from an
earlier pair of grids, which scored the ECE on the teacher-forced factor and ran
the 10^6 tier with a single seed. The tracked grids are the rerun with the
marginal ECE and three seeds at every budget, so the quoted values move: at
`K = 50` and 10^5 simulations, per-dimension total variation is 0.261 in the
paper and 0.250 here, and joint C2ST 0.68 against 0.653.

## Results

| File | What it is |
|------|------------|
| `scaling_study_results.csv` | The D-sweep grid, 90 rows. Figure B-1 |
| `k_sweep_results.csv` | The K-sweep grid, 36 rows. Figure B-2 |
| `k_sweep_capacity_w128.csv` | Width probe at `K` in {25, 50}, 10^5 sims, width 128 |
| `k_sweep_capacity_w256.csv` | The same at width 256, one seed per cell |

The two capacity files are the evidence for the paper's claim that quadrupling the
width at fixed budget roughly halves the error: at `K = 50` and 10^5 simulations,
per-dimension total variation is 0.25 at the sweep's width of 32 and 0.11 at width
128. They are deliberately *not* part of Figure B-2, whose caption states that the
width is held fixed — `k_sweep_figure.py` filters on `hidden_features == 32` so
that stays true.

**Read the `ece_mean` column of the two capacity files with care.** Both carry a
single `ece_mean` / `ece_max_dim` column, which holds the teacher-forced factor
rather than the marginal. Their total variation and C2ST columns are unaffected,
and those are the columns the paper quotes from them. `append_row_checked` will
refuse to add a row to any of them, by design, rather than produce a file whose
header means two things.

## How closely a rerun should agree

One cell was rerun here to calibrate expectations: `D = 2`, 10^4 simulations,
seed 0, on a different machine from the one the grid was trained on.

The network is *structurally identical* — 79,669 parameters, split 4,838 discrete
and 74,831 continuous, matching the tracked row exactly — as is the evaluation
protocol (`ece_baseline`, and the exact-versus-sampled marginalization route).
Those columns are deterministic given the architecture, so a mismatch there would
mean the code no longer runs the same experiment.

The trained network is not the same object, because thread counts and BLAS
reduction order are not pinned, so the metrics agree rather than match:
per-dimension TV within 6e-5, marginal ECE within 4e-6, joint C2ST within 4e-4.
The SBC p-value moved from 0.51 to 0.44, which is what a p-value does — it is a
draw from a uniform under the null, not an estimate converging to a number, and
neither value is anywhere near rejecting.

## Where output goes

Result tables go to `results/scaling/` and are tracked, because the paper quotes
numbers from them and the figures rebuild from them alone. Per-run rows, pickled
posteriors and anything else a rerun produces go to `artifacts/scaling/` and are
not — the posteriors in particular run to gigabytes across a full grid and are
never needed to draw a figure.

## Reproducing the figures without rerunning anything

Both figure scripts read only the tracked CSVs, so they work in a fresh clone:

```bash
uv run python figures/scaling_figure.py
uv run python figures/k_sweep_figure.py
```

Pass `--strict` to fail rather than warn when a `(D, budget)` or `(K, budget)`
cell holds fewer than three seeds. A min–max band over two seeds is visually
indistinguishable from one over three, so a partly-filled grid would otherwise
render as a confident multi-seed result.
