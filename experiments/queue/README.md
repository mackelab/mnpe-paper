# Tandem M/M/c queueing network

The paper's main experiment: three continuous rates and two discrete server
counts, scored against an exact reference rather than against another
approximation.

```
Arrivals(gamma) -> [M/M/c1, mu1] -> [M/M/c2, mu2] -> Departure
```

| Parameter | Prior | Meaning |
|-----------|-------|---------|
| `gamma` | LogNormal(log 9, 0.3) | Arrival rate |
| `mu1` | LogNormal(log 8, 0.3) | Service rate, station 1 |
| `mu2` | LogNormal(log 5, 0.3) | Service rate, station 2 |
| `c1` | Uniform{2,3,4,5,6} | Servers at station 1 |
| `c2` | Uniform{2,3,4,5,6} | Servers at station 2 |

Observations are three Poisson counts and two queue lengths, with `T = 100` and
`sigma_obs = 0.1`. The prior is the independent product prior *conditioned on
both stations being stable*, `rho_i < 0.999` — about 13% of prior draws are
rejected. That conditioning is not a detail: see "Why the reference is built this
way" below.

The model itself is `mnpe_paper.models.queue`: `spec` holds the constants and the
tuned architecture, `config` the priors, simulator and PyMC models, `reference`
the quadrature reference, `targets` the independent JAX reimplementation, `eval`
the shared metric suite, `metrics` the derived metrics, `conditional` the
conditional sampler, and `nets` the masked estimator classes.

## Layout

Scripts are grouped by what they are *for*, since there are twenty-seven of them:

| Directory | Role |
|-----------|------|
| `main/` | The convergence pipeline that produces the headline result |
| `references/` | Building the reference posterior, and checking it against independent methods |
| `ablations/` | Method variants — what changes if MNPE is built differently |
| `diagnostics/` | Whether the fitted posterior is right in ways the headline metric cannot see |

Run everything from the repository root.

## What each script produces

Two scripts produce no number that appears in the paper. They are marked
**validation only** and ship anyway, because they are the evidence that the
reference's evidences and the study's noise floor were independently checked —
an unchecked reference is exactly the failure this experiment is built to avoid.

### `main/`

| Script | Produces | Paper |
|--------|----------|-------|
| `budget_study.py` | `artifacts/queue/budget_study_results.pkl`, `artifacts/queue/posteriors/` | The queueing figure; Section 4.2 convergence claim |
| `reeval_budget_study.py` | `artifacts/queue/budget_study_results_n1000.pkl` | Rescores the saved posteriors on the 1,000-case test set — **this is the file the figure reads** |
| `mnpe_hparam_tuning.py` | `artifacts/queue/hparam_study.db`, `hparam_best_params.json` | The architecture reported in Appendix D.2, recorded as `spec.TUNED_HPARAMS` |
| `train_queue_1M.py` | `artifacts/queue/posterior_queue_1M_rep{k}.pkl` | The 1M-budget result quoted in the text; the high-budget posterior Appendix A.3 tests |
| `eval_queue_1M.py` | `artifacts/queue/queue_1M_eval.pkl` | Same metric suite at 1M, so the 1M result quoted in the text is comparable with the budget study; the figure does not draw it |
| `runtime_comparison.py` | `results/queue/runtime_comparison.csv` | Section 4.2 and Table D-1 (Appendix D.2), except its per-configuration NUTS row |

### `references/`

| Script | Produces | Paper |
|--------|----------|-------|
| `deterministic_reference_run.py` | `artifacts/queue/deterministic_reference.npz` | Section 4.2 and Appendix D.2 — **the reference every metric is scored against** |
| `smc_reference.py` | `artifacts/queue/smc_reference.npz` | Appendix D.2: the independent MCMC route, agreeing with quadrature at C2ST 0.515 and discrete TV 0.004 |
| `reference_yardstick.py` | `results/queue/reference_yardstick_smc.csv`, `..._quadrature.csv` | Appendix D.2; supplies the reference gap the figure prints |
| `pymc_automarginal.py` | `artifacts/queue/pymc_marginal_reference*.npz` | Section 4.2 and Appendix D.2 — evidence for the claim that NUTS does not converge on the marginalized target |
| `numpyro_marginal_reference.py` | `artifacts/queue/numpyro_marginal_reference.npz` | The same claim, in an independent framework |

### `ablations/`

| Script | Produces | Paper |
|--------|----------|-------|
| `ablation_joint_vs_separate.py` | `results/queue/ablation_joint_vs_separate.csv` | Appendix B.2, plus a sentence in Section 3.1 |
| `dequantization_baseline.py` | `results/queue/dequantization_baseline_results.csv` | Appendix B.3 and Table B-1; the open circle in the figure's panel (b) |
| `reverse_factorization_queue.py` | `results/queue/reverse_factorization_queue.csv` | Appendix B.4 and Table B-2 |
| `ablation_discrete_ordering.py` | `results/queue/ablation_discrete_ordering.csv`, `..._paired.csv` | Appendix B.6 and Table B-3 |

### `diagnostics/`

| Script | Produces | Paper |
|--------|----------|-------|
| `posterior_metrics.py` | `results/queue/posterior_metrics.csv`, `c2st_floor.csv` | Section 4.2 and Appendix A.1; `c2st_floor.csv` is the level the figure's panel (b) draws |
| `conditional_diagnostics.py` | `results/queue/conditional_diagnostics_d1.csv`, `..._d2_ks.csv` | Appendix A.3 |
| `conditional_sbc_budget_trend.py` | `results/queue/conditional_sbc_budget_trend.csv`, `..._summary.csv` | Appendix A.3 — the deviation against budget |
| `ece_observed_vs_baseline_vary_b.py` | `results/queue/ece_observed_vs_baseline_vary_b.csv` | Appendix A.1 closing paragraph and Figure A-1 |
| `ece_samples_per_bin.py` | `results/queue/ece_observed_vs_baseline_vary_n.csv` | Appendix A.1 — the noise floor against samples per bin |
| `compare_ece_baselines.py` | `results/queue/ece_baseline_vary_{n,k,b}.csv`, `ece_baseline_across_budgets.csv` | Appendix A.1 — backs the sentence that equal-mass bins give systematically higher baselines |

The paper figure is built by `figures/make_queue_figure.py`, whose docstring
records the four deliberate choices behind it.

### One table has no producer

| Table | Status |
|-------|--------|
| `results/queue/reference_yardstick.csv` | **Kept for one number.** No script in this repository writes it — `reference_yardstick.py` writes `reference_yardstick_{smc,quadrature}.csv`, and its schema differs by an extra `n_samples` column. Its `per-configuration PyMC NUTS reference` and `per-configuration NumPyro NUTS reference` rows are the per-configuration NUTS references at C2ST 0.52–0.53 that Appendix D.2 quotes, and nothing else here carries them. **It is not the yardstick**; use `reference_yardstick_smc.csv`. |

The reason those rows have no producer is the subject of the last paragraph of
the next section.

`reference_yardstick_quadrature.csv` carries the same two rows and the same
`n_samples` column, so rerunning `reference_yardstick.py --target quadrature`
drops them from that file; the committed version is the only copy.

## Order of execution

Almost everything downstream needs the reference and the budget study, in that
order.

```bash
uv run python experiments/queue/references/deterministic_reference_run.py   # first: the reference
uv run python experiments/queue/main/budget_study.py                        # then: the posteriors
uv run python experiments/queue/main/reeval_budget_study.py                 # -> the file the figure reads
uv run python experiments/queue/diagnostics/posterior_metrics.py            # -> c2st_floor.csv
uv run python experiments/queue/references/smc_reference.py                 # the independent reference
uv run python experiments/queue/references/reference_yardstick.py --target smc   # -> the reference gap
uv run python figures/make_queue_figure.py
```

`reference_yardstick.py` scores the saved posteriors, so it runs after the budget
study and stops without writing if there are none.

`mnpe_hparam_tuning.py` is optional: its result is already recorded as
`spec.TUNED_HPARAMS`, so rerunning it re-derives the architecture rather than
changing it. `train_queue_1M.py` and `eval_queue_1M.py` are the expensive
extension. Several scripts take `--smoke` for a fast, smaller run. It shrinks the
training, not the prerequisites: the ablation smokes still need the reference,
and `reverse_factorization_queue.py --smoke` also needs the re-evaluated budget study.

## Why the reference is built this way

A reader who wonders why this experiment needs quadrature and SMC when the rest
of the field uses NUTS should read this section; the scripts point back at it.

**Marginalizing the discrete parameters makes the continuous target multimodal.**
Summing the 25 configurations by log-sum-exp leaves a mixture in `(mu1, mu2)`
with one mode per dominant configuration — `(c1=2, mu1=8)` and `(c1=3, mu1=5)`
produce similar `E[Q1]`. Gradient-based samplers cannot cross well-separated
modes, and the gradient of a log-sum-exp offers no proposal that would. PyMC
reaches R-hat 1.64 and NumPyro 1.89 on the service rates, with bulk ESS near 6,
and both report `p(2,2 | x) = 0.24999` — the signature of one stuck chain in
four. That is what `pymc_automarginal.py` and `numpyro_marginal_reference.py`
exist to demonstrate; both refuse to write an artifact unlabelled when R-hat
exceeds their threshold.

**The M/M/c formula has a numerical trap, and an epsilon does not close it.**
The standard guard `E[Q] = (r^c rho) / (c! (1 - rho + 1e-10)^2) pi_0` returns
nonsensical negative values once `rho >= 1`. The `TruncatedNormal(lower=0)`
observation model then normalizes pathologically: `P(X > 0)` approaches zero and
the log-density at the observed positive queue length becomes extremely negative
but *finite*. That is a spurious attractor, not a divergence — at `mu1 ~ 2.66`
with `c1 = 2`, `rho = 1.69` — and chains that wander in during warmup stay,
because the gradient points back toward it. Its log-probability is about -318
against -3 at the true mode, so it is invisible to anything but per-chain
inspection. The fix is the hard `rho < 0.999` constraint, enforced in every
sampler here.

**So the reference is computed deterministically.** Conditional on `gamma` the
two stations are independent, which reduces each configuration's evidence to
nested one-dimensional integrals; `reference.py` evaluates those by quadrature
and samples by rejection, so neither a Markov chain nor an approximation of the
configuration weights is involved. `smc_reference.py` is the independent check:
exact log-sum-exp marginalization sampled with SMC, which anneals from prior to
posterior and therefore populates every mode in proportion to its mass. **The
multimodality was never the obstacle; NUTS was.**

**Why not `pymc_extras.marginalize()`.** Auto-marginalization works in general,
but it refuses a discrete variable that a `pm.Potential` depends on — and the
stability guard is exactly such a Potential. The workaround, expressing the
constraint as an observed Bernoulli, is accepted but needs a finite epsilon,
which turns an exact `-inf` into a penalty of about -69 and makes the constraint
soft. `config.build_pymc_marginalized_model` therefore sums the 25 terms with an
explicit log-sum-exp. `smc_reference.py --verify-against-pymc-marginalize`
confirms the two densities agree to 1.4e-14, so the explicit version is PyMC's
own marginalization without the epsilon.

**Reading MNPE's residual against a scale.** A C2ST of 0.54 means little in the
abstract. Two scales make it readable: the split floor, 0.502, from disjoint
splits of the reference against itself under the same protocol; and the gap
between two legitimate methods on the same posterior, 0.515 for SMC against
quadrature. Quote both against the same baseline — `reference_yardstick.py` also
reports `excess_over_discrete_bound`, measured against a total-variation lower
bound rather than the floor, and the two are not interchangeable.

**Agreement between two implementations of one algorithm is not validation.**
The independence has to be in the *method*, not just in the codebase. This is
why the per-configuration NUTS references are not used as a yardstick: the PyMC
and NumPyro versions run one chain per configuration and combine them with the
same Laplace evidence approximation, so their agreement measures nothing.
Appendix D.2 reports their distance from quadrature (C2ST 0.52–0.53) as evidence
that the deterministic route was necessary, and those rows are in
`results/queue/reference_yardstick.csv`; the scripts that produced them are
**not** included in this repository. The two marginalized samplers that ship here
are a different thing — exact log-sum-exp targets, kept because they evidence the
non-convergence claim above.

## Where output goes

Result tables go to `results/queue/` and are tracked, because the paper quotes
numbers from them. Everything else — trained posteriors, study pickles, the
tuning database, the reference `.npz` files, cached `_parts` directories, and the
diagnostic plots the scripts draw beside their tables — goes to
`artifacts/queue/` and is not, because it is regenerable and runs to gigabytes.

The one exception is the paper figure, which is a published artifact and so is
written to `figures/`.

## Rebuilding the figure without rerunning anything

`figures/make_queue_figure.py` reads `results/queue/figure_inputs.npz`, which is
tracked, so it works in a fresh clone. If the full budget study is present under
`artifacts/queue/`, it is preferred instead, and the script prints which source
it used. The figure also reads three tracked tables directly — `c2st_floor.csv`,
`reference_yardstick_smc.csv` and `dequantization_baseline_results.csv` — so
those are always the committed numbers.
