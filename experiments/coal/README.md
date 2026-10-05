# Coal-mining-disaster changepoint

Bayesian changepoint detection on UK coal-mining disasters, 1851–1961. Two
continuous parameters and one discrete one, the last with 111 categories:

```
switchpoint  ~ DiscreteUniform(0, 110)
early_rate   ~ Exponential(1)
late_rate    ~ Exponential(1)
disasters[t] ~ Poisson(early_rate if t < switchpoint else late_rate)
```

Its role among the experiments is the size of the discrete space: 111 categories
against the queueing model's two dimensions of five. That is what makes the
classwise ECE uninformative here and the top-label variant the one reported, and
it is why the observations (one count per year, so 111 dimensions) are compressed
by an `FCEmbedding` before the flow. The model is
`mnpe_paper.models.coal.config`.

## Pipeline

Run from the repository root, in this order.

| Step | Script | Produces |
|------|--------|----------|
| 1 | `pymc_reference.py` | `artifacts/coal/pymc_reference_samples.npz` |
| 2 | `mnpe_hparam_tuning.py` *(optional)* | `artifacts/coal/hparam_study.db`, `hparam_best_params.json` |
| 3 | `budget_study.py` | `artifacts/coal/budget_study_results.pkl` |
| 4 | `../../figures/make_coal_figure.py` | `figures/coal_paper_figure.pdf` |

```bash
uv run python experiments/coal/pymc_reference.py
uv run python experiments/coal/budget_study.py
uv run python figures/make_coal_figure.py
```

Step 2 is optional and documents provenance only: the architecture it selected is
written out in `budget_study.py` and pinned by
`tests/test_architecture_pinning.py`, so no published number depends on rerunning
the search.

## The reference posterior

The discrete switchpoint is integrated out with `pymc_extras.marginalize()` and
recovered afterwards with `recover_marginals()`, rather than sampled directly.
This matters for the comparison being meaningful: a mixed-space sampler would have
to move between 111 switchpoint values, and its mixing would then be the suspect
whenever MNPE and MCMC disagreed. Marginalising removes that ambiguity — the
continuous chains see a smooth target, and the discrete posterior is exact given
them.

## Where output goes

Nothing this pipeline writes is tracked: the reference draws, the tuning database
and the budget-study pickle all land in `artifacts/coal/`, which is regenerable
and large (the pickle is about 17 MB, mostly posterior sample arrays). What ships
instead is `results/coal/figure_inputs.npz`, the compact export the figure reads,
written by `scripts/export_figure_inputs.py`.

So `figures/make_coal_figure.py` works in a fresh clone, and prefers your own
artifact over the export if you have rerun the pipeline — printing which of the
two it used either way.

## One example replicate, aggregated numbers

Appendix C.1 quotes a joint C2ST of 0.56 ± 0.01 at 10^5 simulations. The tracked
study gives 0.597 ± 0.023 (mean ± s.e.m. over five runs, two of them at 0.55), and
no table here reproduces the quoted value.

The single-model visual panels — the pairplot, the SBC eCDF and the ECE
reliability histogram — show one replicate, selected as the lowest joint C2ST at
the largest budget. Every *number* in the figure is aggregated over all five
replicates instead, with panels (b) and (d3) plotting mean ± SEM. The selection
rule lives in `mnpe_paper.models.coal.figure_inputs.example_rep_index` so that the
export and the figure script cannot disagree about which replicate is meant, and
the chosen index is printed when the figure builds. It belongs in the caption.
