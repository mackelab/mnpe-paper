# Tractable Gaussian toy model

One continuous parameter and one binary one, with the mixed posterior available in
closed form:

```
theta ~ N(0, 1),  z ~ Bernoulli(1/2),  x | theta, z ~ N(theta + a·z, sigma^2)

p(z=1 | x) = sigmoid((a·x - a^2/2) / (1 + sigma^2))
p(theta | x) = a two-component Gaussian mixture, weights p(z | x)
```

This is the only experiment scored against a known answer rather than against
another sampler, which is what makes it the control for the rest: the analytic
derivation is in [`tractable-example-derivation.md`](tractable-example-derivation.md),
and the model itself is `mnpe_paper.models.toy.config`.

The coupling is the point. The continuous posterior *shifts* with the discrete
state, so at `x = 1` the marginal is genuinely bimodal with the mode weights set
by the discrete posterior — a method that got the discrete part wrong could not
place the continuous mass correctly, and here that failure is exactly measurable.

## Scripts

Run from the repository root. Every script beyond the budget study needs the
trained estimator the budget study writes, and says so if it is missing.

| Script | Produces | Paper |
|--------|----------|-------|
| `budget_study.py` | `artifacts/toy/budget_study_results.pkl`, `artifacts/toy/mnpe_posterior.pt` | Toy convergence panel |
| `dequantization_baseline_toy.py` | `results/toy/dequantization_baseline_toy.csv` | Appendix B.3, Table B-1 |
| `reverse_factorization_toy.py` | `results/toy/reverse_factorization_toy.csv` | Appendix B.4, Table B-2 |
| `sensitivity_discrete_errors.py` | `results/toy/sensitivity_discrete_errors.csv` | Appendix B.5 |

```bash
uv run python experiments/toy/budget_study.py          # first (~6 min): everything else reads its estimator
uv run python experiments/toy/sensitivity_discrete_errors.py   # ~8 min
uv run python figures/make_toy_figure.py               # main figure
uv run python figures/sensitivity_discrete_errors_figure.py   # Appendix B.5 figure
```

`reverse_factorization_toy.py` takes `--smoke` for a fast, smaller run; its output
is written under a `_SMOKE` name and is git-ignored.

## Where output goes

Result tables go to `results/toy/` and are tracked, because the paper quotes
numbers from them. Everything else — the trained estimator, the budget-study
pickle, and the diagnostic plots the scripts draw alongside their tables — goes to
`artifacts/toy/` and is not, because it is regenerable and, in the estimator's
case, cannot be shipped at all: loading it executes its pickle, and the published
one resolves a class from a dependency this repository does not include.

The one exception is the Appendix B.5 figure, which is a published figure and so
is written to `figures/`. The similarly named plot that
`sensitivity_discrete_errors.py` draws is the diagnostic view of the same sweep,
not the published one.

## Reproducing the figure without rerunning anything

`figures/make_toy_figure.py` builds from `results/toy/figure_inputs.npz`, which is
tracked, so it works in a fresh clone. The published posterior panels came from
**unseeded** draws; the tracked export seeds them at 42, so those panels differ
slightly. Panel b is read from the budget study and does match. See the script's
docstring.
