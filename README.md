# MNPE — paper reproduction code

Code and tracked results reproducing the experiments in the
[MNPE paper](https://arxiv.org/abs/2605.13551) (mixed neural posterior estimation for models
with both discrete and continuous parameters).

**The method itself is not here.** MNPE is implemented and maintained in the
[sbi toolbox](https://sbi.readthedocs.io); build on that rather than on anything in this
repository. What this repository adds is the experiments: the models, the reference
posteriors they are scored against, the calibration diagnostics, and a script per paper
figure.

## Install

```bash
git clone https://github.com/mackelab/mnpe-paper.git && cd mnpe-paper
uv sync
```

No local checkouts are required — `uv sync` resolves everything, including sbi, from a
package index.

## Layout

| Path | Contents |
|------|----------|
| `src/mnpe_paper/` | Importable shared machinery: calibration diagnostics, plotting style, path resolution, per-model specifications |
| `experiments/` | Runnable scripts, one directory per experiment |
| `figures/` | One script per paper figure, plus the published PDFs |
| `results/` | Tracked result tables and the compact figure inputs every figure rebuilds from |
| `artifacts/` | Untracked. Full pipeline output, written only if you rerun a pipeline |
| `tests/` | Behaviour tests, including the architecture-pinning guard |

## Figures

Every figure below rebuilds from tracked data alone, in a fresh clone, with no pipeline rerun:

```bash
uv run python figures/make_coal_figure.py
```

| Figure | Producer | Reads | Experiment |
|--------|----------|-------|------------|
| `fig1.pdf` | `figures/make_fig1_panels.py`, panels only: they were assembled into `fig1.svg` by hand in Inkscape and exported to the PDF | nothing | method overview |
| `toy_paper_figure.pdf` | `figures/make_toy_figure.py` | `results/toy/figure_inputs.npz` | [toy](experiments/toy/) |
| `sensitivity_discrete_errors_redesign.pdf` | `figures/sensitivity_discrete_errors_figure.py` | `results/toy/sensitivity_discrete_errors.csv` | [toy](experiments/toy/) |
| `coal_paper_figure.pdf` | `figures/make_coal_figure.py` | `results/coal/figure_inputs.npz` | [coal](experiments/coal/) |
| `queue_paper_figure.pdf` | `figures/make_queue_figure.py` | `results/queue/figure_inputs.npz`, plus three tracked tables | [queue](experiments/queue/) |
| `scaling_figure.pdf` | `figures/scaling_figure.py` | `results/scaling/scaling_study_results.csv` | [scaling](experiments/scaling/) |
| `k_sweep_figure.pdf` | `figures/k_sweep_figure.py` | `results/scaling/k_sweep_results.csv` | [scaling](experiments/scaling/) |
| `figureHH.pdf` | `figures/make_hh_figure.py` | `results/hh/figure_inputs.npz` | [hh](experiments/hh/) |
| `figureHH_supp_1.pdf` | `figures/make_hh_supp_1_figure.py` | `results/hh/figure_inputs.npz` | [hh](experiments/hh/) |
| `figureHH_supp_2.pdf` | `figures/make_hh_supp_2_figure.py` | `results/hh/figure_inputs.npz` | [hh](experiments/hh/) |

## Rebuild all figures

```bash
uv run python scripts/make_all_figures.py
```

This runs every script in `figures/` and exits non-zero if any fails. Afterwards `git status`
shows no modified files: the tracked PDFs rebuild byte for byte. `make_fig1_panels.py`
regenerates Fig. 1's panels only, not the assembled `fig1.pdf`.

## Tests

```bash
uv run pytest
uv run ruff check .
```

## Citation

If you use this code, please cite the paper. The metadata is in
[`CITATION.cff`](CITATION.cff); as BibTeX:

```bibtex
@article{boelts2026mixed,
  title   = {Mixed neural posterior estimation for simulators with discrete and continuous parameters},
  author  = {Boelts, Jan and Schr{\"o}der, Cornelius and Beck, Jonas and Macke, Jakob H. and Deistler, Michael and Gedon, Daniel},
  journal = {Transactions on Machine Learning Research},
  issn    = {2835-8856},
  year    = {2026},
  month   = sep,
  url     = {https://openreview.net/forum?id=fOZZeRfdrC}
}
```

## Reproduction and posture

This repository is a frozen snapshot pinned to `sbi==0.27.0`. Bit-exact reproduction is
not claimed; runs are seeded and CPU-only, but thread counts and BLAS reduction order are not
pinned, so a rerun agrees with the tracked numbers within training noise rather than exactly.
Rerunning a pipeline rewrites its tracked tables; `git checkout results figures` restores them.
Issues reporting genuine reproduction problems are welcome; improvements to the method itself
belong upstream in `sbi`.
