# Hodgkin-Huxley simulator

MNPE on a single-compartment neuron simulated with [jaxley](https://github.com/jaxleyverse/jaxley):
four continuous parameters (`g_Na`, `g_K`, `g_Leak`, `E_Leak`), a binary M-type
potassium channel switch and a three-way calcium channel choice (off, L-type,
T-type). The observation is 14 summary statistics of a 1.45 s voltage trace under
step-current stimulation. The model and prior are in `model.py`.

## Pipeline

Hydra-configured (`configs/`). Run from the repository root; everything is written
under `artifacts/hh/`.

| Step | Script | Produces |
|------|--------|----------|
| 1 | `sample.py` | `artifacts/hh/data/jaxley_data_10k_test.pkl` (test set, seed 43), and with overrides the 100k training set (seed 42) |
| 2 | `inference.py` | `artifacts/hh/{n_train}/run_{seed}/`: the trained estimator and `mse_results_{dropout}.pkl` (MSE, SBC ranks, discrete calibration) |
| 3 | `../../figures/make_hh_figure.py`, `make_hh_supp_1_figure.py`, `make_hh_supp_2_figure.py` | `figures/figureHH*.pdf` |

jaxley and Hydra are not dependencies of this repository, so that building the
figures never needs them; add them per run:

```bash
uv run --with jaxley==0.14.0 --with hydra-core==1.3.7 --with hydra-submitit-launcher==1.2.0 \
    python experiments/hh/sample.py
uv run --with jaxley==0.14.0 --with hydra-core==1.3.7 --with hydra-submitit-launcher==1.2.0 \
    python experiments/hh/sample.py data_config.n_samples=100000 data_config.seed=42 \
    data_config.output_file=data/jaxley_data_100k.pkl
uv run --with jaxley==0.14.0 --with hydra-core==1.3.7 --with hydra-submitit-launcher==1.2.0 \
    python experiments/hh/inference.py --multirun \
    inference_config.n_train_samples=100,500,1000,5000,10000,50000,100000 \
    inference_config.seed=1,2,3
```

The pipeline was checked to import and simulate with these versions; the ones
the published runs used were not recorded.

`sample.py` also stores every raw voltage trace (`save_raw_data` in
`configs/data_config/hh_model.yaml`), which nothing here reads: the 10k test set
is 2.9 GB, so the 100k training set is about 29 GB. Add
`data_config.save_raw_data=false` to skip them.

**Training needs a cluster.** The sweep is 21 trainings with up to 100k simulations
each, and evaluation simulates every posterior-predictive sample. The launcher
config, `configs/launcher/slurm.yaml`, submits to SLURM through submitit and names
the partition the published runs used, which you will need to change. A single
`inference.py` run without `--multirun` trains locally.

## Where the published numbers come from

The evaluation artifacts behind the paper are held by co-author Cornelius
Schroeder. What ships is `results/hh/figure_inputs.npz`, the compact export the
three figures read, written by `scripts/export_figure_inputs.py` from:

- `aggregated_results.csv`: every run's metrics, averaged over the three seeds per
  training budget (panels b and d);
- `100000/run_2/mse_results_0.0.pkl`: the SBC ranks and discrete calibration of the
  100k run at seed 2 (panel c);
- `100000/run_2/obs_{id}_dropout_rate_0.0.pickle` for nine test observations:
  observed traces, posterior samples and posterior-predictive traces (panels a, e,
  Figs. C-2 and C-3).

The last two steps — simulating posterior predictives for the displayed
observations, and averaging runs into the table — are not included in this
repository.

The stored discrete calibration of these runs scores the autoregressive factors,
so K_M, first in the order, is a genuine marginal, while Ca's probabilities are
conditioned on the true K_M, P(Ca | K_M, x); `inference.py` computes the marginal
(`probability_type="marginal"`).

To rebuild the figures from the artifacts rather than the export, put them under
`artifacts/hh/` with the layout above; the figure scripts print which source they
used.

## Calibration test-set size

SBC and discrete calibration use **N = 500** test cases (`num_samples_sbc` and
`num_samples_sbc_discrete` in `configs/inference_config/mnpe_hh.yaml`; the shipped
ranks and probabilities are 500 rows each). The EoD baseline in panel d is
computed for 500 runs accordingly.
