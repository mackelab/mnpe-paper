"""Reading the Hodgkin-Huxley evaluation outputs into the inputs its figures draw.

Three figures share one export: the main figure (traces, MSE and calibration
against the training budget, SBC, and a pairplot), the full 6x6 pairplot of
Fig. C-2, and the grid of observed and posterior-predictive traces of Fig. C-3.

The artifacts are those co-author Cornelius Schroeder supplied for the paper,
laid out as the pipeline writes them (``artifacts/hh/{n_train}/run_{seed}/``). Their
weight is voltage traces of 72,501 samples each; only the posterior-predictive
traces the figures draw survive, at full resolution and in their original
float32, so the export path draws exactly what the artifact path draws.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd

from mnpe_paper import figure_export

__all__ = [
    "ARTIFACTS",
    "DROPOUT_RATE",
    "FIGURE_OBS",
    "MODEL",
    "N_PREDICTIVE_TRACES",
    "OBS_IDS",
    "RUN_DIR",
    "TRACE_GRID_OBS",
    "assemble",
]

MODEL = "hh"

# The run every single-model panel shows: 100k training simulations, seed 2,
# no dropout. The budget-sweep panels read every run through the aggregated table.
RUN_DIR = "100000/run_2"
DROPOUT_RATE = 0.0

# The observation the main figure and Fig. C-2 show.
FIGURE_OBS = "284"

# Fig. C-3's rows, top to bottom, as published. Observation 20 appears twice.
TRACE_GRID_OBS = ("284", "7", "20", "2", "3", "6", "8", "13", "20", "35")
OBS_IDS = tuple(dict.fromkeys(TRACE_GRID_OBS))

# Fig. C-3 shows three posterior-predictive traces per row, the main figure two.
N_PREDICTIVE_TRACES = 3

ARTIFACTS = (
    "aggregated_results.csv",
    f"{RUN_DIR}/mse_results_{DROPOUT_RATE}.pkl",
    *(f"{RUN_DIR}/obs_{obs_id}_dropout_rate_{DROPOUT_RATE}.pickle" for obs_id in OBS_IDS),
)

# What the SBC and discrete-calibration panels read from the evaluation pickle.
_CALIBRATION_KEYS = (
    "num_posterior_samples",
    "ranks_marginals",
    "ece_discrete",
    "probs_discrete",
    "true_vals_discrete",
    "discrete_indices",
    "num_discrete_bins",
)


def assemble(artifacts: Sequence[Union[str, Path]]) -> dict[str, Any]:
    """Read the aggregated sweep, the evaluation run and the observations."""
    paths = [Path(path) for path in artifacts]
    table_path, calibration_path, obs_paths = paths[0], paths[1], paths[2:]

    table = pd.read_csv(table_path)
    evaluation = figure_export.read_pickle(calibration_path)

    observations = {}
    for obs_path in obs_paths:
        obs = figure_export.read_pickle(obs_path)
        observations[str(obs["obs_id"])] = {
            "x_raw_o": np.asarray(obs["x_raw_o"]),
            "x_post": np.asarray(obs["x_post"])[:N_PREDICTIVE_TRACES],
            "samples": _as_array(obs["samples"]),
            "theta_o": _as_array(obs["theta_o"]),
        }

    return {
        "aggregated_results": {column: table[column].to_numpy() for column in table.columns},
        "calibration": {key: _as_array(evaluation[key]) for key in _CALIBRATION_KEYS},
        "observations": observations,
        "provenance": figure_export.provenance(
            MODEL,
            paths,
            run=RUN_DIR,
            dropout_rate=DROPOUT_RATE,
            predictive_traces=(
                f"the first {N_PREDICTIVE_TRACES} posterior-predictive traces per "
                "observation, unresampled; the figures draw no more"
            ),
        ),
    }


def _as_array(value: Any) -> Any:
    """Tensors become arrays of the same dtype; everything else passes through."""
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return value
