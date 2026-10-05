"""Reading the changepoint budget study into the inputs its paper figure draws.

Unlike the queueing figure, the displayed replicate is chosen by a metric rather
than fixed, so which replicate's draws must survive follows from the C2ST values
every replicate keeps.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Union

import numpy as np

from mnpe_paper import figure_export

__all__ = [
    "ARTIFACTS",
    "MAX_BUDGET",
    "MODEL",
    "PRED_OBS_REP",
    "REFERENCE_SHUFFLE_SEED",
    "assemble",
    "example_rep_index",
]

MODEL = "coal"

ARTIFACTS = ("budget_study_results.pkl",)

MAX_BUDGET = 100_000

# The figure reads predictive draws from replicate 0 and posterior draws from its
# example replicate, so the two retained arrays belong to different replicates.
PRED_OBS_REP = 0

REFERENCE_SHUFFLE_SEED = 0

_REPLACED = ("budgets", "budget_results", "ref_samples")


def example_rep_index(reps: Sequence[Mapping[str, Any]]) -> int:
    """The replicate the single-model panels show: the lowest joint C2ST.

    Shared with the figure script so the two cannot disagree about which one is
    meant. It is an example run, which the caption states; every number the figure
    reports is aggregated over all replicates.
    """
    return int(np.argmin([rep["c2st_all"] for rep in reps]))


def assemble(artifacts: Sequence[Union[str, Path]]) -> dict[str, Any]:
    """Read the budget study, capped at the budget the figure stops at."""
    study_path = Path(artifacts[0])
    study = figure_export.read_pickle(study_path)

    budgets = [budget for budget in study["budgets"] if budget <= MAX_BUDGET]
    shipped = {budget: study["budget_results"][budget] for budget in budgets}

    largest = max(budgets)
    example = example_rep_index(shipped[largest]["reps"])
    budget_results = figure_export.compact_reps(
        shipped,
        retain=[(largest, example, "samples"), (largest, PRED_OBS_REP, "pred_obs")],
    )

    displayed = budget_results[largest]["reps"][example]
    reference, reference_record = figure_export.subsample_reference(
        study["ref_samples"],
        n_keep=len(displayed["samples"]),
        seed=REFERENCE_SHUFFLE_SEED,
    )

    return {
        **{key: value for key, value in study.items() if key not in _REPLACED},
        "budgets": budgets,
        "budget_results": budget_results,
        "ref_samples": reference,
        "provenance": figure_export.provenance(
            MODEL,
            [study_path],
            reference_subsample=reference_record,
            max_budget=MAX_BUDGET,
            retained=[
                {
                    "key": "samples",
                    "budget": largest,
                    "rep": example,
                    "selected_by": "lowest joint c2st at the largest budget",
                },
                {
                    "key": "pred_obs",
                    "budget": largest,
                    "rep": PRED_OBS_REP,
                    "selected_by": "the replicate the figure reads predictive draws from",
                },
            ],
        ),
    }
