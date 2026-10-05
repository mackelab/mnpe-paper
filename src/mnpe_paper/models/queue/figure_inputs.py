"""Reading the queueing budget study into the inputs its paper figure draws.

Key names and the displayed replicate are those the figure script reads. The
convergence panels aggregate over every replicate, so only the
posterior and predictive draws can be dropped — the rank and probability arrays
are what those panels are computed from.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional, Union

from mnpe_paper import figure_export

__all__ = [
    "ARTIFACTS",
    "FIGURE_BUDGET",
    "FIGURE_REP",
    "MODEL",
    "OPTIONAL_ARTIFACTS",
    "REFERENCE_SHUFFLE_SEED",
    "assemble",
]

MODEL = "queue"

ARTIFACTS = ("budget_study_results_n1000.pkl",)

# Not required: a reader who reran the budget study but not the 1M training should
# still get their own numbers rather than falling back to the tracked export.
OPTIONAL_ARTIFACTS = ("queue_1M_eval.pkl",)

# Every single-model panel shows this budget and replicate. Both fixed rather than
# chosen on a metric, which would select on the quantity the figure reports.
FIGURE_BUDGET = 100_000
FIGURE_REP = 0

REFERENCE_SHUFFLE_SEED = 0

_REPLACED = ("budget_results", "ref_samples")


def assemble(artifacts: Sequence[Union[str, Path]]) -> dict[str, Any]:
    """Read the budget study, and the 1M evaluation if it was produced."""
    study_path = Path(artifacts[0])
    study = figure_export.read_pickle(study_path)

    budget_results = figure_export.compact_reps(
        study["budget_results"],
        retain=[(FIGURE_BUDGET, FIGURE_REP, key) for key in figure_export.HEAVY_REP_KEYS],
    )
    displayed = budget_results[FIGURE_BUDGET]["reps"][FIGURE_REP]

    reference, reference_record = figure_export.subsample_reference(
        study["ref_samples"],
        n_keep=len(displayed["samples"]),
        seed=REFERENCE_SHUFFLE_SEED,
    )

    one_m, extra_sources = _read_million_run(study_path.parent / OPTIONAL_ARTIFACTS[0])

    return {
        **{key: value for key, value in study.items() if key not in _REPLACED},
        "budget_results": budget_results,
        "ref_samples": reference,
        "one_m": one_m,
        "provenance": figure_export.provenance(
            MODEL,
            [study_path, *extra_sources],
            reference_subsample=reference_record,
            reference_file_note=(
                "reference_file records what the reference was called when the "
                "study ran, with a version tag the file this repository writes "
                "does not carry. Carried unchanged rather than rewritten, since it "
                "is a record of the run; nothing loads from it."
            ),
            retained=[
                {
                    "key": key,
                    "budget": FIGURE_BUDGET,
                    "rep": FIGURE_REP,
                    "selected_by": "the figure's fixed display replicate, not a metric",
                }
                for key in figure_export.HEAVY_REP_KEYS
            ],
        ),
    }


def _read_million_run(path: Path) -> tuple[Optional[Mapping[str, Any]], list[Path]]:
    """Read the 1M evaluation and the sources it adds, or ``None`` if absent.

    Returns ``None`` rather than omitting the key, so the two load paths agree in
    shape whether or not the 1M run was produced.
    """
    if not path.exists():
        return None, []
    evaluation = figure_export.read_pickle(path)
    one_m = {
        **{key: value for key, value in evaluation.items() if key != "reps"},
        "reps": figure_export.compact_rep_list(evaluation["reps"]),
    }
    return one_m, [path]
