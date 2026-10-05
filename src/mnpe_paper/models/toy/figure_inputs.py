"""Reading the toy budget study into the inputs its paper figure draws.

The study is kilobytes, so this reduction is not about size: the figure's
posterior panels are drawn by sampling a trained estimator, and that file cannot
ship. Loading it executes its pickle, and the published one resolves a class from
a dependency this repository does not include, so ``torch.load`` raises in a fresh
clone. Seeded draws travel instead.
"""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from mnpe_paper import figure_export

__all__ = [
    "ARTIFACTS",
    "MODEL",
    "N_POSTERIOR_SAMPLES",
    "SAMPLE_SEED",
    "X_OBS_VALUES",
    "assemble",
]

MODEL = "toy"

ARTIFACTS = ("budget_study_results.pkl", "mnpe_posterior.pt")

# Declared here rather than in the figure script, so the export and the figure
# cannot disagree about what the panels show.
X_OBS_VALUES = (-0.5, 1.0, 2.5)
N_POSTERIOR_SAMPLES = 5000
SAMPLE_SEED = 42


def assemble(
    artifacts: Sequence[Union[str, Path]],
    load_posterior: Optional[Callable[[Path], Any]] = None,
) -> dict[str, Any]:
    """Read the budget study and draw from the trained posterior.

    ``load_posterior`` defaults to ``torch.load``; the tests supply a stub, since
    they must not depend on a trained estimator existing.
    """
    study_path, posterior_path = Path(artifacts[0]), Path(artifacts[1])
    study = figure_export.read_pickle(study_path)
    posterior = (load_posterior or _load_posterior)(posterior_path)

    return {
        **study,
        "x_obs_values": np.asarray(X_OBS_VALUES, dtype=float),
        "n_post_samples": N_POSTERIOR_SAMPLES,
        "posterior_samples": _draw(posterior),
        "provenance": figure_export.provenance(
            MODEL,
            [study_path, posterior_path],
            posterior_sampling={
                "seed": SAMPLE_SEED,
                "n_samples": N_POSTERIOR_SAMPLES,
                "x_obs_values": list(X_OBS_VALUES),
                "note": (
                    "Draws replace the trained estimator, which is not shipped. "
                    "Seeded, so the export is reproducible from the artifact."
                ),
            },
        ),
    }


def _draw(posterior: Any) -> np.ndarray:
    import torch

    torch.manual_seed(SAMPLE_SEED)
    return np.stack(
        [
            posterior.sample(
                (N_POSTERIOR_SAMPLES,),
                x=torch.tensor([[value]], dtype=torch.float32),
                show_progress_bars=False,
            )
            .numpy()
            .astype(np.float32)
            for value in X_OBS_VALUES
        ]
    )


def _load_posterior(path: Path) -> Any:
    import torch

    # The artifact is a pickled posterior object, not a state dictionary, which is
    # why it does not ship and this runs only against a local artifact.
    return torch.load(path, weights_only=False)
