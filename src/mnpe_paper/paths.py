"""Repository-relative paths, resolved once so no script has to guess.

Every script in this repository reads and writes through the three directories
defined here rather than through ``Path(__file__).parent``. The distinction that
matters is between the two output trees:

``RESULTS_DIR``
    Tracked. Result tables and the exported figure inputs — small, text or
    compressed arrays, committed, and sufficient on their own to rebuild every
    published figure.

``ARTIFACTS_DIR``
    Untracked. Full pipeline output: posterior samples, trained estimators,
    tuning databases. Written when a reader reruns a pipeline, gigabytes when
    they rerun all of them, and never needed to rebuild a figure. The figure
    scripts prefer these when present (see :mod:`mnpe_paper.figure_io`).

Set ``MNPE_PAPER_ROOT`` to override the resolved root, which is needed only when
the package is installed from a wheel rather than used from a clone.
"""

import os
from pathlib import Path

__all__ = [
    "ARTIFACTS_DIR",
    "FIGURES_DIR",
    "RESULTS_DIR",
    "model_artifacts_dir",
    "model_results_dir",
    "repo_root",
]

_ROOT_ENV_VAR = "MNPE_PAPER_ROOT"


def _is_repo_root(candidate: Path) -> bool:
    """Both markers are required, so a parent project's metadata cannot match."""
    return (candidate / "pyproject.toml").is_file() and (candidate / "src" / "mnpe_paper").is_dir()


def repo_root() -> Path:
    """Return the repository root.

    Resolution order: the ``MNPE_PAPER_ROOT`` environment variable, then the
    nearest ancestor of this file carrying the repository markers, then the
    nearest ancestor of the working directory.

    Raises:
        RuntimeError: If no candidate carries the markers, which means the
            package was installed from a wheel and has no clone to point at.
    """
    override = os.environ.get(_ROOT_ENV_VAR)
    if override:
        root = Path(override).expanduser().resolve()
        if not _is_repo_root(root):
            raise RuntimeError(
                f"{_ROOT_ENV_VAR} is set to {root}, which is not an mnpe-paper checkout "
                "(expected a pyproject.toml and a src/mnpe_paper directory there)."
            )
        return root

    for start in (Path(__file__).resolve(), Path.cwd().resolve() / "_"):
        for parent in start.parents:
            if _is_repo_root(parent):
                return parent

    raise RuntimeError(
        "Could not locate the mnpe-paper repository root. This happens when the "
        f"package is installed from a wheel rather than used from a clone; set "
        f"{_ROOT_ENV_VAR} to the checkout to continue."
    )


ROOT = repo_root()
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
ARTIFACTS_DIR = ROOT / "artifacts"


def model_results_dir(model: str) -> Path:
    """Return (and create) the tracked results directory for one experiment."""
    path = RESULTS_DIR / model
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_artifacts_dir(model: str) -> Path:
    """Return (and create) the untracked artifact directory for one experiment."""
    path = ARTIFACTS_DIR / model
    path.mkdir(parents=True, exist_ok=True)
    return path
