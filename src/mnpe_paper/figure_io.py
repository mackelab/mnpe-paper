"""Figure input loading, with the data source stated out loud.

Every published figure must rebuild from a fresh clone, which rules out shipping
the full pipeline artifacts — tens of megabytes of posterior samples and pickled
estimators, where the figures need only metrics and one replicate's draws. What
ships instead is a compact per-model export under ``results/<model>/``, written
by ``scripts/export_figure_inputs.py``.

That alone would leave a reader who spent days regenerating a pipeline unable to
draw a figure from their own posteriors, so the loader prefers a full artifact
whenever one is present and falls back to the tracked export otherwise. Because
those two paths produce visually identical figures from different data, the
choice is **printed** rather than inferred: the provenance of any given PDF is
never ambiguous.

Assembling figure inputs from full artifacts is model-specific — it is the same
reading the exporter does — so the caller supplies that step as ``assemble``.
This module owns only the preference, the fallback and the announcement.

Typical use from a figure script::

    inputs = load_figure_inputs(
        "queue",
        artifacts=["budget_study_results_n1000.pkl"],
        assemble=lambda paths: _read_budget_study(paths[0]),
    )
    budgets = inputs["budgets"]
"""

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Optional, Union

import matplotlib as mpl
import numpy as np

# Imported as a module, not as names: binding RESULTS_DIR here would make a second
# copy that a redirected mnpe_paper.paths could no longer reach.
from mnpe_paper import paths

__all__ = [
    "FigureInputs",
    "FIGURE_FORMATS",
    "figure_inputs_path",
    "load_figure_inputs",
    "save_figure",
]

EXPORT_NAME = "figure_inputs.npz"

# What every figure script writes: the PDF the paper uses, plus a PNG for
# convenience. Only the PDF is tracked — see .gitignore.
FIGURE_FORMATS = ("pdf", "png")

Source = Literal["artifact", "export"]


def figure_inputs_path(model: str) -> Path:
    """Return the tracked figure-input export for one experiment.

    Single source of truth for the location, so the exporter that writes the file
    and the figure scripts that read it cannot drift apart.
    """
    return paths.RESULTS_DIR / model / EXPORT_NAME


class FigureInputs(Mapping):
    """A read-only mapping of figure inputs that remembers where it came from.

    Attributes:
        source: ``"artifact"`` if assembled from full pipeline output,
            ``"export"`` if read from the tracked summary.
        paths: The file or files the data was read from.

    Compare values key by key rather than comparing two instances. ``Mapping``
    supplies an ``__eq__`` that compares contents, and since most values here are
    numpy arrays, whole-object equality raises "truth value of an array is
    ambiguous" instead of failing cleanly.
    """

    def __init__(self, data: Mapping[str, Any], source: Source, paths: Sequence[Path]):
        self._data = dict(data)
        self.source = source
        self.paths = tuple(paths)

    def __getitem__(self, key: str) -> Any:
        try:
            return self._data[key]
        except KeyError:
            raise KeyError(
                f"{key!r} is not in the figure inputs read from "
                f"{_describe(self.paths)}. Available keys: {sorted(self._data)}"
            ) from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return (
            f"FigureInputs(source={self.source!r}, paths=({_describe(self.paths)}), "
            f"keys={sorted(self._data)})"
        )


def load_figure_inputs(
    model: str,
    artifacts: Iterable[Union[str, Path]] = (),
    assemble: Optional[Callable[[Sequence[Path]], Mapping[str, Any]]] = None,
    verbose: bool = True,
) -> FigureInputs:
    """Load one experiment's figure inputs, preferring full pipeline artifacts.

    Args:
        model: Experiment key, matching the directory name under ``results/``.
        artifacts: Full pipeline artifacts this figure can be built from.
            Relative paths resolve against ``artifacts/<model>/``. The artifact
            path is taken only if *every* one of them is present.
        assemble: Reads the artifacts into figure inputs. Required for the
            artifact path to be taken at all; without it the export is used.
        verbose: Print which source was used. Leave enabled in figure scripts —
            it is what keeps a rebuilt PDF's provenance unambiguous.

    Returns:
        The figure inputs, carrying the source they were read from.

    Raises:
        FileNotFoundError: If no export exists and the artifact path was not taken.
            Note that the fallback turns on artifact *existence* only: an artifact
            that is present but unreadable surfaces as whatever ``assemble``
            raises, which is deliberate, since silently falling back would hide a
            corrupt regeneration behind a figure that looks fine.
    """
    resolved = [_resolve_artifact(model, a) for a in artifacts]
    missing = [p for p in resolved if not p.exists()]

    if resolved and not missing and assemble is not None:
        data = assemble(resolved)
        if verbose:
            print(
                f"[mnpe-paper] {model} figure inputs: full pipeline artifact "
                f"({_describe(resolved)})"
            )
        return FigureInputs(data, "artifact", resolved)

    export = figure_inputs_path(model)
    if not export.exists():
        also_absent = f", and so are the full artifacts ({_describe(missing)})" if missing else ""
        raise FileNotFoundError(
            f"No figure inputs for {model!r}: the tracked export "
            f"{_describe([export])} is not present{also_absent}. Regenerate it with "
            "scripts/export_figure_inputs.py, which needs the full pipeline artifacts."
        )

    data = _load_export(export)
    if verbose:
        reason = ""
        if missing:
            reason = f" — full artifact not present ({_describe(missing)})"
        elif resolved and assemble is None:
            reason = " — this figure has no artifact reader"
        print(
            f"[mnpe-paper] {model} figure inputs: tracked export "
            f"({_describe([export])}){reason}"
        )
    return FigureInputs(data, "export", [export])


def save_figure(fig, stem: str, formats: Sequence[str] = FIGURE_FORMATS, **savefig_kwargs) -> Path:
    """Write one figure to ``figures/`` in each format, reproducibly.

    Args:
        fig: The figure to write.
        stem: Filename without extension, e.g. ``"coal_paper_figure"``.
        formats: Extensions to write. The first is returned.
        **savefig_kwargs: Passed through, e.g. ``bbox_inches="tight"``.

    Returns:
        The path written in the first format.

    Matplotlib stamps a ``/CreationDate`` into every PDF (and a date and randomly
    salted element ids into every SVG), which makes two builds
    of an unchanged figure differ in bytes and leaves ``git status`` dirty after a
    rebuild — the reader then cannot tell a real change from a new timestamp. This
    suppresses it, so an unchanged figure is byte-identical and ``git diff`` alone
    answers whether a rebuild reproduced the tracked PDF.
    """
    paths.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for extension in formats:
        path = paths.FIGURES_DIR / f"{stem}.{extension}"
        # The PDF and SVG backends stamp the date under different keys; a PNG
        # rejects both.
        metadata = {"pdf": {"CreationDate": None}, "svg": {"Date": None}}.get(extension)
        # SVG element ids are hashed with a fresh random salt unless one is fixed.
        with mpl.rc_context({"svg.hashsalt": "mnpe-paper"}):
            fig.savefig(path, metadata=metadata, **savefig_kwargs)
        written.append(path)
    return written[0]


def _resolve_artifact(model: str, artifact: Union[str, Path]) -> Path:
    path = Path(artifact)
    return path if path.is_absolute() else paths.ARTIFACTS_DIR / model / path


def _load_export(path: Path) -> dict[str, Any]:
    """Read an ``.npz`` export, unwrapping the scalars numpy stores as 0-d arrays.

    ``allow_pickle`` is required because the exports carry object arrays (nested
    metric dictionaries). The file being unpickled is tracked in this repository,
    not fetched, so this does not widen the trust boundary.
    """
    with np.load(path, allow_pickle=True) as handle:
        return {key: _unwrap(handle[key]) for key in handle.files}


def _unwrap(value: np.ndarray) -> Any:
    """Restore a Python object stored as a 0-d array; leave real arrays alone."""
    return value.item() if value.ndim == 0 else value


def _describe(items: Iterable[Path]) -> str:
    """Render paths for a message, relative to the repository root where possible."""
    return ", ".join(_relative_to_root(p) for p in items)


def _relative_to_root(path: Path) -> str:
    """Render a path relative to the repository root when it lies inside it."""
    try:
        return str(path.relative_to(paths.repo_root()))
    except ValueError:
        return str(path)
