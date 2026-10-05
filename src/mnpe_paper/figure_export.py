"""Reducing full pipeline output to the compact inputs a figure draws.

The write side of :mod:`mnpe_paper.figure_io`. Each model's ``figure_inputs``
module owns the reading; both this exporter and the figure scripts call it, so a
key dropped here is dropped on the artifact path too and the two cannot diverge.
"""

import hashlib
import pickle
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

__all__ = [
    "HEAVY_REP_KEYS",
    "as_export_value",
    "assert_plain_data",
    "compact_rep_list",
    "compact_reps",
    "provenance",
    "read_pickle",
    "subsample_reference",
    "write_export",
]

EXPORTER = "scripts/export_figure_inputs.py"

# The only per-replicate arrays large enough to matter, and the only ones no
# convergence panel aggregates over.
HEAVY_REP_KEYS = ("samples", "pred_obs")

_PLAIN_SCALARS = (str, bytes, bool, int, float, complex, np.generic)


def compact_reps(
    budget_results: Mapping[int, Mapping[str, Any]],
    retain: Iterable[tuple[int, int, str]] = (),
) -> dict[int, dict[str, Any]]:
    """Drop :data:`HEAVY_REP_KEYS` except at the ``(budget, rep, key)`` triples given.

    Every triple must exist, so a retention naming the wrong replicate fails here
    rather than shipping a study with no draws in it.
    """
    keep: dict[int, dict[int, list[str]]] = {}
    for budget, index, key in retain:
        if budget not in budget_results:
            raise KeyError(
                f"cannot retain {key!r} at budget {budget}: the study has budgets "
                f"{sorted(budget_results)}"
            )
        reps = budget_results[budget]["reps"]
        if not -len(reps) <= index < len(reps):
            raise IndexError(
                f"cannot retain {key!r} at budget {budget}: replicate {index} of "
                f"{len(reps)} does not exist"
            )
        if key not in reps[index]:
            raise KeyError(
                f"cannot retain {key!r} at budget {budget} replicate {index}: the "
                f"replicate carries {sorted(reps[index])}"
            )
        keep.setdefault(budget, {}).setdefault(index, []).append(key)

    return {
        budget: {
            **{k: v for k, v in entry.items() if k != "reps"},
            "reps": compact_rep_list(entry["reps"], keep.get(budget, {})),
        }
        for budget, entry in budget_results.items()
    }


def compact_rep_list(
    reps: Sequence[Mapping[str, Any]],
    retain: Optional[Mapping[int, Sequence[str]]] = None,
) -> list[dict[str, Any]]:
    """Drop the heavy arrays from a flat list of replicates."""
    kept = {
        (index if index >= 0 else index + len(reps)): tuple(keys)
        for index, keys in (retain or {}).items()
    }
    return [
        {
            key: value
            for key, value in rep.items()
            if key not in HEAVY_REP_KEYS or key in kept.get(index, ())
        }
        for index, rep in enumerate(reps)
    ]


def subsample_reference(
    reference: np.ndarray, n_keep: int, seed: int = 0
) -> tuple[np.ndarray, dict[str, Any]]:
    """Shuffle reference draws, keep a prefix, and describe what was kept.

    The returned record states the reasoning for a reader who has only the export.
    """
    shuffled = np.array(reference, copy=True)
    np.random.default_rng(seed).shuffle(shuffled)
    n_kept = min(int(n_keep), len(shuffled))
    record = {
        "n_total": len(reference),
        "n_kept": n_kept,
        "shuffle_seed": seed,
        "note": (
            "Shuffled with numpy default_rng before the prefix was taken, since "
            "reference draws are ordered by discrete configuration. The count is "
            "what the pairplot displays; do not shuffle again before slicing."
        ),
    }
    return shuffled[:n_kept], record


def artifact_record(path: Union[str, Path]) -> dict[str, Any]:
    """Describe a source artifact so an export can be checked against it."""
    path = Path(path)
    return {
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def read_pickle(path: Union[str, Path]) -> Any:
    """Read a pipeline artifact.

    Unpickling executes what the file names, and unlike the exports this reads a
    path the caller supplies — a local artifact they produced. Never point it at a
    downloaded file; producing exports is what removes the need to.
    """
    with open(path, "rb") as handle:
        return pickle.load(handle)


def provenance(model: str, sources: Sequence[Union[str, Path]], **details: Any) -> dict[str, Any]:
    """Record how an export was derived, for the export to carry."""
    return {
        "model": model,
        "exporter": EXPORTER,
        "sources": [artifact_record(path) for path in sources],
        **details,
    }


def as_export_value(value: Any) -> np.ndarray:
    """Wrap a value for ``np.savez`` so it reloads as the same Python type.

    Left to ``np.array``, a list of budgets becomes an integer array — and the
    queueing figure calls ``budgets.index(...)``.
    """
    if isinstance(value, np.ndarray) and value.dtype != object:
        return value
    wrapped = np.empty((), dtype=object)
    wrapped[()] = value
    return wrapped


def assert_plain_data(data: Mapping[str, Any]) -> None:
    """Reject anything that would have to be unpickled to be read.

    The exports are built by spreading a whole study dictionary, so an artifact
    key holding an estimator or a tensor would otherwise travel unnoticed.
    """
    for path, value in _walk(data, ""):
        if value is None or isinstance(value, (*_PLAIN_SCALARS, np.ndarray)):
            continue
        raise TypeError(
            f"figure inputs must contain only arrays and plain containers, but "
            f"{path} is a {type(value).__name__}. Reduce it to arrays before "
            "exporting."
        )


def write_export(path: Union[str, Path], data: Mapping[str, Any]) -> int:
    """Write figure inputs to a compressed ``.npz``, returning its size in bytes."""
    assert_plain_data(data)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: as_export_value(value) for key, value in data.items()})
    return path.stat().st_size


def _walk(value: Any, path: str) -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _walk(item, f"{path}[{key!r}]" if path else str(key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _walk(item, f"{path}[{index}]")
    elif isinstance(value, np.ndarray) and value.dtype == object:
        for index, item in enumerate(value.reshape(-1)):
            yield from _walk(item, f"{path}[{index}]")
    else:
        yield path or "the export", value
