#!/usr/bin/env python
"""Derive the tracked figure inputs from full pipeline artifacts.

Writes ``results/<model>/figure_inputs.npz``, the compact summary every paper
figure rebuilds from in a fresh clone. It ships alongside its output so the
summary can be checked against the artifacts it claims to come from, together
with the ``provenance`` record inside each export.

The reading lives in the models, which the figure scripts share for their own
artifact path; this only resolves paths, writes and reports. A fresh clone has the
exports already and does not need to run it.

    uv run python scripts/export_figure_inputs.py
    uv run python scripts/export_figure_inputs.py --model queue --artifacts-dir ../run/queue
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from mnpe_paper import figure_export, figure_io, paths
from mnpe_paper.models.coal import figure_inputs as coal_inputs
from mnpe_paper.models.hh import figure_inputs as hh_inputs
from mnpe_paper.models.queue import figure_inputs as queue_inputs
from mnpe_paper.models.toy import figure_inputs as toy_inputs

MODELS = {module.MODEL: module for module in (queue_inputs, coal_inputs, toy_inputs, hh_inputs)}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        nargs="+",
        choices=sorted(MODELS),
        default=sorted(MODELS),
        help="experiments to export (default: all)",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        help="where this experiment's artifacts live, if not artifacts/<model>/",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="write under this directory instead of the tracked results tree",
    )
    args = parser.parse_args(argv)
    if args.artifacts_dir is not None and len(args.model) != 1:
        parser.error("--artifacts-dir applies to one experiment; pass a single --model")
    return args


def export(model: str, artifacts_dir: Optional[Path], out_dir: Optional[Path]) -> Path:
    module = MODELS[model]
    source_dir = artifacts_dir if artifacts_dir is not None else paths.ARTIFACTS_DIR / model
    resolved = [source_dir / name for name in module.ARTIFACTS]

    absent = [path for path in resolved if not path.exists()]
    if absent:
        raise FileNotFoundError(
            f"cannot export {model!r}: {', '.join(p.name for p in absent)} "
            f"{'is' if len(absent) == 1 else 'are'} not in {source_dir}. Point "
            "--artifacts-dir at the run that produced them."
        )

    data = module.assemble(resolved)
    destination = (
        out_dir / model / figure_io.EXPORT_NAME
        if out_dir is not None
        else figure_io.figure_inputs_path(model)
    )
    written = figure_export.write_export(destination, data)
    _report(model, data, destination, written)
    return destination


def _report(model: str, data: dict, destination: Path, written: int) -> None:
    record = data["provenance"]
    read = sum(source["bytes"] for source in record["sources"])
    print(f"[{model}] {destination}")
    for source in record["sources"]:
        print(f"    read  {source['name']:38s} {source['bytes'] / 1e6:8.2f} MB")
    print(
        f"    wrote {figure_io.EXPORT_NAME:38s} {written / 1e6:8.2f} MB"
        f"  ({read / written:.1f}x smaller)"
    )

    subsample = record.get("reference_subsample")
    if subsample:
        print(
            f"    reference draws: kept {subsample['n_kept']:,} of "
            f"{subsample['n_total']:,} (shuffle seed {subsample['shuffle_seed']})"
        )
    for retained in record.get("retained", ()):
        print(
            f"    kept {retained['key']:9s} for budget {retained['budget']:,} "
            f"replicate {retained['rep']} — {retained['selected_by']}"
        )
    sampling = record.get("posterior_sampling")
    if sampling:
        print(
            f"    posterior draws: {sampling['n_samples']:,} at each of "
            f"{sampling['x_obs_values']} (seed {sampling['seed']})"
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    failures = []
    for model in args.model:
        try:
            export(model, args.artifacts_dir, args.out_dir)
        except FileNotFoundError as absent:
            failures.append(str(absent))
            print(f"[{model}] skipped: {absent}", file=sys.stderr)
    if failures:
        print(
            f"\n{len(failures)} of {len(args.model)} experiments were not exported.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
