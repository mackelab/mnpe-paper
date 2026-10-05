"""Every key a figure script reads must exist in the tracked export it falls back to.

The acceptance criterion these guard is "the figure rebuilds in a tree containing
no pipeline artifacts". That fails in a particular way: a figure script reads a key
the exporter drops, and nothing notices until someone clones the repository fresh,
because on the developer's machine the artifact path is taken and supplies it.

``test_figure_input_export`` approaches this from the write side, listing by hand
the keys each figure needs. This module approaches it from the
read side, extracting the accesses from the figure scripts themselves, so the two
cannot both drift in the same direction: adding a key to a figure script without
adding it to the exporter fails here even if the hand-written list is never updated.
"""

import ast
import unittest
from pathlib import Path

import numpy as np

from mnpe_paper import figure_io, paths

# The name each figure script binds its loaded inputs to.
INPUTS_NAME = "inputs"

# (model, script) pairs: one export can feed several figures.
FIGURE_SCRIPTS = (
    ("toy", "make_toy_figure.py"),
    ("coal", "make_coal_figure.py"),
    ("queue", "make_queue_figure.py"),
    ("hh", "make_hh_figure.py"),
    ("hh", "make_hh_supp_1_figure.py"),
    ("hh", "make_hh_supp_2_figure.py"),
)

# Build from a tracked CSV (or, for Fig. 1, from no data) rather than an export, so
# they have no figure inputs to check — but they must still write reproducibly.
CSV_BACKED_SCRIPTS = (
    "sensitivity_discrete_errors_figure.py",
    "scaling_figure.py",
    "k_sweep_figure.py",
    "make_fig1_panels.py",
)


def figure_script_path(name: str) -> Path:
    return paths.repo_root() / "figures" / name


def keys_read_from_inputs(source: str) -> set[str]:
    """Collect the string keys a script subscripts out of ``inputs``.

    Deliberately syntactic: importing a figure script would draw it. Only constant
    string subscripts are collected, which is all the figure scripts use — a
    computed key would be missed, so this asserts what it can see rather than
    claiming completeness.
    """
    found = set()
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == INPUTS_NAME
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            found.add(node.slice.value)
    return found


class FigureScriptInputTests(unittest.TestCase):
    def test_each_script_reads_only_keys_the_export_carries(self):
        for model, script in FIGURE_SCRIPTS:
            with self.subTest(script=script):
                read = keys_read_from_inputs(figure_script_path(script).read_text())
                self.assertTrue(read, f"{script} appears to read no figure inputs")

                export = figure_io.figure_inputs_path(model)
                self.assertTrue(export.is_file(), f"{export} is not tracked")
                with np.load(export, allow_pickle=True) as handle:
                    available = set(handle.files)

                self.assertEqual(
                    read - available,
                    set(),
                    f"{script} reads keys absent from {export.name}; the figure "
                    "would fail in a clone with no pipeline artifacts",
                )

    def test_each_script_loads_through_the_provenance_aware_loader(self):
        """A script reading the export directly would draw without saying so.

        Every figure announces which source it was built from. That property is a
        property of ``load_figure_inputs``, so bypassing it silently loses it.
        """
        for _, script in FIGURE_SCRIPTS:
            with self.subTest(script=script):
                source = figure_script_path(script).read_text()
                self.assertIn("load_figure_inputs", source)
                self.assertNotIn("np.load", source)

    def test_scripts_save_through_the_reproducible_writer(self):
        """A bare ``fig.savefig`` stamps a timestamp and breaks byte comparison.

        Matplotlib writes a ``/CreationDate`` into every PDF, so two builds of an
        unchanged figure differ in bytes and a rebuild leaves ``git status`` dirty
        — at which point a reader cannot tell a real change from a new clock
        reading. ``save_figure`` suppresses it; bypassing it silently gives that up.
        """
        for script in (*(script for _, script in FIGURE_SCRIPTS), *CSV_BACKED_SCRIPTS):
            with self.subTest(script=script):
                source = figure_script_path(script).read_text()
                self.assertIn("save_figure(", source)
                self.assertNotIn("fig.savefig(", source)

    def test_scripts_name_their_artifacts_from_the_model_module(self):
        """The artifact list and the reader must come from one place.

        If a figure script spelled its own artifact filenames, the artifact path
        could go on being taken after the exporter's list changed, and the two
        paths would then build the figure from different data.
        """
        for _, script in FIGURE_SCRIPTS:
            with self.subTest(script=script):
                source = figure_script_path(script).read_text()
                self.assertIn("figure_inputs.ARTIFACTS", source)
                self.assertIn("figure_inputs.assemble", source)


if __name__ == "__main__":
    unittest.main()
