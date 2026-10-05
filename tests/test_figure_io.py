"""Behaviour of the figure-input loader and the path resolver.

The loader's contract is what makes a rebuilt figure's provenance checkable: it
prefers a full pipeline artifact, falls back to the tracked export, and says which
it used. A silent fallback would let a reader believe they had drawn a figure from
their own posteriors when they had not, so the announcement is asserted here
rather than left to inspection.
"""

import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np

from mnpe_paper import figure_io, paths
from mnpe_paper.figure_io import FigureInputs, figure_inputs_path, load_figure_inputs

MODEL = "queue"


def _write_export(root: Path, **arrays) -> Path:
    path = root / MODEL / "figure_inputs.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


class FigureIoTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        root = Path(self._tmp.name)
        self.results = root / "results"
        self.artifacts = root / "artifacts"
        (self.artifacts / MODEL).mkdir(parents=True)
        # Patching mnpe_paper.paths alone must be enough: figure_io reads through the
        # module rather than binding its own copies of these.
        patches = mock.patch.multiple(
            paths, RESULTS_DIR=self.results, ARTIFACTS_DIR=self.artifacts
        )
        patches.start()
        self.addCleanup(patches.stop)
        self.addCleanup(self._tmp.cleanup)

    def _load(self, **kwargs):
        """Load, returning the inputs alongside whatever was printed."""
        out = io.StringIO()
        with redirect_stdout(out):
            inputs = load_figure_inputs(MODEL, **kwargs)
        return inputs, out.getvalue()

    def test_prefers_the_full_artifact_and_says_so(self):
        _write_export(self.results, budgets=np.array([100, 1000]))
        artifact = self.artifacts / MODEL / "budget_study_results.pkl"
        artifact.write_bytes(b"not actually read; assemble does the reading")

        inputs, printed = self._load(
            artifacts=["budget_study_results.pkl"],
            assemble=lambda ps: {"budgets": np.array([7]), "read_from": ps[0].name},
        )

        self.assertEqual(inputs.source, "artifact")
        self.assertEqual(inputs["read_from"], "budget_study_results.pkl")
        np.testing.assert_array_equal(inputs["budgets"], [7])
        self.assertIn("full pipeline artifact", printed)
        self.assertIn("budget_study_results.pkl", printed)

    def test_falls_back_to_the_export_and_names_the_absent_artifact(self):
        _write_export(self.results, budgets=np.array([100, 1000]))

        inputs, printed = self._load(
            artifacts=["budget_study_results.pkl"],
            assemble=lambda ps: {"budgets": np.array([7])},
        )

        self.assertEqual(inputs.source, "export")
        np.testing.assert_array_equal(inputs["budgets"], [100, 1000])
        self.assertIn("tracked export", printed)
        self.assertIn("full artifact not present", printed)
        self.assertIn("budget_study_results.pkl", printed)

    def test_a_partially_present_artifact_set_does_not_count(self):
        """Assembling from half an artifact set would silently mix two sources."""
        _write_export(self.results, budgets=np.array([100]))
        (self.artifacts / MODEL / "present.pkl").write_bytes(b"")

        inputs, printed = self._load(
            artifacts=["present.pkl", "absent.pkl"],
            assemble=lambda ps: {"budgets": np.array([7])},
        )

        self.assertEqual(inputs.source, "export")
        self.assertIn("absent.pkl", printed)
        self.assertNotIn("present.pkl", printed)

    def test_an_artifact_with_no_reader_falls_back_with_that_reason(self):
        _write_export(self.results, budgets=np.array([100]))
        (self.artifacts / MODEL / "results.pkl").write_bytes(b"")

        inputs, printed = self._load(artifacts=["results.pkl"])

        self.assertEqual(inputs.source, "export")
        self.assertIn("no artifact reader", printed)

    def test_absolute_artifact_paths_are_honoured(self):
        _write_export(self.results, budgets=np.array([100]))
        elsewhere = Path(self._tmp.name) / "elsewhere.pkl"
        elsewhere.write_bytes(b"")

        inputs, printed = self._load(
            artifacts=[elsewhere], assemble=lambda ps: {"from": str(ps[0])}
        )

        self.assertEqual(inputs.source, "artifact")
        self.assertEqual(inputs["from"], str(elsewhere))
        self.assertIn("elsewhere.pkl", printed)

    def test_scalars_and_nested_objects_survive_the_export_round_trip(self):
        """numpy stores these as 0-d object arrays; a figure script needs them back."""
        _write_export(
            self.results,
            n_reps=np.array(5),
            subsample=np.array({"ref_samples": 2000}, dtype=object),
            budget_results=np.array([{"c2st": 0.53}], dtype=object),
            ranks=np.arange(6).reshape(3, 2),
        )

        inputs, _ = self._load()

        self.assertEqual(inputs["n_reps"], 5)
        self.assertEqual(inputs["subsample"], {"ref_samples": 2000})
        self.assertEqual(inputs["budget_results"][0]["c2st"], 0.53)
        self.assertEqual(inputs["ranks"].shape, (3, 2))

    def test_redirecting_the_paths_module_redirects_the_loader(self):
        """Guards against figure_io holding its own copy of the path constants."""
        _write_export(self.results, budgets=np.array([100]))
        self.assertEqual(figure_io.figure_inputs_path(MODEL).parent, self.results / MODEL)

    def test_no_inputs_at_all_names_both_places_looked(self):
        with self.assertRaises(FileNotFoundError) as caught:
            self._load(artifacts=["results.pkl"], assemble=lambda ps: {})

        message = str(caught.exception)
        self.assertIn("figure_inputs.npz", message)
        self.assertIn("results.pkl", message)
        self.assertIn("export_figure_inputs.py", message)

    def test_an_unknown_key_reports_what_is_available(self):
        _write_export(self.results, budgets=np.array([100]))
        inputs, _ = self._load()

        with self.assertRaises(KeyError) as caught:
            inputs["ece_marginal_mean"]

        self.assertIn("budgets", str(caught.exception))

    def test_inputs_are_read_only(self):
        _write_export(self.results, budgets=np.array([100]))
        inputs, _ = self._load()

        self.assertIsInstance(inputs, FigureInputs)
        with self.assertRaises(TypeError):
            inputs["budgets"] = np.array([1])

    def test_verbose_can_be_silenced(self):
        _write_export(self.results, budgets=np.array([100]))
        out = io.StringIO()
        with redirect_stdout(out):
            load_figure_inputs(MODEL, verbose=False)
        self.assertEqual(out.getvalue(), "")


class FigureInputsPathTests(unittest.TestCase):
    def test_the_export_lives_under_the_model_results_directory(self):
        """The exporter and every figure script agree on this location."""
        path = figure_inputs_path("toy")
        self.assertEqual(path.parent, paths.RESULTS_DIR / "toy")
        self.assertEqual(path.name, "figure_inputs.npz")


class PathsTests(unittest.TestCase):
    def test_repo_root_finds_this_checkout(self):
        root = paths.repo_root()
        self.assertTrue((root / "pyproject.toml").is_file())
        self.assertEqual(root, Path(__file__).resolve().parent.parent)

    def test_the_override_must_point_at_a_real_checkout(self):
        with TemporaryDirectory() as tmp:
            with mock.patch.dict("os.environ", {"MNPE_PAPER_ROOT": tmp}):
                with self.assertRaises(RuntimeError) as caught:
                    paths.repo_root()
        self.assertIn("MNPE_PAPER_ROOT", str(caught.exception))

    def test_the_override_is_honoured_for_a_real_checkout(self):
        with TemporaryDirectory() as tmp:
            fake = Path(tmp) / "clone"
            (fake / "src" / "mnpe_paper").mkdir(parents=True)
            (fake / "pyproject.toml").write_text("")
            with mock.patch.dict("os.environ", {"MNPE_PAPER_ROOT": str(fake)}):
                self.assertEqual(paths.repo_root(), fake.resolve())

    def test_model_directories_are_created_on_request(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.multiple(
                paths, RESULTS_DIR=root / "results", ARTIFACTS_DIR=root / "artifacts"
            ):
                self.assertTrue(paths.model_results_dir("toy").is_dir())
                self.assertTrue(paths.model_artifacts_dir("toy").is_dir())


if __name__ == "__main__":
    unittest.main()
