"""Behaviour of the figure-input exporter and the per-model artifact readers.

A defect here passes its own acceptance criteria: a dropped key or a wrongly
chosen replicate still yields a figure that rebuilds successfully while showing
something other than what the paper reports. So the tests assert the three
properties that failure mode turns on — every key a figure reads survives, the
retained replicate is the one the figure displays, and nothing that has to be
unpickled travels.
"""

import hashlib
import pickle
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np

from mnpe_paper import figure_export, figure_io, paths
from mnpe_paper.models.coal import figure_inputs as coal_inputs
from mnpe_paper.models.hh import figure_inputs as hh_inputs
from mnpe_paper.models.queue import figure_inputs as queue_inputs
from mnpe_paper.models.toy import figure_inputs as toy_inputs

# The keys each figure script reads from its inputs.

QUEUE_TOP_LEVEL_KEYS = {
    "model_version",
    "budgets",
    "n_reps",
    "x_o",
    "ref_samples",
    "ref_pred_obs",
    "prior_pred_obs",
    "budget_results",
    "discrete_values",
    "reference_file",
    "n_sbc_test_actual",
    "observation_transform",
}

# Read for every replicate at every budget, to aggregate the convergence panels.
QUEUE_PER_REP_KEYS = {"c2st_all", "c2st_marginal", "ece", "ece_probs", "sbc_ranks"}

COAL_TOP_LEVEL_KEYS = {
    "budgets",
    "n_reps",
    "x_o",
    "ref_samples",
    "ref_pred_obs",
    "prior_pred_obs",
    "budget_results",
    "discrete_values",
}

COAL_PER_REP_KEYS = {"c2st_all", "c2st_marginal", "ece", "ece_probs", "sbc_ranks"}

TOY_KEYS = {"budgets", "n_repeats", "all_c2st", "posterior_samples", "x_obs_values"}

# Read by the three Hodgkin-Huxley figure scripts (make_hh_figure.py,
# make_hh_supp_1_figure.py, make_hh_supp_2_figure.py).
HH_KEYS = {"aggregated_results", "calibration", "observations"}
HH_CALIBRATION_KEYS = {
    "ranks_marginals",
    "num_posterior_samples",
    "probs_discrete",
    "true_vals_discrete",
    "ece_discrete",
    "discrete_indices",
    "num_discrete_bins",
}


# Synthetic rather than the real pickles, which are tens of megabytes, untracked,
# and absent from a fresh clone.


def _rep(c2st=0.6, n_sbc=8, n_disc=2, n_cat=5, n_theta=5, n_post=6, n_pred=4, n_obs=5):
    """One replicate, carrying the same keys the real budget studies write."""
    return {
        "samples": np.full((n_post, n_theta), c2st, dtype=np.float32),
        "c2st_marginal": {"gamma": c2st, "mu1": 0.51, "mu2": 0.52},
        "c2st_all": c2st,
        "sbc_ks_pvals": [0.3, 0.4, 0.5],
        "sbc_ranks": np.arange(n_sbc * n_theta, dtype=np.float32).reshape(n_sbc, n_theta),
        "joint_sbc_ks_pval": 0.42,
        "joint_sbc_ranks": np.zeros((n_sbc, 1), dtype=np.float32),
        "ece": [0.02, 0.03],
        "ece_classwise": [0.04, 0.05],
        "ece_probs": np.full((n_sbc, n_disc, n_cat), 1 / n_cat, dtype=np.float32),
        "ece_true_vals": np.zeros((n_sbc, n_disc), dtype=np.int64),
        "pred_obs": np.ones((n_pred, n_obs)),
    }


def _queue_study(budgets=(1000, 100_000), c2st_per_rep=(0.7, 0.6)):
    return {
        "model_version": "stable_joint_rho0999_v1",
        "budgets": list(budgets),
        "n_reps": len(c2st_per_rep),
        "hparams": {"num_bins": 9},
        "x_o": np.arange(5.0),
        "ref_samples": np.arange(200.0).reshape(40, 5),
        "ref_pred_obs": np.zeros((4, 5)),
        "prior_pred_obs": np.zeros((4, 5)),
        "budget_results": {b: {"reps": [_rep(c2st=c) for c in c2st_per_rep]} for b in budgets},
        "reference_file": "deterministic_reference.npz",
        "discrete_values": [2, 3, 4, 5, 6],
        "observation_transform": "log1p on dims [3, 4] (queue lengths)",
        "n_sbc_test_actual": 8,
    }


def _one_m_eval(n_reps=3):
    reps = [_rep(c2st=0.52) for _ in range(n_reps)]
    for rep in reps:
        rep["pred_obs"] = None
    return {
        "model_version": "stable_joint_rho0999_v1",
        "budget": 1_000_000,
        "n_reps": n_reps,
        "reps": reps,
        "reference_file": "deterministic_reference.npz",
        "n_sbc_test_actual": 8,
    }


def _coal_study(budgets=(1000, 100_000), c2st_per_rep=(0.9, 0.8, 0.6, 0.55, 0.7)):
    """The lowest-C2ST replicate is index 3, deliberately not index 0."""
    return {
        "budgets": list(budgets),
        "n_reps": len(c2st_per_rep),
        "hparams": {"num_bins": 10},
        "x_o": np.arange(6.0, dtype=np.float32),
        "observation_transform": "sqrt",
        "ref_samples": np.arange(90.0).reshape(30, 3),
        "ref_pred_obs": np.zeros((4, 6)),
        "prior_pred_obs": np.zeros((4, 6)),
        "budget_results": {
            b: {"reps": [_rep(c2st=c, n_theta=3, n_disc=1, n_cat=6) for c in c2st_per_rep]}
            for b in budgets
        },
        "reference_file": "pymc_reference_samples.npz",
        "discrete_values": list(range(1851, 1857)),
    }


def _toy_study():
    budgets = [100, 10_000]
    return {
        "budgets": budgets,
        "hparams": {"hidden_features": 50},
        "final_hparams": {"hidden_features": 50},
        "n_repeats": 5,
        "n_test_obs": 10,
        "all_c2st": {b: [0.6, 0.55] for b in budgets},
        "all_wass": {b: [0.1, 0.2] for b in budgets},
        "all_tv": {b: [0.01, 0.02] for b in budgets},
    }


class _StubPosterior:
    """Draws from torch so that seeding is observable."""

    def __init__(self):
        self.calls = []

    def sample(self, shape, x=None, show_progress_bars=True):
        import torch

        self.calls.append((shape, float(x[0, 0])))
        return torch.rand(shape[0], 2)


def _write_pickle(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)
    return path


class ExportValueTests(unittest.TestCase):
    """Round-tripping must not change a value's Python type: the queueing figure
    calls ``BUDGETS.index(b)``, which an array does not have."""

    def _round_trip(self, value):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.npz"
            np.savez_compressed(path, value=figure_export.as_export_value(value))
            return figure_io._load_export(path)["value"]

    def test_a_list_of_budgets_comes_back_as_a_list(self):
        restored = self._round_trip([1000, 2000, 100_000])
        self.assertEqual(restored, [1000, 2000, 100_000])
        self.assertEqual(restored.index(2000), 1)

    def test_nested_dictionaries_come_back_as_dictionaries(self):
        restored = self._round_trip({100: {"reps": [{"c2st_all": 0.53}]}})
        self.assertEqual(restored[100]["reps"][0]["c2st_all"], 0.53)

    def test_scalars_and_strings_keep_their_types(self):
        self.assertEqual(self._round_trip("log1p on dims [3, 4]"), "log1p on dims [3, 4]")
        self.assertEqual(self._round_trip(5), 5)
        self.assertIsNone(self._round_trip(None))

    def test_arrays_pass_through_untouched(self):
        value = np.arange(6, dtype=np.float32).reshape(3, 2)
        restored = self._round_trip(value)
        self.assertEqual(restored.dtype, np.float32)
        np.testing.assert_array_equal(restored, value)


class CompactionTests(unittest.TestCase):
    def test_every_replicate_keeps_its_metrics(self):
        study = _queue_study()
        compacted = figure_export.compact_reps(study["budget_results"], retain=())

        for budget, entry in compacted.items():
            for index, rep in enumerate(entry["reps"]):
                for key in QUEUE_PER_REP_KEYS:
                    self.assertIn(key, rep, f"budget {budget}, rep {index} lost {key}")

    def test_heavy_arrays_survive_only_where_they_are_retained(self):
        study = _queue_study()
        compacted = figure_export.compact_reps(
            study["budget_results"], retain=[(100_000, 0, "samples")]
        )

        self.assertIn("samples", compacted[100_000]["reps"][0])
        self.assertNotIn("pred_obs", compacted[100_000]["reps"][0])
        self.assertNotIn("samples", compacted[100_000]["reps"][1])
        self.assertNotIn("samples", compacted[1000]["reps"][0])

    def test_the_source_is_left_alone(self):
        """Compaction must not mutate the artifact dictionary it was handed."""
        study = _queue_study()
        figure_export.compact_reps(study["budget_results"], retain=[(1000, 0, "samples")])
        self.assertIn("samples", study["budget_results"][100_000]["reps"][0])

    def test_retaining_a_replicate_that_does_not_exist_is_an_error(self):
        """A silently ignored retention would drop the displayed replicate."""
        study = _queue_study()
        with self.assertRaises(KeyError) as caught:
            figure_export.compact_reps(study["budget_results"], retain=[(50_000, 0, "samples")])
        self.assertIn("50000", str(caught.exception))

        with self.assertRaises(IndexError):
            figure_export.compact_reps(study["budget_results"], retain=[(1000, 9, "samples")])

    def test_ragged_probability_arrays_survive(self):
        """One real replicate has 110 categories where the rest have 111."""
        study = _queue_study()
        odd = study["budget_results"][1000]["reps"][1]
        odd["ece_probs"] = np.zeros((8, 2, 4), dtype=np.float32)

        compacted = figure_export.compact_reps(study["budget_results"], retain=())

        self.assertEqual(compacted[1000]["reps"][0]["ece_probs"].shape, (8, 2, 5))
        self.assertEqual(compacted[1000]["reps"][1]["ece_probs"].shape, (8, 2, 4))


class ReferenceSubsampleTests(unittest.TestCase):
    def test_the_kept_draws_are_a_shuffled_prefix_and_the_record_says_so(self):
        reference = np.arange(60.0).reshape(20, 3)

        kept, record = figure_export.subsample_reference(reference, n_keep=5, seed=0)

        self.assertEqual(kept.shape, (5, 3))
        self.assertEqual(record["n_total"], 20)
        self.assertEqual(record["n_kept"], 5)
        self.assertEqual(record["shuffle_seed"], 0)
        # Every kept row is a real reference row, and they are not the first five:
        # the reference is ordered by discrete configuration, so an unshuffled
        # prefix would show one mode only.
        rows = {tuple(row) for row in reference}
        self.assertTrue(all(tuple(row) in rows for row in kept))
        self.assertFalse(np.array_equal(kept, reference[:5]))

    def test_subsampling_is_deterministic(self):
        reference = np.arange(60.0).reshape(20, 3)
        first, _ = figure_export.subsample_reference(reference, n_keep=5, seed=0)
        second, _ = figure_export.subsample_reference(reference, n_keep=5, seed=0)
        np.testing.assert_array_equal(first, second)

    def test_the_reference_array_is_not_shuffled_in_place(self):
        reference = np.arange(60.0).reshape(20, 3)
        original = reference.copy()
        figure_export.subsample_reference(reference, n_keep=5, seed=0)
        np.testing.assert_array_equal(reference, original)

    def test_asking_for_more_draws_than_exist_keeps_all_of_them(self):
        reference = np.arange(9.0).reshape(3, 3)
        kept, record = figure_export.subsample_reference(reference, n_keep=99, seed=0)
        self.assertEqual(kept.shape, (3, 3))
        self.assertEqual(record["n_kept"], 3)


class PlainDataGuardTests(unittest.TestCase):
    """The exporter refuses to write anything that would need unpickling."""

    def test_arrays_and_plain_containers_are_accepted(self):
        figure_export.assert_plain_data(
            {
                "budgets": [1, 2],
                "x_o": np.arange(3),
                "budget_results": {100: {"reps": [{"c2st_all": 0.5, "ece": None}]}},
                "note": "text",
                "flag": True,
                "count": np.int64(3),
            }
        )

    def test_an_estimator_anywhere_in_the_payload_is_rejected_by_path(self):
        class Estimator:
            pass

        with self.assertRaises(TypeError) as caught:
            figure_export.assert_plain_data(
                {"budget_results": {100: {"reps": [{"posterior": Estimator()}]}}}
            )

        message = str(caught.exception)
        self.assertIn("Estimator", message)
        self.assertIn("budget_results", message)
        self.assertIn("posterior", message)

    def test_an_object_array_is_searched_too(self):
        class Estimator:
            pass

        payload = np.empty((), dtype=object)
        payload[()] = {"posterior": Estimator()}
        with self.assertRaises(TypeError):
            figure_export.assert_plain_data({"wrapped": payload})


class QueueAssembleTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.study_path = _write_pickle(self.dir / queue_inputs.ARTIFACTS[0], _queue_study())
        self.one_m_path = _write_pickle(
            self.dir / queue_inputs.OPTIONAL_ARTIFACTS[0], _one_m_eval()
        )
        self.data = queue_inputs.assemble([self.study_path])

    def test_every_key_the_load_cell_reads_is_present(self):
        self.assertTrue(
            QUEUE_TOP_LEVEL_KEYS <= set(self.data), QUEUE_TOP_LEVEL_KEYS - set(self.data)
        )

    def test_metrics_survive_for_every_replicate_at_every_budget(self):
        for budget, entry in self.data["budget_results"].items():
            self.assertEqual(len(entry["reps"]), self.data["n_reps"])
            for index, rep in enumerate(entry["reps"]):
                for key in QUEUE_PER_REP_KEYS:
                    self.assertIn(key, rep, f"budget {budget}, rep {index}")

    def test_draws_survive_only_for_the_replicate_the_figure_displays(self):
        displayed = self.data["budget_results"][queue_inputs.FIGURE_BUDGET]["reps"][
            queue_inputs.FIGURE_REP
        ]
        self.assertIn("samples", displayed)
        self.assertIn("pred_obs", displayed)

        for budget, entry in self.data["budget_results"].items():
            for index, rep in enumerate(entry["reps"]):
                if (budget, index) == (queue_inputs.FIGURE_BUDGET, queue_inputs.FIGURE_REP):
                    continue
                self.assertNotIn("samples", rep, f"budget {budget}, rep {index}")
                self.assertNotIn("pred_obs", rep, f"budget {budget}, rep {index}")

    def test_reference_draws_are_subsampled_to_the_pairplot_count(self):
        displayed = self.data["budget_results"][queue_inputs.FIGURE_BUDGET]["reps"][
            queue_inputs.FIGURE_REP
        ]
        expected = min(len(displayed["samples"]), 40)
        self.assertEqual(len(self.data["ref_samples"]), expected)
        self.assertEqual(self.data["provenance"]["reference_subsample"]["n_kept"], expected)
        self.assertEqual(self.data["provenance"]["reference_subsample"]["n_total"], 40)

    def test_the_retained_replicate_is_recorded_in_the_output(self):
        retained = self.data["provenance"]["retained"]
        self.assertEqual(
            sorted((r["key"], r["budget"], r["rep"]) for r in retained),
            [
                ("pred_obs", queue_inputs.FIGURE_BUDGET, queue_inputs.FIGURE_REP),
                ("samples", queue_inputs.FIGURE_BUDGET, queue_inputs.FIGURE_REP),
            ],
        )

    def test_the_source_artifact_is_recorded_by_hash(self):
        sources = self.data["provenance"]["sources"]
        digest = hashlib.sha256(self.study_path.read_bytes()).hexdigest()
        recorded = {s["name"]: s for s in sources}
        self.assertEqual(recorded[self.study_path.name]["sha256"], digest)
        self.assertEqual(recorded[self.study_path.name]["bytes"], self.study_path.stat().st_size)

    def test_the_model_version_travels_so_the_stale_artifact_assertion_still_fires(self):
        self.assertEqual(self.data["model_version"], "stable_joint_rho0999_v1")

    def test_the_million_run_travels_with_its_draws_dropped(self):
        one_m = self.data["one_m"]
        self.assertEqual(one_m["budget"], 1_000_000)
        self.assertEqual(one_m["n_sbc_test_actual"], self.data["n_sbc_test_actual"])
        self.assertEqual(one_m["reference_file"], self.data["reference_file"])
        for rep in one_m["reps"]:
            self.assertNotIn("samples", rep)
            self.assertIn("c2st_all", rep)

    def test_the_million_run_is_absent_rather_than_missing_when_it_was_not_produced(self):
        """The key stays present so the two load paths never differ in shape."""
        self.one_m_path.unlink()
        data = queue_inputs.assemble([self.study_path])
        self.assertIn("one_m", data)
        self.assertIsNone(data["one_m"])


class CoalAssembleTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.study_path = _write_pickle(self.dir / coal_inputs.ARTIFACTS[0], _coal_study())
        self.data = coal_inputs.assemble([self.study_path])

    def test_every_key_the_load_cell_reads_is_present(self):
        self.assertTrue(COAL_TOP_LEVEL_KEYS <= set(self.data), COAL_TOP_LEVEL_KEYS - set(self.data))

    def test_metrics_survive_for_every_replicate(self):
        for budget, entry in self.data["budget_results"].items():
            for index, rep in enumerate(entry["reps"]):
                for key in COAL_PER_REP_KEYS:
                    self.assertIn(key, rep, f"budget {budget}, rep {index}")

    def test_draws_are_kept_for_the_lowest_c2st_replicate_not_the_first(self):
        """The figure selects its example by argmin over C2ST; keeping rep 0
        would leave the figure drawing a replicate it did not select, or
        crashing — and the export must not depend on which."""
        largest = max(self.data["budgets"])
        reps = self.data["budget_results"][largest]["reps"]
        selected = int(np.argmin([r["c2st_all"] for r in reps]))

        self.assertNotEqual(selected, 0, "fixture must not make rep 0 the example")
        self.assertIn("samples", reps[selected])
        for index, rep in enumerate(reps):
            if index != selected:
                self.assertNotIn("samples", rep, f"rep {index} kept draws it does not need")

    def test_predictive_draws_are_kept_for_replicate_zero(self):
        """The figure reads pred_obs from rep 0, not from its example rep."""
        largest = max(self.data["budgets"])
        reps = self.data["budget_results"][largest]["reps"]
        self.assertIn("pred_obs", reps[0])
        for index, rep in enumerate(reps[1:], start=1):
            self.assertNotIn("pred_obs", rep, f"rep {index} kept predictive draws")

    def test_budgets_above_the_figure_ceiling_are_not_shipped(self):
        _write_pickle(
            self.dir / coal_inputs.ARTIFACTS[0],
            _coal_study(budgets=(1000, 100_000, 1_000_000)),
        )
        data = coal_inputs.assemble([self.dir / coal_inputs.ARTIFACTS[0]])

        self.assertEqual(data["budgets"], [1000, 100_000])
        self.assertNotIn(1_000_000, data["budget_results"])

    def test_the_selection_rule_is_recorded_in_the_output(self):
        retained = {r["key"]: r for r in self.data["provenance"]["retained"]}
        self.assertEqual(retained["pred_obs"]["rep"], 0)
        self.assertEqual(retained["samples"]["rep"], 3)
        self.assertIn("c2st", retained["samples"]["selected_by"])


class ToyAssembleTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.study_path = _write_pickle(self.dir / toy_inputs.ARTIFACTS[0], _toy_study())
        self.posterior_path = self.dir / toy_inputs.ARTIFACTS[1]
        self.posterior_path.write_bytes(b"a real estimator, not read by these tests")
        self.stub = _StubPosterior()
        self.data = toy_inputs.assemble(
            [self.study_path, self.posterior_path], load_posterior=lambda _: self.stub
        )

    def test_every_key_the_load_cell_reads_is_present(self):
        self.assertTrue(TOY_KEYS <= set(self.data), TOY_KEYS - set(self.data))

    def test_draws_replace_the_estimator_one_row_per_observation(self):
        self.assertEqual(
            self.data["posterior_samples"].shape,
            (len(toy_inputs.X_OBS_VALUES), toy_inputs.N_POSTERIOR_SAMPLES, 2),
        )
        self.assertEqual([call[1] for call in self.stub.calls], list(toy_inputs.X_OBS_VALUES))

    def test_the_observations_and_the_draw_count_are_recorded(self):
        self.assertEqual(list(self.data["x_obs_values"]), list(toy_inputs.X_OBS_VALUES))
        self.assertEqual(self.data["n_post_samples"], toy_inputs.N_POSTERIOR_SAMPLES)
        self.assertEqual(
            self.data["provenance"]["posterior_sampling"]["seed"], toy_inputs.SAMPLE_SEED
        )

    def test_sampling_is_seeded_so_two_exports_agree(self):
        again = toy_inputs.assemble(
            [self.study_path, self.posterior_path], load_posterior=lambda _: _StubPosterior()
        )
        np.testing.assert_array_equal(self.data["posterior_samples"], again["posterior_samples"])

    def test_the_estimator_itself_does_not_travel(self):
        figure_export.assert_plain_data(self.data)


class HHAssembleTests(unittest.TestCase):
    """The evaluation pickles hold torch tensors and ten traces per observation."""

    N_TIME = 50

    def setUp(self):
        import torch

        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        paths = [self.dir / name for name in hh_inputs.ARTIFACTS]
        paths[0].write_text("dropout_rate,n_train,mse_p50_avg\n0.0,100,1.5\n0.5,100,1.2\n")
        _write_pickle(
            paths[1],
            {
                "mean": np.float32(3.0),
                "num_posterior_samples": 1000,
                "ranks_marginals": torch.zeros(8, 6),
                "ece_discrete": torch.tensor([0.01, 0.05]),
                "probs_discrete": torch.full((8, 2, 3), 1 / 3),
                "true_vals_discrete": torch.zeros(8, 2, dtype=torch.int64),
                "discrete_indices": [4, 5],
                "num_discrete_bins": 10,
            },
        )
        for path, obs_id in zip(paths[2:], hh_inputs.OBS_IDS):
            _write_pickle(
                path,
                {
                    "obs_id": obs_id,
                    "x_raw_o": np.arange(self.N_TIME, dtype=np.float32),
                    "x_post": np.ones((10, self.N_TIME), dtype=np.float32),
                    "samples": torch.zeros(10, 6),
                    "theta_o": torch.zeros(6),
                },
            )
        self.data = hh_inputs.assemble(paths)

    def test_every_key_the_figure_scripts_read_is_present(self):
        self.assertTrue(HH_KEYS <= set(self.data), HH_KEYS - set(self.data))
        self.assertEqual(set(self.data["calibration"]), HH_CALIBRATION_KEYS)

    def test_every_row_of_the_trace_grid_has_its_observation(self):
        self.assertTrue(set(hh_inputs.TRACE_GRID_OBS) <= set(self.data["observations"]))
        self.assertIn(hh_inputs.FIGURE_OBS, self.data["observations"])

    def test_traces_keep_full_resolution_and_only_the_drawn_count(self):
        for obs_id, obs in self.data["observations"].items():
            self.assertEqual(obs["x_raw_o"].shape, (self.N_TIME,), obs_id)
            self.assertEqual(obs["x_raw_o"].dtype, np.float32, obs_id)
            self.assertEqual(
                obs["x_post"].shape, (hh_inputs.N_PREDICTIVE_TRACES, self.N_TIME), obs_id
            )

    def test_the_whole_table_travels_so_the_figure_can_filter_it(self):
        self.assertEqual(list(self.data["aggregated_results"]["dropout_rate"]), [0.0, 0.5])

    def test_tensors_become_arrays(self):
        figure_export.assert_plain_data(self.data)
        self.assertEqual(self.data["calibration"]["true_vals_discrete"].dtype, np.int64)


class WriteAndReloadTests(unittest.TestCase):
    """What the exporter writes must be what a figure script reads back."""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def _assemble_queue(self):
        _write_pickle(self.dir / queue_inputs.ARTIFACTS[0], _queue_study())
        _write_pickle(self.dir / queue_inputs.OPTIONAL_ARTIFACTS[0], _one_m_eval())
        return queue_inputs.assemble([self.dir / queue_inputs.ARTIFACTS[0]])

    def test_a_written_export_reloads_key_for_key(self):
        assembled = self._assemble_queue()
        path = self.dir / "figure_inputs.npz"
        figure_export.write_export(path, assembled)

        reloaded = figure_io._load_export(path)

        self.assertEqual(set(reloaded), set(assembled))
        for key, value in assembled.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(reloaded[key], value, err_msg=key)
            else:
                self.assertEqual(type(reloaded[key]), type(value), key)
        # Spot-check through the nesting, where a type change would hide.
        original = assembled["budget_results"][100_000]["reps"][0]
        restored = reloaded["budget_results"][100_000]["reps"][0]
        self.assertEqual(set(restored), set(original))
        np.testing.assert_array_equal(restored["ece_probs"], original["ece_probs"])
        self.assertEqual(reloaded["budgets"].index(100_000), 1)

    def test_the_loader_reaches_a_written_export(self):
        assembled = self._assemble_queue()
        results = self.dir / "results"
        with mock.patch.multiple(paths, RESULTS_DIR=results, ARTIFACTS_DIR=self.dir / "artifacts"):
            figure_export.write_export(figure_io.figure_inputs_path("queue"), assembled)
            inputs = figure_io.load_figure_inputs("queue", verbose=False)

        self.assertEqual(inputs.source, "export")
        self.assertEqual(inputs["budgets"], assembled["budgets"])

    def test_writing_an_estimator_is_refused(self):
        class Estimator:
            pass

        with self.assertRaises(TypeError):
            figure_export.write_export(self.dir / "bad.npz", {"posterior": Estimator()})
        self.assertFalse((self.dir / "bad.npz").exists(), "a refused export must not be written")


# Ceilings, not measurements: room for a regeneration to move, tight enough that a
# heavy key slipping in trips before the file is committed.
EXPORT_CEILING_MB = {"queue": 4.0, "coal": 10.0, "toy": 0.5, "hh": 10.0}


class TrackedExportSizeTests(unittest.TestCase):
    """These files are tracked, so silent growth is a cost every clone pays."""

    def test_each_export_stays_within_its_ceiling(self):
        for model, ceiling in EXPORT_CEILING_MB.items():
            with self.subTest(model=model):
                size = figure_io.figure_inputs_path(model).stat().st_size / 1e6
                self.assertLess(
                    size,
                    ceiling,
                    f"{model} export is {size:.2f} MB, over its {ceiling} MB ceiling. "
                    "Check that no heavy per-replicate array is being retained for "
                    "replicates no figure displays.",
                )


if __name__ == "__main__":
    unittest.main()
