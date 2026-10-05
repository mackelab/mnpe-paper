"""Architecture pinning guard.

Every network the paper reports must be defined by this repository's own
hyperparameters rather than by whichever sbi version happens to be installed.

There are **two construction paths** with different defaults, and conflating them
is the trap this module exists to document.

*The factory path.* ``posterior_nn(model="mnpe", ...)`` defaults to
``num_bins=10`` and forwards it to the builder explicitly. The toy budget study
(two records), both scaling sweeps and the Hodgkin-Huxley experiment take this
path, so all five train with ten bins.

*The low-level path.* ``build_mnpe`` reaches ``_build_mixed_density_estimator``,
whose own default in the pinned sbi is ``num_bins=5``; later sbi versions raise it
to 10 to match the factory. The queueing and changepoint studies take this path,
and both name the value explicitly (9 and 10).

So no site inherits a value that differs between sbi versions. Pinning still
earns its place: the value the factory-path sites rely on is a *library* default,
and it could drift just as the low-level one does. The pinned numbers are
therefore what the published runs used, not the low-level defaults -- reasoning
from the low-level signature alone gets this backwards and silently halves the bin
count.

Later sbi versions also move the low-level ``num_transforms`` 2 -> 5 and
``num_components`` 5 -> 10. Every site names ``num_transforms``;
``num_components`` is inert here (see ``test_num_components_is_inert_under_nsf``).

Two properties are checked separately, because pinning a value is not the same as
passing it on:

* the hyperparameter record for each experiment pins everything the flow consumes;
* the hand-written forwarding lists actually hand those pins to the builder.

Three sites forward by naming each argument (queueing, changepoint,
Hodgkin-Huxley) rather than by ``**hparams``, so a pin added to the record but not
to the list would otherwise pass unnoticed.

Hyperparameter records are read from source with ``ast`` rather than imported: two
of the budget studies run their entire study at import time.
"""

import ast
import inspect
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PKG = REPO / "src" / "mnpe_paper"

# Architecture parameters that build_nsf actually consumes. Leaving any of these
# implicit hands control of the trained network to an sbi default.
REQUIRED_PINS = {"hidden_features", "num_transforms", "num_bins"}

# The spline bin count each experiment's published results were produced with.
# Only the queueing study differs, because its Optuna search tuned the value; the
# rest sit at 10.
PUBLISHED_NUM_BINS = {
    "toy budget study": 10,
    "toy final model": 10,
    "scaling D-sweep": 10,
    "scaling K-sweep": 10,
    "coal budget study": 10,
    "queue tuned spec": 9,
    "hodgkin-huxley": 10,
}

# Sites that build through the factory rather than calling build_mnpe directly.
# For these the factory's own defaults are what the network inherits, so they are
# the ones worth watching.
FACTORY_PATH_SITES = {
    "toy budget study",
    "toy final model",
    "scaling D-sweep",
    "scaling K-sweep",
    "hodgkin-huxley",
}

TOY = REPO / "experiments" / "toy" / "budget_study.py"
SCALING = REPO / "experiments" / "scaling" / "scaling_study.py"
KSWEEP = REPO / "experiments" / "scaling" / "k_sweep.py"
COAL = REPO / "experiments" / "coal" / "budget_study.py"
QUEUE_SPEC = PKG / "models" / "queue" / "spec.py"
QUEUE_EVAL = PKG / "models" / "queue" / "eval.py"
HH_INFERENCE = REPO / "experiments" / "hh" / "inference.py"
HH_CONFIG = REPO / "experiments" / "hh" / "configs" / "inference_config" / "mnpe_hh.yaml"

# Which files each record is read from, so a missing source is reported by name
# rather than as an obscure failure.
SITE_SOURCES = {
    "toy budget study": (TOY,),
    "toy final model": (TOY,),
    "scaling D-sweep": (SCALING,),
    "scaling K-sweep": (KSWEEP,),
    "coal budget study": (COAL,),
    "queue tuned spec": (QUEUE_EVAL, QUEUE_SPEC),
    "hodgkin-huxley": (HH_CONFIG,),
}


def _rel(path):
    return str(path.relative_to(REPO))


def _dict_call_keywords(node):
    """Keyword names of a ``dict(...)`` call, with literal values where available."""
    out = {}
    for kw in node.keywords:
        try:
            out[kw.arg] = ast.literal_eval(kw.value)
        except (ValueError, TypeError):
            out[kw.arg] = None  # computed or referenced, e.g. width scaling with D
    return out


def _module_dict(path, name):
    """Read a module-level ``NAME = dict(...)``, without importing the module.

    Only module-level statements are considered, so a same-named dict nested in a
    function or a conditional cannot be picked up by accident.
    """
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            if name in [t.id for t in node.targets if isinstance(t, ast.Name)]:
                return _dict_call_keywords(node.value)
    raise AssertionError(f"{name} is not a module-level dict(...) in {path.name}")


def _returned_dict(path, func_name):
    """Read the ``dict(...)`` a named function returns, without importing."""
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            for sub in ast.walk(node):
                if isinstance(sub, ast.Return) and isinstance(sub.value, ast.Call):
                    return _dict_call_keywords(sub.value)
    raise AssertionError(f"{func_name} in {path.name} returns no dict(...)")


def _dict_in_function(path, func_name, var_name):
    """Read ``var_name = dict(...)`` inside a named function, without importing."""
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            for sub in ast.walk(node):
                if isinstance(sub, ast.Assign) and isinstance(sub.value, ast.Call):
                    if var_name in [t.id for t in sub.targets if isinstance(t, ast.Name)]:
                        return _dict_call_keywords(sub.value)
    raise AssertionError(f"no {var_name} = dict(...) in {func_name} of {path.name}")


def _call_keyword_names(path, callee):
    """Keyword names handed to the first call of ``callee`` in a module."""
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name == callee:
                return {kw.arg for kw in node.keywords}
    raise AssertionError(f"no call to {callee}(...) in {path.name}")


def _hh_config():
    import yaml  # deferred: only the Hodgkin-Huxley record needs a YAML reader

    return yaml.safe_load(HH_CONFIG.read_text())


def _queue_num_bins():
    """The queueing bin count, resolved through the spec the forwarding list reads."""
    from mnpe_paper.models.queue.spec import TUNED_HPARAMS

    return TUNED_HPARAMS["num_bins"]


# How each record is read. Kept as thunks so a record with a missing source fails
# only the checks that read it.
SITE_READERS = {
    "toy budget study": lambda: _module_dict(TOY, "HPARAMS"),
    "toy final model": lambda: _module_dict(TOY, "FINAL_HPARAMS"),
    "scaling D-sweep": lambda: _returned_dict(SCALING, "hparams_for"),
    "scaling K-sweep": lambda: _module_dict(KSWEEP, "HPARAMS"),
    "coal budget study": lambda: _module_dict(COAL, "HPARAMS"),
    # Deliberately the forwarding list inside ``build_tuned_mnpe`` rather than
    # TUNED_HPARAMS: that dict also holds training settings which are not builder
    # arguments, so the forwarded subset is what actually determines the network.
    "queue tuned spec": lambda: _dict_in_function(QUEUE_EVAL, "build_tuned_mnpe", "kwargs"),
    "hodgkin-huxley": _hh_config,
}

# The bin count each record resolves to. Every reader here returns the integer
# directly, including the queueing one, whose forwarding list passes
# ``TUNED_HPARAMS["num_bins"]`` by reference and so must be resolved through the
# spec rather than read off the call site.
NUM_BINS_READERS = {
    **{label: (lambda r=read: r()["num_bins"]) for label, read in SITE_READERS.items()},
    "queue tuned spec": _queue_num_bins,
}


class ArchitecturePinningTests(unittest.TestCase):
    def test_every_record_is_read_from_a_present_source(self):
        """A record deleted or renamed must fail here by name, not drop out of the checks.

        A record that quietly stops being checked is exactly the silently-unchecked
        architecture this module guards.
        """
        self.assertEqual(set(SITE_READERS), set(PUBLISHED_NUM_BINS))
        self.assertEqual(set(SITE_SOURCES), set(PUBLISHED_NUM_BINS))
        missing = [_rel(p) for paths in SITE_SOURCES.values() for p in paths if not p.exists()]
        self.assertEqual(missing, [], f"record sources not in the repository: {missing}")

    def test_every_site_pins_the_consumed_architecture(self):
        for label in SITE_READERS:
            with self.subTest(site=label):
                missing = REQUIRED_PINS - set(SITE_READERS[label]())
                self.assertEqual(
                    missing, set(), f"{label} leaves {sorted(missing)} to an sbi default"
                )

    def test_num_bins_matches_the_published_runs(self):
        """A wrong value trains a different network while every other check stays green."""
        resolved = {label: read() for label, read in NUM_BINS_READERS.items()}
        self.assertEqual(resolved, PUBLISHED_NUM_BINS)

    def test_hand_written_forwarding_lists_include_the_pins(self):
        """Pinning a value in a record is useless if the call site never passes it.

        These three sites name each argument instead of splatting the record, so
        the record and the call can drift apart silently.
        """
        readers = {
            "queue tuned spec": lambda: set(
                _dict_in_function(QUEUE_EVAL, "build_tuned_mnpe", "kwargs")
            ),
            "coal budget study": lambda: _call_keyword_names(COAL, "partial"),
            "hodgkin-huxley": lambda: _call_keyword_names(HH_INFERENCE, "posterior_nn"),
        }
        # The H-H forwarding list lives in the training entry point, not the config
        # this record is otherwise read from.
        for label in readers:
            with self.subTest(site=label):
                names = readers[label]()
                self.assertIn("num_bins", names, f"{label} pins num_bins but never forwards it")

    def test_factory_forwards_the_bin_count_the_published_runs_used(self):
        """Measure what the factory hands the builder, rather than read a signature.

        Five records build through ``posterior_nn``, so the value their networks
        inherit is the factory's, not ``_build_mixed_density_estimator``'s. Reading
        the low-level signature instead gives 5 and silently halves the bin count,
        which is exactly the mistake this test exists to prevent.
        """
        import sbi.neural_nets.factory as factory
        import torch

        class _StopBuild(Exception):
            pass

        captured = {}
        real_builder = factory.model_builders["mnpe"]

        def spy(*_args, **kwargs):
            captured.update(kwargs)
            raise _StopBuild

        factory.model_builders["mnpe"] = spy
        try:
            theta = torch.cat(
                [torch.randn(64, 1), torch.randint(0, 2, (64, 1)).float()], dim=1
            )
            build_fn = factory.posterior_nn(model="mnpe")
            try:
                build_fn(theta, torch.randn(64, 1))
            except _StopBuild:
                pass
        finally:
            factory.model_builders["mnpe"] = real_builder

        self.assertEqual(captured.get("num_bins"), 10)
        for label in FACTORY_PATH_SITES:
            with self.subTest(site=label):
                self.assertEqual(PUBLISHED_NUM_BINS[label], captured["num_bins"])

    def test_unpinned_sbi_defaults_have_not_drifted(self):
        """Defaults this project relies on without pinning, so drift must be loud."""
        from sbi.neural_nets.net_builders.flow import build_nsf
        from sbi.neural_nets.net_builders.mixed_nets import _build_mixed_density_estimator

        mixed = inspect.signature(_build_mixed_density_estimator).parameters
        self.assertEqual(mixed["flow_model"].default, "nsf")
        self.assertEqual(mixed["tail_bound"].default, 10.0)
        self.assertEqual(mixed["discrete_hidden_layers"].default, 2)
        # These two size the two heads. Only the scaling D-sweep names
        # discrete_hidden_features, because its width scales with D and both heads
        # have to follow. Every other record -- including the scaling K-sweep,
        # whose published grid is a fixed width 32 -- relies on None, which derives
        # the discrete width from hidden_features.
        self.assertIsNone(mixed["discrete_hidden_features"].default)
        self.assertIsNone(mixed["continuous_hidden_features"].default)

        # The queueing spec does not pin num_blocks; its flows rely on this default.
        nsf = inspect.signature(build_nsf).parameters
        self.assertEqual(nsf["num_blocks"].default, 2)

    def test_num_components_is_inert_under_nsf(self):
        """``num_components`` is forwarded to the continuous head but discarded there.

        build_nsf has no such parameter, so it lands in **kwargs. This is why its
        default differing between sbi versions cannot affect any network here, and
        why the tuned specs deliberately do not pin it. If sbi ever gives build_nsf a real
        ``num_components``, this fails and the specs must start pinning it.
        """
        from sbi.neural_nets.net_builders.flow import build_nsf

        self.assertNotIn("num_components", inspect.signature(build_nsf).parameters)


if __name__ == "__main__":
    unittest.main()
