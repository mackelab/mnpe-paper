"""Two properties of the queueing experiment that go wrong silently.

The queueing model is the largest surface in the repository — twenty-seven
scripts across four role directories — and both properties guarded here fail
without raising anything.

*The provenance table.* A reader's only route from a paper number back to the
script that produced it is the table in ``experiments/queue/README.md``. A script
added or renamed without a table row leaves that number unattributed, and nothing
about a missing row is visible from running the code.

*The estimator module path.* Pickle stores classes by module path, so the two
ablations' cached part files name ``mnpe_paper.models.queue.nets`` inside them.
Moving or renaming that module makes every cached run unloadable, and the failure
surfaces hours into a resumed sweep rather than at import.
"""

import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
QUEUE = REPO / "experiments" / "queue"
QUEUE_README = QUEUE / "README.md"

# The role directories the README describes. A script filed outside them is not
# navigable by that scheme.
ROLE_DIRECTORIES = {"main", "references", "ablations", "diagnostics"}

# Pickle resolves these by module path. Changing either constant invalidates
# every cached part file the ablations wrote.
NETS_MODULE = "mnpe_paper.models.queue.nets"
PICKLED_CLASSES = ("HeadMaskedMDE", "PermutedCategoricalMADE")


class QueueProvenanceTests(unittest.TestCase):
    def test_every_script_has_a_provenance_row(self):
        readme = QUEUE_README.read_text()
        for script in sorted(QUEUE.rglob("*.py")):
            with self.subTest(script=script.name):
                self.assertIn(
                    f"`{script.name}`",
                    readme,
                    f"{script.name} has no row in experiments/queue/README.md, so "
                    "nothing it produces can be traced back to the paper",
                )

    def test_scripts_are_filed_under_a_role_directory(self):
        for script in sorted(QUEUE.rglob("*.py")):
            with self.subTest(script=script.name):
                self.assertEqual(script.parent.parent, QUEUE)
                self.assertIn(script.parent.name, ROLE_DIRECTORIES)


class PickledEstimatorPathTests(unittest.TestCase):
    def test_masked_estimator_classes_resolve_at_the_recorded_path(self):
        module = __import__(NETS_MODULE, fromlist=PICKLED_CLASSES)
        for name in PICKLED_CLASSES:
            with self.subTest(cls=name):
                cls = getattr(module, name, None)
                self.assertIsNotNone(
                    cls, f"{NETS_MODULE}.{name} is gone; cached ablation parts "
                    "referencing it can no longer be unpickled"
                )
                self.assertEqual(cls.__module__, NETS_MODULE)


if __name__ == "__main__":
    unittest.main()
