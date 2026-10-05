"""The scaling sweeps' grid guards: seed completeness and the row-directory merge."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from mnpe_paper.models.scaling.sweep_common import check_seed_completeness, merge_row_dir


def _grid(cells, seeds=3):
    return pd.DataFrame([{"D": d, "budget": b, "seed": s} for d, b in cells for s in range(seeds)])


class SeedCompletenessTests(unittest.TestCase):
    def test_complete_grid_passes(self):
        frame = _grid([(2, 10), (2, 100), (4, 10), (4, 100)])
        self.assertEqual(check_seed_completeness(frame, "D", strict=True), [])

    def test_absent_cell_counts_as_zero_seeds(self):
        frame = _grid([(2, 10), (2, 100), (4, 10)])
        self.assertEqual(check_seed_completeness(frame, "D"), ["D=4, budget=100: 0 seed(s)"])
        with self.assertRaises(SystemExit):
            check_seed_completeness(frame, "D", strict=True)


class MergeRowDirTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.rows = Path(self.dir.name) / "rows"
        self.rows.mkdir()
        self.out = Path(self.dir.name) / "grid.csv"
        _grid([(2, 10), (4, 10)]).to_csv(self.out, index=False)

    def tearDown(self):
        self.dir.cleanup()

    def _write_rows(self, frame):
        for i, row in frame.iterrows():
            row.to_frame().T.to_csv(self.rows / f"D{i}.csv", index=False)

    def _merge(self):
        return merge_row_dir(self.rows, self.out, sort_by=["budget", "D", "seed"], prefix="D")

    def test_refuses_to_drop_runs_already_in_the_table(self):
        self._write_rows(_grid([(4, 10)]))
        before = self.out.read_text()
        with self.assertRaises(SystemExit):
            self._merge()
        self.assertEqual(self.out.read_text(), before)

    def test_merges_a_row_directory_that_covers_the_table(self):
        self._write_rows(_grid([(2, 10), (4, 10), (8, 10)]))
        self.assertEqual(self._merge(), 9)


if __name__ == "__main__":
    unittest.main()
