"""The scaling sweeps' evaluation set must not move.

Every number in Appendix B.1 is scored on a prior-predictive set drawn from a
fixed seed, so the set is part of the result rather than an incidental detail of
a run. Nothing would raise if it moved: the sweeps would simply report different
metrics on different observations, and a rerun would disagree with the tracked
CSVs for a reason no error message would name.

Both sweeps and ``recompute_marginal_ece.py`` draw their sets through
:func:`mnpe_paper.models.scaling.sweep_common.make_test_set`. The hashes below are
of the tensors the published sweeps were evaluated on, so they pin the function
to the published behaviour rather than to itself.
"""

import hashlib
import unittest

from mnpe_paper.models.scaling.sweep_common import TEST_SEED, make_test_set

# (label, kwargs) -> (params digest, x digest) of the sets the published sweeps used.
PUBLISHED_DIGESTS = {
    # D-sweep calibration sets, one per D the grid visits.
    "D=2 n=1000": (dict(n=1000, D=2), ("f5a127394ede9832", "6daec478088f61d6")),
    "D=8 n=1000": (dict(n=1000, D=8), ("ead220ad86eb4d84", "b619f017677a1e69")),
    "D=32 n=1000": (dict(n=1000, D=32), ("470437f9de9464f4", "1fe69cc0eccf309c")),
    "D=128 n=1000": (dict(n=1000, D=128), ("b458406f5d600c39", "83f8c8825cef0a4a")),
    # D-sweep TV/SBC sets, drawn at the smaller size from the same seed.
    "D=2 n=300": (dict(n=300, D=2), ("3c61505e81b89f34", "2260c9f3ff1490e3")),
    "D=8 n=300": (dict(n=300, D=8), ("5c7d7a30a1f753a3", "5abc98e894e45259")),
    "D=32 n=300": (dict(n=300, D=32), ("9aa80508d62e8630", "3940913fc9f45779")),
    # K-sweep evaluation sets.
    "K=5 n=1000": (dict(n=1000, D=2, K=5), ("e4abacff4a087cb8", "b7d5b9930a2398a1")),
    "K=50 n=1000": (dict(n=1000, D=2, K=50), ("b5c835839b79e8f8", "dab0cba4db09a254")),
    # The K-sweep's SBC set is deliberately disjoint, at TEST_SEED + 1.
    "K=5 n=300 sbc": (
        dict(n=300, D=2, K=5, seed=TEST_SEED + 1),
        ("ad01bb0fcd57ca5e", "08984a1f502941b9"),
    ),
    "K=50 n=300 sbc": (
        dict(n=300, D=2, K=50, seed=TEST_SEED + 1),
        ("88ff628391abbdbc", "52a6d4694a9f681d"),
    ),
}


def digest(tensor) -> str:
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()[:16]


class EvaluationSetTests(unittest.TestCase):
    def test_matches_the_sets_the_published_sweeps_scored_on(self):
        for label, (kwargs, expected) in PUBLISHED_DIGESTS.items():
            with self.subTest(case=label):
                params, xs = make_test_set(**kwargs)
                self.assertEqual((digest(params), digest(xs)), expected)

    def test_draw_does_not_disturb_the_global_stream(self):
        """It runs under ``fork_rng``, so calling it must not shift training draws.

        The sweeps build their evaluation set between seeding and training, so a
        leak here would make every trained network depend on how many test points
        were drawn.
        """
        import torch

        torch.manual_seed(0)
        before = torch.randn(3)

        torch.manual_seed(0)
        make_test_set(n=64, D=4)
        after = torch.randn(3)

        self.assertTrue(torch.equal(before, after))

    def test_column_layout_is_continuous_then_discrete(self):
        """Every metric slices on this convention; a swap would invert them all."""
        params, xs = make_test_set(n=32, D=5, K=3)
        self.assertEqual(params.shape, (32, 10))
        self.assertEqual(xs.shape, (32, 5))
        discrete = params[:, 5:]
        self.assertTrue(set(discrete.unique().tolist()).issubset({0.0, 1.0, 2.0}))
        self.assertGreater(len(params[:, :5].unique()), 100)


if __name__ == "__main__":
    unittest.main()
