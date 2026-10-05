"""Invariants of the model specifications.

These are the properties the rest of each experiment quietly relies on, and each
one is a thing that a refactor can break silently. The toy model's
analytic posterior is the ground truth every toy metric is scored against, so a
sign or a constant wrong there would not raise — it would just move every C2ST in
the paper. The changepoint model's discrete parameter is carried as a 0-based index
and displayed as a year, so an off-by-one in that conversion mislabels a figure
axis without failing anywhere. The queueing model carries the same index hazard, and
two further ones: its stability predicate exists in a numpy and a torch copy that
must agree, and its JAX target is only useful as an independent check of the
reference if it computes the same E[Q] as the PyMC/numpy side.
"""

import unittest

import numpy as np
import torch

from mnpe_paper.models.coal import config as coal
from mnpe_paper.models.queue import config as queue
from mnpe_paper.models.queue import spec as queue_spec
from mnpe_paper.models.toy import config as toy
from mnpe_paper.models.toy import eval as toy_eval

X_VALUES = (-1.0, -0.5, 0.0, 1.0, 2.0, 2.5, 3.0)


class ToyAnalyticPosteriorTests(unittest.TestCase):
    """The closed-form posterior must agree with sampling from the same model."""

    def test_discrete_posterior_is_a_distribution(self):
        for x in X_VALUES:
            with self.subTest(x=x):
                p0 = toy.p_z_given_x(x, z=0)
                p1 = toy.p_z_given_x(x, z=1)
                self.assertAlmostEqual(p0 + p1, 1.0, places=12)
                self.assertTrue(0.0 < p1 < 1.0)

    def test_discrete_posterior_increases_with_x(self):
        """z=1 shifts the mean up by a, so larger x must favour it."""
        probs = [toy.p_z_given_x(x, z=1) for x in X_VALUES]
        self.assertEqual(probs, sorted(probs))

    def test_continuous_marginal_integrates_to_one(self):
        grid = np.linspace(-8.0, 10.0, 20001)
        for x in X_VALUES:
            with self.subTest(x=x):
                mass = np.trapezoid(toy.p_theta_given_x(grid, x), grid)
                self.assertAlmostEqual(mass, 1.0, places=6)

    def test_marginal_is_the_mixture_of_the_conditionals(self):
        """The mixture weights must be the discrete posterior, not the prior."""
        grid = np.linspace(-8.0, 10.0, 4001)
        for x in X_VALUES:
            with self.subTest(x=x):
                from scipy import stats

                rebuilt = sum(
                    toy.p_z_given_x(x, z=z)
                    * stats.norm.pdf(
                        grid, loc=toy.posterior_mean_given_z(x, z), scale=np.sqrt(toy.v)
                    )
                    for z in (0, 1)
                )
                np.testing.assert_allclose(toy.p_theta_given_x(grid, x), rebuilt, atol=1e-12)

    def test_ground_truth_sampler_matches_the_analytic_moments(self):
        rng_state = np.random.get_state()
        try:
            np.random.seed(0)
            for x in (-0.5, 1.0, 2.5):
                with self.subTest(x=x):
                    theta, z = toy.sample_gt_posterior(x, 200_000)
                    self.assertAlmostEqual(z.mean(), toy.p_z_given_x(x, z=1), places=2)
                    for z_val in (0, 1):
                        mask = z == z_val
                        if mask.sum() < 1000:
                            continue
                        self.assertAlmostEqual(
                            theta[mask].mean(),
                            toy.posterior_mean_given_z(x, z_val),
                            places=2,
                        )
                        self.assertAlmostEqual(theta[mask].std(), np.sqrt(toy.v), places=2)
        finally:
            np.random.set_state(rng_state)

    def test_simulator_and_prior_shapes(self):
        torch.manual_seed(0)
        params = toy.sample_prior(64)
        self.assertEqual(tuple(params.shape), (64, 2))
        self.assertTrue(set(params[:, 1].unique().tolist()) <= {0.0, 1.0})
        self.assertEqual(tuple(toy.simulator(params).shape), (64, 1))

    def test_evaluation_grid_spans_the_discrete_regimes(self):
        """The shared grid must cover near-0 to near-1 discrete posteriors.

        The three toy studies are comparable only because they score on the same
        grid; a grid that missed a regime would make one study's numbers look
        better for a reason that is not the method.
        """
        probs = [toy.p_z_given_x(x, z=1) for x in toy_eval.X_GRID]
        self.assertLess(min(probs), 0.05)
        self.assertGreater(max(probs), 0.90)


class CoalSpecificationTests(unittest.TestCase):
    def test_year_and_index_round_trip(self):
        for year in (coal.FIRST_YEAR, coal.SWITCHPOINT_TRUE_YEAR, coal.YEARS[-1]):
            self.assertEqual(coal.idx_to_year(coal.year_to_idx(year)), year)
        for idx in (0, coal.SWITCHPOINT_TRUE_IDX, coal.N_YEARS - 1):
            self.assertEqual(coal.year_to_idx(coal.idx_to_year(idx)), idx)

    def test_true_switchpoint_index_matches_its_year(self):
        self.assertEqual(coal.SWITCHPOINT_TRUE_IDX, coal.SWITCHPOINT_TRUE_YEAR - coal.FIRST_YEAR)
        self.assertEqual(coal.idx_to_year(coal.SWITCHPOINT_TRUE_IDX), coal.SWITCHPOINT_TRUE_YEAR)

    def test_data_and_year_axis_agree(self):
        self.assertEqual(len(coal.DISASTER_DATA), coal.N_YEARS)
        self.assertEqual(len(coal.YEARS), coal.N_YEARS)
        self.assertEqual(coal.N_SWITCHPOINTS, coal.N_YEARS)
        self.assertEqual(coal.SWITCHPOINT_MAX, coal.N_YEARS - 1)

    def test_rate_vector_switches_strictly_before_the_switchpoint(self):
        """`year_idx < switchpoint` — the switchpoint year itself gets the late rate.

        This is the convention the PyMC reference model uses; flipping it would
        shift the recovered switchpoint by one year against the MCMC reference.
        """
        rates = coal.compute_rate_vector(39, early_rate=3.0, late_rate=1.0)
        self.assertEqual(rates[38], 3.0)
        self.assertEqual(rates[39], 1.0)
        self.assertEqual(len(rates), coal.N_YEARS)

    def test_transform_x_is_variance_stabilising_sqrt(self):
        raw = torch.tensor([[0.0, 1.0, 4.0, 9.0]])
        torch.testing.assert_close(coal.transform_x(raw), torch.tensor([[0.0, 1.0, 2.0, 3.0]]))
        # Accepts raw arrays too, which is how the observed counts arrive.
        torch.testing.assert_close(
            coal.transform_x(np.array([[4.0]], dtype=np.float32)), torch.tensor([[2.0]])
        )

    def test_prior_puts_the_discrete_parameter_last(self):
        """MNPE requires continuous parameters first and the discrete one last."""
        torch.manual_seed(0)
        samples = coal.build_sbi_prior().sample((512,))
        self.assertEqual(tuple(samples.shape), (512, 3))
        switchpoint = samples[:, 2]
        torch.testing.assert_close(switchpoint, switchpoint.round())
        self.assertGreaterEqual(switchpoint.min().item(), 0)
        self.assertLessEqual(switchpoint.max().item(), coal.N_YEARS - 1)
        self.assertGreater(samples[:, 0].min().item(), 0.0)  # Exponential support

    def test_simulator_shapes_and_determinism_under_seed(self):
        theta = np.array([[3.0, 1.0, 39.0], [2.0, 0.5, 70.0]])
        batched = coal.simulator(theta, seed=7)
        self.assertEqual(batched.shape, (2, coal.N_YEARS))
        np.testing.assert_array_equal(batched, coal.simulator(theta, seed=7))
        self.assertEqual(coal.simulator(theta[0], seed=7).shape, (coal.N_YEARS,))


class QueueSpecificationTests(unittest.TestCase):
    """The queueing model's index convention, stability predicate and E[Q]."""

    def test_server_count_index_round_trip(self):
        """Discrete parameters are 0-indexed inside sbi and physical for display."""
        for index in range(queue_spec.N_C):
            physical = queue_spec.to_physical(index)
            self.assertEqual(int(physical), int(queue_spec.C_VALUES[index]))
            self.assertEqual(int(queue_spec.to_index(physical)), index)
        np.testing.assert_array_equal(
            queue_spec.to_physical(np.arange(queue_spec.N_C)), queue_spec.C_VALUES
        )

    def test_true_parameters_are_inside_the_conditioned_prior(self):
        """The reference observation must come from a point the prior supports."""
        self.assertTrue(bool(queue_spec.stability_mask_numpy(queue.THETA_TRUE_SBI)))
        self.assertEqual(
            int(queue.THETA_TRUE_SBI[3]), int(queue_spec.to_index(queue_spec.C1_TRUE))
        )
        self.assertEqual(
            int(queue.THETA_TRUE_SBI[4]), int(queue_spec.to_index(queue_spec.C2_TRUE))
        )

    def test_numpy_and_torch_stability_masks_agree(self):
        """Two copies of one predicate, used by the simulator and by the prior.

        The numpy copy gates training-data rejection and the torch copy defines
        the prior's support. If they disagreed, the prior would assign mass where
        the simulator returns NaN, and nothing downstream would raise.
        """
        rng = np.random.default_rng(0)
        theta = np.column_stack([
            rng.lognormal(np.log(9.0), 0.5, 4096),
            rng.lognormal(np.log(8.0), 0.5, 4096),
            rng.lognormal(np.log(5.0), 0.5, 4096),
            rng.integers(0, queue_spec.N_C, 4096),
            rng.integers(0, queue_spec.N_C, 4096),
        ])
        numpy_mask = queue_spec.stability_mask_numpy(theta)
        torch_mask = queue.stable_prior_mask(torch.tensor(theta, dtype=torch.float64))
        np.testing.assert_array_equal(numpy_mask, torch_mask.numpy())
        # The draw has to exercise both branches or the comparison is vacuous.
        self.assertTrue(numpy_mask.any() and not numpy_mask.all())

    def test_prior_samples_lie_in_the_conditioned_support(self):
        torch.manual_seed(0)
        samples = queue.build_sbi_prior().sample((512,))
        self.assertEqual(tuple(samples.shape), (512, 5))
        self.assertTrue(bool(queue.stable_prior_mask(samples).all()))
        discrete = samples[:, 3:5]
        torch.testing.assert_close(discrete, discrete.round())

    def test_jax_target_reproduces_the_numpy_expected_queue(self):
        """The independent reimplementation must agree with the NumPy one.

        `targets` is a deliberate second implementation in JAX, behind the NumPyro
        reference of Appendix D.2. Comparing it with the PyMC route only means
        something while the two actually compute the same thing.
        """
        from mnpe_paper.models.queue import targets

        for c in queue_spec.C_VALUES:
            for gamma, mu in ((9.0, 8.0), (9.0, 5.0), (6.0, 2.0)):
                if gamma / (int(c) * mu) >= queue_spec.MAX_UTILIZATION:
                    continue
                with self.subTest(c=int(c), gamma=gamma, mu=mu):
                    self.assertAlmostEqual(
                        float(targets.expected_queue_jax(
                            np.array([float(c)]),
                            np.array([gamma]),
                            np.array([mu]),
                        )[0]),
                        queue.compute_E_Q(int(c), gamma, mu),
                        places=9,
                    )

    def test_reference_observation_is_reproducible(self):
        first = queue.generate_reference_observation(seed=42)
        second = queue.generate_reference_observation(seed=42)
        np.testing.assert_array_equal(first["x_array"], second["x_array"])
        self.assertTrue(first["stable"])
        self.assertLess(first["rho1"], queue_spec.MAX_UTILIZATION)
        self.assertLess(first["rho2"], queue_spec.MAX_UTILIZATION)


if __name__ == "__main__":
    unittest.main()
