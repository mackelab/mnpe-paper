import itertools
import unittest
from types import SimpleNamespace

import torch

from mnpe_paper.calibration import exact_marginal_variable_count, run_discrete_calibration


class _ChainMade:
    """Autoregressive binary MADE with analytically enumerable marginals.

    ``P(d_i = 1 | d_<i) = sigmoid(a_i + b_i * sum(d_<i))``, which respects the
    autoregressive mask (output ``i`` reads only inputs before ``i``) while
    making every conditional depend on the prefix, so a hybrid exact/sampled
    marginalizer cannot pass by ignoring the prefix.
    """

    max_num_categories = 2
    _needs_mapping = False

    def __init__(self, num_variables=4):
        self.num_variables = num_variables
        self.num_categories_per_var = torch.full((num_variables,), 2)
        self.a = torch.linspace(-0.8, 0.9, num_variables)
        self.b = torch.linspace(0.5, -0.6, num_variables)

    def _p_one(self, i, prefix_sum):
        return torch.sigmoid(self.a[i] + self.b[i] * prefix_sum)

    def forward(self, inputs, condition=None):
        batch_size = inputs.shape[0]
        logits = torch.empty(batch_size, self.num_variables, 2)
        for i in range(self.num_variables):
            prefix_sum = inputs[:, :i].sum(dim=1) if i else torch.zeros(batch_size)
            p_one = self._p_one(i, prefix_sum)
            logits[:, i, 0] = torch.log1p(-p_one)
            logits[:, i, 1] = torch.log(p_one)
        return logits.reshape(batch_size, -1)

    def brute_force_marginals(self):
        """Marginals from explicit enumeration of the full joint, shape (V, 2)."""
        marginals = torch.zeros(self.num_variables, 2)
        for config in itertools.product([0.0, 1.0], repeat=self.num_variables):
            joint = torch.tensor(1.0)
            for i, value in enumerate(config):
                p_one = self._p_one(i, torch.tensor(sum(config[:i])))
                joint = joint * (p_one if value else 1.0 - p_one)
            for i, value in enumerate(config):
                marginals[i, int(value)] += joint
        return marginals


class _TwoVariableMade:
    num_variables = 2
    max_num_categories = 2
    num_categories_per_var = torch.tensor([2, 2])
    _needs_mapping = False

    def forward(self, inputs, condition=None):
        batch_size = inputs.shape[0]
        logits = torch.empty(batch_size, 2, 2)
        logits[:, 0, :] = torch.log(torch.tensor([0.2, 0.8]))
        p_second_one = torch.where(
            inputs[:, 0] == 0,
            torch.tensor(0.1),
            torch.tensor(0.7),
        )
        logits[:, 1, 0] = torch.log1p(-p_second_one)
        logits[:, 1, 1] = torch.log(p_second_one)
        return logits.reshape(batch_size, -1)


def _posterior_with(made):
    estimator = SimpleNamespace(
        discrete_net=SimpleNamespace(net=made),
    )
    return SimpleNamespace(
        potential_fn=SimpleNamespace(posterior_estimator=estimator),
    )


class DiscreteCalibrationTests(unittest.TestCase):
    def test_returns_true_marginals_by_default(self):
        posterior = _posterior_with(_TwoVariableMade())
        thetas = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        xs = torch.zeros(2, 1)

        _, probs, _ = run_discrete_calibration(
            thetas,
            xs,
            posterior,
            discrete_indices=[0, 1],
        )

        expected = torch.tensor([0.42, 0.58])
        torch.testing.assert_close(probs[:, 1, :], expected.expand(2, -1))

    def test_factor_mode_preserves_teacher_forced_diagnostic(self):
        posterior = _posterior_with(_TwoVariableMade())
        thetas = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        xs = torch.zeros(2, 1)

        _, probs, _ = run_discrete_calibration(
            thetas,
            xs,
            posterior,
            discrete_indices=[0, 1],
            probability_type="factor",
        )

        torch.testing.assert_close(probs[0, 1], torch.tensor([0.9, 0.1]))
        torch.testing.assert_close(probs[1, 1], torch.tensor([0.3, 0.7]))

    def test_sampled_fallback_estimates_marginals(self):
        """Below the exact budget, the estimate must still find the marginals.

        With ``max_exact_states=1`` this reaches ``_marginal_discrete_probs_hybrid``.
        """
        posterior = _posterior_with(_TwoVariableMade())
        thetas = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        xs = torch.zeros(2, 1)

        _, probs, _ = run_discrete_calibration(
            thetas,
            xs,
            posterior,
            discrete_indices=[0, 1],
            max_exact_states=1,
            num_marginal_samples=20_000,
            marginal_seed=123,
        )

        expected = torch.tensor([0.42, 0.58])
        torch.testing.assert_close(
            probs[:, 1, :], expected.expand(2, -1), atol=0.01, rtol=0
        )

    def test_single_variable_is_unchanged(self):
        made = _TwoVariableMade()
        made.num_variables = 1
        made.num_categories_per_var = torch.tensor([2])

        def forward(inputs, condition=None):
            return torch.log(torch.tensor([0.2, 0.8])).expand(inputs.shape[0], -1)

        made.forward = forward
        posterior = _posterior_with(made)
        thetas = torch.tensor([[0.0], [1.0]])
        xs = torch.zeros(2, 1)

        _, marginal, _ = run_discrete_calibration(
            thetas, xs, posterior, discrete_indices=[0]
        )
        _, factor, _ = run_discrete_calibration(
            thetas,
            xs,
            posterior,
            discrete_indices=[0],
            probability_type="factor",
        )

        torch.testing.assert_close(marginal, factor)


class ExactVariableCountTests(unittest.TestCase):
    def test_counts_leading_variables_within_budget(self):
        binary_32 = [2] * 32
        # Variable i costs 2**i prefixes, so a 32768-state budget buys 0..15.
        self.assertEqual(exact_marginal_variable_count(binary_32, 32768), 16)
        self.assertEqual(exact_marginal_variable_count(binary_32, 4096), 13)
        # Variable 0 needs a single forward pass and is always affordable.
        self.assertEqual(exact_marginal_variable_count(binary_32, 1), 1)

    def test_full_count_matches_the_all_exact_condition(self):
        # The dispatch routes to the untouched exact routine exactly when the
        # joint prefix fits; the helper must agree, or D<=16 results could move.
        for num_variables in (2, 5, 12):
            counts = [2] * num_variables
            joint_prefix = 2 ** (num_variables - 1)
            self.assertEqual(
                exact_marginal_variable_count(counts, joint_prefix), num_variables
            )
            self.assertLess(
                exact_marginal_variable_count(counts, joint_prefix - 1), num_variables
            )


class HybridMarginalizationTests(unittest.TestCase):
    """Covers 1 < num_exact < D-1, where a prefix expansion precedes the
    transition draw. The D=2 cases above all reduce to num_exact == 1."""

    NUM_VARIABLES = 4
    MAX_EXACT_STATES = 2  # buys variables 0 and 1; 2 and 3 are sampled

    def setUp(self):
        self.made = _ChainMade(self.NUM_VARIABLES)
        self.posterior = _posterior_with(self.made)
        self.thetas = torch.zeros(2, self.NUM_VARIABLES)
        self.xs = torch.zeros(2, 1)
        self.expected = self.made.brute_force_marginals()
        self.num_exact = exact_marginal_variable_count(
            [2] * self.NUM_VARIABLES, self.MAX_EXACT_STATES
        )
        assert 1 < self.num_exact < self.NUM_VARIABLES - 1, self.num_exact

    def _probs(self, seed, num_samples=40_000):
        _, probs, _ = run_discrete_calibration(
            self.thetas,
            self.xs,
            self.posterior,
            discrete_indices=list(range(self.NUM_VARIABLES)),
            max_exact_states=self.MAX_EXACT_STATES,
            num_marginal_samples=num_samples,
            marginal_seed=seed,
        )
        return probs

    def test_fully_exact_route_matches_enumeration(self):
        _, probs, _ = run_discrete_calibration(
            self.thetas,
            self.xs,
            self.posterior,
            discrete_indices=list(range(self.NUM_VARIABLES)),
            max_exact_states=10_000,
        )
        torch.testing.assert_close(probs[0], self.expected)

    def test_exact_head_matches_enumeration_without_sampling_error(self):
        head = self._probs(seed=0)[0, : self.num_exact]
        torch.testing.assert_close(head, self.expected[: self.num_exact])

    def test_exact_head_is_seed_independent(self):
        # The point of the hybrid: these dimensions carry no Monte Carlo noise.
        first = self._probs(seed=0)[0, : self.num_exact]
        second = self._probs(seed=999)[0, : self.num_exact]
        self.assertTrue(torch.equal(first, second))

    def test_sampled_tail_is_unbiased(self):
        tail = self._probs(seed=7)[0, self.num_exact :]
        torch.testing.assert_close(
            tail, self.expected[self.num_exact :], atol=0.01, rtol=0
        )

    def test_sampled_tail_does_carry_seed_dependence(self):
        # Guards the converse of the head test: if the tail were secretly
        # deterministic, the head test above would prove nothing.
        first = self._probs(seed=0, num_samples=256)[0, self.num_exact :]
        second = self._probs(seed=999, num_samples=256)[0, self.num_exact :]
        self.assertFalse(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
