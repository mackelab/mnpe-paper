"""Masked estimator classes for the two training and ordering ablations.

Backs Appendix B.2, the joint-versus-separate training comparison, and Appendix
B.6 with Table B-3, the forward-versus-swapped ordering of the discrete head.

These live in the package rather than beside the study scripts because **pickle
resolves classes by module path**. Both ablations cache partial results, and a
part file written under one import path cannot be loaded from another, so
keeping these classes in a script would strand every cached run the first time
the script moved.
"""

from typing import Optional

import torch
from sbi.neural_nets.estimators import MixedDensityEstimator
from sbi.neural_nets.estimators.categorical_net import CategoricalMADE
from torch import Tensor


class HeadMaskedMDE(MixedDensityEstimator):
    """MixedDensityEstimator whose training loss is restricted to one head.

    `_train_part` selects the term:
      "both"        the shipped joint loss, -(disc + cont)
      "discrete"    -disc, so the continuous head receives no gradient and,
                    crucially, does not enter the validation loss
      "continuous"  -cont, obtained as -(joint - disc) so it reuses the
                    estimator's own log_prob rather than duplicating internals

    Restricting the loss is what decouples *early stopping and gradient
    clipping*: with only one head's term present, the validation loss and the
    global gradient norm both see that head alone. Freezing the other head's
    parameters (done separately, via requires_grad) only decouples the
    gradients, which the sum already does.
    """

    _train_part = "both"

    def _disc_log_prob(self, input, condition):
        inp, cond, batch_dim = self._broadcast_and_align(input, condition)
        sample_dim = inp.shape[0]
        n_disc = self.discrete_net.net.num_variables
        lp = self.discrete_net.log_prob(input=inp[..., -n_disc:], condition=cond)
        return lp.reshape(sample_dim, batch_dim)

    def loss(self, input, condition, **kwargs):
        inp3 = input.unsqueeze(0)
        if self._train_part == "both":
            return -self.log_prob(inp3, condition)[0]
        disc = self._disc_log_prob(inp3, condition)[0]
        if self._train_part == "discrete":
            return -disc
        # `disc` is deliberately left attached to the graph: the two forward
        # passes over the discrete net are bitwise identical, so its gradient
        # contribution cancels against the one inside `log_prob` instead of
        # leaking into the discrete parameters (which a `.detach()` would cause).
        return -(self.log_prob(inp3, condition)[0] - disc)


class PermutedCategoricalMADE(CategoricalMADE):
    """Relabels which physical variable occupies which autoregressive slot.

    The MADE's own ordering is fixed by its masks, and `random_mask` randomizes
    rather than permutes, so there is no ordering argument to set. Permuting the
    input on the way in and the samples on the way out is what changes the
    factorization: with ``perm = [1, 0]`` the net fits q(c2|x) q(c1|c2,x).

    ``perm[i]`` is the physical variable that occupies slot ``i``. Everything
    outside this class -- theta, the prior, the continuous head's conditioning,
    the samples handed back to the caller -- stays in physical order, so the
    continuous head trained under the forward ordering remains reusable.

    `perm` and `inv_perm` are deliberately plain attributes rather than buffers:
    a buffer would enter `state_dict` and break interchange with the unpermuted
    class, which reusing Appendix B.2's cached forward-order heads depends on.
    """

    perm: Optional[Tensor] = None
    inv_perm: Optional[Tensor] = None

    def log_prob(self, input: Tensor, condition: Optional[Tensor] = None) -> Tensor:
        return super().log_prob(input[..., self.perm], condition)

    def sample(
        self, sample_shape: torch.Size, context: Optional[Tensor] = None
    ) -> Tensor:
        # `context`, not `condition`: that is CategoricalMADE.sample's signature,
        # and CategoricalMassEstimator.sample forwards it positionally.
        return super().sample(sample_shape, context)[..., self.inv_perm]


def apply_permutation(made: CategoricalMADE, perm) -> CategoricalMADE:
    """Re-type `made` as permuted and attach the slot assignment.

    Asserts the two conditions that make a slot permutation safe to apply to the
    input alone: `num_categories_per_var` and `mask` are indexed by slot, and
    `_map_values_to_indices` uses a per-slot `values_lookup`. With differing
    class counts or differing supports, those would have to be permuted too.
    """
    perm = torch.as_tensor(perm, dtype=torch.long)
    counts = made.num_categories_per_var
    assert int(counts.min()) == int(counts.max()), (
        f"permuting the input alone assumes equal cardinalities, got {counts.tolist()}"
    )
    if made._needs_mapping:
        lookup = made.values_lookup
        assert bool((lookup == lookup[0]).all()), (
            "permuting the input alone assumes identical per-variable value "
            f"lookups, got {lookup.tolist()}"
        )
    made.__class__ = PermutedCategoricalMADE
    made.perm = perm
    made.inv_perm = torch.argsort(perm)
    return made


def enumerate_discrete_joint(made: CategoricalMADE, x_o: Tensor, n_categories: int):
    """Exact q(c1, c2 | x_o) over the physical grid, via the net's own log_prob.

    Routes through `PermutedCategoricalMADE.log_prob`, so it is correct for both
    orderings. `metrics.exact_joint_discrete` cannot be used for the
    swapped arm: it calls `made.forward` and labels slot 0 as c1, which is false
    once the slots are permuted.

    Returns the 25-vector in the same (c1 outer, c2 inner) layout as the
    reference's `configuration_probabilities`.
    """
    grid = torch.tensor(
        [[c1, c2] for c1 in range(n_categories) for c2 in range(n_categories)],
        dtype=torch.float32,
    )
    condition = x_o.unsqueeze(0).expand(len(grid), -1)
    with torch.no_grad():
        log_probs = made.log_prob(grid, condition=condition)
    return log_probs.exp().numpy()


def autoregressive_structure(made: CategoricalMADE, n_categories: int, x_o: Tensor):
    """Which physical variable is unconditioned, read off the network itself.

    The slot-0 conditional sees only x, so its logits must be invariant to the
    value in slot 1; the slot-1 conditional must depend on slot 0. Mapping that
    back through `perm` says which *physical* variable carries the first factor,
    which is the semantic claim the permutation is supposed to change -- not
    something the identity check can establish.
    """
    perm = getattr(made, "perm", None)
    perm = torch.arange(made.num_variables) if perm is None else perm

    condition = x_o.unsqueeze(0).expand(n_categories, -1)
    # Vary slot 1 while slot 0 is held at 0, and vice versa.
    vary_slot1 = torch.zeros(n_categories, made.num_variables)
    vary_slot1[:, 1] = torch.arange(n_categories, dtype=torch.float32)
    vary_slot0 = torch.zeros(n_categories, made.num_variables)
    vary_slot0[:, 0] = torch.arange(n_categories, dtype=torch.float32)

    with torch.no_grad():
        logits_a = made.forward(vary_slot1, condition=condition)
        logits_b = made.forward(vary_slot0, condition=condition)
    logits_a = logits_a.reshape(n_categories, made.num_variables, -1)
    logits_b = logits_b.reshape(n_categories, made.num_variables, -1)

    return {
        # Slot 0's logits must not move when slot 1's value changes.
        "slot0_spread_when_slot1_varies": float(
            (logits_a[:, 0, :] - logits_a[0, 0, :]).abs().max()
        ),
        # Slot 1's logits must move when slot 0's value changes.
        "slot1_spread_when_slot0_varies": float(
            (logits_b[:, 1, :] - logits_b[0, 1, :]).abs().max()
        ),
        "physical_variable_in_slot0": int(perm[0]),
    }
