"""Adversarial contracts for symbolic states that SymPy cannot decide."""

import pytest
import sympy as sp

from probstats.algebraic import (
    ContingencyTable,
    ProbabilityTensor,
    categorical_multi_view_moment,
    moment_tensor,
    recover_multiview_mixture,
    tensor_independent,
)


def test_probability_tensor_preserves_unknown_normalization_and_sign_state():
    x, y = sp.symbols("x y")
    tensor = ProbabilityTensor([[x, y]])

    assert tensor.is_normalized() is None
    assert tensor.is_nonnegative() is None


def test_symbolic_normalized_tensor_can_still_have_unknown_nonnegativity():
    x = sp.Symbol("x")
    tensor = ProbabilityTensor([[x, 1 - x]])

    assert tensor.is_normalized() is True
    assert tensor.is_nonnegative() is None


def test_tensor_independent_returns_none_for_undecidable_segre_residual():
    x = sp.Symbol("x")
    tensor = ProbabilityTensor([[x, 0], [0, 1 - x]])
    result = tensor_independent(tensor)

    assert result.independent is None
    assert any(residual.is_zero is None for residual in result.residuals)


def test_probability_mixture_recovery_rejects_unknown_normalization():
    x, y = sp.symbols("x y", nonnegative=True)
    with pytest.raises(ValueError, match="sum to one"):
        recover_multiview_mixture([[x, y], [0, 0]], 1, method="exact")


def test_mixture_recovery_rejects_unknown_sign():
    x = sp.Symbol("x")
    with pytest.raises(ValueError, match="certified nonnegative"):
        recover_multiview_mixture([[x, 1 - x], [0, 0]], 1, method="exact")


def test_empirical_weights_reject_unknown_sign_and_unknown_positive_total():
    w = sp.Symbol("w")
    with pytest.raises(ValueError, match="certified nonnegative"):
        moment_tensor(((1,), (2,)), 1, weights=(w, 1 - w))

    a, b = sp.symbols("a b", nonnegative=True)
    with pytest.raises(ValueError, match="certified positive total"):
        moment_tensor(((1,), (2,)), 1, weights=(a, b))


def test_symbolic_cardinality_and_category_labels_are_not_assumed_integral():
    n = sp.Symbol("n", integer=True, positive=True)
    with pytest.raises(ValueError, match="cardinality"):
        categorical_multi_view_moment(((0, 1), (0, 1)), cardinalities=(n, 2))

    label = sp.Symbol("k", integer=True, nonnegative=True)
    with pytest.raises(ValueError, match="categorical label"):
        categorical_multi_view_moment(((0, label), (0, 1)), cardinalities=(2, 2))


def test_contingency_table_requires_decidable_nonnegative_integer_counts():
    n = sp.Symbol("n", integer=True, nonnegative=True)
    with pytest.raises((TypeError, ValueError)):
        ContingencyTable([[n, 1], [2, 3]])
