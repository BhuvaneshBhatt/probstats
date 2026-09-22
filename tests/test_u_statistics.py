import numpy as np
import pytest
import sympy as sp

from probstats.u_statistics import UStatisticResult, incomplete_u_statistic, u_statistic


def test_complete_u_statistic_scalar_kernel():
    value = u_statistic([1, 2, 4], lambda x, y: x * y, order=2)
    assert value == pytest.approx((2 + 4 + 8) / 3)


def test_complete_u_statistic_vector_kernel():
    value = u_statistic([1, 2, 3], lambda x, y: np.array([x + y, x * y]), order=2)
    assert np.allclose(value, [4, 11 / 3])


def test_complete_u_statistic_result_metadata():
    result = u_statistic([1, 2, 3, 4], lambda x, y: x + y, order=2, return_result=True)
    assert isinstance(result, UStatisticResult)
    assert result.total_subsets == 6
    assert result.evaluated_subsets == 6
    assert result.randomized is False


def test_incomplete_fixed_budget_is_reproducible():
    first = incomplete_u_statistic(
        range(8),
        lambda x, y: x * y,
        order=2,
        budget=7,
        rng=123,
        selection="fixed",
        return_result=True,
    )
    second = incomplete_u_statistic(
        range(8),
        lambda x, y: x * y,
        order=2,
        budget=7,
        rng=123,
        selection="fixed",
        return_result=True,
    )
    assert first == second
    assert first.evaluated_subsets == 7
    assert first.requested_budget == 7


def test_bernoulli_budget_equal_to_all_subsets_matches_complete_statistic():
    sample = [1, 2, 3, 4]

    def kernel(x, y):
        return (x - y) ** 2

    complete = u_statistic(sample, kernel, order=2)
    incomplete = incomplete_u_statistic(
        sample, kernel, order=2, budget=6, rng=7, return_result=True
    )
    assert incomplete.value == pytest.approx(complete)
    assert incomplete.evaluated_subsets == 6


def test_incomplete_validation():
    with pytest.raises(ValueError, match="budget"):
        incomplete_u_statistic([1, 2, 3], lambda x, y: 0, order=2, budget=4)
    with pytest.raises(ValueError, match="selection"):
        incomplete_u_statistic(
            [1, 2, 3], lambda x, y: 0, order=2, budget=1, selection="other"
        )
    with pytest.raises(ValueError, match="sample size"):
        u_statistic([1], lambda x, y: 0, order=2)


def test_incomplete_result_can_retain_kernel_values():
    result = incomplete_u_statistic(
        [1, 2, 3, 4],
        lambda x: x * x,
        order=1,
        budget=4,
        selection="fixed",
        rng=4,
        return_result=True,
        retain_values=True,
    )
    assert result.kernel_values is not None
    expected = tuple((i + 1) * (i + 1) for (i,) in result.subset_indices)
    assert result.kernel_values == expected
    assert result.value == sum(expected) / len(expected)


def test_empty_bernoulli_selection_has_specific_error():
    from probstats.u_statistics import EmptySubsetSelectionError

    with pytest.raises(EmptySubsetSelectionError):
        incomplete_u_statistic([1, 2, 3, 4, 5], lambda x: x, order=1, budget=1, rng=2)


def test_declared_kernel_dimension_rejects_wrong_vector_shape():
    from probstats.algebraic.testing import Kernel

    kernel = Kernel(order=1, dimension=2, function=lambda x: [x])
    with pytest.raises(ValueError, match="declared dimension 2"):
        u_statistic([1, 2], kernel)


def test_streaming_rejects_shape_change():
    def kernel(x):
        return [x] if x == 1 else [x, x]

    with pytest.raises(ValueError, match="shape changed"):
        u_statistic([1, 2], kernel, order=1)


def test_streaming_preserves_exact_sympy_scalar_arithmetic():
    x = sp.Symbol("x")
    result = u_statistic([x, x + 1], lambda value: value, order=1)
    assert result == x + sp.Rational(1, 2)


def test_generic_u_statistic_uses_caller_supplied_argument_order():
    # Symmetry is a mathematical precondition, not something that can be
    # established reliably from finite runtime probes.
    value = u_statistic([1, 2, 4], lambda x, y: x - y, order=2)
    assert value == pytest.approx((-1 - 3 - 2) / 3)
