import numpy as np
import pytest

from probstats.algebraic.testing import (
    Kernel,
    hajek_projection_estimate,
    sdl_studentize,
)
from probstats.u_statistics import incomplete_u_statistic


def test_hajek_order_one_is_kernel():
    kernel = Kernel(1, lambda x: x)
    result = hajek_projection_estimate([1.0, 2.0, 4.0], kernel)
    np.testing.assert_allclose(result.values[:, 0], [1, 2, 4])
    np.testing.assert_allclose(result.mean, [7 / 3])
    np.testing.assert_allclose(result.variance, [14 / 9])
    assert result.blocks_per_observation == 1


def test_hajek_order_two_partition_estimator():
    kernel = Kernel(2, lambda x, y: x + y)
    result = hajek_projection_estimate([1.0, 2.0, 3.0, 4.0], kernel, n1=2)
    # For i=0, pair with each other point: mean 4; for i=1: mean 14/3.
    np.testing.assert_allclose(result.values[:, 0], [4.0, 14 / 3])
    assert result.blocks_per_observation == 3


def test_hajek_reproducible_random_partition():
    kernel = Kernel(3, lambda x, y, z: x * y + z)
    a = hajek_projection_estimate(range(8), kernel, n1=4, rng=19, shuffle_blocks=True)
    b = hajek_projection_estimate(range(8), kernel, n1=4, rng=19, shuffle_blocks=True)
    np.testing.assert_allclose(a.values, b.values)


def test_studentization_matches_formula_and_reuses_u_result():
    sample = [1.0, 2.0, 3.0, 4.0, 5.0]
    kernel = Kernel(1, lambda x: x - 2.5)
    u = incomplete_u_statistic(
        sample, kernel, budget=5, rng=3, return_result=True, retain_values=True
    )
    result = sdl_studentize(sample, kernel, budget=5, u_result=u)
    selected = np.array([sample[i[0]] - 2.5 for i in u.subset_indices])
    expected_h = np.mean((selected - float(u.value)) ** 2)
    expected_g = np.mean((np.array(sample) - 2.5 - 0.5) ** 2)
    expected_var = expected_g + expected_h
    assert result.alpha == 1.0
    assert result.kernel_variance[0] == pytest.approx(expected_h)
    assert result.variance[0] == pytest.approx(expected_var)
    assert result.coordinate_statistics[0] == pytest.approx(
        np.sqrt(5) * float(u.value) / np.sqrt(expected_var)
    )
    assert result.statistic == pytest.approx(result.coordinate_statistics[0])


def test_studentization_vector_kernel():
    kernel = Kernel(1, lambda x: (x - 2, 1 - x), dimension=2)
    result = sdl_studentize([1.0, 2.0, 3.0, 4.0], kernel, budget=4, rng=1)
    assert result.coordinate_statistics.shape == (2,)
    assert result.variance.shape == (2,)
    assert result.statistic == pytest.approx(max(result.coordinate_statistics))


def test_studentization_validates_reused_result():
    kernel = Kernel(1, lambda x: x)
    u = incomplete_u_statistic([1, 2, 3], kernel, budget=2, rng=2, return_result=True)
    with pytest.raises(ValueError, match="budget"):
        sdl_studentize([1, 2, 3], kernel, budget=3, u_result=u)


def test_studentization_reuses_retained_kernel_evaluations():
    calls = 0

    def kernel_value(x):
        nonlocal calls
        calls += 1
        return x

    kernel = Kernel(1, kernel_value)
    sample = [1.0, 2.0, 3.0, 4.0]
    u = incomplete_u_statistic(
        sample,
        kernel,
        budget=4,
        selection="fixed",
        rng=2,
        return_result=True,
        retain_values=True,
    )
    calls = 0
    sdl_studentize(sample, kernel, budget=4, u_result=u)
    assert calls == len(sample)
