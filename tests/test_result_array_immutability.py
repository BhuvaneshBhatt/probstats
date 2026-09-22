import numpy as np
import pytest

from probstats.algebraic.testing.bootstrap import SDLBootstrapResult
from probstats.algebraic.testing.diagnostics import SDLCalibrationResult


def test_bootstrap_result_owns_and_freezes_arrays():
    source = np.array([1.0, 2.0])
    result = SDLBootstrapResult(source, 0.5, 2, studentization=None)
    source[0] = 99.0
    assert result.statistics[0] == 1.0
    with pytest.raises(ValueError):
        result.statistics[0] = 3.0


def test_calibration_result_owns_and_freezes_arrays():
    p_values = np.array([0.1, 0.2])
    result = SDLCalibrationResult(
        alpha=0.05,
        repetitions=2,
        rejection_rate=0.0,
        monte_carlo_standard_error=0.0,
        p_values=p_values,
        statistics=np.array([1.0, 2.0]),
        realized_budgets=np.array([3, 4]),
        requested_budget=4,
        kernel_orders=np.array([2, 2]),
        projection_sizes=np.array([5, 5]),
        seed=1,
    )
    p_values[0] = 0.9
    assert result.p_values[0] == 0.1
    for array in (
        result.p_values,
        result.statistics,
        result.realized_budgets,
        result.kernel_orders,
        result.projection_sizes,
    ):
        assert not array.flags.writeable
