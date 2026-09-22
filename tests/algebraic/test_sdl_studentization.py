import numpy as np

from probstats.algebraic.testing import sdl_studentize


def test_nonzero_coordinate_with_zero_standard_error_is_infinite():
    result = sdl_studentize(
        [1.0, 2.0, 3.0],
        lambda x: 1.0,
        order=1,
        budget=3,
        rng=np.random.default_rng(4),
    )
    assert result.standard_error[0] == 0
    assert result.coordinate_statistics[0] == np.inf


def test_zero_coordinate_with_zero_standard_error_is_zero():
    result = sdl_studentize(
        [1.0, 2.0, 3.0],
        lambda x: 0.0,
        order=1,
        budget=3,
        rng=np.random.default_rng(4),
    )
    assert result.standard_error[0] == 0
    assert result.coordinate_statistics[0] == 0
