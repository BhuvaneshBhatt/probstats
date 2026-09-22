import numpy as np
import pytest
import sympy as sp

from probstats.bayes.core.backend import (
    as_float_array,
    as_symbol,
    numeric_matrix,
    numeric_vector,
)


def test_as_symbol_preserves_existing_symbol():
    symbol = sp.Symbol("x", positive=True)
    assert as_symbol(symbol) is symbol


def test_as_symbol_builds_real_symbol():
    symbol = as_symbol("theta", real=True)
    assert symbol.name == "theta"
    assert symbol.is_real is True


def test_numeric_vector_and_matrix_conventions():
    vector = numeric_vector([1, 2.5, 3])
    matrix = numeric_matrix([[1, 2], [3, 4]])
    assert vector.dtype == np.float64
    assert vector.shape == (3,)
    assert matrix.shape == (2, 2)


def test_arrays_reject_wrong_dimension_and_nonfinite_values():
    with pytest.raises(ValueError, match="ndim"):
        as_float_array([[1, 2]], ndim=1)
    with pytest.raises(ValueError, match="finite"):
        numeric_vector([1, np.inf])
