import numpy as np
import sympy as sp

from probstats import Normal, cdf, density


def test_scalar_functional_array_shape():
    values = np.array([[0, 1], [sp.Rational(1, 2), -1]], dtype=object)
    dist = Normal(0, 1)
    pdf = density(dist, values)
    probabilities = cdf(dist, values)
    assert pdf.shape == values.shape
    assert probabilities.shape == values.shape
    assert pdf.dtype == object
    assert probabilities.dtype == object
    assert sp.simplify(pdf[0, 0] - 1 / sp.sqrt(2 * sp.pi)) == 0
    assert probabilities[0, 0] == sp.Rational(1, 2)
