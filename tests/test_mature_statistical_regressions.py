"""Independent numerical and edge-case regressions for mature statistics APIs."""

import numpy as np
import pytest
import sympy as sp
from hypothesis import given
from hypothesis import strategies as st

from probstats import Binomial, Exponential, Gamma, Normal, Poisson
from probstats.descriptive import (
    covariance,
    interquartile_range,
    skewness,
    trimmed_mean,
    variance,
)
from probstats.distributions.standard import Geometric, Pareto, Weibull
from probstats.functionals import density, quantile


def test_covariance_numeric_regression():
    assert covariance([1.5, 3, 5, 10], [2, 1.25, 15, 8]) == pytest.approx(
        11.260416666666666
    )


def test_bias_uncorrected_skewness_numeric_regression():
    data = [1.21, 3.4, 2, 4.66, 1.5, 5.61, 7.22]
    assert skewness(data, bias=True) == pytest.approx(0.37496055326162164)


@pytest.mark.parametrize(
    ("data", "proportion", "expected"),
    [
        ([-10, 1, 1, 1, 1, 20], 0.2, 1.0),
        ([5, 10, 4, 25, 2, 1], 0.2, 5.25),
        ([1, 5, 2, 6, 10, 100000, 5, 4, -200, 5], 0.1, 4.75),
        (list(range(1, 11)), 0.0, 5.5),
    ],
)
def test_trimmed_mean_regressions(data, proportion, expected):
    assert trimmed_mean(data, proportion) == pytest.approx(expected)


def test_interquartile_range_numeric_regression():
    # probstats uses NumPy's linear empirical-quantile convention.
    assert interquartile_range([1, 3, 4, 2, 5, 6]) == pytest.approx(2.5)


@pytest.mark.parametrize(
    ("distribution", "value"),
    [
        (Binomial(5, 0), 0),
        (Binomial(5, 1), 5),
        (Binomial(0, sp.Rational(2, 5)), 0),
    ],
)
def test_binomial_degenerate_boundary_mass_is_one(distribution, value):
    assert density(distribution, value) == 1


@pytest.mark.parametrize(
    ("distribution", "value"),
    [
        (Binomial(5, 0), 1),
        (Binomial(5, 1), 4),
    ],
)
def test_binomial_degenerate_boundary_mass_is_zero_off_atom(distribution, value):
    assert density(distribution, value) == 0


def test_nested_array_density_preserves_shape_and_values():
    exponential = np.asarray(density(Exponential(1.1), [[0.2, 0.3, 0.1]]), dtype=float)
    poisson = np.asarray(density(Poisson(0.3), [[1, 3, 2]]), dtype=float)
    gamma = np.asarray(density(Gamma(1.2, 2.1), [[0.3, 1.2]]), dtype=float)
    normal = np.asarray(density(Normal(0, 1), [[1, 2, 3, 4, 5]]), dtype=float)

    assert exponential.shape == (1, 3)
    assert np.allclose(
        exponential, [[0.8827706777587264, 0.7908161067751188, 0.9854175488261812]]
    )
    assert np.allclose(
        poisson, [[0.22224546620451535, 0.0033336819930677303, 0.0333368199306773]]
    )
    assert np.allclose(gamma, [[0.3046467326137857, 0.2618679306656224]])
    assert normal.shape == (1, 5)


@pytest.mark.parametrize(
    "distribution",
    [Geometric(sp.Rational(1, 2)), Weibull(1, 1), Pareto(2, 1)],
)
def test_unbounded_upper_quantile_is_positive_infinity(distribution):
    assert quantile(distribution, 1) == sp.oo


@given(
    st.lists(
        st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=30,
    )
)
def test_sample_variance_equals_self_covariance(data):
    assert variance(data) == pytest.approx(covariance(data, data))


@given(
    st.lists(
        st.floats(-1e4, 1e4, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=30,
    ),
    st.floats(-100, 100, allow_nan=False, allow_infinity=False),
)
def test_covariance_is_translation_invariant(data, shift):
    shifted = [x + shift for x in data]
    assert covariance(shifted, shifted) == pytest.approx(
        covariance(data, data), rel=1e-10, abs=1e-10
    )
