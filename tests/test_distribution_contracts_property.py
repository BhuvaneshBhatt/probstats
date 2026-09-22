"""Property-based contracts over randomized valid scalar parameters."""

from __future__ import annotations

import math

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings
from hypothesis import strategies as st

from probstats import Bernoulli, Exponential, Normal, StudentT, Uniform
from probstats.distributions import Cauchy, Geometric, Laplace, Logistic, Weibull


def _float(value):
    return float(value.evalf()) if hasattr(value, "evalf") else float(value)


positive = st.floats(
    min_value=0.15, max_value=6.0, allow_nan=False, allow_infinity=False
)
location = st.floats(
    min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False
)
probability = st.floats(
    min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False
)


@st.composite
def native_quantile_continuous(draw):
    family = draw(
        st.sampled_from(
            [
                "normal",
                "cauchy",
                "laplace",
                "logistic",
                "uniform",
                "exponential",
                "weibull",
                "student_t",
            ]
        )
    )
    if family == "normal":
        return Normal(draw(location), draw(positive))
    if family == "cauchy":
        return Cauchy(draw(location), draw(positive))
    if family == "laplace":
        return Laplace(draw(location), draw(positive))
    if family == "logistic":
        return Logistic(draw(location), draw(positive))
    if family == "uniform":
        low = draw(location)
        width = draw(positive)
        return Uniform(low, low + width)
    if family == "exponential":
        return Exponential(draw(positive))
    if family == "weibull":
        return Weibull(draw(positive), draw(positive))
    return StudentT(
        draw(location), draw(positive), draw(st.integers(min_value=2, max_value=30))
    )


@given(native_quantile_continuous(), probability)
@settings(max_examples=80, deadline=None)
def test_randomized_continuous_quantile_cdf_inverse(dist, p):
    q = _float(dist.quantile(p))
    assert _float(dist.pdf(q)) >= -1e-12
    assert math.isclose(_float(dist.cdf(q)), p, rel_tol=3e-7, abs_tol=3e-7)


@given(
    native_quantile_continuous(),
    st.floats(-8, 8, allow_nan=False, allow_infinity=False),
    st.floats(-8, 8, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=80, deadline=None)
def test_randomized_continuous_cdf_is_monotone_and_complements_survival(dist, a, b):
    x, y = sorted((a, b))
    fx, fy = _float(dist.cdf(x)), _float(dist.cdf(y))
    assert fx <= fy + 1e-12
    assert math.isclose(
        fx + _float(dist.survival(x)), 1.0, rel_tol=2e-10, abs_tol=2e-10
    )


@given(
    st.one_of(
        st.builds(
            Bernoulli, st.floats(0.02, 0.98, allow_nan=False, allow_infinity=False)
        ),
        st.builds(
            Geometric, st.floats(0.02, 0.98, allow_nan=False, allow_infinity=False)
        ),
    ),
    probability,
)
@settings(max_examples=60, deadline=None)
def test_randomized_discrete_generalized_inverse(dist, p):
    q = int(dist.quantile(p))
    assert _float(dist.cdf(q)) + 1e-12 >= p
    assert _float(dist.cdf(q - 1)) < p + 1e-12


@given(native_quantile_continuous())
@settings(max_examples=30, deadline=None)
def test_randomized_sample_shape_contract(dist):
    draws = np.asarray(dist.sample(size=(3, 4), rng=12345))
    assert draws.shape == (3, 4)
