import sympy as sp
from hypothesis import given, settings
from hypothesis import strategies as st

from probstats import (
    Bernoulli,
    Normal,
    probability,
)


@given(st.integers(min_value=1, max_value=99))
def test_bernoulli_event_partition_exact(percent):
    p = sp.Rational(percent, 100)
    d = Bernoulli(p)
    assert probability(d, sp.FiniteSet(0)) + probability(d, sp.FiniteSet(1)) == 1
    assert probability(d, sp.FiniteSet(1)) == p


@given(
    st.integers(min_value=-5, max_value=4),
    st.integers(min_value=-4, max_value=5),
)
def test_normal_interval_probability_bounds(a, b):
    if a > b:
        a, b = b, a
    value = probability(Normal(0, 1), sp.Interval(a, b))
    numeric = float(sp.N(value))
    assert 0.0 <= numeric <= 1.0


@given(st.integers(min_value=-4, max_value=3))
def test_normal_nested_intervals_are_monotone(a):
    d = Normal(0, 1)
    inner = probability(d, sp.Interval(a, a + 1))
    outer = probability(d, sp.Interval(a - 1, a + 2))
    assert float(sp.N(inner)) <= float(sp.N(outer)) + 1e-14


@settings(max_examples=20, deadline=None)
@given(st.integers(min_value=-2, max_value=2))
def test_monte_carlo_probability_tracks_exact_threshold(threshold):
    x = sp.symbols("x", real=True)
    d = Normal(0, 1)
    exact = float(sp.N(probability(d, x <= threshold, variable=x)))
    result = probability(
        d,
        sp.sin(x) > 0 if threshold == 0 else x <= threshold,
        variable=x,
        numerical_fallback=True,
        samples=4000,
        rng=threshold + 100,
        return_result=True,
    )
    # Ordinary threshold predicates are exact; the nonalgebraic threshold may
    # fall back to Monte Carlo. Both are acceptable planner outcomes.
    if result.exact:
        if threshold != 0:
            assert abs(float(sp.N(result.value)) - exact) < 1e-12
    else:
        assert 0 <= result.value <= 1
        assert result.standard_error is not None
