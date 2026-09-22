import sympy as sp
from hypothesis import given
from hypothesis import strategies as st

from probstats import (
    RandomVariable,
    StatisticalAssumptions,
    conditional_expectation,
    conditional_variance,
    correlation,
    independent,
    moment_generating_function,
    skewness,
    variance,
)
from probstats.random_variable_algebra import Variance
from probstats.random_variable_conditional import ConditionalExpectation

X = RandomVariable("X")
Y = RandomVariable("Y")
Z = RandomVariable("Z")


@given(st.integers(min_value=-7, max_value=7).filter(lambda value: value != 0))
def test_affine_variance_metamorphic_scale(a):
    coefficient = sp.Integer(a)
    assert variance(coefficient * X + 11) == coefficient**2 * Variance(X)


@given(st.integers(min_value=1, max_value=7), st.integers(min_value=-5, max_value=5))
def test_positive_affine_standardized_statistics_are_invariant(a, b):
    expression = sp.Integer(a) * X + sp.Integer(b)
    assert correlation(expression, X) == 1
    assert skewness(expression) == skewness(X)


@given(st.integers(min_value=-5, max_value=5))
def test_deterministic_mgf_matches_exponential_definition(value):
    t = sp.Symbol("t", real=True)
    constant = sp.Integer(value)
    assert moment_generating_function(constant, t) == sp.exp(constant * t)


def test_same_problem_with_and_without_sufficient_joint_independence():
    weak = StatisticalAssumptions(independent(X, Y), independent(X, Z))
    strong = StatisticalAssumptions(
        independent(X, Y), independent(X, Z), independent(Y, Z)
    )
    assert conditional_expectation(
        X, given=(Y, Z), assumptions=weak
    ) == ConditionalExpectation(X, sp.Tuple(Y, Z))
    # Pairwise relations alone remain insufficient even when all three pairs are present.
    assert conditional_expectation(
        X, given=(Y, Z), assumptions=strong
    ) == ConditionalExpectation(X, sp.Tuple(Y, Z))


def test_measurable_factor_conditional_variance_oracle():
    assert conditional_variance(Y * X, given=Y) == Y**2 * conditional_variance(
        X, given=Y
    )
