import numpy as np
import pytest
import sympy as sp

from probstats import (
    Bernoulli,
    Normal,
    expectation,
    probability,
)
from probstats.composition import (
    IndependentDistribution,
    MixtureDistribution,
    ProductDistribution,
    TruncatedDistribution,
)
from probstats.events import ProductEvent
from probstats.solver import EvaluationMethod


def test_probability_accepts_predicates_and_sets():
    x = sp.symbols("x", real=True)
    d = Normal(0, 1)
    assert sp.simplify(probability(d, x > 0, variable=x) - sp.Rational(1, 2)) == 0
    assert sp.simplify(probability(d, sp.Interval(-sp.oo, 0)) - sp.Rational(1, 2)) == 0


def test_discrete_interval_endpoint_semantics():
    d = Bernoulli(sp.Rational(1, 4))
    assert probability(d, sp.FiniteSet(1)) == sp.Rational(1, 4)
    assert probability(d, sp.Interval(0, 1, right_open=True)) == sp.Rational(3, 4)
    assert probability(d, sp.Interval.open(0, 1)) == 0


def test_mixture_probability_is_structural():
    d = MixtureDistribution([Normal(-1, 1), Normal(1, 1)], [sp.Rational(1, 2)] * 2)
    result = probability(d, sp.Interval(-sp.oo, 0), return_result=True)
    assert result.method is EvaluationMethod.STRUCTURAL
    assert sp.simplify(result.value - sp.Rational(1, 2)) == 0


def test_truncated_probability_reuses_base_probability():
    d = TruncatedDistribution(Normal(0, 1), sp.Interval(0, sp.oo))
    assert (
        sp.simplify(probability(d, sp.Interval(0, 1)) - (2 * Normal(0, 1).cdf(1) - 1))
        == 0
    )


def test_product_event_factorizes():
    d = ProductDistribution([Bernoulli(sp.Rational(1, 3)), Normal(0, 1)])
    event = ProductEvent([sp.FiniteSet(1), sp.Interval(0, sp.oo)])
    result = probability(d, event, return_result=True)
    assert result.method is EvaluationMethod.STRUCTURAL
    assert result.value == sp.Rational(1, 6)


def test_iid_product_event_factorizes():
    d = IndependentDistribution(Bernoulli(sp.Rational(1, 2)), 3)
    event = ProductEvent([sp.FiniteSet(1)] * 3)
    assert probability(d, event) == sp.Rational(1, 8)


def test_polynomial_expectation_uses_moments():
    x = sp.symbols("x", real=True)
    d = Normal(0, 2)
    result = expectation(d, 3 * x**2 + 2 * x + 7, variable=x, return_result=True)
    assert result.method is EvaluationMethod.STRUCTURAL
    assert result.value == 19


def test_product_joint_expectation_by_iterated_expectation():
    x, y = sp.symbols("x y", real=True)
    d = ProductDistribution([Normal(2, 1), Bernoulli(sp.Rational(1, 4))])
    assert expectation(d, x * y + y, variables=(x, y)) == sp.Rational(3, 4)


def test_mixture_expectation_uses_distribution_structure():
    x = sp.symbols("x", real=True)
    d = MixtureDistribution(
        [Normal(-2, 1), Normal(4, 1)], [sp.Rational(1, 3), sp.Rational(2, 3)]
    )
    assert sp.simplify(expectation(d, x, variable=x) - 2) == 0


def test_numerical_probability_is_opt_in_and_reports_error():
    # Product distribution with a non-product event has no exact generic route.
    x, y = sp.symbols("x y", real=True)
    d = ProductDistribution([Normal(0, 1), Normal(0, 1)])
    from probstats.functionals import ProbabilityFunctionalError

    with pytest.raises(ProbabilityFunctionalError):
        probability(d, x + y > 0, variable=x)
    # Numerical fallback is tested on a scalar ConditionSet-like predicate that
    # the exact set solver does not simplify reliably.
    z = sp.symbols("z", real=True)
    result = probability(
        Normal(0, 1),
        sp.sin(z) > 0,
        variable=z,
        numerical_fallback=True,
        samples=20_000,
        rng=123,
        return_result=True,
    )
    assert result.method is EvaluationMethod.MONTE_CARLO or result.exact
    if result.method is EvaluationMethod.MONTE_CARLO:
        assert result.standard_error is not None
        assert 0 <= result.value <= 1


def test_numerical_expectation_reproducible():
    # Force MC by using a vector law without structural variables.
    d = IndependentDistribution(Normal(0, 1), 2)

    def f(draws):
        return np.asarray(draws)[:, 0] ** 2

    a = expectation(
        d, f, numerical_fallback=True, samples=5000, rng=77, return_result=True
    )
    b = expectation(
        d, f, numerical_fallback=True, samples=5000, rng=77, return_result=True
    )
    assert a.method is EvaluationMethod.MONTE_CARLO
    assert a.value == b.value
    assert a.standard_error == b.standard_error
    assert abs(a.value - 1.0) < 0.1
