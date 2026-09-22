import sympy as sp

from probstats import (
    Normal,
    RandomVariable,
    StatisticalAssumptions,
    complex_covariance,
    complex_variance,
    delta_method,
    delta_variance,
    expectation,
    independent,
    pseudo_covariance,
    taylor_expectation,
    taylor_variance,
    variance,
)
from probstats.random_variable_algebra import CentralMoment, Expectation, Variance
from probstats.random_variable_complex import (
    ComplexCovariance,
    ComplexVariance,
    PseudoCovariance,
)


def test_taylor_expectation_uses_central_moments_about_mean():
    x = RandomVariable("X")
    result = taylor_expectation(sp.exp(x), order=4)
    mu = Expectation(x)
    expected = sp.exp(mu) * (
        1 + Variance(x) / 2 + CentralMoment(x, 3) / 6 + CentralMoment(x, 4) / 24
    )
    assert sp.simplify(result - expected) == 0


def test_taylor_expectation_is_exact_for_polynomial_with_sufficient_order():
    x = RandomVariable("X")
    a, b = sp.symbols("a b", real=True)
    result = taylor_expectation(a * x**2 + b * x + 3, order=2)
    expected = a * (Variance(x) + Expectation(x) ** 2) + b * Expectation(x) + 3
    assert sp.expand(result - expected) == 0


def test_delta_method_and_second_order_variance():
    x = RandomVariable("X")
    first = delta_method(sp.exp(x))
    assert first.asymptotic_variance == sp.exp(2 * Expectation(x)) * Variance(x)
    assert delta_variance(sp.exp(x)) == first.asymptotic_variance

    second = taylor_variance(x**2, order=2)
    mu = Expectation(x)
    expected = (
        4 * mu**2 * Variance(x)
        + 4 * mu * CentralMoment(x, 3)
        + CentralMoment(x, 4)
        - Variance(x) ** 2
    )
    assert sp.expand(second - expected) == 0


def test_taylor_validation_requires_univariate_random_expression():
    x = RandomVariable("X")
    y = RandomVariable("Y")
    try:
        taylor_expectation(x + y)
    except ValueError as exc:
        assert "variable is required" in str(exc)
    else:
        raise AssertionError("multivariate expression was accepted without variable")


def test_expectation_commutes_with_complex_conjugation():
    z = RandomVariable("Z")
    assert expectation(sp.conjugate(z)) == sp.conjugate(Expectation(z))


def test_complex_covariance_is_sesquilinear_and_hermitian_by_construction():
    x = RandomVariable("X")
    y = RandomVariable("Y")
    a, b = sp.symbols("a b", real=True)
    assert complex_covariance(a * x, b * y) == a * sp.conjugate(b) * ComplexCovariance(
        x, y
    )
    assert complex_covariance(x, x) == ComplexVariance(x)
    assert complex_covariance(y, x) == sp.conjugate(ComplexCovariance(x, y))


def test_complex_variance_scales_by_squared_modulus():
    z = RandomVariable("Z")
    a = sp.Symbol("a", real=True)
    assert complex_variance(a * z) == a * sp.conjugate(a) * ComplexVariance(z)
    assert complex_variance(3 * sp.I * z) == 9 * ComplexVariance(z)


def test_complex_independent_cross_covariances_vanish():
    x = RandomVariable("X")
    y = RandomVariable("Y")
    ctx = StatisticalAssumptions(independent(x, y))
    assert complex_covariance(x, y, assumptions=ctx) == 0
    assert pseudo_covariance(x, y, assumptions=ctx) == 0
    assert complex_variance(x + y, assumptions=ctx) == ComplexVariance(
        x
    ) + ComplexVariance(y)


def test_pseudo_covariance_is_bilinear_and_symmetric():
    x = RandomVariable("X")
    y = RandomVariable("Y")
    a, b = sp.symbols("a b", real=True)
    assert pseudo_covariance(a * x, b * y) == a * b * PseudoCovariance(x, y)
    assert pseudo_covariance(y, x) == pseudo_covariance(x, y)
    assert pseudo_covariance(x) == PseudoCovariance(x, x)


def test_real_variance_scaling_requires_real_coefficient():
    x = RandomVariable("X")
    a = sp.Symbol("a", real=True)
    assert variance(a * x) == a**2 * Variance(x)


def test_real_variance_does_not_apply_complex_scaling_rule():
    x = RandomVariable("X")
    assert variance(sp.I * x) == Variance(sp.I * x)


def test_delta_method_zero_first_order_variance_has_degenerate_limit():
    x = RandomVariable("DeltaZero", distribution=Normal(0, 1))
    result = delta_method(x**2, variable=x)
    assert result.derivative == 0
    assert result.asymptotic_variance == 0
    assert result.limit_distribution == 0
