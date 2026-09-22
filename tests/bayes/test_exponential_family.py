import sympy as sp

from probstats.distributions import (
    Exponential,
    Gamma,
    InverseGamma,
    LogNormal,
    Normal,
    Poisson,
)


def assert_zero(expr):
    assert sp.simplify(expr) == 0


def test_exponential_definition_matches_closed_form_definition():
    lam, x = sp.symbols("lambda x", positive=True)
    dist = Exponential(lam)
    assert dist.natural_parameters == (-lam,)
    assert dist.sufficient_statistics(x) == (x,)
    assert dist.base_measure(x) == 1
    assert_zero(dist.log_partition + sp.log(lam))
    assert_zero(dist.pdf(x) - lam * sp.exp(-lam * x))
    assert dist.support == sp.Interval(0, sp.oo)


def test_normal_definition_and_canonical_density():
    mu = sp.Symbol("mu", real=True)
    sigma, x = sp.symbols("sigma x", positive=True)
    dist = Normal(mu, sigma)
    assert dist.natural_parameters == (mu / sigma**2, -1 / (2 * sigma**2))
    assert dist.sufficient_statistics(x) == (x, x**2)
    assert dist.base_measure(x) == 1 / sp.sqrt(2 * sp.pi)
    expected = sp.exp(-((x - mu) ** 2) / (2 * sigma**2)) / (sp.sqrt(2 * sp.pi) * sigma)
    assert_zero(dist.pdf(x) - expected)


def test_poisson_definition_and_density():
    lam = sp.Symbol("lambda", positive=True)
    k = sp.Symbol("k", integer=True, nonnegative=True)
    dist = Poisson(lam)
    assert dist.natural_parameters == (sp.log(lam),)
    assert dist.sufficient_statistics(k) == (k,)
    assert dist.base_measure(k) == 1 / sp.factorial(k)
    assert_zero(dist.log_partition - lam)
    assert_zero(dist.pdf(k) - sp.exp(-lam) * lam**k / sp.factorial(k))
    assert dist.support == sp.S.Naturals0


def test_lognormal_definition_and_density():
    mu = sp.Symbol("mu", real=True)
    sigma, x = sp.symbols("sigma x", positive=True)
    dist = LogNormal(mu, sigma)
    assert dist.natural_parameters == (mu / sigma**2, -1 / (2 * sigma**2))
    assert dist.sufficient_statistics(x) == (sp.log(x), sp.log(x) ** 2)
    expected = sp.exp(-((sp.log(x) - mu) ** 2) / (2 * sigma**2)) / (
        x * sigma * sp.sqrt(2 * sp.pi)
    )
    assert_zero(dist.pdf(x) - expected)
    assert dist.support == sp.Interval.open(0, sp.oo)


def test_gamma_definition_and_density():
    shape, scale, x = sp.symbols("shape scale x", positive=True)
    dist = Gamma(shape, scale)
    assert dist.natural_parameters == (shape - 1, -1 / scale)
    assert dist.sufficient_statistics(x) == (sp.log(x), x)
    expected = x ** (shape - 1) * sp.exp(-x / scale) / (sp.gamma(shape) * scale**shape)
    assert_zero(sp.expand_func(dist.pdf(x)) - expected)


def test_inverse_gamma_definition_and_density():
    shape, scale, x = sp.symbols("shape scale x", positive=True)
    dist = InverseGamma(shape, scale)
    assert dist.natural_parameters == (-shape - 1, -scale)
    assert dist.sufficient_statistics(x) == (sp.log(x), 1 / x)
    expected = scale**shape * sp.exp(-scale / x) / (sp.gamma(shape) * x ** (shape + 1))
    assert_zero(sp.expand_func(dist.pdf(x)) - expected)


def test_natural_parameter_regions_are_exposed_for_conjugacy_planner():
    eta1, eta2 = sp.symbols("eta1 eta2", real=True)
    assert Exponential(2).natural_parameter_constraints((eta1,)) == (eta1 < 0)
    assert Normal(0, 1).natural_parameter_constraints((eta1, eta2)) == (eta2 < 0)
    assert Gamma(2, 3).natural_parameter_constraints((eta1, eta2)) == sp.And(
        eta1 > -1, eta2 < 0
    )
    assert InverseGamma(2, 3).natural_parameter_constraints((eta1, eta2)) == sp.And(
        eta1 < -1, eta2 < 0
    )


def test_parameter_constraints_and_symbolic_parameters():
    a, b = sp.symbols("a b", real=True)
    gamma = Gamma(a, b)
    assert gamma.parameters == (a, b)
    assert gamma.free_symbols == {a, b}
    assert gamma.parameter_constraints == sp.And(a > 0, b > 0)


def test_natural_parameter_count_and_dimension_validation():
    dist = Normal(0, 1)
    assert dist.natural_parameters_count == 2
    assert len(dist.natural_parameter_symbols) == 2
    try:
        dist.natural_parameter_constraints((sp.Symbol("eta"),))
    except ValueError as exc:
        assert "Expected 2 natural parameters" in str(exc)
    else:
        raise AssertionError("Expected dimension mismatch to raise ValueError")
