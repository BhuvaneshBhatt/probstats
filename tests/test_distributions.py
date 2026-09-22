import pytest
import sympy as sp

from probstats import (
    Bernoulli,
    Beta,
    Binomial,
    Exponential,
    Gamma,
    MultivariateNormal,
    Normal,
    Poisson,
)
from probstats.distributions import (
    Categorical,
    Dirichlet,
    InverseWishart,
    Multinomial,
    Wishart,
)


def test_existing_api_shapes():
    assert Normal(0, 2).support == sp.S.Reals
    assert Exponential(2).support == sp.Interval(0, sp.oo)
    assert Gamma(2, 3).parameters == (2, 3)
    assert sp.simplify(Poisson(2).pdf(3) - sp.Rational(4, 3) * sp.exp(-2)) == 0


def test_bernoulli_and_binomial():
    p = sp.symbols("p", positive=True)
    b = Bernoulli(p)
    assert b.support == sp.FiniteSet(0, 1)
    assert b.pdf(0) == 1 - p and b.pdf(1) == p and b.pdf(2) == 0
    assert Binomial(4, p).support == sp.FiniteSet(0, 1, 2, 3, 4)


def test_beta_formula():
    x = sp.symbols("x", positive=True)
    d = Beta(2, 3)
    assert d.support == sp.Interval(0, 1)
    assert sp.simplify(d.canonical_pdf(x) - x * (1 - x) ** 2 / sp.beta(2, 3)) == 0
    assert d.pdf(sp.Rational(1, 2)) == d.canonical_pdf(sp.Rational(1, 2))
    assert d.pdf(sp.Rational(6, 5)) == 0


def test_categorical_multinomial_dirichlet():
    c = Categorical([sp.Rational(1, 4), sp.Rational(3, 4)])
    assert c.pdf(0) == sp.Rational(1, 4) and c.pdf(1) == sp.Rational(3, 4)
    m = Multinomial(3, [sp.Rational(1, 4), sp.Rational(3, 4)])
    assert (
        sp.simplify(m.pdf([1, 2]) - 3 * sp.Rational(1, 4) * sp.Rational(3, 4) ** 2) == 0
    )
    d = Dirichlet([2, 3])
    x, y = sp.symbols("x y", positive=True)
    assert (
        sp.simplify(
            d.pdf([x, y]) - x * y**2 / (sp.gamma(2) * sp.gamma(3) / sp.gamma(5))
        )
        == 0
    )


def test_multivariate_normal():
    d = MultivariateNormal([0, 0], sp.eye(2))
    assert d.dimension == 2
    assert sp.simplify(d.pdf([0, 0]) - 1 / (2 * sp.pi)) == 0


def test_wishart_inverse_wishart_univariate_reduction():
    x = sp.symbols("x", positive=True)
    w = Wishart(4, [[2]])
    iw = InverseWishart(4, [[2]])
    assert w.dimension == iw.dimension == 1
    assert w.logpdf([[x]]).has(x)
    assert iw.logpdf([[x]]).has(x)


def test_invalid_parameters():
    with pytest.raises(ValueError):
        Bernoulli(2)
    with pytest.raises(ValueError):
        Beta(-1, 2)
    with pytest.raises(ValueError):
        Categorical([sp.Rational(1, 3), sp.Rational(1, 3)])
    with pytest.raises(ValueError):
        MultivariateNormal([0, 0], [[1, 2], [0, 1]])
    with pytest.raises(ValueError):
        Wishart(1, sp.eye(3))


def test_symbolic_parameter_constraints_remain_symbolic_when_unresolved():
    p = sp.Symbol("p", real=True)
    distribution = Bernoulli(p)
    assert distribution.p == p
    assert distribution.parameter_constraints == sp.And(p >= 0, p <= 1)
