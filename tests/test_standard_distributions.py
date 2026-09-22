import numpy as np
import pytest
import sympy as sp

from probstats import (
    Beta,
    Binomial,
    Uniform,
    cdf,
)
from probstats.distributions import (
    BetaBinomial,
    Cauchy,
    ChiSquared,
    Dirichlet,
    DirichletMultinomial,
    Geometric,
    HalfCauchy,
    HalfNormal,
    Laplace,
    Logistic,
    Multinomial,
    NegativeBinomial,
    Pareto,
    Weibull,
)
from probstats.functionals import (
    mean,
    quantile,
    variance,
)


def test_uniform_protocol():
    d = Uniform(-2, 4)
    assert sp.simplify(d.pdf(0) - sp.Rational(1, 6)) == 0
    assert cdf(d, -3) == 0
    assert cdf(d, 4) == 1
    assert mean(d) == 1
    assert variance(d) == 3
    assert quantile(d, sp.Rational(1, 2)) == 1


def test_geometric_convention_and_boundary():
    d = Geometric(sp.Rational(1, 4))
    assert d.pmf(1) == sp.Rational(1, 4)
    assert sp.simplify(d.pmf(3) - sp.Rational(9, 64)) == 0
    assert mean(d) == 4
    assert variance(d) == 12
    assert Geometric(1).quantile(sp.Rational(3, 4)) == 1


def test_negative_binomial_convention():
    d = NegativeBinomial(2, sp.Rational(1, 2))
    assert sp.simplify(d.pmf(0) - sp.Rational(1, 4)) == 0
    assert sp.simplify(d.pmf(1) - sp.Rational(1, 4)) == 0
    assert mean(d) == 2
    assert variance(d) == 4


def test_positive_continuous_closed_forms():
    assert mean(ChiSquared(7)) == 7
    assert variance(ChiSquared(7)) == 14
    assert sp.simplify(mean(Weibull(1, 3)) - 3) == 0
    assert sp.simplify(Weibull(2, 5).cdf(5) - (1 - sp.E**-1)) == 0
    assert Pareto(3, 2).support == sp.Interval(2, sp.oo)
    assert mean(Pareto(3, 2)) == 3
    assert variance(Pareto(3, 2)) == 3


def test_half_distributions():
    h = HalfNormal(2)
    assert h.cdf(0) == 0
    assert (
        sp.simplify(
            h.quantile(sp.Rational(1, 2))
            - 2 * sp.sqrt(2) * sp.erfinv(sp.Rational(1, 2))
        )
        == 0
    )
    hc = HalfCauchy(3)
    assert hc.cdf(3) == sp.Rational(1, 2)
    assert mean(hc) == sp.oo


def test_robust_location_scale_families():
    assert Cauchy(2, 3).cdf(2) == sp.Rational(1, 2)
    assert Cauchy().mean_value is sp.nan
    assert Laplace(2, 3).mean_value == 2
    assert Laplace(2, 3).variance_value == 18
    assert Logistic(2, 3).cdf(2) == sp.Rational(1, 2)
    assert Logistic(2, 3).variance_value == 3 * sp.pi**2


def test_beta_binomial_compound_structure():
    d = BetaBinomial(4, 2, 3)
    assert d.mixing_distribution == Beta(2, 3)
    assert d.conditional_distribution(sp.Symbol("p")) == Binomial(4, sp.Symbol("p"))
    assert sp.simplify(sum(d.pmf(k) for k in range(5)) - 1) == 0
    assert mean(d) == sp.Rational(8, 5)


def test_dirichlet_multinomial_compound_structure_and_covariance():
    d = DirichletMultinomial(5, [2, 3])
    assert d.mixing_distribution == Dirichlet([2, 3])
    p1, p2 = sp.symbols("p1 p2", nonnegative=True)
    assert d.conditional_distribution([p1, p2]) == Multinomial(5, [p1, p2])
    assert d.pmf([2, 3]) > 0
    assert d.mean_value == sp.ImmutableMatrix([2, 3])
    cov = d.variance_value
    assert cov.shape == (2, 2)
    assert cov[0, 1] < 0


def test_invalid_parameters_rejected():
    constructors = [
        lambda: Uniform(1, 1),
        lambda: Geometric(0),
        lambda: NegativeBinomial(0, 0.5),
        lambda: ChiSquared(0),
        lambda: Weibull(-1),
        lambda: Pareto(2, 0),
        lambda: HalfNormal(0),
        lambda: HalfCauchy(-1),
        lambda: Cauchy(0, 0),
        lambda: Laplace(0, -1),
        lambda: Logistic(0, 0),
        lambda: BetaBinomial(-1, 2, 3),
        lambda: DirichletMultinomial(2, [1, -1]),
    ]
    for constructor in constructors:
        with pytest.raises(ValueError):
            constructor()


def test_sampling_reproducible_and_shapes():
    families = [
        Uniform(-1, 1),
        Geometric(0.3),
        NegativeBinomial(2, 0.4),
        ChiSquared(3),
        Weibull(2, 3),
        Pareto(4, 2),
        HalfNormal(2),
        HalfCauchy(2),
        Cauchy(),
        Laplace(),
        Logistic(),
        BetaBinomial(8, 2, 3),
    ]
    for d in families:
        a = np.asarray(d.sample(12, rng=123))
        b = np.asarray(d.sample(12, rng=123))
        assert a.shape == (12,)
        assert np.array_equal(a, b)
    dm = DirichletMultinomial(7, [1, 2, 3])
    draws = dm.sample((4, 5), rng=42)
    assert draws.shape == (4, 5, 3)
    assert np.all(draws.sum(axis=-1) == 7)


def test_parameter_spaces_are_structured_and_usable():
    a, b = sp.symbols("a b", real=True)
    u = Uniform(a, b)
    assert sp.simplify(u.parameter_space_constraints ^ (b > a)) is sp.false
    p = sp.symbols("p", real=True)
    g = Geometric(p)
    assert (
        sp.simplify(g.parameter_space_constraints ^ sp.And(p > 0, p <= 1)) is sp.false
    )
    alphas = sp.symbols("a0:3", positive=True)
    dm = DirichletMultinomial(4, alphas)
    assert dm.parameter_space.names == ("n", "concentration")
    assert dm.parameter_space_constraints is sp.true


def test_weibull_entropy_and_halfnormal_moments_are_exact():
    # Exponential(scale) is Weibull(shape=1), whose entropy is 1 + log(scale).
    assert sp.simplify(Weibull(1, 3).entropy() - (1 + sp.log(3))) == 0
    h = HalfNormal(2)
    assert sp.simplify(h.moment(2) - 4) == 0
    assert not h.moment(2).has(sp.Float)
