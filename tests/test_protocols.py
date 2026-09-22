import pytest
import sympy as sp

from probstats import (
    Bernoulli,
    Beta,
    Binomial,
    Exponential,
    MultivariateNormal,
    Normal,
    Poisson,
    cdf,
    density,
    expectation,
)
from probstats.distributions import (
    Dirichlet,
    Multinomial,
    Wishart,
)
from probstats.functionals import (
    entropy,
    mean,
    moment,
    quantile,
    survival,
    variance,
)
from probstats.spaces import (
    CountVectorEventSpace,
    MatrixEventSpace,
    MeasureType,
    ScalarEventSpace,
    SimplexEventSpace,
    VectorEventSpace,
)
from probstats.transforms import (
    Transform,
    TransformedDistribution,
)


def test_measure_and_event_spaces():
    assert Normal(0, 1).measure_type is MeasureType.CONTINUOUS
    assert Bernoulli(sp.Rational(1, 3)).measure_type is MeasureType.DISCRETE
    assert Normal(0, 1).event_space == ScalarEventSpace(sp.S.Reals)
    assert MultivariateNormal([0, 0], sp.eye(2)).event_space == VectorEventSpace(
        2, sp.S.Reals
    )
    assert Dirichlet([2, 3, 4]).event_space == SimplexEventSpace(3)
    assert Multinomial(
        4, [sp.Rational(1, 2), sp.Rational(1, 2)]
    ).event_space == CountVectorEventSpace(2, 4)
    assert Wishart(4, sp.eye(2)).event_space == MatrixEventSpace(2, 2, True, True)


def test_pdf_pmf_contract():
    b = Bernoulli(sp.Rational(1, 4))
    assert b.pmf(1) == sp.Rational(1, 4)
    assert density(b, 0) == sp.Rational(3, 4)
    with pytest.raises(TypeError):
        Normal(0, 1).pmf(0)


def test_cdf_survival_quantile_closed_forms():
    p = sp.symbols("p", real=True)
    n = Normal(0, 1)
    assert sp.simplify(cdf(n, 0) - sp.Rational(1, 2)) == 0
    assert sp.simplify(survival(n, 0) - sp.Rational(1, 2)) == 0
    assert sp.simplify(quantile(n, sp.Rational(1, 2))) == 0
    e = Exponential(2)
    assert sp.simplify(cdf(e, 1) - (1 - sp.exp(-2))) == 0
    assert sp.simplify(quantile(e, p) + sp.log(1 - p) / 2) == 0


def test_generic_discrete_cdf():
    b = Binomial(2, sp.Rational(1, 2))
    assert sp.simplify(cdf(b, 1) - sp.Rational(3, 4)) == 0


def test_moments_mean_variance_entropy():
    n = Normal(3, 2)
    assert mean(n) == 3 and variance(n) == 4
    assert sp.simplify(moment(Beta(2, 3), 2) - sp.Rational(1, 5)) == 0
    assert mean(Poisson(5)) == 5 and variance(Poisson(5)) == 5
    assert sp.simplify(entropy(Exponential(2)) - (1 - sp.log(2))) == 0


def test_expectation_generic():
    x = sp.symbols("x", real=True)
    b = Bernoulli(sp.Rational(1, 4))
    assert expectation(b, x**2, variable=x) == sp.Rational(1, 4)
    assert (
        sp.simplify(expectation(Exponential(2), x, variable=x) - sp.Rational(1, 2)) == 0
    )


def test_vector_means_and_covariances():
    mv = MultivariateNormal([1, 2], [[2, 0], [0, 3]])
    assert mean(mv) == sp.ImmutableMatrix([1, 2])
    assert variance(mv) == sp.ImmutableMatrix([[2, 0], [0, 3]])
    d = Dirichlet([2, 3])
    assert mean(d) == sp.ImmutableMatrix([sp.Rational(2, 5), sp.Rational(3, 5)])


def test_monotone_transform():
    x, y = sp.symbols("x y", real=True)
    t = Transform.from_expressions(
        sp.exp(x),
        x,
        sp.log(y),
        y,
        domain=sp.S.Reals,
        codomain=sp.Interval.open(0, sp.oo),
    )
    d = TransformedDistribution(Normal(0, 1), t)
    z = sp.symbols("z", positive=True)
    assert (
        sp.simplify(d.pdf(z) - sp.exp(-(sp.log(z) ** 2) / 2) / (sp.sqrt(2 * sp.pi) * z))
        == 0
    )
    assert sp.simplify(quantile(d, sp.Rational(1, 2)) - 1) == 0
