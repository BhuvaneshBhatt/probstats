import sympy as sp

from probstats import (
    Bernoulli,
    JointDistribution,
    MultivariateNormal,
    Uniform,
)
from probstats.joint import (
    ClaytonCopula,
    CopulaDistribution,
    FrankCopula,
    GumbelCopula,
    IndependenceCopula,
    PushforwardDistribution,
    dependent_transform,
)
from probstats.spaces import MeasureType


def test_partial_generic_conditional_marginalizes_nuisance_coordinate():
    x, y, z = sp.symbols("x y z", real=True)
    density = sp.exp(-x) * sp.exp(-y) * sp.exp(-z)
    joint = JointDistribution((x, y, z), density, (sp.Interval(0, sp.oo),) * 3)
    cond = joint.conditional((0,), {1: 2})
    assert sp.simplify(cond.pdf(x) - sp.exp(-x)) == 0


def test_discrete_independence_copula_uses_cdf_differences():
    d = CopulaDistribution(
        (Bernoulli(sp.Rational(1, 3)), Bernoulli(sp.Rational(1, 2))),
        IndependenceCopula(2),
    )
    assert d.measure_type is MeasureType.DISCRETE
    assert sp.simplify(d.pmf((1, 1)) - sp.Rational(1, 6)) == 0
    assert sp.simplify(d.pmf((0, 1)) - sp.Rational(1, 3)) == 0


def test_mixed_copula_mass_density():
    d = CopulaDistribution(
        (Uniform(0, 1), Bernoulli(sp.Rational(1, 4))), IndependenceCopula(2)
    )
    assert d.measure_type is MeasureType.MIXED
    assert sp.simplify(d.pdf((sp.Rational(1, 2), 1)) - sp.Rational(1, 4)) == 0


def test_archimedean_copula_boundaries_and_density():
    u, v = sp.Rational(1, 3), sp.Rational(2, 5)
    for copula in (ClaytonCopula(2), FrankCopula(2), GumbelCopula(2)):
        assert sp.simplify(copula.cdf((u, 1)) - u) == 0
        assert sp.simplify(copula.cdf((1, v)) - v) == 0
        assert copula.density((u, v)) != 0


def test_archimedean_sampling_shapes():
    for copula in (ClaytonCopula(2), FrankCopula(2), GumbelCopula(2)):
        draws = copula.sample_uniform(size=8, rng=123)
        assert draws.shape == (8, 2)
        assert ((draws >= 0) & (draws <= 1)).all()


def test_nonlinear_dependent_pushforward_retains_exact_integral():
    x, y = sp.symbols("x y", real=True)
    joint = JointDistribution(
        (x, y), sp.exp(-x) * sp.exp(-y), (sp.Interval(0, sp.oo),) * 2
    )
    out = dependent_transform(joint, x * y)
    assert isinstance(out, PushforwardDistribution)
    z = sp.symbols("z", positive=True)
    density = out.pdf(z)
    assert (
        density.has(sp.Integral) or density.has(sp.besselk) or density.has(sp.meijerg)
    )


def test_vector_nonlinear_pushforward_formal_distribution():
    x, y = sp.symbols("x y", real=True)
    joint = JointDistribution(
        (x, y), sp.exp(-x) * sp.exp(-y), (sp.Interval(0, sp.oo),) * 2
    )
    out = dependent_transform(joint, (x + y, x * y))
    assert isinstance(out, PushforwardDistribution)
    assert out.event_space.shape == (2,)


def test_affine_mvn_pushforward_closes_to_mvn():
    x, y = sp.symbols("x y", real=True)
    source = MultivariateNormal([1, 2], [[2, 1], [1, 3]])
    out = dependent_transform(source, (x + y, x - y), variables=(x, y))
    assert isinstance(out, MultivariateNormal)
    assert out.mean == sp.ImmutableMatrix([3, -1])
    assert out.covariance == sp.ImmutableMatrix([[7, -1], [-1, 3]])


def test_nonlinear_pushforward_sampling_from_mvn():
    x, y = sp.symbols("x y", real=True)
    source = MultivariateNormal([0, 0], [[1, 0], [0, 1]])
    out = dependent_transform(source, (x**2, x * y), variables=(x, y))
    draws = out.sample(size=12, rng=123)
    assert draws.shape == (12, 2)
    assert (draws[:, 0] >= 0).all()
