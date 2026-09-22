import pytest
import sympy as sp

from probstats import (
    Bernoulli,
    Beta,
    MultivariateNormal,
    Normal,
    cdf,
)
from probstats.composition import (
    ConditionalDistribution,
    IndependentDistribution,
    MarginalDistribution,
    MixtureDistribution,
    ProductDistribution,
    TruncatedDistribution,
)
from probstats.functionals import (
    mean,
    variance,
)
from probstats.spaces import VectorEventSpace
from probstats.transforms import (
    InverseBranch,
    MultivariateTransform,
    Transform,
    TransformedDistribution,
)


def test_product_distribution_and_marginal():
    joint = ProductDistribution([Normal(0, 1), Bernoulli(sp.Rational(1, 3))])
    assert sp.simplify(joint.pdf((0, 1)) - Normal(0, 1).pdf(0) / 3) == 0
    assert joint.marginal(1).distribution == Bernoulli(sp.Rational(1, 3))


def test_iid_distribution_preserves_structure():
    iid = IndependentDistribution(Normal(2, 3), 3)
    assert iid.event_space.shape == (3,)
    assert mean(iid) == sp.ImmutableMatrix([2, 2, 2])
    assert variance(iid) == sp.diag(9, 9, 9)


def test_truncated_normal_normalization_and_support():
    d = TruncatedDistribution(Normal(0, 1), sp.Interval(0, sp.oo))
    assert sp.simplify(d.normalizing_constant - sp.Rational(1, 2)) == 0
    assert d.pdf(-1) == 0
    assert sp.simplify(cdf(d, 0)) == 0


def test_discrete_truncation():
    d = TruncatedDistribution(Bernoulli(sp.Rational(1, 4)), sp.FiniteSet(1))
    assert d.normalizing_constant == sp.Rational(1, 4)
    assert d.pmf(1) == 1
    assert d.pmf(0) == 0


def test_mixture_mean_variance_and_cdf():
    d = MixtureDistribution([Normal(-1, 1), Normal(1, 1)], [sp.Rational(1, 2)] * 2)
    assert mean(d) == 0
    assert variance(d) == 2
    assert sp.simplify(cdf(d, 0) - sp.Rational(1, 2)) == 0


def test_mixture_weight_validation():
    with pytest.raises(ValueError):
        MixtureDistribution(
            [Normal(0, 1), Normal(1, 1)], [sp.Rational(2, 3), sp.Rational(2, 3)]
        )
    with pytest.raises(ValueError):
        MixtureDistribution(
            [Normal(0, 1), Bernoulli(sp.Rational(1, 2))], [sp.Rational(1, 2)] * 2
        )


def test_decreasing_scalar_transform_cdf_and_quantile():
    x, y = sp.symbols("x y", real=True)
    t = Transform.from_expressions(-x, x, -y, y, orientation=-1)
    d = TransformedDistribution(Normal(2, 3), t)
    assert sp.simplify(mean(d) + 2) == 0
    assert sp.simplify(cdf(d, -2) - sp.Rational(1, 2)) == 0
    assert sp.simplify(d.quantile(sp.Rational(1, 2)) + 2) == 0


def test_many_to_one_transform_sums_inverse_branches():
    x, y = sp.symbols("x y", real=True)
    branches = (
        InverseBranch(sp.sqrt(y), y, sp.Interval(0, sp.oo)),
        InverseBranch(-sp.sqrt(y), y, sp.Interval(0, sp.oo)),
    )
    t = Transform.from_branches(x**2, x, y, branches, codomain=sp.Interval(0, sp.oo))
    d = TransformedDistribution(Normal(0, 1), t)
    yp = sp.symbols("yp", positive=True)
    expected = sp.exp(-yp / 2) / sp.sqrt(2 * sp.pi * yp)
    assert sp.simplify(d.pdf(yp) - expected) == 0
    assert d.pdf(-1) == 0


def test_multivariate_transform_jacobian():
    x1, x2, y1, y2 = sp.symbols("x1 x2 y1 y2", real=True)
    t = MultivariateTransform(
        [2 * x1, 3 * x2],
        [x1, x2],
        [y1 / 2, y2 / 3],
        [y1, y2],
        domain=VectorEventSpace(2, sp.S.Reals),
        codomain=VectorEventSpace(2, sp.S.Reals),
    )
    base = MultivariateNormal([0, 0], sp.eye(2))
    d = TransformedDistribution(base, t)
    assert sp.simplify(d.pdf([0, 0]) - base.pdf([0, 0]) / 6) == 0


def test_multivariate_normal_marginal():
    d = MultivariateNormal([1, 2], [[4, 1], [1, 9]])
    m = MarginalDistribution(d, 0).distribution
    assert m == Normal(1, 2)


def test_multivariate_normal_conditional():
    d = MultivariateNormal([0, 0], [[1, sp.Rational(1, 2)], [sp.Rational(1, 2), 1]])
    cond = ConditionalDistribution(d, 1, 2).distribution
    assert sp.simplify(cond.mean - 1) == 0
    assert sp.simplify(cond.sigma - sp.sqrt(sp.Rational(3, 4))) == 0


def test_independent_conditioning_removes_conditioned_factor():
    joint = ProductDistribution([Normal(0, 1), Beta(2, 3)])
    cond = ConditionalDistribution(joint, 0, 0)
    assert cond.distribution == Beta(2, 3)
