import numpy as np
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
    StudentT,
    sample,
)
from probstats.composition import (
    ConditionalDistribution,
    IndependentDistribution,
    MarginalDistribution,
    MixtureDistribution,
    ProductDistribution,
    TruncatedDistribution,
)
from probstats.distributions import (
    Categorical,
    Dirichlet,
    InverseGamma,
    InverseWishart,
    LogNormal,
    Multinomial,
    MultivariateStudentT,
    Wishart,
)
from probstats.sampling import (
    SamplingError,
    as_rng,
)
from probstats.transforms import (
    MultivariateTransform,
    Transform,
    TransformedDistribution,
)


def test_rng_reproducibility_and_method_protocol():
    d = Normal(2, 3)
    a = sample(d, size=6, rng=1234)
    b = d.sample(size=6, rng=1234)
    np.testing.assert_array_equal(a, b)
    assert isinstance(as_rng(10), np.random.Generator)


def test_scalar_primitive_shapes_and_domains():
    assert Bernoulli(sp.Rational(1, 3)).sample(rng=1) in (0, 1)
    assert Binomial(5, sp.Rational(1, 2)).sample(size=7, rng=2).shape == (7,)
    x = Beta(2, 3).sample(size=(2, 4), rng=3)
    assert x.shape == (2, 4)
    assert np.all((0 <= x) & (x <= 1))
    x = Exponential(2).sample(size=20, rng=4)
    assert np.all(x >= 0)
    assert Poisson(3).sample(size=5, rng=5).shape == (5,)
    assert LogNormal(0, 1).sample(size=5, rng=6).shape == (5,)
    assert Gamma(2, 3).sample(size=5, rng=7).shape == (5,)
    assert InverseGamma(3, 2).sample(size=5, rng=8).shape == (5,)
    assert StudentT(0, 1, 5).sample(size=5, rng=9).shape == (5,)


def test_discrete_vector_and_simplex_sampling():
    c = Categorical([sp.Rational(1, 4), sp.Rational(3, 4)])
    assert c.sample(size=8, rng=1).shape == (8,)
    m = Multinomial(10, [sp.Rational(1, 4), sp.Rational(3, 4)]).sample(size=6, rng=2)
    assert m.shape == (6, 2)
    np.testing.assert_array_equal(m.sum(axis=-1), 10)
    d = Dirichlet([1, 2, 3]).sample(size=(2, 5), rng=3)
    assert d.shape == (2, 5, 3)
    np.testing.assert_allclose(d.sum(axis=-1), 1)


def test_multivariate_and_matrix_sampling_shapes():
    mvn = MultivariateNormal([0, 1], [[2, 0.2], [0.2, 1]])
    assert mvn.sample(rng=1).shape == (2,)
    assert mvn.sample(size=(3, 4), rng=1).shape == (3, 4, 2)
    mvt = MultivariateStudentT([0, 0], [[1, 0], [0, 1]], 5)
    assert mvt.sample(size=7, rng=2).shape == (7, 2)
    w = Wishart(6, [[1, 0.1], [0.1, 2]]).sample(size=4, rng=3)
    iw = InverseWishart(6, [[1, 0.1], [0.1, 2]]).sample(size=4, rng=4)
    assert w.shape == iw.shape == (4, 2, 2)
    assert np.all(np.linalg.eigvalsh(w) > 0)
    assert np.all(np.linalg.eigvalsh(iw) > 0)


def test_product_and_iid_sampling_preserve_structure():
    p = ProductDistribution([Normal(0, 1), Bernoulli(sp.Rational(1, 2))])
    draws = p.sample(size=5, rng=1)
    assert isinstance(draws, tuple) and len(draws) == 2
    assert draws[0].shape == draws[1].shape == (5,)
    iid = IndependentDistribution(Normal(0, 1), 3)
    assert iid.sample(size=(2, 4), rng=2).shape == (2, 4, 3)


def test_mixture_sampling_and_reproducibility():
    mix = MixtureDistribution(
        [Normal(-10, 0.01), Normal(10, 0.01)], [sp.Rational(1, 4), sp.Rational(3, 4)]
    )
    a = mix.sample(size=200, rng=99)
    b = mix.sample(size=200, rng=99)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (200,)
    assert np.any(a < 0) and np.any(a > 0)


def test_truncated_rejection_sampling():
    d = TruncatedDistribution(Normal(0, 1), sp.Interval(0, sp.oo))
    x = d.sample(size=100, rng=7)
    assert x.shape == (100,)
    assert np.all(x >= 0)
    with pytest.raises(SamplingError):
        TruncatedDistribution(Normal(0, 1), sp.Interval(10, 11)).sample(
            size=2, rng=1, max_attempts=20
        )


def test_scalar_transformed_sampling():
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
    a = d.sample(size=10, rng=4)
    raw = Normal(0, 1).sample(size=10, rng=4)
    np.testing.assert_allclose(a, np.exp(raw))


def test_multivariate_transformed_sampling():
    x1, x2, y1, y2 = sp.symbols("x1 x2 y1 y2", real=True)
    t = MultivariateTransform([x1 + x2, x2], (x1, x2), [y1 - y2, y2], (y1, y2))
    d = TransformedDistribution(MultivariateNormal([0, 0], sp.eye(2)), t)
    assert d.sample(size=(3, 2), rng=5).shape == (3, 2, 2)


def test_marginal_and_conditional_delegate_sampling():
    joint = MultivariateNormal([0, 0], [[1, 0.5], [0.5, 1]])
    assert MarginalDistribution(joint, 0).sample(size=3, rng=1).shape == (3,)
    assert ConditionalDistribution(joint, 1, 2).sample(size=3, rng=1).shape == (3,)


def test_symbolic_parameters_fail_explicitly():
    mu = sp.symbols("mu", real=True)
    with pytest.raises(SamplingError, match="numeric"):
        Normal(mu, 1).sample(rng=1)


def test_sampling_size_rejects_fractional_and_boolean_entries():
    from probstats import Normal

    dist = Normal(0, 1)
    with pytest.raises(TypeError, match="size"):
        dist.sample(size=2.5, rng=1)
    with pytest.raises(TypeError, match="size"):
        dist.sample(size=(2, 1.5), rng=1)
    with pytest.raises(TypeError, match="size"):
        dist.sample(size=True, rng=1)
