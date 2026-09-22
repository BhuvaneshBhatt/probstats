import sympy as sp

from probstats import (
    Bernoulli,
    Beta,
    Binomial,
    Gamma,
    Normal,
    Poisson,
)
from probstats.algebra import (
    RecognitionKind,
    canonicalize_distribution,
    conjugacy_signature,
    distribution_metadata,
    equivalent_distributions,
    exponential_family_metadata,
    recognize_affine,
    recognize_product,
    recognize_sum,
    recognize_transform,
)
from probstats.distributions import (
    Cauchy,
    ChiSquared,
    LogNormal,
)
from probstats.transforms import Transform


def test_canonicalize_chisquare_to_gamma_and_equivalence():
    c = canonicalize_distribution(ChiSquared(6))
    assert isinstance(c, Gamma)
    assert c.shape == 3 and c.scale == 2
    assert equivalent_distributions(ChiSquared(6), Gamma(3, 2))


def test_normal_affine_recognition():
    r = recognize_affine(Normal(2, 3), -2, 5)
    assert r.recognized and r.kind is RecognitionKind.AFFINE
    assert r.distribution == Normal(1, 6)


def test_exp_normal_recognizes_lognormal():
    x, y = sp.symbols("x y", real=True)
    t = Transform.from_expressions(
        sp.exp(x), x, sp.log(y), y, codomain=sp.Interval.open(0, sp.oo)
    )
    r = recognize_transform(Normal(2, 3), t)
    assert r.distribution == LogNormal(2, 3)


def test_closed_sum_rules():
    assert recognize_sum(Normal(1, 2), Normal(3, 4)).distribution == Normal(
        4, 2 * sp.sqrt(5)
    )
    assert recognize_sum(Poisson(2), Poisson(5)).distribution == Poisson(7)
    assert recognize_sum(
        Binomial(3, sp.Rational(1, 4)), Binomial(7, sp.Rational(1, 4))
    ).distribution == Binomial(10, sp.Rational(1, 4))
    assert recognize_sum(Gamma(2, 3), Gamma(5, 3)).distribution == Gamma(7, 3)
    assert recognize_sum(Cauchy(1, 2), Cauchy(3, 4)).distribution == Cauchy(4, 6)


def test_closed_product_rules():
    assert recognize_product(
        LogNormal(1, 2), LogNormal(3, 4)
    ).distribution == LogNormal(4, 2 * sp.sqrt(5))
    assert recognize_product(
        Bernoulli(sp.Rational(1, 2)), Bernoulli(sp.Rational(1, 3))
    ).distribution == Bernoulli(sp.Rational(1, 6))


def test_exponential_family_metadata_is_canonical():
    d = Beta(2, 3)
    m = exponential_family_metadata(d)
    assert m is not None
    assert m.family == "Beta"
    assert m.natural_parameters == (1, 2)
    assert m.natural_dimension == 2
    assert m.sufficient_statistics == (
        sp.log(sp.Symbol("x", real=True)),
        sp.log(1 - sp.Symbol("x", real=True)),
    )


def test_conjugacy_signature_is_stable_and_decoupled():
    sig = conjugacy_signature(Bernoulli(sp.Rational(1, 3)))
    assert sig.likelihood_family == "Bernoulli"
    assert sig.conjugate_prior_family == "Beta"
    assert sig.natural_dimension == 1


def test_planner_metadata_traits():
    m = distribution_metadata(Normal(0, 1))
    assert m.family == "Normal"
    assert m.is_exponential_family
    assert m.is_location_scale_family
    assert m.event_shape == ()


def test_more_affine_families_and_distribution_methods():
    from probstats import (
        StudentT,
        Uniform,
    )
    from probstats.distributions import (
        Laplace,
        Logistic,
    )

    assert recognize_affine(Laplace(1, 2), -3, 4).distribution == Laplace(1, 6)
    assert recognize_affine(Logistic(1, 2), 2, 3).distribution == Logistic(5, 4)
    assert recognize_affine(StudentT(1, 2, 7), -2, 3).distribution == StudentT(1, 4, 7)
    assert recognize_affine(Uniform(1, 3), -2, 5).distribution == Uniform(-1, 3)
    assert ChiSquared(4).canonicalize() == Gamma(2, 2)
    assert ChiSquared(4).equivalent_to(Gamma(2, 2))
    assert Normal(0, 1).metadata.family == "Normal"


def test_bernoulli_and_multivariate_normal_sum_rules():
    from probstats import MultivariateNormal

    p = sp.Rational(2, 5)
    assert recognize_sum(
        Bernoulli(p), Bernoulli(p), Bernoulli(p)
    ).distribution == Binomial(3, p)
    a = MultivariateNormal([1, 2], [[2, 0], [0, 3]])
    b = MultivariateNormal([4, 5], [[1, 0], [0, 2]])
    assert recognize_sum(a, b).distribution == MultivariateNormal(
        [5, 7], [[3, 0], [0, 5]]
    )


def test_multivariate_exponential_family_flags():
    from probstats import MultivariateNormal
    from probstats.distributions import (
        Categorical,
        Dirichlet,
        Multinomial,
    )

    assert distribution_metadata(
        Categorical([sp.Rational(1, 2), sp.Rational(1, 2)])
    ).is_exponential_family
    assert distribution_metadata(
        Multinomial(2, [sp.Rational(1, 2), sp.Rational(1, 2)])
    ).is_exponential_family
    assert distribution_metadata(Dirichlet([2, 3])).is_exponential_family
    assert distribution_metadata(
        MultivariateNormal([0, 0], sp.eye(2))
    ).is_exponential_family
