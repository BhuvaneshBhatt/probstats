import numpy as np
import sympy as sp

from probstats import (
    Bernoulli,
    Beta,
    Exponential,
    Gamma,
    Normal,
    Poisson,
    probability,
)
from probstats.distributions import (
    CensoredDistribution,
    ParameterMixtureDistribution,
    ProbabilityDistribution,
    SplicedDistribution,
)
from probstats.functionals import (
    likelihood_value,
    mean,
)
from probstats.spaces import MeasureType


def test_user_defined_continuous_distribution():
    x, p = sp.symbols("x p", real=True)
    dist = ProbabilityDistribution(
        x,
        2 * x,
        sp.Interval(0, 1),
        cdf_expression=x**2,
        quantile_expression=sp.sqrt(p),
        probability_symbol=p,
        sampler=lambda *, size, rng: np.sqrt(rng.random(size=size)),
    )
    assert dist.cdf(sp.Rational(1, 2)) == sp.Rational(1, 4)
    assert dist.quantile(sp.Rational(1, 4)) == sp.Rational(1, 2)
    assert mean(dist) == sp.Rational(2, 3)
    draws = dist.sample(size=50, rng=10)
    assert draws.shape == (50,)
    assert np.all((draws >= 0) & (draws <= 1))


def test_user_defined_discrete_distribution():
    k = sp.Symbol("k", integer=True, nonnegative=True)
    dist = ProbabilityDistribution(
        k, sp.Rational(1, 3), sp.FiniteSet(0, 1, 2), discrete=True
    )
    assert dist.pmf(1) == sp.Rational(1, 3)
    assert probability(dist, sp.FiniteSet(0, 2)) == sp.Rational(2, 3)


def test_censoring_creates_endpoint_atoms():
    dist = CensoredDistribution(Normal(0, 1), -1, 1)
    assert dist.measure_type is MeasureType.MIXED
    assert sp.simplify(dist.lower_mass - Normal(0, 1).cdf(-1)) == 0
    assert sp.simplify(dist.upper_mass - (1 - Normal(0, 1).cdf(1))) == 0
    assert probability(dist, sp.FiniteSet(-1, 1)) == sp.simplify(
        dist.lower_mass + dist.upper_mass
    )
    assert sp.simplify(dist.cdf(0) - sp.Rational(1, 2)) == 0


def test_censored_likelihood_uses_atom_mass():
    base = Exponential(1)
    dist = CensoredDistribution(base, 0, 2)
    expected = base.pdf(1) * (1 - base.cdf(2))
    assert sp.simplify(likelihood_value(dist, [1, 2]) - expected) == 0


def test_discrete_censored_expectation():
    dist = CensoredDistribution(Poisson(2), 0, 2)
    expected = 2 - 4 * sp.exp(-2)
    assert sp.simplify(mean(dist) - expected) == 0


def test_censored_sampling_is_clipped():
    draws = CensoredDistribution(Normal(0, 1), -0.5, 0.5).sample(size=100, rng=4)
    assert np.all(draws >= -0.5)
    assert np.all(draws <= 0.5)


def test_beta_poisson_parameter_mixture_density_and_mean():
    # Gamma-Poisson mixture is negative-binomial-like; test exact hierarchy directly.
    theta = sp.Symbol("theta", positive=True)
    dist = ParameterMixtureDistribution(Gamma(2, 3), lambda rate: Poisson(rate), theta)
    assert sp.simplify(dist.pmf(0) - sp.Rational(1, 16)) == 0
    assert mean(dist) == 6


def test_beta_bernoulli_parameter_mixture():
    theta = sp.Symbol("theta", real=True)
    dist = ParameterMixtureDistribution(Beta(2, 3), lambda p: Bernoulli(p), theta)
    assert sp.simplify(dist.pmf(1) - sp.Rational(2, 5)) == 0
    assert sp.simplify(dist.pmf(0) - sp.Rational(3, 5)) == 0


def test_parameter_mixture_sampling():
    dist = ParameterMixtureDistribution(Gamma(2, 1), lambda rate: Poisson(rate))
    draws = dist.sample(size=30, rng=3)
    assert draws.shape == (30,)
    assert np.all(draws >= 0)


def test_spliced_distribution_normalizes_and_selects_regions():
    dist = SplicedDistribution(
        [Normal(-1, 1), Normal(2, 1)],
        [sp.Interval.open(-sp.oo, 0), sp.Interval(0, sp.oo)],
        [sp.Rational(2, 5), sp.Rational(3, 5)],
    )
    assert sp.simplify(probability(dist, dist.support) - 1) == 0
    assert (
        sp.simplify(probability(dist, sp.Interval.open(-sp.oo, 0)) - sp.Rational(2, 5))
        == 0
    )
    assert dist.pdf(-1) != 0
    assert dist.pdf(1) != 0


def test_spliced_sampling():
    dist = SplicedDistribution(
        [Normal(-1, 1), Normal(1, 1)],
        [sp.Interval.open(-sp.oo, 0), sp.Interval(0, sp.oo)],
        [sp.Rational(1, 2), sp.Rational(1, 2)],
    )
    draws = dist.sample(size=40, rng=2)
    assert draws.shape == (40,)
