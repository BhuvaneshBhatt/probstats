"""Differential tests requiring independent inference routes to agree."""

import sympy as sp

from probstats.bayes import (
    Factor,
    Model,
    Parameter,
    RandomVariable,
    infer_conjugate,
)
from probstats.bayes.exact import infer_exact
from probstats.bayes.laplace import infer_laplace
from probstats.distributions import (
    Bernoulli,
    Beta,
    Gamma,
    Normal,
    Poisson,
)


def _expanded(expr):
    return sp.simplify(sp.expand_func(expr))


def test_beta_bernoulli_conjugacy_matches_direct_symbolic_normalization():
    p = Parameter("diff_p", sp.Interval(0, 1))
    y = RandomVariable("diff_y", sp.FiniteSet(0, 1))
    prior = Beta(2, 3)
    likelihood = Bernoulli(p.symbol)
    model = Model(
        (p, y),
        (Factor.from_distribution(p, prior), Factor.from_distribution(y, likelihood)),
    ).observe(diff_y=1)

    direct = infer_exact(model)
    conjugate = infer_conjugate(likelihood, [1], prior)
    expected_density = conjugate.posterior.pdf(p.symbol)

    assert _expanded(direct.posterior.density - expected_density) == 0
    assert _expanded(sp.exp(direct.log_evidence) - sp.exp(conjugate.log_evidence)) == 0


def test_gamma_poisson_conjugacy_matches_direct_symbolic_normalization():
    rate = Parameter("diff_rate", sp.Interval.open(0, sp.oo))
    y = RandomVariable("diff_count", sp.S.Naturals0)
    prior = Gamma(3, 2)
    likelihood = Poisson(rate.symbol)
    model = Model(
        (rate, y),
        (
            Factor.from_distribution(rate, prior),
            Factor.from_distribution(y, likelihood),
        ),
    ).observe(diff_count=2)

    direct = infer_exact(model)
    conjugate = infer_conjugate(likelihood, [2], prior)
    expected_density = conjugate.posterior.pdf(rate.symbol)

    assert _expanded(direct.posterior.density - expected_density) == 0
    assert _expanded(sp.exp(direct.log_evidence) - sp.exp(conjugate.log_evidence)) == 0


def test_laplace_is_exact_for_quadratic_gaussian_location_posterior():
    theta = Parameter("diff_theta", sp.S.Reals)
    y = RandomVariable("diff_obs", sp.S.Reals)
    model = Model(
        (theta, y),
        (
            Factor.from_distribution(theta, Normal(0, 1)),
            Factor.from_distribution(y, Normal(theta.symbol, 1)),
        ),
    ).observe(diff_obs=1)

    direct = infer_exact(model)
    laplace = infer_laplace(model, backend="sympy", initial_guess=[0])

    assert sp.simplify(laplace.posterior.mean[0] - sp.Rational(1, 2)) == 0
    assert sp.simplify(laplace.posterior.covariance[0, 0] - sp.Rational(1, 2)) == 0
    assert abs(float(sp.N(laplace.log_evidence - direct.log_evidence))) < 1e-12
