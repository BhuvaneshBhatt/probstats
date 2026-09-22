import sympy as sp

from probstats import (
    Bernoulli,
    Binomial,
    Exponential,
    Normal,
    Poisson,
)
from probstats.functionals import (
    cumulant,
    cumulative_hazard,
    factorial_moment,
    hazard,
    inverse_survival,
    likelihood_value,
    log_likelihood,
)
from probstats.inference import likelihood
from probstats.symbolic import (
    central_moment_generating_function,
    cumulant_generating_function,
    factorial_moment_generating_function,
)


def test_continuous_hazard_and_inverse_survival():
    rate = sp.Symbol("rate", positive=True)
    x = sp.Symbol("x", nonnegative=True)
    q = sp.Symbol("q", positive=True)
    dist = Exponential(rate)
    assert sp.simplify(hazard(dist, x) - rate) == 0
    assert sp.simplify(cumulative_hazard(dist, x) - rate * x) == 0
    assert sp.simplify(inverse_survival(dist, q) + sp.log(q) / rate) == 0


def test_discrete_hazard_uses_mass_at_current_point():
    p = sp.Symbol("p", positive=True)
    dist = Bernoulli(p)
    assert sp.simplify(hazard(dist, 0) - (1 - p)) == 0
    assert hazard(dist, 1) == 1


def test_cumulants_from_raw_moments():
    mu = sp.Symbol("mu", real=True)
    sigma = sp.Symbol("sigma", positive=True)
    dist = Normal(mu, sigma)
    assert cumulant(dist, 0) == 0
    assert sp.simplify(cumulant(dist, 1) - mu) == 0
    assert sp.simplify(cumulant(dist, 2) - sigma**2) == 0
    assert sp.simplify(cumulant(dist, 3)) == 0
    assert sp.simplify(cumulant(dist, 4)) == 0


def test_cumulant_and_central_moment_generating_functions():
    t = sp.Symbol("t", real=True)
    mu = sp.Symbol("mu", real=True)
    sigma = sp.Symbol("sigma", positive=True)
    dist = Normal(mu, sigma)
    assert (
        sp.simplify(
            cumulant_generating_function(dist, t) - (mu * t + sigma**2 * t**2 / 2)
        )
        == 0
    )
    assert (
        sp.simplify(
            central_moment_generating_function(dist, t) - sp.exp(sigma**2 * t**2 / 2)
        )
        == 0
    )


def test_factorial_moments_and_generating_function():
    n = sp.Symbol("n", integer=True, nonnegative=True)
    p = sp.Symbol("p", nonnegative=True)
    rate = sp.Symbol("rate", positive=True)
    assert sp.simplify(factorial_moment(Binomial(n, p), 2) - n * (n - 1) * p**2) == 0
    assert sp.simplify(factorial_moment(Poisson(rate), 4) - rate**4) == 0

    t = sp.Symbol("t")
    fmgf = factorial_moment_generating_function(Poisson(rate), t)
    assert sp.simplify(fmgf - sp.exp(rate * t)) == 0
    assert (
        sp.simplify(sp.diff(fmgf, t, 3).subs(t, 0) - factorial_moment(Poisson(rate), 3))
        == 0
    )


def test_generic_likelihood_and_log_likelihood():
    p = sp.Symbol("p", positive=True)
    dist = Bernoulli(p)
    data = [1, 0, 1]
    expected = p**2 * (1 - p)
    assert sp.simplify(likelihood_value(dist, data) - expected) == 0
    assert sp.simplify(likelihood(dist, data) - expected) == 0
    assert sp.simplify(log_likelihood(dist, data) - sp.log(expected)) == 0
    assert sp.simplify(dist.likelihood(data) - expected) == 0
    assert sp.simplify(dist.log_likelihood(data) - sp.log(expected)) == 0


def test_distribution_methods_for_completed_functionals():
    rate = sp.Integer(2)
    dist = Exponential(rate)
    assert dist.hazard(3) == 2
    assert dist.cumulative_hazard(3) == 6
    assert sp.simplify(dist.inverse_survival(sp.Rational(1, 4)) - sp.log(2)) == 0

    pois = Poisson(3)
    assert pois.factorial_moment(2) == 9
    assert pois.cumulant(2) == 3
