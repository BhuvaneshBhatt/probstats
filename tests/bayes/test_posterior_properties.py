"""Property-based posterior invariants for named conjugate families."""

import sympy as sp
from hypothesis import given, settings
from hypothesis import strategies as st

from probstats.bayes import infer_conjugate
from probstats.distributions import (
    Bernoulli,
    Beta,
    Gamma,
    Poisson,
)

POSITIVE_SMALL = st.integers(min_value=1, max_value=8)
BINARY_DATA = st.lists(st.integers(min_value=0, max_value=1), min_size=0, max_size=10)
COUNT_DATA = st.lists(st.integers(min_value=0, max_value=6), min_size=0, max_size=8)


@settings(max_examples=40, deadline=None)
@given(alpha=POSITIVE_SMALL, beta=POSITIVE_SMALL, data=BINARY_DATA)
def test_beta_bernoulli_posterior_contract(alpha, beta, data):
    p = sp.Symbol("property_p", real=True)
    prior = Beta(alpha, beta)
    result = infer_conjugate(Bernoulli(p), data, prior)
    posterior = result.posterior

    successes = sum(data)
    failures = len(data) - successes
    assert posterior.alpha == alpha + successes
    assert posterior.beta == beta + failures
    assert posterior.support == prior.support
    assert result.metadata["sufficient_statistics"] == {
        "successes": sp.Integer(successes),
        "failures": sp.Integer(failures),
    }
    assert (
        sp.simplify(sp.expand_func(sp.integrate(posterior.pdf(p), (p, 0, 1)) - 1)) == 0
    )


@settings(max_examples=40, deadline=None)
@given(alpha=POSITIVE_SMALL, beta=POSITIVE_SMALL, data=BINARY_DATA)
def test_beta_bernoulli_batch_and_sequential_updates_are_identical(alpha, beta, data):
    p = sp.Symbol("sequential_p", real=True)
    prior = Beta(alpha, beta)
    batch = infer_conjugate(Bernoulli(p), data, prior)

    posterior = prior
    log_evidence = sp.S.Zero
    for datum in data:
        step = infer_conjugate(Bernoulli(p), [datum], posterior)
        posterior = step.posterior
        log_evidence += step.log_evidence

    assert posterior == batch.posterior
    assert (
        sp.simplify(sp.expand_func(sp.exp(log_evidence) - sp.exp(batch.log_evidence)))
        == 0
    )


@settings(max_examples=30, deadline=None)
@given(shape=POSITIVE_SMALL, scale=POSITIVE_SMALL, data=COUNT_DATA)
def test_gamma_poisson_posterior_is_proper_and_accumulates_count_statistic(
    shape, scale, data
):
    rate = sp.Symbol("property_rate", positive=True)
    prior = Gamma(shape, scale)
    result = infer_conjugate(Poisson(rate), data, prior)
    posterior = result.posterior

    assert posterior.shape == shape + sum(data)
    assert posterior.scale == sp.Rational(scale, 1 + len(data) * scale)
    assert result.metadata["sufficient_statistics"]["count_sum"] == sum(data)
    assert posterior.support == prior.support
    assert sp.simplify(sp.integrate(posterior.pdf(rate), (rate, 0, sp.oo)) - 1) == 0
