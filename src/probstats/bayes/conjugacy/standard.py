"""Named conjugate updates using distributions provided by probstats."""

from __future__ import annotations

from collections.abc import Iterable

import sympy as sp

from ..._symbolic_predicates import TruthValue, certified_equal
from ..distributions import (
    Bernoulli,
    Beta,
    Binomial,
    Categorical,
    Dirichlet,
    Gamma,
    Multinomial,
    MultivariateNormal,
    Poisson,
)
from .multivariate import NormalInverseWishart, update_multivariate_normal


def _multibeta(alpha):
    alpha = tuple(map(sp.sympify, alpha))
    return sp.simplify(sp.prod(sp.gamma(a) for a in alpha) / sp.gamma(sum(alpha)))


def update_beta_bernoulli(likelihood: Bernoulli, data: Iterable, prior: Beta):
    xs = tuple(map(sp.sympify, data))
    n = len(xs)
    s = sp.simplify(sum(xs))
    post = Beta(sp.simplify(prior.alpha + s), sp.simplify(prior.beta + n - s))
    logz = sp.simplify(
        sp.log(sp.beta(post.alpha, post.beta))
        - sp.log(sp.beta(prior.alpha, prior.beta))
    )
    return {
        "prior": prior,
        "posterior": post,
        "log_evidence": logz,
        "prior_predictive": None,
        "posterior_predictive": None,
        "n_observations": n,
        "sufficient_statistics": {"successes": s, "failures": sp.simplify(n - s)},
    }


def update_beta_binomial(likelihood: Binomial, data: Iterable, prior: Beta):
    xs = tuple(map(sp.sympify, data))
    m = len(xs)
    s = sp.simplify(sum(xs))
    failures = sp.simplify(m * likelihood.n - s)
    post = Beta(sp.simplify(prior.alpha + s), sp.simplify(prior.beta + failures))
    log_coeff = sum(sp.log(sp.binomial(likelihood.n, x)) for x in xs)
    logz = sp.simplify(
        log_coeff
        + sp.log(sp.beta(post.alpha, post.beta))
        - sp.log(sp.beta(prior.alpha, prior.beta))
    )
    return {
        "prior": prior,
        "posterior": post,
        "log_evidence": logz,
        "prior_predictive": None,
        "posterior_predictive": None,
        "n_observations": m,
        "sufficient_statistics": {"successes": s, "failures": failures},
    }


def update_gamma_poisson(likelihood: Poisson, data: Iterable, prior: Gamma):
    xs = tuple(map(sp.sympify, data))
    n = len(xs)
    s = sp.simplify(sum(xs))
    rate0 = sp.simplify(1 / prior.scale)
    rate1 = sp.simplify(rate0 + n)
    post = Gamma(sp.simplify(prior.shape + s), sp.simplify(1 / rate1))
    logz = sp.simplify(
        sp.loggamma(post.shape)
        - sp.loggamma(prior.shape)
        + prior.shape * sp.log(rate0)
        - post.shape * sp.log(rate1)
        - sum(sp.loggamma(x + 1) for x in xs)
    )
    return {
        "prior": prior,
        "posterior": post,
        "log_evidence": logz,
        "prior_predictive": None,
        "posterior_predictive": None,
        "n_observations": n,
        "sufficient_statistics": {"count_sum": s},
    }


def update_dirichlet_categorical(
    likelihood: Categorical, data: Iterable, prior: Dirichlet
):
    xs = tuple(map(sp.sympify, data))
    k = prior.dimension
    if len(likelihood.probabilities) != k:
        raise ValueError("Dirichlet and Categorical dimensions differ")
    counts = [sp.S.Zero] * k
    for x in xs:
        if x.is_integer is not True or x.is_number is not True or not (0 <= int(x) < k):
            raise ValueError("Categorical observations must be integer category labels")
        counts[int(x)] += 1
    post = Dirichlet(
        tuple(sp.simplify(a + c) for a, c in zip(prior.concentration, counts))
    )
    logz = sp.simplify(
        sp.log(_multibeta(post.concentration)) - sp.log(_multibeta(prior.concentration))
    )
    return {
        "prior": prior,
        "posterior": post,
        "log_evidence": logz,
        "prior_predictive": None,
        "posterior_predictive": None,
        "n_observations": len(xs),
        "sufficient_statistics": {"counts": tuple(counts)},
    }


def update_dirichlet_multinomial(
    likelihood: Multinomial, data: Iterable, prior: Dirichlet
):
    rows = tuple(tuple(map(sp.sympify, row)) for row in data)
    k = prior.dimension
    if likelihood.dimension != k:
        raise ValueError("Dirichlet and Multinomial dimensions differ")
    counts = [sp.S.Zero] * k
    log_coeff = sp.S.Zero
    for row in rows:
        if len(row) != k:
            raise ValueError("Multinomial observation dimension differs")
        total_relation = certified_equal(sum(row), likelihood.n)
        if total_relation is TruthValue.FALSE:
            raise ValueError("Multinomial counts must sum to n")
        if total_relation is TruthValue.UNKNOWN:
            raise ValueError("could not certify that Multinomial counts sum to n")
        for i, x in enumerate(row):
            counts[i] += x
        log_coeff += sp.loggamma(likelihood.n + 1) - sum(
            sp.loggamma(x + 1) for x in row
        )
    post = Dirichlet(
        tuple(sp.simplify(a + c) for a, c in zip(prior.concentration, counts))
    )
    logz = sp.simplify(
        log_coeff
        + sp.log(_multibeta(post.concentration))
        - sp.log(_multibeta(prior.concentration))
    )
    return {
        "prior": prior,
        "posterior": post,
        "log_evidence": logz,
        "prior_predictive": None,
        "posterior_predictive": None,
        "n_observations": len(rows),
        "sufficient_statistics": {"counts": tuple(counts)},
    }


__all__ = [
    "update_beta_bernoulli",
    "update_beta_binomial",
    "update_dirichlet_categorical",
    "update_dirichlet_multinomial",
    "update_gamma_poisson",
    "update_multivariate_normal_niw",
]


def update_multivariate_normal_niw(
    likelihood: MultivariateNormal, data: Iterable, prior: NormalInverseWishart
):
    """Update a Normal-Inverse-Wishart prior for multivariate-normal observations."""
    rows = tuple(data)
    posterior = update_multivariate_normal(rows, prior)
    return {
        "prior": prior,
        "posterior": posterior,
        "log_evidence": None,
        "prior_predictive": prior.predictive,
        "posterior_predictive": posterior.predictive,
        "n_observations": len(rows),
        "sufficient_statistics": {},
    }
