"""Normal conjugate priors and posterior updates."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import sympy as sp

from ..distributions import InverseGamma, Normal
from ..distributions.student_t import StudentT


@dataclass(frozen=True, slots=True)
class NormalInverseGamma:
    """Normal-Inverse-Gamma in the source package's parameterization.

    variance ~ InverseGamma(nu, beta)
    mean | variance ~ Normal(mu, sqrt(variance/lambda_)).
    """

    mu: sp.Expr
    lambda_: sp.Expr
    beta: sp.Expr
    nu: sp.Expr

    def __post_init__(self):
        for n in ("mu", "lambda_", "beta", "nu"):
            object.__setattr__(self, n, sp.sympify(getattr(self, n)))

    @property
    def parameters(self):
        return (self.mu, self.lambda_, self.beta, self.nu)

    @property
    def parameter_constraints(self):
        return sp.And(self.lambda_ > 0, self.beta > 0, self.nu > 0)

    @property
    def mean_marginal(self):
        return StudentT(
            self.mu, sp.sqrt(self.beta / (self.nu * self.lambda_)), 2 * self.nu
        )

    @property
    def variance_marginal(self):
        return InverseGamma(self.nu, self.beta)

    @property
    def predictive(self):
        return StudentT(
            self.mu,
            sp.sqrt(self.beta * (self.lambda_ + 1) / (self.lambda_ * self.nu)),
            2 * self.nu,
        )

    def logpdf(self, mean, variance):
        return sp.simplify(
            Normal(self.mu, sp.sqrt(sp.sympify(variance) / self.lambda_)).logpdf(mean)
            + self.variance_marginal.logpdf(variance)
        )

    def pdf(self, mean, variance):
        return sp.exp(self.logpdf(mean, variance))


@dataclass(frozen=True, slots=True)
class ConjugateUpdateResult:
    prior: Any
    posterior: Any
    prior_predictive: Any
    posterior_predictive: Any
    log_evidence: sp.Expr | None
    n_observations: int
    sufficient_statistics: dict[str, Any]


def default_normal_prior() -> NormalInverseGamma:
    return NormalInverseGamma(
        0, sp.Rational(1, 100), sp.Rational(1, 200), sp.Rational(1, 200)
    )


def update_normal(
    data: Iterable[Any], prior: NormalInverseGamma | None = None
) -> ConjugateUpdateResult:
    prior = prior or default_normal_prior()
    xs = tuple(map(sp.sympify, data))
    n = len(xs)
    if n == 0:
        return ConjugateUpdateResult(
            prior, prior, prior.predictive, prior.predictive, sp.S.Zero, 0, {"n": 0}
        )
    mean = sp.simplify(sum(xs) / n)
    ss = sp.simplify(sum((x - mean) ** 2 for x in xs))
    lam = sp.simplify(prior.lambda_ + n)
    mu = sp.simplify((prior.lambda_ * prior.mu + n * mean) / lam)
    beta = sp.simplify(
        prior.beta + ss / 2 + prior.lambda_ * n * (mean - prior.mu) ** 2 / (2 * lam)
    )
    nu = sp.simplify(prior.nu + sp.Rational(n, 2))
    post = NormalInverseGamma(mu, lam, beta, nu)
    # Stable exact marginal likelihood from NIG normalization constants.
    logz = sp.simplify(
        sp.loggamma(nu)
        - sp.loggamma(prior.nu)
        + prior.nu * sp.log(prior.beta)
        - nu * sp.log(beta)
        + sp.Rational(1, 2) * (sp.log(prior.lambda_) - sp.log(lam))
        - sp.Rational(n, 2) * sp.log(2 * sp.pi)
    )
    return ConjugateUpdateResult(
        prior,
        post,
        prior.predictive,
        post.predictive,
        logz,
        n,
        {"n": n, "mean": mean, "sum_squared_deviations": ss},
    )
