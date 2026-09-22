"""Closed-form registry for information measures.

The registry keeps distribution-specific formulas separate from the generic
symbolic integration engine in :mod:`probstats.information`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import sympy as sp

from ._exact_linear_algebra import exact_determinant, exact_solve, quadratic_form
from ._symbolic_predicates import TruthValue, certified_equal
from .distributions.basic import (
    Bernoulli,
    Beta,
    Binomial,
    Categorical,
    Dirichlet,
    InverseWishart,
    Multinomial,
    MultivariateNormal,
    Wishart,
)
from .distributions.exponential_family import (
    Exponential,
    Gamma,
    LogNormal,
    Normal,
    Poisson,
)
from .distributions.matrix import MatrixNormal
from .distributions.student_t import StudentT

InformationFormula = Callable[..., Any]
_REGISTRY: dict[tuple[str, type, type], InformationFormula] = {}


def register_information_formula(measure: str, p_type: type, q_type: type):
    """Register a closed-form formula for a distribution pair."""
    key = (measure, p_type, q_type)

    def decorator(function: InformationFormula) -> InformationFormula:
        if key in _REGISTRY:
            raise ValueError(f"information formula already registered for {key}")
        _REGISTRY[key] = function
        return function

    return decorator


def get_information_formula(measure: str, p: object, q: object):
    """Return the most specific registered formula, if one exists."""
    for p_type in type(p).__mro__:
        for q_type in type(q).__mro__:
            formula = _REGISTRY.get((measure, p_type, q_type))
            if formula is not None:
                return formula
    return None


def registered_information_formulas() -> tuple[tuple[str, type, type], ...]:
    """Return the registered measure/type keys in deterministic order."""
    return tuple(sorted(_REGISTRY, key=lambda k: (k[0], k[1].__name__, k[2].__name__)))


@register_information_formula("kl", Normal, Normal)
def _normal_kl(p: Normal, q: Normal):
    variance_ratio = (p.sigma**2 + (p.mean - q.mean) ** 2) / q.sigma**2
    return sp.simplify(sp.log(q.sigma / p.sigma) + (variance_ratio - 1) / 2)


@register_information_formula("kl", MultivariateNormal, MultivariateNormal)
def _mvn_kl(p: MultivariateNormal, q: MultivariateNormal):
    if p.dimension != q.dimension:
        return sp.oo
    d = p.dimension
    delta = q.mean - p.mean
    quadratic = quadratic_form(q.covariance, delta)
    return sp.simplify(
        (
            sp.trace(exact_solve(q.covariance, p.covariance))
            + quadratic
            - d
            + sp.log(q.covariance.det() / p.covariance.det())
        )
        / 2
    )


@register_information_formula("kl", Gamma, Gamma)
def _gamma_kl(p: Gamma, q: Gamma):
    kp, tp = p.shape, p.scale
    kq, tq = q.shape, q.scale
    return sp.simplify(
        (kp - kq) * sp.digamma(kp)
        - sp.loggamma(kp)
        + sp.loggamma(kq)
        + kq * sp.log(tq / tp)
        + kp * (tp / tq - 1)
    )


@register_information_formula("kl", Beta, Beta)
def _beta_kl(p: Beta, q: Beta):
    ap, bp = p.alpha, p.beta
    aq, bq = q.alpha, q.beta
    return sp.simplify(
        sp.log(sp.beta(aq, bq) / sp.beta(ap, bp))
        + (ap - aq) * sp.digamma(ap)
        + (bp - bq) * sp.digamma(bp)
        + (aq + bq - ap - bp) * sp.digamma(ap + bp)
    )


def _log_multibeta(alpha):
    total = sp.Add(*alpha)
    return sp.Add(*(sp.loggamma(a) for a in alpha)) - sp.loggamma(total)


@register_information_formula("kl", Dirichlet, Dirichlet)
def _dirichlet_kl(p: Dirichlet, q: Dirichlet):
    if p.dimension != q.dimension:
        return sp.oo
    ap, aq = p.concentration, q.concentration
    a0 = sp.Add(*ap)
    return sp.simplify(
        _log_multibeta(aq)
        - _log_multibeta(ap)
        + sp.Add(*((a - b) * (sp.digamma(a) - sp.digamma(a0)) for a, b in zip(ap, aq)))
    )


@register_information_formula("entropy", Normal, Normal)
def _normal_entropy(p: Normal, _q: Normal):
    return sp.log(p.sigma * sp.sqrt(2 * sp.pi * sp.E))


@register_information_formula("entropy", MultivariateNormal, MultivariateNormal)
def _mvn_entropy(p: MultivariateNormal, _q: MultivariateNormal):
    d = p.dimension
    return sp.simplify((d * (1 + sp.log(2 * sp.pi)) + sp.log(p.covariance.det())) / 2)


@register_information_formula("entropy", Gamma, Gamma)
def _gamma_entropy(p: Gamma, _q: Gamma):
    return sp.simplify(
        p.shape
        + sp.log(p.scale)
        + sp.loggamma(p.shape)
        + (1 - p.shape) * sp.digamma(p.shape)
    )


@register_information_formula("entropy", Beta, Beta)
def _beta_entropy(p: Beta, _q: Beta):
    a, b = p.alpha, p.beta
    return sp.simplify(
        sp.log(sp.beta(a, b))
        - (a - 1) * sp.digamma(a)
        - (b - 1) * sp.digamma(b)
        + (a + b - 2) * sp.digamma(a + b)
    )


@register_information_formula("entropy", Dirichlet, Dirichlet)
def _dirichlet_entropy(p: Dirichlet, _q: Dirichlet):
    alpha = p.concentration
    a0 = sp.Add(*alpha)
    k = p.dimension
    return sp.simplify(
        _log_multibeta(alpha)
        + (a0 - k) * sp.digamma(a0)
        - sp.Add(*((a - 1) * sp.digamma(a) for a in alpha))
    )


def _validate_renyi_order(alpha):
    a = sp.sympify(alpha)
    if a.is_positive is False or a == 0 or a == 1:
        raise ValueError("closed-form Rényi formula requires alpha > 0 and alpha != 1")
    return a


@register_information_formula("renyi", Normal, Normal)
def _normal_renyi(p: Normal, q: Normal, alpha):
    # This is the one-dimensional specialization of the Gaussian precision formula.
    a = _validate_renyi_order(alpha)
    vp, vq = p.sigma**2, q.sigma**2
    precision = a / vp + (1 - a) / vq
    linear = a * p.mean / vp + (1 - a) * q.mean / vq
    constant = a * p.mean**2 / vp + (1 - a) * q.mean**2 / vq
    log_integral = sp.simplify(
        -a * sp.log(vp) / 2
        - (1 - a) * sp.log(vq) / 2
        - sp.log(precision) / 2
        - (constant - linear**2 / precision) / 2
    )
    return sp.simplify(log_integral / (a - 1))


@register_information_formula("renyi", MultivariateNormal, MultivariateNormal)
def _mvn_renyi(p: MultivariateNormal, q: MultivariateNormal, alpha):
    if p.dimension != q.dimension:
        return sp.oo
    a = _validate_renyi_order(alpha)
    pp = exact_solve(p.covariance, sp.eye(p.dimension))
    pq = exact_solve(q.covariance, sp.eye(q.dimension))
    precision = a * pp + (1 - a) * pq
    linear = a * pp * p.mean + (1 - a) * pq * q.mean
    constant = a * (p.mean.T * pp * p.mean)[0] + (1 - a) * (q.mean.T * pq * q.mean)[0]
    log_integral = sp.simplify(
        -a * sp.log(p.covariance.det()) / 2
        - (1 - a) * sp.log(q.covariance.det()) / 2
        - sp.log(exact_determinant(precision)) / 2
        - (constant - quadratic_form(precision, linear)) / 2
    )
    return sp.simplify(log_integral / (a - 1))


@register_information_formula("renyi", Gamma, Gamma)
def _gamma_renyi(p: Gamma, q: Gamma, alpha):
    a = _validate_renyi_order(alpha)
    shape = sp.simplify(a * p.shape + (1 - a) * q.shape)
    rate = sp.simplify(a / p.scale + (1 - a) / q.scale)
    log_integral = sp.simplify(
        sp.loggamma(shape)
        - a * sp.loggamma(p.shape)
        - (1 - a) * sp.loggamma(q.shape)
        - a * p.shape * sp.log(p.scale)
        - (1 - a) * q.shape * sp.log(q.scale)
        - shape * sp.log(rate)
    )
    return sp.simplify(log_integral / (a - 1))


@register_information_formula("renyi", Beta, Beta)
def _beta_renyi(p: Beta, q: Beta, alpha):
    a = _validate_renyi_order(alpha)
    ar = sp.simplify(a * p.alpha + (1 - a) * q.alpha)
    br = sp.simplify(a * p.beta + (1 - a) * q.beta)
    log_integral = sp.simplify(
        sp.log(sp.beta(ar, br))
        - a * sp.log(sp.beta(p.alpha, p.beta))
        - (1 - a) * sp.log(sp.beta(q.alpha, q.beta))
    )
    return sp.simplify(log_integral / (a - 1))


@register_information_formula("renyi", Dirichlet, Dirichlet)
def _dirichlet_renyi(p: Dirichlet, q: Dirichlet, alpha):
    if p.dimension != q.dimension:
        return sp.oo
    a = _validate_renyi_order(alpha)
    mixed = tuple(
        sp.simplify(a * x + (1 - a) * y)
        for x, y in zip(p.concentration, q.concentration)
    )
    log_integral = sp.simplify(
        _log_multibeta(mixed)
        - a * _log_multibeta(p.concentration)
        - (1 - a) * _log_multibeta(q.concentration)
    )
    return sp.simplify(log_integral / (a - 1))


def _multigamma_log(a, dimension: int):
    return sp.Rational(dimension * (dimension - 1), 4) * sp.log(sp.pi) + sp.Add(
        *(sp.loggamma(a + sp.Rational(1 - j, 2)) for j in range(1, dimension + 1))
    )


def _multidigamma(a, dimension: int):
    return sp.Add(
        *(sp.digamma(a + sp.Rational(1 - j, 2)) for j in range(1, dimension + 1))
    )


@register_information_formula("kl", Bernoulli, Bernoulli)
def _bernoulli_kl(p: Bernoulli, q: Bernoulli):
    return sp.simplify(
        p.p * sp.log(p.p / q.p) + (1 - p.p) * sp.log((1 - p.p) / (1 - q.p))
    )


@register_information_formula("kl", Categorical, Categorical)
def _categorical_kl(p: Categorical, q: Categorical):
    if len(p.probabilities) > len(q.probabilities):
        tail = p.probabilities[len(q.probabilities) :]
        if any(x.is_zero is not True for x in tail):
            return sp.oo
    terms = []
    for i, pp in enumerate(p.probabilities):
        qq = q.probabilities[i] if i < len(q.probabilities) else sp.S.Zero
        terms.append(pp * sp.log(pp / qq))
    return sp.simplify(sp.Add(*terms))


@register_information_formula("kl", Binomial, Binomial)
def _binomial_kl(p: Binomial, q: Binomial):
    if certified_equal(p.n, q.n) is not TruthValue.TRUE:
        raise NotImplementedError(
            "closed-form Binomial KL requires matching trial counts"
        )
    return sp.simplify(p.n * _bernoulli_kl(Bernoulli(p.p), Bernoulli(q.p)))


@register_information_formula("kl", Multinomial, Multinomial)
def _multinomial_kl(p: Multinomial, q: Multinomial):
    if certified_equal(p.n, q.n) is not TruthValue.TRUE or p.dimension != q.dimension:
        raise NotImplementedError(
            "closed-form Multinomial KL requires matching trial counts and dimensions"
        )
    return sp.simplify(
        p.n
        * _categorical_kl(Categorical(p.probabilities), Categorical(q.probabilities))
    )


@register_information_formula("kl", Poisson, Poisson)
def _poisson_kl(p: Poisson, q: Poisson):
    return sp.simplify(p.rate * sp.log(p.rate / q.rate) + q.rate - p.rate)


@register_information_formula("kl", Exponential, Exponential)
def _exponential_kl(p: Exponential, q: Exponential):
    return sp.simplify(sp.log(p.rate / q.rate) + q.rate / p.rate - 1)


@register_information_formula("kl", LogNormal, LogNormal)
def _lognormal_kl(p: LogNormal, q: LogNormal):
    variance_ratio = (p.sigma**2 + (p.mean - q.mean) ** 2) / q.sigma**2
    return sp.simplify(sp.log(q.sigma / p.sigma) + (variance_ratio - 1) / 2)


@register_information_formula("kl", MatrixNormal, MatrixNormal)
def _matrix_normal_kl(p: MatrixNormal, q: MatrixNormal):
    if p.mean.shape != q.mean.shape:
        return sp.oo
    n, m = p.rows, p.cols
    delta = q.mean - p.mean
    trace_term = sp.trace(exact_solve(q.row_covariance, p.row_covariance)) * sp.trace(
        exact_solve(q.column_covariance, p.column_covariance)
    )
    row_solved = exact_solve(q.row_covariance, delta)
    quadratic = sp.trace(exact_solve(q.column_covariance, delta.T * row_solved))
    logdet_ratio = n * sp.log(
        exact_determinant(q.column_covariance) / exact_determinant(p.column_covariance)
    ) + m * sp.log(
        exact_determinant(q.row_covariance) / exact_determinant(p.row_covariance)
    )
    return sp.simplify((trace_term + quadratic - n * m + logdet_ratio) / 2)


@register_information_formula("kl", Wishart, Wishart)
def _wishart_kl(p: Wishart, q: Wishart):
    if p.dimension != q.dimension:
        return sp.oo
    d = p.dimension
    vp, vq = p.df, q.df
    return sp.simplify(
        vq / 2 * sp.log(exact_determinant(q.scale) / exact_determinant(p.scale))
        + vp / 2 * (sp.trace(exact_solve(q.scale, p.scale)) - d)
        + _multigamma_log(vq / 2, d)
        - _multigamma_log(vp / 2, d)
        + (vp - vq) / 2 * _multidigamma(vp / 2, d)
    )


@register_information_formula("kl", InverseWishart, InverseWishart)
def _inverse_wishart_kl(p: InverseWishart, q: InverseWishart):
    if p.dimension != q.dimension:
        return sp.oo
    d = p.dimension
    vp, vq = p.df, q.df
    return sp.simplify(
        vq / 2 * sp.log(exact_determinant(p.scale) / exact_determinant(q.scale))
        + vp / 2 * (sp.trace(exact_solve(p.scale, q.scale)) - d)
        + _multigamma_log(vq / 2, d)
        - _multigamma_log(vp / 2, d)
        + (vp - vq) / 2 * _multidigamma(vp / 2, d)
    )


@register_information_formula("entropy", Bernoulli, Bernoulli)
def _bernoulli_entropy(p: Bernoulli, _q: Bernoulli):
    return sp.simplify(-p.p * sp.log(p.p) - (1 - p.p) * sp.log(1 - p.p))


@register_information_formula("entropy", Categorical, Categorical)
def _categorical_entropy(p: Categorical, _q: Categorical):
    return sp.simplify(-sp.Add(*(x * sp.log(x) for x in p.probabilities)))


@register_information_formula("entropy", Exponential, Exponential)
def _exponential_entropy(p: Exponential, _q: Exponential):
    return sp.simplify(1 - sp.log(p.rate))


@register_information_formula("entropy", LogNormal, LogNormal)
def _lognormal_entropy(p: LogNormal, _q: LogNormal):
    return sp.simpl(p.mean + sp.log(p.sigma * sp.sqrt(2 * sp.pi * sp.E)))


@register_information_formula("entropy", StudentT, StudentT)
def _student_t_entropy(p: StudentT, _q: StudentT):
    v = p.df
    return sp.simplify(
        sp.log(sp.sqrt(v) * sp.beta(v / 2, sp.Rational(1, 2)) * p.scale)
        + (v + 1) / 2 * (sp.digamma((v + 1) / 2) - sp.digamma(v / 2))
    )


@register_information_formula("entropy", MatrixNormal, MatrixNormal)
def _matrix_normal_entropy(p: MatrixNormal, _q: MatrixNormal):
    n, m = p.rows, p.cols
    return sp.simplify(
        sp.Rational(n * m, 2) * (1 + sp.log(2 * sp.pi))
        + sp.Rational(m, 2) * sp.log(exact_determinant(p.row_covariance))
        + sp.Rational(n, 2) * sp.log(exact_determinant(p.column_covariance))
    )


@register_information_formula("entropy", Wishart, Wishart)
def _wishart_entropy(p: Wishart, _q: Wishart):
    d, v = p.dimension, p.df
    elogdet = (
        _multidigamma(v / 2, d) + d * sp.log(2) + sp.log(exact_determinant(p.scale))
    )
    return sp.simplify(
        v * d / 2 * sp.log(2)
        + v / 2 * sp.log(exact_determinant(p.scale))
        + _multigamma_log(v / 2, d)
        - (v - d - 1) / 2 * elogdet
        + v * d / 2
    )


@register_information_formula("entropy", InverseWishart, InverseWishart)
def _inverse_wishart_entropy(p: InverseWishart, _q: InverseWishart):
    d, v = p.dimension, p.df
    elogdet = (
        sp.log(exact_determinant(p.scale)) - d * sp.log(2) - _multidigamma(v / 2, d)
    )
    return sp.simplify(
        -v / 2 * sp.log(exact_determinant(p.scale))
        + (v + d + 1) / 2 * elogdet
        + v * d / 2 * sp.log(2)
        + _multigamma_log(v / 2, d)
        + v * d / 2
    )


@register_information_formula("renyi", Bernoulli, Bernoulli)
def _bernoulli_renyi(p: Bernoulli, q: Bernoulli, alpha):
    a = _validate_renyi_order(alpha)
    integral = p.p**a * q.p ** (1 - a) + (1 - p.p) ** a * (1 - q.p) ** (1 - a)
    return sp.simplify(sp.log(integral) / (a - 1))


@register_information_formula("renyi", Categorical, Categorical)
def _categorical_renyi(p: Categorical, q: Categorical, alpha):
    a = _validate_renyi_order(alpha)
    size = max(len(p.probabilities), len(q.probabilities))
    terms = []
    for i in range(size):
        pp = p.probabilities[i] if i < len(p.probabilities) else sp.S.Zero
        qq = q.probabilities[i] if i < len(q.probabilities) else sp.S.Zero
        terms.append(pp**a * qq ** (1 - a))
    return sp.simplify(sp.log(sp.Add(*terms)) / (a - 1))


@register_information_formula("renyi", Binomial, Binomial)
def _binomial_renyi(p: Binomial, q: Binomial, alpha):
    if certified_equal(p.n, q.n) is not TruthValue.TRUE:
        raise NotImplementedError(
            "closed-form Binomial Rényi divergence requires matching trial counts"
        )
    return sp.simplify(p.n * _bernoulli_renyi(Bernoulli(p.p), Bernoulli(q.p), alpha))


@register_information_formula("renyi", Multinomial, Multinomial)
def _multinomial_renyi(p: Multinomial, q: Multinomial, alpha):
    if certified_equal(p.n, q.n) is not TruthValue.TRUE or p.dimension != q.dimension:
        raise NotImplementedError(
            "closed-form Multinomial Rényi divergence requires matching trial counts "
            "and dimensions"
        )
    return sp.simplify(
        p.n
        * _categorical_renyi(
            Categorical(p.probabilities), Categorical(q.probabilities), alpha
        )
    )


@register_information_formula("renyi", Poisson, Poisson)
def _poisson_renyi(p: Poisson, q: Poisson, alpha):
    a = _validate_renyi_order(alpha)
    log_integral = -a * p.rate - (1 - a) * q.rate + p.rate**a * q.rate ** (1 - a)
    return sp.simplify(log_integral / (a - 1))


@register_information_formula("renyi", Exponential, Exponential)
def _exponential_renyi(p: Exponential, q: Exponential, alpha):
    a = _validate_renyi_order(alpha)
    mixed_rate = sp.simplify(a * p.rate + (1 - a) * q.rate)
    if mixed_rate.is_positive is False:
        return sp.oo
    log_integral = a * sp.log(p.rate) + (1 - a) * sp.log(q.rate) - sp.log(mixed_rate)
    return sp.simplify(log_integral / (a - 1))


@register_information_formula("renyi", LogNormal, LogNormal)
def _lognormal_renyi(p: LogNormal, q: LogNormal, alpha):
    # KL/Rényi divergence is invariant under the common bijection x -> log(x).
    return _normal_renyi(Normal(p.mean, p.sigma), Normal(q.mean, q.sigma), alpha)


@register_information_formula("renyi", MatrixNormal, MatrixNormal)
def _matrix_normal_renyi(p: MatrixNormal, q: MatrixNormal, alpha):
    if p.mean.shape != q.mean.shape:
        return sp.oo
    a = _validate_renyi_order(alpha)
    cp = sp.kronecker_product(p.column_covariance, p.row_covariance)
    cq = sp.kronecker_product(q.column_covariance, q.row_covariance)
    mp = sp.ImmutableMatrix(list(p.mean))
    mq = sp.ImmutableMatrix(list(q.mean))
    pp = exact_solve(cp, sp.eye(cp.rows))
    pq = exact_solve(cq, sp.eye(cq.rows))
    precision = a * pp + (1 - a) * pq
    if precision.is_positive_definite is False:
        return sp.oo
    linear = a * pp * mp + (1 - a) * pq * mq
    constant = a * (mp.T * pp * mp)[0] + (1 - a) * (mq.T * pq * mq)[0]
    log_integral = sp.simplify(
        -a * sp.log(exact_determinant(cp)) / 2
        - (1 - a) * sp.log(exact_determinant(cq)) / 2
        - sp.log(exact_determinant(precision)) / 2
        - (constant - quadratic_form(precision, linear)) / 2
    )
    return sp.simplify(log_integral / (a - 1))


@register_information_formula("renyi", Wishart, Wishart)
def _wishart_renyi(p: Wishart, q: Wishart, alpha):
    if p.dimension != q.dimension:
        return sp.oo
    a = _validate_renyi_order(alpha)
    d = p.dimension
    vr = sp.simplify(a * p.df + (1 - a) * q.df)
    p_precision = exact_solve(p.scale, sp.eye(d))
    q_precision = exact_solve(q.scale, sp.eye(d))
    precision = a * p_precision + (1 - a) * q_precision
    if (vr - (d - 1)).is_positive is False or precision.is_positive_definite is False:
        return sp.oo
    log_integral = sp.simplify(
        -vr / 2 * sp.log(exact_determinant(precision))
        + _multigamma_log(vr / 2, d)
        - a
        * (p.df / 2 * sp.log(exact_determinant(p.scale)) + _multigamma_log(p.df / 2, d))
        - (1 - a)
        * (q.df / 2 * sp.log(exact_determinant(q.scale)) + _multigamma_log(q.df / 2, d))
    )
    return sp.simplify(log_integral / (a - 1))


@register_information_formula("renyi", InverseWishart, InverseWishart)
def _inverse_wishart_renyi(p: InverseWishart, q: InverseWishart, alpha):
    if p.dimension != q.dimension:
        return sp.oo
    a = _validate_renyi_order(alpha)
    d = p.dimension
    vr = sp.simplify(a * p.df + (1 - a) * q.df)
    sr = sp.simplify(a * p.scale + (1 - a) * q.scale)
    if (vr - (d - 1)).is_positive is False or sr.is_positive_definite is False:
        return sp.oo
    log_integral = sp.simplify(
        a
        * (p.df / 2 * sp.log(exact_determinant(p.scale)) - _multigamma_log(p.df / 2, d))
        + (1 - a)
        * (q.df / 2 * sp.log(exact_determinant(q.scale)) - _multigamma_log(q.df / 2, d))
        - vr / 2 * sp.log(exact_determinant(sr))
        + _multigamma_log(vr / 2, d)
    )
    return sp.simplify(log_integral / (a - 1))


__all__ = [
    "get_information_formula",
    "register_information_formula",
    "registered_information_formulas",
]
