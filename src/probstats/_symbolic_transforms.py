"""Exact univariate probability transforms."""

from __future__ import annotations

from typing import Any

import sympy as sp

from .distributions import (
    Bernoulli,
    Binomial,
    Cauchy,
    ChiSquared,
    DiscreteUniform,
    Exponential,
    Gamma,
    Geometric,
    Gumbel,
    InverseGaussian,
    Laplace,
    Logistic,
    LogNormal,
    NegativeBinomial,
    NoncentralT,
    Normal,
    Poisson,
    Skellam,
    StudentT,
    Uniform,
    Zipf,
)
from .distributions.base import Distribution
from .spaces import MeasureType


def _s(x: Any) -> sp.Expr:
    return sp.sympify(x)


def moment_generating_function(distribution: Distribution, t: Any | None = None):
    """Return the MGF ``E[exp(t X)]`` when a closed form is known.

    The returned expression is symbolic and may carry Piecewise convergence
    conditions for heavy-tailed/one-sided families.
    """
    t = sp.Symbol("t", real=True) if t is None else _s(t)
    d = distribution
    if isinstance(d, Normal):
        return sp.exp(d.mean * t + d.sigma**2 * t**2 / 2)
    if isinstance(d, Bernoulli):
        return 1 - d.p + d.p * sp.exp(t)
    if isinstance(d, Binomial):
        return (1 - d.p + d.p * sp.exp(t)) ** d.n
    if isinstance(d, Poisson):
        return sp.exp(d.rate * (sp.exp(t) - 1))
    if isinstance(d, Exponential):
        return sp.Piecewise((d.rate / (d.rate - t), t < d.rate), (sp.oo, True))
    if isinstance(d, Gamma):
        return sp.Piecewise(
            ((1 - d.scale * t) ** (-d.shape), t < 1 / d.scale), (sp.oo, True)
        )
    if isinstance(d, ChiSquared):
        return sp.Piecewise(
            ((1 - 2 * t) ** (-d.df / 2), t < sp.Rational(1, 2)), (sp.oo, True)
        )
    if isinstance(d, InverseGaussian):
        bound = d.shape / (2 * d.mean**2)
        value = sp.exp(
            d.shape / d.mean * (1 - sp.sqrt(1 - 2 * d.mean**2 * t / d.shape))
        )
        return sp.Piecewise((value, t <= bound), (sp.oo, True))
    if isinstance(d, Gumbel):
        value = sp.exp(d.location * t) * sp.gamma(1 - d.scale * t)
        return sp.Piecewise((value, t < 1 / d.scale), (sp.oo, True))
    if isinstance(d, Geometric):
        return sp.Piecewise(
            (d.p * sp.exp(t) / (1 - (1 - d.p) * sp.exp(t)), (1 - d.p) * sp.exp(t) < 1),
            (sp.oo, True),
        )
    if isinstance(d, NegativeBinomial):
        return sp.Piecewise(
            ((d.p / (1 - (1 - d.p) * sp.exp(t))) ** d.r, (1 - d.p) * sp.exp(t) < 1),
            (sp.oo, True),
        )
    if isinstance(d, Laplace):
        return sp.Piecewise(
            (sp.exp(d.location * t) / (1 - d.scale**2 * t**2), sp.Abs(t) < 1 / d.scale),
            (sp.oo, True),
        )
    if isinstance(d, Logistic):
        expression = (
            sp.exp(d.location * t) * sp.pi * d.scale * t / sp.sin(sp.pi * d.scale * t)
        )
        return sp.Piecewise(
            (expression, sp.And(sp.Ne(t, 0), sp.Abs(t) < 1 / d.scale)),
            (1, sp.Eq(t, 0)),
            (sp.oo, True),
        )
    if isinstance(d, Uniform):
        return sp.Piecewise(
            (
                (sp.exp(t * d.high) - sp.exp(t * d.low)) / (t * (d.high - d.low)),
                sp.Ne(t, 0),
            ),
            (1, True),
        )
    if isinstance(d, LogNormal):
        x = sp.Symbol("x", positive=True)
        negative_branch = sp.Integral(sp.exp(t * x) * d.pdf(x), (x, 0, sp.oo))
        return sp.Piecewise(
            (1, sp.Eq(t, 0)),
            (negative_branch, t < 0),
            (sp.oo, True),
        )
    if isinstance(d, (Cauchy, StudentT, NoncentralT)):
        return sp.Piecewise((1, sp.Eq(t, 0)), (sp.oo, True))
    x = sp.Symbol("x", real=True)
    try:
        expression = distribution.expectation(sp.exp(t * x), variable=x)
    except (TypeError, ValueError, NotImplementedError, sp.PolynomialError):
        expression = sp.Integral(
            sp.exp(t * x) * distribution.pdf(x), (x, distribution.support)
        )
    return sp.Piecewise((1, sp.Eq(t, 0)), (expression, True))


def cumulant_generating_function(distribution: Distribution, t: Any | None = None):
    """Return the cumulant-generating function ``log(E[exp(t X)])``."""
    t = sp.Symbol("t", real=True) if t is None else _s(t)
    return sp.log(moment_generating_function(distribution, t))


def central_moment_generating_function(
    distribution: Distribution, t: Any | None = None
):
    """Return the MGF of the centered variable ``X - E[X]``."""
    from .functionals import mean

    t = sp.Symbol("t", real=True) if t is None else _s(t)
    return sp.simplify(
        sp.exp(-mean(distribution) * t) * moment_generating_function(distribution, t)
    )


def characteristic_function(distribution: Distribution, t: Any | None = None):
    """Return the characteristic function ``E[exp(i t X)]``."""
    t = sp.Symbol("t", real=True) if t is None else _s(t)
    d = distribution
    if isinstance(d, Normal):
        return sp.exp(sp.I * d.mean * t - d.sigma**2 * t**2 / 2)
    if isinstance(d, Cauchy):
        return sp.exp(sp.I * d.location * t - d.scale * sp.Abs(t))
    if isinstance(d, Laplace):
        return sp.exp(sp.I * d.location * t) / (1 + d.scale**2 * t**2)
    if isinstance(d, Logistic):
        expression = (
            sp.exp(sp.I * d.location * t)
            * sp.pi
            * d.scale
            * t
            / sp.sinh(sp.pi * d.scale * t)
        )
        return sp.Piecewise((1, sp.Eq(t, 0)), (expression, True))
    if isinstance(d, Uniform):
        mid = (d.low + d.high) / 2
        width = d.high - d.low
        return sp.exp(sp.I * mid * t) * sp.sinc(width * t / 2)
    if isinstance(d, Exponential):
        return d.rate / (d.rate - sp.I * t)
    if isinstance(d, Gamma):
        return (1 - sp.I * d.scale * t) ** (-d.shape)
    if isinstance(d, ChiSquared):
        return (1 - 2 * sp.I * t) ** (-d.df / 2)
    if isinstance(d, Geometric):
        return d.p * sp.exp(sp.I * t) / (1 - (1 - d.p) * sp.exp(sp.I * t))
    if isinstance(d, NegativeBinomial):
        return (d.p / (1 - (1 - d.p) * sp.exp(sp.I * t))) ** d.r
    if isinstance(d, Skellam):
        return sp.exp(
            d.rate1 * (sp.exp(sp.I * t) - 1) + d.rate2 * (sp.exp(-sp.I * t) - 1)
        )
    if isinstance(d, Gumbel):
        return sp.exp(sp.I * d.location * t) * sp.gamma(1 - sp.I * d.scale * t)
    if isinstance(d, InverseGaussian):
        return sp.exp(
            d.shape / d.mean * (1 - sp.sqrt(1 - 2 * sp.I * d.mean**2 * t / d.shape))
        )
    if isinstance(d, DiscreteUniform):
        n = d.high - d.low + 1
        q = sp.exp(sp.I * t)
        expr = q**d.low * (1 - q**n) / (n * (1 - q))
        return sp.Piecewise((1, sp.Eq(t, 0)), (expr, True))
    if isinstance(d, Poisson):
        return sp.exp(d.rate * (sp.exp(sp.I * t) - 1))
    if isinstance(d, Bernoulli):
        return 1 - d.p + d.p * sp.exp(sp.I * t)
    if isinstance(d, Binomial):
        return (1 - d.p + d.p * sp.exp(sp.I * t)) ** d.n
    x = sp.Symbol("x", real=True)
    try:
        expression = distribution.expectation(sp.exp(sp.I * t * x), variable=x)
    except (TypeError, ValueError, NotImplementedError, sp.PolynomialError):
        expression = sp.Integral(
            sp.exp(sp.I * t * x) * distribution.pdf(x), (x, distribution.support)
        )
    return sp.Piecewise((1, sp.Eq(t, 0)), (expression, True))


def probability_generating_function(distribution: Distribution, z: Any | None = None):
    """Return the PGF ``E[z**X]`` for nonnegative integer-valued laws."""
    if distribution.measure_type is not MeasureType.DISCRETE:
        raise TypeError("PGF is defined here only for discrete distributions")
    z = sp.Symbol("z") if z is None else _s(z)
    d = distribution
    if isinstance(d, Bernoulli):
        return 1 - d.p + d.p * z
    if isinstance(d, Binomial):
        return (1 - d.p + d.p * z) ** d.n
    if isinstance(d, Poisson):
        return sp.exp(d.rate * (z - 1))
    if isinstance(d, Geometric):
        return d.p * z / (1 - (1 - d.p) * z)
    if isinstance(d, NegativeBinomial):
        return (d.p / (1 - (1 - d.p) * z)) ** d.r
    if isinstance(d, DiscreteUniform):
        n = d.high - d.low + 1
        expr = z**d.low * (1 - z**n) / (n * (1 - z))
        return sp.Piecewise((1, sp.Eq(z, 1)), (expr, True))
    if isinstance(d, Zipf):
        return sp.polylog(d.exponent, z) / sp.zeta(d.exponent)
    x = sp.Symbol("x", integer=True, nonnegative=True)
    return distribution.expectation(z**x, variable=x)


def factorial_moment_generating_function(
    distribution: Distribution, t: Any | None = None
):
    """Return ``E[(1+t)**X]``, the factorial-moment generating function.

    Its ``n``-th derivative at zero is the ``n``-th falling-factorial moment.
    """
    if distribution.measure_type is not MeasureType.DISCRETE:
        raise TypeError(
            "factorial-moment generating functions require a discrete distribution"
        )
    t = sp.Symbol("t") if t is None else _s(t)
    return sp.simplify(probability_generating_function(distribution, 1 + t))


__all__ = [
    "central_moment_generating_function",
    "characteristic_function",
    "cumulant_generating_function",
    "factorial_moment_generating_function",
    "moment_generating_function",
    "probability_generating_function",
]
