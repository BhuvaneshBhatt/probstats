"""Standard univariate and compound probability distributions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import reduce
from operator import and_
from typing import Any

import numpy as np
import sympy as sp

from ..spaces import (
    CountVectorEventSpace,
    MeasureType,
    NaturalNumberSpace,
    ParameterSpace,
    PositiveRealSpace,
    PositiveVectorSpace,
    RealSpace,
    UnitIntervalSpace,
)
from .base import Distribution, scalar_batch, validate_distribution_parameters
from .basic import Beta, Binomial, Dirichlet, Multinomial


def _and(*conditions):
    return reduce(and_, (sp.sympify(c) for c in conditions), sp.true)


def _sym(value):
    return sp.sympify(value)


def _float(value, name):
    value = sp.sympify(value)
    if value.free_symbols:
        from ..sampling import SamplingError

        raise SamplingError(f"{name} must be numeric for sampling")
    return float(value)


def _int(value, name):
    value = sp.sympify(value)
    if value.free_symbols or value.is_integer is not True:
        from ..sampling import SamplingError

        raise SamplingError(f"{name} must be a concrete integer for sampling")
    return int(value)


@dataclass(frozen=True, slots=True)
class Uniform(Distribution):
    low: sp.Expr
    high: sp.Expr

    def __post_init__(self):
        object.__setattr__(self, "low", _sym(self.low))
        object.__setattr__(self, "high", _sym(self.high))
        validate_distribution_parameters(self)

    @property
    def parameters(self):
        return (self.low, self.high)

    @property
    def support(self):
        return sp.Interval(self.low, self.high)

    @property
    def parameter_constraints(self):
        return self.high > self.low

    @property
    def parameter_space(self):
        return ParameterSpace(
            (("low", RealSpace()), ("high", RealSpace())),
            relations=(lambda m: m["high"] > m["low"],),
        )

    @scalar_batch
    def logpdf(self, value):
        x = _sym(value)
        return sp.Piecewise(
            (-sp.log(self.high - self.low), _and(x >= self.low, x <= self.high)),
            (-sp.oo, True),
        )

    def _cdf(self, x):
        return sp.Piecewise(
            (0, x < self.low),
            ((x - self.low) / (self.high - self.low), x < self.high),
            (1, True),
        )

    def _quantile(self, p):
        return self.low + p * (self.high - self.low)

    def _mean(self):
        return (self.low + self.high) / 2

    def _variance(self):
        return (self.high - self.low) ** 2 / 12

    def _raw_moment(self, n):
        return (self.high ** (n + 1) - self.low ** (n + 1)) / (
            (n + 1) * (self.high - self.low)
        )

    def _entropy(self):
        return sp.log(self.high - self.low)

    def _sample(self, size, rng):
        return rng.uniform(
            _float(self.low, "low"), _float(self.high, "high"), size=size
        )


@dataclass(frozen=True, slots=True)
class Geometric(Distribution):
    """Number of trials until the first success (support 1, 2, ...)."""

    p: sp.Expr

    def __post_init__(self):
        object.__setattr__(self, "p", _sym(self.p))
        validate_distribution_parameters(self)

    @property
    def parameters(self):
        return (self.p,)

    @property
    def support(self):
        return sp.S.Naturals

    @property
    def parameter_constraints(self):
        return _and(self.p > 0, self.p <= 1)

    @property
    def parameter_space(self):
        return ParameterSpace((("p", UnitIntervalSpace(open_left=True)),))

    @scalar_batch
    def pdf(self, value):
        k = _sym(value)
        valid = _and(k >= 1, sp.Eq(sp.floor(k), k))
        mass = sp.Piecewise(
            (1, sp.And(sp.Eq(self.p, 1), sp.Eq(k, 1))),
            (self.p * (1 - self.p) ** (k - 1), valid),
            (0, True),
        )
        return mass

    @scalar_batch
    def logpdf(self, value):
        return sp.log(self.pdf(value))

    def _cdf(self, x):
        return sp.Piecewise((0, x < 1), (1 - (1 - self.p) ** sp.floor(x), True))

    def _quantile(self, q):
        return sp.Piecewise(
            (sp.oo, sp.Eq(q, 1)),
            (1, sp.Eq(self.p, 1)),
            (sp.ceiling(sp.log(1 - q) / sp.log(1 - self.p)), True),
        )

    def _mean(self):
        return 1 / self.p

    def _variance(self):
        return (1 - self.p) / self.p**2

    def _entropy(self):
        return -sp.log(self.p) - (1 - self.p) / self.p * sp.log(1 - self.p)

    def _sample(self, size, rng):
        return rng.geometric(_float(self.p, "p"), size=size)


@dataclass(frozen=True, slots=True)
class NegativeBinomial(Distribution):
    """Failures before ``r`` successes, with success probability ``p``."""

    r: sp.Expr
    p: sp.Expr

    def __post_init__(self):
        object.__setattr__(self, "r", _sym(self.r))
        object.__setattr__(self, "p", _sym(self.p))
        validate_distribution_parameters(self)

    @property
    def parameters(self):
        return (self.r, self.p)

    @property
    def support(self):
        return sp.S.Naturals0

    @property
    def parameter_constraints(self):
        return _and(self.r > 0, self.p > 0, self.p <= 1)

    @property
    def parameter_space(self):
        return ParameterSpace(
            (("r", PositiveRealSpace()), ("p", UnitIntervalSpace(open_left=True)))
        )

    @scalar_batch
    def pdf(self, value):
        k = _sym(value)
        valid = _and(k >= 0, sp.Eq(sp.floor(k), k))
        mass = (
            sp.gamma(k + self.r)
            / (sp.gamma(self.r) * sp.gamma(k + 1))
            * self.p**self.r
            * (1 - self.p) ** k
        )
        return sp.Piecewise(
            (1, sp.And(sp.Eq(self.p, 1), sp.Eq(k, 0))),
            (sp.simplify(mass), valid),
            (0, True),
        )

    @scalar_batch
    def logpdf(self, value):
        return sp.log(self.pdf(value))

    def _mean(self):
        return self.r * (1 - self.p) / self.p

    def _variance(self):
        return self.r * (1 - self.p) / self.p**2

    def _sample(self, size, rng):
        return rng.negative_binomial(
            _float(self.r, "r"), _float(self.p, "p"), size=size
        )


@dataclass(frozen=True, slots=True)
class ChiSquared(Distribution):
    df: sp.Expr

    def __post_init__(self):
        object.__setattr__(self, "df", _sym(self.df))
        validate_distribution_parameters(self)

    @property
    def parameters(self):
        return (self.df,)

    @property
    def support(self):
        return sp.Interval(0, sp.oo)

    @property
    def parameter_constraints(self):
        return self.df > 0

    @property
    def parameter_space(self):
        return ParameterSpace((("df", PositiveRealSpace()),))

    @scalar_batch
    def logpdf(self, value):
        x = _sym(value)
        a = self.df / 2
        lp = (a - 1) * sp.log(x) - x / 2 - a * sp.log(2) - sp.loggamma(a)
        return sp.Piecewise((lp, x >= 0), (-sp.oo, True))

    def _cdf(self, x):
        return sp.Piecewise(
            (0, x < 0),
            (sp.lowergamma(self.df / 2, x / 2) / sp.gamma(self.df / 2), True),
        )

    def _mean(self):
        return self.df

    def _variance(self):
        return 2 * self.df

    def _raw_moment(self, n):
        return 2**n * sp.rf(self.df / 2, n)

    def _entropy(self):
        a = self.df / 2
        return a + sp.log(2 * sp.gamma(a)) + (1 - a) * sp.polygamma(0, a)

    def _sample(self, size, rng):
        return rng.chisquare(_float(self.df, "df"), size=size)


@dataclass(frozen=True, slots=True)
class Weibull(Distribution):
    shape: sp.Expr
    scale: sp.Expr = sp.S.One

    def __post_init__(self):
        object.__setattr__(self, "shape", _sym(self.shape))
        object.__setattr__(self, "scale", _sym(self.scale))
        validate_distribution_parameters(self)

    @property
    def parameters(self):
        return (self.shape, self.scale)

    @property
    def support(self):
        return sp.Interval(0, sp.oo)

    @property
    def parameter_constraints(self):
        return _and(self.shape > 0, self.scale > 0)

    @property
    def parameter_space(self):
        return ParameterSpace(
            (("shape", PositiveRealSpace()), ("scale", PositiveRealSpace()))
        )

    @scalar_batch
    def logpdf(self, value):
        x = _sym(value)
        k = self.shape
        s = self.scale
        lp = sp.log(k / s) + (k - 1) * sp.log(x / s) - (x / s) ** k
        return sp.Piecewise((lp, x >= 0), (-sp.oo, True))

    def _cdf(self, x):
        return sp.Piecewise(
            (0, x < 0), (1 - sp.exp(-((x / self.scale) ** self.shape)), True)
        )

    def _quantile(self, p):
        return sp.Piecewise(
            (sp.oo, sp.Eq(p, 1)),
            (self.scale * (-sp.log(1 - p)) ** (1 / self.shape), True),
        )

    def _raw_moment(self, n):
        return self.scale**n * sp.gamma(1 + n / self.shape)

    def _mean(self):
        return self.scale * sp.gamma(1 + 1 / self.shape)

    def _variance(self):
        return self.scale**2 * (
            sp.gamma(1 + 2 / self.shape) - sp.gamma(1 + 1 / self.shape) ** 2
        )

    def _entropy(self):
        return (
            1
            - sp.EulerGamma
            + sp.EulerGamma / self.shape
            + sp.log(self.scale / self.shape)
        )

    def _sample(self, size, rng):
        return _float(self.scale, "scale") * rng.weibull(
            _float(self.shape, "shape"), size=size
        )


@dataclass(frozen=True, slots=True)
class Pareto(Distribution):
    shape: sp.Expr
    scale: sp.Expr = sp.S.One

    def __post_init__(self):
        object.__setattr__(self, "shape", _sym(self.shape))
        object.__setattr__(self, "scale", _sym(self.scale))
        validate_distribution_parameters(self)

    @property
    def parameters(self):
        return (self.shape, self.scale)

    @property
    def support(self):
        return sp.Interval(self.scale, sp.oo)

    @property
    def parameter_constraints(self):
        return _and(self.shape > 0, self.scale > 0)

    @property
    def parameter_space(self):
        return ParameterSpace(
            (("shape", PositiveRealSpace()), ("scale", PositiveRealSpace()))
        )

    @scalar_batch
    def logpdf(self, value):
        x = _sym(value)
        lp = (
            sp.log(self.shape)
            + self.shape * sp.log(self.scale)
            - (self.shape + 1) * sp.log(x)
        )
        return sp.Piecewise((lp, x >= self.scale), (-sp.oo, True))

    def _cdf(self, x):
        return sp.Piecewise(
            (0, x < self.scale), (1 - (self.scale / x) ** self.shape, True)
        )

    def _quantile(self, p):
        return sp.Piecewise(
            (sp.oo, sp.Eq(p, 1)),
            (self.scale / (1 - p) ** (1 / self.shape), True),
        )

    def _raw_moment(self, n):
        return sp.Piecewise(
            (self.shape * self.scale**n / (self.shape - n), self.shape > n),
            (sp.oo, True),
        )

    def _mean(self):
        return sp.Piecewise(
            (self.shape * self.scale / (self.shape - 1), self.shape > 1), (sp.oo, True)
        )

    def _variance(self):
        return sp.Piecewise(
            (
                self.shape * self.scale**2 / ((self.shape - 1) ** 2 * (self.shape - 2)),
                self.shape > 2,
            ),
            (sp.oo, True),
        )

    def _entropy(self):
        return sp.log(self.scale / self.shape) + 1 + 1 / self.shape

    def _sample(self, size, rng):
        return _float(self.scale, "scale") * (
            rng.pareto(_float(self.shape, "shape"), size=size) + 1
        )


@dataclass(frozen=True, slots=True)
class HalfNormal(Distribution):
    sigma: sp.Expr = sp.S.One

    def __post_init__(self):
        object.__setattr__(self, "sigma", _sym(self.sigma))
        validate_distribution_parameters(self)

    @property
    def parameters(self):
        return (self.sigma,)

    @property
    def support(self):
        return sp.Interval(0, sp.oo)

    @property
    def parameter_constraints(self):
        return self.sigma > 0

    @property
    def parameter_space(self):
        return ParameterSpace((("sigma", PositiveRealSpace()),))

    @scalar_batch
    def logpdf(self, value):
        x = _sym(value)
        lp = sp.log(sp.sqrt(2 / sp.pi) / self.sigma) - x**2 / (2 * self.sigma**2)
        return sp.Piecewise((lp, x >= 0), (-sp.oo, True))

    def _cdf(self, x):
        return sp.Piecewise((0, x < 0), (sp.erf(x / (sp.sqrt(2) * self.sigma)), True))

    def _quantile(self, p):
        return sp.sqrt(2) * self.sigma * sp.erfinv(p)

    def _raw_moment(self, n):
        n = sp.sympify(n)
        return self.sigma**n * 2 ** (n / 2) * sp.gamma((n + 1) / 2) / sp.sqrt(sp.pi)

    def _mean(self):
        return self.sigma * sp.sqrt(2 / sp.pi)

    def _variance(self):
        return self.sigma**2 * (1 - 2 / sp.pi)

    def _entropy(self):
        return sp.Rational(1, 2) * sp.log(sp.pi * self.sigma**2 / 2) + sp.Rational(1, 2)

    def _sample(self, size, rng):
        return np.abs(rng.normal(0, _float(self.sigma, "sigma"), size=size))


@dataclass(frozen=True, slots=True)
class HalfCauchy(Distribution):
    scale: sp.Expr = sp.S.One

    def __post_init__(self):
        object.__setattr__(self, "scale", _sym(self.scale))
        validate_distribution_parameters(self)

    @property
    def parameters(self):
        return (self.scale,)

    @property
    def support(self):
        return sp.Interval(0, sp.oo)

    @property
    def parameter_constraints(self):
        return self.scale > 0

    @property
    def parameter_space(self):
        return ParameterSpace((("scale", PositiveRealSpace()),))

    @scalar_batch
    def logpdf(self, value):
        x = _sym(value)
        lp = sp.log(2 / (sp.pi * self.scale)) - sp.log(1 + (x / self.scale) ** 2)
        return sp.Piecewise((lp, x >= 0), (-sp.oo, True))

    def _cdf(self, x):
        return sp.Piecewise((0, x < 0), (2 / sp.pi * sp.atan(x / self.scale), True))

    def _quantile(self, p):
        return self.scale * sp.tan(sp.pi * p / 2)

    def _mean(self):
        return sp.oo

    def _variance(self):
        return sp.oo

    def _entropy(self):
        return sp.log(2 * sp.pi * self.scale)

    def _sample(self, size, rng):
        return np.abs(_float(self.scale, "scale") * rng.standard_cauchy(size=size))


@dataclass(frozen=True, slots=True)
class Cauchy(Distribution):
    location: sp.Expr = sp.S.Zero
    scale: sp.Expr = sp.S.One

    def __post_init__(self):
        object.__setattr__(self, "location", _sym(self.location))
        object.__setattr__(self, "scale", _sym(self.scale))
        validate_distribution_parameters(self)

    @property
    def parameters(self):
        return (self.location, self.scale)

    @property
    def support(self):
        return sp.S.Reals

    @property
    def parameter_constraints(self):
        return self.scale > 0

    @property
    def parameter_space(self):
        return ParameterSpace(
            (("location", RealSpace()), ("scale", PositiveRealSpace()))
        )

    @scalar_batch
    def logpdf(self, value):
        x = _sym(value)
        return -sp.log(sp.pi * self.scale) - sp.log(
            1 + ((x - self.location) / self.scale) ** 2
        )

    def _cdf(self, x):
        return sp.Rational(1, 2) + sp.atan((x - self.location) / self.scale) / sp.pi

    def _quantile(self, p):
        return self.location + self.scale * sp.tan(sp.pi * (p - sp.Rational(1, 2)))

    def _mean(self):
        return sp.nan

    def _variance(self):
        return sp.nan

    def _entropy(self):
        return sp.log(4 * sp.pi * self.scale)

    def _sample(self, size, rng):
        return _float(self.location, "location") + _float(
            self.scale, "scale"
        ) * rng.standard_cauchy(size=size)


@dataclass(frozen=True, slots=True)
class Laplace(Distribution):
    location: sp.Expr = sp.S.Zero
    scale: sp.Expr = sp.S.One

    def __post_init__(self):
        object.__setattr__(self, "location", _sym(self.location))
        object.__setattr__(self, "scale", _sym(self.scale))
        validate_distribution_parameters(self)

    @property
    def parameters(self):
        return (self.location, self.scale)

    @property
    def support(self):
        return sp.S.Reals

    @property
    def parameter_constraints(self):
        return self.scale > 0

    @property
    def parameter_space(self):
        return ParameterSpace(
            (("location", RealSpace()), ("scale", PositiveRealSpace()))
        )

    @scalar_batch
    def logpdf(self, value):
        return (
            -sp.log(2 * self.scale) - sp.Abs(_sym(value) - self.location) / self.scale
        )

    def _cdf(self, x):
        return sp.Piecewise(
            (sp.exp((x - self.location) / self.scale) / 2, x < self.location),
            (1 - sp.exp(-(x - self.location) / self.scale) / 2, True),
        )

    def _quantile(self, p):
        return sp.Piecewise(
            (self.location + self.scale * sp.log(2 * p), p < sp.Rational(1, 2)),
            (self.location - self.scale * sp.log(2 * (1 - p)), True),
        )

    def _mean(self):
        return self.location

    def _variance(self):
        return 2 * self.scale**2

    def _entropy(self):
        return 1 + sp.log(2 * self.scale)

    def _sample(self, size, rng):
        return rng.laplace(
            _float(self.location, "location"), _float(self.scale, "scale"), size=size
        )


@dataclass(frozen=True, slots=True)
class Logistic(Distribution):
    location: sp.Expr = sp.S.Zero
    scale: sp.Expr = sp.S.One

    def __post_init__(self):
        object.__setattr__(self, "location", _sym(self.location))
        object.__setattr__(self, "scale", _sym(self.scale))
        validate_distribution_parameters(self)

    @property
    def parameters(self):
        return (self.location, self.scale)

    @property
    def support(self):
        return sp.S.Reals

    @property
    def parameter_constraints(self):
        return self.scale > 0

    @property
    def parameter_space(self):
        return ParameterSpace(
            (("location", RealSpace()), ("scale", PositiveRealSpace()))
        )

    @scalar_batch
    def logpdf(self, value):
        z = (_sym(value) - self.location) / self.scale
        return -sp.log(self.scale) - z - 2 * sp.log(1 + sp.exp(-z))

    def _cdf(self, x):
        return 1 / (1 + sp.exp(-(x - self.location) / self.scale))

    def _quantile(self, p):
        return self.location + self.scale * sp.log(p / (1 - p))

    def _mean(self):
        return self.location

    def _variance(self):
        return sp.pi**2 * self.scale**2 / 3

    def _entropy(self):
        return 2 + sp.log(self.scale)

    def _sample(self, size, rng):
        return rng.logistic(
            _float(self.location, "location"), _float(self.scale, "scale"), size=size
        )


@dataclass(frozen=True, slots=True)
class BetaBinomial(Distribution):
    n: sp.Expr
    alpha: sp.Expr
    beta: sp.Expr

    def __post_init__(self):
        for name in ("n", "alpha", "beta"):
            object.__setattr__(self, name, _sym(getattr(self, name)))
        validate_distribution_parameters(self)

    @property
    def parameters(self):
        return (self.n, self.alpha, self.beta)

    @property
    def support(self):
        if (
            self.n.is_integer is True
            and self.n.is_nonnegative is True
            and self.n.is_number
        ):
            return sp.FiniteSet(*range(int(self.n) + 1))
        return sp.Intersection(sp.S.Naturals0, sp.Interval(0, self.n))

    @property
    def parameter_constraints(self):
        return _and(
            self.n >= 0, sp.Eq(sp.floor(self.n), self.n), self.alpha > 0, self.beta > 0
        )

    @property
    def parameter_space(self):
        return ParameterSpace(
            (
                ("n", NaturalNumberSpace()),
                ("alpha", PositiveRealSpace()),
                ("beta", PositiveRealSpace()),
            )
        )

    @property
    def mixing_distribution(self):
        return Beta(self.alpha, self.beta)

    def conditional_distribution(self, p):
        return Binomial(self.n, p)

    @scalar_batch
    def pdf(self, value):
        k = _sym(value)
        valid = _and(k >= 0, k <= self.n, sp.Eq(sp.floor(k), k))
        mass = (
            sp.binomial(self.n, k)
            * sp.gamma(k + self.alpha)
            * sp.gamma(self.n - k + self.beta)
            * sp.gamma(self.alpha + self.beta)
            / (
                sp.gamma(self.n + self.alpha + self.beta)
                * sp.gamma(self.alpha)
                * sp.gamma(self.beta)
            )
        )
        return sp.Piecewise((sp.simplify(mass), valid), (0, True))

    @scalar_batch
    def logpdf(self, value):
        return sp.log(self.pdf(value))

    def _mean(self):
        return self.n * self.alpha / (self.alpha + self.beta)

    def _variance(self):
        a, b = self.alpha, self.beta
        return self.n * a * b * (a + b + self.n) / ((a + b) ** 2 * (a + b + 1))

    def _sample(self, size, rng):
        p = rng.beta(_float(self.alpha, "alpha"), _float(self.beta, "beta"), size=size)
        return rng.binomial(_int(self.n, "n"), p)


@dataclass(frozen=True, slots=True)
class DirichletMultinomial(Distribution):
    n: sp.Expr
    concentration: tuple[sp.Expr, ...]

    def __init__(self, n: Any, concentration: Sequence[Any]):
        n = _sym(n)
        alpha = tuple(_sym(a) for a in concentration)
        if not alpha:
            raise ValueError("concentration must be nonempty")
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "concentration", alpha)
        validate_distribution_parameters(self)

    @property
    def dimension(self):
        return len(self.concentration)

    @property
    def parameters(self):
        return (self.n, *self.concentration)

    @property
    def support(self):
        return sp.ProductSet(*([sp.S.Naturals0] * self.dimension))

    @property
    def event_space(self):
        return CountVectorEventSpace(self.dimension, self.n)

    @property
    def measure_type(self):
        return MeasureType.DISCRETE

    @property
    def parameter_constraints(self):
        return _and(
            self.n >= 0,
            sp.Eq(sp.floor(self.n), self.n),
            *(a > 0 for a in self.concentration),
        )

    @property
    def parameter_space(self):
        return ParameterSpace(
            (
                ("n", NaturalNumberSpace()),
                ("concentration", PositiveVectorSpace(self.dimension)),
            )
        )

    @property
    def parameter_space_constraints(self):
        return self.parameter_space.constraints((self.n, self.concentration))

    @property
    def mixing_distribution(self):
        return Dirichlet(self.concentration)

    def conditional_distribution(self, probabilities):
        return Multinomial(self.n, probabilities)

    @scalar_batch
    def logpdf(self, value):
        xs = tuple(map(_sym, value))
        if len(xs) != self.dimension:
            raise ValueError("count vector has wrong dimension")
        a0 = sp.Add(*self.concentration)
        valid = _and(
            *(x >= 0 for x in xs),
            *(sp.Eq(sp.floor(x), x) for x in xs),
            sp.Eq(sp.Add(*xs), self.n),
        )
        lp = (
            sp.loggamma(self.n + 1)
            - sum(sp.loggamma(x + 1) for x in xs)
            + sp.loggamma(a0)
            - sp.loggamma(a0 + self.n)
            + sum(
                sp.loggamma(x + a) - sp.loggamma(a)
                for x, a in zip(xs, self.concentration)
            )
        )
        return sp.Piecewise((sp.simplify(lp), valid), (-sp.oo, True))

    def _mean(self):
        a0 = sp.Add(*self.concentration)
        return sp.ImmutableMatrix([self.n * a / a0 for a in self.concentration])

    def _variance(self):
        a0 = sp.Add(*self.concentration)
        p = [a / a0 for a in self.concentration]
        factor = self.n * (self.n + a0) / (1 + a0)
        return sp.ImmutableMatrix(
            self.dimension,
            self.dimension,
            lambda i, j: factor * (p[i] * (1 - p[i]) if i == j else -p[i] * p[j]),
        )

    def _sample(self, size, rng):
        alpha = np.asarray(
            [_float(a, "concentration") for a in self.concentration], dtype=float
        )
        probs = rng.dirichlet(alpha, size=size)
        n = _int(self.n, "n")
        if size is None:
            return rng.multinomial(n, probs)
        flat = probs.reshape((-1, self.dimension))
        draws = np.asarray([rng.multinomial(n, p) for p in flat])
        return draws.reshape(probs.shape)


__all__ = [
    "BetaBinomial",
    "Cauchy",
    "ChiSquared",
    "DirichletMultinomial",
    "Geometric",
    "HalfCauchy",
    "HalfNormal",
    "Laplace",
    "Logistic",
    "NegativeBinomial",
    "Pareto",
    "Uniform",
    "Weibull",
]
