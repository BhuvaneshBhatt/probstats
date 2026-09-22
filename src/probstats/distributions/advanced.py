"""Additional univariate probability distributions."""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from operator import and_

import numpy as np
import sympy as sp

from ..spaces import (
    IntegerSpace,
    NaturalNumberSpace,
    ParameterSpace,
    PositiveRealSpace,
    RealSpace,
)
from .base import Distribution, scalar_batch, validate_distribution_parameters


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


def _reg_beta(a, b, z):
    return sp.betainc(a, b, 0, z) / sp.beta(a, b)


@dataclass(frozen=True, slots=True)
class FDistribution(Distribution):
    df1: sp.Expr
    df2: sp.Expr

    def __post_init__(self):
        object.__setattr__(self, "df1", _sym(self.df1))
        object.__setattr__(self, "df2", _sym(self.df2))
        validate_distribution_parameters(self)

    @property
    def parameters(self):
        return (self.df1, self.df2)

    @property
    def support(self):
        return sp.Interval(0, sp.oo)

    @property
    def parameter_constraints(self):
        return _and(self.df1 > 0, self.df2 > 0)

    @property
    def parameter_space(self):
        return ParameterSpace(
            (("df1", PositiveRealSpace()), ("df2", PositiveRealSpace()))
        )

    @scalar_batch
    def logpdf(self, value):
        x = _sym(value)
        a = self.df1 / 2
        b = self.df2 / 2
        lp = (
            a * sp.log(self.df1 / self.df2)
            + (a - 1) * sp.log(x)
            - sp.log(sp.beta(a, b))
            - (a + b) * sp.log(1 + self.df1 * x / self.df2)
        )
        return sp.Piecewise((lp, x > 0), (-sp.oo, True))

    def _cdf(self, x):
        z = self.df1 * x / (self.df1 * x + self.df2)
        return sp.Piecewise(
            (0, x <= 0), (_reg_beta(self.df1 / 2, self.df2 / 2, z), True)
        )

    def _mean(self):
        return sp.Piecewise((self.df2 / (self.df2 - 2), self.df2 > 2), (sp.oo, True))

    def _variance(self):
        num = 2 * self.df2**2 * (self.df1 + self.df2 - 2)
        den = self.df1 * (self.df2 - 2) ** 2 * (self.df2 - 4)
        return sp.Piecewise((num / den, self.df2 > 4), (sp.oo, True))

    def _raw_moment(self, n):
        n = sp.sympify(n)
        expr = (
            (self.df2 / self.df1) ** n
            * sp.gamma(self.df1 / 2 + n)
            * sp.gamma(self.df2 / 2 - n)
            / (sp.gamma(self.df1 / 2) * sp.gamma(self.df2 / 2))
        )
        return sp.Piecewise((expr, self.df2 > 2 * n), (sp.oo, True))

    def _sample(self, size, rng):
        return rng.f(_float(self.df1, "df1"), _float(self.df2, "df2"), size=size)


@dataclass(frozen=True, slots=True)
class NoncentralChiSquared(Distribution):
    df: sp.Expr
    noncentrality: sp.Expr

    def __post_init__(self):
        object.__setattr__(self, "df", _sym(self.df))
        object.__setattr__(self, "noncentrality", _sym(self.noncentrality))
        validate_distribution_parameters(self)

    @property
    def parameters(self):
        return (self.df, self.noncentrality)

    @property
    def support(self):
        return sp.Interval(0, sp.oo)

    @property
    def parameter_constraints(self):
        return _and(self.df > 0, self.noncentrality >= 0)

    @property
    def parameter_space(self):
        return ParameterSpace(
            (("df", PositiveRealSpace()), ("noncentrality", RealSpace())),
            relations=(lambda m: m["noncentrality"] >= 0,),
        )

    @scalar_batch
    def pdf(self, value):
        x = _sym(value)
        nu = self.df / 2 - 1
        lam = self.noncentrality
        central = (
            x ** (self.df / 2 - 1)
            * sp.exp(-x / 2)
            / (2 ** (self.df / 2) * sp.gamma(self.df / 2))
        )
        nc = (
            sp.exp(-(x + lam) / 2)
            / 2
            * (x / lam) ** (nu / 2)
            * sp.besseli(nu, sp.sqrt(lam * x))
        )
        return sp.Piecewise((0, x < 0), (central, sp.Eq(lam, 0)), (nc, True))

    @scalar_batch
    def logpdf(self, value):
        return sp.log(self.pdf(value))

    def _mean(self):
        return self.df + self.noncentrality

    def _variance(self):
        return 2 * (self.df + 2 * self.noncentrality)

    def _sample(self, size, rng):
        return rng.noncentral_chisquare(
            _float(self.df, "df"),
            _float(self.noncentrality, "noncentrality"),
            size=size,
        )


@dataclass(frozen=True, slots=True)
class NoncentralF(Distribution):
    df1: sp.Expr
    df2: sp.Expr
    noncentrality: sp.Expr

    def __post_init__(self):
        for name in ("df1", "df2", "noncentrality"):
            object.__setattr__(self, name, _sym(getattr(self, name)))
        validate_distribution_parameters(self)

    @property
    def parameters(self):
        return (self.df1, self.df2, self.noncentrality)

    @property
    def support(self):
        return sp.Interval(0, sp.oo)

    @property
    def parameter_constraints(self):
        return _and(self.df1 > 0, self.df2 > 0, self.noncentrality >= 0)

    @property
    def parameter_space(self):
        return ParameterSpace(
            (
                ("df1", PositiveRealSpace()),
                ("df2", PositiveRealSpace()),
                ("noncentrality", RealSpace()),
            ),
            relations=(lambda m: m["noncentrality"] >= 0,),
        )

    @scalar_batch
    def pdf(self, value):
        x = _sym(value)
        j = sp.Dummy("j", integer=True, nonnegative=True)
        a = self.df1 / 2 + j
        b = self.df2 / 2
        q = self.df1 * x / self.df2
        term = (
            sp.exp(-self.noncentrality / 2)
            * (self.noncentrality / 2) ** j
            / sp.factorial(j)
            * (self.df1 / self.df2) ** a
            * x ** (a - 1)
            / (sp.beta(a, b) * (1 + q) ** (a + b))
        )
        return sp.Piecewise((0, x <= 0), (sp.Sum(term, (j, 0, sp.oo)), True))

    @scalar_batch
    def logpdf(self, value):
        return sp.log(self.pdf(value))

    def _mean(self):
        expr = self.df2 * (self.df1 + self.noncentrality) / (self.df1 * (self.df2 - 2))
        return sp.Piecewise((expr, self.df2 > 2), (sp.oo, True))

    def _sample(self, size, rng):
        return rng.noncentral_f(
            _float(self.df1, "df1"),
            _float(self.df2, "df2"),
            _float(self.noncentrality, "noncentrality"),
            size=size,
        )


@dataclass(frozen=True, slots=True)
class NoncentralT(Distribution):
    df: sp.Expr
    noncentrality: sp.Expr = sp.S.Zero

    def __post_init__(self):
        object.__setattr__(self, "df", _sym(self.df))
        object.__setattr__(self, "noncentrality", _sym(self.noncentrality))
        validate_distribution_parameters(self)

    @property
    def parameters(self):
        return (self.df, self.noncentrality)

    @property
    def support(self):
        return sp.S.Reals

    @property
    def parameter_constraints(self):
        return self.df > 0

    @property
    def parameter_space(self):
        return ParameterSpace(
            (("df", PositiveRealSpace()), ("noncentrality", RealSpace()))
        )

    @scalar_batch
    def pdf(self, value):
        x = _sym(value)
        v = sp.Dummy("v", positive=True)
        df = self.df
        delta = self.noncentrality
        normal = sp.exp(-((x * sp.sqrt(v / df) - delta) ** 2) / 2) / sp.sqrt(2 * sp.pi)
        chi = v ** (df / 2 - 1) * sp.exp(-v / 2) / (2 ** (df / 2) * sp.gamma(df / 2))
        return sp.Integral(normal * sp.sqrt(v / df) * chi, (v, 0, sp.oo))

    @scalar_batch
    def logpdf(self, value):
        return sp.log(self.pdf(value))

    def _mean(self):
        expr = (
            self.noncentrality
            * sp.sqrt(self.df / 2)
            * sp.gamma((self.df - 1) / 2)
            / sp.gamma(self.df / 2)
        )
        return sp.Piecewise((expr, self.df > 1), (sp.nan, True))

    def _variance(self):
        mean = self._mean().args[0][0]
        finite = self.df * (1 + self.noncentrality**2) / (self.df - 2) - mean**2
        return sp.Piecewise(
            (finite, self.df > 2),
            (sp.oo, self.df > 1),
            (sp.nan, True),
        )

    def _sample(self, size, rng):
        z = rng.normal(_float(self.noncentrality, "noncentrality"), 1, size=size)
        v = rng.chisquare(_float(self.df, "df"), size=size)
        return z / np.sqrt(v / _float(self.df, "df"))


@dataclass(frozen=True, slots=True)
class DiscreteUniform(Distribution):
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
        if (
            self.low.is_integer is True
            and self.high.is_integer is True
            and self.low.is_number
            and self.high.is_number
        ):
            return sp.FiniteSet(*range(int(self.low), int(self.high) + 1))
        return sp.Intersection(sp.S.Integers, sp.Interval(self.low, self.high))

    @property
    def parameter_constraints(self):
        return _and(
            sp.Eq(sp.floor(self.low), self.low),
            sp.Eq(sp.floor(self.high), self.high),
            self.high >= self.low,
        )

    @property
    def parameter_space(self):
        return ParameterSpace(
            (("low", IntegerSpace()), ("high", IntegerSpace())),
            relations=(lambda m: m["high"] >= m["low"],),
        )

    @scalar_batch
    def pdf(self, value):
        k = _sym(value)
        valid = _and(k >= self.low, k <= self.high, sp.Eq(sp.floor(k), k))
        return sp.Piecewise((1 / (self.high - self.low + 1), valid), (0, True))

    @scalar_batch
    def logpdf(self, value):
        return sp.log(self.pdf(value))

    def _cdf(self, x):
        return sp.Piecewise(
            (0, x < self.low),
            (1, x >= self.high),
            ((sp.floor(x) - self.low + 1) / (self.high - self.low + 1), True),
        )

    def _quantile(self, p):
        return sp.Piecewise(
            (self.low, sp.Eq(p, 0)),
            (self.low + sp.ceiling(p * (self.high - self.low + 1)) - 1, True),
        )

    def _mean(self):
        return (self.low + self.high) / 2

    def _variance(self):
        return ((self.high - self.low + 1) ** 2 - 1) / 12

    def _sample(self, size, rng):
        return rng.integers(
            _int(self.low, "low"), _int(self.high, "high") + 1, size=size
        )


@dataclass(frozen=True, slots=True)
class Hypergeometric(Distribution):
    successes: sp.Expr
    failures: sp.Expr
    draws: sp.Expr

    def __post_init__(self):
        for name in ("successes", "failures", "draws"):
            object.__setattr__(self, name, _sym(getattr(self, name)))
        validate_distribution_parameters(self)

    @property
    def parameters(self):
        return (self.successes, self.failures, self.draws)

    @property
    def parameter_constraints(self):
        return _and(
            *(sp.Eq(sp.floor(v), v) for v in self.parameters),
            self.successes >= 0,
            self.failures >= 0,
            self.draws >= 0,
            self.draws <= self.successes + self.failures,
        )

    @property
    def parameter_space(self):
        return ParameterSpace(
            (
                ("successes", NaturalNumberSpace()),
                ("failures", NaturalNumberSpace()),
                ("draws", NaturalNumberSpace()),
            ),
            relations=(lambda m: m["draws"] <= m["successes"] + m["failures"],),
        )

    @property
    def support(self):
        lo = sp.Max(0, self.draws - self.failures)
        hi = sp.Min(self.draws, self.successes)
        if all(v.is_number for v in self.parameters):
            return sp.FiniteSet(*range(int(lo), int(hi) + 1))
        return sp.Intersection(sp.S.Integers, sp.Interval(lo, hi))

    @scalar_batch
    def pdf(self, value):
        k = _sym(value)
        lo = sp.Max(0, self.draws - self.failures)
        hi = sp.Min(self.draws, self.successes)
        valid = _and(k >= lo, k <= hi, sp.Eq(sp.floor(k), k))
        mass = (
            sp.binomial(self.successes, k)
            * sp.binomial(self.failures, self.draws - k)
            / sp.binomial(self.successes + self.failures, self.draws)
        )
        return sp.Piecewise((mass, valid), (0, True))

    @scalar_batch
    def logpdf(self, value):
        return sp.log(self.pdf(value))

    def _mean(self):
        return self.draws * self.successes / (self.successes + self.failures)

    def _variance(self):
        n = self.successes + self.failures
        expr = (
            self.draws
            * self.successes
            * self.failures
            * (n - self.draws)
            / (n**2 * (n - 1))
        )
        return sp.Piecewise((0, sp.Eq(n, 1)), (expr, True))

    def _factorial_moment(self, order):
        n = sp.Integer(order)
        total = self.successes + self.failures
        return sp.ff(self.draws, n) * sp.ff(self.successes, n) / sp.ff(total, n)

    def _sample(self, size, rng):
        return rng.hypergeometric(
            _int(self.successes, "successes"),
            _int(self.failures, "failures"),
            _int(self.draws, "draws"),
            size=size,
        )


@dataclass(frozen=True, slots=True)
class Skellam(Distribution):
    rate1: sp.Expr
    rate2: sp.Expr

    def __post_init__(self):
        object.__setattr__(self, "rate1", _sym(self.rate1))
        object.__setattr__(self, "rate2", _sym(self.rate2))
        validate_distribution_parameters(self)

    @property
    def parameters(self):
        return (self.rate1, self.rate2)

    @property
    def support(self):
        return sp.S.Integers

    @property
    def parameter_constraints(self):
        return _and(self.rate1 > 0, self.rate2 > 0)

    @property
    def parameter_space(self):
        return ParameterSpace(
            (("rate1", PositiveRealSpace()), ("rate2", PositiveRealSpace()))
        )

    @scalar_batch
    def pdf(self, value):
        k = _sym(value)
        mass = (
            sp.exp(-(self.rate1 + self.rate2))
            * (self.rate1 / self.rate2) ** (k / 2)
            * sp.besseli(sp.Abs(k), 2 * sp.sqrt(self.rate1 * self.rate2))
        )
        return sp.Piecewise((mass, sp.Eq(sp.floor(k), k)), (0, True))

    @scalar_batch
    def logpdf(self, value):
        return sp.log(self.pdf(value))

    def _cdf(self, x):
        k = sp.Dummy("k", integer=True)
        return sp.Sum(self.pdf(k), (k, -sp.oo, sp.floor(x)))

    def _mean(self):
        return self.rate1 - self.rate2

    def _variance(self):
        return self.rate1 + self.rate2

    def _sample(self, size, rng):
        return rng.poisson(_float(self.rate1, "rate1"), size=size) - rng.poisson(
            _float(self.rate2, "rate2"), size=size
        )


@dataclass(frozen=True, slots=True)
class Zipf(Distribution):
    exponent: sp.Expr

    def __post_init__(self):
        object.__setattr__(self, "exponent", _sym(self.exponent))
        validate_distribution_parameters(self)

    @property
    def parameters(self):
        return (self.exponent,)

    @property
    def support(self):
        return sp.S.Naturals

    @property
    def parameter_constraints(self):
        return self.exponent > 1

    @property
    def parameter_space(self):
        return ParameterSpace(
            (("exponent", PositiveRealSpace()),),
            relations=(lambda m: m["exponent"] > 1,),
        )

    @scalar_batch
    def pdf(self, value):
        k = _sym(value)
        valid = _and(k >= 1, sp.Eq(sp.floor(k), k))
        return sp.Piecewise(
            (k ** (-self.exponent) / sp.zeta(self.exponent), valid), (0, True)
        )

    @scalar_batch
    def logpdf(self, value):
        return sp.log(self.pdf(value))

    def _cdf(self, x):
        return sp.Piecewise(
            (0, x < 1),
            (sp.harmonic(sp.floor(x), self.exponent) / sp.zeta(self.exponent), True),
        )

    def _mean(self):
        return sp.Piecewise(
            (sp.zeta(self.exponent - 1) / sp.zeta(self.exponent), self.exponent > 2),
            (sp.oo, True),
        )

    def _variance(self):
        e = self.exponent
        expr = sp.zeta(e - 2) / sp.zeta(e) - (sp.zeta(e - 1) / sp.zeta(e)) ** 2
        return sp.Piecewise((expr, e > 3), (sp.oo, True))

    def _sample(self, size, rng):
        return rng.zipf(_float(self.exponent, "exponent"), size=size)


@dataclass(frozen=True, slots=True)
class InverseGaussian(Distribution):
    mean: sp.Expr
    shape: sp.Expr

    def __post_init__(self):
        object.__setattr__(self, "mean", _sym(self.mean))
        object.__setattr__(self, "shape", _sym(self.shape))
        validate_distribution_parameters(self)

    @property
    def parameters(self):
        return (self.mean, self.shape)

    @property
    def support(self):
        return sp.Interval(0, sp.oo)

    @property
    def parameter_constraints(self):
        return _and(self.mean > 0, self.shape > 0)

    @property
    def parameter_space(self):
        return ParameterSpace(
            (("mean", PositiveRealSpace()), ("shape", PositiveRealSpace()))
        )

    @scalar_batch
    def logpdf(self, value):
        x = _sym(value)
        lp = sp.Rational(1, 2) * sp.log(
            self.shape / (2 * sp.pi * x**3)
        ) - self.shape * (x - self.mean) ** 2 / (2 * self.mean**2 * x)
        return sp.Piecewise((lp, x > 0), (-sp.oo, True))

    def _cdf(self, x):
        z1 = sp.sqrt(self.shape / x) * (x / self.mean - 1)
        z2 = -sp.sqrt(self.shape / x) * (x / self.mean + 1)
        phi = lambda z: (1 + sp.erf(z / sp.sqrt(2))) / 2
        expr = phi(z1) + sp.exp(2 * self.shape / self.mean) * phi(z2)
        return sp.Piecewise((0, x <= 0), (expr, True))

    def _mean(self):
        return self.mean

    def _variance(self):
        return self.mean**3 / self.shape

    def _sample(self, size, rng):
        return rng.wald(
            _float(self.mean, "mean"), _float(self.shape, "shape"), size=size
        )


@dataclass(frozen=True, slots=True)
class BetaPrime(Distribution):
    alpha: sp.Expr
    beta: sp.Expr
    scale: sp.Expr = sp.S.One

    def __post_init__(self):
        for name in ("alpha", "beta", "scale"):
            object.__setattr__(self, name, _sym(getattr(self, name)))
        validate_distribution_parameters(self)

    @property
    def parameters(self):
        return (self.alpha, self.beta, self.scale)

    @property
    def support(self):
        return sp.Interval(0, sp.oo)

    @property
    def parameter_constraints(self):
        return _and(self.alpha > 0, self.beta > 0, self.scale > 0)

    @property
    def parameter_space(self):
        return ParameterSpace(
            (
                ("alpha", PositiveRealSpace()),
                ("beta", PositiveRealSpace()),
                ("scale", PositiveRealSpace()),
            )
        )

    @scalar_batch
    def logpdf(self, value):
        x = _sym(value)
        lp = (
            (self.alpha - 1) * sp.log(x)
            - self.alpha * sp.log(self.scale)
            - sp.log(sp.beta(self.alpha, self.beta))
            - (self.alpha + self.beta) * sp.log(1 + x / self.scale)
        )
        return sp.Piecewise((lp, x > 0), (-sp.oo, True))

    def _cdf(self, x):
        return sp.Piecewise(
            (0, x <= 0), (_reg_beta(self.alpha, self.beta, x / (x + self.scale)), True)
        )

    def _mean(self):
        return sp.Piecewise(
            (self.scale * self.alpha / (self.beta - 1), self.beta > 1), (sp.oo, True)
        )

    def _variance(self):
        expr = (
            self.scale**2
            * self.alpha
            * (self.alpha + self.beta - 1)
            / ((self.beta - 2) * (self.beta - 1) ** 2)
        )
        return sp.Piecewise((expr, self.beta > 2), (sp.oo, True))

    def _sample(self, size, rng):
        a = rng.gamma(_float(self.alpha, "alpha"), 1, size=size)
        b = rng.gamma(_float(self.beta, "beta"), 1, size=size)
        return _float(self.scale, "scale") * a / b


@dataclass(frozen=True, slots=True)
class Rayleigh(Distribution):
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
        return sp.Piecewise(
            (sp.log(x) - 2 * sp.log(self.scale) - x**2 / (2 * self.scale**2), x > 0),
            (-sp.oo, True),
        )

    def _cdf(self, x):
        return sp.Piecewise(
            (0, x < 0), (1 - sp.exp(-(x**2) / (2 * self.scale**2)), True)
        )

    def _quantile(self, p):
        return self.scale * sp.sqrt(-2 * sp.log(1 - p))

    def _raw_moment(self, n):
        return (
            self.scale**n * 2 ** (sp.sympify(n) / 2) * sp.gamma(1 + sp.sympify(n) / 2)
        )

    def _mean(self):
        return self.scale * sp.sqrt(sp.pi / 2)

    def _variance(self):
        return (2 - sp.pi / 2) * self.scale**2

    def _sample(self, size, rng):
        return rng.rayleigh(_float(self.scale, "scale"), size=size)


@dataclass(frozen=True, slots=True)
class Rice(Distribution):
    noncentrality: sp.Expr
    scale: sp.Expr = sp.S.One

    def __post_init__(self):
        object.__setattr__(self, "noncentrality", _sym(self.noncentrality))
        object.__setattr__(self, "scale", _sym(self.scale))
        validate_distribution_parameters(self)

    @property
    def parameters(self):
        return (self.noncentrality, self.scale)

    @property
    def support(self):
        return sp.Interval(0, sp.oo)

    @property
    def parameter_constraints(self):
        return _and(self.noncentrality >= 0, self.scale > 0)

    @property
    def parameter_space(self):
        return ParameterSpace(
            (("noncentrality", RealSpace()), ("scale", PositiveRealSpace())),
            relations=(lambda m: m["noncentrality"] >= 0,),
        )

    @scalar_batch
    def logpdf(self, value):
        x = _sym(value)
        lp = (
            sp.log(x)
            - 2 * sp.log(self.scale)
            - (x**2 + self.noncentrality**2) / (2 * self.scale**2)
            + sp.log(sp.besseli(0, x * self.noncentrality / self.scale**2))
        )
        return sp.Piecewise((lp, x > 0), (-sp.oo, True))

    def _mean(self):
        q = self.noncentrality**2 / (4 * self.scale**2)
        return (
            self.scale
            * sp.sqrt(sp.pi / 2)
            * sp.exp(-q)
            * ((1 + 2 * q) * sp.besseli(0, q) + 2 * q * sp.besseli(1, q))
        )

    def _variance(self):
        return 2 * self.scale**2 + self.noncentrality**2 - self._mean() ** 2

    def _sample(self, size, rng):
        x = rng.normal(
            _float(self.noncentrality, "noncentrality"),
            _float(self.scale, "scale"),
            size=size,
        )
        y = rng.normal(0, _float(self.scale, "scale"), size=size)
        return np.hypot(x, y)


@dataclass(frozen=True, slots=True)
class Nakagami(Distribution):
    shape: sp.Expr
    spread: sp.Expr = sp.S.One

    def __post_init__(self):
        object.__setattr__(self, "shape", _sym(self.shape))
        object.__setattr__(self, "spread", _sym(self.spread))
        validate_distribution_parameters(self)

    @property
    def parameters(self):
        return (self.shape, self.spread)

    @property
    def support(self):
        return sp.Interval(0, sp.oo)

    @property
    def parameter_constraints(self):
        return _and(self.shape >= sp.Rational(1, 2), self.spread > 0)

    @property
    def parameter_space(self):
        return ParameterSpace(
            (("shape", PositiveRealSpace()), ("spread", PositiveRealSpace())),
            relations=(lambda m: m["shape"] >= sp.Rational(1, 2),),
        )

    @scalar_batch
    def logpdf(self, value):
        x = _sym(value)
        m = self.shape
        o = self.spread
        lp = (
            sp.log(2)
            + m * sp.log(m)
            - sp.loggamma(m)
            - m * sp.log(o)
            + (2 * m - 1) * sp.log(x)
            - m * x**2 / o
        )
        return sp.Piecewise((lp, x > 0), (-sp.oo, True))

    def _raw_moment(self, n):
        n = sp.sympify(n)
        return (
            (self.spread / self.shape) ** (n / 2)
            * sp.gamma(self.shape + n / 2)
            / sp.gamma(self.shape)
        )

    def _mean(self):
        return self._raw_moment(1)

    def _variance(self):
        return self.spread - self._mean() ** 2

    def _sample(self, size, rng):
        return np.sqrt(
            rng.gamma(
                _float(self.shape, "shape"),
                _float(self.spread / self.shape, "spread/shape"),
                size=size,
            )
        )


@dataclass(frozen=True, slots=True)
class Maxwell(Distribution):
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
        lp = (
            sp.log(sp.sqrt(2 / sp.pi))
            + 2 * sp.log(x)
            - 3 * sp.log(self.scale)
            - x**2 / (2 * self.scale**2)
        )
        return sp.Piecewise((lp, x > 0), (-sp.oo, True))

    def _cdf(self, x):
        expr = sp.erf(x / (sp.sqrt(2) * self.scale)) - sp.sqrt(2 / sp.pi) * (
            x / self.scale
        ) * sp.exp(-(x**2) / (2 * self.scale**2))
        return sp.Piecewise((0, x < 0), (expr, True))

    def _raw_moment(self, n):
        n = sp.sympify(n)
        return self.scale**n * 2 ** (n / 2 + 1) * sp.gamma((n + 3) / 2) / sp.sqrt(sp.pi)

    def _mean(self):
        return 2 * self.scale * sp.sqrt(2 / sp.pi)

    def _variance(self):
        return self.scale**2 * (3 - 8 / sp.pi)

    def _sample(self, size, rng):
        shape = (
            () if size is None else ((size,) if isinstance(size, int) else tuple(size))
        )
        arr = rng.normal(0, _float(self.scale, "scale"), size=shape + (3,))
        return np.linalg.norm(arr, axis=-1)


@dataclass(frozen=True, slots=True)
class Gumbel(Distribution):
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
        return -sp.log(self.scale) - z - sp.exp(-z)

    def _cdf(self, x):
        return sp.exp(-sp.exp(-(x - self.location) / self.scale))

    def _quantile(self, p):
        return self.location - self.scale * sp.log(-sp.log(p))

    def _mean(self):
        return self.location + sp.EulerGamma * self.scale

    def _variance(self):
        return sp.pi**2 * self.scale**2 / 6

    def _sample(self, size, rng):
        return rng.gumbel(
            _float(self.location, "location"), _float(self.scale, "scale"), size=size
        )


@dataclass(frozen=True, slots=True)
class Frechet(Distribution):
    shape: sp.Expr
    scale: sp.Expr = sp.S.One
    location: sp.Expr = sp.S.Zero

    def __post_init__(self):
        for name in ("shape", "scale", "location"):
            object.__setattr__(self, name, _sym(getattr(self, name)))
        validate_distribution_parameters(self)

    @property
    def parameters(self):
        return (self.shape, self.scale, self.location)

    @property
    def support(self):
        return sp.Interval(self.location, sp.oo)

    @property
    def parameter_constraints(self):
        return _and(self.shape > 0, self.scale > 0)

    @property
    def parameter_space(self):
        return ParameterSpace(
            (
                ("shape", PositiveRealSpace()),
                ("scale", PositiveRealSpace()),
                ("location", RealSpace()),
            )
        )

    @scalar_batch
    def logpdf(self, value):
        x = _sym(value)
        z = (x - self.location) / self.scale
        lp = (
            sp.log(self.shape / self.scale)
            - (1 + self.shape) * sp.log(z)
            - z ** (-self.shape)
        )
        return sp.Piecewise((lp, x > self.location), (-sp.oo, True))

    def _cdf(self, x):
        z = (x - self.location) / self.scale
        return sp.Piecewise(
            (0, x <= self.location), (sp.exp(-(z ** (-self.shape))), True)
        )

    def _quantile(self, p):
        return self.location + self.scale * (-sp.log(p)) ** (-1 / self.shape)

    def _mean(self):
        return sp.Piecewise(
            (self.location + self.scale * sp.gamma(1 - 1 / self.shape), self.shape > 1),
            (sp.oo, True),
        )

    def _variance(self):
        expr = self.scale**2 * (
            sp.gamma(1 - 2 / self.shape) - sp.gamma(1 - 1 / self.shape) ** 2
        )
        return sp.Piecewise((expr, self.shape > 2), (sp.oo, True))

    def _sample(self, size, rng):
        return (
            _float(self.location, "location")
            + _float(self.scale, "scale")
            * rng.weibull(_float(self.shape, "shape"), size=size) ** -1
        )


@dataclass(frozen=True, slots=True)
class Kumaraswamy(Distribution):
    alpha: sp.Expr
    beta: sp.Expr

    def __post_init__(self):
        object.__setattr__(self, "alpha", _sym(self.alpha))
        object.__setattr__(self, "beta", _sym(self.beta))
        validate_distribution_parameters(self)

    @property
    def parameters(self):
        return (self.alpha, self.beta)

    @property
    def support(self):
        return sp.Interval(0, 1)

    @property
    def parameter_constraints(self):
        return _and(self.alpha > 0, self.beta > 0)

    @property
    def parameter_space(self):
        return ParameterSpace(
            (("alpha", PositiveRealSpace()), ("beta", PositiveRealSpace()))
        )

    @scalar_batch
    def logpdf(self, value):
        x = _sym(value)
        lp = (
            sp.log(self.alpha * self.beta)
            + (self.alpha - 1) * sp.log(x)
            + (self.beta - 1) * sp.log(1 - x**self.alpha)
        )
        return sp.Piecewise((lp, _and(x > 0, x < 1)), (-sp.oo, True))

    def _cdf(self, x):
        return sp.Piecewise(
            (0, x <= 0), (1, x >= 1), (1 - (1 - x**self.alpha) ** self.beta, True)
        )

    def _quantile(self, p):
        return (1 - (1 - p) ** (1 / self.beta)) ** (1 / self.alpha)

    def _raw_moment(self, n):
        return self.beta * sp.beta(1 + sp.sympify(n) / self.alpha, self.beta)

    def _mean(self):
        return self._raw_moment(1)

    def _variance(self):
        return self._raw_moment(2) - self._mean() ** 2

    def _sample(self, size, rng):
        u = rng.random(size=size)
        return (1 - (1 - u) ** (1 / _float(self.beta, "beta"))) ** (
            1 / _float(self.alpha, "alpha")
        )


@dataclass(frozen=True, slots=True)
class PowerDistribution(Distribution):
    shape: sp.Expr

    def __post_init__(self):
        object.__setattr__(self, "shape", _sym(self.shape))
        validate_distribution_parameters(self)

    @property
    def parameters(self):
        return (self.shape,)

    @property
    def support(self):
        return sp.Interval(0, 1)

    @property
    def parameter_constraints(self):
        return self.shape > 0

    @property
    def parameter_space(self):
        return ParameterSpace((("shape", PositiveRealSpace()),))

    @scalar_batch
    def logpdf(self, value):
        x = _sym(value)
        return sp.Piecewise(
            (sp.log(self.shape) + (self.shape - 1) * sp.log(x), _and(x > 0, x <= 1)),
            (-sp.oo, True),
        )

    def _cdf(self, x):
        return sp.Piecewise((0, x <= 0), (1, x >= 1), (x**self.shape, True))

    def _quantile(self, p):
        return p ** (1 / self.shape)

    def _raw_moment(self, n):
        return self.shape / (self.shape + sp.sympify(n))

    def _mean(self):
        return self.shape / (self.shape + 1)

    def _variance(self):
        return self.shape / ((self.shape + 1) ** 2 * (self.shape + 2))

    def _sample(self, size, rng):
        return rng.power(_float(self.shape, "shape"), size=size)


@dataclass(frozen=True, slots=True)
class Triangular(Distribution):
    low: sp.Expr
    mode: sp.Expr
    high: sp.Expr

    def __post_init__(self):
        for name in ("low", "mode", "high"):
            object.__setattr__(self, name, _sym(getattr(self, name)))
        validate_distribution_parameters(self)

    @property
    def parameters(self):
        return (self.low, self.mode, self.high)

    @property
    def support(self):
        return sp.Interval(self.low, self.high)

    @property
    def parameter_constraints(self):
        return _and(self.low < self.high, self.mode >= self.low, self.mode <= self.high)

    @property
    def parameter_space(self):
        return ParameterSpace(
            (("low", RealSpace()), ("mode", RealSpace()), ("high", RealSpace())),
            relations=(
                lambda m: m["low"] < m["high"],
                lambda m: m["mode"] >= m["low"],
                lambda m: m["mode"] <= m["high"],
            ),
        )

    @scalar_batch
    def pdf(self, value):
        x = _sym(value)
        width = self.high - self.low
        rising = 2 * (x - self.low) / (width * (self.mode - self.low))
        falling = 2 * (self.high - x) / (width * (self.high - self.mode))
        left_mode = 2 * (self.high - x) / width**2
        right_mode = 2 * (x - self.low) / width**2
        return sp.Piecewise(
            (0, _and(x < self.low)),
            (left_mode, _and(sp.Eq(self.mode, self.low), x <= self.high)),
            (
                right_mode,
                _and(sp.Eq(self.mode, self.high), x >= self.low, x <= self.high),
            ),
            (rising, _and(x >= self.low, x < self.mode)),
            (2 / width, sp.Eq(x, self.mode)),
            (falling, _and(x > self.mode, x <= self.high)),
            (0, True),
        )

    @scalar_batch
    def logpdf(self, value):
        return sp.log(self.pdf(value))

    def _cdf(self, x):
        width = self.high - self.low
        rising = (x - self.low) ** 2 / (width * (self.mode - self.low))
        falling = 1 - (self.high - x) ** 2 / (width * (self.high - self.mode))
        left_mode = 1 - (self.high - x) ** 2 / width**2
        right_mode = (x - self.low) ** 2 / width**2
        return sp.Piecewise(
            (0, x <= self.low),
            (left_mode, sp.Eq(self.mode, self.low)),
            (right_mode, sp.Eq(self.mode, self.high)),
            (rising, x < self.mode),
            (falling, x < self.high),
            (1, True),
        )

    def _mean(self):
        return (self.low + self.mode + self.high) / 3

    def _variance(self):
        return (
            self.low**2
            + self.mode**2
            + self.high**2
            - self.low * self.mode
            - self.low * self.high
            - self.mode * self.high
        ) / 18

    def _sample(self, size, rng):
        return rng.triangular(
            _float(self.low, "low"),
            _float(self.mode, "mode"),
            _float(self.high, "high"),
            size=size,
        )


@dataclass(frozen=True, slots=True)
class VonMises(Distribution):
    location: sp.Expr = sp.S.Zero
    concentration: sp.Expr = sp.S.One

    def __post_init__(self):
        object.__setattr__(self, "location", _sym(self.location))
        object.__setattr__(self, "concentration", _sym(self.concentration))
        validate_distribution_parameters(self)

    @property
    def parameters(self):
        return (self.location, self.concentration)

    @property
    def support(self):
        return sp.Interval(self.location - sp.pi, self.location + sp.pi)

    @property
    def parameter_constraints(self):
        return self.concentration > 0

    @property
    def parameter_space(self):
        return ParameterSpace(
            (("location", RealSpace()), ("concentration", PositiveRealSpace()))
        )

    @scalar_batch
    def logpdf(self, value):
        x = _sym(value)
        lp = self.concentration * sp.cos(x - self.location) - sp.log(
            2 * sp.pi * sp.besseli(0, self.concentration)
        )
        return sp.Piecewise(
            (lp, _and(x >= self.location - sp.pi, x <= self.location + sp.pi)),
            (-sp.oo, True),
        )

    def _mean(self):
        return self.location

    def _sample(self, size, rng):
        return _float(self.location, "location") + rng.vonmises(
            0, _float(self.concentration, "concentration"), size=size
        )


__all__ = [
    "BetaPrime",
    "DiscreteUniform",
    "FDistribution",
    "Frechet",
    "Gumbel",
    "Hypergeometric",
    "InverseGaussian",
    "Kumaraswamy",
    "Maxwell",
    "Nakagami",
    "NoncentralChiSquared",
    "NoncentralF",
    "NoncentralT",
    "PowerDistribution",
    "Rayleigh",
    "Rice",
    "Skellam",
    "Triangular",
    "VonMises",
    "Zipf",
]
