"""Core scalar and multivariate probability distributions for probstats."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import reduce
from operator import and_
from typing import Any

import sympy as sp

from .._exact_linear_algebra import exact_determinant, exact_solve, quadratic_form
from ..spaces import (
    CountVectorEventSpace,
    MatrixEventSpace,
    MeasureType,
    NaturalNumberSpace,
    ParameterSpace,
    PositiveDefiniteMatrixSpace,
    PositiveRealSpace,
    PositiveVectorSpace,
    SimplexEventSpace,
    UnitIntervalSpace,
    VectorEventSpace,
)
from .base import (
    Distribution,
    scalar_batch,
    validate_distribution_parameters,
    validate_parameter_condition,
)
from .exponential_family import ExponentialFamily, _sympify_dataclass


def _and(*conditions):
    return reduce(and_, (sp.sympify(c) for c in conditions), sp.true)


def _probability_vector(values: Sequence[Any]) -> tuple[sp.Expr, ...]:
    out = tuple(sp.sympify(v) for v in values)
    if not out:
        raise ValueError("probability vector must be nonempty")
    cond = _and(*(p >= 0 for p in out), sp.Eq(sp.Add(*out), 1))
    validate_parameter_condition(
        cond, message="probabilities must be nonnegative and sum to 1"
    )
    return out


def _positive_vector(values: Sequence[Any], *, name: str) -> tuple[sp.Expr, ...]:
    out = tuple(sp.sympify(v) for v in values)
    if not out:
        raise ValueError(f"{name} must be nonempty")
    cond = _and(*(v > 0 for v in out))
    validate_parameter_condition(cond, message=f"{name} entries must be positive")
    return out


def _as_column(value: Any) -> sp.ImmutableDenseMatrix:
    m = sp.ImmutableMatrix(value)
    if m.cols == 1:
        return m
    if m.rows == 1:
        return sp.ImmutableMatrix(list(m))
    raise ValueError("value must be a vector")


def _validate_symmetric_pd(matrix: sp.ImmutableDenseMatrix, *, name: str) -> None:
    if matrix.rows != matrix.cols:
        raise ValueError(f"{name} must be square")
    if matrix != matrix.T:
        raise ValueError(f"{name} must be symmetric")
    if not matrix.free_symbols and matrix.is_positive_definite is not True:
        raise ValueError(f"{name} must be positive definite")


@dataclass(frozen=True, slots=True)
class Bernoulli(ExponentialFamily):
    """Bernoulli distribution on ``{0, 1}`` with success probability ``p``."""

    p: sp.Expr

    def __post_init__(self):
        _sympify_dataclass(self)
        validate_distribution_parameters(self)

    @property
    def parameters(self):
        return (self.p,)

    @property
    def support(self):
        return sp.FiniteSet(0, 1)

    @property
    def parameter_constraints(self):
        return _and(self.p >= 0, self.p <= 1)

    @property
    def parameter_space(self):
        return ParameterSpace((("p", UnitIntervalSpace()),))

    @property
    def natural_parameters(self):
        return (sp.log(self.p / (1 - self.p)),)

    def sufficient_statistics(self, value):
        return (sp.sympify(value),)

    def base_measure(self, value):
        return sp.S.One

    def log_partition_from_natural(self, eta):
        return sp.log(1 + sp.exp(sp.sympify(eta[0])))

    def _natural_parameter_constraints(self, eta):
        return sp.Contains(eta[0], sp.S.Reals)

    @scalar_batch
    def pdf(self, value):
        x = sp.sympify(value)
        return sp.Piecewise((1 - self.p, sp.Eq(x, 0)), (self.p, sp.Eq(x, 1)), (0, True))

    @scalar_batch
    def logpdf(self, value):
        return sp.log(self.pdf(value))

    def _mean(self):
        return self.p

    def _variance(self):
        return self.p * (1 - self.p)

    def _entropy(self):
        return -self.p * sp.log(self.p) - (1 - self.p) * sp.log(1 - self.p)

    def _cdf(self, x):
        return sp.Piecewise((0, x < 0), (1 - self.p, x < 1), (1, True))

    def _quantile(self, q):
        return sp.Piecewise((0, q <= 1 - self.p), (1, True))


@dataclass(frozen=True, slots=True)
class Binomial(ExponentialFamily):
    """Binomial distribution with ``n`` trials and success probability ``p``."""

    n: sp.Expr
    p: sp.Expr

    def __post_init__(self):
        _sympify_dataclass(self)
        validate_distribution_parameters(self)
        if self.n.is_integer is False or self.n.is_negative is True:
            raise ValueError("Binomial n must be a nonnegative integer")

    @property
    def parameters(self):
        return (self.n, self.p)

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
            self.n >= 0, sp.Eq(sp.floor(self.n), self.n), self.p >= 0, self.p <= 1
        )

    @property
    def parameter_space(self):
        return ParameterSpace((("n", NaturalNumberSpace()), ("p", UnitIntervalSpace())))

    @property
    def natural_parameters(self):
        return (sp.log(self.p / (1 - self.p)),)

    def sufficient_statistics(self, value):
        return (sp.sympify(value),)

    def base_measure(self, value):
        return sp.binomial(self.n, sp.sympify(value))

    def log_partition_from_natural(self, eta):
        return self.n * sp.log(1 + sp.exp(sp.sympify(eta[0])))

    def _natural_parameter_constraints(self, eta):
        return sp.Contains(eta[0], sp.S.Reals)

    def _mean(self):
        return self.n * self.p

    def _variance(self):
        return self.n * self.p * (1 - self.p)

    @scalar_batch
    def pdf(self, value):
        x = sp.sympify(value)
        valid = _and(x >= 0, x <= self.n, sp.Eq(sp.floor(x), x))
        mass = sp.binomial(self.n, x) * self.p**x * (1 - self.p) ** (self.n - x)
        # SymPy represents 0**0 as 1, so this direct mass formula also handles
        # the degenerate p=0 and p=1 boundary distributions exactly.
        return sp.Piecewise((sp.simplify(mass), valid), (sp.S.Zero, True))

    @scalar_batch
    def logpdf(self, value):
        mass = self.pdf(value)
        return sp.Piecewise((sp.log(mass), sp.Ne(mass, 0)), (-sp.oo, True))


@dataclass(frozen=True, slots=True)
class Beta(ExponentialFamily):
    """Beta distribution with positive shape parameters ``alpha`` and ``beta``."""

    alpha: sp.Expr
    beta: sp.Expr

    def __post_init__(self):
        _sympify_dataclass(self)
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

    @property
    def natural_parameters(self):
        return (self.alpha - 1, self.beta - 1)

    def sufficient_statistics(self, value):
        x = sp.sympify(value)
        return (sp.log(x), sp.log(1 - x))

    def base_measure(self, value):
        return sp.S.One

    def log_partition_from_natural(self, eta):
        a, b = map(sp.sympify, eta)
        return sp.log(sp.beta(a + 1, b + 1))

    def _natural_parameter_constraints(self, eta):
        a, b = eta
        return _and(a > -1, b > -1)

    def _mean(self):
        return self.alpha / (self.alpha + self.beta)

    def _variance(self):
        a, b = self.alpha, self.beta
        return a * b / ((a + b) ** 2 * (a + b + 1))

    def _raw_moment(self, n):
        return sp.rf(self.alpha, n) / sp.rf(self.alpha + self.beta, n)

    def _entropy(self):
        a, b = self.alpha, self.beta
        return (
            sp.log(sp.beta(a, b))
            - (a - 1) * sp.polygamma(0, a)
            - (b - 1) * sp.polygamma(0, b)
            + (a + b - 2) * sp.polygamma(0, a + b)
        )


@dataclass(frozen=True, slots=True)
class Categorical(Distribution):
    """Categorical distribution over integer labels ``0, ..., k-1``."""

    probabilities: tuple[sp.Expr, ...]

    def __init__(self, probabilities: Sequence[Any]):
        object.__setattr__(self, "probabilities", _probability_vector(probabilities))

    @property
    def parameters(self):
        return self.probabilities

    @property
    def support(self):
        return sp.FiniteSet(*range(len(self.probabilities)))

    @property
    def parameter_constraints(self):
        return _and(
            *(p >= 0 for p in self.probabilities), sp.Eq(sp.Add(*self.probabilities), 1)
        )

    @property
    def parameter_space(self):
        return ParameterSpace(
            (("probabilities", SimplexEventSpace(len(self.probabilities))),)
        )

    @property
    def parameter_space_constraints(self):
        return self.parameter_space.constraints((self.probabilities,))

    @scalar_batch
    def pdf(self, value):
        x = sp.sympify(value)
        return sp.Piecewise(
            *[(p, sp.Eq(x, i)) for i, p in enumerate(self.probabilities)], (0, True)
        )

    @scalar_batch
    def logpdf(self, value):
        return sp.log(self.pdf(value))

    def _mean(self):
        return sp.Add(*(i * p for i, p in enumerate(self.probabilities)))

    def _variance(self):
        mu = self._mean()
        return sp.Add(*(i**2 * p for i, p in enumerate(self.probabilities))) - mu**2

    def _entropy(self):
        return -sp.Add(*(p * sp.log(p) for p in self.probabilities))


@dataclass(frozen=True, slots=True)
class Multinomial(Distribution):
    """Multinomial count-vector distribution."""

    n: sp.Expr
    probabilities: tuple[sp.Expr, ...]

    def __init__(self, n: Any, probabilities: Sequence[Any]):
        n = sp.sympify(n)
        if n.is_integer is False or n.is_negative is True:
            raise ValueError("Multinomial n must be a nonnegative integer")
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "probabilities", _probability_vector(probabilities))

    @property
    def parameters(self):
        return (self.n, *self.probabilities)

    @property
    def dimension(self):
        return len(self.probabilities)

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
            *(p >= 0 for p in self.probabilities),
            sp.Eq(sp.Add(*self.probabilities), 1),
        )

    @property
    def parameter_space(self):
        return ParameterSpace(
            (
                ("n", NaturalNumberSpace()),
                ("probabilities", SimplexEventSpace(self.dimension)),
            )
        )

    @property
    def parameter_space_constraints(self):
        return self.parameter_space.constraints((self.n, self.probabilities))

    @scalar_batch
    def logpdf(self, value):
        xs = tuple(map(sp.sympify, value))
        if len(xs) != self.dimension:
            raise ValueError("count vector has wrong dimension")
        return sp.simplify(
            sp.loggamma(self.n + 1)
            - sum(sp.loggamma(x + 1) for x in xs)
            + sum(x * sp.log(p) for x, p in zip(xs, self.probabilities))
        )

    @scalar_batch
    def pdf(self, value):
        return sp.exp(self.logpdf(value))

    def _mean(self):
        return sp.ImmutableMatrix([self.n * p for p in self.probabilities])

    def _variance(self):
        ps = self.probabilities
        n = self.n
        return sp.ImmutableMatrix(
            self.dimension,
            self.dimension,
            lambda i, j: n * ps[i] * (1 - ps[i]) if i == j else -n * ps[i] * ps[j],
        )


@dataclass(frozen=True, slots=True)
class Dirichlet(Distribution):
    """Dirichlet distribution over a probability simplex."""

    concentration: tuple[sp.Expr, ...]

    def __init__(self, concentration: Sequence[Any]):
        object.__setattr__(
            self,
            "concentration",
            _positive_vector(concentration, name="Dirichlet concentration"),
        )

    @property
    def parameters(self):
        return self.concentration

    @property
    def dimension(self):
        return len(self.concentration)

    @property
    def support(self):
        return sp.ProductSet(*([sp.Interval(0, 1)] * self.dimension))

    @property
    def event_space(self):
        return SimplexEventSpace(self.dimension)

    @property
    def measure_type(self):
        return MeasureType.CONTINUOUS

    def _mean(self):
        a0 = sp.Add(*self.concentration)
        return sp.ImmutableMatrix([a / a0 for a in self.concentration])

    def _variance(self):
        a0 = sp.Add(*self.concentration)
        return sp.ImmutableMatrix(
            [a * (a0 - a) / (a0**2 * (a0 + 1)) for a in self.concentration]
        )

    @property
    def parameter_constraints(self):
        return _and(*(a > 0 for a in self.concentration))

    @property
    def parameter_space(self):
        return ParameterSpace((("concentration", PositiveVectorSpace(self.dimension)),))

    @property
    def parameter_space_constraints(self):
        return self.parameter_space.constraints((self.concentration,))

    @scalar_batch
    def logpdf(self, value):
        xs = tuple(map(sp.sympify, value))
        if len(xs) != self.dimension:
            raise ValueError("simplex vector has wrong dimension")
        a0 = sp.Add(*self.concentration)
        return sp.simplify(
            sp.loggamma(a0)
            - sum(sp.loggamma(a) for a in self.concentration)
            + sum((a - 1) * sp.log(x) for a, x in zip(self.concentration, xs))
        )

    @scalar_batch
    def pdf(self, value):
        return sp.exp(self.logpdf(value))


@dataclass(frozen=True, slots=True)
class MultivariateNormal(Distribution):
    """Multivariate normal distribution with mean vector and covariance matrix."""

    mean: sp.ImmutableDenseMatrix
    covariance: sp.ImmutableDenseMatrix

    def __init__(self, mean: Sequence[Any], covariance: Any):
        m = _as_column(mean)
        c = sp.ImmutableMatrix(covariance)
        if c.rows != m.rows:
            raise ValueError("mean and covariance dimensions are incompatible")
        _validate_symmetric_pd(c, name="covariance")
        object.__setattr__(self, "mean", m)
        object.__setattr__(self, "covariance", c)

    @property
    def parameters(self):
        return tuple(self.mean) + tuple(self.covariance)

    @property
    def dimension(self):
        return self.mean.rows

    @property
    def support(self):
        return sp.ProductSet(*([sp.S.Reals] * self.dimension))

    @property
    def event_space(self):
        return VectorEventSpace(self.dimension, sp.S.Reals)

    @property
    def measure_type(self):
        return MeasureType.CONTINUOUS

    def _mean(self):
        return self.mean

    def _variance(self):
        return self.covariance

    @property
    def parameter_constraints(self):
        return sp.true

    @property
    def parameter_space(self):
        return ParameterSpace(
            (
                ("mean", VectorEventSpace(self.dimension, sp.S.Reals)),
                ("covariance", PositiveDefiniteMatrixSpace(self.dimension)),
            )
        )

    @property
    def parameter_space_constraints(self):
        return self.parameter_space.constraints((self.mean, self.covariance))

    @scalar_batch
    def logpdf(self, value):
        x = _as_column(value)
        if x.rows != self.dimension:
            raise ValueError("value has wrong dimension")
        d = self.dimension
        delta = x - self.mean
        q = quadratic_form(self.covariance, delta)
        return sp.simplify(
            -sp.Rational(d, 2) * sp.log(2 * sp.pi)
            - sp.Rational(1, 2) * sp.log(self.covariance.det())
            - q / 2
        )

    @scalar_batch
    def pdf(self, value):
        return sp.exp(self.logpdf(value))


@dataclass(frozen=True, slots=True)
class Wishart(Distribution):
    """Wishart distribution with degrees of freedom ``df`` and scale matrix ``scale``."""

    df: sp.Expr
    scale: sp.ImmutableDenseMatrix

    def __init__(self, df: Any, scale: Any):
        df = sp.sympify(df)
        s = sp.ImmutableMatrix(scale)
        _validate_symmetric_pd(s, name="scale")
        if (df - (s.rows - 1)).is_nonpositive is True:
            raise ValueError("Wishart df must exceed dimension - 1")
        object.__setattr__(self, "df", df)
        object.__setattr__(self, "scale", s)

    @property
    def dimension(self):
        return self.scale.rows

    @property
    def parameters(self):
        return (self.df, *tuple(self.scale))

    @property
    def support(self):
        return sp.S.UniversalSet

    @property
    def event_space(self):
        return MatrixEventSpace(
            self.dimension, self.dimension, symmetric=True, positive_definite=True
        )

    @property
    def measure_type(self):
        return MeasureType.CONTINUOUS

    @property
    def parameter_constraints(self):
        return self.df > self.dimension - 1

    @property
    def parameter_space(self):
        d = self.dimension
        return ParameterSpace(
            (("df", PositiveRealSpace()), ("scale", PositiveDefiniteMatrixSpace(d))),
            relations=(lambda values, d=d: sp.sympify(values["df"]) > d - 1,),
        )

    @property
    def parameter_space_constraints(self):
        return self.parameter_space.constraints((self.df, self.scale))

    @scalar_batch
    def logpdf(self, value):
        x = sp.ImmutableMatrix(value)
        if x.rows != self.dimension or x.cols != self.dimension:
            raise ValueError("matrix has wrong dimension")
        p = self.dimension
        v = self.df
        multi_gamma = sp.Rational(p * (p - 1), 4) * sp.log(sp.pi) + sum(
            sp.loggamma(v / 2 + sp.Rational(1 - j, 2)) for j in range(1, p + 1)
        )
        return sp.simplify(
            (v - p - 1) / 2 * sp.log(x.det())
            - sp.trace(exact_solve(self.scale, x)) / 2
            - v * p / 2 * sp.log(2)
            - v / 2 * sp.log(exact_determinant(self.scale))
            - multi_gamma
        )


@dataclass(frozen=True, slots=True)
class InverseWishart(Distribution):
    """Inverse-Wishart distribution with degrees of freedom ``df`` and scale matrix ``scale``."""

    df: sp.Expr
    scale: sp.ImmutableDenseMatrix

    def __init__(self, df: Any, scale: Any):
        df = sp.sympify(df)
        s = sp.ImmutableMatrix(scale)
        _validate_symmetric_pd(s, name="scale")
        if (df - (s.rows - 1)).is_nonpositive is True:
            raise ValueError("InverseWishart df must exceed dimension - 1")
        object.__setattr__(self, "df", df)
        object.__setattr__(self, "scale", s)

    @property
    def dimension(self):
        return self.scale.rows

    @property
    def parameters(self):
        return (self.df, *tuple(self.scale))

    @property
    def support(self):
        return sp.S.UniversalSet

    @property
    def event_space(self):
        return MatrixEventSpace(
            self.dimension, self.dimension, symmetric=True, positive_definite=True
        )

    @property
    def measure_type(self):
        return MeasureType.CONTINUOUS

    @property
    def parameter_constraints(self):
        return self.df > self.dimension - 1

    @property
    def parameter_space(self):
        d = self.dimension
        return ParameterSpace(
            (("df", PositiveRealSpace()), ("scale", PositiveDefiniteMatrixSpace(d))),
            relations=(lambda values, d=d: sp.sympify(values["df"]) > d - 1,),
        )

    @property
    def parameter_space_constraints(self):
        return self.parameter_space.constraints((self.df, self.scale))

    @scalar_batch
    def logpdf(self, value):
        x = sp.ImmutableMatrix(value)
        if x.rows != self.dimension or x.cols != self.dimension:
            raise ValueError("matrix has wrong dimension")
        p = self.dimension
        v = self.df
        multi_gamma = sp.Rational(p * (p - 1), 4) * sp.log(sp.pi) + sum(
            sp.loggamma(v / 2 + sp.Rational(1 - j, 2)) for j in range(1, p + 1)
        )
        return sp.simplify(
            v / 2 * sp.log(exact_determinant(self.scale))
            - (v + p + 1) / 2 * sp.log(x.det())
            - sp.trace(exact_solve(x, self.scale)) / 2
            - v * p / 2 * sp.log(2)
            - multi_gamma
        )


__all__ = [
    "Bernoulli",
    "Beta",
    "Binomial",
    "Categorical",
    "Dirichlet",
    "InverseWishart",
    "Multinomial",
    "MultivariateNormal",
    "Wishart",
]
