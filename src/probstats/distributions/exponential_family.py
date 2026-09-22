"""Symbolic scalar exponential-family distributions.

Each distribution exposes the canonical exponential-family decomposition

    p(x | theta) = h(x) exp(eta(theta) . T(x) - A(eta(theta))).

The symbolic representation supports conjugacy recognition and exact
marginalisation.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, fields
from functools import reduce
from operator import and_
from typing import Any

import numpy as np
import sympy as sp

from ..spaces import ParameterSpace, PositiveRealSpace, RealSpace
from .base import Distribution, scalar_batch, validate_distribution_parameters


def _sympify_dataclass(instance: object) -> None:
    """Canonicalise all dataclass fields to SymPy expressions in-place."""
    for field in fields(instance):
        object.__setattr__(
            instance, field.name, sp.sympify(getattr(instance, field.name))
        )


def _and(*conditions: sp.Basic | bool) -> sp.Basic:
    if not conditions:
        return sp.true
    return reduce(and_, (sp.sympify(condition) for condition in conditions), sp.true)


def _support_condition(support: sp.Set, value: sp.Expr) -> sp.Basic:
    """Return a decidable membership predicate when numeric input permits it.

    SymPy leaves membership such as ``1.0 in Naturals0`` unevaluated.
    Numeric statistical code commonly supplies integer observations as floats, so
    normalize those exact integer-valued floats without weakening symbolic support
    conditions.
    """
    condition = support.contains(value)
    if condition in (sp.true, sp.false):
        return condition
    if value.is_number and value.is_real is not False:
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError):
            return condition
        if np.isfinite(numeric):
            integer = numeric.is_integer()
            if support == sp.S.Naturals0:
                return sp.true if integer and numeric >= 0 else sp.false
            if support == sp.S.Naturals:
                return sp.true if integer and numeric >= 1 else sp.false
            if support == sp.S.Integers:
                return sp.true if integer else sp.false
    return condition


class ExponentialFamily(Distribution):
    """Base class for regular scalar exponential-family distributions.

    Subclasses provide both ordinary distribution parameters and their canonical
    representation.  ``natural_parameter_constraints`` describes the natural
    parameter space independently of a particular parameterisation; this is useful
    for conjugacy analysis and exact inference.
    """

    @property
    @abstractmethod
    def natural_parameters(self) -> tuple[sp.Expr, ...]:
        """Natural parameter vector ``eta(theta)``."""

    @abstractmethod
    def sufficient_statistics(self, value: Any) -> tuple[sp.Expr, ...]:
        """Sufficient-statistic vector ``T(value)``."""

    @abstractmethod
    def base_measure(self, value: Any) -> sp.Expr:
        """Base measure ``h(value)``."""

    @abstractmethod
    def log_partition_from_natural(self, eta: tuple[Any, ...]) -> sp.Expr:
        """Log-partition function ``A(eta)`` in natural coordinates."""

    @property
    @abstractmethod
    def parameter_constraints(self) -> sp.Basic:
        """Conditions under which ordinary distribution parameters are valid."""

    def natural_parameter_constraints(
        self, eta: tuple[Any, ...] | None = None
    ) -> sp.Basic:
        """Conditions defining the natural-parameter space.

        With no argument, fresh real symbols ``eta1, eta2, ...`` are used.
        """
        if eta is None:
            eta = self.natural_parameter_symbols
        if len(eta) != self.natural_parameters_count:
            raise ValueError(
                f"Expected {self.natural_parameters_count} natural parameters, got {len(eta)}."
            )
        return self._natural_parameter_constraints(tuple(map(sp.sympify, eta)))

    @abstractmethod
    def _natural_parameter_constraints(self, eta: tuple[sp.Expr, ...]) -> sp.Basic:
        """Subclass implementation for ``natural_parameter_constraints``."""

    @property
    def natural_parameters_count(self) -> int:
        return len(self.natural_parameters)

    @property
    def natural_parameter_symbols(self) -> tuple[sp.Symbol, ...]:
        return tuple(
            sp.Symbol(f"eta{index}", real=True)
            for index in range(1, self.natural_parameters_count + 1)
        )

    @property
    def log_partition(self) -> sp.Expr:
        return sp.simplify(self.log_partition_from_natural(self.natural_parameters))

    def canonical_logpdf(self, value: Any) -> sp.Expr:
        """Return ``log(h(x)) + eta.T(x) - A(eta)`` symbolically."""
        statistics = self.sufficient_statistics(value)
        if len(statistics) != self.natural_parameters_count:
            raise ValueError(
                "Natural-parameter and sufficient-statistic dimensions differ."
            )
        inner = sp.Add(
            *(
                eta * statistic
                for eta, statistic in zip(self.natural_parameters, statistics)
            )
        )
        return sp.simplify(
            sp.log(self.base_measure(value)) + inner - self.log_partition
        )

    def canonical_pdf(self, value: Any) -> sp.Expr:
        """Canonical density/mass ``h(x) exp(eta.T(x) - A(eta))``."""
        statistics = self.sufficient_statistics(value)
        inner = sp.Add(
            *(
                eta * statistic
                for eta, statistic in zip(self.natural_parameters, statistics)
            )
        )
        return sp.simplify(
            self.base_measure(value) * sp.exp(inner - self.log_partition)
        )

    @scalar_batch
    def logpdf(self, value: Any) -> sp.Expr:
        if np.asarray(value).ndim > 0:
            return super().logpdf(value)
        x = sp.sympify(value)
        return sp.Piecewise(
            (self.canonical_logpdf(x), _support_condition(self.support, x)),
            (-sp.oo, True),
        )

    @scalar_batch
    def pdf(self, value: Any) -> sp.Expr:
        if np.asarray(value).ndim > 0:
            return super().pdf(value)
        x = sp.sympify(value)
        return sp.Piecewise(
            (self.canonical_pdf(x), _support_condition(self.support, x)),
            (sp.S.Zero, True),
        )


@dataclass(frozen=True, slots=True)
class Exponential(ExponentialFamily):
    """Exponential distribution with rate ``rate``."""

    rate: sp.Expr

    def __post_init__(self) -> None:
        _sympify_dataclass(self)
        validate_distribution_parameters(self)

    @property
    def parameters(self) -> tuple[sp.Basic, ...]:
        return (self.rate,)

    @property
    def support(self) -> sp.Set:
        return sp.Interval(0, sp.oo)

    @property
    def parameter_constraints(self) -> sp.Basic:
        return self.rate > 0

    @property
    def parameter_space(self):
        return ParameterSpace((("rate", PositiveRealSpace()),))

    @property
    def natural_parameters(self) -> tuple[sp.Expr, ...]:
        return (-self.rate,)

    def sufficient_statistics(self, value: Any) -> tuple[sp.Expr, ...]:
        return (sp.sympify(value),)

    def base_measure(self, value: Any) -> sp.Expr:
        return sp.S.One

    def log_partition_from_natural(self, eta: tuple[Any, ...]) -> sp.Expr:
        (eta1,) = map(sp.sympify, eta)
        return -sp.log(-eta1)

    def _natural_parameter_constraints(self, eta: tuple[sp.Expr, ...]) -> sp.Basic:
        (eta1,) = eta
        return eta1 < 0

    def _cdf(self, x):
        return sp.Piecewise((0, x < 0), (1 - sp.exp(-self.rate * x), True))

    def _quantile(self, p):
        return -sp.log(1 - p) / self.rate

    def _raw_moment(self, n):
        return sp.factorial(n) / self.rate**n

    def _mean(self):
        return 1 / self.rate

    def _variance(self):
        return 1 / self.rate**2

    def _entropy(self):
        return 1 - sp.log(self.rate)


@dataclass(frozen=True, slots=True)
class Normal(ExponentialFamily):
    """Univariate normal distribution with mean ``mean`` and std. dev. ``sigma``."""

    mean: sp.Expr
    sigma: sp.Expr

    def __post_init__(self) -> None:
        _sympify_dataclass(self)
        validate_distribution_parameters(self)

    @property
    def parameters(self) -> tuple[sp.Basic, ...]:
        return (self.mean, self.sigma)

    @property
    def support(self) -> sp.Set:
        return sp.S.Reals

    @property
    def parameter_constraints(self) -> sp.Basic:
        return self.sigma > 0

    @property
    def parameter_space(self):
        return ParameterSpace((("mean", RealSpace()), ("sigma", PositiveRealSpace())))

    @property
    def natural_parameters(self) -> tuple[sp.Expr, ...]:
        return (self.mean / self.sigma**2, -sp.S.One / (2 * self.sigma**2))

    def sufficient_statistics(self, value: Any) -> tuple[sp.Expr, ...]:
        value = sp.sympify(value)
        return (value, value**2)

    def base_measure(self, value: Any) -> sp.Expr:
        return 1 / sp.sqrt(2 * sp.pi)

    def log_partition_from_natural(self, eta: tuple[Any, ...]) -> sp.Expr:
        eta1, eta2 = map(sp.sympify, eta)
        return -(eta1**2) / (4 * eta2) - sp.log(-2 * eta2) / 2

    def _natural_parameter_constraints(self, eta: tuple[sp.Expr, ...]) -> sp.Basic:
        _, eta2 = eta
        return eta2 < 0

    def _cdf(self, x):
        return sp.Rational(1, 2) * (
            1 + sp.erf((x - self.mean) / (self.sigma * sp.sqrt(2)))
        )

    def _quantile(self, p):
        return self.mean + self.sigma * sp.sqrt(2) * sp.erfinv(2 * p - 1)

    def _mean(self):
        return self.mean

    def _variance(self):
        return self.sigma**2

    def _entropy(self):
        return sp.log(self.sigma * sp.sqrt(2 * sp.pi * sp.E))


@dataclass(frozen=True, slots=True)
class Poisson(ExponentialFamily):
    """Poisson distribution with positive rate ``rate``."""

    rate: sp.Expr

    def __post_init__(self) -> None:
        _sympify_dataclass(self)
        validate_distribution_parameters(self)

    @property
    def parameters(self) -> tuple[sp.Basic, ...]:
        return (self.rate,)

    @property
    def support(self) -> sp.Set:
        return sp.S.Naturals0

    @property
    def parameter_constraints(self) -> sp.Basic:
        return self.rate > 0

    @property
    def parameter_space(self):
        return ParameterSpace((("rate", PositiveRealSpace()),))

    @property
    def natural_parameters(self) -> tuple[sp.Expr, ...]:
        return (sp.log(self.rate),)

    def sufficient_statistics(self, value: Any) -> tuple[sp.Expr, ...]:
        return (sp.sympify(value),)

    def base_measure(self, value: Any) -> sp.Expr:
        return 1 / sp.factorial(sp.sympify(value))

    def log_partition_from_natural(self, eta: tuple[Any, ...]) -> sp.Expr:
        (eta1,) = map(sp.sympify, eta)
        return sp.exp(eta1)

    def _natural_parameter_constraints(self, eta: tuple[sp.Expr, ...]) -> sp.Basic:
        # log(rate) ranges over every real value when rate > 0.
        return sp.Contains(eta[0], sp.S.Reals)

    def _mean(self):
        return self.rate

    def _variance(self):
        return self.rate


@dataclass(frozen=True, slots=True)
class LogNormal(ExponentialFamily):
    """Log-normal distribution, where log(X) ~ Normal(mean, sigma)."""

    mean: sp.Expr
    sigma: sp.Expr

    def __post_init__(self) -> None:
        _sympify_dataclass(self)
        validate_distribution_parameters(self)

    @property
    def parameters(self) -> tuple[sp.Basic, ...]:
        return (self.mean, self.sigma)

    @property
    def support(self) -> sp.Set:
        return sp.Interval.open(0, sp.oo)

    @property
    def parameter_constraints(self) -> sp.Basic:
        return self.sigma > 0

    @property
    def parameter_space(self):
        return ParameterSpace((("mean", RealSpace()), ("sigma", PositiveRealSpace())))

    @property
    def natural_parameters(self) -> tuple[sp.Expr, ...]:
        return (self.mean / self.sigma**2, -sp.S.One / (2 * self.sigma**2))

    def sufficient_statistics(self, value: Any) -> tuple[sp.Expr, ...]:
        value = sp.sympify(value)
        return (sp.log(value), sp.log(value) ** 2)

    def base_measure(self, value: Any) -> sp.Expr:
        value = sp.sympify(value)
        return 1 / (sp.sqrt(2 * sp.pi) * value)

    def log_partition_from_natural(self, eta: tuple[Any, ...]) -> sp.Expr:
        eta1, eta2 = map(sp.sympify, eta)
        return -(eta1**2) / (4 * eta2) - sp.log(-2 * eta2) / 2

    def _natural_parameter_constraints(self, eta: tuple[sp.Expr, ...]) -> sp.Basic:
        _, eta2 = eta
        return eta2 < 0


@dataclass(frozen=True, slots=True)
class Gamma(ExponentialFamily):
    """Gamma distribution with shape ``shape`` and scale ``scale``."""

    shape: sp.Expr
    scale: sp.Expr

    def __post_init__(self) -> None:
        _sympify_dataclass(self)
        validate_distribution_parameters(self)

    @property
    def parameters(self) -> tuple[sp.Basic, ...]:
        return (self.shape, self.scale)

    @property
    def support(self) -> sp.Set:
        return sp.Interval.open(0, sp.oo)

    @property
    def parameter_constraints(self) -> sp.Basic:
        return _and(self.shape > 0, self.scale > 0)

    @property
    def parameter_space(self):
        return ParameterSpace(
            (("shape", PositiveRealSpace()), ("scale", PositiveRealSpace()))
        )

    @property
    def natural_parameters(self) -> tuple[sp.Expr, ...]:
        return (self.shape - 1, -1 / self.scale)

    def sufficient_statistics(self, value: Any) -> tuple[sp.Expr, ...]:
        value = sp.sympify(value)
        return (sp.log(value), value)

    def base_measure(self, value: Any) -> sp.Expr:
        return sp.S.One

    def log_partition_from_natural(self, eta: tuple[Any, ...]) -> sp.Expr:
        eta1, eta2 = map(sp.sympify, eta)
        return sp.log(sp.gamma(eta1 + 1)) - (eta1 + 1) * sp.log(-eta2)

    def _natural_parameter_constraints(self, eta: tuple[sp.Expr, ...]) -> sp.Basic:
        eta1, eta2 = eta
        return _and(eta1 > -1, eta2 < 0)

    def _raw_moment(self, n):
        return self.scale**n * sp.gamma(self.shape + n) / sp.gamma(self.shape)

    def _mean(self):
        return self.shape * self.scale

    def _variance(self):
        return self.shape * self.scale**2

    def _entropy(self):
        return (
            self.shape
            + sp.log(self.scale * sp.gamma(self.shape))
            + (1 - self.shape) * sp.polygamma(0, self.shape)
        )


@dataclass(frozen=True, slots=True)
class InverseGamma(ExponentialFamily):
    """Inverse-gamma distribution with shape ``shape`` and scale ``scale``."""

    shape: sp.Expr
    scale: sp.Expr

    def __post_init__(self) -> None:
        _sympify_dataclass(self)
        validate_distribution_parameters(self)

    @property
    def parameters(self) -> tuple[sp.Basic, ...]:
        return (self.shape, self.scale)

    @property
    def support(self) -> sp.Set:
        return sp.Interval.open(0, sp.oo)

    @property
    def parameter_constraints(self) -> sp.Basic:
        return _and(self.shape > 0, self.scale > 0)

    @property
    def parameter_space(self):
        return ParameterSpace(
            (("shape", PositiveRealSpace()), ("scale", PositiveRealSpace()))
        )

    @property
    def natural_parameters(self) -> tuple[sp.Expr, ...]:
        return (-self.shape - 1, -self.scale)

    def sufficient_statistics(self, value: Any) -> tuple[sp.Expr, ...]:
        value = sp.sympify(value)
        return (sp.log(value), 1 / value)

    def base_measure(self, value: Any) -> sp.Expr:
        return sp.S.One

    def log_partition_from_natural(self, eta: tuple[Any, ...]) -> sp.Expr:
        eta1, eta2 = map(sp.sympify, eta)
        return sp.log(sp.gamma(-eta1 - 1)) - (-eta1 - 1) * sp.log(-eta2)

    def _natural_parameter_constraints(self, eta: tuple[sp.Expr, ...]) -> sp.Basic:
        eta1, eta2 = eta
        return _and(eta1 < -1, eta2 < 0)


__all__ = [
    "Exponential",
    "ExponentialFamily",
    "Gamma",
    "InverseGamma",
    "LogNormal",
    "Normal",
    "Poisson",
]
