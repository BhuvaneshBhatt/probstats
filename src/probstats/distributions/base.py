"""Core distribution interfaces and shared convenience methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
from typing import Any

import numpy as np
import sympy as sp

from ..spaces import MeasureType, ParameterSpace, RealSpace, ScalarEventSpace


def scalar_batch(method):
    """Apply a scalar distribution method elementwise to array-like values.

    Distribution formulas remain scalar implementations. This decorator adds the
    package's object-array batching contract without rewriting subclasses at class
    creation time.
    """

    @wraps(method)
    def wrapped(self, value, *args, **kwargs):
        if isinstance(self.event_space, ScalarEventSpace):
            array = np.asarray(value)
            if array.ndim > 0:
                out = np.empty(array.shape, dtype=object)
                for index in np.ndindex(array.shape):
                    item = array[index]
                    if hasattr(item, "item"):
                        item = item.item()
                    out[index] = method(self, item, *args, **kwargs)
                return out
        return method(self, value, *args, **kwargs)

    return wrapped


class Distribution(ABC):
    @property
    @abstractmethod
    def support(self) -> sp.Set: ...

    @abstractmethod
    def logpdf(self, value: Any) -> sp.Expr: ...

    def pdf(self, value: Any) -> sp.Expr:
        log_value = self.logpdf(value)
        if isinstance(log_value, np.ndarray):
            out = np.empty(log_value.shape, dtype=object)
            for index in np.ndindex(log_value.shape):
                out[index] = sp.exp(log_value[index])
            return out
        return sp.exp(log_value)

    @property
    def measure_type(self) -> MeasureType:
        return (
            MeasureType.DISCRETE
            if self.support.is_subset(sp.S.Integers) is True
            else MeasureType.CONTINUOUS
        )

    @property
    def event_space(self):
        return ScalarEventSpace(self.support)

    def pmf(self, value: Any) -> sp.Expr:
        if self.measure_type is not MeasureType.DISCRETE:
            raise TypeError(f"{type(self).__name__} is not discrete")
        return self.pdf(value)

    def logpmf(self, value: Any) -> sp.Expr:
        if self.measure_type is not MeasureType.DISCRETE:
            raise TypeError(f"{type(self).__name__} is not discrete")
        return self.logpdf(value)

    def cdf(self, value):
        from ..functionals import cdf

        return cdf(self, value)

    def survival(self, value):
        from ..functionals import survival

        return survival(self, value)

    def quantile(self, p):
        from ..functionals import quantile

        return quantile(self, p)

    def inverse_survival(self, p):
        from ..functionals import inverse_survival

        return inverse_survival(self, p)

    def hazard(self, value):
        from ..functionals import hazard

        return hazard(self, value)

    def cumulative_hazard(self, value):
        from ..functionals import cumulative_hazard

        return cumulative_hazard(self, value)

    def probability(
        self,
        event,
        *,
        variable=None,
        numerical_fallback=False,
        samples=100_000,
        rng=None,
        return_result=False,
    ):
        from ..functionals import probability

        return probability(
            self,
            event,
            variable=variable,
            numerical_fallback=numerical_fallback,
            samples=samples,
            rng=rng,
            return_result=return_result,
        )

    def expectation(
        self,
        expression=None,
        *,
        variable=None,
        variables=None,
        numerical_fallback=False,
        samples=100_000,
        rng=None,
        return_result=False,
    ):
        from ..functionals import expectation

        return expectation(
            self,
            expression,
            variable=variable,
            variables=variables,
            numerical_fallback=numerical_fallback,
            samples=samples,
            rng=rng,
            return_result=return_result,
        )

    def moment(self, order: int, *, central=False):
        from ..functionals import moment

        return moment(self, order, central=central)

    def factorial_moment(self, order: int):
        from ..functionals import factorial_moment

        return factorial_moment(self, order)

    def cumulant(self, order: int):
        from ..functionals import cumulant

        return cumulant(self, order)

    @property
    def mean_value(self):
        from ..functionals import mean

        return mean(self)

    @property
    def variance_value(self):
        from ..functionals import variance

        return variance(self)

    def entropy(self):
        from ..functionals import entropy

        return entropy(self)

    def kl_divergence(self, other, **kwargs):
        from ..information import kl_divergence

        return kl_divergence(self, other, **kwargs)

    def cross_entropy(self, other, **kwargs):
        from ..information import cross_entropy

        return cross_entropy(self, other, **kwargs)

    def renyi_divergence(self, other, alpha, **kwargs):
        from ..information import renyi_divergence

        return renyi_divergence(self, other, alpha, **kwargs)

    def jensen_shannon_divergence(self, other, **kwargs):
        from ..information import jensen_shannon_divergence

        return jensen_shannon_divergence(self, other, **kwargs)

    def hellinger_distance(self, other, **kwargs):
        from ..information import hellinger_distance

        return hellinger_distance(self, other, **kwargs)

    def bhattacharyya_distance(self, other, **kwargs):
        from ..information import bhattacharyya_distance

        return bhattacharyya_distance(self, other, **kwargs)

    def canonicalize(self):
        from ..algebra import canonicalize_distribution

        return canonicalize_distribution(self)

    def equivalent_to(self, other):
        from ..algebra import equivalent_distributions

        return equivalent_distributions(self, other)

    @property
    def metadata(self):
        from ..algebra import distribution_metadata

        return distribution_metadata(self)

    def characteristic_function(self, t=None):
        from ..symbolic import characteristic_function

        return characteristic_function(self, t)

    def moment_generating_function(self, t=None):
        from ..symbolic import moment_generating_function

        return moment_generating_function(self, t)

    def cumulant_generating_function(self, t=None):
        from ..symbolic import cumulant_generating_function

        return cumulant_generating_function(self, t)

    def central_moment_generating_function(self, t=None):
        from ..symbolic import central_moment_generating_function

        return central_moment_generating_function(self, t)

    def factorial_moment_generating_function(self, t=None):
        from ..symbolic import factorial_moment_generating_function

        return factorial_moment_generating_function(self, t)

    def probability_generating_function(self, z=None):
        from ..symbolic import probability_generating_function

        return probability_generating_function(self, z)

    def likelihood(self, observations):
        from ..functionals import likelihood_value

        return likelihood_value(self, observations)

    def log_likelihood(self, observations):
        from ..functionals import log_likelihood

        return log_likelihood(self, observations)

    def sample(self, size=None, rng=None, *, max_attempts: int = 100_000):
        from ..sampling import sample

        return sample(self, size=size, rng=rng, max_attempts=max_attempts)

    @property
    def parameter_constraints(self) -> sp.Basic:
        return sp.true

    @property
    def parameter_space(self) -> ParameterSpace:
        return ParameterSpace(
            tuple(
                (f"parameter_{i}", RealSpace()) for i, _ in enumerate(self.parameters)
            )
        )

    @property
    def parameter_space_constraints(self) -> sp.Basic:
        return self.parameter_space.constraints(self.parameters)

    @property
    def parameters(self) -> tuple[sp.Basic, ...]:
        return ()

    @property
    def free_symbols(self) -> set[sp.Symbol]:
        symbols = set()
        for parameter in self.parameters:
            if hasattr(parameter, "free_symbols"):
                symbols.update(parameter.free_symbols)
        return symbols


@dataclass(frozen=True, slots=True)
class SymbolicDistribution(Distribution):
    value_symbol: sp.Symbol
    log_density: sp.Expr
    domain: sp.Set = sp.S.Reals
    normalizer_known: bool = False
    discrete: bool = False

    def __post_init__(self):
        object.__setattr__(self, "value_symbol", sp.sympify(self.value_symbol))
        object.__setattr__(self, "log_density", sp.sympify(self.log_density))
        if not isinstance(self.value_symbol, sp.Symbol):
            raise TypeError("value_symbol must be a SymPy Symbol.")
        if not isinstance(self.domain, sp.Set):
            raise TypeError("domain must be a SymPy Set.")

    @property
    def support(self):
        return self.domain

    @property
    def measure_type(self):
        return MeasureType.DISCRETE if self.discrete else MeasureType.CONTINUOUS

    @scalar_batch
    def logpdf(self, value):
        return self.log_density.subs(self.value_symbol, sp.sympify(value))

    @property
    def parameters(self):
        return tuple(
            sorted(
                self.log_density.free_symbols - {self.value_symbol},
                key=sp.default_sort_key,
            )
        )


def validate_parameter_condition(condition: Any, *, message: str) -> sp.Basic:
    """Reject a parameter condition only when SymPy can prove it false."""
    condition = sp.sympify(condition)
    try:
        simplified = sp.simplify(condition)
    except (TypeError, ValueError, NotImplementedError, sp.PolynomialError):
        simplified = condition
    if simplified is sp.false or simplified is False:
        raise ValueError(message)
    return simplified


def _reject_nonfinite_numeric_parameters(distribution: Distribution) -> None:
    """Reject concrete NaN/infinite parameters while preserving symbolic ones."""
    for parameter in distribution.parameters:
        values = (
            tuple(parameter) if isinstance(parameter, sp.MatrixBase) else (parameter,)
        )
        for value in values:
            expression = sp.sympify(value)
            if expression.free_symbols:
                continue
            if (
                expression in (sp.nan, sp.oo, -sp.oo, sp.zoo)
                or expression.is_finite is False
            ):
                raise ValueError(
                    f"Invalid parameters for {type(distribution).__name__}: "
                    "concrete parameters must be finite."
                )


def validate_distribution_parameters(distribution: Distribution) -> sp.Basic:
    _reject_nonfinite_numeric_parameters(distribution)
    declared = sp.sympify(distribution.parameter_constraints)
    try:
        space_constraints = sp.sympify(distribution.parameter_space_constraints)
    except (TypeError, ValueError, NotImplementedError, AttributeError):
        space_constraints = sp.true
    constraints = sp.And(declared, space_constraints)
    return validate_parameter_condition(
        constraints,
        message=(
            f"Invalid parameters for {type(distribution).__name__}: "
            f"required {constraints}."
        ),
    )
