"""Validated probabilistic-model intermediate representation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

import sympy as sp

from .exceptions import ModelValidationError, ObservationError
from .factor import Factor
from .variable import Observation, Variable


@dataclass(frozen=True, slots=True)
class Model:
    """Immutable probabilistic model.

    A model contains variables, probability factors, and zero or more observations.
    Factors are kept separate from variables so inference engines can rewrite/factorize
    the model without mutating the user's variable objects.
    """

    variables: tuple[Variable, ...]
    factors: tuple[Factor, ...] = ()
    observations: Mapping[str, Any] = field(default_factory=dict)
    name: str | None = None

    def __post_init__(self) -> None:
        variables = tuple(self.variables)
        factors = tuple(self.factors)
        names = [variable.name for variable in variables]
        if len(names) != len(set(names)):
            duplicates = sorted({name for name in names if names.count(name) > 1})
            raise ModelValidationError(f"Duplicate variable names: {duplicates}.")

        symbols = [variable.symbol for variable in variables]
        if len(symbols) != len(set(symbols)):
            raise ModelValidationError(
                "Model variables must have distinct SymPy symbols."
            )

        variable_by_name = {variable.name: variable for variable in variables}
        variable_symbols = set(symbols)
        effective_supports = {
            name: variable.support for name, variable in variable_by_name.items()
        }
        for factor in factors:
            if factor.target.name not in variable_by_name:
                raise ModelValidationError(
                    f"Factor target {factor.target.name!r} is not a model variable."
                )
            if factor.distribution is not None:
                effective_supports[factor.target.name] = sp.Intersection(
                    effective_supports[factor.target.name], factor.distribution.support
                )
                if effective_supports[factor.target.name] is sp.S.EmptySet:
                    raise ModelValidationError(
                        f"Distribution support for {factor.target.name!r} is incompatible with its declared support."
                    )
            unknown = factor.free_symbols - variable_symbols
            if unknown:
                # External symbolic constants/hyperparameters are allowed. This check is
                # therefore informational rather than rejecting them.
                pass

        canonical_observations = self._validate_observations(
            self.observations, variable_by_name, factors
        )
        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "factors", factors)
        object.__setattr__(
            self, "observations", MappingProxyType(canonical_observations)
        )

    @staticmethod
    def _validate_observations(
        observations: Mapping[str, Any],
        variable_by_name: Mapping[str, Variable],
        factors: tuple[Factor, ...],
    ) -> dict[str, Any]:
        canonical: dict[str, Any] = {}
        for key, value in observations.items():
            if key not in variable_by_name:
                raise ObservationError(f"Unknown observed variable {key!r}.")
            variable = variable_by_name[key]
            symbolic_value = sp.sympify(value)
            support = variable.support
            for factor in factors:
                if factor.target.name == key and factor.distribution is not None:
                    support = sp.Intersection(support, factor.distribution.support)
            contains = support.contains(symbolic_value)
            if contains is sp.false:
                raise ObservationError(
                    f"Observed value {value!r} is outside support; effective support is {support} for variable {key!r}."
                )
            canonical[key] = value
        return canonical

    @classmethod
    def build(
        cls,
        variables: Iterable[Variable],
        factors: Iterable[Factor] = (),
        *,
        observations: Iterable[Observation] = (),
        name: str | None = None,
    ) -> Model:
        observed = {
            observation.variable.name: observation.value for observation in observations
        }
        return cls(tuple(variables), tuple(factors), observed, name)

    @property
    def variable_map(self) -> Mapping[str, Variable]:
        return MappingProxyType(
            {variable.name: variable for variable in self.variables}
        )

    def effective_support(self, variable: str | Variable) -> sp.Set:
        """Return the declared support intersected with all distribution-factor supports.

        Distribution factors are authoritative about where their target has nonzero
        probability. A variable's declared support is an additional user restriction,
        never an override of the distribution's support.
        """
        name = variable.name if isinstance(variable, Variable) else variable
        try:
            declared = self.variable_map[name]
        except KeyError as exc:
            raise KeyError(f"Unknown variable {name!r}.") from exc
        support: sp.Set = declared.support
        for factor in self.factor_for(name):
            if factor.distribution is not None:
                support = sp.Intersection(support, factor.distribution.support)
        return support

    def effective_variable(self, variable: str | Variable) -> Variable:
        """Return a variable carrying its model-effective support."""
        name = variable.name if isinstance(variable, Variable) else variable
        declared = self.variable_map[name]
        support = self.effective_support(name)
        if support == declared.support:
            return declared
        return replace(declared, support=support)

    @property
    def latent_variables(self) -> tuple[Variable, ...]:
        return tuple(
            self.effective_variable(v)
            for v in self.variables
            if v.name not in self.observations
        )

    @property
    def observed_variables(self) -> tuple[Variable, ...]:
        return tuple(
            self.effective_variable(v)
            for v in self.variables
            if v.name in self.observations
        )

    def factor_for(self, variable: str | Variable) -> tuple[Factor, ...]:
        name = variable.name if isinstance(variable, Variable) else variable
        return tuple(factor for factor in self.factors if factor.target.name == name)

    def parents_of(self, variable: str | Variable) -> tuple[Variable, ...]:
        name = variable.name if isinstance(variable, Variable) else variable
        symbols: set[sp.Symbol] = set()
        for factor in self.factor_for(name):
            symbols.update(factor.parent_symbols)
        return tuple(v for v in self.variables if v.symbol in symbols)

    def observe(self, **values: Any) -> Model:
        observations = dict(self.observations)
        observations.update(values)
        return replace(self, observations=observations)

    def unobserve(self, *variables: str | Variable) -> Model:
        observations = dict(self.observations)
        for variable in variables:
            name = variable.name if isinstance(variable, Variable) else variable
            observations.pop(name, None)
        return replace(self, observations=observations)

    def joint_log_density(self, *, substitute_observations: bool = True) -> sp.Expr:
        expression = sp.Add(*(factor.expression for factor in self.factors))
        if substitute_observations and self.observations:
            substitutions = {
                self.variable_map[name].symbol: sp.sympify(value)
                for name, value in self.observations.items()
            }
            expression = expression.subs(substitutions)
        return sp.simplify(expression)

    def joint_density(self, *, substitute_observations: bool = True) -> sp.Expr:
        return sp.exp(
            self.joint_log_density(substitute_observations=substitute_observations)
        )
