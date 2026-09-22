"""Variable objects used by the probabilistic-model intermediate representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import sympy as sp

from .backend import as_symbol


@dataclass(frozen=True, slots=True)
class Variable:
    """A named mathematical variable.

    ``symbol`` is the canonical identity used inside symbolic expressions. ``support``
    describes values the variable is allowed to take; it defaults to the real line.
    """

    name: str | sp.Symbol
    support: sp.Set = sp.S.Reals
    symbol: sp.Symbol = field(init=False, compare=True)

    def __post_init__(self) -> None:
        symbol = as_symbol(
            self.name, real=True if self.support.is_subset(sp.S.Reals) else None
        )
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "name", symbol.name)
        if not isinstance(self.support, sp.Set):
            raise TypeError("support must be a SymPy Set.")

    def __str__(self) -> str:
        return self.name

    def __sympy__(self) -> sp.Symbol:
        return self.symbol


@dataclass(frozen=True, slots=True)
class Parameter(Variable):
    """An unknown model variable, usually assigned a prior factor."""


@dataclass(frozen=True, slots=True)
class RandomVariable(Variable):
    """A stochastic variable whose probability law is represented by model factors."""


@dataclass(frozen=True, slots=True)
class Observation:
    """An observed value attached to a model variable."""

    variable: Variable
    value: Any

    def symbolic_value(self) -> sp.Basic:
        return sp.sympify(self.value)
