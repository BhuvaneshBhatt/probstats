"""Probability factors for the model intermediate representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sympy as sp

from ..distributions.base import (
    Distribution,
    validate_distribution_parameters,
)
from .variable import Variable


@dataclass(frozen=True, slots=True)
class Factor:
    """A probability factor associated with a target variable.

    A factor may be specified either as a ``Distribution`` for its target or directly as
    an unnormalised symbolic log-density. The latter is useful for custom likelihoods.
    """

    target: Variable
    distribution: Distribution | None = None
    log_density: sp.Expr | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if (self.distribution is None) == (self.log_density is None):
            raise ValueError("Specify exactly one of distribution or log_density.")
        if self.log_density is not None:
            object.__setattr__(self, "log_density", sp.sympify(self.log_density))
        if self.distribution is not None:
            validate_distribution_parameters(self.distribution)

    @classmethod
    def from_distribution(
        cls, target: Variable, distribution: Distribution, *, name: str | None = None
    ) -> Factor:
        return cls(target=target, distribution=distribution, name=name)

    @classmethod
    def from_log_density(
        cls, target: Variable, expression: Any, *, name: str | None = None
    ) -> Factor:
        return cls(target=target, log_density=sp.sympify(expression), name=name)

    @property
    def expression(self) -> sp.Expr:
        """Return this factor's symbolic log-density."""
        if self.distribution is not None:
            return self.distribution.logpdf(self.target.symbol)
        if self.log_density is None:  # guarded by __post_init__
            raise RuntimeError("factor has neither a distribution nor a log density")
        return self.log_density

    @property
    def free_symbols(self) -> set[sp.Symbol]:
        return self.expression.free_symbols

    @property
    def parent_symbols(self) -> set[sp.Symbol]:
        return self.free_symbols - {self.target.symbol}
