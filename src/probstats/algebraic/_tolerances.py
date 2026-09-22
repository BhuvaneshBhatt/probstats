"""Numerical tolerance policy for algebraic-statistics numerical backends."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AlgebraicTolerancePolicy:
    """Defaults for numerical algebraic-statistics operations.

    Exact symbolic decisions do not use these values. They apply only to
    numerical reconstruction, stochastic normalization, and iterative fitting.
    """

    decomposition: float = 1e-10
    stochastic: float = 1e-8
    fitting: float = 1e-10


DEFAULT_TOLERANCES = AlgebraicTolerancePolicy()
DECOMPOSITION_TOLERANCE = DEFAULT_TOLERANCES.decomposition
FITTING_TOLERANCE = DEFAULT_TOLERANCES.fitting
STOCHASTIC_TOLERANCE = DEFAULT_TOLERANCES.stochastic


def validate_tolerance(value, *, name: str = "tolerance") -> float:
    """Return a finite nonnegative numerical tolerance."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite nonnegative real number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite nonnegative real number") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite nonnegative real number")
    return result


__all__ = [
    "DECOMPOSITION_TOLERANCE",
    "DEFAULT_TOLERANCES",
    "FITTING_TOLERANCE",
    "STOCHASTIC_TOLERANCE",
    "AlgebraicTolerancePolicy",
    "validate_tolerance",
]
