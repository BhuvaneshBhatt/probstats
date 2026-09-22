"""Certified symbolic predicates used at mathematical decision boundaries."""

from __future__ import annotations

from enum import Enum
from typing import Any

import sympy as sp
from exprtest import zerotest


class TruthValue(Enum):
    """Three-valued result of a certified symbolic proposition."""

    TRUE = 1
    FALSE = 0
    UNKNOWN = -1

    def __bool__(self) -> bool:
        raise TypeError(
            "TruthValue has no implicit Boolean interpretation; compare explicitly"
        )


def _truth(value: bool | None) -> TruthValue:
    if value is True:
        return TruthValue.TRUE
    if value is False:
        return TruthValue.FALSE
    return TruthValue.UNKNOWN


def certified_zero(value: Any) -> TruthValue:
    """Classify whether ``value`` is provably zero, nonzero, or unresolved."""
    return _truth(zerotest(sp.sympify(value), confidence="certified"))


def certified_equal(left: Any, right: Any) -> TruthValue:
    """Classify whether two symbolic values are provably equal or unequal."""
    return certified_zero(sp.sympify(left) - sp.sympify(right))
