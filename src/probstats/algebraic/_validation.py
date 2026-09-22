"""Shared normalization for algebraic-statistics public inputs."""

from __future__ import annotations

from collections.abc import Iterable

import sympy as sp


def exact_integer(value, *, name: str, minimum: int | None = None) -> int:
    """Return an exact integer without silently truncating numeric inputs."""
    item = sp.sympify(value)
    if item.is_Integer is not True:
        raise ValueError(f"{name} must be an integer")
    result = int(item)
    if minimum is not None and result < minimum:
        qualifier = "positive" if minimum == 1 else f">= {minimum}"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def exact_integer_tuple(
    values: Iterable[object],
    *,
    name: str,
    minimum: int | None = None,
) -> tuple[int, ...]:
    """Normalize an iterable of exact integers under one public contract."""
    return tuple(
        exact_integer(value, name=f"{name} value", minimum=minimum) for value in values
    )


def cardinalities(values, *, minimum_variables: int = 1) -> tuple[int, ...]:
    """Normalize finite-state cardinalities without lossy integer coercion."""
    dims = exact_integer_tuple(values, name="cardinality", minimum=1)
    if len(dims) < minimum_variables:
        noun = "variable" if minimum_variables == 1 else "variables"
        raise ValueError(
            f"cardinalities must describe at least {minimum_variables} {noun}"
        )
    return dims


def count_vector(values, *, width: int, name: str = "counts") -> tuple[int, ...]:
    """Normalize a fixed-width nonnegative integer count vector."""
    raw = tuple(values)
    if len(raw) != width:
        raise ValueError(f"{name} length must be {width}")
    counts: list[int] = []
    for value in raw:
        item = sp.sympify(value)
        if item.is_Integer is not True or item.is_nonnegative is not True:
            raise ValueError(f"{name} must be nonnegative integers")
        counts.append(int(item))
    return tuple(counts)


def choice(value, *, name: str, allowed: tuple[str, ...]) -> str:
    """Return a validated public string choice."""
    if not isinstance(value, str) or value not in allowed:
        options = ", ".join(repr(item) for item in allowed)
        raise ValueError(f"{name} must be one of {options}")
    return value
