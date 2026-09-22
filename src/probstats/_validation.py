"""Small shared validators for public integer and sample-shape arguments."""

from __future__ import annotations

from operator import index

import numpy as np


def integer(
    value,
    *,
    name: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Return an exact integer argument without truncating numeric values."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        result = index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    result = int(result)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return result


def sample_shape(size, *, none_as_empty: bool = True) -> tuple[int, ...] | None:
    """Normalize a NumPy-style sample shape without accepting truncating casts."""
    if size is None:
        return () if none_as_empty else None
    try:
        scalar = integer(size, name="size", minimum=0)
    except TypeError:
        pass
    else:
        return (scalar,)
    try:
        values = tuple(size)
    except TypeError as exc:
        raise TypeError("size must be an integer or an iterable of integers") from exc
    return tuple(integer(value, name="size entry", minimum=0) for value in values)
