"""Small conversion helpers defining the NumPy/SymPy boundary used by the package."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import sympy as sp


def as_symbol(value: str | sp.Symbol, *, real: bool | None = None) -> sp.Symbol:
    """Return *value* as a SymPy symbol.

    Existing symbols are preserved exactly. Strings are converted to symbols and may
    optionally receive a ``real`` assumption.
    """
    if isinstance(value, sp.Symbol):
        return value
    if not isinstance(value, str) or not value:
        raise TypeError("A variable name must be a non-empty string or a SymPy Symbol.")
    assumptions: dict[str, bool] = {}
    if real is not None:
        assumptions["real"] = real
    return sp.Symbol(value, **assumptions)


def sympify(value: Any) -> sp.Basic:
    """Convert a scalar/expression into the package's symbolic representation."""
    return sp.sympify(value)


def as_float_array(value: Any, *, ndim: int | None = None) -> np.ndarray:
    """Convert numerical input to a finite ``float64`` NumPy array.

    Object arrays and silently propagated infinities are rejected at the
    model/backend boundary because they otherwise create difficult-to-diagnose failures
    in inference engines.
    """
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"Expected an array with ndim={ndim}, got ndim={array.ndim}.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Numerical arrays must contain only finite values.")
    return array


def numeric_vector(value: Iterable[Any]) -> np.ndarray:
    """Return a finite one-dimensional float vector."""
    return as_float_array(value, ndim=1)


def numeric_matrix(value: Iterable[Iterable[Any]]) -> np.ndarray:
    """Return a finite two-dimensional float matrix."""
    return as_float_array(value, ndim=2)
