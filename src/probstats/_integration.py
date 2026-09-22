"""Shared exact-integration helpers for rectangular continuous domains."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from importlib import import_module
from typing import Any

import sympy as sp

IntegrationRange = tuple[sp.Symbol, sp.Expr, sp.Expr]


class StructuredIntegrationUnavailable(ImportError):
    """Raised when the optional structured integration backend is unavailable."""


class StructuredIntegrationFailure(RuntimeError):
    """Raised when structured integration cannot evaluate an integral."""


def load_multiple_integrator() -> Callable[..., Any]:
    try:
        module = import_module("multiple_integrate")
    except ImportError as exc:
        raise StructuredIntegrationUnavailable(
            "multiple-integrate is not installed; install probstats[exact]."
        ) from exc
    return module.multiple_integrate


def structured_integrate(
    expr: sp.Expr,
    ranges: Iterable[IntegrationRange],
    *,
    assumptions: Any | None = None,
    integrator: Callable[..., Any] | None = None,
) -> sp.Expr:
    """Evaluate one rectangular continuous integral with ``multiple-integrate``.

    The caller supplies explicit ``(symbol, lower, upper)`` ranges.  This keeps
    the integration layer independent of probability and Bayesian variable
    classes while preserving one backend implementation for both.
    """
    ranges = tuple(ranges)
    if not ranges:
        return sp.sympify(expr)
    integrate = load_multiple_integrator() if integrator is None else integrator
    kwargs = {} if assumptions is None else {"assumptions": assumptions}
    try:
        value = integrate(sp.simplify(sp.sympify(expr)), *ranges, **kwargs)
    except (TypeError, ValueError, NotImplementedError, RuntimeError) as exc:
        raise StructuredIntegrationFailure(f"multiple-integrate failed: {exc}") from exc
    value = sp.sympify(value)
    if value.has(sp.Integral, sp.Sum):
        raise StructuredIntegrationFailure(
            "multiple-integrate returned an unevaluated integral/sum."
        )
    return sp.simplify(value)


def sympy_integrate_rectangular(
    expr: sp.Expr, ranges: Iterable[IntegrationRange]
) -> sp.Expr:
    """Integrate sequentially over rectangular continuous ranges with SymPy."""
    value = sp.sympify(expr)
    for symbol, lower, upper in ranges:
        value = sp.integrate(value, (symbol, lower, upper))
    return sp.simplify(value)


_STRUCTURED_UNAVAILABLE = object()


def _try_structured(expr, ranges, assumptions):
    """Return a structured integral or a sentinel when the backend cannot handle it."""
    try:
        return structured_integrate(expr, ranges, assumptions=assumptions)
    except (StructuredIntegrationUnavailable, StructuredIntegrationFailure):
        return _STRUCTURED_UNAVAILABLE


def integrate_rectangular(
    expr: sp.Expr,
    ranges: Iterable[IntegrationRange],
    *,
    assumptions: Any | None = None,
    structured_min_dim: int = 2,
) -> sp.Expr:
    """Integrate over a continuous product domain with structured fallback.

    ``multiple-integrate`` is attempted for genuinely multivariate integrals.
    Missing optional dependencies or unsupported structured forms fall back to
    sequential SymPy integration.  Explicit backend selection remains available
    through :mod:`probstats.bayes.exact`.
    """
    ranges = tuple(ranges)
    if len(ranges) >= structured_min_dim:
        value = _try_structured(expr, ranges, assumptions)
        if value is not _STRUCTURED_UNAVAILABLE:
            return value
    return sympy_integrate_rectangular(expr, ranges)
