"""Exact-integration backend selection and fallback routing."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import sympy as sp

from ..core import Variable
from .backends import (
    ExactBackendFailure,
    ExactBackendUnavailableError,
    ExactIntegrationBackend,
    ExactIntegrationTrace,
    MultipleIntegrateBackend,
    SymPyExactIntegrationBackend,
)

_NOT_AVAILABLE = object()


def _try_backend(backend, expr, variables, assumptions):
    """Run a fallback-capable backend and return a sentinel when it declines."""
    try:
        return backend.integrate(expr, variables, assumptions=assumptions)
    except (ExactBackendUnavailableError, ExactBackendFailure):
        return _NOT_AVAILABLE


class AutoExactIntegrationBackend:
    """Try structured multiple integration first, then universal SymPy fallback."""

    name = "auto"

    def __init__(
        self,
        structured: ExactIntegrationBackend | None = None,
        sympy: ExactIntegrationBackend | None = None,
    ) -> None:
        self.structured = structured or MultipleIntegrateBackend()
        self.sympy = sympy or SymPyExactIntegrationBackend()
        self.last_trace: ExactIntegrationTrace | None = None

    def integrate(
        self,
        expr: sp.Expr,
        variables: Iterable[Variable],
        *,
        assumptions: Any | None = None,
    ) -> sp.Expr:
        """Return the first exact result from structured or SymPy integration."""
        variables = tuple(variables)
        attempts: list[str] = []
        if variables and all(
            variable.support == sp.S.Reals or isinstance(variable.support, sp.Interval)
            for variable in variables
        ):
            attempts.append(self.structured.name)
            value = _try_backend(self.structured, expr, variables, assumptions)
            if value is not _NOT_AVAILABLE:
                self.last_trace = ExactIntegrationTrace(
                    self.structured.name,
                    tuple(variable.name for variable in variables),
                    tuple(attempts),
                )
                return value
        attempts.append(self.sympy.name)
        value = self.sympy.integrate(expr, variables, assumptions=assumptions)
        self.last_trace = ExactIntegrationTrace(
            self.sympy.name,
            tuple(variable.name for variable in variables),
            tuple(attempts),
        )
        return value


def get_exact_integration_backend(
    backend: str | ExactIntegrationBackend = "auto",
) -> ExactIntegrationBackend:
    """Resolve an exact-integration backend by name or return a supplied backend.

    Supported names are ``"auto"``, ``"sympy"``, and
    ``"multiple-integrate"``. ``"auto"`` prefers structured multiple integration for purely
    continuous interval problems and falls back to SymPy otherwise.
    """
    if not isinstance(backend, str):
        if not isinstance(backend, ExactIntegrationBackend):
            raise TypeError(
                "backend must be a backend name or ExactIntegrationBackend instance"
            )
        return backend
    normalized = backend.strip().lower()
    if normalized == "auto":
        return AutoExactIntegrationBackend()
    if normalized == "sympy":
        return SymPyExactIntegrationBackend()
    if normalized == "multiple-integrate":
        return MultipleIntegrateBackend()
    raise ValueError(f"Unknown exact-integration backend: {backend!r}.")
