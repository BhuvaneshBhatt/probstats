"""Pluggable exact-integration backends for symbolic Bayesian inference."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import sympy as sp

from ..._integration import (
    StructuredIntegrationFailure,
    StructuredIntegrationUnavailable,
    load_multiple_integrator,
    structured_integrate,
)
from ..core.variable import Variable


class ExactBackendUnavailableError(ImportError):
    """Raised when a requested optional exact-integration backend is unavailable."""


class ExactBackendFailure(RuntimeError):
    """Raised when an exact backend cannot evaluate a requested integral."""


@dataclass(frozen=True, slots=True)
class ExactIntegrationTrace:
    """Provenance for one successful exact elimination operation.

    Parameters
    ----------
    backend:
        Name of the backend that produced the returned value.
    variables:
        Variable names eliminated by the backend call.
    attempts:
        Ordered backend names attempted before success.
    metadata:
        Backend-specific non-semantic details useful for diagnostics.
    """

    backend: str
    variables: tuple[str, ...]
    attempts: tuple[str, ...] = ()
    metadata: dict[str, Any] | None = None


@runtime_checkable
class ExactIntegrationBackend(Protocol):
    """Protocol implemented by exact continuous-integration backends."""

    name: str

    def integrate(
        self,
        expr: sp.Expr,
        variables: Iterable[Variable],
        *,
        assumptions: Any | None = None,
    ) -> sp.Expr:
        """Eliminate ``variables`` exactly from ``expr`` or raise ``ExactBackendFailure``."""


class SymPyExactIntegrationBackend:
    """Universal exact backend using SymPy integration and summation sequentially."""

    name = "sympy"

    @staticmethod
    def _integrate_one(expr: sp.Expr, variable: Variable) -> sp.Expr:
        symbol = variable.symbol
        domain = variable.support
        expr = sp.simplify(expr)
        if domain == sp.S.Reals:
            value = sp.integrate(expr, (symbol, -sp.oo, sp.oo))
        elif isinstance(domain, sp.Interval):
            value = sp.integrate(expr, (symbol, domain.start, domain.end))
        elif domain == sp.S.Naturals0:
            value = sp.summation(expr, (symbol, 0, sp.oo))
        elif domain == sp.S.Naturals:
            value = sp.summation(expr, (symbol, 1, sp.oo))
        elif domain == sp.S.Integers:
            value = sp.summation(expr, (symbol, -sp.oo, sp.oo))
        elif isinstance(domain, sp.FiniteSet):
            value = sp.Add(*(expr.subs(symbol, point) for point in domain))
        else:
            raise ExactBackendFailure(
                f"SymPy exact backend does not support domain {domain} for {symbol}."
            )
        if value.has(sp.Integral, sp.Sum):
            raise ExactBackendFailure(
                f"Exact integration remained unevaluated for {symbol}."
            )
        return sp.simplify(value)

    def integrate(
        self,
        expr: sp.Expr,
        variables: Iterable[Variable],
        *,
        assumptions: Any | None = None,
    ) -> sp.Expr:
        """Integrate/sum in declared variable order using native SymPy operations."""
        result = sp.sympify(expr)
        for variable in variables:
            result = self._integrate_one(result, variable)
        return sp.simplify(result)


class MultipleIntegrateBackend:
    """Structured exact backend powered by :mod:`multiple_integrate`.

    The backend batches continuous real/interval variables into one structured
    call, allowing ``multiple-integrate`` to recognize multivariate Gaussian,
    beta/gamma, simplex/Dirichlet, moment, and related exact families.  Discrete
    supports are rejected so the automatic composite backend can
    fall back to SymPy's exact summation machinery.
    """

    name = "multiple-integrate"

    def __init__(self, integrator: Any | None = None, *, loader=None) -> None:
        self._integrator = integrator
        self._loader = load_multiple_integrator if loader is None else loader

    def _load(self):
        if self._integrator is None:
            try:
                self._integrator = self._loader()
            except StructuredIntegrationUnavailable as exc:
                raise ExactBackendUnavailableError(str(exc)) from exc
        return self._integrator

    @staticmethod
    def _range(variable: Variable) -> tuple[sp.Symbol, sp.Expr, sp.Expr]:
        domain = variable.support
        if domain == sp.S.Reals:
            return (variable.symbol, -sp.oo, sp.oo)
        if isinstance(domain, sp.Interval):
            # Measure-zero endpoint openness does not alter ordinary continuous
            # Lebesgue/Riemann density integrals; the numerical bounds therefore
            # match multiple_integrate's range representation.
            return (variable.symbol, domain.start, domain.end)
        raise ExactBackendFailure(
            f"multiple-integrate requires continuous interval support; "
            f"{variable.name} has {domain}."
        )

    def integrate(
        self,
        expr: sp.Expr,
        variables: Iterable[Variable],
        *,
        assumptions: Any | None = None,
    ) -> sp.Expr:
        """Evaluate a structured continuous multiple integral exactly.

        Ranges are passed in the same inner-first order as ``sympy.integrate``
        and ``multiple_integrate``.  Parameter assumptions are forwarded
        unchanged to the backend.
        """
        variables = tuple(variables)
        if not variables:
            return sp.sympify(expr)
        ranges = tuple(self._range(variable) for variable in variables)
        try:
            return structured_integrate(
                expr,
                ranges,
                assumptions=assumptions,
                integrator=self._load(),
            )
        except StructuredIntegrationUnavailable as exc:
            raise ExactBackendUnavailableError(str(exc)) from exc
        except StructuredIntegrationFailure as exc:
            raise ExactBackendFailure(str(exc)) from exc
