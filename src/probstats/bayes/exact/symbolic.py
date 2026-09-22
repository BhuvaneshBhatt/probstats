"""Exact symbolic integration and normalized posterior representations."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import sympy as sp

from ..certification import ExpressionCertificate, certify_zero
from ..core.variable import Variable
from ..reasoning import PropertyCertificate, certify_nonpositive, certify_positive
from .backends import (
    ExactBackendFailure,
    ExactIntegrationBackend,
    ExactIntegrationTrace,
)
from .routing import AutoExactIntegrationBackend, get_exact_integration_backend


class ExactIntegrationError(RuntimeError):
    """Raised when no configured exact backend can certify a marginal/normalizer."""


def _integrate_over_with_trace(
    expr: Any,
    variables: Iterable[Variable],
    *,
    backend: str | ExactIntegrationBackend = "auto",
    assumptions: Any | None = None,
) -> tuple[sp.Expr, ExactIntegrationTrace]:
    variables = tuple(variables)
    resolved = get_exact_integration_backend(backend)
    try:
        value = resolved.integrate(sp.sympify(expr), variables, assumptions=assumptions)
    except ExactBackendFailure as exc:
        raise ExactIntegrationError(str(exc)) from exc
    if (
        isinstance(resolved, AutoExactIntegrationBackend)
        and resolved.last_trace is not None
    ):
        trace = resolved.last_trace
    else:
        trace = ExactIntegrationTrace(
            resolved.name,
            tuple(variable.name for variable in variables),
            (resolved.name,),
        )
    return sp.simplify(value), trace


def integrate_over(
    expr: Any,
    variables: Iterable[Variable],
    *,
    backend: str | ExactIntegrationBackend = "auto",
    assumptions: Any | None = None,
) -> sp.Expr:
    """Exactly eliminate variables using a selectable integration backend.

    ``backend="auto"`` first gives purely continuous real/interval problems to
    :mod:`multiple_integrate` when installed, then falls back to native SymPy.
    Discrete variables are always handled by SymPy's exact summation path.
    """
    value, _ = _integrate_over_with_trace(
        expr, variables, backend=backend, assumptions=assumptions
    )
    return value


@dataclass(frozen=True, slots=True)
class NormalizerValidation:
    """Unified exact-posterior normalizer validation result."""

    zero_certificate: ExpressionCertificate
    positive_certificate: PropertyCertificate
    nonpositive_certificate: PropertyCertificate

    @property
    def certified_nonzero(self) -> bool:
        return self.zero_certificate.proven and self.zero_certificate.is_zero is False

    @property
    def certified_positive(self) -> bool:
        return self.positive_certificate.value is True


def _validate_normalizer(
    value: sp.Expr,
    *,
    assumptions: Any | None = None,
    exprtest_loader: Callable[[], Any] | None = None,
    semialg_loader: Callable[[], Any] | None = None,
) -> NormalizerValidation:
    """Apply one validity contract to every exact posterior normalizer."""
    if value in (sp.oo, -sp.oo, sp.zoo, sp.nan) or value.has(sp.oo, sp.zoo, sp.nan):
        raise ExactIntegrationError(f"Normalizer is not finite: {value}.")
    zero = certify_zero(value, exprtest_loader=exprtest_loader)
    if zero.is_zero is True:
        raise ExactIntegrationError(
            f"Normalizer is zero ({zero.backend}: {zero.reason})."
        )
    positive = certify_positive(
        value, assumptions=assumptions, semialg_loader=semialg_loader
    )
    nonpositive = certify_nonpositive(
        value, assumptions=assumptions, semialg_loader=semialg_loader
    )
    if nonpositive.value is True:
        raise ExactIntegrationError(
            "Posterior normalizer is certified nonpositive under the supplied assumptions."
        )
    if value.is_negative is True:
        raise ExactIntegrationError(f"Normalizer is negative: {value}.")
    return NormalizerValidation(zero, positive, nonpositive)


@dataclass(frozen=True, slots=True)
class SymbolicJointDistribution:
    """A normalized symbolic density/mass over one or more variables."""

    variables: tuple[Variable, ...]
    density: sp.Expr
    normalizing_constant: sp.Expr = sp.S.One

    def __init__(
        self, variables: Iterable[Variable], density: Any, normalizing_constant: Any = 1
    ):
        variables = tuple(variables)
        if not variables:
            raise ValueError(
                "A symbolic joint distribution needs at least one variable."
            )
        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "density", sp.simplify(sp.sympify(density)))
        object.__setattr__(
            self, "normalizing_constant", sp.simplify(sp.sympify(normalizing_constant))
        )

    @property
    def symbols(self) -> tuple[sp.Symbol, ...]:
        """Return the SymPy symbols corresponding to the distribution variables."""
        return tuple(variable.symbol for variable in self.variables)

    @property
    def logpdf(self) -> sp.Expr:
        """Return the symbolic logarithm of the normalized density/mass."""
        return sp.simplify(sp.log(self.density))

    def marginalize(
        self,
        *variables: str | Variable,
        backend: str | ExactIntegrationBackend = "auto",
        assumptions: Any | None = None,
    ) -> SymbolicJointDistribution:
        """Integrate/sum named variables out of this symbolic distribution."""
        names = {v.name if isinstance(v, Variable) else v for v in variables}
        if not names:
            return self
        known = {v.name for v in self.variables}
        unknown = names - known
        if unknown:
            raise KeyError(f"Unknown variables to marginalize: {sorted(unknown)}.")
        eliminated = tuple(v for v in self.variables if v.name in names)
        retained = tuple(v for v in self.variables if v.name not in names)
        if not retained:
            raise ValueError(
                "Marginalizing every variable yields a scalar, not a distribution."
            )
        density = integrate_over(
            self.density, eliminated, backend=backend, assumptions=assumptions
        )
        return SymbolicJointDistribution(retained, density, self.normalizing_constant)

    def marginal(
        self,
        *variables: str | Variable,
        backend: str | ExactIntegrationBackend = "auto",
        assumptions: Any | None = None,
    ) -> SymbolicJointDistribution:
        """Retain only named variables by exactly eliminating all others."""
        names = {v.name if isinstance(v, Variable) else v for v in variables}
        known = {v.name for v in self.variables}
        unknown = names - known
        if unknown:
            raise KeyError(f"Unknown marginal variables: {sorted(unknown)}.")
        eliminate = tuple(v for v in self.variables if v.name not in names)
        if not names:
            raise ValueError("Select at least one marginal variable.")
        return (
            self.marginalize(*eliminate, backend=backend, assumptions=assumptions)
            if eliminate
            else self
        )
