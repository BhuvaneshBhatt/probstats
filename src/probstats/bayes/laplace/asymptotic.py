"""Concrete integration with the optional :mod:`asymptotic` package.

The integration uses ``asymptotic.laplace_asymptotic_integral`` on an auxiliary
concentration family ``exp(t * log_density)`` as ``t -> +oo``.  Evaluating the
finite expansion at ``t = 1`` produces a formal higher-order refinement of the
ordinary Laplace approximation.  For one-dimensional or separable degenerate
modes, the same route also supplies a singular Laplace approximation instead of
forcing an invalid Gaussian Hessian.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np
import sympy as sp


class AsymptoticIntegrationError(RuntimeError):
    """Raised when the optional asymptotic Laplace engine cannot refine a model."""


@dataclass(frozen=True, slots=True)
class AsymptoticLaplaceAnalysis:
    """Structured result returned by :class:`AsymptoticCorrector`.

    ``expression`` is the finite asymptotic approximation to the evidence as a
    function of the artificial concentration parameter.  ``log_evidence`` is
    that expression evaluated at ``evaluation_parameter`` and converted to log
    scale.  ``local_orders`` records the first nonzero local derivative order in
    each separable coordinate; an order greater than two identifies a singular
    (degenerate) mode.
    """

    expression: sp.Expr
    log_evidence: sp.Expr
    correction: sp.Expr | None
    parameter: sp.Symbol
    evaluation_parameter: sp.Expr
    status: str
    certified: bool
    local_orders: tuple[int, ...]
    local_coefficients: tuple[sp.Expr, ...]
    raw_results: tuple[Any, ...]
    remainder: Any = None
    certificate: Any = None

    @property
    def singular(self) -> bool:
        """Return whether at least one local coordinate has degenerate curvature."""
        return any(order > 2 for order in self.local_orders)


def _import_asymptotic(loader: Callable[[], Any] | None = None) -> Any:
    try:
        module = import_module("asymptotic") if loader is None else loader()
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise AsymptoticIntegrationError(
            "The asymptotic package is not installed; install probstats[asymptotic]."
        ) from exc
    if not hasattr(module, "laplace_asymptotic_integral"):
        raise AsymptoticIntegrationError(
            "Installed asymptotic does not expose laplace_asymptotic_integral()."
        )
    return module


def _as_interval(support: sp.Set) -> sp.Interval:
    if support is sp.S.Reals:
        return sp.Interval(-sp.oo, sp.oo)
    if isinstance(support, sp.Interval):
        return support
    raise AsymptoticIntegrationError(
        f"Laplace asymptotics require continuous real interval support, got {support}."
    )


def _point_substitutions(
    variables: Sequence[sp.Symbol], point: Sequence[float]
) -> dict[sp.Symbol, sp.Expr]:
    if len(variables) != len(point):
        raise ValueError("point dimension does not match variables")
    return {var: sp.Float(value) for var, value in zip(variables, point, strict=True)}


def _local_order_and_coefficient(
    log_density: sp.Expr,
    variable: sp.Symbol,
    substitutions: dict[sp.Symbol, sp.Expr],
    *,
    max_order: int = 12,
) -> tuple[int, sp.Expr]:
    """Find the first nonzero local decay derivative at the mode.

    For ``ell(x) = ell(x0) - a (x-x0)^m + ...`` this returns ``(m, a)``.
    A usable real local maximum requires an even ``m`` and positive ``a``.
    """
    for order in range(2, max_order + 1):
        derivative = sp.simplify(
            sp.diff(log_density, variable, order).subs(substitutions)
        )
        if derivative == 0 or derivative.is_zero is True:
            continue
        try:
            numeric = float(sp.N(derivative))
        except (TypeError, ValueError):
            numeric = None
        if derivative.is_zero is None and numeric is not None and abs(numeric) < 1e-12:
            continue
        if order % 2:
            raise AsymptoticIntegrationError(
                f"First nonzero local derivative for {variable} has odd order {order}; "
                "the point is not an even-order interior maximum."
            )
        coefficient = sp.simplify(-derivative / sp.factorial(order))
        positive = coefficient.is_positive
        if positive is not True:
            try:
                if float(sp.N(coefficient)) <= 0:
                    raise AsymptoticIntegrationError(
                        f"Local order-{order} coefficient for {variable} is not positive."
                    )
            except (TypeError, ValueError):
                if positive is not True:
                    raise AsymptoticIntegrationError(
                        f"Could not certify positive local decay for {variable}."
                    )
        return order, coefficient
    raise AsymptoticIntegrationError(
        f"No nonzero even local derivative through order {max_order} for {variable}."
    )


def _separable_slices(
    log_density: sp.Expr,
    variables: tuple[sp.Symbol, ...],
    point: tuple[float, ...],
) -> tuple[sp.Expr, tuple[sp.Expr, ...], sp.Expr]:
    subs = _point_substitutions(variables, point)
    mode_value = sp.simplify(log_density.subs(subs))
    slices: list[sp.Expr] = []
    for variable in variables:
        other = {symbol: value for symbol, value in subs.items() if symbol != variable}
        slices.append(sp.simplify(log_density.subs(other) - mode_value))
    reconstructed = sp.simplify(mode_value + sp.Add(*slices))
    difference = sp.simplify(sp.expand(log_density - reconstructed))
    if difference != 0:
        raise AsymptoticIntegrationError(
            "The current asymptotic Laplace backend supports one-dimensional or "
            "coordinate-separable integrals; mixed multivariate higher-order terms "
            "require a future multivariate Laplace integral API in asymptotic."
        )
    return mode_value, tuple(slices), reconstructed


class AsymptoticCorrector:
    """Higher-order and singular Laplace integration via :mod:`asymptotic`.

    Parameters
    ----------
    terms:
        Number of asymptotic terms requested from
        ``asymptotic.laplace_asymptotic_integral``. If omitted, ``order`` passed
        by :func:`probstats.bayes.infer_laplace` is converted to a conservative
        term count.
    certify:
        Request the optional asymptotic package's global real-Laplace
        certification when supported.
    parameter:
        Symbol used for the auxiliary concentration parameter. A fresh positive
        symbol is created by default.
    evaluation_parameter:
        Value at which the asymptotic evidence expansion is evaluated. ``1``
        gives the conventional formal refinement of the target integral.
    correction_function:
        Optional correction callable used instead of the asymptotic analysis path.
    loader:
        Optional dependency loader. Supplying one is useful for controlled
        environments and backend integration tests.
    """

    name = "asymptotic"

    def __init__(
        self,
        correction_function: Callable[..., Any] | None = None,
        *,
        terms: int | None = None,
        certify: bool = True,
        parameter: sp.Symbol | None = None,
        evaluation_parameter: Any = 1,
        loader: Callable[[], Any] | None = None,
    ) -> None:
        if terms is not None and terms < 1:
            raise ValueError("terms must be positive")
        self._correction_function = correction_function
        self.terms = terms
        self.certify = bool(certify)
        self.parameter = parameter
        self.evaluation_parameter = sp.sympify(evaluation_parameter)
        self._loader = loader

    def analyze(
        self,
        *,
        log_density: sp.Expr,
        variables: tuple[sp.Symbol, ...],
        point: tuple[float, ...],
        supports: Sequence[sp.Set] | None = None,
        precision: np.ndarray | None = None,
        order: int = 4,
        base_log_evidence: sp.Expr | None = None,
    ) -> AsymptoticLaplaceAnalysis:
        """Run the concrete asymptotic Laplace expansion.

        The current ``asymptotic`` public API exposes a one-dimensional Laplace
        integral engine. Multivariate models are therefore accepted only when
        the log density separates additively by coordinate at the MAP; the
        evidence then factorizes into independent one-dimensional Laplace
        integrals. This includes mixed regular/singular separable modes.
        """
        del precision  # reserved for multivariate integration
        module = _import_asymptotic(self._loader)
        variables = tuple(variables)
        supports = tuple(supports or (sp.S.Reals,) * len(variables))
        if len(supports) != len(variables):
            raise ValueError("supports must match variables")
        if not variables:
            raise ValueError("at least one variable is required")

        lam = self.parameter or sp.Dummy("laplace_scale", positive=True)
        if lam in sp.sympify(log_density).free_symbols:
            raise AsymptoticIntegrationError(
                "The auxiliary asymptotic parameter occurs in the log density; "
                "pass a distinct AsymptoticCorrector(parameter=...)."
            )

        mode_value, slices, _ = _separable_slices(
            sp.sympify(log_density), variables, point
        )
        subs = _point_substitutions(variables, point)
        local = tuple(
            _local_order_and_coefficient(log_density, variable, subs)
            for variable in variables
        )
        local_orders = tuple(item[0] for item in local)
        local_coefficients = tuple(item[1] for item in local)

        requested_terms = self.terms or max(2, order // 2 + 1)
        raw_results: list[Any] = []
        factor_expressions: list[sp.Expr] = []
        statuses: list[str] = []
        certified = True
        remainders: list[Any] = []
        certificates: list[Any] = []
        laplace = module.laplace_asymptotic_integral

        for variable, support, local_slice in zip(
            variables, supports, slices, strict=True
        ):
            domain = _as_interval(support)
            integrand = sp.exp(lam * local_slice)
            try:
                result = laplace(
                    integrand,
                    variable,
                    domain,
                    parameter=lam,
                    point=sp.oo,
                    terms=requested_terms,
                    certify=self.certify,
                )
            except (ArithmeticError, RuntimeError, TypeError, ValueError) as exc:
                raise AsymptoticIntegrationError(
                    f"asymptotic Laplace expansion failed for {variable}: {exc}"
                ) from exc
            expression = getattr(result, "expression", None)
            if expression is None:
                raise AsymptoticIntegrationError(
                    "asymptotic Laplace result does not expose an expression."
                )
            raw_results.append(result)
            factor_expressions.append(sp.sympify(expression))
            status = str(getattr(result, "status", "FORMAL"))
            statuses.append(status)
            certified = certified and bool(getattr(result, "certified", False))
            remainder = getattr(result, "remainder", None)
            certificate = getattr(result, "certificate", None)
            if remainder is not None:
                remainders.append(remainder)
            if certificate is not None:
                certificates.append(certificate)

        evidence_expression = sp.simplify(
            sp.exp(lam * mode_value) * sp.Mul(*factor_expressions)
        )
        evaluated = sp.simplify(
            evidence_expression.subs(lam, self.evaluation_parameter)
        )
        if evaluated.is_positive is False or evaluated == 0:
            raise AsymptoticIntegrationError(
                "Higher-order evidence expansion is nonpositive at the requested evaluation point."
            )
        log_evidence = sp.simplify(sp.log(evaluated))
        correction = (
            None
            if base_log_evidence is None
            else sp.simplify(log_evidence - base_log_evidence)
        )
        overall_status = (
            "EXACT"
            if statuses and all(s == "EXACT" for s in statuses)
            else "CERTIFIED"
            if certified
            else "FORMAL"
        )
        return AsymptoticLaplaceAnalysis(
            expression=evidence_expression,
            log_evidence=log_evidence,
            correction=correction,
            parameter=lam,
            evaluation_parameter=self.evaluation_parameter,
            status=overall_status,
            certified=certified,
            local_orders=local_orders,
            local_coefficients=local_coefficients,
            raw_results=tuple(raw_results),
            remainder=tuple(remainders) if remainders else None,
            certificate=tuple(certificates) if certificates else None,
        )

    def correction(
        self,
        *,
        log_density: sp.Expr,
        variables: tuple[sp.Symbol, ...],
        point: tuple[float, ...],
        precision: np.ndarray,
        order: int,
        supports: Sequence[sp.Set] | None = None,
        base_log_evidence: sp.Expr | None = None,
    ) -> Any:
        """Return an additive higher-order log-evidence correction.

        A supplied ``correction_function`` preserves the callback adapter
        behavior. Otherwise this runs :meth:`analyze` and subtracts the supplied
        second-order log evidence.
        """
        if self._correction_function is not None:
            return self._correction_function(
                log_density=log_density,
                variables=variables,
                point=point,
                precision=precision,
                order=order,
            )
        if base_log_evidence is None:
            raise AsymptoticIntegrationError(
                "base_log_evidence is required for a concrete additive correction."
            )
        analysis = self.analyze(
            log_density=log_density,
            variables=variables,
            point=point,
            supports=supports,
            precision=precision,
            order=order,
            base_log_evidence=base_log_evidence,
        )
        return analysis.correction
