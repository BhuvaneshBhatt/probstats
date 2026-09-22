"""Optional exact and structural reasoning for Bayesian inference.

This module integrates the optional :mod:`semialg` and :mod:`funcprops`
packages without making either a required dependency.  All decisions are
tri-state: an unsupported problem or failed proof remains ``UNKNOWN``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from importlib import import_module
from types import MappingProxyType
from typing import Any

import sympy as sp

from ..._exact_linear_algebra import exact_determinant


class ProofStatus(str, Enum):
    """Tri-state status for exact property claims."""

    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class PropertyCertificate:
    """A proof-oriented mathematical property decision.

    ``TRUE`` and ``FALSE`` mean that the selected backend supplied evidence for
    the property or its failure.  ``UNKNOWN`` is preserved when a
    backend is unavailable, inapplicable, or inconclusive.
    """

    property: str
    status: ProofStatus
    backend: str
    method: str = ""
    reason: str = ""
    evidence: Any | None = None

    @property
    def proven(self) -> bool:
        """Whether the property or its negation was certified."""
        return self.status is not ProofStatus.UNKNOWN

    @property
    def value(self) -> bool | None:
        """Return a Boolean for certified decisions and ``None`` otherwise."""
        if self.status is ProofStatus.TRUE:
            return True
        if self.status is ProofStatus.FALSE:
            return False
        return None


def _normalize_assumptions(assumptions: Any) -> Any:
    if assumptions is None:
        return True
    if isinstance(assumptions, (tuple, list, set, frozenset)):
        if not assumptions:
            return True
        return sp.And(*(sp.sympify(item) for item in assumptions))
    return sp.sympify(assumptions)


def certify_sign(
    expr: Any,
    relation: str,
    *,
    variables: Sequence[sp.Symbol] | None = None,
    assumptions: Any = True,
    use_semialg: bool = True,
    semialg_loader: Callable[[], Any] | None = None,
) -> PropertyCertificate:
    """Certify a sign relation using semialg, with exact SymPy fallback.

    ``relation`` is one of ``positive``, ``nonnegative``, ``negative``,
    ``nonpositive``, ``zero``, or ``nonzero``.  A failed semialg universal
    proof is classified ``FALSE`` only when the structured result contains a
    counterexample; otherwise it remains ``UNKNOWN``.
    """
    relation = str(relation).lower()
    valid = {"positive", "nonnegative", "negative", "nonpositive", "zero", "nonzero"}
    if relation not in valid:
        raise ValueError(f"unknown sign relation {relation!r}")
    value = sp.simplify(sp.sympify(expr))
    asm = _normalize_assumptions(assumptions)
    if use_semialg:
        try:
            semialg = (
                import_module("semialg") if semialg_loader is None else semialg_loader()
            )
        except ImportError:
            pass
        else:
            fn = getattr(semialg, f"prove_{relation}", None)
            if fn is not None:
                try:
                    result = fn(value, variables, assumptions=asm, return_result=True)
                except (
                    TypeError,
                    ValueError,
                    NotImplementedError,
                    AttributeError,
                    RuntimeError,
                    sp.PolynomialError,
                ) as exc:
                    semialg_reason = f"semialg could not decide the relation: {exc}"
                else:
                    if bool(getattr(result, "proven", False)):
                        return PropertyCertificate(
                            relation,
                            ProofStatus.TRUE,
                            "semialg",
                            str(getattr(result, "method", "sign-proof")),
                            f"semialg certified {relation}",
                            result,
                        )
                    if getattr(result, "counterexample", None) is not None:
                        return PropertyCertificate(
                            relation,
                            ProofStatus.FALSE,
                            "semialg",
                            str(getattr(result, "method", "counterexample")),
                            f"semialg produced a counterexample to {relation}",
                            result,
                        )
                    semialg_reason = f"semialg did not certify {relation}"
            else:
                semialg_reason = f"semialg has no prove_{relation} API"
    else:
        semialg_reason = "semialg disabled"

    # Exact dependency-free fallback.  This conservative fallback
    # never turns a floating approximation into a proof.
    pred = {
        "positive": value.is_positive,
        "nonnegative": value.is_nonnegative,
        "negative": value.is_negative,
        "nonpositive": value.is_nonpositive,
        "zero": value.is_zero,
        "nonzero": None if value.is_zero is None else not value.is_zero,
    }[relation]
    if pred is True:
        return PropertyCertificate(
            relation,
            ProofStatus.TRUE,
            "sympy",
            f"is_{relation}",
            f"SymPy certified {relation}",
        )
    if pred is False:
        return PropertyCertificate(
            relation,
            ProofStatus.FALSE,
            "sympy",
            f"is_{relation}",
            f"SymPy disproved {relation}",
        )
    return PropertyCertificate(
        relation, ProofStatus.UNKNOWN, "semialg/sympy", "", semialg_reason
    )


def certify_positive(expr: Any, **kwargs: Any) -> PropertyCertificate:
    """Certify that an expression is strictly positive."""
    return certify_sign(expr, "positive", **kwargs)


def certify_nonnegative(expr: Any, **kwargs: Any) -> PropertyCertificate:
    """Certify that an expression is nonnegative."""
    return certify_sign(expr, "nonnegative", **kwargs)


def certify_negative(expr: Any, **kwargs: Any) -> PropertyCertificate:
    """Certify that an expression is strictly negative."""
    return certify_sign(expr, "negative", **kwargs)


def certify_nonpositive(expr: Any, **kwargs: Any) -> PropertyCertificate:
    """Certify that an expression is nonpositive."""
    return certify_sign(expr, "nonpositive", **kwargs)


@dataclass(frozen=True, slots=True)
class MatrixDefinitenessCertificate:
    """Sylvester-style positive-definiteness certificate for a symmetric matrix."""

    status: ProofStatus
    leading_principal_minors: tuple[sp.Expr, ...]
    minor_certificates: tuple[PropertyCertificate, ...]
    backend: str = "semialg/sympy"

    @property
    def positive_definite(self) -> bool | None:
        """Return the certified positive-definiteness decision."""
        if self.status is ProofStatus.TRUE:
            return True
        if self.status is ProofStatus.FALSE:
            return False
        return None


def certify_positive_definite(
    matrix: Any,
    *,
    assumptions: Any = True,
    use_semialg: bool = True,
    semialg_loader: Callable[[], Any] | None = None,
) -> MatrixDefinitenessCertificate:
    """Certify positive definiteness via Sylvester's criterion.

    For a real symmetric matrix, positivity of every leading principal minor is
    necessary and sufficient for positive definiteness.  Each minor is passed
    through :func:`certify_positive`, so semialg can prove parameter-conditioned
    inequalities that SymPy's local assumptions may leave unresolved.
    """
    m = sp.Matrix(matrix)
    if m.rows != m.cols:
        raise ValueError("positive definiteness requires a square matrix")
    if m != m.T:
        return MatrixDefinitenessCertificate(ProofStatus.FALSE, (), (), "structural")
    minors = tuple(exact_determinant(m[:k, :k]) for k in range(1, m.rows + 1))
    certificates = tuple(
        certify_positive(
            minor,
            assumptions=assumptions,
            use_semialg=use_semialg,
            semialg_loader=semialg_loader,
        )
        for minor in minors
    )
    if all(cert.status is ProofStatus.TRUE for cert in certificates):
        status = ProofStatus.TRUE
    elif any(cert.status is ProofStatus.FALSE for cert in certificates):
        status = ProofStatus.FALSE
    else:
        status = ProofStatus.UNKNOWN
    return MatrixDefinitenessCertificate(status, minors, certificates)


@dataclass(frozen=True, slots=True)
class PosteriorGeometry:
    """Certified/high-level structural properties of a posterior log density."""

    convexity: str = "unknown"
    globally_concave: bool | None = None
    continuity: bool | None = None
    singularities: tuple[Any, ...] = ()
    source: str = "none"
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))

    @property
    def has_singularities(self) -> bool | None:
        """Whether funcprops found singularities on the queried real domain."""
        if self.singularities:
            return True
        if (
            self.source == "funcprops"
            and self.details.get("singularity_complete") is True
        ):
            return False
        return None

    @property
    def laplace_favorable(self) -> bool:
        """Whether global geometry supplies positive evidence for Laplace routing."""
        return (
            self.globally_concave is True
            and self.continuity is not False
            and self.has_singularities is not True
        )


def _singularity_condition(item: Any) -> sp.Basic | None:
    condition = getattr(item, "condition", None)
    guard = getattr(item, "guard", sp.true)
    if condition is not None:
        return sp.And(sp.sympify(condition), sp.sympify(guard))
    if isinstance(item, sp.Basic) and (
        getattr(item, "is_Relational", False) or item.func in (sp.And, sp.Or, sp.Not)
    ):
        return item
    return None


def _singularity_intersects_domain(
    item: Any,
    domain: Any,
    variable: sp.Symbol,
    *,
    semialg_loader: Callable[[], Any] | None = None,
) -> bool | None:
    condition = _singularity_condition(item)
    if condition is None or domain in (True, sp.true, sp.S.Reals):
        return True
    formula = sp.And(sp.sympify(domain), condition)
    try:
        reduced = sp.reduce_inequalities([sp.sympify(domain), condition], variable)
        if reduced is sp.false:
            return False
        if reduced is sp.true:
            return True
    except (TypeError, ValueError, NotImplementedError, sp.PolynomialError):
        pass
    try:
        semialg = (
            import_module("semialg") if semialg_loader is None else semialg_loader()
        )
        result = semialg.is_satisfiable(formula, (variable,), return_result=True)
        if hasattr(result, "__bool__"):
            return bool(result)
    except (
        ImportError,
        TypeError,
        ValueError,
        NotImplementedError,
        AttributeError,
        RuntimeError,
        sp.PolynomialError,
    ):
        pass
    try:
        simplified = sp.simplify_logic(formula)
    except (TypeError, ValueError, NotImplementedError):
        return None
    if simplified is sp.false:
        return False
    if simplified is sp.true:
        return True
    return None


def _enum_value(value: Any) -> str:
    primary = getattr(value, "primary", value)
    raw = getattr(primary, "value", primary)
    return str(raw).lower()


def analyze_posterior_geometry(
    log_density: Any,
    variables: Sequence[sp.Symbol],
    *,
    domain: Any = True,
    use_funcprops: bool = True,
    funcprops_loader: Callable[[], Any] | None = None,
    semialg_loader: Callable[[], Any] | None = None,
) -> PosteriorGeometry:
    """Analyze posterior geometry with funcprops when available.

    Multivariate convexity uses funcprops' public tuple-variable interface.
    Continuity and singularity analysis are currently queried only for the
    one-dimensional case, where the package exposes complete real-domain
    machinery.  Unsupported or unresolved properties remain unknown.
    """
    expr = sp.sympify(log_density)
    vv = tuple(sp.sympify(v) for v in variables)
    if not use_funcprops:
        return PosteriorGeometry(source="none")
    try:
        funcprops = (
            import_module("funcprops")
            if funcprops_loader is None
            else funcprops_loader()
        )
    except ImportError:
        return PosteriorGeometry(source="none", details={"available": False})
    if funcprops_loader is not None or semialg_loader is not None:
        return _analyze_posterior_geometry_uncached(
            expr, vv, domain, True, funcprops, semialg_loader
        )
    fingerprint = (
        id(getattr(funcprops, "function_convexity", None)),
        id(getattr(funcprops, "function_continuous", None)),
        id(getattr(funcprops, "function_singularities", None)),
    )
    try:
        hash((expr, vv, sp.sympify(domain), fingerprint))
    except (TypeError, ValueError):
        return _analyze_posterior_geometry_uncached(
            expr, vv, domain, True, funcprops, None
        )
    return _analyze_posterior_geometry_cached(expr, vv, sp.sympify(domain), fingerprint)


@lru_cache(maxsize=256)
def _analyze_posterior_geometry_cached(
    expr: sp.Expr,
    vv: tuple[sp.Symbol, ...],
    domain: Any,
    _backend_fingerprint: tuple[int, ...] = (),
) -> PosteriorGeometry:
    try:
        funcprops = import_module("funcprops")
    except ImportError:
        return PosteriorGeometry(source="none", details={"available": False})
    return _analyze_posterior_geometry_uncached(expr, vv, domain, True, funcprops, None)


def _analyze_posterior_geometry_uncached(
    expr: sp.Expr,
    vv: tuple[sp.Symbol, ...],
    domain: Any,
    use_funcprops: bool,
    funcprops: Any | None = None,
    semialg_loader: Callable[[], Any] | None = None,
) -> PosteriorGeometry:
    if not vv or not use_funcprops:
        return PosteriorGeometry(source="none")
    if funcprops is None:
        try:
            funcprops = import_module("funcprops")
        except ImportError:
            return PosteriorGeometry(source="none", details={"available": False})

    details: dict[str, Any] = {"available": True}
    convexity = "unknown"
    globally_concave: bool | None = None
    try:
        raw_convexity = funcprops.function_convexity(
            expr, vv[0] if len(vv) == 1 else vv, domain=domain, details=True
        )
        convexity = _enum_value(raw_convexity)
        details["convexity_raw"] = raw_convexity
        details["convexity_source"] = getattr(raw_convexity, "source", None)
        concave_values = {"concave", "strictly_concave", "strongly_concave", "affine"}
        convex_values = {"convex", "strictly_convex", "strongly_convex", "neither"}
        if convexity in concave_values:
            globally_concave = True
        elif convexity in convex_values:
            globally_concave = False
    except (
        TypeError,
        ValueError,
        NotImplementedError,
        AttributeError,
        RuntimeError,
        sp.PolynomialError,
    ) as exc:
        details["convexity_error"] = str(exc)

    continuity: bool | None = None
    singularities: tuple[Any, ...] = ()
    if len(vv) == 1:
        var = vv[0]
        try:
            raw_continuity = funcprops.function_continuous(expr, var, domain=domain)
            continuity_name = _enum_value(raw_continuity)
            details["continuity_raw"] = raw_continuity
            if continuity_name in {"continuous", "true"}:
                continuity = True
            elif continuity_name in {"discontinuous", "false"}:
                continuity = False
        except (
            TypeError,
            ValueError,
            NotImplementedError,
            AttributeError,
            RuntimeError,
            sp.PolynomialError,
        ) as exc:
            details["continuity_error"] = str(exc)
        try:
            raw_singularities = funcprops.function_singularities(
                expr, var, domain=sp.S.Reals, assumptions=domain
            )
            if raw_singularities is None:
                singularities = ()
            elif isinstance(raw_singularities, (tuple, list, set, frozenset)):
                singularities = tuple(raw_singularities)
            else:
                singularities = tuple(getattr(raw_singularities, "singularities", ()))
                if not singularities and getattr(raw_singularities, "primary", None):
                    primary = raw_singularities.primary
                    if isinstance(primary, (tuple, list, set, frozenset)):
                        singularities = tuple(primary)
            if singularities and domain not in (True, sp.true, sp.S.Reals):
                filtered = []
                unresolved_domain_filter = False
                for item in singularities:
                    intersects = _singularity_intersects_domain(
                        item, domain, var, semialg_loader=semialg_loader
                    )
                    if intersects is False:
                        continue
                    if intersects is None:
                        unresolved_domain_filter = True
                    filtered.append(item)
                singularities = tuple(filtered)
                details["singularity_domain_filter_unresolved"] = (
                    unresolved_domain_filter
                )
            details["singularity_raw"] = raw_singularities
            # A returned concrete collection is a complete enough response for
            # planner metadata; an empty tuple from a compact API means none found.
            details["singularity_complete"] = isinstance(
                raw_singularities, (tuple, list, set, frozenset)
            )
        except (
            TypeError,
            ValueError,
            NotImplementedError,
            AttributeError,
            RuntimeError,
            sp.PolynomialError,
        ) as exc:
            details["singularity_error"] = str(exc)

    return PosteriorGeometry(
        convexity=convexity,
        globally_concave=globally_concave,
        continuity=continuity,
        singularities=singularities,
        source="funcprops",
        details=details,
    )
