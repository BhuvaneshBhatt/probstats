"""Optional proof-oriented expression certification backed by exprtest."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from importlib import import_module
from typing import Any

import sympy as sp


class CertificationStatus(str, Enum):
    """Strength and direction of a mathematical expression certificate."""

    ZERO = "zero"
    NONZERO = "nonzero"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ExpressionCertificate:
    """Proof-oriented zero/nonzero classification for a symbolic expression.

    ``proven`` is true only when the selected backend supplied a proof.  A
    numerical witness or heuristic zero indication is represented
    as ``UNKNOWN`` because Bayesian normalization and curvature checks must not
    promote suggestive evidence to a mathematical certificate.
    """

    expression: sp.Expr
    status: CertificationStatus
    proven: bool
    backend: str
    method: str = ""
    reason: str = ""
    raw: Any | None = None

    @property
    def is_zero(self) -> bool | None:
        """Return ``True``/``False`` only for a proved zero/nonzero result."""
        if not self.proven:
            return None
        if self.status is CertificationStatus.ZERO:
            return True
        if self.status is CertificationStatus.NONZERO:
            return False
        return None


def certify_zero(
    expr: Any,
    *,
    use_exprtest: bool = True,
    exprtest_loader: Callable[[], Any] | None = None,
) -> ExpressionCertificate:
    """Try to prove that ``expr`` is zero or nonzero.

    The optional :mod:`exprtest` backend is tried first when installed, using
    ``confidence="certified"`` so a ``False`` result means *proved* nonzero
    rather than merely a strong numerical witness. Native SymPy exact
    predicates are then used as a dependency-free fallback.
    """
    value = sp.simplify(sp.sympify(expr))
    if use_exprtest:
        try:
            exprtest = (
                import_module("exprtest")
                if exprtest_loader is None
                else exprtest_loader()
            )
        except ImportError:
            pass
        else:
            try:
                direct = exprtest.zerotest(value, confidence="certified")
            except (
                TypeError,
                ValueError,
                NotImplementedError,
                AttributeError,
                RuntimeError,
                sp.PolynomialError,
            ):
                direct = None
            if direct is True:
                return ExpressionCertificate(
                    value,
                    CertificationStatus.ZERO,
                    True,
                    "exprtest",
                    "zerotest",
                    "exprtest certified the expression as zero",
                    direct,
                )
            if direct is False:
                return ExpressionCertificate(
                    value,
                    CertificationStatus.NONZERO,
                    True,
                    "exprtest",
                    "zerotest",
                    "exprtest certified the expression as nonzero",
                    direct,
                )
            exprtest_unknown = True

    else:
        exprtest_unknown = False

    if value == 0 or value.is_zero is True:
        return ExpressionCertificate(
            value,
            CertificationStatus.ZERO,
            True,
            "sympy",
            "is_zero",
            "SymPy proved zero",
        )
    if value.is_zero is False:
        return ExpressionCertificate(
            value,
            CertificationStatus.NONZERO,
            True,
            "sympy",
            "is_zero",
            "SymPy proved nonzero",
        )
    reason = (
        "exprtest returned UNKNOWN and SymPy could not prove zero/nonzero"
        if use_exprtest and "exprtest_unknown" in locals() and exprtest_unknown
        else "Neither exprtest nor SymPy proved zero/nonzero"
    )
    return ExpressionCertificate(
        value, CertificationStatus.UNKNOWN, False, "sympy", "is_zero", reason
    )


def certify_equal(
    left: Any,
    right: Any,
    *,
    use_exprtest: bool = True,
    exprtest_loader: Callable[[], Any] | None = None,
) -> ExpressionCertificate:
    """Certify equality by applying :func:`certify_zero` to ``left - right``."""
    return certify_zero(
        sp.sympify(left) - sp.sympify(right),
        use_exprtest=use_exprtest,
        exprtest_loader=exprtest_loader,
    )
