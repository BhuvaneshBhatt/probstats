"""Optional proof-oriented expression certification backed by exprtools."""

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
    use_exprtools: bool | None = None,
    exprtools_loader: Callable[[], Any] | None = None,
    use_exprtest: bool = True,
    exprtest_loader: Callable[[], Any] | None = None,
) -> ExpressionCertificate:
    """Try to prove that ``expr`` is zero or nonzero.

    The :mod:`exprtools` backend is tried first when installed, using
    ``confidence="certified"`` so a ``False`` result means *proved* nonzero
    rather than merely a strong numerical witness. Native SymPy exact
    predicates are then used as a dependency-free fallback.

    ``use_exprtest`` and ``exprtest_loader`` remain compatibility keywords.
    ``use_exprtools`` takes precedence when supplied. An explicit legacy loader
    retains the legacy backend label; the default and new loader use exprtools.
    """
    if exprtools_loader is not None and exprtest_loader is not None:
        raise ValueError("Supply only one of exprtools_loader and exprtest_loader")
    enabled = use_exprtest if use_exprtools is None else use_exprtools
    loader = exprtools_loader if exprtools_loader is not None else exprtest_loader
    backend = "exprtest" if exprtest_loader is not None else "exprtools"
    backend_unknown = False
    value = sp.simplify(sp.sympify(expr))
    if enabled:
        try:
            oracle = import_module("exprtools") if loader is None else loader()
        except ImportError:
            pass
        else:
            try:
                direct = oracle.zerotest(value, confidence="certified")
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
                    backend,
                    "zerotest",
                    f"{backend} certified the expression as zero",
                    direct,
                )
            if direct is False:
                return ExpressionCertificate(
                    value,
                    CertificationStatus.NONZERO,
                    True,
                    backend,
                    "zerotest",
                    f"{backend} certified the expression as nonzero",
                    direct,
                )
            backend_unknown = True

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
        f"{backend} returned UNKNOWN and SymPy could not prove zero/nonzero"
        if backend_unknown
        else f"Neither {backend} nor SymPy proved zero/nonzero"
    )
    return ExpressionCertificate(
        value, CertificationStatus.UNKNOWN, False, "sympy", "is_zero", reason
    )


def certify_equal(
    left: Any,
    right: Any,
    *,
    use_exprtools: bool | None = None,
    exprtools_loader: Callable[[], Any] | None = None,
    use_exprtest: bool = True,
    exprtest_loader: Callable[[], Any] | None = None,
) -> ExpressionCertificate:
    """Certify equality by applying :func:`certify_zero` to ``left - right``."""
    return certify_zero(
        sp.sympify(left) - sp.sympify(right),
        use_exprtools=use_exprtools,
        exprtools_loader=exprtools_loader,
        use_exprtest=use_exprtest,
        exprtest_loader=exprtest_loader,
    )
