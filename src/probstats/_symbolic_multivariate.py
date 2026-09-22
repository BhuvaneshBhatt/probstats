"""Multivariate transforms and transform-equivalence proofs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import sympy as sp

from ._integration import integrate_rectangular
from ._symbolic_transforms import characteristic_function, moment_generating_function
from .composition import ProductDistribution
from .distributions import MultivariateNormal
from .distributions.base import Distribution
from .joint import CopulaDistribution, JointDistribution, _eliminate_coordinates


def multivariate_characteristic_function(distribution: Distribution, t=None):
    """Return the joint characteristic function ``E[exp(i t.T X)]``.

    Closed forms are used for multivariate normals and independent products;
    exact iterated integrals are retained for generic continuous joint laws.
    """
    shape = distribution.event_space.shape
    if distribution.event_space.rank != 1:
        raise TypeError(
            "multivariate characteristic functions require vector-valued laws"
        )
    dim = shape[0]
    ts = (
        tuple(sp.symbols(f"t0:{dim}", real=True))
        if t is None
        else tuple(map(sp.sympify, t))
    )
    if len(ts) != dim:
        raise ValueError("one transform variable is required per event coordinate")
    tv = sp.ImmutableMatrix([ts])
    if isinstance(distribution, MultivariateNormal):
        mu = sp.ImmutableMatrix(distribution.mean)
        cov = sp.ImmutableMatrix(distribution.covariance)
        return sp.simplify(
            sp.exp(sp.I * (tv * mu)[0] - sp.Rational(1, 2) * (tv * cov * tv.T)[0])
        )
    if isinstance(distribution, ProductDistribution):
        if any(c.event_space.rank != 0 for c in distribution.components):
            raise NotImplementedError(
                "nested vector product transforms are not implemented"
            )
        return sp.simplify(
            sp.prod(
                characteristic_function(c, ti)
                for c, ti in zip(distribution.components, ts)
            )
        )
    if isinstance(distribution, JointDistribution):
        expr = distribution.joint_density * sp.exp(
            sp.I * sum(ti * xi for ti, xi in zip(ts, distribution.variables))
        )
        return _eliminate_coordinates(
            expr,
            distribution.variables,
            distribution.supports,
            range(distribution.dimension),
            discrete=distribution.discrete,
        )
    if isinstance(distribution, CopulaDistribution):
        xs = tuple(sp.Symbol(f"x{i}", real=True) for i in range(dim))
        expr = distribution.pdf(xs) * sp.exp(
            sp.I * sum(ti * xi for ti, xi in zip(ts, xs))
        )
        ranges = [
            (x, marginal.support.inf, marginal.support.sup)
            for x, marginal in zip(xs, distribution.marginals)
        ]
        return integrate_rectangular(expr, ranges)
    raise NotImplementedError(
        f"no multivariate characteristic transform for {type(distribution).__name__}"
    )


def multivariate_moment_generating_function(distribution: Distribution, t=None):
    """Return the joint MGF ``E[exp(t.T X)]`` when it exists symbolically."""
    shape = distribution.event_space.shape
    if distribution.event_space.rank != 1:
        raise TypeError("multivariate MGFs require vector-valued laws")
    dim = shape[0]
    ts = (
        tuple(sp.symbols(f"t0:{dim}", real=True))
        if t is None
        else tuple(map(sp.sympify, t))
    )
    if len(ts) != dim:
        raise ValueError("one transform variable is required per event coordinate")
    tv = sp.ImmutableMatrix([ts])
    if isinstance(distribution, MultivariateNormal):
        mu = sp.ImmutableMatrix(distribution.mean)
        cov = sp.ImmutableMatrix(distribution.covariance)
        return sp.simplify(
            sp.exp((tv * mu)[0] + sp.Rational(1, 2) * (tv * cov * tv.T)[0])
        )
    if isinstance(distribution, ProductDistribution):
        if any(c.event_space.rank != 0 for c in distribution.components):
            raise NotImplementedError(
                "nested vector product transforms are not implemented"
            )
        return sp.simplify(
            sp.prod(
                moment_generating_function(c, ti)
                for c, ti in zip(distribution.components, ts)
            )
        )
    if isinstance(distribution, JointDistribution):
        expr = distribution.joint_density * sp.exp(
            sum(ti * xi for ti, xi in zip(ts, distribution.variables))
        )
        return _eliminate_coordinates(
            expr,
            distribution.variables,
            distribution.supports,
            range(distribution.dimension),
            discrete=distribution.discrete,
        )
    raise NotImplementedError(f"no multivariate MGF for {type(distribution).__name__}")


class ProofStatus(str, Enum):
    PROVED = "proved"
    DISPROVED = "disproved"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class TransformEquivalenceProof:
    """Certificate/result for transform-based equality of probability laws."""

    status: ProofStatus
    transform_kind: str
    left_transform: sp.Expr
    right_transform: sp.Expr
    residual: sp.Expr
    uniqueness_domain: sp.Basic
    reason: str

    @property
    def proved(self):
        return self.status is ProofStatus.PROVED


def _transform_residual(left, right):
    try:
        return sp.simplify(sp.expand_func(left - right))
    except (TypeError, ValueError):
        return sp.sympify(left) - sp.sympify(right)


def _residual_witness_nonzero(residual, variables):
    """Return True when a simple exact substitution witnesses inequality."""
    variables = tuple(variables)
    probes = (sp.Rational(1, 2), sp.Integer(1), sp.Integer(2), sp.Integer(-1))
    if not variables:
        return residual != 0
    for value in probes:
        try:
            candidate = sp.simplify(residual.subs({v: value for v in variables}))
            if candidate.is_zero is False or (candidate.is_number and candidate != 0):
                return True
        except (TypeError, ValueError, ZeroDivisionError):
            continue
    return False


def prove_equivalent_by_transform(
    left: Distribution, right: Distribution, *, kind="auto"
) -> TransformEquivalenceProof:
    """Try to prove distribution equality using a uniqueness theorem.

    Characteristic functions uniquely determine probability distributions on
    Euclidean spaces, so exact CF equality is a proof.  MGF equality is a proof
    only when both MGFs are finite on an open neighborhood of zero; this helper
    uses MGF proofs only for families whose implementation supplies such a
    neighborhood without an unevaluated integral.
    """
    if left.event_space.shape != right.event_space.shape:
        return TransformEquivalenceProof(
            ProofStatus.DISPROVED,
            "shape",
            sp.nan,
            sp.nan,
            sp.nan,
            sp.false,
            "event shapes differ",
        )

    vector = left.event_space.rank == 1
    if vector != (right.event_space.rank == 1):
        return TransformEquivalenceProof(
            ProofStatus.DISPROVED,
            "rank",
            sp.nan,
            sp.nan,
            sp.nan,
            sp.false,
            "event ranks differ",
        )

    if kind not in {"auto", "characteristic", "mgf"}:
        raise ValueError("kind must be 'auto', 'characteristic', or 'mgf'")

    # Prefer CF because its uniqueness theorem applies to every probability law.
    if kind in {"auto", "characteristic"}:
        try:
            if vector:
                ts = tuple(sp.symbols(f"_t0:{left.event_space.shape[0]}", real=True))
                lt = multivariate_characteristic_function(left, ts)
                rt = multivariate_characteristic_function(right, ts)
            else:
                t = sp.Symbol("_t", real=True)
                lt = characteristic_function(left, t)
                rt = characteristic_function(right, t)
            residual = _transform_residual(lt, rt)
            if residual == 0:
                return TransformEquivalenceProof(
                    ProofStatus.PROVED,
                    "characteristic",
                    lt,
                    rt,
                    residual,
                    sp.S.Reals,
                    "characteristic functions are equal; CF uniqueness applies",
                )
            cf_vars = ts if vector else (t,)
            if residual.is_zero is False or _residual_witness_nonzero(
                residual, cf_vars
            ):
                return TransformEquivalenceProof(
                    ProofStatus.DISPROVED,
                    "characteristic",
                    lt,
                    rt,
                    residual,
                    sp.S.Reals,
                    "characteristic functions differ",
                )
            if kind == "characteristic":
                return TransformEquivalenceProof(
                    ProofStatus.UNKNOWN,
                    "characteristic",
                    lt,
                    rt,
                    residual,
                    sp.S.Reals,
                    "symbolic equality is undecided",
                )
        except (NotImplementedError, TypeError, ValueError):
            if kind == "characteristic":
                raise

    # Conservative MGF fallback.  An explicit finite expression near t=0 is
    # accepted; Piecewise/integral forms are left unknown rather than overstated.
    try:
        if vector:
            ts = tuple(sp.symbols(f"_t0:{left.event_space.shape[0]}", real=True))
            lt = multivariate_moment_generating_function(left, ts)
            rt = multivariate_moment_generating_function(right, ts)
        else:
            t = sp.Symbol("_t", real=True)
            lt = moment_generating_function(left, t)
            rt = moment_generating_function(right, t)
        residual = _transform_residual(lt, rt)
        unsafe = any(
            expr.has(sp.Integral, sp.Sum, sp.oo, -sp.oo)
            or isinstance(expr, sp.Piecewise)
            for expr in (lt, rt)
        )
        if residual == 0 and not unsafe:
            return TransformEquivalenceProof(
                ProofStatus.PROVED,
                "mgf",
                lt,
                rt,
                residual,
                sp.Contains(0, sp.Interval.open(-1, 1)),
                "MGFs agree on an open neighborhood of zero",
            )
        mgf_vars = ts if vector else (t,)
        if residual.is_zero is False or _residual_witness_nonzero(residual, mgf_vars):
            return TransformEquivalenceProof(
                ProofStatus.DISPROVED, "mgf", lt, rt, residual, sp.true, "MGFs differ"
            )
        return TransformEquivalenceProof(
            ProofStatus.UNKNOWN,
            "mgf",
            lt,
            rt,
            residual,
            sp.true,
            "MGF equality/uniqueness conditions are not certified",
        )
    except (NotImplementedError, TypeError, ValueError):
        return TransformEquivalenceProof(
            ProofStatus.UNKNOWN,
            "none",
            sp.nan,
            sp.nan,
            sp.nan,
            sp.true,
            "no supported transform proof path",
        )


__all__ = [
    "ProofStatus",
    "TransformEquivalenceProof",
    "multivariate_characteristic_function",
    "multivariate_moment_generating_function",
    "prove_equivalent_by_transform",
]
