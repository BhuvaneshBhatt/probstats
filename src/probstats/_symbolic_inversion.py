"""Recognition and exact inversion of univariate probability transforms."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from ._symbolic_predicates import TruthValue, certified_equal
from ._symbolic_transforms import characteristic_function, moment_generating_function
from .distributions import (
    Bernoulli,
    Cauchy,
    Exponential,
    Gamma,
    Laplace,
    Logistic,
    NegativeBinomial,
    Normal,
    Poisson,
    Uniform,
)
from .distributions.base import Distribution


@dataclass(frozen=True, slots=True)
class TransformInversionDistribution(Distribution):
    """Exact symbolic law defined by inversion of a characteristic transform."""

    transform_expression: sp.Expr
    transform_variable: sp.Symbol
    transform_kind: str = "characteristic"

    def __post_init__(self):
        object.__setattr__(
            self, "transform_expression", sp.sympify(self.transform_expression)
        )
        object.__setattr__(
            self, "transform_variable", sp.sympify(self.transform_variable)
        )
        if not isinstance(self.transform_variable, sp.Symbol):
            raise TypeError("transform_variable must be a SymPy Symbol")
        if self.transform_kind not in {"characteristic", "mgf"}:
            raise ValueError("transform_kind must be 'characteristic' or 'mgf'")

    @property
    def support(self):
        return sp.S.Reals

    @property
    def parameters(self):
        return (self.transform_expression,)

    def _characteristic(self):
        if self.transform_kind == "characteristic":
            return self.transform_expression
        u = self.transform_variable
        return sp.simplify(self.transform_expression.subs(u, sp.I * u))

    def pdf(self, value):
        x = sp.sympify(value)
        t = self.transform_variable
        phi = self._characteristic()
        return sp.Integral(sp.exp(-sp.I * t * x) * phi, (t, -sp.oo, sp.oo)) / (
            2 * sp.pi
        )

    def logpdf(self, value):
        return sp.log(self.pdf(value))


def _same_expr(a, b):
    try:
        return certified_equal(a, b) is TruthValue.TRUE
    except (TypeError, ValueError):
        return False


def _valid_candidate(factory):
    """Construct a candidate distribution, returning None for invalid parameters."""
    try:
        return factory()
    except ValueError:
        return None


def _add_candidate(candidates, factory):
    candidate = _valid_candidate(factory)
    if candidate is not None:
        candidates.append(candidate)


def distribution_from_mgf(mgf, t) -> Distribution:
    """Recognize/invert a moment-generating function.

    Recognition is verified by substitution into the candidate's own MGF;
    otherwise an exact Fourier-inversion distribution is retained via
    ``M(i t)``.
    """
    t = sp.sympify(t)
    m = sp.sympify(mgf)
    if not isinstance(t, sp.Symbol):
        raise TypeError("t must be a SymPy Symbol")
    if _same_expr(m.subs(t, 0), 1) is False:
        raise ValueError("an MGF must equal 1 at t=0")

    # Cumulant-based candidates, always verified against the original MGF.
    logm = sp.log(m)
    mean = sp.simplify(sp.diff(logm, t).subs(t, 0))
    var = sp.simplify(sp.diff(logm, t, 2).subs(t, 0))
    candidates = []
    if var.is_positive is True:
        candidates.append(Normal(mean, sp.sqrt(var)))
    if mean.is_positive is True and var.is_positive is True:
        shape = sp.simplify(mean**2 / var)
        scale = sp.simplify(var / mean)
        candidates.append(Gamma(shape, scale))
    # Poisson has equal first two cumulants.
    if _same_expr(mean, var) and mean.is_nonnegative is not False:
        candidates.append(Poisson(mean))
    # Additional two-cumulant candidates are harmless because every candidate
    # is verified by reconstructing its exact MGF.
    if var.is_positive is True:
        for factory in (
            lambda: Laplace(mean, sp.sqrt(var / 2)),
            lambda: Logistic(mean, sp.sqrt(3 * var) / sp.pi),
            lambda: Uniform(mean - sp.sqrt(3 * var), mean + sp.sqrt(3 * var)),
        ):
            _add_candidate(candidates, factory)
    if mean.is_positive is True and var.is_positive is True:
        _add_candidate(candidates, lambda: Exponential(sp.simplify(1 / mean)))
        if (var - mean).is_positive is True:
            nb_p = sp.simplify(mean / var)
            nb_r = sp.simplify(mean**2 / (var - mean))
            _add_candidate(candidates, lambda: NegativeBinomial(nb_r, nb_p))
    # Bernoulli can be recovered from its first derivative and exact verification.
    p = sp.simplify(sp.diff(m, t).subs(t, 0))
    _add_candidate(candidates, lambda: Bernoulli(p))
    for candidate in candidates:
        try:
            cm = moment_generating_function(candidate, t)
            if isinstance(cm, sp.Piecewise):
                cm = cm.args[0][0]
            if _same_expr(m, cm):
                return candidate
        except (TypeError, ValueError):
            continue
    return TransformInversionDistribution(m, t, "mgf")


def _recognized_mgf(mgf, variable):
    try:
        return distribution_from_mgf(mgf, variable)
    except (TypeError, ValueError):
        return None


def _matches_characteristic(candidate, transform, variable):
    try:
        candidate_transform = characteristic_function(candidate, variable)
        # Some exact characteristic functions use a Piecewise branch only to
        # define a removable singularity at t = 0.  The input transform has
        # already been certified to equal 1 there, so compare the generic
        # branch rather than sending relational Piecewise conditions through
        # the algebraic identity checker.
        if isinstance(candidate_transform, sp.Piecewise):
            branches = candidate_transform.args
            if (
                len(branches) == 2
                and branches[0][0] == 1
                and branches[0][1] == sp.Eq(variable, 0)
                and branches[1][1] is sp.S.true
            ):
                candidate_transform = branches[1][0]
        return _same_expr(transform, candidate_transform)
    except (TypeError, ValueError, NotImplementedError):
        return False


def distribution_from_characteristic_function(phi, t) -> Distribution:
    """Recognize/invert a characteristic function exactly when possible."""
    t = sp.sympify(t)
    p = sp.sympify(phi)
    if not isinstance(t, sp.Symbol):
        raise TypeError("t must be a SymPy Symbol")
    if _same_expr(p.subs(t, 0), 1) is False:
        raise ValueError("a characteristic function must equal 1 at t=0")

    # Normal from first two derivatives of log(phi), with exact verification.
    logp = sp.log(p)
    mu = sp.simplify(sp.diff(logp, t).subs(t, 0) / sp.I)
    variance = sp.simplify(-sp.diff(logp, t, 2).subs(t, 0))
    candidates = []
    if variance.is_positive is True:
        candidates.append(Normal(mu, sp.sqrt(variance)))
        for factory in (
            lambda: Laplace(mu, sp.sqrt(variance / 2)),
            lambda: Logistic(mu, sp.sqrt(3 * variance) / sp.pi),
        ):
            _add_candidate(candidates, factory)
    # Cauchy is not twice differentiable at zero, so detect its canonical
    # ``exp(i*mu*t - scale*Abs(t))`` form directly and verify it below.
    wild_mu = sp.Wild("mu", exclude=[t])
    wild_scale = sp.Wild("scale", exclude=[t])
    match = p.match(sp.exp(sp.I * wild_mu * t - wild_scale * sp.Abs(t)))
    if match and wild_scale in match and wild_mu in match:
        _add_candidate(candidates, lambda: Cauchy(match[wild_mu], match[wild_scale]))
    # Poisson/Bernoulli candidates via MGF substitution when analytically meaningful.
    mgf_guess = sp.simplify(p.subs(t, -sp.I * t))
    mgf_candidate = _recognized_mgf(mgf_guess, t)
    if mgf_candidate is not None:
        candidates.append(mgf_candidate)
    for candidate in candidates:
        if isinstance(candidate, TransformInversionDistribution):
            continue
        if _matches_characteristic(candidate, p, t):
            return candidate
    return TransformInversionDistribution(p, t, "characteristic")


__all__ = [
    "TransformInversionDistribution",
    "distribution_from_characteristic_function",
    "distribution_from_mgf",
]
