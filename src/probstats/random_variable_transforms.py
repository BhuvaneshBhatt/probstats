"""Generating-function algebra for symbolic random variables."""

from __future__ import annotations

import sympy as sp

from ._random_variable_semantics import (
    affine_in_one_variable,
    certified_independent,
    normalize_assumptions,
    stochastic_variables,
)
from .random_variable_distributions import RandomExpressionLaw
from .random_variable_distributions import distribution as expression_distribution
from .random_variables import RandomVariable


class MomentGeneratingFunction(sp.Function):
    nargs = 2


class CumulantGeneratingFunction(sp.Function):
    nargs = 2


class CharacteristicFunction(sp.Function):
    nargs = 2


class ProbabilityGeneratingFunction(sp.Function):
    nargs = 2


def _known(expr, t, kind, assumptions=None):
    law = expression_distribution(expr, assumptions=assumptions)
    if law is None or isinstance(law, RandomExpressionLaw):
        return None
    from . import symbolic

    fn = {
        "mgf": symbolic.moment_generating_function,
        "cgf": symbolic.cumulant_generating_function,
        "cf": symbolic.characteristic_function,
        "pgf": symbolic.probability_generating_function,
    }[kind]
    try:
        return fn(law, t)
    except (TypeError, ValueError, NotImplementedError):
        return None


def moment_generating_function(expr, t=None, *, assumptions=None):
    """Return the MGF of a distribution or symbolic random expression."""
    from .distributions.base import Distribution

    if isinstance(expr, Distribution):
        if assumptions is not None:
            raise TypeError("assumptions apply only to random-variable expressions")
        from .symbolic import moment_generating_function as fn

        return fn(expr, t)
    t = sp.Symbol("t", real=True) if t is None else sp.sympify(t)
    expr = sp.sympify(expr)
    ctx = normalize_assumptions(assumptions)
    if not stochastic_variables(expr):
        return sp.exp(expr * t)
    aff = affine_in_one_variable(expr)
    if aff:
        x, a, b = aff
        if isinstance(x, RandomVariable) and x.distribution is not None:
            from .symbolic import moment_generating_function as mgf

            return sp.simplify(sp.exp(b * t) * mgf(x.distribution, a * t))
        return sp.exp(b * t) * MomentGeneratingFunction(x, a * t)
    if isinstance(expr, sp.Add):
        terms = tuple(t0 for t0 in expr.args if stochastic_variables(t0))
        vars_ = tuple(stochastic_variables(expr))
        if len(terms) > 1 and certified_independent(vars_, ctx):
            return sp.prod(
                moment_generating_function(term, t, assumptions=ctx) for term in terms
            ) * sp.exp(
                t
                * sum(
                    (term for term in expr.args if not stochastic_variables(term)),
                    sp.S.Zero,
                )
            )
    known = _known(expr, t, "mgf", assumptions=ctx)
    return known if known is not None else MomentGeneratingFunction(expr, t)


def cumulant_generating_function(expr, t=None, *, assumptions=None):
    """Return the CGF of a distribution or symbolic random expression."""
    from .distributions.base import Distribution

    if isinstance(expr, Distribution):
        if assumptions is not None:
            raise TypeError("assumptions apply only to random-variable expressions")
        from .symbolic import cumulant_generating_function as fn

        return fn(expr, t)
    t = sp.Symbol("t", real=True) if t is None else sp.sympify(t)
    expr = sp.sympify(expr)
    ctx = normalize_assumptions(assumptions)
    if not stochastic_variables(expr):
        return sp.expand(expr * t)
    aff = affine_in_one_variable(expr)
    if aff:
        x, a, b = aff
        if x.distribution is not None:
            from .symbolic import cumulant_generating_function as cgf

            return sp.simplify(b * t + cgf(x.distribution, a * t))
        return b * t + CumulantGeneratingFunction(x, a * t)
    if isinstance(expr, sp.Add):
        vars_ = tuple(stochastic_variables(expr))
        if len(vars_) > 1 and certified_independent(vars_, ctx):
            return sp.Add(
                *(
                    cumulant_generating_function(term, t, assumptions=ctx)
                    for term in expr.args
                )
            )
    return CumulantGeneratingFunction(expr, t)


def characteristic_function(expr, t=None, *, assumptions=None):
    """Return the characteristic function of a distribution or random expression."""
    from .distributions.base import Distribution

    if isinstance(expr, Distribution):
        if assumptions is not None:
            raise TypeError("assumptions apply only to random-variable expressions")
        from .symbolic import characteristic_function as fn

        return fn(expr, t)
    t = sp.Symbol("t", real=True) if t is None else sp.sympify(t)
    expr = sp.sympify(expr)
    ctx = normalize_assumptions(assumptions)
    if not stochastic_variables(expr):
        return sp.exp(sp.I * expr * t)
    aff = affine_in_one_variable(expr)
    if aff:
        x, a, b = aff
        if x.distribution is not None:
            from .symbolic import characteristic_function as cf

            return sp.simplify(sp.exp(sp.I * b * t) * cf(x.distribution, a * t))
        return sp.exp(sp.I * b * t) * CharacteristicFunction(x, a * t)
    if isinstance(expr, sp.Add):
        vars_ = tuple(stochastic_variables(expr))
        if len(vars_) > 1 and certified_independent(vars_, ctx):
            return sp.prod(
                characteristic_function(term, t, assumptions=ctx) for term in expr.args
            )
    return CharacteristicFunction(expr, t)


def probability_generating_function(expr, z=None, *, assumptions=None):
    """Return the PGF of a distribution or symbolic random expression."""
    from .distributions.base import Distribution

    if isinstance(expr, Distribution):
        if assumptions is not None:
            raise TypeError("assumptions apply only to random-variable expressions")
        from .symbolic import probability_generating_function as fn

        return fn(expr, z)
    z = sp.Symbol("z") if z is None else sp.sympify(z)
    expr = sp.sympify(expr)
    ctx = normalize_assumptions(assumptions)
    if not stochastic_variables(expr):
        if expr.is_integer is True and expr.is_nonnegative is True:
            return z**expr
        return ProbabilityGeneratingFunction(expr, z)
    if isinstance(expr, RandomVariable) and expr.distribution is not None:
        from .symbolic import probability_generating_function as pgf

        return pgf(expr.distribution, z)
    if isinstance(expr, sp.Add):
        vars_ = tuple(stochastic_variables(expr))
        if len(vars_) > 1 and certified_independent(vars_, ctx):
            return sp.prod(
                probability_generating_function(term, z, assumptions=ctx)
                for term in expr.args
            )
    return ProbabilityGeneratingFunction(expr, z)


__all__ = [
    "CharacteristicFunction",
    "CumulantGeneratingFunction",
    "MomentGeneratingFunction",
    "ProbabilityGeneratingFunction",
    "characteristic_function",
    "cumulant_generating_function",
    "moment_generating_function",
    "probability_generating_function",
]
