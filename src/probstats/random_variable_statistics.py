"""Standardized statistical algebra for symbolic random expressions."""

from __future__ import annotations

import sympy as sp

from ._random_variable_semantics import (
    affine_in_one_variable,
    normalize_assumptions,
    stochastic_variables,
)
from .assumptions import finite_moment
from .random_variable_algebra import (
    central_moment,
    covariance,
    expectation,
    variance,
)


def _known_nonfinite(expression, order, context):
    variables = stochastic_variables(expression)
    return any(
        finite_moment(variable, order, assumptions=context) is False
        for variable in variables
    )


class Correlation(sp.Function):
    nargs = 2


class Skewness(sp.Function):
    nargs = 1


class Kurtosis(sp.Function):
    nargs = 1


class StandardizedMoment(sp.Function):
    nargs = 2


class CoefficientOfVariation(sp.Function):
    nargs = 1


class FactorialMoment(sp.Function):
    nargs = 2


def correlation(left, right, *, assumptions=None):
    """Return symbolic Pearson correlation."""
    left = sp.sympify(left)
    right = sp.sympify(right)
    context = normalize_assumptions(assumptions)
    if _known_nonfinite(left, 2, context) or _known_nonfinite(right, 2, context):
        return sp.nan
    if left == right:
        return sp.S.One
    left_vars = stochastic_variables(left)
    right_vars = stochastic_variables(right)
    if len(left_vars) == len(right_vars) == 1 and left_vars == right_vars:
        variable = next(iter(left_vars))
        la = affine_in_one_variable(left, variable)
        ra = affine_in_one_variable(right, variable)
        if la is not None and ra is not None:
            _, a, _ = la
            _, c, _ = ra
            if (
                a.is_real is True
                and c.is_real is True
                and a.is_zero is False
                and c.is_zero is False
            ):
                return sp.sign(a * c)
    cov = covariance(left, right, assumptions=context)
    vl = variance(left, assumptions=context)
    vr = variance(right, assumptions=context)
    if cov == 0:
        return sp.S.Zero
    if vl == 0 or vr == 0:
        return sp.nan
    return sp.simplify(cov / sp.sqrt(vl * vr))


def standardized_moment(expression, order, *, assumptions=None):
    """Return a standardized central moment."""
    order = int(sp.sympify(order))
    if order < 0:
        raise ValueError("order must be nonnegative")
    if order == 0:
        return sp.S.One
    context = normalize_assumptions(assumptions)
    if _known_nonfinite(expression, max(order, 2), context):
        return sp.nan
    if order == 1:
        return sp.S.Zero
    var = variance(expression, assumptions=context)
    if var == 0:
        return sp.nan
    return sp.simplify(
        central_moment(expression, order, assumptions=context)
        / var ** sp.Rational(order, 2)
    )


def skewness(expression, *, assumptions=None):
    """Return the third standardized moment."""
    return standardized_moment(expression, 3, assumptions=assumptions)


def kurtosis(expression, *, excess=False, assumptions=None):
    """Return ordinary or excess kurtosis."""
    value = standardized_moment(expression, 4, assumptions=assumptions)
    return sp.simplify(value - 3) if excess else value


def coefficient_of_variation(expression, *, assumptions=None):
    """Return standard deviation divided by mean."""
    context = normalize_assumptions(assumptions)
    if _known_nonfinite(expression, 2, context):
        return sp.nan
    mu = expectation(expression, assumptions=context)
    if mu == 0:
        return sp.nan
    return sp.simplify(sp.sqrt(variance(expression, assumptions=context)) / mu)


def factorial_moment(expression, order, *, assumptions=None):
    """Return a falling-factorial moment."""
    order = int(sp.sympify(order))
    if order < 0:
        raise ValueError("order must be nonnegative")
    if order == 0:
        return sp.S.One
    context = normalize_assumptions(assumptions)
    expr = sp.sympify(expression)
    if _known_nonfinite(expr, order, context):
        return sp.nan
    from .random_variables import RandomVariable

    if isinstance(expr, RandomVariable) and expr.distribution is not None:
        try:
            return sp.simplify(expr.distribution.factorial_moment(order))
        except (TypeError, ValueError, NotImplementedError):
            pass
    return sp.expand(expectation(sp.ff(expr, order), assumptions=context))


__all__ = [
    "CoefficientOfVariation",
    "Correlation",
    "FactorialMoment",
    "Kurtosis",
    "Skewness",
    "StandardizedMoment",
    "coefficient_of_variation",
    "correlation",
    "factorial_moment",
    "kurtosis",
    "skewness",
    "standardized_moment",
]
