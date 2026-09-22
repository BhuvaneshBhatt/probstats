"""Moment-Taylor approximations and delta-method semantics."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from ._random_variable_semantics import normalize_assumptions, stochastic_variables
from .random_variable_algebra import _order, central_moment, expectation, variance
from .random_variables import RandomVariable


@dataclass(frozen=True, slots=True)
class TaylorApproximationResult:
    expression: sp.Expr
    order: int
    expansion_point: sp.Expr
    required_moments: tuple[int, ...]
    exact: bool


@dataclass(frozen=True, slots=True)
class DeltaMethodResult:
    transformed_center: sp.Expr
    derivative: sp.Expr
    asymptotic_variance: sp.Expr
    limit_distribution: object


def _resolve_variable(expression, variable):
    expression = sp.sympify(expression)
    variables = stochastic_variables(expression)
    if variable is None:
        if len(variables) != 1:
            raise ValueError(
                "variable is required unless expression depends on exactly one random variable"
            )
        variable = next(iter(variables))
    if not isinstance(variable, RandomVariable):
        raise TypeError("variable must be a RandomVariable")
    if variable not in variables:
        raise ValueError("expression does not depend on variable")
    if variables != frozenset((variable,)):
        raise ValueError(
            "moment-Taylor approximation currently requires one random variable"
        )
    return expression, variable


def _central(variable, degree, context):
    return (
        sp.S.One
        if degree == 0
        else central_moment(variable, degree, assumptions=context)
    )


def taylor_expectation(
    expression, *, variable=None, order=2, assumptions=None, return_result=False
):
    """Approximate an expectation by a central-moment Taylor expansion."""
    expression, variable = _resolve_variable(expression, variable)
    degree = _order(order)
    context = normalize_assumptions(assumptions)
    center = expectation(variable, assumptions=context)
    result = sp.S.Zero
    for index in range(degree + 1):
        derivative = sp.diff(expression, variable, index).subs(variable, center)
        result += derivative * _central(variable, index, context) / sp.factorial(index)
    result = sp.simplify(result)
    if return_result:
        return TaylorApproximationResult(
            result, degree, center, tuple(range(2, degree + 1)), degree == 0
        )
    return result


def taylor_variance(
    expression, *, variable=None, order=1, assumptions=None, return_result=False
):
    """Approximate a variance from a truncated Taylor polynomial."""
    expression, variable = _resolve_variable(expression, variable)
    degree = _order(order)
    context = normalize_assumptions(assumptions)
    center = expectation(variable, assumptions=context)
    if degree == 0:
        result = sp.S.Zero
    else:
        coefficients = {
            index: sp.diff(expression, variable, index).subs(variable, center)
            / sp.factorial(index)
            for index in range(1, degree + 1)
        }
        mean_shift = sp.Add(
            *(
                coefficients[index] * _central(variable, index, context)
                for index in range(1, degree + 1)
            )
        )
        second = sp.Add(
            *(
                coefficients[left]
                * coefficients[right]
                * _central(variable, left + right, context)
                for left in range(1, degree + 1)
                for right in range(1, degree + 1)
            )
        )
        result = sp.simplify(second - mean_shift**2)
    if return_result:
        return TaylorApproximationResult(
            result,
            degree,
            center,
            tuple(range(2, 2 * degree + 1)),
            degree == 0,
        )
    return result


def delta_variance(expression, *, variable=None, assumptions=None):
    """Return the first-order propagated variance approximation."""
    return taylor_variance(
        expression, variable=variable, order=1, assumptions=assumptions
    )


def delta_method(
    expression,
    *,
    variable=None,
    center=None,
    asymptotic_variance=None,
    assumptions=None,
):
    """Apply the scalar first-order delta method.

    If ``sqrt(n)(T_n-center) -> Normal(0, asymptotic_variance)``, return a
    :class:`DeltaMethodResult` describing the transformed center, derivative,
    asymptotic variance, and implied normal limit. Use :func:`delta_variance`
    when only the propagated first-order variance is wanted.
    """
    expression, variable = _resolve_variable(expression, variable)
    context = normalize_assumptions(assumptions)
    center = (
        expectation(variable, assumptions=context)
        if center is None
        else sp.sympify(center)
    )
    asymptotic_variance = (
        variance(variable, assumptions=context)
        if asymptotic_variance is None
        else sp.sympify(asymptotic_variance)
    )
    derivative = sp.simplify(sp.diff(expression, variable).subs(variable, center))
    propagated = sp.simplify(derivative**2 * asymptotic_variance)
    from .distributions import Normal

    limit_distribution = (
        sp.S.Zero if propagated == 0 else Normal(0, sp.sqrt(propagated))
    )
    return DeltaMethodResult(
        transformed_center=sp.simplify(expression.subs(variable, center)),
        derivative=derivative,
        asymptotic_variance=propagated,
        limit_distribution=limit_distribution,
    )


__all__ = [
    "DeltaMethodResult",
    "TaylorApproximationResult",
    "delta_method",
    "delta_variance",
    "taylor_expectation",
    "taylor_variance",
]
