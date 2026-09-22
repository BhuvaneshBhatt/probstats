"""Conditional expectation, moments, covariance, and variance algebra."""

from __future__ import annotations

from itertools import combinations

import sympy as sp

from ._random_variable_semantics import (
    certified_independent_collections,
    is_measurable_with_respect_to,
    is_random_expression,
    normalize_assumptions,
    split_measurable_product,
    stochastic_variables,
)
from .assumptions import conditionally_independent
from .random_variable_algebra import covariance, expectation, variance
from .random_variables import RandomVariable


class ConditionalExpectation(sp.Function):
    nargs = 2


class ConditionalVariance(sp.Function):
    nargs = 2


class ConditionalCovariance(sp.Function):
    nargs = 3


class ConditionalMoment(sp.Function):
    nargs = 4


def _given_tuple(given):
    if isinstance(given, RandomVariable):
        values = (given,)
    else:
        try:
            values = tuple(given)
        except TypeError as exc:
            raise TypeError(
                "given must be a RandomVariable or iterable of them"
            ) from exc
    if not values or not all(isinstance(value, RandomVariable) for value in values):
        raise TypeError("given must contain RandomVariable objects")
    return tuple(sorted(set(values), key=sp.default_sort_key))


def _independent_of_given(expression, given, context):
    variables = stochastic_variables(expression)
    remaining = variables.difference(given)
    if not remaining:
        return False
    # Independence from every conditioning variable separately is insufficient.
    return certified_independent_collections(remaining, given, context)


def conditional_expectation(expression, *, given, assumptions=None):
    """Normalize conditional expectation using measurability and independence rules."""
    expr = sp.sympify(expression)
    gs = _given_tuple(given)
    context = normalize_assumptions(assumptions)
    if not is_random_expression(expr):
        return expr
    if is_measurable_with_respect_to(expr, gs):
        return expr
    # X ⟂ Y | Z implies E[X | Y,Z] = E[X | Z].
    if isinstance(expr, RandomVariable) and len(gs) >= 2:
        for redundant in gs:
            remaining = tuple(g for g in gs if g != redundant)
            if (
                remaining
                and conditionally_independent(
                    expr, redundant, given=remaining, assumptions=context
                )
                is True
            ):
                return conditional_expectation(
                    expr, given=remaining, assumptions=context
                )
    if isinstance(expr, sp.Add):
        return sp.Add(
            *(
                conditional_expectation(term, given=gs, assumptions=context)
                for term in expr.args
            )
        )
    if isinstance(expr, sp.Mul):
        measurable, remainder = split_measurable_product(expr, gs)
        if measurable != 1:
            return sp.simplify(
                measurable
                * conditional_expectation(remainder, given=gs, assumptions=context)
            )
    if _independent_of_given(expr, gs, context):
        return expectation(expr, assumptions=context)
    return ConditionalExpectation(expr, sp.Tuple(*gs))


def conditional_variance(expression, *, given, assumptions=None):
    """Normalize conditional variance using measurability and covariance rules."""
    expr = sp.sympify(expression)
    gs = _given_tuple(given)
    context = normalize_assumptions(assumptions)
    if not is_random_expression(expr) or is_measurable_with_respect_to(expr, gs):
        return sp.S.Zero
    if isinstance(expr, sp.Add):
        measurable = sp.Add(
            *(term for term in expr.args if is_measurable_with_respect_to(term, gs))
        )
        if measurable != 0:
            return conditional_variance(
                sp.simplify(expr - measurable), given=gs, assumptions=context
            )
        terms = tuple(term for term in expr.args if is_random_expression(term))
        diagonal = sp.Add(
            *(
                conditional_variance(term, given=gs, assumptions=context)
                for term in terms
            )
        )
        cross = sp.Add(
            *(
                2 * conditional_covariance(left, right, given=gs, assumptions=context)
                for left, right in combinations(terms, 2)
            )
        )
        return sp.expand(diagonal + cross)
    if isinstance(expr, sp.Mul):
        measurable, remainder = split_measurable_product(expr, gs)
        if measurable != 1 and (
            measurable.is_real is True or is_random_expression(measurable)
        ):
            return sp.simplify(
                measurable**2
                * conditional_variance(remainder, given=gs, assumptions=context)
            )
    if _independent_of_given(expr, gs, context):
        return variance(expr, assumptions=context)
    return ConditionalVariance(expr, sp.Tuple(*gs))


def conditional_covariance(left, right, *, given, assumptions=None):
    """Normalize conditional covariance using bilinearity and conditional independence."""
    left = sp.sympify(left)
    right = sp.sympify(right)
    gs = _given_tuple(given)
    context = normalize_assumptions(assumptions)
    if (
        not is_random_expression(left)
        or not is_random_expression(right)
        or is_measurable_with_respect_to(left, gs)
        or is_measurable_with_respect_to(right, gs)
    ):
        return sp.S.Zero
    if isinstance(left, sp.Add):
        return sp.Add(
            *(
                conditional_covariance(term, right, given=gs, assumptions=context)
                for term in left.args
            )
        )
    if isinstance(right, sp.Add):
        return sp.Add(
            *(
                conditional_covariance(left, term, given=gs, assumptions=context)
                for term in right.args
            )
        )
    lm, lr = split_measurable_product(left, gs)
    rm, rr = split_measurable_product(right, gs)
    if (lm != 1 or rm != 1) and lm.is_real is True and rm.is_real is True:
        return sp.simplify(
            lm * rm * conditional_covariance(lr, rr, given=gs, assumptions=context)
        )
    if left == right:
        return conditional_variance(left, given=gs, assumptions=context)
    if (
        len(gs) == 1
        and isinstance(left, RandomVariable)
        and isinstance(right, RandomVariable)
        and conditionally_independent(left, right, given=gs, assumptions=context)
        is True
    ):
        return sp.S.Zero
    first, second = sorted((left, right), key=sp.default_sort_key)
    return ConditionalCovariance(first, second, sp.Tuple(*gs))


def conditional_moment(expression, order, *, given, central=False, assumptions=None):
    """Return a raw or central conditional moment."""
    expr = sp.sympify(expression)
    degree = int(sp.sympify(order))
    if degree < 0:
        raise ValueError("order must be nonnegative")
    gs = _given_tuple(given)
    context = normalize_assumptions(assumptions)
    if degree == 0:
        return sp.S.One
    if not central:
        return conditional_expectation(expr**degree, given=gs, assumptions=context)
    if degree == 1:
        return sp.S.Zero
    if degree == 2:
        return conditional_variance(expr, given=gs, assumptions=context)
    center = conditional_expectation(expr, given=gs, assumptions=context)
    expanded = conditional_expectation(
        (expr - center) ** degree, given=gs, assumptions=context
    )
    if isinstance(expanded, ConditionalExpectation):
        return ConditionalMoment(expr, sp.Integer(degree), sp.true, sp.Tuple(*gs))
    return sp.simplify(expanded)


def total_expectation(expression, *, given, assumptions=None, expanded=False):
    """Return the tower-property identity, optionally in expanded form."""
    gs = _given_tuple(given)
    context = normalize_assumptions(assumptions)
    expr = sp.sympify(expression)
    if expanded:
        return expectation(
            conditional_expectation(expr, given=gs, assumptions=context),
            assumptions=context,
        )
    return expectation(expr, assumptions=context)


def total_variance(expression, *, given, assumptions=None, expanded=False):
    """Return the law of total variance, optionally in expanded form."""
    gs = _given_tuple(given)
    context = normalize_assumptions(assumptions)
    expr = sp.sympify(expression)
    if not expanded:
        return variance(expr, assumptions=context)
    cv = conditional_variance(expr, given=gs, assumptions=context)
    ce = conditional_expectation(expr, given=gs, assumptions=context)
    return expectation(cv, assumptions=context) + variance(ce, assumptions=context)


def total_covariance(left, right, *, given, assumptions=None, expanded=False):
    """Return the law of total covariance, optionally in expanded form."""
    gs = _given_tuple(given)
    context = normalize_assumptions(assumptions)
    left = sp.sympify(left)
    right = sp.sympify(right)
    if not expanded:
        return covariance(left, right, assumptions=context)
    inner = conditional_covariance(left, right, given=gs, assumptions=context)
    left_mean = conditional_expectation(left, given=gs, assumptions=context)
    right_mean = conditional_expectation(right, given=gs, assumptions=context)
    return expectation(inner, assumptions=context) + covariance(
        left_mean, right_mean, assumptions=context
    )


__all__ = [
    "ConditionalCovariance",
    "ConditionalExpectation",
    "ConditionalMoment",
    "ConditionalVariance",
    "conditional_covariance",
    "conditional_expectation",
    "conditional_moment",
    "conditional_variance",
    "total_covariance",
    "total_expectation",
    "total_variance",
]
