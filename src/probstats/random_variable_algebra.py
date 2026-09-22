"""Exact symbolic algebra for random-variable statistical operators."""

from __future__ import annotations

from itertools import combinations

import sympy as sp

from ._random_variable_semantics import (
    certified_independent as _certified_independent,
)
from ._random_variable_semantics import (
    disjoint_expressions_independent as _disjoint_expressions_independent,
)
from ._random_variable_semantics import (
    independent_addends as _independent_addends,
)
from ._random_variable_semantics import (
    is_random_expression,
    stochastic_variables,
)
from ._random_variable_semantics import (
    normalize_assumptions as _context,
)
from ._random_variable_semantics import (
    split_deterministic_product as _split_deterministic_product,
)
from .assumptions import independent, uncorrelated
from .random_variables import RandomVariable


class Expectation(sp.Function):
    """Unevaluated expectation of a random expression."""

    nargs = 1


class Variance(sp.Function):
    """Unevaluated variance of a random expression."""

    nargs = 1


class Covariance(sp.Function):
    """Unevaluated covariance of two random expressions."""

    nargs = 2


class RawMoment(sp.Function):
    """Unevaluated raw moment of a random expression."""

    nargs = 2


class CentralMoment(sp.Function):
    """Unevaluated central moment of a random expression."""

    nargs = 2


class Cumulant(sp.Function):
    """Unevaluated cumulant of a random expression."""

    nargs = 2


def _order(value, *, name="order") -> int:
    value = sp.sympify(value)
    if not isinstance(value, sp.Integer):
        raise TypeError(f"{name} must be an exact nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


_STATISTICAL_OPERATORS = (
    Expectation,
    Variance,
    Covariance,
    RawMoment,
    CentralMoment,
    Cumulant,
)


def _single_variable_groups_product(expression):
    groups: dict[RandomVariable, list[sp.Expr]] = {}
    for factor in sp.Mul.make_args(expression):
        variables = stochastic_variables(factor)
        if len(variables) != 1:
            return None
        variable = next(iter(variables))
        groups.setdefault(variable, []).append(factor)
    return {variable: sp.Mul(*factors) for variable, factors in groups.items()}


def expectation(expression, *, assumptions=None):
    """Normalize the expectation of a symbolic random expression.

    Linearity is unconditional. Products factor only when the supplied
    statistical assumptions certify the required independent relation.
    """
    expression = sp.sympify(expression)
    context = _context(assumptions)
    if (
        expression.func.__name__ == "ConditionalExpectation"
        and len(expression.args) == 2
    ):
        return expectation(expression.args[0], assumptions=context)
    if not is_random_expression(expression):
        return expression
    if expression.func == sp.conjugate and len(expression.args) == 1:
        return sp.conjugate(expectation(expression.args[0], assumptions=context))
    if isinstance(expression, sp.Add):
        return sp.Add(
            *(expectation(term, assumptions=context) for term in expression.args)
        )
    if isinstance(expression, sp.Mul):
        coefficient, random_part = _split_deterministic_product(expression)
        if coefficient != 1:
            return sp.simplify(
                coefficient * expectation(random_part, assumptions=context)
            )
        groups = _single_variable_groups_product(random_part)
        if groups and len(groups) > 1 and _certified_independent(groups, context):
            return sp.Mul(
                *(expectation(group, assumptions=context) for group in groups.values())
            )
    if isinstance(expression, sp.Pow):
        base, exponent = expression.as_base_exp()
        if (
            isinstance(base, RandomVariable)
            and exponent.is_Integer is True
            and exponent.is_nonnegative is True
        ):
            return raw_moment(base, int(exponent), assumptions=context)
    if isinstance(expression, RandomVariable):
        if expression.distribution is not None:
            from .functionals import mean as distribution_mean

            return sp.simplify(distribution_mean(expression.distribution))
        return Expectation(expression)
    return Expectation(expression)


def covariance(left, right, *, assumptions=None):
    """Normalize covariance using bilinearity and certified relationships."""
    left = sp.sympify(left)
    right = sp.sympify(right)
    context = _context(assumptions)
    if not is_random_expression(left) or not is_random_expression(right):
        return sp.S.Zero
    if isinstance(left, sp.Add):
        return sp.Add(
            *(covariance(term, right, assumptions=context) for term in left.args)
        )
    if isinstance(right, sp.Add):
        return sp.Add(
            *(covariance(left, term, assumptions=context) for term in right.args)
        )
    left_coefficient, left_random = _split_deterministic_product(left)
    right_coefficient, right_random = _split_deterministic_product(right)
    if (left_coefficient != 1 or right_coefficient != 1) and (
        left_coefficient.is_real is True and right_coefficient.is_real is True
    ):
        return sp.simplify(
            left_coefficient
            * right_coefficient
            * covariance(left_random, right_random, assumptions=context)
        )
    if left == right:
        return variance(left, assumptions=context)
    if isinstance(left, RandomVariable) and isinstance(right, RandomVariable):
        if uncorrelated(left, right, assumptions=context) is True:
            return sp.S.Zero
        if independent(left, right, assumptions=context) is True:
            return sp.S.Zero
    elif _disjoint_expressions_independent(left, right, context):
        return sp.S.Zero
    first, second = sorted((left, right), key=sp.default_sort_key)
    return Covariance(first, second)


def variance(expression, *, assumptions=None):
    """Normalize variance of a symbolic random expression."""
    expression = sp.sympify(expression)
    context = _context(assumptions)
    if not is_random_expression(expression):
        return sp.S.Zero
    if isinstance(expression, sp.Add):
        terms = tuple(term for term in expression.args if is_random_expression(term))
        diagonal = sp.Add(*(variance(term, assumptions=context) for term in terms))
        cross = sp.Add(
            *(
                2 * covariance(left, right, assumptions=context)
                for left, right in combinations(terms, 2)
            )
        )
        return sp.expand(diagonal + cross)
    if isinstance(expression, sp.Mul):
        coefficient, random_part = _split_deterministic_product(expression)
        if coefficient != 1 and coefficient.is_real is True:
            return sp.simplify(
                coefficient**2 * variance(random_part, assumptions=context)
            )
    if isinstance(expression, RandomVariable) and expression.distribution is not None:
        from .functionals import variance as distribution_variance

        return sp.simplify(distribution_variance(expression.distribution))
    return Variance(expression)


def raw_moment(expression, order, *, assumptions=None):
    """Return the symbolic raw moment ``E[X**order]``."""
    expression = sp.sympify(expression)
    degree = _order(order)
    if degree == 0:
        return sp.S.One
    if degree == 1:
        return expectation(expression, assumptions=assumptions)
    if not is_random_expression(expression):
        return sp.expand(expression**degree)
    if isinstance(expression, RandomVariable):
        if expression.distribution is not None:
            from .functionals import raw_moment as distribution_raw_moment

            return sp.simplify(distribution_raw_moment(expression.distribution, degree))
        return RawMoment(expression, sp.Integer(degree))
    expanded = sp.expand(expression**degree)
    return sp.expand(expectation(expanded, assumptions=assumptions))


def moment(expression, order, *, central=False, assumptions=None):
    """Return a raw or central symbolic moment of a random expression."""
    if central:
        return central_moment(expression, order, assumptions=assumptions)
    return raw_moment(expression, order, assumptions=assumptions)


def central_moment(expression, order, *, assumptions=None):
    """Return the symbolic central moment of a random expression."""
    expression = sp.sympify(expression)
    degree = _order(order)
    if degree == 0:
        return sp.S.One
    if degree == 1 or not is_random_expression(expression):
        return sp.S.Zero
    if degree == 2:
        return variance(expression, assumptions=assumptions)
    if isinstance(expression, sp.Add):
        deterministic = sp.Add(
            *(term for term in expression.args if not is_random_expression(term))
        )
        random_part = sp.simplify(expression - deterministic)
        if deterministic != 0:
            return central_moment(random_part, degree, assumptions=assumptions)
    if isinstance(expression, sp.Mul):
        coefficient, random_part = _split_deterministic_product(expression)
        if coefficient != 1 and coefficient.is_real is True:
            return sp.simplify(
                coefficient**degree
                * central_moment(random_part, degree, assumptions=assumptions)
            )
    if isinstance(expression, RandomVariable):
        if expression.distribution is not None:
            from .functionals import central_moment as distribution_central_moment

            return sp.simplify(
                distribution_central_moment(expression.distribution, degree)
            )
        return CentralMoment(expression, sp.Integer(degree))
    return CentralMoment(expression, sp.Integer(degree))


def mixed_moment(*expressions, orders=None, assumptions=None):
    """Return a mixed raw moment of one or more random expressions."""
    if not expressions:
        raise ValueError("mixed_moment requires at least one expression")
    if orders is None:
        orders = (1,) * len(expressions)
    orders = tuple(orders)
    if len(orders) != len(expressions):
        raise ValueError("orders must match the number of expressions")
    powers = [
        sp.sympify(expr) ** _order(order)
        for expr, order in zip(expressions, orders, strict=True)
    ]
    return expectation(sp.Mul(*powers), assumptions=assumptions)


def cumulant(expression, order, *, assumptions=None):
    """Normalize a symbolic cumulant, including independent-sum additivity."""
    expression = sp.sympify(expression)
    degree = _order(order)
    context = _context(assumptions)
    if degree == 0:
        return sp.S.Zero
    if degree == 1:
        return expectation(expression, assumptions=context)
    if degree == 2:
        return variance(expression, assumptions=context)
    if not is_random_expression(expression):
        return sp.S.Zero
    if isinstance(expression, sp.Add) and _independent_addends(
        expression.args, context
    ):
        return sp.Add(
            *(cumulant(term, degree, assumptions=context) for term in expression.args)
        )
    if isinstance(expression, sp.Mul):
        coefficient, random_part = _split_deterministic_product(expression)
        if coefficient != 1 and coefficient.is_real is True:
            return sp.simplify(
                coefficient**degree * cumulant(random_part, degree, assumptions=context)
            )
    if isinstance(expression, RandomVariable):
        if expression.distribution is not None:
            from .functionals import cumulant as distribution_cumulant

            return sp.simplify(distribution_cumulant(expression.distribution, degree))
        return Cumulant(expression, sp.Integer(degree))
    return Cumulant(expression, sp.Integer(degree))


__all__ = [
    "CentralMoment",
    "Covariance",
    "Cumulant",
    "Expectation",
    "RawMoment",
    "Variance",
    "central_moment",
    "covariance",
    "cumulant",
    "expectation",
    "is_random_expression",
    "mixed_moment",
    "moment",
    "raw_moment",
    "stochastic_variables",
    "variance",
]
