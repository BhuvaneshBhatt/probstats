"""Complex-valued random-variable expectation, covariance, and variance algebra."""

from __future__ import annotations

import sympy as sp

from ._random_variable_semantics import (
    disjoint_expressions_independent as _disjoint_expressions_independent,
)
from ._random_variable_semantics import (
    is_random_expression,
)
from ._random_variable_semantics import (
    normalize_assumptions as _context,
)
from ._random_variable_semantics import (
    split_deterministic_product as _split_deterministic_product,
)


class ComplexCovariance(sp.Function):
    """Unevaluated Hermitian covariance ``E[(X-EX) conjugate(Y-EY)]``."""

    nargs = 2


class ComplexVariance(sp.Function):
    """Unevaluated complex variance ``E[abs(X-EX)**2]``."""

    nargs = 1


class PseudoCovariance(sp.Function):
    """Unevaluated pseudo-covariance ``E[(X-EX)(Y-EY)]``."""

    nargs = 2


def complex_covariance(left, right, *, assumptions=None):
    """Normalize Hermitian covariance for complex-valued random expressions."""
    left = sp.sympify(left)
    right = sp.sympify(right)
    context = _context(assumptions)
    if not is_random_expression(left) or not is_random_expression(right):
        return sp.S.Zero
    if isinstance(left, sp.Add):
        return sp.Add(
            *(
                complex_covariance(term, right, assumptions=context)
                for term in left.args
            )
        )
    if isinstance(right, sp.Add):
        return sp.Add(
            *(
                complex_covariance(left, term, assumptions=context)
                for term in right.args
            )
        )
    left_coefficient, left_random = _split_deterministic_product(left)
    right_coefficient, right_random = _split_deterministic_product(right)
    if left_coefficient != 1 or right_coefficient != 1:
        return sp.simplify(
            left_coefficient
            * sp.conjugate(right_coefficient)
            * complex_covariance(left_random, right_random, assumptions=context)
        )
    if left == right:
        return complex_variance(left, assumptions=context)
    if _disjoint_expressions_independent(left, right, context):
        return sp.S.Zero
    if sp.default_sort_key(left) > sp.default_sort_key(right):
        return sp.conjugate(complex_covariance(right, left, assumptions=context))
    return ComplexCovariance(left, right)


def complex_variance(expression, *, assumptions=None):
    """Normalize the nonnegative variance of a complex-valued random expression."""
    expression = sp.sympify(expression)
    context = _context(assumptions)
    if not is_random_expression(expression):
        return sp.S.Zero
    if isinstance(expression, sp.Add):
        terms = tuple(term for term in expression.args if is_random_expression(term))
        diagonal = sp.Add(
            *(complex_variance(term, assumptions=context) for term in terms)
        )
        cross = sp.Add(
            *(
                complex_covariance(left, right, assumptions=context)
                + complex_covariance(right, left, assumptions=context)
                for index, left in enumerate(terms)
                for right in terms[index + 1 :]
            )
        )
        return sp.expand(diagonal + cross)
    if isinstance(expression, sp.Mul):
        coefficient, random_part = _split_deterministic_product(expression)
        if coefficient != 1:
            return sp.simplify(
                coefficient
                * sp.conjugate(coefficient)
                * complex_variance(random_part, assumptions=context)
            )
    return ComplexVariance(expression)


def pseudo_covariance(left, right=None, *, assumptions=None):
    """Normalize pseudo-covariance for complex-valued random expressions."""
    left = sp.sympify(left)
    right = left if right is None else sp.sympify(right)
    context = _context(assumptions)
    if not is_random_expression(left) or not is_random_expression(right):
        return sp.S.Zero
    if isinstance(left, sp.Add):
        return sp.Add(
            *(pseudo_covariance(term, right, assumptions=context) for term in left.args)
        )
    if isinstance(right, sp.Add):
        return sp.Add(
            *(pseudo_covariance(left, term, assumptions=context) for term in right.args)
        )
    left_coefficient, left_random = _split_deterministic_product(left)
    right_coefficient, right_random = _split_deterministic_product(right)
    if left_coefficient != 1 or right_coefficient != 1:
        return sp.simplify(
            left_coefficient
            * right_coefficient
            * pseudo_covariance(left_random, right_random, assumptions=context)
        )
    if _disjoint_expressions_independent(left, right, context):
        return sp.S.Zero
    first, second = sorted((left, right), key=sp.default_sort_key)
    return PseudoCovariance(first, second)


__all__ = [
    "ComplexCovariance",
    "ComplexVariance",
    "PseudoCovariance",
    "complex_covariance",
    "complex_variance",
    "pseudo_covariance",
]
