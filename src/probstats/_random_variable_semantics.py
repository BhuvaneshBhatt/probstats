"""Shared semantics for symbolic random-variable algebra.

This internal module owns context normalization, stochastic dependency inspection,
measurability, collection independence, and affine decomposition.  Keeping these
rules here prevents the moment, distribution, transform, and conditional layers
from drifting apart.
"""

from __future__ import annotations

from itertools import combinations

import sympy as sp

from .assumptions import (
    NegatedStatisticalRelation,
    StatisticalAssumptions,
    StatisticalRelation,
    independent,
    independent_collections,
    mutually_independent,
)
from .random_variables import RandomVariable


def normalize_assumptions(value=None) -> StatisticalAssumptions:
    if value is None:
        return StatisticalAssumptions()
    if isinstance(value, StatisticalAssumptions):
        return value
    if isinstance(value, (StatisticalRelation, NegatedStatisticalRelation)):
        return StatisticalAssumptions(value)
    try:
        return StatisticalAssumptions(*value)
    except TypeError as exc:
        raise TypeError(
            "assumptions must be a statistical relation, an assumptions context, "
            "or an iterable of relations"
        ) from exc


def stochastic_variables(expression) -> frozenset[RandomVariable]:
    """Return variables on which ``expression`` remains stochastic."""
    expression = sp.sympify(expression)
    if isinstance(expression, RandomVariable):
        return frozenset((expression,))
    # Conditional summaries are random only through their conditioning variables.
    if (
        expression.func.__name__
        in {
            "ConditionalExpectation",
            "ConditionalVariance",
            "ConditionalCovariance",
            "ConditionalMoment",
        }
        and len(expression.args) >= 2
    ):
        given = expression.args[-1]
        if isinstance(given, sp.Tuple):
            return frozenset(v for v in given if isinstance(v, RandomVariable))
    # Ordinary statistical summaries are deterministic quantities.
    if expression.func.__name__ in {
        "Expectation",
        "Variance",
        "Covariance",
        "RawMoment",
        "CentralMoment",
        "Cumulant",
        "ComplexVariance",
        "ComplexCovariance",
        "PseudoCovariance",
        "Correlation",
        "Skewness",
        "Kurtosis",
        "StandardizedMoment",
        "CoefficientOfVariation",
        "FactorialMoment",
        "InformationEntropy",
        "MutualInformation",
    }:
        return frozenset()
    variables = frozenset()
    for argument in expression.args:
        variables |= stochastic_variables(argument)
    return variables


def is_random_expression(expression) -> bool:
    return bool(stochastic_variables(expression))


def split_deterministic_product(expression):
    deterministic = []
    random = []
    for factor in sp.Mul.make_args(sp.sympify(expression)):
        (random if is_random_expression(factor) else deterministic).append(factor)
    return sp.Mul(*deterministic), sp.Mul(*random)


def split_measurable_product(expression, given):
    given = frozenset(given)
    measurable = []
    remainder = []
    for factor in sp.Mul.make_args(sp.sympify(expression)):
        variables = stochastic_variables(factor)
        if not variables or variables.issubset(given):
            measurable.append(factor)
        else:
            remainder.append(factor)
    return sp.Mul(*measurable), sp.Mul(*remainder)


def is_measurable_with_respect_to(expression, given) -> bool:
    return stochastic_variables(expression).issubset(frozenset(given))


def certified_independent(variables, context: StatisticalAssumptions) -> bool:
    variables = tuple(sorted(set(variables), key=sp.default_sort_key))
    if len(variables) < 2:
        return True
    if len(variables) == 2:
        return independent(*variables, assumptions=context) is True
    return mutually_independent(*variables, assumptions=context) is True


def certified_independent_collections(left, right, context) -> bool:
    left = frozenset(left)
    right = frozenset(right)
    if not left or not right or left.intersection(right):
        return False
    if independent_collections(left, right, assumptions=context) is True:
        return True
    # Mutual independence of the union is sufficient for independence of any
    # two disjoint subcollections.  Pairwise independence is not.
    return certified_independent(left | right, context)


def disjoint_expressions_independent(left, right, context) -> bool:
    return certified_independent_collections(
        stochastic_variables(left), stochastic_variables(right), context
    )


def independent_addends(terms, context) -> bool:
    random_terms = tuple(term for term in terms if is_random_expression(term))
    if len(random_terms) < 2:
        return True
    variable_sets = [stochastic_variables(term) for term in random_terms]
    if any(left.intersection(right) for left, right in combinations(variable_sets, 2)):
        return False
    union = frozenset().union(*variable_sets)
    return certified_independent(union, context)


def affine_in_one_variable(expression, variable=None):
    expression = sp.sympify(expression)
    variables = stochastic_variables(expression)
    if variable is None:
        if len(variables) != 1:
            return None
        variable = next(iter(variables))
    if variable not in variables:
        return None
    coefficient = sp.simplify(sp.diff(expression, variable))
    if coefficient.has(variable):
        return None
    offset = sp.simplify(expression - coefficient * variable)
    if offset.has(variable):
        return None
    return variable, coefficient, offset


__all__ = [
    "affine_in_one_variable",
    "certified_independent",
    "certified_independent_collections",
    "disjoint_expressions_independent",
    "independent_addends",
    "is_measurable_with_respect_to",
    "is_random_expression",
    "normalize_assumptions",
    "split_deterministic_product",
    "split_measurable_product",
    "stochastic_variables",
]
