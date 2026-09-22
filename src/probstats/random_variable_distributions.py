"""Distribution rules and closure registry for symbolic random expressions."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from ._random_variable_semantics import (
    affine_in_one_variable,
    certified_independent,
    normalize_assumptions,
    stochastic_variables,
)
from .distributions import Binomial, Gamma, Normal, Poisson
from .distributions.base import Distribution
from .random_variables import RandomVariable


@dataclass(frozen=True, slots=True)
class RandomExpressionLaw:
    expression: sp.Expr
    component_laws: tuple[tuple[RandomVariable, Distribution], ...]
    exact: bool = True


_AFFINE_CLOSURES = []
_SUM_CLOSURES = []


def register_affine_closure(distribution_type):
    """Register a process-global affine pushforward rule.

    Re-registering the same function is idempotent. Registry rules are tried in
    registration order; exceptions raised by a rule propagate to the caller.
    """

    def decorator(function):
        entry = (distribution_type, function)
        if entry not in _AFFINE_CLOSURES:
            _AFFINE_CLOSURES.append(entry)
        return function

    return decorator


def register_sum_closure(function):
    """Register a process-global independent-sum closure rule.

    Re-registering the same function is idempotent. Rules are tried in
    registration order and rule exceptions are not interpreted as non-matches.
    """
    if function not in _SUM_CLOSURES:
        _SUM_CLOSURES.append(function)
    return function


@register_affine_closure(Normal)
def _normal_affine(law, coefficient, offset):
    return Normal(
        sp.simplify(coefficient * law.mean + offset),
        sp.simplify(sp.Abs(coefficient) * law.sigma),
    )


@register_sum_closure
def _normal_sum(laws, coefficients, deterministic):
    if not all(isinstance(law, Normal) for law in laws):
        return None
    mean = deterministic + sum(
        (a * law.mean for a, law in zip(coefficients, laws, strict=True)), sp.S.Zero
    )
    var = sum(
        (a**2 * law.sigma**2 for a, law in zip(coefficients, laws, strict=True)),
        sp.S.Zero,
    )
    return Normal(sp.simplify(mean), sp.sqrt(sp.simplify(var)))


@register_sum_closure
def _poisson_sum(laws, coefficients, deterministic):
    if deterministic != 0 or not all(a == 1 for a in coefficients):
        return None
    if not all(isinstance(law, Poisson) for law in laws):
        return None
    return Poisson(sp.simplify(sum(law.rate for law in laws)))


@register_sum_closure
def _binomial_sum(laws, coefficients, deterministic):
    if deterministic != 0 or not all(a == 1 for a in coefficients):
        return None
    if not all(isinstance(law, Binomial) for law in laws):
        return None
    probabilities = {law.p for law in laws}
    if len(probabilities) != 1:
        return None
    return Binomial(sp.simplify(sum(law.n for law in laws)), next(iter(probabilities)))


@register_sum_closure
def _gamma_sum(laws, coefficients, deterministic):
    if deterministic != 0 or not all(a == 1 for a in coefficients):
        return None
    if not all(isinstance(law, Gamma) for law in laws):
        return None
    scales = {law.scale for law in laws}
    if len(scales) != 1:
        return None
    return Gamma(sp.simplify(sum(law.shape for law in laws)), next(iter(scales)))


def _apply_affine_closure(law, coefficient, offset):
    if coefficient == 1 and offset == 0:
        return law
    for distribution_type, function in _AFFINE_CLOSURES:
        if isinstance(law, distribution_type):
            result = function(law, coefficient, offset)
            if result is not None:
                return result
    return None


def _linear_additive_parts(expr, variables):
    deterministic = sp.Add(
        *(term for term in sp.Add.make_args(expr) if not stochastic_variables(term))
    )
    coefficients = []
    for variable in variables:
        variable_terms = sum(
            (
                term
                for term in sp.Add.make_args(expr)
                if variable in stochastic_variables(term)
            ),
            sp.S.Zero,
        )
        affine = affine_in_one_variable(variable_terms, variable)
        if affine is None:
            return None
        _, coefficient, offset = affine
        if offset != 0:
            deterministic += offset
        coefficients.append(coefficient)
    return tuple(coefficients), sp.simplify(deterministic)


def distribution(expression, *, assumptions=None):
    """Return the exact or symbolic law of a random expression when derivable."""
    expr = sp.sympify(expression)
    context = normalize_assumptions(assumptions)
    if isinstance(expr, RandomVariable):
        return expr.distribution
    variables = tuple(sorted(stochastic_variables(expr), key=sp.default_sort_key))
    if not variables:
        raise TypeError("distribution requires a random expression")
    laws = {variable: variable.distribution for variable in variables}
    if any(law is None for law in laws.values()):
        return None
    if len(variables) == 1:
        variable = variables[0]
        affine = affine_in_one_variable(expr, variable)
        if affine is not None:
            _, coefficient, offset = affine
            closed = _apply_affine_closure(laws[variable], coefficient, offset)
            if closed is not None:
                return closed
    if (
        isinstance(expr, sp.Add)
        and len(variables) >= 2
        and certified_independent(variables, context)
    ):
        parts = _linear_additive_parts(expr, variables)
        if parts is not None:
            coefficients, deterministic = parts
            ordered_laws = tuple(laws[variable] for variable in variables)
            for closure in _SUM_CLOSURES:
                result = closure(ordered_laws, coefficients, deterministic)
                if result is not None:
                    return result
    return RandomExpressionLaw(
        expr, tuple((variable, laws[variable]) for variable in variables)
    )


__all__ = [
    "RandomExpressionLaw",
    "distribution",
    "register_affine_closure",
    "register_sum_closure",
]
