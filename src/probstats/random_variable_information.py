"""Information-theoretic algebra for symbolic random variables."""

from __future__ import annotations

import sympy as sp

from ._random_variable_semantics import (
    certified_independent,
    normalize_assumptions,
    stochastic_variables,
)
from ._transformation_certification import certify_random_expression_bijection
from .random_variables import RandomVariable


def _mutual_information_under_bijections(left, right, *, assumptions=None):
    """Apply mutual-information invariance after independent bijection certification."""
    left_source = certify_random_expression_bijection(left)
    right_source = certify_random_expression_bijection(right)
    if left_source is None or right_source is None:
        return None
    if left_source == left and right_source == right:
        return None
    return mutual_information(left_source, right_source, assumptions=assumptions)


class InformationEntropy(sp.Function):
    nargs = 1


class JointEntropy(sp.Function):
    nargs = None


class ConditionalEntropy(sp.Function):
    nargs = 2


class MutualInformation(sp.Function):
    nargs = 2


def random_entropy(expression, *, assumptions=None):
    """Return entropy of a random variable when its law is known, else formally."""
    expression = sp.sympify(expression)
    normalize_assumptions(assumptions)
    if isinstance(expression, RandomVariable) and expression.distribution is not None:
        from .functionals import entropy

        return sp.simplify(entropy(expression.distribution))
    return InformationEntropy(expression)


def joint_entropy(*expressions, assumptions=None):
    """Return joint entropy, simplifying independent collections additively."""
    if len(expressions) < 2:
        raise ValueError("joint_entropy requires at least two random expressions")
    expressions = tuple(map(sp.sympify, expressions))
    context = normalize_assumptions(assumptions)
    variables = frozenset().union(*(stochastic_variables(expr) for expr in expressions))
    disjoint = sum(len(stochastic_variables(expr)) for expr in expressions) == len(
        variables
    )
    if disjoint and certified_independent(variables, context):
        return sp.Add(
            *(random_entropy(expr, assumptions=context) for expr in expressions)
        )
    return JointEntropy(*expressions)


def conditional_entropy(expression, *, given, assumptions=None):
    """Return conditional entropy with independence simplification."""
    expression = sp.sympify(expression)
    given = sp.sympify(given)
    context = normalize_assumptions(assumptions)
    left_vars = stochastic_variables(expression)
    right_vars = stochastic_variables(given)
    if left_vars and right_vars and not left_vars.intersection(right_vars):
        from ._random_variable_semantics import certified_independent_collections

        if certified_independent_collections(left_vars, right_vars, context):
            return random_entropy(expression, assumptions=context)
    return ConditionalEntropy(expression, given)


def mutual_information(left, right, *, assumptions=None):
    """Return symbolic mutual information with independence and bijection simplification."""
    left = sp.sympify(left)
    right = sp.sympify(right)
    context = normalize_assumptions(assumptions)
    if left != right:
        invariant = _mutual_information_under_bijections(
            left, right, assumptions=context
        )
        if invariant is not None and (invariant.args != (left, right)):
            return invariant
    left_vars = stochastic_variables(left)
    right_vars = stochastic_variables(right)
    if left_vars and right_vars and not left_vars.intersection(right_vars):
        from ._random_variable_semantics import certified_independent_collections

        if certified_independent_collections(left_vars, right_vars, context):
            return sp.S.Zero
    first, second = sorted((left, right), key=sp.default_sort_key)
    return MutualInformation(first, second)


def entropy_chain_rule(*expressions, assumptions=None):
    """Return the entropy chain-rule decomposition for an ordered tuple."""
    if len(expressions) < 2:
        raise ValueError("entropy_chain_rule requires at least two expressions")
    context = normalize_assumptions(assumptions)
    result = random_entropy(expressions[0], assumptions=context)
    previous = expressions[0]
    for expression in expressions[1:]:
        result += conditional_entropy(expression, given=previous, assumptions=context)
        previous = sp.Tuple(previous, expression)
    return result


def mutual_information_chain_rule(left, *right, assumptions=None):
    """Return ``I(X;Y1)+I(X;Y2|Y1)+...`` as a symbolic chain rule."""
    if len(right) < 2:
        raise ValueError(
            "mutual_information_chain_rule requires at least two right expressions"
        )
    context = normalize_assumptions(assumptions)
    result = mutual_information(left, right[0], assumptions=context)
    conditioned = right[0]
    for expression in right[1:]:
        result += ConditionalMutualInformation(
            sp.sympify(left), sp.sympify(expression), sp.sympify(conditioned)
        )
        conditioned = sp.Tuple(conditioned, expression)
    return result


class ConditionalMutualInformation(sp.Function):
    nargs = 3


def product_kl_divergence(pairs, **kwargs):
    """Apply KL additivity to a finite product of independent component laws."""
    from .information import kl_divergence

    pairs = tuple(pairs)
    if not pairs:
        return sp.S.Zero
    return sp.simplify(sum(kl_divergence(p, q, **kwargs) for p, q in pairs))


__all__ = [
    "ConditionalEntropy",
    "ConditionalMutualInformation",
    "InformationEntropy",
    "JointEntropy",
    "MutualInformation",
    "conditional_entropy",
    "entropy_chain_rule",
    "joint_entropy",
    "mutual_information",
    "mutual_information_chain_rule",
    "product_kl_divergence",
    "random_entropy",
]
