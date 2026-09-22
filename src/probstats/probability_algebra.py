"""Exact probability identities over first-class events."""

from __future__ import annotations

from itertools import combinations

import sympy as sp

from ._symbolic_predicates import TruthValue, certified_equal
from .events import event_intersection, event_union
from .functionals import probability


def conditional_probability(dist, event, *, given, variable=None):
    """Return ``P(event | given)`` from the defining ratio."""
    denominator = probability(dist, given, variable=variable)
    if denominator == 0:
        raise ZeroDivisionError("conditional probability requires P(given) != 0")
    numerator = probability(dist, event_intersection(event, given), variable=variable)
    return sp.simplify(numerator / denominator)


def bayes_probability(dist, event, *, given, variable=None):
    """Evaluate Bayes' rule for two events under ``dist``."""
    prior = probability(dist, event, variable=variable)
    evidence = probability(dist, given, variable=variable)
    if evidence == 0:
        raise ZeroDivisionError("Bayes' rule requires P(given) != 0")
    likelihood = conditional_probability(dist, given, given=event, variable=variable)
    return sp.simplify(likelihood * prior / evidence)


def total_probability(dist, event, *, partition, variable=None):
    """Apply the law of total probability over a supplied finite partition."""
    parts = tuple(partition)
    if not parts:
        raise ValueError("partition must contain at least one event")
    total = sp.S.Zero
    for part in parts:
        p_part = probability(dist, part, variable=variable)
        if p_part == 0:
            continue
        total += (
            conditional_probability(dist, event, given=part, variable=variable) * p_part
        )
    return sp.simplify(total)


def inclusion_exclusion(dist, *events, variable=None):
    """Return the finite inclusion-exclusion probability of an event union."""
    if not events:
        return sp.S.Zero
    total = sp.S.Zero
    for size in range(1, len(events) + 1):
        sign = 1 if size % 2 else -1
        for subset in combinations(events, size):
            total += sign * probability(
                dist,
                event_intersection(*subset) if size > 1 else subset[0],
                variable=variable,
            )
    return sp.simplify(total)


def events_independent(dist, left, right, *, variable=None):
    """Return exact event-independence status when the probabilities decide it."""
    joint = probability(dist, event_intersection(left, right), variable=variable)
    product = probability(dist, left, variable=variable) * probability(
        dist, right, variable=variable
    )
    relation = certified_equal(joint, product)
    if relation is TruthValue.TRUE:
        return True
    if relation is TruthValue.FALSE:
        return False
    return None


def union_probability(dist, *events, variable=None):
    """Return ``P(union(events))``; exact evaluation is delegated to probability()."""
    return probability(dist, event_union(*events), variable=variable)


__all__ = [
    "bayes_probability",
    "conditional_probability",
    "events_independent",
    "inclusion_exclusion",
    "total_probability",
    "union_probability",
]
