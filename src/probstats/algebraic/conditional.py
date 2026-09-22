"""Exact conditional inference for toric count models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import factorial

import sympy as sp

from ._validation import count_vector
from .fibers import Fiber
from .tables import ContingencyTable


def _base_weight(table: tuple[int, ...]) -> sp.Rational:
    denominator = 1
    for count in table:
        denominator *= factorial(count)
    return sp.Rational(1, denominator)


@dataclass(frozen=True, slots=True)
class ExactConditionalTestResult:
    """Result of exact inference after conditioning on sufficient statistics."""

    pvalue: sp.Rational
    observed_score: object
    observed_probability: sp.Rational
    fiber_size: int
    method: str
    exact: bool = True


def conditional_test(
    observed,
    model,
    *,
    statistic: str | Callable[[tuple[int, ...]], object] = "probability",
    alternative: str = "greater",
    max_candidates: int = 1_000_000,
) -> ExactConditionalTestResult:
    """Perform an exact test on the sufficient-statistic fiber of a toric model.

    The conditional mass is proportional to ``1 / prod(x_i!)``. With
    ``statistic="probability"`` the p-value sums all fiber points whose
    conditional mass is no greater than the observed mass, the standard
    probability-ordering rule for exact conditional tests.

    A callable statistic uses an exact lower or upper tail selected by
    ``alternative``.
    """
    raw_observed = (
        observed.flat_counts
        if isinstance(observed, ContingencyTable)
        else tuple(observed)
    )
    fiber: Fiber = model.fiber(raw_observed)
    vector = count_vector(raw_observed, width=fiber.width, name="observed")
    tables = fiber.enumerate(max_candidates=max_candidates)
    weights = tuple(_base_weight(table) for table in tables)
    total_weight = sum(weights, sp.Rational(0))
    observed_index = tables.index(vector)
    observed_weight = weights[observed_index]
    observed_probability = sp.cancel(observed_weight / total_weight)

    if statistic == "probability":
        tail_weight = sum(
            (weight for weight in weights if weight <= observed_weight),
            sp.Rational(0),
        )
        return ExactConditionalTestResult(
            sp.cancel(tail_weight / total_weight),
            observed_probability,
            observed_probability,
            len(tables),
            "probability",
        )
    if not callable(statistic):
        raise TypeError("statistic must be 'probability' or a callable")
    if alternative not in {"greater", "less"}:
        raise ValueError("alternative must be 'greater' or 'less'")
    scores = tuple(statistic(table) for table in tables)
    observed_score = scores[observed_index]
    if alternative == "greater":
        selected = (
            weight
            for score, weight in zip(scores, weights, strict=True)
            if score >= observed_score
        )
    else:
        selected = (
            weight
            for score, weight in zip(scores, weights, strict=True)
            if score <= observed_score
        )
    tail_weight = sum(selected, sp.Rational(0))
    return ExactConditionalTestResult(
        sp.cancel(tail_weight / total_weight),
        observed_score,
        observed_probability,
        len(tables),
        f"statistic-{alternative}",
    )


__all__ = ["ExactConditionalTestResult", "conditional_test"]
