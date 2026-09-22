"""Shared primitives for hypothesis tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._validation import integer
from .results import HypothesisResult


@dataclass(frozen=True)
class HypothesisTestResult(HypothesisResult):
    statistic: float
    pvalue: float
    alternative: str
    method: str
    sample_sizes: tuple[int | float, ...]
    null_value: float | None = None
    degrees_of_freedom: float | None = None
    estimate: float | None = None
    standard_error: float | None = None

    @property
    def reject_05(self):
        return self.pvalue < 0.05


def _x(data):
    x = np.asarray(data, dtype=float)
    if x.size == 0:
        raise ValueError("data must not be empty")
    if not np.all(np.isfinite(x)):
        raise ValueError("data must contain only finite values")
    return x.reshape(-1)


def _binomial_counts(successes, n):
    n = integer(n, name="n", minimum=1)
    successes = integer(successes, name="successes", minimum=0, maximum=n)
    return successes, n


def _proportion_counts(successes1, n1, successes2, n2):
    successes1, n1 = _binomial_counts(successes1, n1)
    successes2, n2 = _binomial_counts(successes2, n2)
    return successes1, n1, successes2, n2


def _tail(cdf, stat, alt):
    if alt == "less":
        return float(cdf(stat))
    if alt == "greater":
        return float(1 - cdf(stat))
    if alt == "two-sided":
        return float(2 * min(cdf(stat), 1 - cdf(stat)))
    raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")
