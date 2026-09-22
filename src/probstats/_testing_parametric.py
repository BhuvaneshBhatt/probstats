"""Internal implementation for testing parametric."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ._special import chi2_sf, normal_cdf, t_cdf
from ._testing_core import (
    HypothesisTestResult,
    _binomial_counts,
    _proportion_counts,
    _tail,
    _x,
)
from .results import HypothesisResult


def one_sample_t_test(data, mu=0, alternative="two-sided"):
    """Perform a one-sample Student t test for a population mean."""
    x = _x(data)
    n = x.size
    if n < 2:
        raise ValueError("t-test requires at least two observations")
    se = float(x.std(ddof=1) / math.sqrt(n))
    delta = float(x.mean()) - mu
    if se == 0:
        stat = 0.0 if delta == 0 else math.copysign(math.inf, delta)
        pvalue = 1.0 if delta == 0 else 0.0
    else:
        stat = delta / se
        pvalue = _tail(lambda z: t_cdf(z, n - 1), stat, alternative)
    return HypothesisTestResult(
        stat,
        pvalue,
        alternative,
        "one-sample t-test",
        (n,),
        mu,
        n - 1,
        float(x.mean()),
        se,
    )


def welch_t_test(x, y, alternative="two-sided"):
    """Perform Welch’s unequal-variance two-sample t test."""
    a = _x(x)
    b = _x(y)
    n1 = a.size
    n2 = b.size
    if min(n1, n2) < 2:
        raise ValueError("Welch t-test requires at least two observations per sample")
    v1 = float(a.var(ddof=1))
    v2 = float(b.var(ddof=1))
    se = math.sqrt(v1 / n1 + v2 / n2)
    stat = (float(a.mean()) - float(b.mean())) / se
    df = (v1 / n1 + v2 / n2) ** 2 / (
        (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    )
    return HypothesisTestResult(
        stat,
        _tail(lambda z: t_cdf(z, df), stat, alternative),
        alternative,
        "Welch two-sample t-test",
        (n1, n2),
        0.0,
        df,
        float(a.mean() - b.mean()),
        se,
    )


def one_sample_z_test(data, mu=0, sigma=None, alternative="two-sided"):
    """Perform a one-sample z test for a population mean."""
    x = _x(data)
    n = x.size
    if sigma is None:
        sigma = float(x.std(ddof=0))
    se = sigma / math.sqrt(n)
    stat = (float(x.mean()) - mu) / se
    return HypothesisTestResult(
        stat,
        _tail(normal_cdf, stat, alternative),
        alternative,
        "one-sample z-test",
        (n,),
        mu,
        None,
        float(x.mean()),
        se,
    )


def one_proportion_z_test(successes, n, p0=0.5, alternative="two-sided"):
    """Perform a large-sample z test for one binomial proportion."""
    successes, n = _binomial_counts(successes, n)
    if not 0 < p0 < 1:
        raise ValueError("p0 must lie strictly between 0 and 1")
    phat = successes / n
    se = math.sqrt(p0 * (1 - p0) / n)
    stat = (phat - p0) / se
    return HypothesisTestResult(
        stat,
        _tail(normal_cdf, stat, alternative),
        alternative,
        "one-proportion z-test",
        (n,),
        p0,
        None,
        phat,
        se,
    )


def exact_binomial_test(successes, n, p=0.5, alternative="two-sided"):
    """Perform an exact test for a binomial success probability."""
    successes, n = _binomial_counts(successes, n)
    if not 0 <= p <= 1:
        raise ValueError("p must lie in [0, 1]")
    probs = [math.comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(n + 1)]
    obs = probs[successes]
    if alternative == "less":
        pv = sum(probs[: successes + 1])
    elif alternative == "greater":
        pv = sum(probs[successes:])
    elif alternative == "two-sided":
        pv = sum(q for q in probs if q <= obs + 1e-15)
    else:
        raise ValueError("invalid alternative")
    return HypothesisTestResult(
        float(successes),
        min(1.0, float(pv)),
        alternative,
        "exact binomial test",
        (n,),
        p,
        None,
        successes / n,
        None,
    )


def chi_square_goodness_of_fit(observed, expected=None):
    """Perform a Pearson chi-square goodness-of-fit test."""
    o = _x(observed)
    n = o.size
    if np.any(o < 0) or o.sum() <= 0:
        raise ValueError("observed counts must be nonnegative with positive total")
    e = np.full(n, o.sum() / n) if expected is None else _x(expected)
    if e.size != n or np.any(e <= 0):
        raise ValueError("expected counts must match and be positive")
    if not np.isclose(o.sum(), e.sum()):
        e = e * (o.sum() / e.sum())
    stat = float(np.sum((o - e) ** 2 / e))
    df = n - 1
    pv = chi2_sf(stat, df)
    return HypothesisTestResult(
        stat, pv, "greater", "chi-square goodness-of-fit", (float(o.sum()),), None, df
    )


def pearson_correlation_test(x, y, alternative="two-sided"):
    """Test the null hypothesis of zero Pearson correlation."""
    a = _x(x)
    b = _x(y)
    if a.size != b.size or a.size < 3:
        raise ValueError("correlation test requires equal samples of length >= 3")
    r = float(np.corrcoef(a, b)[0, 1])
    df = a.size - 2
    if abs(r) >= 1:
        stat = math.copysign(math.inf, r)
        pv = 0.0
    else:
        stat = r * math.sqrt(df / (1 - r * r))
        pv = _tail(lambda z: t_cdf(z, df), stat, alternative)
    return HypothesisTestResult(
        stat, pv, alternative, "Pearson correlation test", (a.size,), 0.0, df, r, None
    )


def paired_t_test(x, y, alternative="two-sided"):
    """Paired Student-t test, implemented as a one-sample test of differences."""
    a = _x(x)
    b = _x(y)
    if a.size != b.size:
        raise ValueError("paired samples must have equal lengths")
    result = one_sample_t_test(a - b, 0.0, alternative)
    return HypothesisTestResult(
        result.statistic,
        result.pvalue,
        alternative,
        "paired t-test",
        (a.size, a.size),
        0.0,
        result.degrees_of_freedom,
        float(np.mean(a - b)),
        result.standard_error,
    )


def two_proportion_z_test(successes1, n1, successes2, n2, alternative="two-sided"):
    """Pooled two-sample z test for equality of proportions."""
    successes1, n1, successes2, n2 = _proportion_counts(successes1, n1, successes2, n2)
    p1 = successes1 / n1
    p2 = successes2 / n2
    pooled = (successes1 + successes2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        stat = 0.0 if p1 == p2 else math.copysign(math.inf, p1 - p2)
    else:
        stat = (p1 - p2) / se
    return HypothesisTestResult(
        stat,
        _tail(normal_cdf, stat, alternative) if math.isfinite(stat) else 0.0,
        alternative,
        "two-proportion z-test",
        (n1, n2),
        0.0,
        None,
        p1 - p2,
        se,
    )


def chi_square_independence(table):
    """Pearson chi-square test of independence for a contingency table."""
    observed = np.asarray(table, dtype=float)
    if (
        observed.ndim != 2
        or min(observed.shape) < 2
        or not np.all(np.isfinite(observed))
        or np.any(observed < 0)
    ):
        raise ValueError(
            "table must be a finite nonnegative 2D array with at least 2x2 cells"
        )
    total = observed.sum()
    if total <= 0:
        raise ValueError("contingency table total must be positive")
    row = observed.sum(axis=1, keepdims=True)
    col = observed.sum(axis=0, keepdims=True)
    expected = row @ col / total
    if np.any(expected <= 0):
        raise ValueError("all expected cell counts must be positive")
    statistic = float(np.sum((observed - expected) ** 2 / expected))
    df = (observed.shape[0] - 1) * (observed.shape[1] - 1)
    pvalue = chi2_sf(statistic, df)
    return HypothesisTestResult(
        statistic,
        pvalue,
        "greater",
        "chi-square test of independence",
        (float(total),),
        None,
        df,
    )


@dataclass(frozen=True)
class EquivalenceTestResult(HypothesisResult):
    """Result of an equivalence test based on two one-sided tests (TOST)."""

    estimate: float
    lower_bound: float
    upper_bound: float
    lower_statistic: float
    upper_statistic: float
    lower_pvalue: float
    upper_pvalue: float
    standard_error: float
    degrees_of_freedom: float
    method: str
    sample_sizes: tuple[int, ...]

    @property
    def pvalue(self):
        return max(self.lower_pvalue, self.upper_pvalue)

    @property
    def equivalent_05(self):
        return self.pvalue < 0.05
