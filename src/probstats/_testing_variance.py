"""Internal implementation for testing variance."""

from __future__ import annotations

import math

import numpy as np

from ._special import chi2_sf, f_cdf, normal_cdf, t_cdf
from ._testing_core import HypothesisTestResult, _proportion_counts, _tail, _x
from ._testing_parametric import EquivalenceTestResult, one_sample_t_test, welch_t_test


def one_sample_variance_test(data, variance, alternative="two-sided", *, mean=None):
    """Chi-square test for the variance of a Normal population."""
    x = _x(data).reshape(-1)
    if x.size < 2 or variance <= 0:
        raise ValueError(
            "at least two observations and positive null variance are required"
        )
    if mean is None:
        observed = float(x.var(ddof=1))
        df = x.size - 1
        statistic = df * observed / variance
    else:
        observed = float(np.mean((x - float(mean)) ** 2))
        df = x.size
        statistic = float(np.sum((x - float(mean)) ** 2) / variance)
    from ._special import chi2_cdf

    pvalue = _tail(lambda z: chi2_cdf(z, df), statistic, alternative)
    return HypothesisTestResult(
        statistic,
        pvalue,
        alternative,
        "one-sample chi-square variance test",
        (x.size,),
        variance,
        df,
        observed,
        None,
    )


def variance_ratio_test(x, y, alternative="two-sided"):
    """Classical F test for equality of two Normal-population variances."""
    a = _x(x).reshape(-1)
    b = _x(y).reshape(-1)
    if min(a.size, b.size) < 2:
        raise ValueError(
            "variance-ratio test requires at least two observations per sample"
        )
    v1 = float(a.var(ddof=1))
    v2 = float(b.var(ddof=1))
    if v2 == 0:
        if v1 == 0:
            statistic, pvalue = 1.0, 1.0
        else:
            statistic, pvalue = math.inf, 0.0
    else:
        statistic = v1 / v2

        def cdf(z):
            return f_cdf(z, a.size - 1, b.size - 1)

        pvalue = _tail(cdf, statistic, alternative)
    return HypothesisTestResult(
        statistic,
        pvalue,
        alternative,
        "two-sample F variance-ratio test",
        (a.size, b.size),
        1.0,
        None,
        statistic,
        None,
    )


def bartlett_test(*samples):
    """Bartlett test of equality of variances across two or more groups."""
    groups = [_x(sample).reshape(-1) for sample in samples]
    if len(groups) < 2 or min(group.size for group in groups) < 2:
        raise ValueError("Bartlett test requires at least two groups of size >= 2")
    dfs = np.array([group.size - 1 for group in groups], dtype=float)
    variances = np.array([group.var(ddof=1) for group in groups], dtype=float)
    if np.any(variances <= 0):
        raise ValueError("Bartlett test requires positive within-group variances")
    total_df = float(dfs.sum())
    pooled = float(np.dot(dfs, variances) / total_df)
    numerator = total_df * math.log(pooled) - float(np.dot(dfs, np.log(variances)))
    correction = 1 + (float(np.sum(1 / dfs)) - 1 / total_df) / (3 * (len(groups) - 1))
    statistic = numerator / correction
    df = len(groups) - 1
    return HypothesisTestResult(
        statistic,
        chi2_sf(statistic, df),
        "greater",
        "Bartlett test for equal variances",
        tuple(group.size for group in groups),
        None,
        df,
    )


def _one_way_anova_statistic(groups):
    sizes = np.array([g.size for g in groups], dtype=float)
    means = np.array([g.mean() for g in groups], dtype=float)
    grand = float(np.dot(sizes, means) / sizes.sum())
    ss_between = float(np.dot(sizes, (means - grand) ** 2))
    ss_within = float(sum(np.sum((g - g.mean()) ** 2) for g in groups))
    df1 = len(groups) - 1
    df2 = int(sizes.sum() - len(groups))
    if ss_within == 0:
        return (0.0 if ss_between == 0 else math.inf), df1, df2
    return (ss_between / df1) / (ss_within / df2), df1, df2


def levene_test(*samples, center="median"):
    """Levene variance-homogeneity test; ``center='median'`` gives Brown-Forsythe."""
    groups = [_x(sample).reshape(-1) for sample in samples]
    if len(groups) < 2 or min(group.size for group in groups) < 2:
        raise ValueError("Levene test requires at least two groups of size >= 2")
    if center == "mean":
        centers = [float(g.mean()) for g in groups]
        method = "Levene test (mean centered)"
    elif center == "median":
        centers = [float(np.median(g)) for g in groups]
        method = "Brown-Forsythe test (median centered)"
    else:
        raise ValueError("center must be 'mean' or 'median'")
    deviations = [np.abs(g - c) for g, c in zip(groups, centers, strict=True)]
    statistic, df1, df2 = _one_way_anova_statistic(deviations)
    pvalue = 0.0 if math.isinf(statistic) else 1 - f_cdf(statistic, df1, df2)
    return HypothesisTestResult(
        statistic,
        pvalue,
        "greater",
        method,
        tuple(group.size for group in groups),
        None,
        float(df2),
    )


def brown_forsythe_test(*samples):
    return levene_test(*samples, center="median")


def one_sample_tost(data, lower, upper, *, mu=0.0):
    """TOST equivalence test for a one-sample mean difference."""
    if not lower < upper:
        raise ValueError("lower equivalence bound must be below upper bound")
    x = _x(data).reshape(-1)
    if x.size < 2:
        raise ValueError("TOST requires at least two observations")
    estimate = float(x.mean()) - mu
    se = float(x.std(ddof=1) / math.sqrt(x.size))
    df = x.size - 1
    if se == 0:
        lower_stat = math.inf if estimate > lower else -math.inf
        upper_stat = -math.inf if estimate < upper else math.inf
    else:
        lower_stat = (estimate - lower) / se
        upper_stat = (estimate - upper) / se
    lower_p = 1 - t_cdf(lower_stat, df)
    upper_p = t_cdf(upper_stat, df)
    return EquivalenceTestResult(
        estimate,
        lower,
        upper,
        lower_stat,
        upper_stat,
        lower_p,
        upper_p,
        se,
        df,
        "one-sample TOST",
        (x.size,),
    )


def welch_tost(x, y, lower, upper):
    """Welch two-sample TOST for equivalence of means."""
    if not lower < upper:
        raise ValueError("lower equivalence bound must be below upper bound")
    a = _x(x).reshape(-1)
    b = _x(y).reshape(-1)
    if min(a.size, b.size) < 2:
        raise ValueError("Welch TOST requires at least two observations per group")
    v1 = float(a.var(ddof=1))
    v2 = float(b.var(ddof=1))
    se2 = v1 / a.size + v2 / b.size
    se = math.sqrt(se2)
    df = se2**2 / (
        (v1 / a.size) ** 2 / (a.size - 1) + (v2 / b.size) ** 2 / (b.size - 1)
    )
    estimate = float(a.mean() - b.mean())
    lower_stat = (estimate - lower) / se
    upper_stat = (estimate - upper) / se
    return EquivalenceTestResult(
        estimate,
        lower,
        upper,
        lower_stat,
        upper_stat,
        1 - t_cdf(lower_stat, df),
        t_cdf(upper_stat, df),
        se,
        df,
        "Welch two-sample TOST",
        (a.size, b.size),
    )


def two_proportion_tost(successes1, n1, successes2, n2, lower, upper):
    """Asymptotic TOST equivalence test for a difference of two proportions."""
    if not lower < upper:
        raise ValueError("lower equivalence bound must be below upper bound")
    successes1, n1, successes2, n2 = _proportion_counts(successes1, n1, successes2, n2)
    p1 = successes1 / n1
    p2 = successes2 / n2
    estimate = p1 - p2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    if se == 0:
        lower_stat = math.inf if estimate > lower else -math.inf
        upper_stat = -math.inf if estimate < upper else math.inf
    else:
        lower_stat = (estimate - lower) / se
        upper_stat = (estimate - upper) / se
    return EquivalenceTestResult(
        estimate,
        lower,
        upper,
        lower_stat,
        upper_stat,
        1 - normal_cdf(lower_stat),
        normal_cdf(upper_stat),
        se,
        math.inf,
        "two-proportion z TOST",
        (n1, n2),
    )


def proportion_noninferiority_z_test(
    successes1,
    n1,
    successes2,
    n2,
    *,
    margin=0.0,
    direction="greater",
):
    """One-sided noninferiority z test for a difference of two proportions."""
    if margin < 0 or direction not in {"greater", "less"}:
        raise ValueError("margin must be nonnegative and direction 'greater' or 'less'")
    successes1, n1, successes2, n2 = _proportion_counts(successes1, n1, successes2, n2)
    p1 = successes1 / n1
    p2 = successes2 / n2
    estimate = p1 - p2
    null = -margin if direction == "greater" else margin
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    if se == 0:
        statistic = (
            0.0 if estimate == null else math.copysign(math.inf, estimate - null)
        )
        pvalue = 1.0 if estimate == null else (0.0 if statistic > 0 else 1.0)
        if direction == "less" and estimate != null:
            pvalue = 0.0 if statistic < 0 else 1.0
    else:
        statistic = (estimate - null) / se
        pvalue = _tail(normal_cdf, statistic, direction)
    return HypothesisTestResult(
        statistic,
        pvalue,
        direction,
        "two-proportion noninferiority z-test",
        (n1, n2),
        null,
        None,
        estimate,
        se,
    )


def noninferiority_t_test(x, y=None, *, margin=0.0, mu=0.0, direction="greater"):
    """One-sided noninferiority t test for a mean or a difference of means.

    ``direction='greater'`` tests that the estimand is greater than ``-margin``;
    ``direction='less'`` tests that it is less than ``margin``.
    """
    if margin < 0 or direction not in {"greater", "less"}:
        raise ValueError("margin must be nonnegative and direction 'greater' or 'less'")
    if y is None:
        null = -margin if direction == "greater" else margin
        return one_sample_t_test(_x(x) - mu, null, direction)
    null = -margin if direction == "greater" else margin
    a = _x(x).reshape(-1)
    b = _x(y).reshape(-1)
    result = welch_t_test(a, b + null, direction)
    return HypothesisTestResult(
        result.statistic,
        result.pvalue,
        direction,
        "Welch noninferiority t-test",
        result.sample_sizes,
        null,
        result.degrees_of_freedom,
        float(a.mean() - b.mean()),
        result.standard_error,
    )
