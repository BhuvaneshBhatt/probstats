"""Small distribution-function helpers shared by statistical routines."""

from __future__ import annotations

import math
from statistics import NormalDist

import mpmath as mp

_STANDARD_NORMAL = NormalDist()


def normal_cdf(value: float) -> float:
    """Return the standard Normal cumulative distribution function."""
    return _STANDARD_NORMAL.cdf(float(value))


def normal_ppf(probability: float) -> float:
    """Return the standard Normal quantile for a probability in ``[0, 1]``."""
    if probability == 0:
        return -math.inf
    if probability == 1:
        return math.inf
    if not 0 < probability < 1:
        raise ValueError("probability must lie in [0, 1]")
    return _STANDARD_NORMAL.inv_cdf(float(probability))


def t_cdf(value: float, df: float) -> float:
    """Return the Student-t cumulative distribution function."""
    if df <= 0:
        raise ValueError("degrees of freedom must be positive")
    ratio = df / (df + value * value)
    beta = mp.betainc(df / 2, 0.5, 0, ratio, regularized=True)
    return float(1 - 0.5 * beta if value >= 0 else 0.5 * beta)


def t_ppf(probability: float, df: float) -> float:
    """Return a Student-t quantile using monotone bisection."""
    if not 0 < probability < 1:
        raise ValueError("probability must lie in (0, 1)")
    if df <= 0:
        raise ValueError("degrees of freedom must be positive")
    if probability < 0.5:
        return -t_ppf(1 - probability, df)
    if probability == 0.5:
        return 0.0
    low, high = 0.0, 1.0
    while t_cdf(high, df) < probability:
        high *= 2
        if not math.isfinite(high):
            raise OverflowError("Student-t quantile could not be bracketed")
    for _ in range(120):
        middle = (low + high) / 2
        if t_cdf(middle, df) < probability:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def f_cdf(value: float, df_num: float, df_den: float) -> float:
    """Return the F cumulative distribution function."""
    if df_num <= 0 or df_den <= 0:
        raise ValueError("degrees of freedom must be positive")
    if value <= 0:
        return 0.0
    ratio = df_num * value / (df_num * value + df_den)
    return float(mp.betainc(df_num / 2, df_den / 2, 0, ratio, regularized=True))


def chi2_cdf(value: float, df: float) -> float:
    """Return the chi-square cumulative distribution function."""
    if df <= 0:
        raise ValueError("degrees of freedom must be positive")
    if value <= 0:
        return 0.0
    return float(mp.gammainc(df / 2, 0, value / 2, regularized=True))


def chi2_sf(value: float, df: float) -> float:
    """Return the chi-square survival function."""
    if df <= 0:
        raise ValueError("degrees of freedom must be positive")
    if value <= 0:
        return 1.0
    return float(mp.gammainc(df / 2, value / 2, mp.inf) / mp.gamma(df / 2))
