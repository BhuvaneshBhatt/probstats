"""Generic descriptive and distribution statistics.

These functions dispatch on the first argument so the same mathematical
operation works for observed data and probability distributions. Numerical
sample statistics remain implemented in :mod:`probstats.descriptive`, while
exact distribution statistics remain implemented in :mod:`probstats.functionals`.
"""

from __future__ import annotations

import sympy as sp

from . import descriptive, functionals
from .distributions.base import Distribution
from .random_variable_algebra import is_random_expression


def mean(value, *, weights=None, assumptions=None):
    """Return the arithmetic mean of data or the mean of a distribution."""
    if isinstance(value, sp.Basic) and is_random_expression(value):
        if weights is not None:
            raise TypeError("weights apply only to observed data")
        return functionals.expectation(value, assumptions=assumptions)
    if assumptions is not None:
        raise TypeError(
            "assumptions apply only to symbolic random-variable expressions"
        )
    if isinstance(value, Distribution):
        if weights is not None:
            raise TypeError("weights apply only to observed data")
        return functionals.mean(value)
    return descriptive.mean(value, weights=weights)


def median(value):
    """Return the sample median or the median of a distribution."""
    if isinstance(value, Distribution):
        return functionals.quantile(value, sp.Rational(1, 2))
    return descriptive.median(value)


def quantile(value, q, *, method="linear"):
    """Return sample or distribution quantiles.

    ``method`` controls interpolation for observed data. Distribution quantiles
    have their own mathematical definition, so non-default interpolation
    methods are rejected rather than ignored.
    """
    if isinstance(value, Distribution):
        if method != "linear":
            raise TypeError("method applies only to observed data")
        return functionals.quantile(value, q)
    return descriptive.quantile(value, q, method=method)


def variance(value, *, ddof=None, assumptions=None):
    """Return sample variance or distribution variance.

    Observed data use the usual sample default ``ddof=1``. Distribution
    variance is a population quantity and therefore does not accept nonzero
    ``ddof``.
    """
    if isinstance(value, sp.Basic) and is_random_expression(value):
        if ddof not in (None, 0):
            raise TypeError("ddof applies only to observed data")
        from .random_variable_algebra import variance as random_variance

        return random_variance(value, assumptions=assumptions)
    if assumptions is not None:
        raise TypeError(
            "assumptions apply only to symbolic random-variable expressions"
        )
    if isinstance(value, Distribution):
        return functionals.variance(value, ddof=ddof)
    return descriptive.variance(value, ddof=1 if ddof is None else ddof)


def covariance(left, right=None, *, ddof=None, assumptions=None):
    """Return covariance for random expressions or paired observations."""
    if isinstance(left, sp.Basic) and is_random_expression(left):
        if (
            right is None
            or not isinstance(right, sp.Basic)
            or not is_random_expression(right)
        ):
            raise TypeError("symbolic covariance requires two random expressions")
        if ddof not in (None, 0):
            raise TypeError("ddof applies only to observed data")
        from .random_variable_algebra import covariance as random_covariance

        return random_covariance(left, right, assumptions=assumptions)
    if assumptions is not None:
        raise TypeError(
            "assumptions apply only to symbolic random-variable expressions"
        )
    return descriptive.covariance(left, right, ddof=1 if ddof is None else ddof)


def standard_deviation(value, *, ddof=None, assumptions=None):
    """Return sample or distribution standard deviation."""
    if isinstance(value, sp.Basic) and is_random_expression(value):
        return sp.sqrt(variance(value, ddof=ddof, assumptions=assumptions))
    if assumptions is not None:
        raise TypeError(
            "assumptions apply only to symbolic random-variable expressions"
        )
    if isinstance(value, Distribution):
        return sp.sqrt(functionals.variance(value, ddof=ddof))
    return descriptive.standard_deviation(value, ddof=1 if ddof is None else ddof)


__all__ = [
    "covariance",
    "mean",
    "median",
    "quantile",
    "standard_deviation",
    "variance",
]
