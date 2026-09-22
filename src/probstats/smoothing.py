"""Smoothing, rolling, weighted-data, and binned-data namespace."""

from .data import (
    BinnedData,
    SmootherResult,
    WeightedData,
    bin_data,
    exponentially_weighted_covariance,
    exponentially_weighted_mean,
    exponentially_weighted_variance,
    kernel_smooth,
    local_polynomial_smooth,
    lowess,
    rolling_mean,
    rolling_quantile,
    rolling_sum,
    running_quantile,
)

__all__ = [
    "BinnedData",
    "SmootherResult",
    "WeightedData",
    "bin_data",
    "exponentially_weighted_covariance",
    "exponentially_weighted_mean",
    "exponentially_weighted_variance",
    "kernel_smooth",
    "local_polynomial_smooth",
    "lowess",
    "rolling_mean",
    "rolling_quantile",
    "rolling_sum",
    "running_quantile",
]
