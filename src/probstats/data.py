"""Data-analysis, smoothing, and weighted-data utilities."""

from ._data_core import (
    BinnedData,
    WeightedData,
    bin_data,
)
from ._data_exponential import (
    exponentially_weighted_correlation,
    exponentially_weighted_covariance,
    exponentially_weighted_mean,
    exponentially_weighted_standard_deviation,
    exponentially_weighted_variance,
)
from ._data_rolling import (
    rolling_apply,
    rolling_correlation,
    rolling_covariance,
    rolling_max,
    rolling_mean,
    rolling_min,
    rolling_quantile,
    rolling_standard_deviation,
    rolling_sum,
    rolling_variance,
    running_mean,
    running_quantile,
    running_standard_deviation,
    running_sum,
    running_variance,
)
from ._data_sequence import (
    SequenceSummary,
    differences,
    run_length_encode,
    sequence_summary,
)
from ._data_smoothing import (
    SmootherResult,
    kernel_smooth,
    local_polynomial_smooth,
    lowess,
)

__all__ = [
    "BinnedData",
    "SequenceSummary",
    "SmootherResult",
    "WeightedData",
    "bin_data",
    "differences",
    "exponentially_weighted_correlation",
    "exponentially_weighted_covariance",
    "exponentially_weighted_mean",
    "exponentially_weighted_standard_deviation",
    "exponentially_weighted_variance",
    "kernel_smooth",
    "local_polynomial_smooth",
    "lowess",
    "rolling_apply",
    "rolling_correlation",
    "rolling_covariance",
    "rolling_max",
    "rolling_mean",
    "rolling_min",
    "rolling_quantile",
    "rolling_standard_deviation",
    "rolling_sum",
    "rolling_variance",
    "run_length_encode",
    "running_mean",
    "running_quantile",
    "running_standard_deviation",
    "running_sum",
    "running_variance",
    "sequence_summary",
]
