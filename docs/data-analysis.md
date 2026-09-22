# Data analysis and smoothing

`probstats` provides a NumPy-centered data-analysis layer that complements its probability distributions and statistical tests. Pandas input is supported at tabular boundaries when the optional `tabular` extra is installed; numerical smoothing, rolling, and weighted-statistics kernels remain NumPy-based. The same observation-weight semantics are shared by descriptive summaries, histogram/KDE construction, and smoothers.

## Weighted data

`WeightedData(values, weights)` stores observations along the first axis. Weights are nonnegative and need not sum to one. The object exposes the original total weight, normalized weights, Kish effective sample size, weighted mean/variance/covariance, weighted quantiles, resampling, histogram construction, and KDE construction.

```python
from probstats.smoothing import WeightedData
from probstats.stats import (
    mean,
    quantile,
)

x = WeightedData([0, 2, 4], weights=[1, 2, 1])
mean(x)  # 2.0
quantile(x, 0.5)  # 2.0
x.effective_n
```

`WeightedData` can also contain vector observations with shape `(n, d)`. Means and marginal quantiles are computed columnwise, while `covariance()` returns the full weighted covariance matrix.

## Binning and histograms

`bin_data(...)` returns `BinnedData`, which records bin edges and weighted counts and derives widths, midpoints, relative frequencies, density heights, and cumulative frequencies. `BinnedData.to_distribution()` converts the summary to `HistogramDistribution` without re-binning.

```python
from probstats.smoothing import bin_data

hist = bin_data(x, bins=[-1, 1, 3, 5])
hist.frequencies
hist.density
law = hist.to_distribution()
```

## Rolling and running statistics

Trailing-window functions include `rolling_sum`, `rolling_mean`, `rolling_variance`, `rolling_standard_deviation`, `rolling_min`, `rolling_max`, `rolling_quantile`, and the general `rolling_apply`. `min_periods` controls when a partial window becomes valid; `center=True` uses centered windows.

Expanding statistics include `running_sum`, `running_mean`, numerically stable Welford `running_variance`, `running_standard_deviation`, and exact-so-far `running_quantile`.

## Exponentially weighted statistics

`exponentially_weighted_mean`, `exponentially_weighted_variance`, and `exponentially_weighted_standard_deviation` accept exactly one decay parameter: `alpha`, `span`, `com`, or `halflife`.

The default exponentially weighted mean has online (`adjust=False`) semantics,

```text
m[t] = alpha*x[t] + (1-alpha)*m[t-1].
```

Set `adjust=True` to normalize the finite history by the accumulated geometric weights.

## Kernel and local-polynomial smoothing

`local_polynomial_smooth` fits a weighted polynomial around every evaluation point. `kernel_smooth` computes the Nadaraya-Watson estimator, corresponding to local-polynomial degree 0. Supported kernels are Gaussian, Epanechnikov, tricube, and uniform.

```python
fit = local_polynomial_smooth(x, y, bandwidth=0.5, degree=1)
fit.x
fit.fitted
fit.residuals
```

Observation weights can be supplied directly or through `WeightedData`. Local-linear fits exactly reproduce a linear signal apart from floating-point roundoff, including near boundaries where ordinary kernel regression is biased.

## LOWESS

`lowess` implements locally weighted regression with nearest-neighbor spans and Cleveland-style robust bisquare residual reweighting. `span` is the fraction of observations used locally; `robust_iterations=0` disables the outlier-resistant iterations.

```python
fit = lowess(x, y, span=0.4, robust_iterations=2)
```

LOWESS accepts polynomial degrees 0, 1, and 2. Its tricube neighborhood adapts to local data density rather than requiring a fixed distance bandwidth.

## Sequence statistics

`sequence_summary` collects order-sensitive diagnostics: first/last/change, lag-1 autocorrelation, runs relative to the median, turning points, and longest increasing/decreasing runs. `differences` computes lagged differences and `run_length_encode` exposes consecutive-value run structure.

These utilities are descriptive. Formal independence/randomness inference belongs in the hypothesis-testing layer (`ljung_box_test`, runs/rank tests, and related procedures).

## Weighted covariance and correlation

`WeightedData` supports both covariance and correlation matrices for vector observations. The public descriptive functions also accept a weighted series together with a second plain or weighted series:

```python
from probstats.smoothing import WeightedData
from probstats.stats import (
    correlation,
    covariance,
)

x = WeightedData([1, 2, 4, 8], weights=[1, 2, 2, 1])
y = [2, 3, 7, 9]

covariance(x, y)
correlation(x, y)
```

If both arguments are `WeightedData`, they must use the same weights. This avoids silently combining incompatible weighting schemes.
