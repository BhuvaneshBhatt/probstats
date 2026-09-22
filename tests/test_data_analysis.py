import numpy as np
import pytest

from probstats.data import (
    WeightedData,
    bin_data,
    exponentially_weighted_correlation,
    exponentially_weighted_covariance,
    exponentially_weighted_mean,
    exponentially_weighted_variance,
    kernel_smooth,
    local_polynomial_smooth,
    lowess,
    rolling_correlation,
    rolling_covariance,
    rolling_mean,
    rolling_quantile,
    rolling_variance,
    run_length_encode,
    running_mean,
    running_quantile,
    running_variance,
    sequence_summary,
)
from probstats.distributions import (
    HistogramDistribution,
    KernelDensityDistribution,
)
from probstats.functionals import (
    mean,
    variance,
)


def test_weighted_data_statistics_sampling_and_existing_api_integration():
    data = WeightedData([0.0, 2.0, 4.0], [1.0, 2.0, 1.0])
    assert data.total_weight == 4.0
    assert data.effective_n == pytest.approx(8 / 3)
    assert data.mean() == pytest.approx(2.0)
    assert data.quantile(0.5) == pytest.approx(2.0)
    assert mean(data) == pytest.approx(2.0)
    assert variance(data, ddof=0) == pytest.approx(2.0)

    draws = data.sample(20_000, rng=12)
    assert draws.mean() == pytest.approx(2.0, abs=0.04)

    kde = KernelDensityDistribution(data, bandwidth=0.2)
    assert float(mean(kde)) == pytest.approx(2.0)
    hist = HistogramDistribution(data, bins=[-1, 1, 3, 5])
    assert np.asarray([float(p) for p in hist.probabilities]) == pytest.approx(
        [0.25, 0.5, 0.25]
    )


def test_weighted_vector_covariance_and_quantiles():
    data = WeightedData([[0.0, 0.0], [2.0, 4.0], [4.0, 8.0]], [1, 2, 1])
    assert np.allclose(data.mean(), [2.0, 4.0])
    assert np.allclose(data.covariance(), [[2.0, 4.0], [4.0, 8.0]])
    assert np.allclose(data.quantile(0.5), [2.0, 4.0])


def test_binning_frequency_density_and_distribution_round_trip():
    binned = bin_data(WeightedData([0.2, 0.7, 1.5, 1.8], [1, 1, 3, 1]), bins=[0, 1, 2])
    assert np.allclose(binned.counts, [2, 4])
    assert np.allclose(binned.frequencies, [1 / 3, 2 / 3])
    assert np.dot(binned.density, binned.widths) == pytest.approx(1.0)
    dist = binned.to_distribution()
    assert float(dist.cdf(1)) == pytest.approx(1 / 3)
    auto = WeightedData([0.1, 0.2, 1.5], [1, 2, 3]).histogram()
    assert auto.counts.sum() == pytest.approx(6.0)


def test_weighted_binned_result_contracts():
    data = WeightedData([0.0, 1.0, 10.0], [1.0, 2.0, 7.0])
    with pytest.raises(ValueError):
        data.values[0] = 3.0
    with pytest.raises(ValueError):
        data.weights[0] = 3.0

    binned = bin_data(data, bins=[0.0, 2.0], range=(0.0, 2.0))
    assert binned.total_weight == pytest.approx(3.0)
    assert binned.frequencies.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        binned.counts[0] = 0.0


def test_unbiased_ew_requires_two_effective_observations():
    x = [1.0, 2.0, 3.0]
    assert np.all(np.isnan(exponentially_weighted_variance(x, alpha=1.0)))
    assert np.all(np.isnan(exponentially_weighted_covariance(x, x, alpha=1.0)))


def test_rolling_and_running_statistics_match_direct_calculations():
    x = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    assert np.allclose(
        rolling_mean(x, 3),
        [np.nan, np.nan, 7 / 3, 14 / 3, 28 / 3],
        equal_nan=True,
    )
    assert rolling_quantile(x, 3, 0.5)[-1] == 8
    assert rolling_variance(x, 3, ddof=1)[-1] == pytest.approx(
        np.var([4, 8, 16], ddof=1)
    )
    y = 2 * x + 1
    assert rolling_covariance(x, y, 3)[-1] == pytest.approx(
        np.cov(x[-3:], y[-3:])[0, 1]
    )
    assert rolling_correlation(x, y, 3)[-1] == pytest.approx(1.0)
    assert np.allclose(running_mean(x), [1, 1.5, 7 / 3, 3.75, 6.2])
    expected_var = np.asarray(
        [np.var(x[:i], ddof=1) if i > 1 else np.nan for i in range(1, 6)]
    )
    assert np.allclose(running_variance(x), expected_var, equal_nan=True)
    assert np.allclose(running_quantile(x, 0.5), [1, 1.5, 2, 3, 4])


def test_exponentially_weighted_mean_and_variance_have_online_semantics():
    x = np.array([1.0, 3.0, 5.0])
    assert np.allclose(exponentially_weighted_mean(x, alpha=0.5), [1.0, 2.0, 3.5])
    assert np.allclose(
        exponentially_weighted_mean(x, span=3),
        exponentially_weighted_mean(x, alpha=0.5),
    )
    var = exponentially_weighted_variance(x, alpha=0.5, bias=True)
    assert np.all(var >= 0)
    assert var[0] == 0
    y = 2 * x + 5
    assert exponentially_weighted_covariance(x, y, alpha=0.5)[-1] > 0
    assert exponentially_weighted_correlation(x, y, alpha=0.5)[-1] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        exponentially_weighted_mean(x, alpha=0.5, span=3)


def test_local_polynomial_linear_signal():
    x = np.linspace(-2, 2, 21)
    y = 1.5 + 2.25 * x
    result = local_polynomial_smooth(x, y, bandwidth=0.7, degree=1)
    assert np.max(np.abs(result.fitted - y)) < 1e-10
    assert np.max(np.abs(result.residuals)) < 1e-10

    kernel = kernel_smooth(x, y, bandwidth=0.7)
    assert np.all(np.isfinite(kernel.fitted))
    assert kernel.method == "kernel"


def test_smoother_preserves_zero_weights():
    x = np.array([0.0, 1.0, 2.0, 3.0, 100.0])
    y = np.array([0.0, 1.0, 2.0, 3.0, -1e9])
    weights = np.array([1.0, 1.0, 1.0, 1.0, 0.0])

    weighted = local_polynomial_smooth(
        x,
        y,
        x_eval=[100.0],
        bandwidth=0.01,
        degree=1,
        kernel="uniform",
        weights=weights,
    )
    reference = local_polynomial_smooth(
        x[:-1], y[:-1], x_eval=[100.0], bandwidth=0.01, degree=1, kernel="uniform"
    )
    assert weighted.fitted == pytest.approx(reference.fitted)


def test_variance_paths_reject_invalid_ddof_without_runtime_warnings():
    data = WeightedData([1.0, 2.0, 3.0])
    for ddof in (-1, np.nan):
        with pytest.raises(ValueError):
            data.variance(ddof=ddof)
        with pytest.raises(ValueError):
            rolling_variance([1.0, 2.0, 3.0], 2, ddof=ddof)
        with pytest.raises(ValueError):
            running_variance([1.0, 2.0, 3.0], ddof=ddof)


def test_lowess_is_robust_to_single_large_outlier():
    x = np.linspace(0, 10, 51)
    truth = 3 + 0.5 * x
    y = truth.copy()
    y[25] += 40
    plain = lowess(x, y, span=0.35, robust_iterations=0)
    robust = lowess(x, y, span=0.35, robust_iterations=3)
    center = 25
    assert abs(robust.fitted[center] - truth[center]) < abs(
        plain.fitted[center] - truth[center]
    )
    assert abs(robust.fitted[center] - truth[center]) < 0.2


def test_sequence_statistics_and_run_length_encoding():
    assert run_length_encode([1, 1, 2, 2, 2, 1]) == ((1, 2), (2, 3), (1, 1))
    summary = sequence_summary([1, 2, 3, 2, 1, 2])
    assert summary.count == 6
    assert summary.turning_points == 2
    assert summary.longest_increasing_run == 3
    assert summary.longest_decreasing_run == 3
    assert summary.change == 1


def test_smoother_result_tracks_training_coordinates_for_residuals():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = 1.0 + 2.0 * x
    result = local_polynomial_smooth(
        x,
        y,
        x_eval=np.array([0.5, 1.5]),
        bandwidth=1.0,
        degree=1,
    )
    assert result.x.shape == (2,)
    assert result.fitted.shape == (2,)
    assert np.array_equal(result.residual_x, x)
    assert result.residuals.shape == x.shape
    with pytest.raises(ValueError):
        result.residual_x[0] = 10.0


def test_lowess_fallback_preserves_zero_robust_weights():
    x = np.array([0.0, 1.0, 2.0, 3.0, 100.0])
    truth = 2.0 + x
    y = truth.copy()
    y[-1] = 1e8
    result = lowess(
        x,
        y,
        x_eval=np.array([100.0]),
        span=0.6,
        degree=1,
        robust_iterations=3,
    )
    assert np.isfinite(result.fitted[0])
    assert result.residual_x.shape == x.shape


def test_integer_analysis_parameters_reject_booleans_and_fractional_values():
    with pytest.raises(TypeError, match="window must be an integer"):
        rolling_mean([1.0, 2.0], True)
    with pytest.raises(TypeError, match="degree must be an integer"):
        local_polynomial_smooth([0.0, 1.0], [0.0, 1.0], bandwidth=1.0, degree=1.5)
    with pytest.raises(TypeError, match="robust_iterations must be an integer"):
        lowess([0.0, 1.0], [0.0, 1.0], robust_iterations=1.5)


def test_weighted_correlation_matrix_and_public_dispatch():
    from probstats.data import WeightedData
    from probstats.descriptive import correlation

    data = WeightedData([[0.0, 1.0], [1.0, 3.0], [2.0, 5.0]], [1.0, 2.0, 1.0])
    expected = np.ones((2, 2))
    assert np.allclose(data.correlation(), expected)
    assert np.allclose(correlation(data), expected)


def test_rolling_sum_and_mean_match_window_reference():
    from probstats.data import (
        rolling_mean,
        rolling_sum,
    )

    x = np.arange(1.0, 9.0)
    for center in (False, True):
        for min_periods in (1, 3):
            sums = rolling_sum(x, 4, min_periods=min_periods, center=center)
            means = rolling_mean(x, 4, min_periods=min_periods, center=center)
            bounds = []
            for i in range(x.size):
                if center:
                    left, right = 1, 3
                    bounds.append((max(0, i - left), min(x.size, i + right)))
                else:
                    bounds.append((max(0, i - 3), i + 1))
            expected_sum = np.array(
                [
                    np.sum(x[lo:hi]) if hi - lo >= min_periods else np.nan
                    for lo, hi in bounds
                ]
            )
            expected_mean = np.array(
                [
                    np.mean(x[lo:hi]) if hi - lo >= min_periods else np.nan
                    for lo, hi in bounds
                ]
            )
            assert np.allclose(sums, expected_sum, equal_nan=True)
            assert np.allclose(means, expected_mean, equal_nan=True)


def test_running_quantile_matches_numpy_for_fast_methods():
    rng = np.random.default_rng(2026)
    x = rng.normal(size=80)
    for method in ("linear", "lower", "higher", "midpoint", "nearest"):
        actual = running_quantile(x, 0.37, method=method)
        expected = np.array(
            [np.quantile(x[: i + 1], 0.37, method=method) for i in range(x.size)]
        )
        assert np.allclose(actual, expected)
