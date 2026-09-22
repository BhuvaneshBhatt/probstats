import math

import numpy as np
import pytest

from probstats.testing import (
    bartlett_test,
    box_m_test,
    box_pierce_test,
    brown_forsythe_test,
    cramers_v,
    distance_correlation,
    distance_correlation_test,
    durbin_watson,
    hotelling_t2_test,
    kendall_tau,
    kendall_tau_test,
    levene_test,
    ljung_box_test,
    mardia_normality_test,
    noninferiority_t_test,
    one_sample_tost,
    one_sample_variance_test,
    proportion_noninferiority_z_test,
    spearman_correlation,
    spearman_correlation_test,
    two_proportion_tost,
    two_sample_hotelling_t2_test,
    variance_ratio_test,
    welch_tost,
)


def test_rank_correlations_handle_ties_and_direction():
    x = [1, 2, 2, 4, 5, 6]
    y = [6, 5, 5, 3, 2, 1]
    assert spearman_correlation(x, y) == pytest.approx(-1)
    assert kendall_tau(x, y) == pytest.approx(-1)
    assert spearman_correlation_test(x, y).pvalue < 1e-8
    assert kendall_tau_test(x, y).pvalue < 0.02


def test_distance_correlation_nonlinear():
    x = np.linspace(-2, 2, 21)
    y = x**2
    assert abs(np.corrcoef(x, y)[0, 1]) < 1e-12
    assert distance_correlation(x, y) > 0.45
    result = distance_correlation_test(x, y, permutations=99, rng=123)
    assert result.pvalue <= 0.05


def test_cramers_v_independent_and_strong_tables():
    assert cramers_v([[10, 20], [20, 40]]) == pytest.approx(0)
    assert cramers_v([[40, 0], [0, 40]]) > 0.95


def test_variance_tests_basic_behavior():
    x = [-2, -1, 0, 1, 2]
    one = one_sample_variance_test(x, variance=2.5)
    assert one.statistic == pytest.approx(4)
    ratio = variance_ratio_test([1, 2, 3, 4], [2, 4, 6, 8])
    assert ratio.statistic == pytest.approx(0.25)
    equal1 = [1, 2, 3, 4, 5]
    equal2 = [11, 12, 13, 14, 15]
    assert bartlett_test(equal1, equal2).pvalue == pytest.approx(1)
    assert levene_test(equal1, equal2, center="mean").pvalue == pytest.approx(1)
    assert brown_forsythe_test(equal1, equal2).pvalue == pytest.approx(1)


def test_tost_and_noninferiority():
    x = [9.9, 10.0, 10.1, 10.05, 9.95, 10.02]
    result = one_sample_tost(x, -0.2, 0.2, mu=10)
    assert result.equivalent_05
    y = [9.92, 10.02, 10.08, 10.03, 9.97, 10.00]
    assert welch_tost(x, y, -0.25, 0.25).equivalent_05
    ni = noninferiority_t_test(x, y, margin=0.25, direction="greater")
    assert ni.pvalue < 0.05
    assert two_proportion_tost(52, 100, 50, 100, -0.2, 0.2).equivalent_05
    prop_ni = proportion_noninferiority_z_test(55, 100, 50, 100, margin=0.1)
    assert prop_ni.pvalue < 0.05


def test_portmanteau_and_durbin_watson():
    rng = np.random.default_rng(123)
    white = rng.normal(size=400)
    assert ljung_box_test(white, lags=5).pvalue > 0.01
    assert box_pierce_test(white, lags=5).pvalue > 0.01
    ar = np.empty(400)
    ar[0] = 0
    noise = rng.normal(scale=0.3, size=400)
    for i in range(1, ar.size):
        ar[i] = 0.9 * ar[i - 1] + noise[i]
    assert ljung_box_test(ar, lags=5).pvalue < 1e-6
    assert durbin_watson(ar) < 1


def test_hotelling_one_sample_matches_univariate_t_squared():
    x = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    result = hotelling_t2_test(x, [0])
    t = 3 / (np.std(x[:, 0], ddof=1) / math.sqrt(5))
    assert result.statistic == pytest.approx(t * t)
    assert result.pvalue < 0.05


def test_two_sample_hotelling_separates_shifted_groups():
    a = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.2], [0.2, 0.8]])
    b = a + np.array([4.0, -3.0])
    result = two_sample_hotelling_t2_test(a, b)
    assert result.pvalue < 1e-5


def test_box_m_and_mardia_smoke_and_null_like_behavior():
    rng = np.random.default_rng(321)
    a = rng.multivariate_normal([0, 0], [[1, 0.2], [0.2, 1.5]], size=80)
    b = rng.multivariate_normal([1, -1], [[1, 0.2], [0.2, 1.5]], size=90)
    assert box_m_test(a, b).pvalue > 0.01
    normality = mardia_normality_test(np.vstack([a, b - [1, -1]]))
    assert 0 <= normality.skewness_pvalue <= 1
    assert 0 <= normality.kurtosis_pvalue <= 1


def test_kendall_tau_matches_quadratic_definition_with_ties():
    rng = np.random.default_rng(27)
    x = rng.integers(0, 8, size=90)
    y = rng.integers(0, 9, size=90)
    concordant = 0
    discordant = 0
    for i in range(x.size - 1):
        products = np.sign(x[i + 1 :] - x[i]) * np.sign(y[i + 1 :] - y[i])
        concordant += np.count_nonzero(products > 0)
        discordant += np.count_nonzero(products < 0)
    n0 = x.size * (x.size - 1) / 2
    cx = np.unique(x, return_counts=True)[1]
    cy = np.unique(y, return_counts=True)[1]
    tx = sum(c * (c - 1) / 2 for c in cx)
    ty = sum(c * (c - 1) / 2 for c in cy)
    expected = (concordant - discordant) / math.sqrt((n0 - tx) * (n0 - ty))
    assert kendall_tau(x, y) == pytest.approx(expected)


def test_portmanteau_rejects_fractional_lag_entries():
    series = np.arange(10.0)
    with pytest.raises(TypeError, match="lag"):
        ljung_box_test(series, lags=[1, 2.5])


def test_proportion_equivalence_tests_require_integer_counts():
    with pytest.raises(TypeError, match="successes"):
        two_proportion_tost(10.5, 20, 10, 20, -0.2, 0.2)
    with pytest.raises(TypeError, match="n"):
        proportion_noninferiority_z_test(10, 20.5, 10, 20)
