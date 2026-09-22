import math

import numpy as np
import pytest

from probstats import (
    Bernoulli,
    Exponential,
    Gamma,
    Normal,
    Poisson,
    Uniform,
)
from probstats.data import WeightedData
from probstats.descriptive import (
    EmpiricalDistribution,
    correlation,
    covariance,
    describe,
    mean,
    median,
    median_absolute_deviation,
    trimmed_mean,
    winsorize,
)
from probstats.descriptive import (
    variance as descriptive_variance,
)
from probstats.estimation import (
    maximum_likelihood,
    mean_confidence_interval,
    method_of_moments,
    proportion_confidence_interval,
)
from probstats.functionals import (
    quantile,
    variance,
)
from probstats.testing import (
    chi_square_goodness_of_fit,
    exact_binomial_test,
    one_proportion_z_test,
    one_sample_t_test,
    one_sample_z_test,
    pearson_correlation_test,
    welch_t_test,
)


def test_descriptive_and_robust_statistics():
    x = [1, 2, 3, 4, 100]
    assert mean(x) == 22
    assert descriptive_variance([1, 2, 3], ddof=1) == 1
    assert trimmed_mean(x, 0.2) == 3
    assert np.array_equal(winsorize(x, 0.2), [2, 2, 3, 4, 4])
    assert median_absolute_deviation(x) == 1
    s = describe([1, 2, 3, 4, 5])
    assert s.count == 5 and s.mean == 3 and s.median == 3 and s.iqr == 2


def test_descriptive_weighted_data_semantics_are_explicit():
    data = WeightedData([0.0, 10.0], [9.0, 1.0])
    assert mean(data) == pytest.approx(1.0)
    assert median(data) == pytest.approx(data.quantile(0.5))
    assert quantile(data, 0.5) == pytest.approx(data.quantile(0.5))
    assert variance(data, ddof=0) == pytest.approx(9.0)
    assert covariance(WeightedData([[0.0, 0.0], [2.0, 4.0]], [1, 1]), ddof=0)[
        0, 1
    ] == pytest.approx(2.0)
    assert correlation(data, data) == pytest.approx(1.0)
    assert covariance(data, data, ddof=0) == pytest.approx(9.0)
    with pytest.raises(ValueError, match="same weights"):
        correlation(data, WeightedData([0.0, 10.0], [1.0, 9.0]))


def test_descriptive_statistics_reject_nonfinite_observations():
    with pytest.raises(ValueError):
        mean([1.0, np.nan])


def test_covariance_correlation():
    assert covariance([1, 2, 3], [2, 4, 6]) == 2
    assert correlation([1, 2, 3], [2, 4, 6]) == pytest.approx(1)


def test_empirical_distribution():
    e = EmpiricalDistribution([3, 1, 2, 2])
    assert e.cdf(2) == 0.75
    assert e.quantile(0.5) == 2
    assert e.sample(5, rng=1).shape == (5,)


def test_mle_common_families():
    n = maximum_likelihood(Normal, [1, 2, 3])
    assert isinstance(n.distribution, Normal) and n.parameters["mean"] == 2
    assert n.parameters["sigma"] == pytest.approx(math.sqrt(2 / 3))
    assert math.isfinite(n.aic) and math.isfinite(n.bic)
    b = maximum_likelihood(Bernoulli, [1, 0, 1, 1])
    assert b.parameters["p"] == 0.75
    p = maximum_likelihood(Poisson, [0, 1, 2, 1])
    assert p.parameters["rate"] == 1
    e = maximum_likelihood(Exponential, [1, 2, 3])
    assert e.parameters["rate"] == 0.5
    u = maximum_likelihood(Uniform, [2, 4, 3])
    assert u.parameters == {"low": 2.0, "high": 4.0}


def test_method_of_moments_gamma():
    r = method_of_moments(Gamma, [1, 2, 3, 4])
    m = np.mean([1, 2, 3, 4])
    v = np.var([1, 2, 3, 4])
    assert r.parameters["shape"] == pytest.approx(m * m / v)
    assert r.parameters["scale"] == pytest.approx(v / m)


def test_confidence_intervals():
    ci = mean_confidence_interval([1, 2, 3, 4, 5])
    assert ci.low < 3 < ci.high and ci.confidence == 0.95
    zci = mean_confidence_interval([1, 2, 3, 4, 5], sigma=1)
    assert zci.low < 3 < zci.high
    pci = proportion_confidence_interval(50, 100)
    assert pci.low < 0.5 < pci.high


def test_mean_tests():
    r = one_sample_t_test([10, 11, 9, 10, 10], mu=0)
    assert r.pvalue < 0.001 and r.reject_05
    w = welch_t_test([1, 2, 3, 4, 5], [10, 11, 12, 13, 14])
    assert w.pvalue < 0.001
    z = one_sample_z_test([1, 1, 1, 1], mu=0, sigma=1)
    assert z.statistic == 2


def test_proportion_and_exact_binomial_tests():
    z = one_proportion_z_test(80, 100, p0=0.5)
    assert z.pvalue < 0.001
    exact = exact_binomial_test(10, 10, p=0.5)
    assert exact.pvalue == pytest.approx(2 / 1024)


def test_chi_square_and_correlation_tests():
    c = chi_square_goodness_of_fit([10, 10, 10])
    assert c.statistic == 0 and c.pvalue == pytest.approx(1)
    r = pearson_correlation_test([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
    assert r.estimate == pytest.approx(1) and r.pvalue == 0


def test_estimation_rejects_invalid_data():
    with pytest.raises(ValueError):
        maximum_likelihood(Bernoulli, [0, 2])
    with pytest.raises(ValueError):
        maximum_likelihood(Poisson, [1, 0.5])


def test_paired_and_two_proportion_tests():
    from probstats.testing import (
        paired_t_test,
        two_proportion_z_test,
    )

    paired = paired_t_test([5, 6, 7, 8], [1, 2, 3, 4])
    assert paired.estimate == pytest.approx(4)
    assert paired.pvalue == 0.0 or paired.pvalue < 1e-6
    prop = two_proportion_z_test(80, 100, 50, 100)
    assert prop.statistic > 0
    assert prop.pvalue < 0.001


def test_chi_square_independence():
    from probstats.testing import chi_square_independence

    independent = chi_square_independence([[10, 20], [20, 40]])
    assert independent.statistic == pytest.approx(0)
    assert independent.pvalue == pytest.approx(1)
    dependent = chi_square_independence([[30, 5], [5, 30]])
    assert dependent.pvalue < 0.001


def test_extreme_student_t_confidence_interval_is_not_artificially_capped():
    interval = mean_confidence_interval([0.0, 1.0], confidence=0.999)
    assert interval.high - interval.low > 100


def test_proportion_tests_require_integer_counts():
    import pytest

    from probstats.testing import (
        exact_binomial_test,
        one_proportion_z_test,
        two_proportion_z_test,
    )

    with pytest.raises(TypeError, match="successes"):
        one_proportion_z_test(3.5, 10)
    with pytest.raises(TypeError, match="n"):
        exact_binomial_test(3, 10.5)
    with pytest.raises(TypeError, match="successes"):
        two_proportion_z_test(3, 10, 4.5, 10)
    with pytest.raises(ValueError, match="at least 1"):
        one_proportion_z_test(0, 0)


def test_classical_tests_reject_nonfinite_data_and_flatten_column_vectors():
    import pytest

    from probstats.testing import (
        chi_square_goodness_of_fit,
        one_sample_t_test,
    )

    with pytest.raises(ValueError, match="finite"):
        one_sample_t_test([1.0, np.nan, 2.0])
    flat = one_sample_t_test([1.0, 2.0, 3.0])
    column = one_sample_t_test([[1.0], [2.0], [3.0]])
    assert column.statistic == pytest.approx(flat.statistic)
    with pytest.raises(ValueError, match="nonnegative"):
        chi_square_goodness_of_fit([3, -1, 2])
