import math

import numpy as np
import pytest

from probstats import Normal
from probstats.inference import (
    bootstrap,
    estimator_covariance,
    expected_fisher_information,
    kolmogorov_smirnov_test,
    kruskal_wallis_test,
    likelihood,
    likelihood_ratio_test,
    linear_regression,
    mann_whitney_u_test,
    numerical_map,
    numerical_mle,
    observed_fisher_information,
    one_way_anova,
    permutation_test,
    score_test,
    wald_test,
    wilcoxon_signed_rank_test,
)


def test_generic_likelihood_numerical_mle_and_information():
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    lik = likelihood(Normal, data, ("mean", "sigma"))
    fit = numerical_mle(lik, [2.5, 2.0], bounds=[(None, None), (1e-4, None)])
    assert fit.converged
    assert fit.parameters["mean"] == pytest.approx(3.0, abs=1e-4)
    assert fit.parameters["sigma"] == pytest.approx(math.sqrt(2), rel=1e-3)
    assert fit.standard_errors["mean"] > 0
    info = observed_fisher_information(lik, fit.estimate)
    cov = estimator_covariance(lik, fit.estimate)
    assert info.shape == cov.shape == (2, 2)
    assert np.all(np.linalg.eigvalsh(info) > 0)
    expected = expected_fisher_information(lik, fit.estimate, samples=50, rng=2)
    assert expected.shape == (2, 2)


def test_map_with_distribution_prior():
    lik = likelihood(Normal, [4.8, 5.0, 5.2], ("mean",), fixed={"sigma": 1.0})
    fit = numerical_map(lik, [4.0], priors={"mean": Normal(0, 1)})
    assert fit.converged
    assert 3.0 < fit.parameters["mean"] < 5.0
    assert fit.log_prior < 0


def test_lr_wald_score_tests():
    data = [1.8, 2.0, 2.2, 2.1, 1.9]
    full_lik = likelihood(Normal, data, ("mean",), fixed={"sigma": 0.2})
    full = numerical_mle(full_lik, [1.5])
    restricted_ll = full_lik.log_likelihood([0.0])
    from probstats.inference import OptimizationResult

    restricted = OptimizationResult(
        {"mean": 0.0},
        np.array([0.0]),
        -restricted_ll,
        restricted_ll,
        True,
        0,
        "fixed",
        "fixed",
    )
    assert likelihood_ratio_test(full, restricted, 1).pvalue < 1e-6
    assert wald_test(full.estimate, [0], full.covariance).pvalue < 1e-6
    assert score_test(full_lik, [0]).pvalue < 1e-6


def test_bootstrap_and_permutation_reproducible():
    b1 = bootstrap([1, 2, 3, 4, 5], iterations=200, rng=123)
    b2 = bootstrap([1, 2, 3, 4, 5], iterations=200, rng=123)
    assert np.array_equal(b1.replicates, b2.replicates)
    assert b1.confidence_interval[0] <= 3 <= b1.confidence_interval[1]
    p = permutation_test([1, 2, 3, 4, 5], [10, 11, 12, 13, 14], iterations=499, rng=1)
    assert p.pvalue < 0.02
    assert p.method == "exact permutation test"


def test_anova_and_regression():
    a = one_way_anova([1, 2, 1], [5, 6, 5], [10, 9, 10])
    assert a.statistic > 20 and a.pvalue < 0.001
    x = np.arange(1, 8, dtype=float)
    y = 1 + 2 * x + np.array([0.1, -0.1, 0.05, -0.05, 0.1, -0.1, 0])
    r = linear_regression(x, y)
    assert r.coefficients[0] == pytest.approx(1, abs=0.15)
    assert r.coefficients[1] == pytest.approx(2, abs=0.05)
    assert r.r_squared > 0.999
    assert r.pvalues[1] < 1e-5
    assert r.predict([8])[0] == pytest.approx(17, abs=0.2)


def test_nonparametric_tests():
    mw = mann_whitney_u_test([1, 2, 3, 4], [10, 11, 12, 13])
    assert mw.pvalue < 0.05
    wx = wilcoxon_signed_rank_test([5, 6, 7, 8, 9], [1, 2, 3, 4, 5])
    assert wx.statistic > 0 and wx.pvalue < 0.1
    kw = kruskal_wallis_test([1, 2, 3], [5, 6, 7], [9, 10, 11])
    assert kw.pvalue < 0.05


def test_gof_known_and_fitted_distribution():
    rng = np.random.default_rng(5)
    data = rng.normal(0, 1, 100)
    known = kolmogorov_smirnov_test(data, Normal(0, 1))
    assert 0 <= known.pvalue <= 1
    fitted = kolmogorov_smirnov_test(
        data, family=Normal, fit=True, bootstrap_iterations=30, rng=1
    )
    assert 0 <= fitted.pvalue <= 1
    assert "parametric-bootstrap" in fitted.method
