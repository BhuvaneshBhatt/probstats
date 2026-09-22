import math

import numpy as np
import pytest

from probstats import (
    Bernoulli,
    Normal,
)
from probstats.inference import (
    adjust_pvalues,
    bootstrap,
    likelihood,
    likelihood_confidence_region,
    linear_regression,
    mann_whitney_u_test,
    numerical_mle,
    parameter_transform_plan,
    profile_likelihood,
    robust_regression_covariance,
    wilcoxon_signed_rank_test,
)


def test_multiple_testing_adjustments_preserve_order_and_monotonicity():
    p = np.array([0.01, 0.04, 0.03, 0.20])
    holm = adjust_pvalues(p, method="holm")
    assert holm.adjusted_pvalues.tolist() == pytest.approx([0.04, 0.09, 0.09, 0.20])
    bh = adjust_pvalues(p, method="benjamini-hochberg")
    assert bh.adjusted_pvalues.tolist() == pytest.approx(
        [0.04, 0.0533333333, 0.0533333333, 0.20]
    )
    assert holm.rejected.tolist() == [True, False, False, False]


def test_hc_covariance_variants_and_result_recalibration():
    x = np.arange(1.0, 11.0)
    y = 1 + 2 * x + np.array([0.1, -0.1, 0.2, -0.2, 0.4, -0.4, 0.7, -0.7, 1.0, -1.0])
    fit = linear_regression(x, y)
    for kind in ("HC0", "HC1", "HC2", "HC3", "HC4"):
        cov = robust_regression_covariance(fit, kind)
        assert cov.shape == (2, 2)
        assert np.all(np.isfinite(cov))
    hc3 = fit.with_covariance("HC3")
    assert hc3.covariance_type == "HC3"
    assert not np.allclose(hc3.covariance, fit.covariance)
    direct = linear_regression(x, y, covariance="HC3")
    assert np.allclose(direct.covariance, hc3.covariance)


def test_parameter_space_auto_constraints_normal_and_bernoulli():
    lik = likelihood(Normal, [1, 2, 3, 4, 5], ("mean", "sigma"))
    plan = parameter_transform_plan(lik, [2.0, 1.0])
    assert plan.parameter_names == ("mean", "sigma")
    roundtrip = plan.from_unconstrained(plan.to_unconstrained([2.0, 1.0]))
    assert roundtrip == pytest.approx([2.0, 1.0])
    fit = numerical_mle(lik, [2.5, 1.0], constraints="auto")
    assert fit.converged
    assert fit.parameters["sigma"] > 0
    assert "ParameterSpace" in fit.method

    bern = likelihood(Bernoulli, [0, 1, 1, 1, 0], ("p",))
    bfit = numerical_mle(bern, [0.5], constraints="auto")
    assert bfit.converged
    assert bfit.parameters["p"] == pytest.approx(0.6, abs=1e-4)
    assert 0 < bfit.parameters["p"] < 1


def test_bootstrap_bca_and_bootstrap_t():
    data = np.array([1.0, 2, 3, 4, 8])
    bca = bootstrap(data, np.mean, iterations=400, method="bca", rng=4)
    assert bca.method.endswith("BCa")
    assert bca.confidence_interval[0] < np.mean(data) < bca.confidence_interval[1]

    def se_mean(sample):
        return float(np.std(sample, ddof=1) / math.sqrt(len(sample)))

    bt = bootstrap(
        data,
        np.mean,
        iterations=300,
        method="bootstrap-t",
        standard_error=se_mean,
        rng=5,
    )
    assert bt.method.endswith("bootstrap-t")
    assert bt.confidence_interval[0] < np.mean(data) < bt.confidence_interval[1]


def test_exact_rank_distributions_with_ties():
    mw = mann_whitney_u_test([1, 1, 2], [2, 3], exact=True)
    assert "exact conditional" in mw.method
    # Verify exact p-value by direct assignment count is a rational multiple of 1/C(5,3).
    assert mw.pvalue * math.comb(5, 3) == pytest.approx(
        round(mw.pvalue * math.comb(5, 3))
    )

    wx = wilcoxon_signed_rank_test([1, 2, -2, 3], exact=True)
    assert "exact sign-randomization" in wx.method
    assert wx.pvalue * 16 == pytest.approx(round(wx.pvalue * 16))


def test_profile_likelihood_and_lr_confidence_region():
    data = [1.8, 2.0, 2.2, 2.1, 1.9, 2.05]
    lik = likelihood(Normal, data, ("mean",), fixed={"sigma": 0.2})
    fit = numerical_mle(lik, [2.0], constraints="auto")
    assert fit.converged
    profile = profile_likelihood(lik, fit, "mean", points=15, span=3)
    assert profile.interval is not None
    assert profile.interval[0] <= fit.parameters["mean"] <= profile.interval[1]
    assert np.nanmin(profile.statistic()) == pytest.approx(0, abs=0.1)

    region = likelihood_confidence_region(lik, fit, parameters=("mean",))
    assert region.contains([fit.parameters["mean"]])
    assert not region.contains([0.0])
