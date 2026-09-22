import math

import numpy as np

from probstats.survival import SurvivalData
from probstats.survival_regression import (
    cox_ph,
    exponential_regression,
    loglogistic_regression,
    lognormal_regression,
    parametric_survival_regression,
    weibull_regression,
)


def test_cox_single_binary_covariate_direction_and_inference():
    data = SurvivalData([1, 2, 3, 4, 5, 6], [1, 1, 1, 1, 1, 1])
    x = np.array([[1], [1], [1], [0], [0], [0]], dtype=float)
    fit = cox_ph(data, x, ties="efron", covariate_names=["treated"])
    assert fit.converged
    assert fit.coefficients[0] > 0
    assert fit.hazard_ratio[0] > 1
    assert fit.standard_error[0] > 0
    assert fit.baseline_cum_hazard[-1] > 0
    assert fit.survival(3.0, [1.0]) < fit.survival(3.0, [0.0])


def test_cox_breslow_and_efron_ties_are_finite():
    data = SurvivalData([1, 1, 2, 3, 4, 5], [1, 1, 1, 1, 0, 1])
    x = np.array([[0], [1], [0], [1], [0], [1]], dtype=float)
    breslow = cox_ph(data, x, ties="breslow")
    efron = cox_ph(data, x, ties="efron")
    assert np.isfinite(breslow.log_partial_likelihood)
    assert np.isfinite(efron.log_partial_likelihood)
    assert np.isfinite(breslow.coefficients[0])
    assert np.isfinite(efron.coefficients[0])


def test_cox_residual_shapes_and_ph_test():
    data = SurvivalData([1, 2, 3, 4, 5, 6, 7, 8], [1, 1, 1, 1, 1, 1, 0, 1])
    x = np.column_stack([np.arange(8) % 2, np.linspace(-1, 1, 8)])
    fit = cox_ph(data, x)
    sch = fit.schoenfeld_residuals()
    assert sch.residuals.shape == (data.events, 2)
    assert sch.scaled.shape == sch.residuals.shape
    assert fit.martingale_residuals().shape == (8,)
    assert fit.deviance_residuals().shape == (8,)
    ph = fit.proportional_hazards_test()
    assert ph.df == 2
    assert 0 <= ph.pvalue <= 1


def test_cox_delayed_entry_supported():
    data = SurvivalData([3, 4, 6, 7, 8], [1, 1, 0, 1, 1], entry=[0, 1, 2, 3, 0])
    x = np.array([[0], [1], [0], [1], [0]], dtype=float)
    fit = cox_ph(data, x)
    assert np.isfinite(fit.log_partial_likelihood)
    assert np.all(np.isfinite(fit.martingale_residuals()))


def test_exponential_intercept_only_matches_closed_form_without_censoring():
    times = np.array([1.0, 2.0, 3.0, 4.0])
    x = np.ones((4, 1))
    fit = exponential_regression(times, x)
    # scale MLE for an exponential with all events is the sample mean.
    assert fit.converged
    assert math.isclose(math.exp(fit.coefficients[0]), np.mean(times), rel_tol=2e-4)


def test_parametric_families_fit_and_predict():
    times = np.array([1.0, 1.5, 2.0, 2.8, 3.2, 4.0, 5.0, 6.0])
    event = np.array([1, 1, 0, 1, 1, 0, 1, 1])
    x = np.column_stack([np.ones(8), np.linspace(-1, 1, 8)])
    for fitter in (weibull_regression, lognormal_regression, loglogistic_regression):
        fit = fitter(times, x, event)
        assert np.isfinite(fit.log_likelihood)
        assert fit.ancillary > 0
        s1 = float(fit.survival(1.0, [1.0, 0.0]))
        s2 = float(fit.survival(5.0, [1.0, 0.0]))
        assert 0 <= s2 <= s1 <= 1
        assert float(fit.median([1.0, 0.0])) > 0


def test_delayed_entry_parametric_likelihood_is_finite():
    data = SurvivalData([2, 3, 4, 5, 6], [1, 0, 1, 1, 0], entry=[0, 1, 1, 2, 0])
    x = np.ones((5, 1))
    fit = parametric_survival_regression(data, x, "weibull")
    assert np.isfinite(fit.log_likelihood)


def test_public_cox_partial_log_likelihood_at_zero():
    from probstats.survival_regression import cox_partial_log_likelihood

    data = SurvivalData([1, 2, 3], [1, 1, 1])
    x = np.array([[0.0], [1.0], [2.0]])
    value = cox_partial_log_likelihood(data, x, [0.0], ties="breslow")
    assert math.isclose(value, -math.log(3) - math.log(2), rel_tol=1e-12)


def test_efron_schoenfeld_residuals_sum_to_zero_at_fit():
    data = SurvivalData([1, 1, 2, 3, 4, 5], [1, 1, 1, 1, 0, 1])
    x = np.array([[0.0], [1.0], [0.2], [1.2], [0.4], [0.8]])
    fit = cox_ph(data, x, ties="efron")
    assert abs(float(np.sum(fit.schoenfeld_residuals().residuals))) < 1e-6
