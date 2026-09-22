import math

import numpy as np

from probstats.modeling import (
    NegativeBinomialFamily,
    ancova,
    coefficient_contrast,
    compare_models,
    design_matrix,
    estimated_marginal_means,
    glm,
    joint_wald_test,
    linear_model,
)


def test_formula_transforms_polynomials_and_prediction_reuse():
    data = {"y": [1, 4, 9, 16], "x": [1, 2, 3, 4]}
    dm = design_matrix("y ~ log(x) + I(x**2) + poly(x, 3)", data)
    assert dm.column_names == (
        "Intercept",
        "log(x)",
        "I(x**2)",
        "poly(x,3)[1]",
        "poly(x,3)[2]",
        "poly(x,3)[3]",
    )
    transformed = dm.transform({"x": [2.0]})[0]
    np.testing.assert_allclose(transformed, [1, math.log(2), 4, 2, 4, 8])


def test_weighted_formula_ols():
    data = {"y": [0.0, 1.0, 10.0], "x": [0.0, 1.0, 2.0], "w": [10.0, 10.0, 0.1]}
    weighted = linear_model("y ~ x", data, weights="w")
    unweighted = linear_model("y ~ x", data)
    assert weighted.coefficients[1] < unweighted.coefficients[1]


def test_poisson_offset_exposure():
    exposure = np.array([1, 2, 4, 8, 16, 32], dtype=float)
    counts = 3 * exposure
    data = {"count": counts, "exposure": exposure}
    fit = glm("count ~ 1", data, family="poisson", offset=np.log(exposure))
    assert fit.converged
    assert abs(fit.coefficients[0] - math.log(3)) < 1e-6


def test_grouped_binomial_trials():
    data = {
        "successes": [1, 3, 7, 9, 2, 8],
        "trials": [10, 10, 10, 10, 10, 10],
        "x": [-2, -1, 1, 2, -1.5, 1.5],
    }
    fit = glm("successes ~ x", data, family="binomial", trials="trials")
    assert fit.trials is not None
    assert fit.coefficients[1] > 0
    assert math.isfinite(fit.log_likelihood)


def test_negative_binomial_and_quasi_poisson():
    data = {"y": [0, 1, 1, 2, 4, 8, 12, 20], "x": np.arange(8, dtype=float)}
    nb = glm("y ~ x", data, family=NegativeBinomialFamily(alpha=0.5))
    qp = glm("y ~ x", data, family="quasipoisson")
    assert nb.family == "negative_binomial"
    assert nb.coefficients[1] > 0
    assert qp.family == "quasipoisson"
    assert qp.dispersion > 0
    assert math.isnan(qp.aic)


def test_coefficient_contrast_and_joint_wald():
    x = np.arange(10, dtype=float)
    data = {"y": 1 + 2 * x + 0.5 * x**2, "x": x}
    fit = linear_model("y ~ x + I(x**2)", data)
    contrast = coefficient_contrast(fit, {"x": 1.0}, value=2.0)
    assert abs(contrast.estimate - 2.0) < 1e-10
    assert contrast.pvalue > 0.99
    joint = joint_wald_test(fit, [[0, 1, 0], [0, 0, 1]], value=[0, 0])
    assert joint.df == 2
    assert joint.pvalue < 1e-8


def test_estimated_marginal_means_balanced_grid():
    data = {
        "y": [1, 2, 3, 4, 5, 6, 7, 8],
        "group": ["a", "a", "a", "a", "b", "b", "b", "b"],
        "site": ["x", "x", "y", "y", "x", "x", "y", "y"],
    }
    fit = linear_model("y ~ C(group) + C(site)", data)
    emm = estimated_marginal_means(fit, "group")
    assert emm.mean("b").estimate > emm.mean("a").estimate
    assert abs((emm.mean("b").estimate - emm.mean("a").estimate) - 4) < 1e-10


def test_ancova_and_model_comparison():
    data = {
        "y": [1, 2, 3, 4, 3, 5, 7, 9],
        "x": [0, 1, 2, 3, 0, 1, 2, 3],
        "group": ["a"] * 4 + ["b"] * 4,
    }
    result = ancova("y ~ x + C(group)", data, type=2)
    assert result.term("x").df == 1
    reduced = linear_model("y ~ x", data)
    full = linear_model("y ~ x + C(group)", data)
    table = compare_models(reduced, full, names=("reduced", "full"))
    assert table.rows[1].df_difference == 1
    assert table.rows[1].pvalue is not None


def test_glm_deviance_comparison():
    x = np.arange(1, 9, dtype=float)
    data = {"y": [1, 1, 2, 3, 5, 8, 13, 20], "x": x}
    reduced = glm("y ~ 1", data, family="poisson")
    full = glm("y ~ x", data, family="poisson")
    comparison = compare_models(reduced, full)
    assert comparison.rows[1].statistic > 0
    assert comparison.rows[1].pvalue < 0.05


def test_intercept_only_regression_has_no_overall_predictor_test():
    data = {"y": [1.0, 2.0, 3.0, 4.0], "w": [1.0, 2.0, 1.0, 2.0]}
    ordinary = linear_model("y ~ 1", data)
    weighted = linear_model("y ~ 1", data, weights="w")
    assert ordinary.f_statistic == 0.0
    assert ordinary.f_pvalue == 1.0
    assert weighted.f_statistic == 0.0
    assert weighted.f_pvalue == 1.0


def test_glm_prediction_reuses_named_offset_column():
    exposure = np.array([1, 2, 4, 8, 16, 32], dtype=float)
    data = {
        "count": 3 * exposure,
        "log_exposure": np.log(exposure),
    }
    fit = glm("count ~ 1", data, family="poisson", offset="log_exposure")
    prediction = fit.predict({"log_exposure": [math.log(64.0)]})
    np.testing.assert_allclose(prediction, [192.0], rtol=1e-6)


def test_model_comparison_rejects_different_response_data():
    reduced = linear_model("y ~ x", {"y": [1, 2, 3, 4], "x": [0, 1, 2, 3]})
    full = linear_model(
        "y ~ x + I(x**2)",
        {"y": [1, 2, 4, 8], "x": [0, 1, 2, 3]},
    )
    import pytest

    with pytest.raises(ValueError, match="same response data"):
        compare_models(reduced, full)
