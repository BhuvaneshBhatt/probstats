import numpy as np
import pytest

from probstats.inference import one_way_anova
from probstats.modeling import (
    BinomialFamily,
    SumContrast,
    TreatmentContrast,
    design_matrix,
    factorial_anova,
    glm,
    linear_model,
    parse_formula,
)


def test_formula_parser_star_expands_hierarchy_and_intercept_control():
    response, terms, intercept = parse_formula("y ~ x * C(group) + 0")
    assert response == "y"
    assert not intercept
    assert [term.name for term in terms] == ["x", "C(group)", "x:C(group)"]


def test_design_matrix_treatment_contrasts_and_interaction():
    data = {
        "y": [1, 2, 3, 4, 5, 6],
        "x": [0, 1, 0, 1, 0, 1],
        "group": ["a", "a", "b", "b", "c", "c"],
    }
    dm = design_matrix("y ~ x * C(group)", data)
    assert dm.matrix.shape == (6, 6)
    assert dm.column_names == (
        "Intercept",
        "x",
        "C(group)[T.b]",
        "C(group)[T.c]",
        "x:C(group)[T.b]",
        "x:C(group)[T.c]",
    )
    assert dm.term_slices["x:C(group)"] == (4, 5)
    new = dm.transform({"x": [1], "group": ["b"]})
    assert np.array_equal(new, [[1, 1, 1, 0, 1, 0]])


def test_sum_contrast_is_zero_sum_by_level():
    data = {"y": [1, 2, 3, 4, 5, 6], "g": ["a", "a", "b", "b", "c", "c"]}
    dm = design_matrix("y ~ C(g)", data, contrasts={"g": "sum"})
    encoded = dm.matrix[:, 1:]
    assert np.allclose(encoded.sum(axis=0), 0)
    assert isinstance(dm.factor_encodings["C(g)"].contrast, SumContrast)


def test_formula_linear_model_prediction_and_low_level_reuse():
    data = {"y": [1, 3, 5, 7, 9], "x": [0, 1, 2, 3, 4]}
    fit = linear_model("y ~ x", data)
    assert fit.coefficient_names == ("Intercept", "x")
    assert fit.coefficients == pytest.approx([1, 2])
    assert fit.predict({"x": [5, 6]}) == pytest.approx([11, 13])


def test_binomial_logit_glm_recovers_direction_and_predicts():
    x = np.array([-3, -2, -1, 0, 1, 2, 3] * 8, dtype=float)
    # Deterministic but non-separated pattern around a logistic trend.
    y = np.array(
        ([0, 0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 0, 1, 1]) * 4, dtype=float
    ).reshape(-1)
    fit = glm("y ~ x", {"y": y, "x": x}, family="binomial")
    assert fit.converged
    assert fit.coefficients[1] > 0
    pred = fit.predict({"x": [-2, 2]})
    assert 0 < pred[0] < pred[1] < 1
    assert fit.family == "binomial" and fit.link == "logit"


def test_poisson_log_glm():
    x = np.repeat(np.arange(4, dtype=float), 5)
    y = np.array(
        [1, 1, 2, 1, 1, 2, 3, 2, 2, 3, 4, 5, 4, 5, 6, 8, 9, 10, 8, 9], dtype=float
    )
    fit = glm("y ~ x", {"y": y, "x": x}, family="poisson")
    assert fit.converged
    assert fit.coefficients[1] > 0
    assert np.all(fit.fitted_mean > 0)
    assert np.isfinite(fit.aic)


def test_gaussian_glm_matches_ols_coefficients():
    data = {"y": [1, 3, 5, 7, 9, 11], "x": [0, 1, 2, 3, 4, 5]}
    ols = linear_model("y ~ x", data)
    gl = glm("y ~ x", data, family="gaussian")
    assert gl.coefficients == pytest.approx(ols.coefficients, abs=1e-8)


def test_factorial_anova_detects_main_and_interaction_effects():
    # Balanced 2x2 design with strong A main effect and A:B interaction.
    a = []
    b = []
    y = []
    cells = {("a0", "b0"): 0, ("a0", "b1"): 0, ("a1", "b0"): 4, ("a1", "b1"): 9}
    jitter = [-0.2, 0.0, 0.2, -0.1, 0.1]
    for (ai, bi), mean in cells.items():
        for e in jitter:
            a.append(ai)
            b.append(bi)
            y.append(mean + e)
    data = {"y": y, "a": a, "b": b}
    result = factorial_anova("y ~ C(a) * C(b)", data, type=2)
    assert result.term("C(a)").pvalue < 1e-6
    assert result.term("C(a):C(b)").pvalue < 1e-6
    assert result.residual_df == 16


def test_type_three_anova_with_sum_contrasts_and_shared_terms():
    data = {
        "y": [1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
        "a": ["a", "a", "a", "a", "b", "b", "b", "b", "c", "c", "c", "c"],
        "b": ["x", "x", "y", "y"] * 3,
    }
    result = factorial_anova(
        "y ~ C(a) * C(b)", data, type=3, contrasts={"a": "sum", "b": "sum"}
    )
    assert {term.term for term in result.terms} == {"C(a)", "C(b)", "C(a):C(b)"}
    assert all(term.df > 0 for term in result.terms)


def test_explicit_contrast_objects_are_supported():
    data = {"y": [1, 2, 3, 4], "g": ["a", "b", "a", "b"]}
    dm = design_matrix(
        "y ~ C(g)", data, contrasts={"g": TreatmentContrast(reference="b")}
    )
    assert dm.column_names[1] == "C(g)[T.a]"
    assert isinstance(BinomialFamily(), BinomialFamily)


def test_formula_one_factor_anova_matches_one_way_decomposition():
    groups = ([1.0, 2.0, 1.5, 2.2], [4.0, 5.0, 4.5, 5.2], [8.0, 9.0, 8.5, 9.2])
    direct = one_way_anova(*groups)
    y = [value for group in groups for value in group]
    g = [label for label, group in enumerate(groups) for _ in group]
    formula_result = factorial_anova("y ~ C(g)", {"y": y, "g": g}, type=3)
    term = formula_result.term("C(g)")
    assert term.sum_squares == pytest.approx(direct.ss_between)
    assert formula_result.residual_sum_squares == pytest.approx(direct.ss_within)
    assert term.statistic == pytest.approx(direct.statistic)
    assert term.pvalue == pytest.approx(direct.pvalue)


def test_formula_regression_overall_f_excludes_intercept_df():
    fit = linear_model("y ~ x", {"y": [1, 3, 5, 7, 9, 11], "x": [0, 1, 2, 3, 4, 5]})
    assert fit.df_resid == 4
    assert fit.f_statistic > 1e10
