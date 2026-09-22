import numpy as np
import pandas as pd
import sympy as sp

import probstats
from probstats import Normal, mean, median, quantile, standard_deviation, variance
from probstats.bayes import (
    BayesianLinearRegression,
    GaussianProcessRegressor,
    InferenceKind,
    InferenceResult,
    PosteriorData,
    RBFKernel,
    compare_models,
    posterior_predictive,
    predict,
    predict_distribution,
    predictive_interval,
)
from probstats.distributions import Bernoulli, Geometric
from probstats.inference import linear_regression
from probstats.modeling import glm, linear_model
from probstats.testing import one_sample_t_test


def test_result_protocol_and_coefficient_tables():
    test = one_sample_t_test([1.0, 2.0, 3.0])
    assert "HypothesisTestResult" in test.summary()
    assert test.to_dict()["method"] == "one-sample t-test"
    assert list(test.to_frame().columns)

    frame = pd.DataFrame({"y": [1.0, 2.0, 4.0, 5.0], "x": [0.0, 1.0, 2.0, 3.0]})
    fit = linear_model("y ~ x", frame)
    table = fit.coefficient_table()
    assert list(table.index) == ["Intercept", "x"]
    assert list(table.columns) == ["estimate", "standard_error", "statistic", "pvalue"]


def test_pandas_formula_input_preserves_prediction_index():
    frame = pd.DataFrame(
        {"y": [0.0, 1.0, 2.0, 3.0], "x": [0.0, 1.0, 2.0, 3.0]},
        index=["a", "b", "c", "d"],
    )
    fit = linear_model("y ~ x", frame)
    new = pd.DataFrame({"x": [4.0, 5.0]}, index=["future-a", "future-b"])
    pred = fit.predict(new)
    assert isinstance(pred, pd.Series)
    assert pred.index.tolist() == ["future-a", "future-b"]
    assert pred.name == "y"

    gfit = glm("y ~ x", frame)
    gpred = gfit.predict(new)
    assert isinstance(gpred, pd.Series)
    assert gpred.index.equals(new.index)


def test_direct_dataframe_regression_retains_feature_and_response_labels():
    x = pd.DataFrame(
        {"temperature": [1.0, 2.0, 3.0, 4.0], "rain": [0.0, 1.0, 0.0, 1.0]},
        index=["a", "b", "c", "d"],
    )
    y = pd.Series([2.0, 4.0, 6.0, 8.0], index=x.index, name="growth")
    fit = linear_regression(x, y)
    assert fit.feature_names == ("Intercept", "temperature", "rain")
    assert fit.response_name == "growth"
    assert list(fit.coefficient_table().index) == ["Intercept", "temperature", "rain"]
    future = pd.DataFrame(
        {"temperature": [5.0, 6.0], "rain": [0.0, 1.0]},
        index=["e", "f"],
    )
    prediction = fit.predict(future)
    assert isinstance(prediction, pd.Series)
    assert prediction.index.equals(future.index)
    assert prediction.name == "growth"


def test_array_probability_functionals_preserve_shape():
    dist = Normal(0, 1)
    x = np.array([[-1.0, 0.0], [1.0, 2.0]])
    log_values = dist.logpdf(x)
    pdf_values = dist.pdf(x)
    values = dist.cdf(x)
    hazard_values = dist.hazard(x)
    assert log_values.shape == x.shape
    assert pdf_values.shape == x.shape
    assert values.shape == x.shape
    assert hazard_values.shape == x.shape
    assert values.dtype == object
    assert values[0, 1] == sp.Rational(1, 2)
    quantiles = dist.quantile(np.array([0.25, 0.5, 0.75]))
    assert quantiles.shape == (3,)

    for discrete in (Bernoulli(sp.Rational(1, 3)), Geometric(sp.Rational(1, 2))):
        values = discrete.pdf(np.array([0, 1, 2]))
        assert values.shape == (3,)
        assert values.dtype == object


def test_domain_namespaces_expose_specialized_workflows():
    assert hasattr(probstats.stats, "linear_regression")
    assert hasattr(probstats.smoothing, "lowess")
    assert hasattr(probstats.random_matrix, "TracyWidom")
    assert hasattr(probstats.survival, "kaplan_meier")


def test_bayesian_model_comparison_and_predictive_protocol():
    fit1 = BayesianLinearRegression().fit([[1], [2], [3]], [1, 2, 3])
    fit2 = BayesianLinearRegression().fit([[1], [2], [3]], [1, 2, 4])
    comparison = compare_models([fit1, fit2], names=["linear-a", "linear-b"])
    assert comparison.log_bayes_factors.shape == (2, 2)
    assert np.isclose(comparison.posterior_probabilities.sum(), 1.0)
    assert comparison.best_model in comparison.names
    assert list(comparison.to_frame().index) == ["linear-a", "linear-b"]
    assert np.isclose(
        comparison.bayes_factor("linear-a", "linear-b"),
        np.exp(comparison.log_bayes_factor("linear-a", "linear-b")),
    )
    assert np.isclose(
        comparison.posterior_probability("linear-a"),
        comparison.posterior_probabilities[0],
    )

    laws = predict_distribution(fit1, [[4], [5]])
    assert len(laws) == 2
    means = predict(fit1, [[4], [5]])
    assert means.shape == (2,)
    intervals = predictive_interval(fit1, [[4], [5]], level=0.8)
    assert len(intervals) == 2


def test_gp_predictive_protocol_supports_sampling_and_intervals():
    gp = GaussianProcessRegressor(RBFKernel(1.0, 1.0), noise_variance=0.1).fit(
        [[0.0], [1.0], [2.0]], [0.0, 1.0, 0.0]
    )
    points = [[0.5], [1.5]]
    law = predict_distribution(gp, points)
    assert law.mean.shape == (2,)
    draws = posterior_predictive(gp, points, size=7, rng=123)
    assert np.asarray(draws).shape == (7, 2)
    intervals = predictive_interval(gp, points)
    assert len(intervals) == 2


def test_sampled_inference_uses_empirical_predictive_contract():
    posterior = PosteriorData({"slope": np.full((2, 4), 2.0)})
    result = InferenceResult(posterior=posterior, kind=InferenceKind.SAMPLED)

    def predictive(draw, x, *, rng):
        return draw["slope"] * np.asarray(x, dtype=float)

    law = result.predict_distribution([1.0, 3.0], predictive=predictive, size=20, rng=5)
    assert np.allclose(law.mean, [2.0, 6.0])
    assert np.allclose(
        result.predict([1.0, 3.0], predictive=predictive, size=20, rng=5), [2.0, 6.0]
    )
    draws = result.posterior_predictive(
        [1.0, 3.0], predictive=predictive, size=7, rng=5
    )
    assert draws.shape == (7, 2)
    lower, upper = result.predictive_interval(
        [1.0, 3.0], predictive=predictive, size=20, rng=5
    )
    assert np.allclose(lower, [2.0, 6.0])
    assert np.allclose(upper, [2.0, 6.0])


def test_multivariate_bayesian_fit_uses_same_predictive_contract():
    from probstats.bayes import BayesianMultivariateLinearRegression

    fit = BayesianMultivariateLinearRegression().fit(
        [[1], [2], [3]], [[1, 2], [2, 1], [3, 0]]
    )
    mean = fit.predict([[4], [5]])
    assert mean.shape == (2, 2)
    draws = fit.posterior_predictive([[4], [5]], size=3, rng=7)
    assert draws.shape == (3, 2, 2)
    intervals = fit.predictive_interval([[4], [5]], level=0.8)
    assert len(intervals) == 2
    assert len(intervals[0]) == 2


def test_root_statistics_dispatch_between_data_and_distributions():
    data = [1.0, 2.0, 3.0, 4.0]
    dist = Normal(0, 1)

    assert mean(data) == 2.5
    assert mean(dist) == 0
    assert median(data) == 2.5
    assert median(dist) == 0
    assert quantile(data, 0.5) == 2.5
    assert quantile(dist, 0.5) == 0
    assert np.isclose(variance(data), np.var(data, ddof=1))
    assert variance(dist) == 1
    assert np.isclose(standard_deviation(data), np.std(data, ddof=1))
    assert standard_deviation(dist) == 1


def test_distribution_statistics_reject_sample_only_options():
    dist = Normal(0, 1)
    with np.testing.assert_raises(TypeError):
        quantile(dist, 0.5, method="nearest")
    with np.testing.assert_raises(ValueError):
        variance(dist, ddof=1)
    with np.testing.assert_raises(ValueError):
        standard_deviation(dist, ddof=1)
    with np.testing.assert_raises(TypeError):
        mean(dist, weights=[1.0])
