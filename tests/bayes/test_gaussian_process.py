import math

import numpy as np
import pytest

from probstats.bayes import (
    GaussianProcessRegressor,
    InferenceKind,
    RBFKernel,
)
from probstats.bayes.gp import (
    ConstantKernel,
    LinearKernel,
    gaussian_process_log_likelihood,
    predict_hyperparameter_mixture,
)


def test_rbf_kernel_matrix_is_symmetric_and_has_expected_diagonal():
    kernel = RBFKernel(length_scale=2.0, variance=3.0)
    x = [[0.0], [1.0], [2.0]]
    matrix = kernel.matrix(x)
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 3.0)
    assert matrix[0, 1] == pytest.approx(3.0 * math.exp(-1.0 / 8.0))


def test_kernel_sum_matches_expected_shape():
    kernel = ConstantKernel(2.0) + RBFKernel(length_scale=3.0, variance=1.0)
    matrix = kernel.matrix([[0.0], [1.0]])
    assert matrix[0, 0] == pytest.approx(3.0)
    assert matrix[0, 1] == pytest.approx(2.0 + math.exp(-1.0 / 18.0))


def test_gp_log_evidence_matches_direct_multivariate_normal_formula():
    x = np.array([[0.0], [1.0]])
    y = np.array([1.0, -0.5])
    kernel = RBFKernel(length_scale=1.2, variance=1.7)
    noise = 0.3
    model = GaussianProcessRegressor(kernel, noise_variance=noise, jitter=0.0)
    fit = model.fit(x, y)

    covariance = kernel.matrix(x) + noise * np.eye(2)
    sign, logdet = np.linalg.slogdet(covariance)
    expected = -0.5 * (
        2 * np.log(2 * np.pi) + logdet + y @ np.linalg.solve(covariance, y)
    )
    assert sign > 0
    assert fit.log_evidence == pytest.approx(expected, abs=1e-12)
    assert gaussian_process_log_likelihood(
        x, y, kernel, noise_variance=noise, jitter=0.0
    ) == pytest.approx(expected, abs=1e-12)


def test_prediction_matches_manual_gaussian_conditioning():
    x = np.array([[0.0], [1.0]])
    y = np.array([1.0, 2.0])
    x_star = np.array([[0.5]])
    kernel = RBFKernel(length_scale=1.0, variance=2.0)
    noise = 0.25
    fit = GaussianProcessRegressor(kernel, noise_variance=noise, jitter=0.0).fit(x, y)
    pred = fit.predict_distribution(x_star, observation=False).marginal(0)

    k = kernel.matrix(x) + noise * np.eye(2)
    ks = kernel.matrix(x, x_star).reshape(2)
    expected_mean = ks @ np.linalg.solve(k, y)
    expected_var = kernel.matrix(x_star)[0, 0] - ks @ np.linalg.solve(k, ks)
    assert pred.mean == pytest.approx(expected_mean, abs=1e-12)
    assert pred.variance == pytest.approx(expected_var, abs=1e-12)


def test_observation_prediction_adds_noise_variance():
    fit = GaussianProcessRegressor(RBFKernel(), noise_variance=0.4, jitter=0.0).fit(
        [[0.0]], [1.0]
    )
    latent = fit.predict_distribution([[0.3]], observation=False).marginal(0)
    observed = fit.predict_distribution([[0.3]], observation=True).marginal(0)
    assert observed.mean == pytest.approx(latent.mean)
    assert observed.variance == pytest.approx(latent.variance + 0.4)


def test_nonzero_mean_function_is_conditioned_correctly():
    mean = lambda points: np.asarray(points)[:, 0] + 10.0
    fit = GaussianProcessRegressor(
        RBFKernel(length_scale=1.0), noise_variance=0.1, mean_function=mean, jitter=0.0
    ).fit([[0.0], [1.0]], [10.0, 11.0])
    prediction = fit.predict_distribution([[2.0]], observation=False).marginal(0)
    # Residual observations are exactly zero, so posterior mean remains the prior mean.
    assert prediction.mean == pytest.approx(12.0)


def test_linear_kernel_cross_checks_bayesian_linear_model_with_known_noise():
    # beta ~ N(0, 1), y = x beta + eps, eps ~ N(0, sigma2) is a GP with k(x,x')=xx'.
    x = np.array([[1.0], [2.0]])
    y = np.array([1.0, 2.5])
    sigma2 = 0.5
    fit = GaussianProcessRegressor(
        LinearKernel(), noise_variance=sigma2, jitter=0.0
    ).fit(x, y)
    pred = fit.predict_distribution([[3.0]], observation=False).marginal(0)

    precision = 1.0 + float((x.T @ x)[0, 0]) / sigma2
    beta_var = 1.0 / precision
    beta_mean = beta_var * float((x.T @ y)[0]) / sigma2
    assert pred.mean == pytest.approx(3.0 * beta_mean, abs=1e-12)
    assert pred.variance == pytest.approx(9.0 * beta_var, abs=1e-12)


def test_hyperparameter_samples_produce_weighted_predictive_mixture():
    samples = [{"length_scale": 0.5}, {"length_scale": 2.0}]
    weights = [0.25, 0.75]

    def factory(params):
        return GaussianProcessRegressor(
            RBFKernel(length_scale=params["length_scale"]),
            noise_variance=0.1,
            jitter=0.0,
        )

    mixture = predict_hyperparameter_mixture(
        [[0.0], [1.0]], [0.0, 1.0], [[0.5]], samples, weights, factory
    )[0]
    component_means = np.array([component.mean for component in mixture.components])
    assert mixture.mean == pytest.approx(np.dot(np.array(weights), component_means))
    assert mixture.variance >= 0
    assert mixture.pdf(0.5) > 0


def test_gp_inference_result_has_exact_conditioning_provenance():
    result = GaussianProcessRegressor(RBFKernel(), noise_variance=0.1).infer(
        [[0.0], [1.0]], [0.0, 1.0]
    )
    assert result.kind is InferenceKind.EXACT
    assert result.steps[0].method == "gaussian-process-conditioning"
    assert result.metadata["evidence"] == pytest.approx(math.exp(result.log_evidence))
    assert "condition_number" in result.diagnostics


def test_invalid_shapes_and_singular_covariance_fail_cleanly():
    model = GaussianProcessRegressor(
        ConstantKernel(0.0), noise_variance=0.0, jitter=0.0
    )
    with pytest.raises(ValueError, match="positive definite"):
        model.fit([[0.0], [1.0]], [1.0, 2.0])
    with pytest.raises(ValueError, match="one-dimensional"):
        GaussianProcessRegressor(RBFKernel()).fit(
            [[0.0], [1.0]], [[1.0, 2.0], [3.0, 4.0]]
        )


def test_input_dependent_nugget_matches_diagonal_noise_semantics():
    nugget = lambda points: 0.1 + 0.2 * np.asarray(points)[:, 0] ** 2
    fit = GaussianProcessRegressor(RBFKernel(), noise_variance=nugget, jitter=0.0).fit(
        [[0.0], [1.0]], [0.0, 1.0]
    )
    assert np.diag(fit.covariance)[0] == pytest.approx(1.1)
    assert np.diag(fit.covariance)[1] == pytest.approx(1.3)
    latent = fit.predict_distribution([[2.0]], observation=False).marginal(0)
    observed = fit.predict_distribution([[2.0]], observation=True).marginal(0)
    assert observed.variance == pytest.approx(latent.variance + 0.9)
