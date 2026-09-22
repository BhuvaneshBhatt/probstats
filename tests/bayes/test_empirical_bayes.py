import math

import numpy as np
import pytest
import sympy as sp

from probstats.bayes import (
    BayesianMultivariateLinearRegression,
    Factor,
    Model,
    Parameter,
    RandomVariable,
)
from probstats.bayes.empirical_bayes import (
    EvidenceEvaluation,
    Hyperparameter,
    mackay_fixed_point,
    mackay_laplace,
    mackay_precision_noise_update,
    mackay_precision_update,
    optimize_evidence,
    optimize_laplace_evidence,
)
from probstats.bayes.laplace import GaussianLaplacePosterior
from probstats.bayes.models import MatrixNormalInverseWishartPrior
from probstats.distributions import Normal


def test_hyperparameter_transforms_round_trip():
    specs = [
        Hyperparameter("x", 2.0),
        Hyperparameter("alpha", 3.0, transform="log"),
        Hyperparameter("rho", 0.25, transform="logit", bounds=(0.0, 1.0)),
    ]
    for spec in specs:
        got = spec.from_unconstrained(spec.to_unconstrained(spec.initial))
        assert math.isclose(got, spec.initial, rel_tol=0, abs_tol=1e-12)


def test_hyperparameter_bounds_and_optimizer_options_are_validated():
    with pytest.raises(ValueError, match="within"):
        Hyperparameter("x", 2.0, bounds=(0.0, 1.0))
    with pytest.raises(ValueError, match="finite"):
        Hyperparameter("x", 1.0, bounds=(0.0, math.inf))

    objective = lambda params: -(params["x"] ** 2)
    with pytest.raises(ValueError, match="tolerance"):
        optimize_evidence(
            objective,
            [Hyperparameter("x", 1.0)],
            method="coordinate",
            tolerance=math.nan,
        )
    with pytest.raises(ValueError, match="initial_step"):
        optimize_evidence(
            objective,
            [Hyperparameter("x", 1.0)],
            method="coordinate",
            options={"initial_step": 0.0},
        )


def test_evidence_results_are_structurally_immutable():
    result = optimize_evidence(
        lambda params: -((params["x"] - 2.0) ** 2),
        [Hyperparameter("x", 0.0)],
        method="coordinate",
        estimate_covariance=True,
    )
    with pytest.raises(TypeError):
        result.parameters["x"] = 3.0
    with pytest.raises(TypeError):
        result.history[0][0]["x"] = 3.0
    if result.hyper_covariance is not None:
        with pytest.raises(ValueError):
            result.hyper_covariance[0, 0] = 1.0


def test_optimize_evidence_generic_positive_hyperparameter():
    def objective(params):
        x = params["scale"]
        return EvidenceEvaluation(-((math.log(x) - math.log(2.5)) ** 2) + 3.0, fit=x)

    result = optimize_evidence(
        objective,
        [Hyperparameter("scale", 0.5, transform="log")],
        method="coordinate",
        tolerance=1e-8,
    )
    assert result.success
    assert math.isclose(result.parameters["scale"], 2.5, rel_tol=1e-5)
    assert math.isclose(result.log_evidence, 3.0, rel_tol=0, abs_tol=1e-10)
    assert result.fit is not None
    assert result.history


def test_generic_evidence_optimizer_on_exact_mniw_evidence():
    x = [[1, 0], [1, 1], [1, 2], [1, 3]]
    y = [[1, 2], [2, 1], [3, 4], [5, 3]]

    def objective(params):
        alpha = params["alpha"]
        prior = MatrixNormalInverseWishartPrior(
            sp.zeros(2, 2), sp.eye(2) * alpha, sp.eye(2), 3
        )
        return BayesianMultivariateLinearRegression(prior=prior).fit(x, y)

    result = optimize_evidence(
        objective,
        [Hyperparameter("alpha", 1.0, transform="log")],
        method="coordinate",
        tolerance=2e-6,
    )
    grid = np.geomspace(0.05, 2.0, 80)
    grid_values = []
    for alpha in grid:
        fit = objective({"alpha": float(alpha)})
        grid_values.append(float(sp.N(fit.log_evidence)))
    grid_best = float(grid[int(np.argmax(grid_values))])
    assert abs(math.log(result.parameters["alpha"] / grid_best)) < 0.04
    assert hasattr(result.fit, "posterior")


def _gaussian_model_factory(params):
    alpha = params["alpha"]
    theta = Parameter("theta")
    obs = RandomVariable("y")
    return Model(
        (theta, obs),
        (
            Factor.from_distribution(theta, Normal(0, 1 / math.sqrt(alpha))),
            Factor.from_distribution(obs, Normal(theta.symbol, 1)),
        ),
    ).observe(y=2)


def test_optimize_laplace_evidence_recovers_gaussian_type2_optimum():
    result = optimize_laplace_evidence(
        _gaussian_model_factory,
        [Hyperparameter("alpha", 1.0, transform="log")],
        laplace_options={"backend": "sympy"},
        method="coordinate",
        tolerance=2e-6,
    )
    assert math.isclose(result.parameters["alpha"], 1 / 3, rel_tol=2e-4)


def test_mackay_one_precision_update_and_laplace_fixed_point():
    mean = np.array([1.5, -0.5])
    covariance = np.diag([0.2, 0.3])
    expected = 2 / (2.5 + 0.5)
    assert math.isclose(mackay_precision_update(1.0, mean, covariance), expected)

    result = mackay_laplace(
        _gaussian_model_factory,
        initial_alpha=1.0,
        laplace_options={"backend": "sympy"},
        tolerance=1e-9,
    )
    assert result.converged
    assert math.isclose(result.parameters["alpha"], 1 / 3, rel_tol=1e-7)
    assert result.history


def test_mackay_two_precision_formula_and_generic_iteration():
    mean = np.array([1.0, 2.0])
    covariance = np.diag([0.1, 0.2])
    alpha_new, beta_new = mackay_precision_noise_update(
        2.0,
        4.0,
        mean,
        covariance,
        n_observations=10,
        squared_error=3.0,
    )
    assert math.isclose(alpha_new, 2 / 5.3)
    gamma = 2 - 2 * 0.3
    assert math.isclose(beta_new, (10 - gamma) / 3.0)

    target_alpha = 0.8

    class Fit:
        log_evidence = 0.0
        posterior = GaussianLaplacePosterior(
            (sp.Symbol("w"),),
            (1.0,),
            np.array([[0.25]]),
            np.array([[4.0]]),
        )

    # Smoke the generic fixed-point protocol independently of Model/Laplace rebuilding.
    result = mackay_fixed_point(
        lambda params: EvidenceEvaluation(
            -((math.log(params["alpha"]) - math.log(target_alpha)) ** 2), Fit()
        ),
        initial_alpha=1.0,
        max_iterations=3,
    )
    assert result.iterations >= 1
    assert result.log_evidence <= 0
