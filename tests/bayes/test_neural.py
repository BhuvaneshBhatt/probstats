import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from probstats.bayes import infer
from probstats.bayes.diagnostics import QualityLevel
from probstats.bayes.neural import (
    JAXRegressionNetwork,
    RegressionNetworkConfig,
    alpha_divergence_loss,
    batchnorm_to_chain,
    fit_regression_network,
    flatten_parameters,
    gaussian_loss,
    network_objective,
    network_regularization_loss,
    regression_network,
    sample_trained_network,
)
from probstats.bayes.planner import PlannerConfig


def test_gaussian_loss_matches_supported_parameterizations():
    y = jnp.array([1.0, 2.0])
    pred = jnp.array([[2.0, math.log(4.0)], [1.0, math.log(0.5)]])
    got = np.asarray(gaussian_loss(y, pred))
    expected = np.array([4.0 - math.log(4.0), 0.5 - math.log(0.5)])
    assert np.allclose(got, expected)


def test_alpha_divergence_special_cases_and_finite_formula():
    losses = jnp.array([1.0, 2.0, 4.0])
    assert np.isclose(float(alpha_divergence_loss(losses, 0)), np.mean([1, 2, 4]))
    assert np.isclose(float(alpha_divergence_loss(losses, math.inf)), 1.0)
    assert np.isclose(float(alpha_divergence_loss(losses, -math.inf)), 4.0)
    alpha = 0.5
    expected = -math.log(np.mean(np.exp(-alpha * np.array([1, 2, 4])))) / alpha
    assert np.isclose(float(alpha_divergence_loss(losses, alpha)), expected)


def test_regression_network_heteroscedastic_and_homoscedastic_shapes():
    hetero = regression_network(2, network_depth=2, layer_size=5, dropout_probability=0)
    state = hetero.initialize(1)
    pred, _ = hetero.apply(state, np.ones((3, 2)))
    assert pred.shape == (3, 2)

    homo = regression_network(
        2,
        network_depth=1,
        layer_size=4,
        dropout_probability=0,
        error_model="homoscedastic",
    )
    state2 = homo.initialize(2)
    pred2, _ = homo.apply(state2, np.ones((3, 2)))
    assert pred2.shape == (3, 2)
    assert np.allclose(np.asarray(pred2[:, 1]), np.asarray(pred2[0, 1]))


def test_mc_dropout_predictive_decomposes_uncertainty():
    net = regression_network(1, network_depth=2, layer_size=8, dropout_probability=0.4)
    state = net.initialize(3)
    predictive = sample_trained_network(net, state, [[0.0], [1.0]], samples=30, rng=4)
    assert predictive.mean.shape == (2,)
    assert predictive.samples.shape == (30, 2, 2)
    assert np.all(predictive.standard_deviation > 0)
    assert np.any(predictive.epistemic_variance > 0)
    assert np.allclose(
        predictive.variance,
        predictive.epistemic_variance + predictive.aleatoric_variance,
    )


def test_batchnorm_freeze_retains_stochastic_dropout_without_state_updates():
    net = JAXRegressionNetwork(
        RegressionNetworkConfig(
            input_dim=1,
            depth=1,
            layer_size=6,
            dropout_probability=0.3,
            batch_normalization=True,
        )
    )
    state = net.initialize(5)
    # Update moving statistics once.
    _, updated = net.apply(
        state,
        np.arange(8.0)[:, None],
        training=True,
        rng=jax.random.PRNGKey(6),
        update_batch_stats=True,
    )
    frozen = batchnorm_to_chain(net, updated)
    a = np.asarray(frozen(np.ones((4, 1)), training=True, rng=jax.random.PRNGKey(7)))
    b = np.asarray(frozen(np.ones((4, 1)), training=True, rng=jax.random.PRNGKey(8)))
    assert a.shape == b.shape == (4, 2)
    assert not np.allclose(a, b)
    assert all(
        np.allclose(np.asarray(x["mean"]), np.asarray(y["mean"]))
        for x, y in zip(updated.batch_stats, frozen.state.batch_stats)
    )


def test_flatten_parameters_round_trip():
    net = regression_network(1, network_depth=1, layer_size=3)
    state = net.initialize(9)
    flat, restore = flatten_parameters(state)
    rebuilt = restore(flat)
    pred1, _ = net.apply(state, [[0.25]])
    pred2, _ = net.apply(rebuilt, [[0.25]])
    assert flat.ndim == 1 and flat.size > 0
    assert np.allclose(np.asarray(pred1), np.asarray(pred2))


def test_network_objective_is_finite_and_regularized():
    net = regression_network(1, network_depth=1, layer_size=4, dropout_probability=0.2)
    state = net.initialize(10)
    x = np.linspace(-1, 1, 6)[:, None]
    y = x[:, 0]
    loose = float(network_objective(net, state, x, y, 0.0, samples=5, rng=11))
    penalized = float(network_objective(net, state, x, y, 0.1, samples=5, rng=11))
    assert np.isfinite(loose)
    assert penalized < loose
    assert float(network_regularization_loss(state.params, 0.1, 2)) > 0


def test_small_training_run_reduces_simple_regression_loss():
    net = regression_network(
        1,
        network_depth=1,
        layer_size=8,
        activation_function="tanh",
        dropout_probability=0.0,
        error_model="homoscedastic",
    )
    x = np.linspace(-1, 1, 20)[:, None]
    y = 1.5 * x[:, 0] - 0.25
    fit = fit_regression_network(
        net,
        x,
        y,
        epochs=40,
        learning_rate=0.02,
        lambda_=1e-5,
        batch_size=20,
        rng=12,
    )
    assert np.isfinite(fit.history.final_loss)
    assert fit.history.final_loss < fit.history.losses[0]
    pred = fit.predict([[0.0]], samples=5, rng=13)
    assert pred.mean.shape == (1,)


def test_planner_dispatches_jax_network_and_attaches_quality():
    net = regression_network(
        1,
        network_depth=1,
        layer_size=4,
        dropout_probability=0.1,
        error_model="homoscedastic",
    )
    x = np.linspace(-1, 1, 8)[:, None]
    y = x[:, 0]
    result = infer(
        net,
        x=x,
        y=y,
        epochs=4,
        learning_rate=0.01,
        rng=14,
        config=PlannerConfig(assess_diagnostics=True),
    )
    assert result.metadata["planner"]["selected"] == "jax-mc-dropout-regression"
    assert result.kind.value == "approximate"
    assert result.quality.level is QualityLevel.CAUTION
    assert "dropout" in " ".join(result.quality.reasons).lower()
