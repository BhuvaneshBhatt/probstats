import numpy as np
import pytest
import sympy as sp

from probstats.bayes import (
    BayesianLinearRegression,
    Factor,
    GaussianProcessRegressor,
    InferenceKind,
    InferenceResult,
    Model,
    Parameter,
    RandomVariable,
)
from probstats.bayes.diagnostics import (
    QualityLevel,
    assess_quality,
    effective_sample_size,
    mcse_mean,
    sampled_diagnostics,
    split_rhat,
    to_posterior_data,
)
from probstats.bayes.gp import LinearKernel
from probstats.bayes.laplace import SingularLaplacePosterior, infer_laplace
from probstats.bayes.nested import infer_nested
from probstats.distributions import Normal


def test_scalar_chain_diagnostics_good_independent_draws():
    rng = np.random.default_rng(123)
    x = rng.normal(size=(4, 2000))
    assert 0.98 < split_rhat(x) < 1.02
    assert effective_sample_size(x) > 3000
    assert mcse_mean(x) < 0.03


def test_rhat_detects_shifted_chain():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(4, 1000))
    x[-1] += 2.0
    assert split_rhat(x) > 1.1


def test_laplace_to_posterior_data_and_quality():
    theta = Parameter("theta")
    y = RandomVariable("y")
    model = Model(
        (theta, y),
        (
            Factor.from_distribution(theta, Normal(0, 1)),
            Factor.from_distribution(y, Normal(theta.symbol, 1)),
        ),
    ).observe(y=1)
    result = infer_laplace(model, backend="sympy")
    data = to_posterior_data(result, draws=50, chains=2, rng=1)
    assert data.posterior["theta"].shape == (2, 50)
    quality = assess_quality(result)
    assert quality.level is QualityLevel.CAUTION
    assert quality.trustworthy


def test_regression_and_gp_common_posterior_data():
    reg = BayesianLinearRegression().infer([[1], [2], [3]], [1, 2, 3])
    rdata = to_posterior_data(reg, draws=25, chains=2, rng=3)
    assert rdata.posterior["beta"].shape[:2] == (2, 25)
    assert rdata.posterior["sigma2"].shape == (2, 25)

    gp = GaussianProcessRegressor(LinearKernel(), noise_variance=0.1).infer(
        [[1], [2]], [1, 2]
    )
    gdata = to_posterior_data(gp, draws=25, chains=2, rng=4)
    assert gdata.posterior["f"].shape == (2, 25, 2)
    assert assess_quality(gp).trustworthy


def test_nested_single_run_has_ess_and_rhat_unavailable_quality():
    def prior(rng):
        return np.array([rng.normal()])

    def logl(p):
        return -0.5 * p[0] ** 2

    result = infer_nested(
        logl, prior, ndim=1, n_live=30, min_iterations=20, max_iterations=200, rng=5
    )
    data = to_posterior_data(result, draws=50, chains=2, rng=6)
    stats = sampled_diagnostics(data)
    assert "theta_0" in stats
    # Resampling one weighted run into several pseudo-chains must not manufacture R-hat evidence.
    quality = assess_quality(result, posterior_data=data)
    assert any("R-hat is unavailable" in reason for reason in quality.reasons)


def test_arviz_interop_when_installed():
    pytest.importorskip("arviz")
    result = BayesianLinearRegression().infer([[1], [2]], [1, 2])
    idata = result.to_arviz(draws=20, chains=2, rng=7)
    assert hasattr(idata, "posterior")
    assert "beta" in idata.posterior


def test_planner_attaches_quality_report():
    from probstats.bayes import infer

    theta = Parameter("theta")
    y = RandomVariable("y")
    model = Model(
        (theta, y),
        (
            Factor.from_distribution(theta, Normal(0, 1)),
            Factor.from_distribution(y, Normal(theta.symbol, 1)),
        ),
    ).observe(y=1)
    result = infer(model)
    assert result.quality is not None
    assert result.quality.level is QualityLevel.GOOD
    assert result.metadata["planner"]["trustworthy"] is True


def test_singular_laplace_converts_to_posterior_data_and_is_cautionary():
    x = sp.Symbol("x", real=True)
    posterior = SingularLaplacePosterior((x,), (0.0,), (4,), (sp.Integer(1),))
    result = InferenceResult(
        posterior=posterior,
        kind=InferenceKind.APPROXIMATE,
        diagnostics={"singular_laplace": True, "asymptotic_certified": True},
    )
    data = to_posterior_data(result, chains=2, draws=2000, rng=123)
    assert data.posterior["x"].shape == (2, 2000)
    assert data.attrs["source"] == "singular-laplace"
    assert abs(float(np.mean(data.posterior["x"]))) < 0.1
    quality = assess_quality(result)
    assert quality.trustworthy is True
    assert quality.level.value == "caution"
    assert quality.metrics["asymptotic_certified"] is True
