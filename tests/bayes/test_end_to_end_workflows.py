"""End-to-end Bayesian workflows spanning model construction, planning, inference, and prediction."""

import sympy as sp

from probstats.bayes import (
    Factor,
    InferenceKind,
    Model,
    NormalInverseGamma,
    Parameter,
    RandomVariable,
)
from probstats.bayes.planner import (
    InferencePlanner,
    PlannerConfig,
)
from probstats.distributions import Normal


def _gaussian_location_model(observation=1):
    observation = sp.sympify(observation)
    theta = Parameter("workflow_theta", sp.S.Reals)
    y = RandomVariable("workflow_y", sp.S.Reals)
    model = Model(
        (theta, y),
        (
            Factor.from_distribution(theta, Normal(0, 1)),
            Factor.from_distribution(y, Normal(theta.symbol, 1)),
        ),
    ).observe(workflow_y=observation)
    return model


def test_planner_to_conjugate_posterior_predictive_and_evidence_workflow():
    model = _gaussian_location_model()
    mu = sp.Symbol("mu", real=True)
    sigma = sp.Symbol("sigma", positive=True)
    likelihood = Normal(mu, sigma)
    prior = NormalInverseGamma(0, 2, 3, 4)
    data = [1, 2, 4]

    planner = InferencePlanner(config=PlannerConfig(max_exact_operations=0))
    plan = planner.plan(model, likelihood=likelihood, data=data, prior=prior)
    assert plan.selected is not None
    assert plan.selected.method == "conjugacy"

    result = planner.infer(model, likelihood=likelihood, data=data, prior=prior)
    assert result.kind is InferenceKind.EXACT
    assert result.log_evidence is not None
    assert result.metadata["planner"]["selected"] == "conjugacy"
    assert result.metadata["posterior_predictive"] == result.posterior.predictive
    assert result.metadata["posterior_predictive"].df == 2 * result.posterior.nu


def test_symbolic_workflow_posterior_evidence():
    from probstats.bayes.exact import (
        infer_exact,
        integrate_over,
    )

    model = _gaussian_location_model()
    result = infer_exact(model)
    posterior = result.posterior

    assert result.kind is InferenceKind.EXACT
    assert result.log_evidence is not None
    normalized = integrate_over(posterior.density, model.latent_variables)
    assert sp.simplify(normalized - 1) == 0
    assert sp.simplify(sp.exp(result.log_evidence) - result.metadata["evidence"]) == 0
