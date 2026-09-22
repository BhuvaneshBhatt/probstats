import numpy as np
import sympy as sp

from probstats.bayes import (
    BayesianLinearRegression,
    Factor,
    GaussianProcessRegressor,
    InferenceKind,
    Model,
    NormalInverseGamma,
    Parameter,
    RandomVariable,
    RBFKernel,
    infer,
    plan_inference,
)
from probstats.bayes.models import LinearRegressionPrior
from probstats.bayes.planner import (
    InferencePlanner,
    PlannerConfig,
)
from probstats.distributions import Normal


def gaussian_model():
    theta = Parameter("theta")
    y = RandomVariable("y")
    return Model(
        (theta, y),
        (
            Factor.from_distribution(theta, Normal(0, 1)),
            Factor.from_distribution(y, Normal(theta.symbol, 1)),
        ),
    ).observe(y=1)


def test_planner_prefers_symbolic_exact_for_small_model():
    model = gaussian_model()
    plan = plan_inference(model)
    assert plan.selected.method == "symbolic-exact"
    result = infer(model)
    assert result.kind is InferenceKind.EXACT
    assert result.metadata["planner"]["selected"] == "symbolic-exact"
    assert result.steps[0].method == "planner-dispatch"


def test_planner_can_choose_conjugacy_when_symbolic_budget_disabled():
    model = gaussian_model()
    likelihood = Normal(sp.Symbol("mu"), sp.Symbol("sigma", positive=True))
    prior = NormalInverseGamma(0, 1, 1, 1)
    planner = InferencePlanner(config=PlannerConfig(max_exact_operations=0))
    plan = planner.plan(model, likelihood=likelihood, data=[1, 2], prior=prior)
    assert plan.selected.method == "conjugacy"
    result = planner.infer(model, likelihood=likelihood, data=[1, 2], prior=prior)
    assert result.kind is InferenceKind.EXACT
    assert result.metadata["planner"]["selected"] == "conjugacy"


def test_planner_routes_linear_regression_analytically():
    prior = LinearRegressionPrior([0], [[1]], 2, 2)
    model = BayesianLinearRegression(prior=prior)
    plan = plan_inference(model, x=[[1], [2]], y=[1, 2])
    assert plan.selected.method == "analytic-linear-regression"
    result = infer(model, x=[[1], [2]], y=[1, 2])
    assert result.kind is InferenceKind.EXACT
    assert result.metadata["planner"]["selected"] == "analytic-linear-regression"


def test_planner_routes_gp_analytically():
    gp = GaussianProcessRegressor(RBFKernel(), noise_variance=0.1)
    result = infer(gp, x=[[0.0], [1.0]], y=[0.0, 1.0])
    assert result.kind is InferenceKind.EXACT
    assert result.metadata["planner"]["selected"] == "analytic-gaussian-process"


def test_planner_uses_laplace_when_exact_is_over_budget():
    model = gaussian_model()
    planner = InferencePlanner(config=PlannerConfig(max_exact_operations=0))
    plan = planner.plan(model, initial_guess=[0.0], laplace_backend="sympy")
    assert plan.selected.method == "laplace"
    result = planner.infer(model, initial_guess=[0.0], laplace_backend="sympy")
    assert result.kind is InferenceKind.APPROXIMATE
    assert result.metadata["planner"]["selected"] == "laplace"


def test_planner_nested_is_available_with_prior_sampler():
    model = gaussian_model()
    # Disable Laplace so the planner reaches nested sampling.
    planner = InferencePlanner(
        config=PlannerConfig(max_exact_operations=0, allow_laplace=False)
    )
    plan = planner.plan(model, prior_sampler=lambda rng: np.array([rng.normal()]))
    assert plan.selected.method == "nested-sampling"
    result = planner.infer(
        model,
        prior_sampler=lambda rng: np.array([rng.normal()]),
        n_live=30,
        min_iterations=20,
        max_iterations=250,
        termination_fraction=0.05,
        rng=123,
    )
    assert result.kind is InferenceKind.SAMPLED
    assert result.metadata["planner"]["selected"] == "nested-sampling"


def test_plan_explanation_records_rejections():
    model = gaussian_model()
    plan = plan_inference(model)
    text = plan.explain()
    assert "nested-sampling" in text
    assert "rejected" in text


def test_planner_accepts_prior_transform_for_slice_nested_sampling():
    theta = Parameter("slice_theta")
    y = RandomVariable("slice_y")
    model = Model(
        (theta, y),
        (
            Factor.from_distribution(theta, Normal(0, 1)),
            Factor.from_distribution(y, Normal(theta.symbol, 1)),
        ),
    ).observe(slice_y=1)
    plan = plan_inference(
        model,
        config=PlannerConfig(max_exact_latents=0, allow_laplace=False),
        prior_transform=lambda u: np.array([2.0 * u[0] - 1.0]),
    )
    assert plan.selected.method == "nested-sampling"
    assert "slice" in plan.selected.reason.lower()


def test_planner_conjugacy_candidate_contains_probstats_metadata():
    from probstats.distributions import (
        Bernoulli,
        Beta,
    )

    model = gaussian_model()
    planner = InferencePlanner(config=PlannerConfig(max_exact_operations=0))
    plan = planner.plan(
        model, likelihood=Bernoulli(sp.Rational(1, 2)), data=[1, 0], prior=Beta(1, 1)
    )
    c = next(x for x in plan.candidates if x.method == "conjugacy")
    assert c.metadata["conjugacy_signature"].likelihood_family == "Bernoulli"
    assert c.metadata["distribution_metadata"].measure_type.value == "discrete"
