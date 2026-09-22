import math

import numpy as np

from probstats.bayes import (
    Factor,
    Model,
    Parameter,
    RandomVariable,
)
from probstats.bayes.nested import (
    EmpiricalPosterior,
    combine_nested_runs,
    infer_nested,
    infer_nested_model,
    nested_sample,
)
from probstats.distributions import Normal


def test_empirical_posterior_normalizes_and_resamples():
    p = EmpiricalPosterior(np.array([[0.0], [1.0]]), np.log([0.25, 0.75]), ("x",))
    assert np.isclose(p.weights.sum(), 1)
    assert np.allclose(p.mean, [0.75])
    assert p.resample(7, rng=1).shape == (7, 1)


def test_nested_sampling_constant_likelihood_has_exact_evidence():
    run = nested_sample(
        lambda x: math.log(3.0),
        lambda rng: np.array([rng.uniform(-1, 1)]),
        ndim=1,
        n_live=20,
        min_iterations=10,
        max_iterations=30,
        termination_fraction=0.2,
        rng=123,
    )
    assert abs(run.log_evidence - math.log(3.0)) < 1e-12
    assert run.posterior.points.shape[1] == 1


def test_nested_sampling_gaussian_likelihood_uniform_prior():
    # prior x~Uniform[-5,5], likelihood exp(-x^2/2)/sqrt(2*pi)
    # evidence = (Phi(5)-Phi(-5))/10.
    log_norm = 0.5 * math.log(2 * math.pi)
    result = infer_nested(
        lambda x: -0.5 * x[0] ** 2 - log_norm,
        lambda rng: np.array([rng.uniform(-5, 5)]),
        ndim=1,
        n_live=150,
        min_iterations=150,
        max_iterations=1500,
        termination_fraction=0.005,
        rng=7,
    )
    exact = math.log(math.erf(5 / math.sqrt(2)) / 10)
    assert abs(result.log_evidence - exact) < 0.18
    assert abs(result.posterior.mean[0]) < 0.15
    assert result.diagnostics["posterior_ess"] > 10


def test_model_adapter_excludes_prior_factor():
    theta = Parameter("theta")
    y = RandomVariable("y")
    model = Model(
        (theta, y),
        (
            Factor.from_distribution(theta, Normal(0, 1)),
            Factor.from_distribution(y, Normal(theta.symbol, 1)),
        ),
    ).observe(y=1)
    result = infer_nested_model(
        model,
        lambda rng: np.array([rng.normal()]),
        n_live=120,
        min_iterations=120,
        max_iterations=1200,
        termination_fraction=0.01,
        rng=22,
    )
    # y marginal is Normal(0, sqrt(2)) at y=1.
    exact = -0.5 * math.log(4 * math.pi) - 0.25
    assert abs(result.log_evidence - exact) < 0.2
    # posterior theta|y=1 is Normal(1/2, 1/sqrt(2)).
    assert abs(result.posterior.mean[0] - 0.5) < 0.16


def test_combine_runs():
    args = {
        "log_likelihood": lambda x: -0.5 * x[0] ** 2,
        "prior_sampler": lambda rng: np.array([rng.uniform(-2, 2)]),
        "ndim": 1,
        "n_live": 30,
        "min_iterations": 20,
        "max_iterations": 100,
        "termination_fraction": 0.05,
    }
    a = nested_sample(**args, rng=1, parameter_names=["x"])
    b = nested_sample(**args, rng=2, parameter_names=["x"])
    combined = combine_nested_runs(a, b)
    assert combined.posterior.names == ("x",)
    assert combined.diagnostics["runs"] == 2
    assert np.isclose(combined.posterior.weights.sum(), 1)


def test_evidence_resampling_flat_likelihood_is_degenerate():
    from probstats.bayes.nested import evidence_resampling

    run = nested_sample(
        lambda x: math.log(2.5),
        lambda rng: np.array([rng.uniform()]),
        ndim=1,
        n_live=20,
        min_iterations=10,
        max_iterations=30,
        termination_fraction=0.2,
        rng=4,
    )
    draws = evidence_resampling(run, samples=30, rng=5)
    assert np.allclose(draws, math.log(2.5), atol=1e-12)


def test_unit_cube_slice_sampler_gaussian_likelihood():
    from probstats.bayes.nested import SliceConstrainedSampler

    log_norm = 0.5 * math.log(2 * math.pi)
    run = nested_sample(
        lambda x: -0.5 * x[0] ** 2 - log_norm,
        prior_transform=lambda u: np.array([-5.0 + 10.0 * u[0]]),
        ndim=1,
        n_live=100,
        min_iterations=100,
        max_iterations=1000,
        termination_fraction=0.01,
        constrained_sampler=SliceConstrainedSampler(steps=6),
        rng=17,
    )
    exact = math.log(math.erf(5 / math.sqrt(2)) / 10)
    assert abs(run.log_evidence - exact) < 0.28
    assert abs(run.posterior.mean[0]) < 0.20
    assert run.constrained_sampler == "SliceConstrainedSampler"
    assert 0 < run.acceptance_rate <= 1


def test_prior_transform_selects_slice_sampler_by_default():
    run = nested_sample(
        lambda x: -0.5 * x[0] ** 2,
        prior_transform=lambda u: np.array([2.0 * u[0] - 1.0]),
        ndim=1,
        n_live=25,
        min_iterations=10,
        max_iterations=30,
        termination_fraction=0.2,
        rng=19,
    )
    assert run.constrained_sampler == "SliceConstrainedSampler"


def test_slice_sampler_handles_concentrated_2d_contour():
    # A small circular high-likelihood region inside a broad square prior.  Once
    # nested sampling has live points on the contour, chord slices move locally
    # instead of repeatedly trying to rediscover the region from the full prior.
    run = nested_sample(
        lambda x: -80.0 * ((x[0] - 0.35) ** 2 + (x[1] + 0.2) ** 2),
        prior_transform=lambda u: np.array([4.0 * u[0] - 2.0, 4.0 * u[1] - 2.0]),
        ndim=2,
        n_live=80,
        min_iterations=100,
        max_iterations=700,
        termination_fraction=0.02,
        rng=23,
    )
    assert run.constrained_sampler == "SliceConstrainedSampler"
    assert np.linalg.norm(run.posterior.mean - np.array([0.35, -0.2])) < 0.25
    assert run.acceptance_rate > 0.02
