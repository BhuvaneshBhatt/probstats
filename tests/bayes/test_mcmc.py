import math

import numpy as np
import pytest

from probstats.bayes import (
    Factor,
    Model,
    Parameter,
    RandomVariable,
    infer,
)
from probstats.bayes.mcmc import (
    AdaptiveMetropolis,
    ComponentMetropolis,
    GibbsSampler,
    HamiltonianMonteCarlo,
    IndependentMetropolis,
    MetropolisHastings,
    NoUTurnSampler,
    TransformedMetropolis,
    diagnose_chains,
    geweke,
    heidelberger_welch,
    infer_mcmc_model,
    raftery_lewis,
    run_chains,
    sample_mcmc,
)
from probstats.bayes.nested import (
    MCMCConstrainedSampler,
    nested_sample,
)
from probstats.bayes.planner import PlannerConfig
from probstats.distributions import Normal


def std_normal_logp(x):
    return -0.5 * float(np.dot(x, x))


def std_normal_grad(x):
    return -np.asarray(x, dtype=float)


def test_metropolis_chain_is_resumable_and_state_counters_continue():
    sampler = MetropolisHastings(proposal_cov=0.4)
    first = sample_mcmc(
        std_normal_logp, np.array([3.0]), sampler=sampler, draws=300, warmup=100, rng=10
    )
    second = first.resume(std_normal_logp, sampler, 200)
    assert first.state.proposed == 400
    assert second.state.proposed == 600
    assert second.state.iteration == 600
    assert np.isfinite(second.acceptance_rate)
    assert abs(second.samples.mean()) < 0.35


def test_independent_metropolis_corrects_for_proposal_density():
    logq = lambda x: -0.5 * float(np.dot(x, x)) / 4 - 0.5 * math.log(8 * math.pi)
    sampler = IndependentMetropolis(lambda rng: np.array([rng.normal(scale=2.0)]), logq)
    chain = sample_mcmc(
        std_normal_logp, [0.0], sampler=sampler, draws=1200, warmup=200, rng=11
    )
    assert abs(chain.samples.mean()) < 0.15
    assert 0.4 < chain.samples.std() < 1.4


class ExpTransform:
    def forward(self, u):
        return np.exp(u)

    def inverse(self, x):
        return np.log(x)

    def log_abs_det_jacobian(self, u):
        return float(np.sum(u))


def test_transformed_metropolis_includes_jacobian():
    # Exponential(1) target on x>0; transformed target in u has log density -exp(u)+u.
    def logp(x):
        return -x[0] if x[0] > 0 else -np.inf

    chain = sample_mcmc(
        logp,
        [1.0],
        sampler=TransformedMetropolis(ExpTransform(), 0.5),
        draws=1500,
        warmup=300,
        rng=12,
    )
    assert 0.75 < chain.samples[:, 0].mean() < 1.25


def test_adaptive_metropolis_learns_covariance_on_correlated_gaussian():
    cov = np.array([[1.0, 0.8], [0.8, 1.5]])
    precision = np.linalg.inv(cov)
    logp = lambda x: -0.5 * float(x @ precision @ x)
    chain = sample_mcmc(
        logp,
        [2.0, -2.0],
        sampler=AdaptiveMetropolis(initial_cov=0.2, adapt_start=50),
        draws=2000,
        warmup=500,
        rng=13,
    )
    np.testing.assert_allclose(chain.samples.mean(axis=0), [0, 0], atol=0.18)
    assert np.sign(np.cov(chain.samples.T)[0, 1]) > 0


def test_hmc_gaussian_target_is_accurate_and_nondivergent():
    sampler = HamiltonianMonteCarlo(
        gradient=std_normal_grad, step_size=0.25, leapfrog_steps=8
    )
    chain = sample_mcmc(
        std_normal_logp, [2.0], sampler=sampler, draws=1000, warmup=100, rng=14
    )
    assert abs(chain.samples.mean()) < 0.12
    assert 0.85 < chain.samples.std() < 1.15
    assert not np.asarray(chain.sample_stats["divergent"]).any()


def test_nuts_gaussian_target_is_accurate_and_reports_tree_statistics():
    sampler = NoUTurnSampler(gradient=std_normal_grad, step_size=0.3, max_tree_depth=7)
    chain = sample_mcmc(
        std_normal_logp, [2.5], sampler=sampler, draws=1200, warmup=100, rng=15
    )
    assert abs(chain.samples.mean()) < 0.12
    assert 0.82 < chain.samples.std() < 1.18
    assert "tree_depth" in chain.sample_stats
    assert not np.asarray(chain.sample_stats["divergent"]).any()


def test_gibbs_and_componentwise_sampling():
    # Independent standard normal conditionals are exact Gibbs updates.
    updates = {0: lambda q, rng: rng.normal(), 1: lambda q, rng: rng.normal()}
    gibbs = sample_mcmc(
        std_normal_logp,
        [3.0, -3.0],
        sampler=GibbsSampler(updates),
        draws=800,
        warmup=20,
        rng=16,
    )
    np.testing.assert_allclose(gibbs.samples.mean(axis=0), [0, 0], atol=0.12)
    comp = sample_mcmc(
        std_normal_logp,
        [2.0, -2.0],
        sampler=ComponentMetropolis([0.8, 0.8]),
        draws=1000,
        warmup=200,
        rng=17,
    )
    np.testing.assert_allclose(comp.samples.mean(axis=0), [0, 0], atol=0.2)


def test_run_chains_respects_supplied_generator_and_count_validation():
    kwargs = {
        "log_density": std_normal_logp,
        "initial_positions": [[-1.0], [1.0]],
        "sampler_factory": lambda: MetropolisHastings(0.3),
        "draws": 20,
        "warmup": 5,
    }
    first = run_chains(**kwargs, rng=np.random.default_rng(123))
    second = run_chains(**kwargs, rng=np.random.default_rng(123))
    for left, right in zip(first, second, strict=True):
        assert np.array_equal(left.samples, right.samples)

    with pytest.raises(TypeError):
        sample_mcmc(std_normal_logp, [0.0], sampler=MetropolisHastings(), draws=True)


def test_multiple_chains_and_convergence_diagnostics():
    chains = run_chains(
        std_normal_logp,
        [[-3.0], [-1.0], [1.0], [3.0]],
        sampler_factory=lambda: NoUTurnSampler(gradient=std_normal_grad, step_size=0.3),
        draws=700,
        warmup=100,
        rng=18,
        parameter_names=["x"],
    )
    diag = diagnose_chains(chains)
    assert diag.rank_rhat[0] < 1.05
    assert diag.bulk_ess[0] > 100
    assert diag.tail_ess[0] > 100
    assert diag.divergence_count == 0
    values = chains[0].samples[:, 0]
    assert abs(geweke(values)) < 4
    assert raftery_lewis(values)["recommended"] > 0
    assert isinstance(heidelberger_welch(values)["stationary"], bool)


def gaussian_model():
    theta = Parameter("theta")
    y = RandomVariable("y")
    return Model(
        (theta, y),
        (
            Factor.from_distribution(theta, Normal(0, 1)),
            Factor.from_distribution(y, Normal(theta.symbol, 1)),
        ),
    ).observe(y=1), theta


def test_planner_chooses_nuts_for_symbolic_gradient():
    model, _theta = gaussian_model()
    result = infer_mcmc_model(
        model,
        initial_positions=[[-2.0], [0.0], [2.0], [4.0]],
        draws=600,
        warmup=100,
        rng=19,
    )
    samples = result.posterior.posterior["theta"]
    assert abs(samples.mean() - 0.5) < 0.12
    assert result.metadata["sampler"] == "nuts"

    planned = infer(
        model,
        initial_positions=[[-2.0], [0.0], [2.0], [4.0]],
        mcmc_draws=300,
        mcmc_warmup=50,
        # Make exact and Laplace inapplicable so planner exercises MCMC.
        config=PlannerConfig(
            max_exact_latents=0, allow_laplace=False, allow_nested=False
        ),
        rng=20,
    )
    assert planned.metadata["planner"]["selected"] == "mcmc"
    assert planned.posterior.posterior["theta"].shape == (4, 300)


def test_nested_mcmc_constrained_sampler_preserves_unit_cube_prior():
    run = nested_sample(
        lambda x: -0.5 * x[0] ** 2,
        prior_transform=lambda u: np.array([-4.0 + 8.0 * u[0]]),
        ndim=1,
        n_live=50,
        min_iterations=40,
        max_iterations=200,
        termination_fraction=0.05,
        constrained_sampler=MCMCConstrainedSampler(steps=20, proposal_scale=0.08),
        rng=21,
    )
    assert run.constrained_sampler == "MCMCConstrainedSampler"
    assert abs(run.posterior.mean[0]) < 0.4
    assert 0 <= run.acceptance_rate <= 1


def test_chain_and_state_arrays_are_immutable():
    logp = lambda x: -0.5 * float(np.dot(x, x))
    chain = sample_mcmc(
        logp, [0.0], sampler=MetropolisHastings(0.2), draws=4, warmup=0, rng=3
    )
    with pytest.raises(ValueError):
        chain.samples[0, 0] = 1.0
    with pytest.raises(ValueError):
        chain.log_prob[0] = 0.0
    with pytest.raises(ValueError):
        chain.state.position[0] = 1.0
