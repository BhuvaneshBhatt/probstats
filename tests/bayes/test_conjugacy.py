import pytest
import sympy as sp

from probstats.bayes import (
    NormalInverseGamma,
    NormalInverseWishart,
    conjugate_update,
    infer_conjugate,
)
from probstats.bayes.conjugacy import (
    DEFAULT_REGISTRY,
    NaturalConjugatePrior,
    default_normal_prior,
    update_multivariate_normal,
    update_normal,
)
from probstats.distributions import Bernoulli, Beta, Exponential, Normal, StudentT


def assert_expr_equal(a, b):
    assert sp.simplify(a - b) == 0


def test_natural_conjugate_exponential_kernel_and_normalizer():
    rate = sp.Symbol("rate", positive=True)
    eta = sp.Symbol("eta", negative=True)
    prior = NaturalConjugatePrior(Exponential(rate), (sp.Integer(3),), sp.Integer(2))
    assert_expr_equal(prior.kernel((eta,)), sp.exp(3 * eta + 2 * sp.log(-eta)))
    assert_expr_equal(prior.normalizer, sp.Rational(27, 2))


def test_natural_conjugate_update_uses_sufficient_statistics():
    prior = NaturalConjugatePrior(
        Exponential(sp.Symbol("r", positive=True)), (sp.Integer(4),), sp.Integer(3)
    )
    post = prior.update([1, 2, 5])
    assert post.chi == (12,)
    assert post.nu == 6


def test_natural_conjugate_predictive_is_normalizer_ratio():
    x = sp.Symbol("x", positive=True)
    prior = NaturalConjugatePrior(
        Exponential(sp.Symbol("r", positive=True)),
        (sp.Symbol("chi", positive=True),),
        sp.Symbol("nu", positive=True),
    )
    expected = (
        (prior.nu + 1)
        * prior.chi[0] ** (prior.nu + 1)
        / (prior.chi[0] + x) ** (prior.nu + 2)
    )
    assert_expr_equal(prior.predictive_pdf(x), expected)


def test_normal_inverse_gamma_marginals_match_source_parameterization():
    p = NormalInverseGamma(2, 3, 5, 7)
    assert p.mean_marginal == StudentT(2, sp.sqrt(sp.Rational(5, 21)), 14)
    assert p.variance_marginal.shape == 7
    assert p.variance_marginal.scale == 5
    assert p.predictive == StudentT(2, sp.sqrt(sp.Rational(20, 21)), 14)


def test_normal_update_matches_closed_form_source_equations():
    prior = NormalInverseGamma(1, 2, 3, 4)
    result = update_normal([2, 4, 6], prior)
    post = result.posterior
    assert_expr_equal(post.mu, sp.Rational(14, 5))
    assert post.lambda_ == 5
    assert post.nu == sp.Rational(11, 2)
    # mean=4, SSD=8: beta=3+4+(2*3/(2*5))*9 = 62/5
    assert_expr_equal(post.beta, sp.Rational(62, 5))
    assert result.sufficient_statistics == {
        "n": 3,
        "mean": sp.Integer(4),
        "sum_squared_deviations": sp.Integer(8),
    }


def test_normal_single_observation_uses_zero_scatter_not_fake_variance():
    prior = NormalInverseGamma(0, 1, 2, 3)
    result = update_normal([5], prior)
    assert result.sufficient_statistics["sum_squared_deviations"] == 0
    assert_expr_equal(result.posterior.beta, sp.Rational(33, 4))


def test_empty_normal_update_is_identity_with_zero_log_evidence():
    prior = NormalInverseGamma(0, 2, 3, 4)
    result = update_normal([], prior)
    assert result.posterior == prior
    assert result.log_evidence == 0


def test_normal_log_evidence_is_sequentially_consistent():
    prior = NormalInverseGamma(0, 2, 3, 4)
    all_at_once = update_normal([1, 2, 4], prior)
    first = update_normal([1, 2], prior)
    second = update_normal([4], first.posterior)
    assert_expr_equal(
        all_at_once.log_evidence, first.log_evidence + second.log_evidence
    )
    assert all_at_once.posterior == second.posterior


def test_registry_dispatches_normal_nig():
    likelihood = Normal(sp.Symbol("mu"), sp.Symbol("sigma", positive=True))
    prior = NormalInverseGamma(0, 1, 1, 1)
    out = conjugate_update(likelihood, [1, 2], prior)
    assert out.posterior.lambda_ == 3


def test_infer_conjugate_uses_default_normal_prior_and_records_provenance():
    likelihood = Normal(sp.Symbol("mu"), sp.Symbol("sigma", positive=True))
    result = infer_conjugate(likelihood, [1, 2, 3])
    assert result.is_exact
    assert result.metadata["prior"] == default_normal_prior()
    assert result.metadata["posterior_predictive"] == result.posterior.predictive
    assert "normal-normal-inverse-gamma" in result.explain()


def test_infer_conjugate_generic_family_checks_family_match():
    ep = Exponential(sp.Symbol("r", positive=True))
    prior = NaturalConjugatePrior(ep, (2,), 3)
    result = infer_conjugate(ep, [1, 4], prior)
    assert result.posterior.chi == (7,)
    assert result.posterior.nu == 5
    with pytest.raises(LookupError):
        infer_conjugate(Normal(0, 1), [1], prior)


def test_niw_update_matches_matrix_closed_form():
    prior = NormalInverseWishart([0, 0], 2, sp.eye(2), 4)
    post = update_multivariate_normal([[1, 2], [3, 4]], prior)
    assert post.lambda_ == 4
    assert post.nu == 6
    assert post.mu == sp.ImmutableMatrix([1, sp.Rational(3, 2)])
    mean = sp.Matrix([2, 3])
    scatter = sp.Matrix([[2, 2], [2, 2]])
    expected = sp.eye(2) + scatter + (2 * 2 / sp.Integer(4)) * (mean * mean.T)
    assert post.psi == sp.ImmutableMatrix(expected)


def test_niw_predictive_parameters_match_closed_form():
    prior = NormalInverseWishart([0, 0], 2, sp.eye(2) * 3, 5)
    pred = prior.predictive
    assert pred.df == 4
    assert pred.scale == sp.ImmutableMatrix(sp.eye(2) * sp.Rational(9, 8))


def test_niw_rejects_dimension_mismatch():
    prior = NormalInverseWishart([0, 0], 1, sp.eye(2), 3)
    with pytest.raises(ValueError):
        update_multivariate_normal([[1, 2, 3]], prior)


def test_registry_exposes_probstats_signature_protocol():
    likelihood = Bernoulli(sp.Rational(1, 3))
    signature = DEFAULT_REGISTRY.signature_for(likelihood)
    assert signature.likelihood_family == "Bernoulli"
    assert signature.conjugate_prior_family == "Beta"
    rule = DEFAULT_REGISTRY.resolve(likelihood, Beta(2, 3))
    assert rule is not None and rule.name == "beta-bernoulli"


def test_conjugate_result_records_distribution_metadata():
    result = infer_conjugate(Bernoulli(sp.Rational(1, 2)), [1, 0, 1], Beta(1, 1))
    assert result.metadata["conjugacy_signature"].conjugate_prior_family == "Beta"
    assert result.metadata["distribution_metadata"].is_exponential_family is True
