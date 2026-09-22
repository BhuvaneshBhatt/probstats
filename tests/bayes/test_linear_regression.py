import pytest
import sympy as sp

from probstats.bayes import (
    BayesianLinearRegression,
    Factor,
    InferenceKind,
    Model,
    Parameter,
    RandomVariable,
)
from probstats.bayes.exact import evidence
from probstats.bayes.models import LinearRegressionPrior
from probstats.distributions import (
    InverseGamma,
    Normal,
)


def test_linear_regression_closed_form_update():
    prior = LinearRegressionPrior([0], [[1]], 2, 2)
    fit = BayesianLinearRegression(prior=prior).fit([[1]], [1])

    assert fit.posterior.coef_mean == sp.ImmutableMatrix([sp.Rational(1, 2)])
    assert fit.posterior.precision == sp.ImmutableMatrix([[2]])
    assert fit.posterior.scale == sp.Rational(5, 2)
    assert fit.posterior.df == 3


def test_coefficient_marginal_is_multivariate_student_t():
    prior = LinearRegressionPrior([0], [[1]], 2, 2)
    fit = BayesianLinearRegression(prior=prior).fit([[1]], [1])
    dist = fit.coefficient_distribution

    assert dist.location == sp.ImmutableMatrix([sp.Rational(1, 2)])
    assert dist.scale == sp.ImmutableMatrix([[sp.Rational(5, 12)]])
    assert dist.df == 3


def test_posterior_predictive_and_underlying_value_distribution():
    fit = BayesianLinearRegression(prior=LinearRegressionPrior([0], [[1]], 2, 2)).fit(
        [[1]], [1]
    )

    predictive = fit.predictive([1])
    latent_mean = fit.predictive([1], observation=False)

    assert predictive.location == sp.Rational(1, 2)
    assert predictive.scale == sp.sqrt(5) / 2
    assert predictive.df == 3
    assert latent_mean.location == sp.Rational(1, 2)
    assert latent_mean.scale == sp.sqrt(15) / 6
    assert latent_mean.df == 3


def test_exact_model_evidence_formula():
    prior = LinearRegressionPrior([0], [[1]], 2, 2)
    fit = BayesianLinearRegression(prior=prior).fit([[1]], [1])

    assert sp.simplify(fit.evidence - 2 * sp.sqrt(5) / 25) == 0


def test_sequential_update_matches_batch_update():
    prior = LinearRegressionPrior([0, 0], sp.eye(2), 2, 2)
    model = BayesianLinearRegression(prior=prior)
    batch = model.fit([[1, 0], [1, 1], [1, 2]], [1, 2, 4])

    first = model.fit([[1, 0], [1, 1]], [1, 2])
    second = BayesianLinearRegression(prior=first.posterior).fit([[1, 2]], [4])

    assert second.posterior.coef_mean == batch.posterior.coef_mean
    assert second.posterior.precision == batch.posterior.precision
    assert sp.simplify(second.posterior.scale - batch.posterior.scale) == 0
    assert second.posterior.df == batch.posterior.df
    assert (
        sp.simplify(first.log_evidence + second.log_evidence - batch.log_evidence) == 0
    )


def test_include_intercept_augments_design_matrix():
    fit = BayesianLinearRegression(include_intercept=True).fit(
        [[0], [1], [2]], [1, 2, 3]
    )
    assert fit.design_matrix == sp.ImmutableMatrix([[1, 0], [1, 1], [1, 2]])
    assert fit.posterior.dimension == 2


def test_inference_result_exposes_exact_provenance():
    result = BayesianLinearRegression(
        prior=LinearRegressionPrior([0], [[1]], 2, 2)
    ).infer([[1]], [1])
    assert result.kind is InferenceKind.EXACT
    assert result.steps[0].method == "bayesian-linear-regression"
    assert result.metadata["evidence"] == result.posterior.evidence


def test_invalid_prior_dimension_is_rejected():
    with pytest.raises(ValueError, match="match coef_mean"):
        LinearRegressionPrior([0, 0], [[1]], 2, 2)

    model = BayesianLinearRegression(prior=LinearRegressionPrior([0], [[1]], 2, 2))
    with pytest.raises(ValueError, match="dimension"):
        model.fit([[1, 2]], [1])


def _p3_intercept_model(values):
    beta = Parameter(sp.Symbol("beta", real=True))
    variance = Parameter(
        sp.Symbol("variance", positive=True), support=sp.Interval.open(0, sp.oo)
    )
    responses = tuple(RandomVariable(f"y{index}") for index in range(len(values)))
    factors = [
        Factor.from_distribution(variance, InverseGamma(1, 1)),
        Factor.from_distribution(beta, Normal(0, sp.sqrt(variance.symbol))),
    ]
    factors.extend(
        Factor.from_distribution(
            response, Normal(beta.symbol, sp.sqrt(variance.symbol))
        )
        for response in responses
    )
    model = Model((beta, variance, *responses), tuple(factors))
    return model.observe(**{rv.name: value for rv, value in zip(responses, values)})


def test_closed_form_evidence_cross_checks_p3_generic_exact_integration():
    prior = LinearRegressionPrior([0], [[1]], 2, 2)
    fit = BayesianLinearRegression(prior=prior).fit([[1]], [1])
    generic = evidence(_p3_intercept_model([1]))

    assert abs(float(sp.N(generic - fit.evidence, 15))) < 1e-12


def test_predictive_density_cross_checks_p3_evidence_ratio():
    prior = LinearRegressionPrior([0], [[1]], 2, 2)
    fit = BayesianLinearRegression(prior=prior).fit([[1]], [1])
    predictive_density = fit.predictive([1]).pdf(0)

    z1 = evidence(_p3_intercept_model([1]))
    z2 = evidence(_p3_intercept_model([1, 0]))
    generic_predictive = sp.N(z2 / z1, 15)

    assert abs(float(generic_predictive - sp.N(predictive_density, 15))) < 1e-12
