import sympy as sp

from probstats.bayes import (
    BayesianMultivariateLinearRegression,
    InferenceKind,
    plan_inference,
)
from probstats.bayes.models import (
    LinearRegressionPrior,
    MatrixNormalInverseWishartPrior,
)
from probstats.distributions import MultivariateStudentT


def _prior():
    return MatrixNormalInverseWishartPrior([[0, 0]], [[1]], [[2, 0], [0, 2]], 4)


def test_closed_form_mniw_update():
    fit = BayesianMultivariateLinearRegression(prior=_prior()).fit([[1]], [[1, 2]])
    assert fit.posterior.coef_mean == sp.ImmutableMatrix([[sp.Rational(1, 2), 1]])
    assert fit.posterior.precision == sp.ImmutableMatrix([[2]])
    assert fit.posterior.df == 5
    assert fit.posterior.scale == sp.ImmutableMatrix([[sp.Rational(5, 2), 1], [1, 4]])


def test_predictive_and_coefficient_marginals_are_multivariate_student_t():
    fit = BayesianMultivariateLinearRegression(prior=_prior()).fit([[1]], [[1, 2]])
    pred = fit.predictive([1])
    latent = fit.predictive([1], observation=False)
    row = fit.coefficient_row_marginal(0)
    column = fit.response_coefficient_marginal(0)
    assert isinstance(pred, MultivariateStudentT)
    assert pred.df == 4
    assert pred.location == sp.ImmutableMatrix([sp.Rational(1, 2), 1])
    assert pred.scale == sp.ImmutableMatrix(
        [
            [sp.Rational(15, 16), sp.Rational(3, 8)],
            [sp.Rational(3, 8), sp.Rational(3, 2)],
        ]
    )
    assert latent.scale == row.scale
    assert column.df == 4
    assert column.location == sp.ImmutableMatrix([sp.Rational(1, 2)])
    assert latent.scale == sp.ImmutableMatrix(
        [
            [sp.Rational(5, 16), sp.Rational(1, 8)],
            [sp.Rational(1, 8), sp.Rational(1, 2)],
        ]
    )


def test_covariance_marginal_and_matrix_t_density():
    fit = BayesianMultivariateLinearRegression(prior=_prior()).fit([[1]], [[1, 2]])
    covariance = fit.covariance_distribution
    assert covariance.df == 5
    assert covariance.scale == fit.posterior.scale
    density_at_mean = fit.coefficient_distribution.pdf(fit.coef_mean)
    assert density_at_mean.is_positive is True


def test_sequential_update_matches_batch_posterior_and_evidence():
    model = BayesianMultivariateLinearRegression(prior=_prior())
    batch = model.fit([[1], [2]], [[1, 2], [2, 1]])
    sequential = model.fit([[1]], [[1, 2]]).update([[2]], [[2, 1]])
    assert sequential.posterior.coef_mean == batch.posterior.coef_mean
    assert sequential.posterior.precision == batch.posterior.precision
    assert sequential.posterior.scale == batch.posterior.scale
    assert sequential.posterior.df == batch.posterior.df
    assert sp.simplify(sequential.log_evidence - batch.log_evidence) == 0


def test_evidence_matches_direct_integral_in_scalar_response_special_case():
    # m=1 MNIW is the Normal-Inverse-Gamma regression parameterization with
    # sigma^2 ~ IW_1(psi, nu), i.e. IG(nu/2, psi/2).
    prior = MatrixNormalInverseWishartPrior([[0]], [[1]], [[2]], 2)
    fit = BayesianMultivariateLinearRegression(prior=prior).fit([[1]], [[1]])
    expected = 2 * sp.sqrt(5) / 25
    assert sp.simplify(fit.evidence - expected) == 0


def test_intercept_and_response_dimension_validation():
    model = BayesianMultivariateLinearRegression(include_intercept=True)
    fit = model.fit([[0], [1]], [[1, 2], [2, 3]])
    assert fit.design_matrix == sp.ImmutableMatrix([[1, 0], [1, 1]])
    assert fit.posterior.coefficient_dimension == 2
    assert fit.posterior.response_dimension == 2


def test_inference_result_and_planner_use_analytic_route():
    model = BayesianMultivariateLinearRegression(prior=_prior())
    plan = plan_inference(model, x=[[1]], y=[[1, 2]])
    assert plan.selected.method == "analytic-multivariate-linear-regression"
    result = model.infer([[1]], [[1, 2]])
    assert result.kind is InferenceKind.EXACT
    assert result.steps[0].method == "bayesian-multivariate-linear-regression"
    assert result.metadata["evidence"] == result.posterior.evidence


def test_joint_predictive_preserves_cross_row_dependence():
    fit = BayesianMultivariateLinearRegression(prior=_prior()).fit([[1]], [[1, 2]])
    joint = fit.joint_predictive([[1], [2]])
    assert joint.mean == sp.ImmutableMatrix([[sp.Rational(1, 2), 1], [1, 2]])
    row_covariance = joint.precision.inv()
    assert row_covariance[0, 1] != 0
    one = fit.joint_predictive([[1]])
    assert one.row_marginal(0).location == fit.predictive([1]).location
    assert one.row_marginal(0).scale == fit.predictive([1]).scale


def test_scalar_response_exactly_matches_univariate_regression():
    from probstats.bayes import BayesianLinearRegression

    x = [[1, 0], [1, 1], [1, 2]]
    y = [1, 2, 4]
    scalar = BayesianLinearRegression(
        prior=LinearRegressionPrior([0, 0], sp.eye(2), 2, 2)
    ).fit(x, y)
    multi = BayesianMultivariateLinearRegression(
        prior=MatrixNormalInverseWishartPrior([[0], [0]], sp.eye(2), [[2]], 2)
    ).fit(x, [[v] for v in y])
    assert multi.posterior.coef_mean == scalar.posterior.coef_mean
    assert multi.posterior.precision == scalar.posterior.precision
    assert multi.posterior.scale[0, 0] == scalar.posterior.scale
    assert multi.posterior.df == scalar.posterior.df
    assert sp.simplify(multi.log_evidence - scalar.log_evidence) == 0
