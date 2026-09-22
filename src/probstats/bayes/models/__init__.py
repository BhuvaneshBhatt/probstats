"""Ready-to-use Bayesian model families."""

from .linear import (
    BayesianLinearRegression,
    BayesianLinearRegressionFit,
    LinearRegressionPrior,
)
from .multivariate_linear import (
    BayesianMultivariateLinearRegression,
    BayesianMultivariateLinearRegressionFit,
    MatrixNormalInverseWishartPrior,
    MatrixStudentT,
)

__all__ = [
    "BayesianLinearRegression",
    "BayesianLinearRegressionFit",
    "BayesianMultivariateLinearRegression",
    "BayesianMultivariateLinearRegressionFit",
    "LinearRegressionPrior",
    "MatrixNormalInverseWishartPrior",
    "MatrixStudentT",
]
