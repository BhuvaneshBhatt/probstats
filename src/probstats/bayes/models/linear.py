"""Analytic Bayesian linear regression with a Normal-Inverse-Gamma prior."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import sympy as sp

from ..._exact_linear_algebra import exact_solve
from ..core import InferenceKind, InferenceResult, InferenceStep
from ..distributions import InverseGamma
from ..distributions.student_t import MultivariateStudentT, StudentT


def _matrix(value: Any, *, column: bool = False) -> sp.ImmutableDenseMatrix:
    if isinstance(value, sp.MatrixBase):
        mat = sp.ImmutableMatrix(value)
    else:
        mat = sp.ImmutableMatrix(value)
    if column and mat.cols != 1:
        if mat.rows == 1:
            mat = sp.ImmutableMatrix(list(mat))
        else:
            raise ValueError("Expected a vector or one-column matrix.")
    return mat


def _column(value: Sequence[Any] | sp.MatrixBase) -> sp.ImmutableDenseMatrix:
    if isinstance(value, sp.MatrixBase):
        mat = sp.ImmutableMatrix(value)
        if mat.cols == 1:
            return mat
        if mat.rows == 1:
            return sp.ImmutableMatrix(list(mat))
        raise ValueError("Expected a vector.")
    return sp.ImmutableMatrix([sp.sympify(item) for item in value])


def _symmetrize(matrix: sp.MatrixBase) -> sp.ImmutableDenseMatrix:
    matrix = sp.ImmutableMatrix(matrix)
    return sp.ImmutableMatrix((matrix + matrix.T) / 2)


@dataclass(frozen=True, slots=True)
class LinearRegressionPrior:
    """Normal-Inverse-Gamma prior for univariate Bayesian regression.

    ``beta | sigma2 ~ N(coef_mean, sigma2 * precision_inverse)`` and
    ``sigma2 ~ InverseGamma(df/2, scale/2)``.
    """

    coef_mean: sp.ImmutableDenseMatrix
    precision: sp.ImmutableDenseMatrix
    scale: sp.Expr
    df: sp.Expr

    def __init__(
        self,
        coef_mean: Sequence[Any] | sp.MatrixBase,
        precision: Any,
        scale: Any,
        df: Any,
    ):
        mean = _column(coef_mean)
        prec = _symmetrize(_matrix(precision))
        if prec.rows != prec.cols or prec.rows != mean.rows:
            raise ValueError("precision must be square and match coef_mean dimension.")
        object.__setattr__(self, "coef_mean", mean)
        object.__setattr__(self, "precision", prec)
        object.__setattr__(self, "scale", sp.sympify(scale))
        object.__setattr__(self, "df", sp.sympify(df))
        if self.scale.is_nonpositive is True:
            raise ValueError("LinearRegressionPrior scale must be positive.")
        if self.df.is_nonpositive is True:
            raise ValueError("LinearRegressionPrior df must be positive.")
        if (
            not self.precision.free_symbols
            and self.precision.is_positive_definite is not True
        ):
            raise ValueError(
                "LinearRegressionPrior precision must be positive definite."
            )

    @property
    def dimension(self) -> int:
        return self.coef_mean.rows

    @property
    def precision_inverse(self) -> sp.ImmutableDenseMatrix:
        return exact_solve(self.precision, sp.eye(self.precision.rows))

    @property
    def variance_distribution(self) -> InverseGamma:
        return InverseGamma(self.df / 2, self.scale / 2)

    @property
    def coefficient_distribution(self) -> MultivariateStudentT:
        return MultivariateStudentT(
            self.coef_mean,
            sp.simplify(self.scale / self.df) * self.precision_inverse,
            self.df,
        )


@dataclass(frozen=True, slots=True)
class BayesianLinearRegressionFit:
    """Closed-form posterior and prediction object for Bayesian linear regression."""

    prior: LinearRegressionPrior
    posterior: LinearRegressionPrior
    design_matrix: sp.ImmutableDenseMatrix
    response: sp.ImmutableDenseMatrix
    log_evidence: sp.Expr

    @property
    def coef_mean(self) -> sp.ImmutableDenseMatrix:
        return self.posterior.coef_mean

    @property
    def coef_precision(self) -> sp.ImmutableDenseMatrix:
        return self.posterior.precision

    @property
    def coef_covariance_scale(self) -> sp.ImmutableDenseMatrix:
        """Scale matrix of the marginal multivariate Student-t for coefficients."""
        return self.posterior.coefficient_distribution.scale

    @property
    def coefficient_distribution(self) -> MultivariateStudentT:
        return self.posterior.coefficient_distribution

    @property
    def variance_distribution(self) -> InverseGamma:
        return self.posterior.variance_distribution

    @property
    def evidence(self) -> sp.Expr:
        return sp.simplify(sp.exp(self.log_evidence))

    def predictive(
        self, row: Sequence[Any] | sp.MatrixBase, *, observation: bool = True
    ) -> StudentT:
        x = _column(row)
        if x.rows != self.posterior.dimension:
            raise ValueError(
                "Prediction row width must match regression coefficient dimension."
            )
        inv_precision = self.posterior.precision_inverse
        leverage = sp.simplify((x.T * inv_precision * x)[0])
        noise = sp.S.One if observation else sp.S.Zero
        variance_scale = sp.simplify(
            self.posterior.scale / self.posterior.df * (leverage + noise)
        )
        location = sp.simplify((x.T * self.posterior.coef_mean)[0])
        return StudentT(location, sp.sqrt(variance_scale), self.posterior.df)

    def predict_distribution(self, rows, *, observation: bool = True):
        """Return predictive law(s) for one row or a collection of rows."""
        try:
            return self.predictive(rows, observation=observation)
        except (TypeError, ValueError):
            return tuple(self.predictive(row, observation=observation) for row in rows)

    def predict(self, rows, *, observation: bool = True):
        """Return posterior-predictive mean(s)."""
        laws = self.predict_distribution(rows, observation=observation)
        if isinstance(laws, tuple):
            return tuple(law.mean_value for law in laws)
        return laws.mean_value

    def posterior_predictive(
        self, rows, *, size=1000, rng=None, observation: bool = True
    ):
        from ..predictive import posterior_predictive

        return posterior_predictive(
            self, rows, size=size, rng=rng, observation=observation
        )

    def predictive_interval(
        self, rows, *, level: float = 0.95, observation: bool = True
    ):
        from ..predictive import predictive_interval

        return predictive_interval(self, rows, level=level, observation=observation)

    def to_inference_result(self) -> InferenceResult:
        return InferenceResult(
            posterior=self,
            kind=InferenceKind.EXACT,
            log_evidence=self.log_evidence,
            steps=(
                InferenceStep(
                    "bayesian-linear-regression",
                    f"Applied analytic Normal-Inverse-Gamma regression update to {self.design_matrix.rows} observations.",
                    True,
                    {
                        "n_observations": self.design_matrix.rows,
                        "n_coefficients": self.design_matrix.cols,
                    },
                ),
            ),
            metadata={
                "prior": self.prior,
                "coefficient_distribution": self.coefficient_distribution,
                "variance_distribution": self.variance_distribution,
                "evidence": self.evidence,
            },
        )


class BayesianLinearRegression:
    """Bayesian linear regression with exact Normal-Inverse-Gamma updating.

    The model is ``y = X beta + epsilon`` with ``epsilon ~ N(0, sigma2 I)``.
    """

    def __init__(
        self,
        *,
        prior: LinearRegressionPrior | None = None,
        include_intercept: bool = False,
    ):
        self.prior = prior
        self.include_intercept = bool(include_intercept)

    @staticmethod
    def default_prior(dimension: int) -> LinearRegressionPrior:
        if dimension < 1:
            raise ValueError("dimension must be positive.")
        return LinearRegressionPrior(
            [0] * dimension,
            sp.eye(dimension) / 100,
            sp.Rational(1, 100),
            sp.Rational(1, 100),
        )

    def design_dimension(self, x: Any) -> int:
        """Return the coefficient dimension implied by a design input."""
        return self._prepare_design(x).cols

    def _prepare_design(self, x: Any) -> sp.ImmutableDenseMatrix:
        design = _matrix(x)
        if design.rows == 0 or design.cols == 0:
            raise ValueError("Design matrix must be non-empty.")
        if self.include_intercept:
            design = sp.ImmutableMatrix.hstack(sp.ones(design.rows, 1), design)
        return sp.ImmutableMatrix(design)

    def fit(
        self, x: Any, y: Sequence[Any] | sp.MatrixBase
    ) -> BayesianLinearRegressionFit:
        design = self._prepare_design(x)
        response = _column(y)
        if design.rows != response.rows:
            raise ValueError(
                "Design matrix and response must have the same number of rows."
            )
        prior = self.prior or self.default_prior(design.cols)
        if prior.dimension != design.cols:
            raise ValueError(
                "Prior coefficient dimension does not match design matrix width."
            )

        xt = design.T
        precision_n = _symmetrize(prior.precision + xt * design)
        precision_inv_n = exact_solve(precision_n, sp.eye(precision_n.rows))
        mean_n = sp.ImmutableMatrix(
            precision_inv_n * (xt * response + prior.precision * prior.coef_mean)
        )
        residual = response - design * mean_n
        mean_diff = mean_n - prior.coef_mean
        scale_n = sp.simplify(
            prior.scale
            + (residual.T * residual)[0]
            + (mean_diff.T * prior.precision * mean_diff)[0]
        )
        df_n = sp.simplify(prior.df + design.rows)
        posterior = LinearRegressionPrior(mean_n, precision_n, scale_n, df_n)

        n = design.rows
        log_evidence = sp.simplify(
            -sp.Rational(n, 2) * sp.log(2 * sp.pi)
            + sp.Rational(1, 2) * sp.log(prior.precision.det())
            - sp.Rational(1, 2) * sp.log(precision_n.det())
            + sp.loggamma(df_n / 2)
            - sp.loggamma(prior.df / 2)
            + prior.df / 2 * sp.log(prior.scale / 2)
            - df_n / 2 * sp.log(scale_n / 2)
        )

        return BayesianLinearRegressionFit(
            prior, posterior, design, response, log_evidence
        )

    def infer(self, x: Any, y: Sequence[Any] | sp.MatrixBase) -> InferenceResult:
        return self.fit(x, y).to_inference_result()
