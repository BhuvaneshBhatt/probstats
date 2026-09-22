"""Exact multivariate Bayesian linear regression.

The conjugate model is

    Y | B, Sigma ~ MN(X B, I_n, Sigma)
    B | Sigma    ~ MN(M, Lambda**-1, Sigma)
    Sigma        ~ IW(Psi, nu)

where rows of ``Y`` are observations, rows of ``B`` correspond to regression
coefficients, and columns correspond to response dimensions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import sympy as sp

from ..._exact_linear_algebra import exact_determinant, exact_solve, quadratic_form
from ..._validation import sample_shape
from ...distributions import InverseWishart, MatrixNormal
from ...sampling import as_rng, sample
from ..core import InferenceKind, InferenceResult, InferenceStep
from ..distributions.student_t import MultivariateStudentT


def _matrix(value: Any, *, name: str) -> sp.ImmutableDenseMatrix:
    try:
        matrix = sp.ImmutableMatrix(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be matrix-like") from exc
    if matrix.rows < 1 or matrix.cols < 1:
        raise ValueError(f"{name} must be non-empty")
    return matrix


def _symmetrize(value: sp.MatrixBase) -> sp.ImmutableDenseMatrix:
    matrix = sp.ImmutableMatrix(value)
    return sp.ImmutableMatrix((matrix + matrix.T) / 2)


def _validate_pd(value: sp.MatrixBase, *, name: str) -> None:
    if value.rows != value.cols:
        raise ValueError(f"{name} must be square")
    if value != value.T and not isinstance(value, sp.MatrixExpr):
        raise ValueError(f"{name} must be symmetric")
    if not value.free_symbols and value.is_positive_definite is not True:
        raise ValueError(f"{name} must be positive definite")
    if value.is_positive_definite is False:
        raise ValueError(f"{name} must be positive definite")


def _multigamma_log(a: sp.Expr, dimension: int) -> sp.Expr:
    return sp.simplify(
        sp.Rational(dimension * (dimension - 1), 4) * sp.log(sp.pi)
        + sp.Add(
            *(sp.loggamma(a + sp.Rational(1 - j, 2)) for j in range(1, dimension + 1))
        )
    )


@dataclass(frozen=True, slots=True)
class MatrixNormalInverseWishartPrior:
    """Conjugate prior for multivariate Gaussian linear regression.

    ``B | Sigma ~ MatrixNormal(coef_mean, precision**-1, Sigma)`` and
    ``Sigma ~ InverseWishart(df, scale)``.
    """

    coef_mean: sp.ImmutableDenseMatrix
    precision: sp.ImmutableDenseMatrix
    scale: sp.ImmutableDenseMatrix
    df: sp.Expr

    def __init__(self, coef_mean: Any, precision: Any, scale: Any, df: Any):
        mean = _matrix(coef_mean, name="coef_mean")
        precision_matrix = _symmetrize(_matrix(precision, name="precision"))
        scale_matrix = _symmetrize(_matrix(scale, name="scale"))
        if precision_matrix.shape != (mean.rows, mean.rows):
            raise ValueError("precision dimension must match coefficient rows")
        if scale_matrix.shape != (mean.cols, mean.cols):
            raise ValueError("scale dimension must match response dimension")
        _validate_pd(precision_matrix, name="precision")
        _validate_pd(scale_matrix, name="scale")
        degrees = sp.sympify(df)
        if (degrees - (mean.cols - 1)).is_nonpositive is True:
            raise ValueError("df must exceed response dimension - 1")
        object.__setattr__(self, "coef_mean", mean)
        object.__setattr__(self, "precision", precision_matrix)
        object.__setattr__(self, "scale", scale_matrix)
        object.__setattr__(self, "df", degrees)

    @property
    def coefficient_dimension(self) -> int:
        return self.coef_mean.rows

    @property
    def response_dimension(self) -> int:
        return self.coef_mean.cols

    @property
    def precision_inverse(self) -> sp.ImmutableDenseMatrix:
        return exact_solve(self.precision, sp.eye(self.precision.rows))

    @property
    def covariance_distribution(self) -> InverseWishart:
        return InverseWishart(self.df, self.scale)

    @property
    def coefficient_distribution(self) -> MatrixStudentT:
        return MatrixStudentT(self.coef_mean, self.precision, self.scale, self.df)


@dataclass(frozen=True, slots=True)
class MatrixStudentT:
    """Matrix-t marginal induced by a Matrix-Normal/Inverse-Wishart law.

    This parameterization retains the conjugate parameters
    ``precision``, ``scale`` and ``df`` rather than introducing an ambiguous
    matrix-t degrees-of-freedom convention.  It is exactly the marginal law of
    ``B`` when ``B | Sigma ~ MN(mean, precision**-1, Sigma)`` and
    ``Sigma ~ IW(scale, df)``.
    """

    mean: sp.ImmutableDenseMatrix
    precision: sp.ImmutableDenseMatrix
    scale: sp.ImmutableDenseMatrix
    df: sp.Expr

    def __init__(self, mean: Any, precision: Any, scale: Any, df: Any):
        prior = MatrixNormalInverseWishartPrior(mean, precision, scale, df)
        object.__setattr__(self, "mean", prior.coef_mean)
        object.__setattr__(self, "precision", prior.precision)
        object.__setattr__(self, "scale", prior.scale)
        object.__setattr__(self, "df", prior.df)

    @property
    def rows(self) -> int:
        return self.mean.rows

    @property
    def cols(self) -> int:
        return self.mean.cols

    def logpdf(self, value: Any) -> sp.Expr:
        x = _matrix(value, name="value")
        if x.shape != self.mean.shape:
            raise ValueError("value shape must match matrix-t mean")
        delta = x - self.mean
        updated_scale = sp.ImmutableMatrix(
            self.scale + delta.T * self.precision * delta
        )
        p, m = self.rows, self.cols
        return sp.simplify(
            _multigamma_log((self.df + p) / 2, m)
            - _multigamma_log(self.df / 2, m)
            - sp.Rational(p * m, 2) * sp.log(sp.pi)
            + sp.Rational(m, 2) * sp.log(exact_determinant(self.precision))
            + self.df / 2 * sp.log(exact_determinant(self.scale))
            - (self.df + p) / 2 * sp.log(exact_determinant(updated_scale))
        )

    def pdf(self, value: Any) -> sp.Expr:
        return sp.simplify(sp.exp(self.logpdf(value)))

    def sample(self, size=None, rng=None):
        """Draw matrix-t samples through the defining MNIW hierarchy."""
        shape = sample_shape(size)
        generator = as_rng(rng)
        row_covariance = exact_solve(self.precision, sp.eye(self.precision.rows))
        covariance_law = InverseWishart(self.df, self.scale)

        def one():
            sigma = sample(covariance_law, rng=generator)
            law = MatrixNormal(self.mean, row_covariance, sp.ImmutableMatrix(sigma))
            return sample(law, rng=generator)

        if not shape:
            return one()
        import numpy as np

        out = np.empty(shape + (self.rows, self.cols), dtype=float)
        for index in np.ndindex(shape):
            out[index] = one()
        return out

    def row_marginal(self, index: int) -> MultivariateStudentT:
        """Marginal distribution of one coefficient row across responses."""
        if not 0 <= index < self.rows:
            raise IndexError("coefficient row index out of range")
        inv_precision = exact_solve(self.precision, sp.eye(self.precision.rows))
        degrees = sp.simplify(self.df - self.cols + 1)
        factor = sp.simplify(inv_precision[index, index])
        location = sp.ImmutableMatrix(list(self.mean.row(index)))
        return MultivariateStudentT(
            location,
            sp.ImmutableMatrix(factor * self.scale / degrees),
            degrees,
        )

    def column_marginal(self, index: int) -> MultivariateStudentT:
        """Marginal coefficient vector for one response dimension."""
        if not 0 <= index < self.cols:
            raise IndexError("response column index out of range")
        degrees = sp.simplify(self.df - self.cols + 1)
        location = sp.ImmutableMatrix(list(self.mean.col(index)))
        return MultivariateStudentT(
            location,
            sp.ImmutableMatrix(
                self.scale[index, index]
                * exact_solve(self.precision, sp.eye(self.precision.rows))
                / degrees
            ),
            degrees,
        )

    def linear_marginal(
        self, row: Sequence[Any] | sp.MatrixBase
    ) -> MultivariateStudentT:
        """Marginal distribution of ``row.T * B`` across responses."""
        x = sp.ImmutableMatrix(row)
        if x.cols != 1:
            if x.rows == 1:
                x = sp.ImmutableMatrix(list(x))
            else:
                raise ValueError("row must be a vector")
        if x.rows != self.rows:
            raise ValueError("row width must match coefficient dimension")
        degrees = sp.simplify(self.df - self.cols + 1)
        leverage = quadratic_form(self.precision, x)
        location = sp.ImmutableMatrix(self.mean.T * x)
        return MultivariateStudentT(
            location,
            sp.ImmutableMatrix(leverage * self.scale / degrees),
            degrees,
        )


@dataclass(frozen=True, slots=True)
class BayesianMultivariateLinearRegressionFit:
    """Exact posterior, evidence and prediction for multivariate regression."""

    prior: MatrixNormalInverseWishartPrior
    posterior: MatrixNormalInverseWishartPrior
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
    def coefficient_distribution(self) -> MatrixStudentT:
        return self.posterior.coefficient_distribution

    @property
    def covariance_distribution(self) -> InverseWishart:
        return self.posterior.covariance_distribution

    @property
    def evidence(self) -> sp.Expr:
        return sp.simplify(sp.exp(self.log_evidence))

    def coefficient_row_marginal(self, index: int) -> MultivariateStudentT:
        return self.coefficient_distribution.row_marginal(index)

    def response_coefficient_marginal(self, index: int) -> MultivariateStudentT:
        """Marginal regression-coefficient vector for one response column."""
        return self.coefficient_distribution.column_marginal(index)

    def predictive(
        self, row: Sequence[Any] | sp.MatrixBase, *, observation: bool = True
    ) -> MultivariateStudentT:
        x = sp.ImmutableMatrix(row)
        if x.cols != 1:
            if x.rows == 1:
                x = sp.ImmutableMatrix(list(x))
            else:
                raise ValueError("Prediction row must be a vector")
        if x.rows != self.posterior.coefficient_dimension:
            raise ValueError("Prediction row width must match coefficient dimension")
        inv_precision = exact_solve(
            self.posterior.precision, sp.eye(self.posterior.precision.rows)
        )
        leverage = sp.simplify((x.T * inv_precision * x)[0])
        noise = sp.S.One if observation else sp.S.Zero
        degrees = sp.simplify(self.posterior.df - self.posterior.response_dimension + 1)
        location = sp.ImmutableMatrix(self.posterior.coef_mean.T * x)
        scale = sp.ImmutableMatrix(
            sp.simplify(leverage + noise) * self.posterior.scale / degrees
        )
        return MultivariateStudentT(location, scale, degrees)

    def predict(self, rows: Any, *, observation: bool = True):
        """Return posterior-predictive means for new design rows."""
        law = self.predict_distribution(rows, observation=observation)
        return sp.ImmutableMatrix(law.mean)

    def predict_distribution(self, rows: Any, *, observation: bool = True):
        """Return the joint posterior-predictive distribution for new rows."""
        return self.joint_predictive(rows, observation=observation)

    def joint_predictive(
        self, rows: Any, *, observation: bool = True
    ) -> MatrixStudentT:
        """Joint matrix-t predictive law for several new design rows.

        Unlike :meth:`predict`, this preserves posterior dependence between the
        future response rows induced by uncertainty in the coefficient matrix.
        """
        design = _matrix(rows, name="prediction design matrix")
        if design.cols != self.posterior.coefficient_dimension:
            raise ValueError("Prediction design width must match coefficient dimension")
        row_covariance = sp.ImmutableMatrix(
            design * exact_solve(self.posterior.precision, design.T)
        )
        if observation:
            row_covariance = sp.ImmutableMatrix(row_covariance + sp.eye(design.rows))
        try:
            row_precision = exact_solve(row_covariance, sp.eye(row_covariance.rows))
        except (ValueError, ZeroDivisionError, sp.NonInvertibleMatrixError) as exc:
            raise ValueError(
                "Joint latent predictive row covariance is singular; use fewer "
                "linearly independent rows or observation=True."
            ) from exc
        return MatrixStudentT(
            sp.ImmutableMatrix(design * self.posterior.coef_mean),
            row_precision,
            self.posterior.scale,
            self.posterior.df,
        )

    def posterior_predictive(
        self, rows: Any, *, size=1000, rng=None, observation: bool = True
    ):
        from ..predictive import posterior_predictive

        return posterior_predictive(
            self, rows, size=size, rng=rng, observation=observation
        )

    def predictive_interval(
        self, rows: Any, *, level: float = 0.95, observation: bool = True
    ):
        from ..predictive import predictive_interval

        return predictive_interval(self, rows, level=level, observation=observation)

    def update(self, x: Any, y: Any) -> BayesianMultivariateLinearRegressionFit:
        """Sequential conjugate update, retaining cumulative evidence and data."""
        chunk = BayesianMultivariateLinearRegression(prior=self.posterior).fit(x, y)
        design = sp.ImmutableMatrix.vstack(self.design_matrix, chunk.design_matrix)
        response = sp.ImmutableMatrix.vstack(self.response, chunk.response)
        return BayesianMultivariateLinearRegressionFit(
            self.prior,
            chunk.posterior,
            design,
            response,
            sp.simplify(self.log_evidence + chunk.log_evidence),
        )

    def to_inference_result(self) -> InferenceResult:
        return InferenceResult(
            posterior=self,
            kind=InferenceKind.EXACT,
            log_evidence=self.log_evidence,
            steps=(
                InferenceStep(
                    "bayesian-multivariate-linear-regression",
                    f"Applied analytic Matrix-Normal/Inverse-Wishart regression update to {self.design_matrix.rows} observations.",
                    True,
                    {
                        "n_observations": self.design_matrix.rows,
                        "n_coefficients": self.design_matrix.cols,
                        "n_responses": self.response.cols,
                    },
                ),
            ),
            metadata={
                "prior": self.prior,
                "coefficient_distribution": self.coefficient_distribution,
                "covariance_distribution": self.covariance_distribution,
                "evidence": self.evidence,
            },
        )


class BayesianMultivariateLinearRegression:
    """Exact multivariate Gaussian linear regression with MNIW conjugacy."""

    def __init__(
        self,
        *,
        prior: MatrixNormalInverseWishartPrior | None = None,
        include_intercept: bool = False,
    ):
        self.prior = prior
        self.include_intercept = bool(include_intercept)

    @staticmethod
    def default_prior(
        coefficient_dimension: int, response_dimension: int
    ) -> MatrixNormalInverseWishartPrior:
        if coefficient_dimension < 1 or response_dimension < 1:
            raise ValueError("coefficient and response dimensions must be positive")
        return MatrixNormalInverseWishartPrior(
            sp.zeros(coefficient_dimension, response_dimension),
            sp.eye(coefficient_dimension) / 100,
            sp.eye(response_dimension) / 100,
            response_dimension - 1 + sp.Rational(1, 100),
        )

    def _prepare_design(self, x: Any) -> sp.ImmutableDenseMatrix:
        design = _matrix(x, name="design matrix")
        if self.include_intercept:
            design = sp.ImmutableMatrix.hstack(sp.ones(design.rows, 1), design)
        return sp.ImmutableMatrix(design)

    def design_dimension(self, x: Any) -> int:
        return self._prepare_design(x).cols

    def fit(self, x: Any, y: Any) -> BayesianMultivariateLinearRegressionFit:
        design = self._prepare_design(x)
        response = _matrix(y, name="response")
        if design.rows != response.rows:
            raise ValueError(
                "Design matrix and response must have the same number of rows"
            )
        prior = self.prior or self.default_prior(design.cols, response.cols)
        if prior.coefficient_dimension != design.cols:
            raise ValueError(
                "Prior coefficient dimension does not match design matrix width"
            )
        if prior.response_dimension != response.cols:
            raise ValueError("Prior response dimension does not match response width")

        xt = design.T
        precision_n = _symmetrize(prior.precision + xt * design)
        mean_n = sp.ImmutableMatrix(
            exact_solve(precision_n, prior.precision * prior.coef_mean + xt * response)
        )
        scale_n = _symmetrize(
            prior.scale
            + response.T * response
            + prior.coef_mean.T * prior.precision * prior.coef_mean
            - mean_n.T * precision_n * mean_n
        )
        df_n = sp.simplify(prior.df + design.rows)
        posterior = MatrixNormalInverseWishartPrior(mean_n, precision_n, scale_n, df_n)

        n = design.rows
        m = response.cols
        log_evidence = sp.simplify(
            -sp.Rational(n * m, 2) * sp.log(sp.pi)
            + sp.Rational(m, 2)
            * (
                sp.log(exact_determinant(prior.precision))
                - sp.log(exact_determinant(precision_n))
            )
            + prior.df / 2 * sp.log(exact_determinant(prior.scale))
            - df_n / 2 * sp.log(exact_determinant(scale_n))
            + _multigamma_log(df_n / 2, m)
            - _multigamma_log(prior.df / 2, m)
        )
        return BayesianMultivariateLinearRegressionFit(
            prior, posterior, design, response, log_evidence
        )

    def infer(self, x: Any, y: Any) -> InferenceResult:
        return self.fit(x, y).to_inference_result()


__all__ = [
    "BayesianMultivariateLinearRegression",
    "BayesianMultivariateLinearRegressionFit",
    "MatrixNormalInverseWishartPrior",
    "MatrixStudentT",
]
