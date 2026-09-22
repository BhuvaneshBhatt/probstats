"""Gaussian-process regression with exact Gaussian conditioning."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from ..._special import normal_ppf
from ..._validation import sample_shape
from ..core import InferenceKind, InferenceResult, InferenceStep

ArrayLike = Any
MeanFunction = Callable[[np.ndarray], np.ndarray | float]
NoiseFunction = Callable[[np.ndarray], np.ndarray | float]


def _points(value: ArrayLike) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim != 2:
        raise ValueError("Input points must be a 1D or 2D array-like object.")
    if arr.shape[0] == 0:
        raise ValueError("Input points must be non-empty.")
    return arr


def _response(value: ArrayLike, n: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 1 or arr.shape[0] != n:
        raise ValueError(
            "GP response must be one-dimensional and match the number of inputs."
        )
    return arr


def _mean_vector(mean: MeanFunction | float | None, x: np.ndarray) -> np.ndarray:
    if mean is None:
        return np.zeros(x.shape[0], dtype=float)
    if np.isscalar(mean):
        return np.full(x.shape[0], float(mean), dtype=float)
    try:
        out = np.asarray(mean(x), dtype=float)
        if out.ndim == 0:
            return np.full(x.shape[0], float(out), dtype=float)
        if out.shape == (x.shape[0],):
            return out
        if out.shape == (x.shape[0], 1):
            return out[:, 0]
    except (TypeError, ValueError, IndexError):
        pass
    return np.asarray([mean(row) for row in x], dtype=float).reshape(-1)


def _noise_vector(noise: NoiseFunction | float, x: np.ndarray) -> np.ndarray:
    if np.isscalar(noise):
        value = float(noise)
        if value < 0:
            raise ValueError("noise_variance must be nonnegative.")
        return np.full(x.shape[0], value, dtype=float)
    try:
        out = np.asarray(noise(x), dtype=float)
        if out.ndim == 0:
            out = np.full(x.shape[0], float(out), dtype=float)
        elif out.shape == (x.shape[0], 1):
            out = out[:, 0]
        elif out.shape != (x.shape[0],):
            raise ValueError
    except (TypeError, ValueError, IndexError):
        out = np.asarray([noise(row) for row in x], dtype=float).reshape(-1)
    if np.any(out < 0):
        raise ValueError("noise_variance must be nonnegative.")
    return out


class Kernel(Protocol):
    """Covariance-kernel protocol."""

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float: ...

    def matrix(self, x: ArrayLike, y: ArrayLike | None = None) -> np.ndarray: ...


class KernelBase:
    """Convenience base class implementing covariance-matrix construction."""

    def matrix(self, x: ArrayLike, y: ArrayLike | None = None) -> np.ndarray:
        xa = _points(x)
        ya = xa if y is None else _points(y)
        return np.asarray([[self(a, b) for b in ya] for a in xa], dtype=float)

    def diag(self, x: ArrayLike) -> np.ndarray:
        xa = _points(x)
        return np.asarray([self(row, row) for row in xa], dtype=float)

    def __add__(self, other: KernelBase) -> SumKernel:
        return SumKernel(self, other)

    def __mul__(self, other: KernelBase) -> ProductKernel:
        return ProductKernel(self, other)


@dataclass(frozen=True, slots=True)
class CallableKernel(KernelBase):
    function: Callable[[np.ndarray, np.ndarray], float]

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        return float(self.function(x, y))


@dataclass(frozen=True, slots=True)
class ConstantKernel(KernelBase):
    variance: float = 1.0

    def __post_init__(self) -> None:
        if self.variance < 0:
            raise ValueError("variance must be nonnegative.")

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        return float(self.variance)


@dataclass(frozen=True, slots=True)
class RBFKernel(KernelBase):
    length_scale: float | Sequence[float] = 1.0
    variance: float = 1.0

    def __post_init__(self) -> None:
        ls = np.asarray(self.length_scale, dtype=float)
        if np.any(ls <= 0):
            raise ValueError("length_scale must be positive.")
        if self.variance < 0:
            raise ValueError("variance must be nonnegative.")

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        ls = np.asarray(self.length_scale, dtype=float)
        delta = (np.asarray(x, dtype=float) - np.asarray(y, dtype=float)) / ls
        return float(self.variance * np.exp(-0.5 * np.dot(delta, delta)))


@dataclass(frozen=True, slots=True)
class LinearKernel(KernelBase):
    variance: float = 1.0
    offset: float = 0.0

    def __post_init__(self) -> None:
        if self.variance < 0:
            raise ValueError("variance must be nonnegative.")

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        return float(
            self.variance
            * np.dot(np.asarray(x) - self.offset, np.asarray(y) - self.offset)
        )


@dataclass(frozen=True, slots=True)
class SumKernel(KernelBase):
    left: KernelBase
    right: KernelBase

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        return float(self.left(x, y) + self.right(x, y))


@dataclass(frozen=True, slots=True)
class ProductKernel(KernelBase):
    left: KernelBase
    right: KernelBase

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        return float(self.left(x, y) * self.right(x, y))


@dataclass(frozen=True, slots=True)
class GaussianPredictive:
    """Univariate Gaussian predictive distribution."""

    mean: float
    variance: float

    @property
    def standard_deviation(self) -> float:
        return float(np.sqrt(max(self.variance, 0.0)))

    def logpdf(self, value: float) -> float:
        if self.variance <= 0:
            return 0.0 if value == self.mean else -np.inf
        return float(
            -0.5
            * (
                np.log(2 * np.pi * self.variance)
                + (value - self.mean) ** 2 / self.variance
            )
        )

    def pdf(self, value: float) -> float:
        return float(np.exp(self.logpdf(value)))

    def quantile(self, probability: float) -> float:
        if not 0 <= probability <= 1:
            raise ValueError("probability must lie in [0, 1]")
        if self.variance == 0:
            return self.mean
        return float(self.mean + self.standard_deviation * normal_ppf(probability))

    def sample(self, size=None, rng=None):
        shape = sample_shape(size)
        generator = np.random.default_rng(rng)
        return generator.normal(self.mean, self.standard_deviation, size=shape or None)


@dataclass(frozen=True, slots=True)
class MultivariateGaussianPredictive:
    """Joint multivariate Gaussian GP predictive distribution."""

    mean: np.ndarray
    covariance: np.ndarray

    def marginal(self, index: int) -> GaussianPredictive:
        return GaussianPredictive(
            float(self.mean[index]), float(self.covariance[index, index])
        )

    def sample(self, size=None, rng=None):
        shape = sample_shape(size)
        generator = np.random.default_rng(rng)
        return generator.multivariate_normal(
            self.mean, self.covariance, size=shape or None
        )


@dataclass(frozen=True, slots=True)
class GaussianProcessFit:
    """A fitted GP with cached Cholesky factorization."""

    x_train: np.ndarray
    y_train: np.ndarray
    kernel: KernelBase
    noise_variance: NoiseFunction | float
    mean_function: MeanFunction | float | None
    covariance: np.ndarray
    cholesky: np.ndarray
    alpha: np.ndarray
    log_evidence: float
    jitter: float

    @property
    def evidence(self) -> float:
        return float(np.exp(self.log_evidence))

    def predict_joint(
        self,
        x: ArrayLike,
        *,
        observation: bool = False,
        include_jitter: bool = False,
    ) -> MultivariateGaussianPredictive:
        xt = _points(x)
        k_cross = self.kernel.matrix(self.x_train, xt)
        mean_train_adjusted = self.alpha
        pred_mean = (
            _mean_vector(self.mean_function, xt) + k_cross.T @ mean_train_adjusted
        )
        solve = np.linalg.solve(self.cholesky, k_cross)
        pred_cov = self.kernel.matrix(xt) - solve.T @ solve
        pred_cov = 0.5 * (pred_cov + pred_cov.T)
        if observation:
            pred_cov = pred_cov + np.diag(_noise_vector(self.noise_variance, xt))
        if include_jitter:
            pred_cov = pred_cov + self.jitter * np.eye(xt.shape[0])
        diag = np.diag(pred_cov).copy()
        tiny_negative = (diag < 0) & (diag > -1e-10)
        if np.any(tiny_negative):
            pred_cov = pred_cov.copy()
            idx = np.diag_indices_from(pred_cov)
            pred_cov[idx] = np.maximum(np.diag(pred_cov), 0.0)
        return MultivariateGaussianPredictive(
            np.asarray(pred_mean), np.asarray(pred_cov)
        )

    def predict_distribution(
        self, x: ArrayLike, *, observation: bool = True
    ) -> MultivariateGaussianPredictive:
        """Return the joint Gaussian posterior-predictive distribution."""
        return self.predict_joint(x, observation=observation)

    def predict(self, x: ArrayLike, *, observation: bool = True) -> np.ndarray:
        """Return posterior-predictive means for new points."""
        return self.predict_joint(x, observation=observation).mean.copy()

    def predict_marginals(
        self, x: ArrayLike, *, observation: bool = True
    ) -> tuple[GaussianPredictive, ...]:
        """Return one Gaussian predictive law per requested point."""
        joint = self.predict_joint(x, observation=observation)
        return tuple(joint.marginal(index) for index in range(len(joint.mean)))

    def posterior_predictive(
        self, x: ArrayLike, *, size=1000, rng=None, observation: bool = True
    ):
        return self.predict_distribution(x, observation=observation).sample(
            size=size, rng=rng
        )

    def predictive_interval(
        self, x: ArrayLike, *, level: float = 0.95, observation: bool = True
    ):
        if not 0 < level < 1:
            raise ValueError("level must lie strictly between 0 and 1")
        alpha = (1 - level) / 2
        joint = self.predict_distribution(x, observation=observation)
        return tuple(
            (joint.marginal(i).quantile(alpha), joint.marginal(i).quantile(1 - alpha))
            for i in range(len(joint.mean))
        )

    def to_inference_result(self) -> InferenceResult:
        return InferenceResult(
            posterior=self,
            kind=InferenceKind.EXACT,
            log_evidence=self.log_evidence,
            steps=(
                InferenceStep(
                    "gaussian-process-conditioning",
                    f"Conditioned a Gaussian process on {self.x_train.shape[0]} observations using Cholesky factorization.",
                    True,
                    {
                        "n_observations": self.x_train.shape[0],
                        "input_dimension": self.x_train.shape[1],
                    },
                ),
            ),
            diagnostics={
                "condition_number": float(np.linalg.cond(self.covariance)),
                "jitter": self.jitter,
            },
            metadata={
                "kernel": self.kernel,
                "noise_variance": self.noise_variance,
                "evidence": self.evidence,
            },
        )


class GaussianProcessRegressor:
    """Exact GP regression for a fixed covariance kernel and Gaussian observation noise."""

    def __init__(
        self,
        kernel: KernelBase | Callable[[np.ndarray, np.ndarray], float],
        *,
        noise_variance: NoiseFunction | float = 0.0,
        mean_function: MeanFunction | float | None = None,
        jitter: float = 1e-10,
    ) -> None:
        if np.isscalar(noise_variance) and float(noise_variance) < 0:
            raise ValueError("noise_variance must be nonnegative.")
        if jitter < 0:
            raise ValueError("jitter must be nonnegative.")
        self.kernel = (
            kernel if isinstance(kernel, KernelBase) else CallableKernel(kernel)
        )
        self.noise_variance = (
            float(noise_variance) if np.isscalar(noise_variance) else noise_variance
        )
        self.mean_function = mean_function
        self.jitter = float(jitter)

    def fit(self, x: ArrayLike, y: ArrayLike) -> GaussianProcessFit:
        xa = _points(x)
        ya = _response(y, xa.shape[0])
        mean = _mean_vector(self.mean_function, xa)
        residual = ya - mean
        covariance = self.kernel.matrix(xa)
        covariance = 0.5 * (covariance + covariance.T)
        covariance = covariance + np.diag(
            _noise_vector(self.noise_variance, xa) + self.jitter
        )
        try:
            cholesky = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "GP covariance matrix is not positive definite; increase jitter or fix the kernel."
            ) from exc
        tmp = np.linalg.solve(cholesky, residual)
        alpha = np.linalg.solve(cholesky.T, tmp)
        log_det = 2.0 * np.sum(np.log(np.diag(cholesky)))
        log_evidence = float(
            -0.5 * residual @ alpha
            - 0.5 * log_det
            - 0.5 * xa.shape[0] * np.log(2.0 * np.pi)
        )
        return GaussianProcessFit(
            xa,
            ya,
            self.kernel,
            self.noise_variance,
            self.mean_function,
            covariance,
            cholesky,
            alpha,
            log_evidence,
            self.jitter,
        )

    def infer(self, x: ArrayLike, y: ArrayLike) -> InferenceResult:
        return self.fit(x, y).to_inference_result()


def gaussian_process_log_likelihood(
    x: ArrayLike,
    y: ArrayLike,
    kernel: KernelBase | Callable[[np.ndarray, np.ndarray], float],
    *,
    noise_variance: NoiseFunction | float = 0.0,
    mean_function: MeanFunction | float | None = None,
    jitter: float = 1e-10,
) -> float:
    """Exact Gaussian-process log marginal likelihood."""
    return (
        GaussianProcessRegressor(
            kernel,
            noise_variance=noise_variance,
            mean_function=mean_function,
            jitter=jitter,
        )
        .fit(x, y)
        .log_evidence
    )


@dataclass(frozen=True, slots=True)
class WeightedGaussianMixture:
    """Finite mixture of Gaussian predictive distributions."""

    weights: np.ndarray
    components: tuple[GaussianPredictive, ...]

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=float)
        if weights.ndim != 1 or len(weights) != len(self.components):
            raise ValueError("weights must be one-dimensional and match components.")
        if np.any(weights < 0) or not np.isfinite(weights).all() or weights.sum() <= 0:
            raise ValueError(
                "weights must be finite, nonnegative, and have positive sum."
            )
        object.__setattr__(self, "weights", weights / weights.sum())

    @property
    def mean(self) -> float:
        means = np.asarray([component.mean for component in self.components])
        return float(self.weights @ means)

    @property
    def variance(self) -> float:
        means = np.asarray([component.mean for component in self.components])
        second = np.asarray(
            [component.variance + component.mean**2 for component in self.components]
        )
        return float(self.weights @ second - (self.weights @ means) ** 2)

    def pdf(self, value: float) -> float:
        return float(
            sum(w * c.pdf(value) for w, c in zip(self.weights, self.components))
        )


def predict_hyperparameter_mixture(
    x_train: ArrayLike,
    y_train: ArrayLike,
    x_new: ArrayLike,
    parameter_samples: Sequence[Mapping[str, Any]],
    weights: Sequence[float],
    factory: Callable[[Mapping[str, Any]], GaussianProcessRegressor],
    *,
    observation: bool = False,
) -> tuple[WeightedGaussianMixture, ...]:
    """Average GP predictions over weighted hyperparameter posterior samples.

    The returned mixture retains hyperparameter uncertainty instead of collapsing
    predictions to a single plug-in kernel configuration.
    """
    if len(parameter_samples) == 0:
        raise ValueError("parameter_samples must be non-empty.")
    if len(parameter_samples) != len(weights):
        raise ValueError("parameter_samples and weights must have the same length.")
    predictions = [
        factory(sample)
        .fit(x_train, y_train)
        .predict_marginals(x_new, observation=observation)
        for sample in parameter_samples
    ]
    n_points = len(predictions[0])
    if any(len(item) != n_points for item in predictions):
        raise ValueError("All GP predictions must have the same number of points.")
    w = np.asarray(weights, dtype=float)
    return tuple(
        WeightedGaussianMixture(
            w, tuple(prediction[index] for prediction in predictions)
        )
        for index in range(n_points)
    )
