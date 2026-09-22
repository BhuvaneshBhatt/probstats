"""Empirical-Bayes hyperparameter optimization and MacKay fixed-point updates."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from math import exp, isfinite, log
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np
import sympy as sp

from ..core import InferenceResult, Model
from ..laplace import infer_laplace


class EvidenceOptimizationError(RuntimeError):
    """Raised when an evidence objective or optimizer cannot produce a valid result."""


@dataclass(frozen=True, slots=True)
class Hyperparameter:
    """One optimizable hyperparameter with a stable unconstrained parameterization.

    ``transform`` may be ``"identity"``, ``"log"`` (strictly positive), or
    ``"logit"`` (finite lower and upper bounds).  Bounds are expressed on the
    natural parameter scale.
    """

    name: str
    initial: float
    transform: str = "identity"
    bounds: tuple[float | None, float | None] = (None, None)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Hyperparameter name must be non-empty.")
        initial = float(self.initial)
        if not isfinite(initial):
            raise ValueError(f"Initial value for {self.name!r} must be finite.")
        if self.transform not in {"identity", "log", "logit"}:
            raise ValueError("transform must be 'identity', 'log', or 'logit'.")
        lo, hi = self.bounds
        lo = None if lo is None else float(lo)
        hi = None if hi is None else float(hi)
        if lo is not None and not isfinite(lo):
            raise ValueError("Hyperparameter lower bound must be finite or None.")
        if hi is not None and not isfinite(hi):
            raise ValueError("Hyperparameter upper bound must be finite or None.")
        if lo is not None and hi is not None and not lo < hi:
            raise ValueError("Hyperparameter lower bound must be below upper bound.")
        if lo is not None and initial < lo or hi is not None and initial > hi:
            raise ValueError("Initial value must lie within the hyperparameter bounds.")
        if self.transform == "log":
            if initial <= 0:
                raise ValueError("Log-transformed hyperparameters must start positive.")
            if lo is not None and lo < 0:
                raise ValueError("A log-transformed lower bound cannot be negative.")
        if self.transform == "logit":
            if lo is None or hi is None:
                raise ValueError(
                    "Logit transforms require finite lower and upper bounds."
                )
            if not lo < initial < hi:
                raise ValueError(
                    "Logit-transformed initial value must lie strictly inside bounds."
                )
        object.__setattr__(self, "initial", initial)
        object.__setattr__(self, "bounds", (lo, hi))

    def to_unconstrained(self, value: float) -> float:
        value = float(value)
        if not isfinite(value):
            raise ValueError(f"{self.name} must be finite.")
        lo, hi = self.bounds
        if lo is not None and value < float(lo) or hi is not None and value > float(hi):
            raise ValueError(f"{self.name} must lie within its bounds.")
        if self.transform == "identity":
            return value
        if self.transform == "log":
            if value <= 0:
                raise ValueError(f"{self.name} must be positive.")
            return log(value)
        if lo is None or hi is None:
            raise ValueError("Logit transforms require finite bounds.")
        if not float(lo) < value < float(hi):
            raise ValueError(f"{self.name} must lie strictly inside its bounds.")
        ratio = (value - float(lo)) / (float(hi) - value)
        return log(ratio)

    def from_unconstrained(self, value: float) -> float:
        value = float(value)
        lo, hi = self.bounds
        if self.transform == "identity":
            natural = value
        elif self.transform == "log":
            natural = exp(value)
        else:
            if lo is None or hi is None:
                raise ValueError("Logit transforms require finite bounds.")
            if value >= 0:
                z = exp(-value)
                sigmoid = 1.0 / (1.0 + z)
            else:
                z = exp(value)
                sigmoid = z / (1.0 + z)
            natural = float(lo) + (float(hi) - float(lo)) * sigmoid
        if lo is not None and natural < float(lo):
            natural = float(lo)
        if hi is not None and natural > float(hi):
            natural = float(hi)
        return natural


@dataclass(frozen=True, slots=True)
class EvidenceEvaluation:
    """One log-evidence evaluation and the fit/object that produced it."""

    log_evidence: float
    fit: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        value = float(self.log_evidence)
        if not isfinite(value):
            raise EvidenceOptimizationError(
                "Log evidence must evaluate to a finite number."
            )
        object.__setattr__(self, "log_evidence", value)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


class EvidenceObjective(Protocol):
    """Callable mapping natural-scale hyperparameters to a log-evidence evaluation."""

    def __call__(self, parameters: Mapping[str, float]) -> Any: ...


@dataclass(frozen=True, slots=True)
class EvidenceOptimizationResult:
    """Backend-independent empirical-Bayes optimization result."""

    parameters: Mapping[str, float]
    log_evidence: float
    success: bool
    method: str
    iterations: int
    evaluations: int
    history: tuple[tuple[Mapping[str, float], float], ...]
    fit: Any = None
    message: str = ""
    hyper_covariance: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))
        object.__setattr__(
            self,
            "history",
            tuple(
                (MappingProxyType(dict(params)), float(value))
                for params, value in self.history
            ),
        )
        if self.hyper_covariance is not None:
            cov = np.asarray(self.hyper_covariance, dtype=float).copy()
            if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
                raise ValueError("hyper_covariance must be square")
            if not np.all(np.isfinite(cov)) or not np.allclose(cov, cov.T):
                raise ValueError("hyper_covariance must be finite and symmetric")
            cov.setflags(write=False)
            object.__setattr__(self, "hyper_covariance", cov)


@dataclass(frozen=True, slots=True)
class MacKayFixedPointResult:
    """Result of MacKay/fixed-point empirical-Bayes iteration."""

    parameters: Mapping[str, float]
    log_evidence: float
    fit: Any
    converged: bool
    iterations: int
    history: tuple[tuple[Mapping[str, float], float], ...]
    message: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))
        object.__setattr__(
            self,
            "history",
            tuple(
                (MappingProxyType(dict(params)), float(value))
                for params, value in self.history
            ),
        )


def _coerce_evaluation(value: Any) -> EvidenceEvaluation:
    if isinstance(value, EvidenceEvaluation):
        return value
    if isinstance(value, InferenceResult):
        if value.log_evidence is None:
            raise EvidenceOptimizationError("Inference result has no log evidence.")
        return EvidenceEvaluation(float(sp.N(value.log_evidence)), fit=value)
    if hasattr(value, "log_evidence"):
        log_evidence = value.log_evidence
        return EvidenceEvaluation(float(sp.N(log_evidence)), fit=value)
    try:
        return EvidenceEvaluation(float(sp.N(value)))
    except (TypeError, ValueError) as exc:
        raise EvidenceOptimizationError(
            "Evidence objective must return a number, EvidenceEvaluation, "
            "InferenceResult, or object with a log_evidence attribute."
        ) from exc


def _natural_parameters(
    specs: Sequence[Hyperparameter], vector: Sequence[float]
) -> dict[str, float]:
    return {
        spec.name: spec.from_unconstrained(float(value))
        for spec, value in zip(specs, vector, strict=True)
    }


def _initial_vector(specs: Sequence[Hyperparameter]) -> np.ndarray:
    return np.asarray(
        [spec.to_unconstrained(spec.initial) for spec in specs], dtype=float
    )


def _finite_difference_hessian(
    objective: Callable[[np.ndarray], float], point: np.ndarray, step: float = 1e-4
) -> np.ndarray:
    """Central finite-difference Hessian in unconstrained coordinates."""
    point = np.asarray(point, dtype=float)
    n = point.size
    hessian = np.empty((n, n), dtype=float)
    f0 = objective(point)
    for i in range(n):
        ei = np.zeros(n)
        hi = step * max(1.0, abs(point[i]))
        ei[i] = hi
        hessian[i, i] = (objective(point + ei) - 2 * f0 + objective(point - ei)) / (
            hi * hi
        )
        for j in range(i + 1, n):
            ej = np.zeros(n)
            hj = step * max(1.0, abs(point[j]))
            ej[j] = hj
            value = (
                objective(point + ei + ej)
                - objective(point + ei - ej)
                - objective(point - ei + ej)
                + objective(point - ei - ej)
            ) / (4 * hi * hj)
            hessian[i, j] = value
            hessian[j, i] = value
    return hessian


def _coordinate_search(
    objective: Callable[[np.ndarray], float],
    initial: np.ndarray,
    *,
    max_iterations: int,
    tolerance: float,
    initial_step: float,
) -> tuple[np.ndarray, float, int, bool, str]:
    """Small dependency-free fallback maximizer in unconstrained coordinates."""
    point = np.asarray(initial, dtype=float).copy()
    best = objective(point)
    step = float(initial_step)
    iterations = 0
    while iterations < max_iterations and step > tolerance:
        iterations += 1
        improved = False
        for index in range(point.size):
            for direction in (1.0, -1.0):
                candidate = point.copy()
                candidate[index] += direction * step
                value = objective(candidate)
                if value > best:
                    point, best = candidate, value
                    improved = True
        if not improved:
            step *= 0.5
    success = step <= tolerance
    message = (
        "coordinate-search tolerance reached"
        if success
        else "maximum iterations reached"
    )
    return point, best, iterations, success, message


def optimize_evidence(
    objective: EvidenceObjective,
    hyperparameters: Sequence[Hyperparameter],
    *,
    method: str = "auto",
    max_iterations: int = 500,
    tolerance: float = 1e-7,
    options: Mapping[str, Any] | None = None,
    estimate_covariance: bool = True,
) -> EvidenceOptimizationResult:
    """Maximize marginal log evidence over hyperparameters.

    Optimization is performed in each hyperparameter's unconstrained coordinate.
    ``method='auto'`` prefers SciPy's L-BFGS-B when SciPy is installed and otherwise
    uses a deterministic dependency-free coordinate search.  The objective itself
    may be exact (for example a conjugate regression fit) or approximate (for
    example a Laplace inference result).
    """
    specs = tuple(hyperparameters)
    if not specs:
        raise ValueError("At least one hyperparameter is required.")
    if len({spec.name for spec in specs}) != len(specs):
        raise ValueError("Hyperparameter names must be unique.")
    if isinstance(max_iterations, (bool, np.bool_)) or not isinstance(
        max_iterations, (int, np.integer)
    ):
        raise TypeError("max_iterations must be an integer.")
    max_iterations = int(max_iterations)
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive.")
    tolerance = float(tolerance)
    if not isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive.")
    opts = dict(options or {})
    history: list[tuple[Mapping[str, float], float]] = []
    cache: dict[tuple[float, ...], EvidenceEvaluation] = {}

    def evaluate(vector: np.ndarray) -> EvidenceEvaluation:
        key = tuple(float(v) for v in np.asarray(vector, dtype=float))
        if key not in cache:
            params = _natural_parameters(specs, key)
            evaluation = _coerce_evaluation(objective(params))
            cache[key] = evaluation
            history.append((params, evaluation.log_evidence))
        return cache[key]

    def maximize_value(vector: np.ndarray) -> float:
        return evaluate(vector).log_evidence

    initial = _initial_vector(specs)
    selected = method.lower().replace("_", "-")
    point: np.ndarray
    best: float
    iterations: int
    success: bool
    message: str
    actual_method: str | None = None

    if selected in {"auto", "scipy", "l-bfgs-b", "lbfgsb"}:
        try:
            from scipy.optimize import minimize
        except ImportError:
            if selected != "auto":
                raise EvidenceOptimizationError(
                    "SciPy evidence optimization requires scipy>=1.10."
                ) from None
        else:
            result = minimize(
                lambda vector: -maximize_value(np.asarray(vector, dtype=float)),
                initial,
                method="L-BFGS-B",
                options={"maxiter": max_iterations, "ftol": tolerance, **opts},
            )
            point = np.asarray(result.x, dtype=float)
            best = maximize_value(point)
            iterations = int(getattr(result, "nit", 0))
            success = bool(result.success)
            message = str(result.message)
            actual_method = "scipy-l-bfgs-b"
    if actual_method is None:
        if selected not in {"auto", "coordinate", "coordinate-search"}:
            raise ValueError(f"Unknown evidence-optimization method {method!r}.")
        initial_step = float(opts.get("initial_step", 1.0))
        if not isfinite(initial_step) or initial_step <= 0:
            raise ValueError("initial_step must be finite and positive.")
        point, best, iterations, success, message = _coordinate_search(
            maximize_value,
            initial,
            max_iterations=max_iterations,
            tolerance=tolerance,
            initial_step=initial_step,
        )
        actual_method = "coordinate-search"

    if actual_method is None:  # all valid methods assign a backend above
        raise RuntimeError("evidence optimizer did not select a backend")
    evaluation = evaluate(point)
    covariance = None
    if estimate_covariance:
        try:
            hessian = _finite_difference_hessian(maximize_value, point)
            precision = -0.5 * (hessian + hessian.T)
            if (
                np.all(np.isfinite(precision))
                and np.linalg.eigvalsh(precision).min() > 0
            ):
                covariance = np.linalg.inv(precision)
        except (ArithmeticError, EvidenceOptimizationError, np.linalg.LinAlgError):
            covariance = None

    return EvidenceOptimizationResult(
        parameters=_natural_parameters(specs, point),
        log_evidence=best,
        success=success,
        method=actual_method,
        iterations=iterations,
        evaluations=len(cache),
        history=tuple(history),
        fit=evaluation.fit,
        message=message,
        hyper_covariance=covariance,
    )


@dataclass(frozen=True, slots=True)
class LaplaceEvidenceObjective:
    """Evidence objective that rebuilds a model and runs Laplace inference."""

    model_factory: Callable[[Mapping[str, float]], Model]
    laplace_options: Mapping[str, Any] = field(default_factory=dict)
    hyperprior_logpdf: Callable[[Mapping[str, float]], Any] | None = None

    def __call__(self, parameters: Mapping[str, float]) -> EvidenceEvaluation:
        result = infer_laplace(
            self.model_factory(parameters), **dict(self.laplace_options)
        )
        if result.log_evidence is None:
            raise EvidenceOptimizationError(
                "Laplace inference did not return log evidence."
            )
        value = float(sp.N(result.log_evidence))
        hyperprior = 0.0
        if self.hyperprior_logpdf is not None:
            hyperprior = float(sp.N(self.hyperprior_logpdf(parameters)))
        return EvidenceEvaluation(
            value + hyperprior,
            fit=result,
            metadata={
                "conditional_log_evidence": value,
                "hyperprior_logpdf": hyperprior,
            },
        )


def optimize_laplace_evidence(
    model_factory: Callable[[Mapping[str, float]], Model],
    hyperparameters: Sequence[Hyperparameter],
    *,
    laplace_options: Mapping[str, Any] | None = None,
    hyperprior_logpdf: Callable[[Mapping[str, float]], Any] | None = None,
    **optimization_options: Any,
) -> EvidenceOptimizationResult:
    """Convenience wrapper for type-II ML/MAP optimization of Laplace evidence."""
    objective = LaplaceEvidenceObjective(
        model_factory,
        dict(laplace_options or {}),
        hyperprior_logpdf,
    )
    return optimize_evidence(objective, hyperparameters, **optimization_options)


def _laplace_mean_covariance(fit: Any) -> tuple[np.ndarray, np.ndarray]:
    result = fit.fit if isinstance(fit, EvidenceEvaluation) else fit
    if isinstance(result, InferenceResult):
        posterior = result.posterior
        if not hasattr(posterior, "mean") or not hasattr(posterior, "covariance"):
            raise EvidenceOptimizationError(
                "MacKay updates require a Gaussian Laplace posterior with mean and covariance."
            )
        return np.asarray(posterior.mean, dtype=float), np.asarray(
            posterior.covariance, dtype=float
        )
    if hasattr(result, "posterior") and hasattr(result.posterior, "mean"):
        return np.asarray(result.posterior.mean, dtype=float), np.asarray(
            result.posterior.covariance, dtype=float
        )
    if hasattr(result, "mean") and hasattr(result, "covariance"):
        return np.asarray(result.mean, dtype=float), np.asarray(
            result.covariance, dtype=float
        )
    raise EvidenceOptimizationError(
        "Cannot extract Laplace posterior mean/covariance from fit."
    )


def mackay_precision_update(
    alpha: float,
    mean: Sequence[float],
    covariance: np.ndarray,
    *,
    prior_log_derivative: Callable[[float], float] | None = None,
) -> float:
    """MacKay fixed-point update for one isotropic prior precision ``alpha``.

    The update is ``alpha_new = k / (||m||^2 + tr(S) - 2 d log p(alpha)/d log alpha)``.
    With no hyperprior derivative this is the standard evidence update.
    """
    alpha = float(alpha)
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    mean_arr = np.asarray(mean, dtype=float).reshape(-1)
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (mean_arr.size, mean_arr.size):
        raise ValueError("covariance shape must match mean dimension.")
    derivative = (
        0.0 if prior_log_derivative is None else float(prior_log_derivative(log(alpha)))
    )
    denominator = float(mean_arr @ mean_arr + np.trace(covariance) - 2.0 * derivative)
    if not isfinite(denominator) or denominator <= 0:
        raise EvidenceOptimizationError(
            "MacKay alpha update produced a non-positive denominator."
        )
    return float(mean_arr.size / denominator)


def mackay_precision_noise_update(
    alpha: float,
    beta: float,
    mean: Sequence[float],
    covariance: np.ndarray,
    *,
    n_observations: int,
    squared_error: float,
    prior_log_derivatives: tuple[
        Callable[[float], float] | None, Callable[[float], float] | None
    ] = (None, None),
) -> tuple[float, float]:
    """Classical two-precision MacKay update for Gaussian regression."""
    if n_observations < 1:
        raise ValueError("n_observations must be positive.")
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha and beta must be positive.")
    mean_arr = np.asarray(mean, dtype=float).reshape(-1)
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (mean_arr.size, mean_arr.size):
        raise ValueError("covariance shape must match mean dimension.")
    da = (
        0.0
        if prior_log_derivatives[0] is None
        else float(prior_log_derivatives[0](log(alpha)))
    )
    db = (
        0.0
        if prior_log_derivatives[1] is None
        else float(prior_log_derivatives[1](log(beta)))
    )
    trace_cov = float(np.trace(covariance))
    alpha_den = float(mean_arr @ mean_arr + trace_cov - 2.0 * da)
    beta_den = float(squared_error - 2.0 * db)
    if alpha_den <= 0 or beta_den <= 0:
        raise EvidenceOptimizationError("MacKay update denominator must be positive.")
    gamma = float(mean_arr.size - alpha * trace_cov)
    alpha_new = float(mean_arr.size / alpha_den)
    beta_new = float((n_observations - gamma) / beta_den)
    if alpha_new <= 0 or beta_new <= 0 or not isfinite(alpha_new + beta_new):
        raise EvidenceOptimizationError(
            "MacKay update produced invalid precision values."
        )
    return alpha_new, beta_new


def mackay_fixed_point(
    evaluator: EvidenceObjective,
    *,
    initial_alpha: float,
    initial_beta: float | None = None,
    n_observations: int | None = None,
    squared_error: Callable[[Any], float] | None = None,
    prior_log_derivatives: tuple[
        Callable[[float], float] | None, Callable[[float], float] | None
    ] = (None, None),
    tolerance: float = 1e-6,
    max_iterations: int = 1000,
) -> MacKayFixedPointResult:
    """Run MacKay fixed-point evidence optimization over precision hyperparameters.

    ``evaluator`` receives natural-scale ``{"alpha": ..., "beta": ...}`` values and
    should return an :class:`InferenceResult`, :class:`EvidenceEvaluation`, or any
    fit object with ``log_evidence``.  One-parameter mode updates only ``alpha``.
    Two-parameter mode additionally requires ``n_observations`` and a callable that
    computes residual sum of squares from the current fit.

    The returned point is the final fixed point (or last iterate), not merely the
    best intermediate evidence value.  ``history`` retains every evaluated iterate.
    """
    if tolerance <= 0 or max_iterations < 1:
        raise ValueError("tolerance and max_iterations must be positive.")
    if initial_alpha <= 0 or (initial_beta is not None and initial_beta <= 0):
        raise ValueError("Initial precisions must be positive.")
    if initial_beta is not None and (n_observations is None or squared_error is None):
        raise ValueError(
            "Two-parameter MacKay updates require n_observations and squared_error."
        )

    parameters = {"alpha": float(initial_alpha)}
    if initial_beta is not None:
        parameters["beta"] = float(initial_beta)
    history: list[tuple[Mapping[str, float], float]] = []
    converged = False

    for iteration in range(1, max_iterations + 1):
        evaluation = _coerce_evaluation(evaluator(parameters))
        history.append((dict(parameters), evaluation.log_evidence))
        mean, covariance = _laplace_mean_covariance(evaluation.fit)
        if initial_beta is None:
            updated = {
                "alpha": mackay_precision_update(
                    parameters["alpha"],
                    mean,
                    covariance,
                    prior_log_derivative=prior_log_derivatives[0],
                )
            }
        else:
            if n_observations is None or squared_error is None:
                raise EvidenceOptimizationError(
                    "Two-parameter MacKay update is missing regression metadata."
                )
            alpha_new, beta_new = mackay_precision_noise_update(
                parameters["alpha"],
                parameters["beta"],
                mean,
                covariance,
                n_observations=n_observations,
                squared_error=float(squared_error(evaluation.fit)),
                prior_log_derivatives=prior_log_derivatives,
            )
            updated = {"alpha": alpha_new, "beta": beta_new}
        delta = max(abs(log(updated[key]) - log(parameters[key])) for key in updated)
        parameters = updated
        if delta < tolerance:
            converged = True
            break

    final_evaluation = _coerce_evaluation(evaluator(parameters))
    if not history or history[-1][0] != parameters:
        history.append((dict(parameters), final_evaluation.log_evidence))
    message = (
        "fixed-point tolerance reached" if converged else "maximum iterations reached"
    )
    return MacKayFixedPointResult(
        parameters=parameters,
        log_evidence=final_evaluation.log_evidence,
        fit=final_evaluation.fit,
        converged=converged,
        iterations=iteration,
        history=tuple(history),
        message=message,
    )


def mackay_laplace(
    model_factory: Callable[[Mapping[str, float]], Model],
    *,
    initial_alpha: float,
    initial_beta: float | None = None,
    n_observations: int | None = None,
    squared_error: Callable[[Any], float] | None = None,
    laplace_options: Mapping[str, Any] | None = None,
    hyperprior_logpdf: Callable[[Mapping[str, float]], Any] | None = None,
    **fixed_point_options: Any,
) -> MacKayFixedPointResult:
    """Convenience MacKay optimizer using repeated Laplace model fits."""
    objective = LaplaceEvidenceObjective(
        model_factory,
        dict(laplace_options or {}),
        hyperprior_logpdf,
    )
    return mackay_fixed_point(
        objective,
        initial_alpha=initial_alpha,
        initial_beta=initial_beta,
        n_observations=n_observations,
        squared_error=squared_error,
        **fixed_point_options,
    )


__all__ = [
    "EvidenceEvaluation",
    "EvidenceObjective",
    "EvidenceOptimizationError",
    "EvidenceOptimizationResult",
    "Hyperparameter",
    "LaplaceEvidenceObjective",
    "MacKayFixedPointResult",
    "mackay_fixed_point",
    "mackay_laplace",
    "mackay_precision_noise_update",
    "mackay_precision_update",
    "optimize_evidence",
    "optimize_laplace_evidence",
]
