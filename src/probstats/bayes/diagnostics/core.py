"""Posterior-data interoperability and inference-quality diagnostics."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import Any

import numpy as np
import sympy as sp

from ..core import InferenceKind, InferenceResult


class PosteriorConversionError(TypeError):
    """Raised when an inference posterior has no sensible sampling adapter."""


@dataclass(frozen=True, slots=True)
class PosteriorData:
    """Small ArviZ-shaped posterior container.

    Every posterior variable is stored with leading ``(chain, draw)`` dimensions;
    trailing dimensions represent vector/matrix-valued variables.  The class is
    independent of xarray/ArviZ so interoperability is optional.
    """

    posterior: Mapping[str, np.ndarray]
    sample_stats: Mapping[str, np.ndarray] = field(default_factory=dict)
    attrs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        posterior = {str(k): np.asarray(v) for k, v in self.posterior.items()}
        if not posterior:
            raise ValueError("posterior must contain at least one variable")
        shapes = {(v.shape[0], v.shape[1]) for v in posterior.values() if v.ndim >= 2}
        if len(shapes) != 1 or any(v.ndim < 2 for v in posterior.values()):
            raise ValueError(
                "posterior arrays must share leading (chain, draw) dimensions"
            )
        object.__setattr__(self, "posterior", MappingProxyType(posterior))
        object.__setattr__(
            self,
            "sample_stats",
            MappingProxyType(
                {str(k): np.asarray(v) for k, v in self.sample_stats.items()}
            ),
        )
        object.__setattr__(self, "attrs", MappingProxyType(dict(self.attrs)))

    @property
    def chains(self) -> int:
        return next(iter(self.posterior.values())).shape[0]

    @property
    def draws(self) -> int:
        return next(iter(self.posterior.values())).shape[1]

    def to_arviz(self) -> Any:
        """Convert to ``arviz.InferenceData`` when ArviZ is installed."""
        try:
            import arviz as az
        except ImportError as exc:
            raise ImportError(
                "ArviZ interoperability requires `pip install probstats[arviz]`."
            ) from exc
        posterior = {k: v for k, v in self.posterior.items()}
        sample_stats = (
            {k: v for k, v in self.sample_stats.items()} if self.sample_stats else None
        )
        # ArviZ 0.x accepts named group arguments, while ArviZ >= 1.0 accepts a
        # single nested group dictionary. Detect the API shape rather than using
        # a version comparison so prereleases and downstream builds remain safe.
        import inspect

        parameters = inspect.signature(az.from_dict).parameters
        if "posterior" in parameters:
            return az.from_dict(posterior=posterior, sample_stats=sample_stats)
        groups = {"posterior": posterior}
        if sample_stats is not None:
            groups["sample_stats"] = sample_stats
        return az.from_dict(groups)


def _rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    return np.random.default_rng(rng)


def _sample_mvt(
    location: np.ndarray,
    scale: np.ndarray,
    df: float,
    size: tuple[int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    z = rng.multivariate_normal(np.zeros(len(location)), scale, size=size)
    u = rng.chisquare(df, size=size)
    return location + z / np.sqrt(u[..., None] / df)


def _sympy_float(value: Any) -> float:
    return float(sp.N(value))


def _result_posterior_data(
    result: InferenceResult, *, draws: int, chains: int, rng: np.random.Generator
) -> PosteriorData:
    posterior = result.posterior
    if isinstance(posterior, PosteriorData):
        return posterior
    # Import adapters only when conversion is requested.
    from ..conjugacy import NormalInverseGamma
    from ..gp import GaussianProcessFit
    from ..laplace import GaussianLaplacePosterior, SingularLaplacePosterior
    from ..models import BayesianLinearRegressionFit
    from ..nested import EmpiricalPosterior

    if isinstance(posterior, EmpiricalPosterior):
        runs = result.metadata.get("runs")
        if runs and len(runs) >= 2:
            chains = len(runs)
            arrays = [run.posterior.resample(draws, rng=rng) for run in runs]
            samples = np.stack(arrays, axis=0)
        else:
            samples = posterior.resample(chains * draws, rng=rng).reshape(
                chains, draws, -1
            )
        names = posterior.names or tuple(f"theta_{i}" for i in range(samples.shape[-1]))
        return PosteriorData(
            {name: samples[..., i] for i, name in enumerate(names)},
            attrs={
                "source": "nested-sampling",
                "weighted_ess": posterior.effective_sample_size,
                "independent_chains": bool(runs and len(runs) >= 2),
            },
        )

    if isinstance(posterior, GaussianLaplacePosterior):
        samples = rng.multivariate_normal(
            np.asarray(posterior.mean), posterior.covariance, size=(chains, draws)
        )
        return PosteriorData(
            {
                symbol.name: samples[..., i]
                for i, symbol in enumerate(posterior.variables)
            },
            attrs={"source": "laplace", "synthetic_draws": True},
        )

    if isinstance(posterior, SingularLaplacePosterior):
        arrays = {}
        for symbol, center, order, coefficient in zip(
            posterior.variables,
            posterior.mode,
            posterior.local_orders,
            posterior.local_coefficients,
            strict=True,
        ):
            a = float(sp.N(coefficient))
            if not np.isfinite(a) or a <= 0:
                raise PosteriorConversionError(
                    f"Singular Laplace coefficient for {symbol} is not positive numeric."
                )
            radial = rng.gamma(shape=1.0 / order, scale=1.0, size=(chains, draws))
            signs = rng.choice(np.array([-1.0, 1.0]), size=(chains, draws))
            arrays[symbol.name] = center + signs * (radial / a) ** (1.0 / order)
        return PosteriorData(
            arrays,
            attrs={
                "source": "singular-laplace",
                "synthetic_draws": True,
                "local_orders": posterior.local_orders,
            },
        )

    if isinstance(posterior, BayesianLinearRegressionFit):
        fit = posterior
        alpha = _sympy_float(fit.posterior.df / 2)
        beta_scale = _sympy_float(fit.posterior.scale / 2)
        sigma2 = 1.0 / rng.gamma(alpha, scale=1.0 / beta_scale, size=(chains, draws))
        mean = np.asarray(fit.posterior.coef_mean, dtype=float).reshape(-1)
        inv_precision = np.asarray(fit.posterior.precision_inverse, dtype=float)
        z = rng.multivariate_normal(
            np.zeros(len(mean)), inv_precision, size=(chains, draws)
        )
        coef = mean + z * np.sqrt(sigma2)[..., None]
        return PosteriorData(
            {"beta": coef, "sigma2": sigma2},
            attrs={"source": "analytic-linear-regression", "synthetic_draws": True},
        )

    if isinstance(posterior, GaussianProcessFit):
        # Posterior over latent f at training inputs: K - K C^-1 K.
        x = posterior.x_train
        k = posterior.kernel.matrix(x)
        cross_solve = np.linalg.solve(posterior.cholesky, k)
        cov = k - cross_solve.T @ cross_solve
        cov = 0.5 * (cov + cov.T)
        eigmin = np.min(np.linalg.eigvalsh(cov))
        if eigmin < 0:
            cov = cov + np.eye(cov.shape[0]) * (-eigmin + 1e-12)
        mean = posterior.predict_joint(x, observation=False).mean
        samples = rng.multivariate_normal(mean, cov, size=(chains, draws))
        return PosteriorData(
            {"f": samples},
            attrs={"source": "analytic-gaussian-process", "synthetic_draws": True},
        )

    if isinstance(posterior, NormalInverseGamma):
        alpha = _sympy_float(posterior.nu)
        beta_scale = _sympy_float(posterior.beta)
        sigma2 = 1.0 / rng.gamma(alpha, scale=1.0 / beta_scale, size=(chains, draws))
        mu0 = _sympy_float(posterior.mu)
        lam = _sympy_float(posterior.lambda_)
        mu = rng.normal(mu0, np.sqrt(sigma2 / lam))
        return PosteriorData(
            {"mu": mu, "sigma2": sigma2},
            attrs={"source": "conjugate", "synthetic_draws": True},
        )

    raise PosteriorConversionError(
        f"No posterior-data adapter is defined for {type(posterior).__name__}; "
        "generic symbolic posteriors are not sampled implicitly."
    )


def to_posterior_data(
    result: InferenceResult,
    *,
    draws: int = 1000,
    chains: int = 4,
    rng: np.random.Generator | int | None = None,
) -> PosteriorData:
    if draws < 4 or chains < 1:
        raise ValueError("draws must be >=4 and chains must be >=1")
    return _result_posterior_data(result, draws=draws, chains=chains, rng=_rng(rng))


def to_arviz(
    result: InferenceResult,
    *,
    draws: int = 1000,
    chains: int = 4,
    rng: np.random.Generator | int | None = None,
) -> Any:
    return to_posterior_data(result, draws=draws, chains=chains, rng=rng).to_arviz()


def _as_scalar_chains(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("diagnostic input must have shape (chain, draw)")
    if values.shape[1] < 4:
        raise ValueError("at least four draws per chain are required")
    return values


def split_rhat(values: np.ndarray) -> float:
    """Classical split-Rhat; returns NaN when fewer than two chains are available."""
    x = _as_scalar_chains(values)
    if x.shape[0] < 2:
        return math.nan
    half = x.shape[1] // 2
    if half < 2:
        return math.nan
    x = np.concatenate([x[:, :half], x[:, -half:]], axis=0)
    n = x.shape[1]
    chain_means = x.mean(axis=1)
    between = n * np.var(chain_means, ddof=1)
    within = np.mean(np.var(x, axis=1, ddof=1))
    if within == 0:
        return 1.0 if between == 0 else math.inf
    var_hat = ((n - 1) / n) * within + between / n
    return float(np.sqrt(var_hat / within))


def effective_sample_size(values: np.ndarray) -> float:
    """Autocorrelation ESS using Geyer's initial-positive paired sequence."""
    x = _as_scalar_chains(values)
    m, n = x.shape
    centered = x - x.mean(axis=1, keepdims=True)
    variances = np.sum(centered * centered, axis=1) / n
    if np.all(variances == 0):
        return float(m * n)
    rhos = []
    for lag in range(1, n):
        cov = np.mean(
            np.sum(centered[:, : n - lag] * centered[:, lag:], axis=1) / (n - lag)
        )
        denom = float(np.mean(variances))
        rhos.append(cov / denom if denom > 0 else 0.0)
    total = 0.0
    for i in range(0, len(rhos) - 1, 2):
        pair = rhos[i] + rhos[i + 1]
        if pair <= 0:
            break
        total += pair
    tau = max(1.0, 1.0 + 2.0 * total)
    return float(min(m * n, m * n / tau))


def mcse_mean(values: np.ndarray) -> float:
    x = _as_scalar_chains(values)
    ess = effective_sample_size(x)
    return float(np.std(x.reshape(-1), ddof=1) / np.sqrt(ess)) if ess > 0 else math.inf


def sampled_diagnostics(data: PosteriorData) -> Mapping[str, Mapping[str, float]]:
    """ESS, MCSE(mean), and split-Rhat for every scalar posterior component."""
    out: dict[str, Mapping[str, float]] = {}
    for name, values in data.posterior.items():
        trailing = values.shape[2:]
        if not trailing:
            components = [(name, values)]
        else:
            components = []
            for index in np.ndindex(trailing):
                components.append(
                    (
                        name + "[" + ",".join(map(str, index)) + "]",
                        values[(slice(None), slice(None)) + index],
                    )
                )
        for label, scalar in components:
            out[label] = MappingProxyType(
                {
                    "ess": effective_sample_size(scalar),
                    "mcse_mean": mcse_mean(scalar),
                    "rhat": split_rhat(scalar),
                }
            )
    return MappingProxyType(out)


class QualityLevel(str, Enum):
    GOOD = "good"
    CAUTION = "caution"
    POOR = "poor"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class DiagnosticQuality:
    level: QualityLevel
    trustworthy: bool
    reasons: tuple[str, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "reasons", tuple(self.reasons))
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))


def assess_quality(
    result: InferenceResult, *, posterior_data: PosteriorData | None = None
) -> DiagnosticQuality:
    """Assess whether a returned inference result has adequate diagnostic support."""
    reasons: list[str] = []
    metrics: dict[str, Any] = {}

    if result.kind is InferenceKind.EXACT:
        cert = result.metadata.get("normalizer_certificate")
        if cert is not None:
            metrics["normalizer_certification_backend"] = getattr(cert, "backend", None)
            metrics["normalizer_certified_nonzero"] = (
                getattr(cert, "is_zero", None) is False
            )
            if getattr(cert, "is_zero", None) is False:
                reasons.append("The exact posterior normalizer is certified nonzero.")
            elif getattr(cert, "proven", False) is False:
                reasons.append(
                    "The exact normalizer was evaluated symbolically but nonzeroness was not independently certified."
                )
        positive_cert = result.metadata.get("positive_certificate")
        if positive_cert is not None:
            metrics["normalizer_positive_certification_backend"] = getattr(
                positive_cert, "backend", None
            )
            metrics["normalizer_certified_positive"] = (
                getattr(positive_cert, "value", None) is True
            )
            if getattr(positive_cert, "value", None) is True:
                reasons.append(
                    "The exact posterior normalizer is certified strictly positive on the supplied parameter domain."
                )
        cond = result.diagnostics.get("condition_number")
        if cond is not None:
            metrics["condition_number"] = float(cond)
            if not np.isfinite(cond) or cond > 1e12:
                reasons.append("Analytic GP covariance is severely ill-conditioned.")
                return DiagnosticQuality(QualityLevel.POOR, False, reasons, metrics)
            if cond > 1e8:
                reasons.append("Analytic GP covariance is ill-conditioned.")
                return DiagnosticQuality(QualityLevel.CAUTION, True, reasons, metrics)
        reasons.append("Inference is analytic/exact for the stated model.")
        return DiagnosticQuality(QualityLevel.GOOD, True, reasons, metrics)

    if result.kind is InferenceKind.APPROXIMATE:
        det_cert = result.diagnostics.get("precision_determinant_certificate")
        if det_cert is not None:
            metrics["precision_determinant_certification_backend"] = getattr(
                det_cert, "backend", None
            )
            metrics["precision_determinant_certified_zero"] = (
                getattr(det_cert, "is_zero", None) is True
            )
            metrics["precision_determinant_certified_nonzero"] = (
                getattr(det_cert, "is_zero", None) is False
            )
        pd_cert = result.diagnostics.get("precision_definiteness_certificate")
        if pd_cert is not None:
            metrics["precision_certified_positive_definite"] = (
                getattr(pd_cert, "positive_definite", None) is True
            )
            if getattr(pd_cert, "positive_definite", None) is True:
                reasons.append(
                    "The MAP precision matrix is symbolically certified positive definite."
                )
        geometry = result.diagnostics.get("posterior_geometry") or result.metadata.get(
            "posterior_geometry"
        )
        if geometry is not None:
            metrics["posterior_global_concavity"] = getattr(
                geometry, "globally_concave", None
            )
            metrics["posterior_continuity"] = getattr(geometry, "continuity", None)
            metrics["posterior_has_singularities"] = getattr(
                geometry, "has_singularities", None
            )
            if getattr(geometry, "globally_concave", None) is True:
                reasons.append(
                    "funcprops certified global concavity of the posterior log density."
                )
            if getattr(geometry, "has_singularities", None) is True:
                reasons.append(
                    "funcprops found singularities on the queried posterior domain."
                )
            if getattr(geometry, "continuity", None) is False:
                reasons.append(
                    "funcprops found a posterior discontinuity on the queried domain."
                )
        if geometry is not None and getattr(geometry, "continuity", None) is False:
            return DiagnosticQuality(QualityLevel.POOR, False, reasons, metrics)
        approximation = result.diagnostics.get("approximation")
        if approximation == "mc-dropout":
            training_loss = result.diagnostics.get("training_loss")
            if training_loss is not None:
                metrics["training_loss"] = float(training_loss)
                if not np.isfinite(float(training_loss)):
                    return DiagnosticQuality(
                        QualityLevel.POOR,
                        False,
                        ("Neural-network training ended with a non-finite objective.",),
                        metrics,
                    )
            reasons.append(
                "Monte Carlo dropout is an approximate function-posterior method; predictive calibration "
                "should be validated on held-out data."
            )
            return DiagnosticQuality(QualityLevel.CAUTION, True, reasons, metrics)
        if result.diagnostics.get("singular_laplace"):
            metrics["asymptotic_certified"] = bool(
                result.diagnostics.get("asymptotic_certified", False)
            )
            reasons.append(
                "Singular Laplace uses an even-order local asymptotic law rather than Gaussian curvature; "
                "global posterior shape should still be checked."
            )
            if metrics["asymptotic_certified"]:
                reasons.append(
                    "The asymptotic integral engine supplied a certified local/global Laplace remainder claim."
                )
            return DiagnosticQuality(QualityLevel.CAUTION, True, reasons, metrics)
        eig = result.diagnostics.get("precision_eigenvalues")
        if eig is not None:
            eig = np.asarray(eig, dtype=float)
            condition = (
                float(np.max(eig) / np.min(eig)) if np.min(eig) > 0 else math.inf
            )
            metrics["precision_condition_number"] = condition
            if not np.isfinite(condition) or condition > 1e10:
                reasons.append(
                    "Laplace posterior curvature is severely ill-conditioned."
                )
                return DiagnosticQuality(QualityLevel.POOR, False, reasons, metrics)
            if condition > 1e6:
                reasons.append("Laplace posterior curvature is ill-conditioned.")
                return DiagnosticQuality(QualityLevel.CAUTION, True, reasons, metrics)
        reasons.append(
            "Laplace is a local asymptotic approximation; diagnostics cannot certify global posterior shape."
        )
        return DiagnosticQuality(QualityLevel.CAUTION, True, reasons, metrics)

    # Sampled results.
    weighted_ess = result.diagnostics.get("posterior_ess")
    if weighted_ess is not None:
        metrics["weighted_ess"] = float(weighted_ess)
        if weighted_ess < 20:
            reasons.append("Posterior effective sample size is below 20.")
        elif weighted_ess < 100:
            reasons.append("Posterior effective sample size is below 100.")
    logz_error = result.diagnostics.get("log_evidence_error")
    if logz_error is not None:
        metrics["log_evidence_error"] = float(logz_error)
        if float(logz_error) > 1.0:
            reasons.append("Log-evidence uncertainty exceeds 1 nat.")
    termination = result.diagnostics.get("termination_reason")
    if termination == "max_iterations":
        reasons.append(
            "Nested sampling hit max_iterations before the evidence-fraction criterion."
        )

    if (
        posterior_data is not None
        and posterior_data.chains >= 2
        and posterior_data.attrs.get("independent_chains", False)
    ):
        per_var = sampled_diagnostics(posterior_data)
        max_rhat = max(
            (v["rhat"] for v in per_var.values() if np.isfinite(v["rhat"])),
            default=math.nan,
        )
        min_ess = min((v["ess"] for v in per_var.values()), default=math.nan)
        metrics.update({"max_rhat": max_rhat, "min_chain_ess": min_ess})
        if np.isfinite(max_rhat) and max_rhat > 1.05:
            reasons.append(f"Split R-hat is high ({max_rhat:.3g}).")
        if np.isfinite(min_ess) and min_ess < 100:
            reasons.append(f"Chain ESS is low ({min_ess:.3g}).")
    elif result.kind is InferenceKind.SAMPLED:
        reasons.append("R-hat is unavailable without multiple independent chains/runs.")

    severe = any(
        "below 20" in r
        or "exceeds 1 nat" in r
        or "max_iterations" in r
        or "R-hat is high" in r
        for r in reasons
    )
    if severe:
        return DiagnosticQuality(QualityLevel.POOR, False, reasons, metrics)
    if reasons:
        return DiagnosticQuality(QualityLevel.CAUTION, True, reasons, metrics)
    return DiagnosticQuality(
        QualityLevel.GOOD,
        True,
        ("Sampling diagnostics meet default quality thresholds.",),
        metrics,
    )


def with_quality(
    result: InferenceResult, quality: DiagnosticQuality
) -> InferenceResult:
    diagnostics = dict(result.diagnostics)
    diagnostics["quality"] = quality
    return replace(result, diagnostics=diagnostics)
