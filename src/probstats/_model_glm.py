"""Generalized linear-model families, links, and fitting."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import mpmath as mp
import numpy as np

from ._model_design import DesignMatrix, design_matrix
from ._special import normal_cdf
from ._tabular import column_mapping, labeled_vector
from .results import ModelFitResult, coefficient_frame


class Link:
    name = "link"

    def link(self, mu):
        raise NotImplementedError

    def inverse(self, eta):
        raise NotImplementedError

    def dmu_deta(self, eta):
        raise NotImplementedError


class IdentityLink(Link):
    name = "identity"

    def link(self, mu):
        return np.asarray(mu, dtype=float)

    def inverse(self, eta):
        return np.asarray(eta, dtype=float)

    def dmu_deta(self, eta):
        return np.ones_like(np.asarray(eta, dtype=float))


class LogLink(Link):
    name = "log"

    def link(self, mu):
        return np.log(np.clip(mu, 1e-15, None))

    def inverse(self, eta):
        return np.exp(np.clip(eta, -700, 700))

    def dmu_deta(self, eta):
        return self.inverse(eta)


class LogitLink(Link):
    name = "logit"

    def link(self, mu):
        mu = np.clip(mu, 1e-15, 1 - 1e-15)
        return np.log(mu / (1 - mu))

    def inverse(self, eta):
        eta = np.asarray(eta, dtype=float)
        out = np.empty_like(eta)
        mask = eta >= 0
        z = np.exp(-np.clip(eta[mask], None, 700))
        out[mask] = 1 / (1 + z)
        z = np.exp(np.clip(eta[~mask], -700, None))
        out[~mask] = z / (1 + z)
        return np.clip(out, 1e-15, 1 - 1e-15)

    def dmu_deta(self, eta):
        mu = self.inverse(eta)
        return mu * (1 - mu)


class ProbitLink(Link):
    name = "probit"

    def link(self, mu):
        mu = np.clip(np.asarray(mu, dtype=float), 1e-15, 1 - 1e-15)
        return np.array([float(mp.sqrt(2) * mp.erfinv(2 * v - 1)) for v in mu])

    def inverse(self, eta):
        return np.array([normal_cdf(float(v)) for v in np.asarray(eta, dtype=float)])

    def dmu_deta(self, eta):
        eta = np.asarray(eta, dtype=float)
        return np.exp(-0.5 * eta**2) / math.sqrt(2 * math.pi)


class InverseLink(Link):
    name = "inverse"

    def link(self, mu):
        return 1 / np.clip(np.asarray(mu, dtype=float), 1e-15, None)

    def inverse(self, eta):
        eta = np.asarray(eta, dtype=float)
        return 1 / np.where(
            np.abs(eta) < 1e-15, np.sign(eta) * 1e-15 + (eta == 0) * 1e-15, eta
        )

    def dmu_deta(self, eta):
        eta = np.asarray(eta, dtype=float)
        return -1 / np.where(np.abs(eta) < 1e-15, 1e-15, eta) ** 2


def _resolve_link(link: str | Link | None, default: Link) -> Link:
    if link is None:
        return default
    if isinstance(link, Link):
        return link
    table = {
        "identity": IdentityLink,
        "log": LogLink,
        "logit": LogitLink,
        "probit": ProbitLink,
        "inverse": InverseLink,
    }
    try:
        return table[str(link).lower()]()
    except KeyError as exc:
        raise ValueError(f"unknown link {link!r}") from exc


class GLMFamily:
    name = "family"
    fixed_dispersion = False
    default_link: Link = IdentityLink()

    def variance(self, mu):
        raise NotImplementedError

    def deviance(self, y, mu):
        raise NotImplementedError

    def log_likelihood(self, y, mu, dispersion=1.0):
        raise NotImplementedError

    def validate(self, y):
        if not np.all(np.isfinite(y)):
            raise ValueError("response must be finite")

    def initialize(self, y):
        return np.asarray(y, dtype=float)


class GaussianFamily(GLMFamily):
    name = "gaussian"
    default_link = IdentityLink()

    def variance(self, mu):
        return np.ones_like(mu)

    def deviance(self, y, mu):
        return float(np.sum((y - mu) ** 2))

    def log_likelihood(self, y, mu, dispersion=1.0):
        return float(
            -0.5 * np.sum(np.log(2 * math.pi * dispersion) + (y - mu) ** 2 / dispersion)
        )


class BinomialFamily(GLMFamily):
    name = "binomial"
    fixed_dispersion = True
    default_link = LogitLink()

    def validate(self, y):
        super().validate(y)
        if np.any((y < 0) | (y > 1)):
            raise ValueError("binomial GLM response must lie in [0, 1]")

    def initialize(self, y):
        return np.clip((y + 0.5) / 2, 1e-6, 1 - 1e-6)

    def variance(self, mu):
        return np.clip(mu * (1 - mu), 1e-15, None)

    def deviance(self, y, mu):
        mu = np.clip(mu, 1e-15, 1 - 1e-15)
        a = np.zeros_like(y, dtype=float)
        b = np.zeros_like(y, dtype=float)
        positive = y > 0
        below_one = y < 1
        a[positive] = y[positive] * np.log(y[positive] / mu[positive])
        b[below_one] = (1 - y[below_one]) * np.log(
            (1 - y[below_one]) / (1 - mu[below_one])
        )
        return float(2 * np.sum(a + b))

    def log_likelihood(self, y, mu, dispersion=1.0):
        mu = np.clip(mu, 1e-15, 1 - 1e-15)
        return float(np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu)))


class PoissonFamily(GLMFamily):
    name = "poisson"
    fixed_dispersion = True
    default_link = LogLink()

    def validate(self, y):
        super().validate(y)
        if np.any(y < 0):
            raise ValueError("Poisson GLM response must be nonnegative")

    def initialize(self, y):
        return np.maximum(y, 0.1)

    def variance(self, mu):
        return np.clip(mu, 1e-15, None)

    def deviance(self, y, mu):
        mu = np.clip(mu, 1e-15, None)
        part = np.asarray(mu, dtype=float).copy()
        positive = y > 0
        part[positive] = y[positive] * np.log(y[positive] / mu[positive]) - (
            y[positive] - mu[positive]
        )
        return float(2 * np.sum(part))

    def log_likelihood(self, y, mu, dispersion=1.0):
        mu = np.clip(mu, 1e-15, None)
        return float(
            np.sum(y * np.log(mu) - mu - np.array([math.lgamma(v + 1) for v in y]))
        )


class GammaFamily(GLMFamily):
    name = "gamma"
    default_link = InverseLink()

    def validate(self, y):
        super().validate(y)
        if np.any(y <= 0):
            raise ValueError("Gamma GLM response must be positive")

    def variance(self, mu):
        return np.clip(mu, 1e-15, None) ** 2

    def deviance(self, y, mu):
        mu = np.clip(mu, 1e-15, None)
        return float(2 * np.sum((y - mu) / mu - np.log(y / mu)))

    def log_likelihood(self, y, mu, dispersion=1.0):
        # GLM Gamma with Var(Y)=dispersion*mu^2: shape=1/dispersion, scale=mu*dispersion.
        phi = max(float(dispersion), 1e-15)
        shape = 1 / phi
        return float(
            np.sum(
                (shape - 1) * np.log(y)
                - y / (mu * phi)
                - shape * np.log(mu * phi)
                - math.lgamma(shape)
            )
        )


@dataclass(frozen=True)
class NegativeBinomialFamily(GLMFamily):
    """NB2 GLM family with Var(Y)=mu + alpha*mu**2."""

    alpha: float = 1.0
    name = "negative_binomial"
    fixed_dispersion = True
    default_link = LogLink()

    def __post_init__(self):
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("negative-binomial alpha must be positive")

    def validate(self, y):
        super().validate(y)
        if np.any(y < 0):
            raise ValueError("negative-binomial response must be nonnegative")

    def initialize(self, y):
        return np.maximum(y, 0.1)

    def variance(self, mu):
        mu = np.clip(np.asarray(mu, dtype=float), 1e-15, None)
        return mu + self.alpha * mu**2

    def deviance(self, y, mu):
        mu = np.clip(mu, 1e-15, None)
        inv_alpha = 1.0 / self.alpha
        first = np.zeros_like(y, dtype=float)
        positive = y > 0
        first[positive] = y[positive] * np.log(y[positive] / mu[positive])
        second = (y + inv_alpha) * np.log((y + inv_alpha) / (mu + inv_alpha))
        return float(2 * np.sum(first - second))

    def log_likelihood(self, y, mu, dispersion=1.0):
        mu = np.clip(mu, 1e-15, None)
        r = 1.0 / self.alpha
        return float(
            np.sum(
                [
                    math.lgamma(v + r)
                    - math.lgamma(r)
                    - math.lgamma(v + 1)
                    + r * math.log(r / (r + m))
                    + v * math.log(m / (r + m))
                    for v, m in zip(y, mu, strict=True)
                ]
            )
        )


class QuasiPoissonFamily(PoissonFamily):
    name = "quasipoisson"
    fixed_dispersion = False

    def log_likelihood(self, y, mu, dispersion=1.0):
        return math.nan


class QuasiBinomialFamily(BinomialFamily):
    name = "quasibinomial"
    fixed_dispersion = False

    def log_likelihood(self, y, mu, dispersion=1.0):
        return math.nan


def _observation_vector(
    value, data: Mapping[str, Any], n: int, *, name: str, default: float
) -> np.ndarray:
    if value is None:
        return np.full(n, default, dtype=float)
    if isinstance(value, str):
        if value not in data:
            raise ValueError(f"missing {name} column {value!r}")
        arr = np.asarray(data[value], dtype=float)
    elif np.isscalar(value):
        arr = np.full(n, float(value), dtype=float)
    else:
        arr = np.asarray(value, dtype=float)
    arr = arr.reshape(-1)
    if len(arr) != n or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain one finite value per observation")
    return arr


def _weighted_deviance(fam: GLMFamily, y, mu, weights, trials=None) -> float:
    scale = np.asarray(weights, dtype=float)
    if trials is not None:
        scale = scale * np.asarray(trials, dtype=float)
    # Family deviance implementations are sums; evaluate per observation to preserve weights.
    return float(
        sum(
            float(w) * fam.deviance(np.asarray([yy]), np.asarray([mm]))
            for yy, mm, w in zip(y, mu, scale, strict=True)
        )
    )


def _weighted_log_likelihood(fam: GLMFamily, y, mu, weights, trials=None) -> float:
    if isinstance(fam, (QuasiPoissonFamily, QuasiBinomialFamily)):
        return math.nan
    weights = np.asarray(weights, dtype=float)
    if isinstance(fam, BinomialFamily) and trials is not None:
        total = 0.0
        for prop, m, w, tr in zip(y, mu, weights, trials, strict=True):
            successes = prop * tr
            total += w * (
                math.lgamma(tr + 1)
                - math.lgamma(successes + 1)
                - math.lgamma(tr - successes + 1)
                + successes * math.log(max(m, 1e-15))
                + (tr - successes) * math.log(max(1 - m, 1e-15))
            )
        return float(total)
    return float(
        sum(
            float(w) * fam.log_likelihood(np.asarray([yy]), np.asarray([mm]))
            for yy, mm, w in zip(y, mu, weights, strict=True)
        )
    )


def resolve_family(
    family: str | GLMFamily, link: str | Link | None = None
) -> tuple[GLMFamily, Link]:
    if isinstance(family, GLMFamily):
        fam = family
    else:
        table = {
            "gaussian": GaussianFamily,
            "normal": GaussianFamily,
            "binomial": BinomialFamily,
            "poisson": PoissonFamily,
            "gamma": GammaFamily,
            "negative_binomial": NegativeBinomialFamily,
            "negativebinomial": NegativeBinomialFamily,
            "nb": NegativeBinomialFamily,
            "quasipoisson": QuasiPoissonFamily,
            "quasibinomial": QuasiBinomialFamily,
        }
        try:
            fam = table[str(family).lower()]()
        except KeyError as exc:
            raise ValueError(f"unknown GLM family {family!r}") from exc
    return fam, _resolve_link(link, fam.default_link)


@dataclass(frozen=True)
class GLMResult(ModelFitResult):
    design_matrix: DesignMatrix
    coefficients: np.ndarray
    standard_errors: np.ndarray
    statistics: np.ndarray
    pvalues: np.ndarray
    covariance: np.ndarray
    fitted_mean: np.ndarray
    linear_predictor: np.ndarray
    residuals: np.ndarray
    deviance: float
    null_deviance: float
    dispersion: float
    df_resid: int
    log_likelihood: float
    aic: float
    family: str
    link: str
    converged: bool
    iterations: int
    observation_weights: np.ndarray | None = None
    offset: np.ndarray | None = None
    trials: np.ndarray | None = None
    offset_name: str | None = None

    @property
    def coefficient_names(self) -> tuple[str, ...]:
        return self.design_matrix.column_names

    def coefficient_table(self):
        """Return coefficient estimates as a labeled pandas table."""
        return coefficient_frame(
            self.coefficient_names,
            self.coefficients,
            self.standard_errors,
            self.statistics,
            self.pvalues,
        )

    def to_frame(self):
        return self.coefficient_table()

    def predict(self, data: Mapping[str, Any], *, linear=False, offset=None):
        x = self.design_matrix.transform(data)
        _, index = column_mapping(data)
        eta = x @ self.coefficients
        offset_arg = self.offset_name if offset is None else offset
        if offset_arg is not None:
            off = _observation_vector(
                offset_arg, data, len(eta), name="offset", default=0.0
            )
            eta = eta + off
        if linear:
            return labeled_vector(eta, index, name="linear_predictor")
        _, link = resolve_family(self.family, self.link)
        return labeled_vector(
            link.inverse(eta), index, name=self.design_matrix.response_name
        )


def glm(
    formula: str,
    data: Mapping[str, Any],
    *,
    family="gaussian",
    link=None,
    contrasts=None,
    offset=None,
    weights=None,
    trials=None,
    max_iter=100,
    tol=1e-8,
) -> GLMResult:
    """Fit a generalized linear model by iteratively reweighted least squares."""
    dm = design_matrix(formula, data, contrasts=contrasts)
    x, y_raw = dm.matrix, dm.response
    n, p = x.shape
    if p == 0 or n <= p:
        raise ValueError(
            "GLM requires a nonempty design with residual degrees of freedom"
        )
    if np.linalg.matrix_rank(x) < p:
        raise ValueError("GLM design matrix is rank deficient")
    fam, link_obj = resolve_family(family, link)
    obs_weights = _observation_vector(weights, data, n, name="weights", default=1.0)
    if np.any(obs_weights <= 0):
        raise ValueError("weights must be positive")
    offset_values = _observation_vector(offset, data, n, name="offset", default=0.0)
    trial_values = None
    y = np.asarray(y_raw, dtype=float)
    if trials is not None:
        if not isinstance(fam, BinomialFamily):
            raise ValueError("trials is only valid for binomial/quasibinomial GLMs")
        trial_values = _observation_vector(trials, data, n, name="trials", default=1.0)
        if np.any(trial_values <= 0) or np.any(
            np.abs(trial_values - np.round(trial_values)) > 1e-10
        ):
            raise ValueError("binomial trials must be positive integers")
        if np.any(y < 0) or np.any(y > trial_values):
            raise ValueError(
                "grouped-binomial response must be successes between 0 and trials"
            )
        y = y / trial_values
    fam.validate(y)
    mu0 = np.asarray(fam.initialize(y), dtype=float)
    # Start from a constant valid mean when an intercept is present. This is
    # substantially safer for inverse/log links than regressing transformed y.
    if dm.intercept:
        beta = np.zeros(p, dtype=float)
        mean0 = float(np.mean(mu0))
        beta[0] = float(np.asarray(link_obj.link(np.asarray([mean0])))[0]) - float(
            np.average(offset_values, weights=obs_weights)
        )
    else:
        beta = np.linalg.lstsq(
            x, np.asarray(link_obj.link(mu0), dtype=float) - offset_values, rcond=None
        )[0]

    def valid_mean(mu):
        return (
            np.all(np.isfinite(mu))
            and (not isinstance(fam, BinomialFamily) or np.all((mu > 0) & (mu < 1)))
            and (
                not isinstance(
                    fam, (PoissonFamily, GammaFamily, NegativeBinomialFamily)
                )
                or np.all(mu > 0)
            )
        )

    converged = False
    weights = np.ones(n)
    old_deviance = math.inf
    for iteration in range(1, max_iter + 1):
        eta = x @ beta + offset_values
        mu = np.asarray(link_obj.inverse(eta), dtype=float)
        if not valid_mean(mu):
            raise ValueError(
                f"{fam.name}/{link_obj.name} produced an invalid mean during IRLS"
            )
        current_deviance = _weighted_deviance(fam, y, mu, obs_weights, trial_values)
        derivative = np.asarray(link_obj.dmu_deta(eta), dtype=float)
        variance = np.asarray(fam.variance(mu), dtype=float)
        safe_derivative = np.where(
            np.abs(derivative) < 1e-15,
            np.copysign(1e-15, derivative + (derivative == 0)),
            derivative,
        )
        working_weights = (
            np.maximum(derivative**2 / variance, 1e-15)
            * obs_weights
            * (trial_values if trial_values is not None else 1.0)
        )
        z = eta + (y - mu) / safe_derivative
        sqrt_w = np.sqrt(working_weights)
        wx = x * sqrt_w[:, None]
        wz = (z - offset_values) * sqrt_w
        raw_candidate = np.linalg.lstsq(wx, wz, rcond=None)[0]

        # Fisher scoring can overshoot for non-identity links. Step halve until
        # the mean is in-domain and deviance does not increase materially.
        step = 1.0
        candidate = raw_candidate
        while step >= 2**-20:
            candidate = beta + step * (raw_candidate - beta)
            candidate_mu = np.asarray(
                link_obj.inverse(x @ candidate + offset_values), dtype=float
            )
            if valid_mean(candidate_mu):
                candidate_deviance = _weighted_deviance(
                    fam, y, candidate_mu, obs_weights, trial_values
                )
                if (
                    math.isfinite(candidate_deviance)
                    and candidate_deviance <= current_deviance + 1e-10
                ):
                    break
            step *= 0.5
        else:
            break
        if np.linalg.norm(candidate - beta) <= tol * (1 + np.linalg.norm(beta)):
            beta = candidate
            converged = True
            break
        if abs(old_deviance - candidate_deviance) <= tol * (
            1 + abs(candidate_deviance)
        ):
            beta = candidate
            converged = True
            break
        beta = candidate
        old_deviance = candidate_deviance

    eta = x @ beta + offset_values
    mu = np.asarray(link_obj.inverse(eta), dtype=float)
    if not valid_mean(mu):
        raise ValueError(f"{fam.name}/{link_obj.name} produced an invalid fitted mean")
    deviance = _weighted_deviance(fam, y, mu, obs_weights, trial_values)
    df_resid = n - p
    dispersion = (
        1.0
        if fam.fixed_dispersion
        else max(
            float(
                np.sum(
                    obs_weights
                    * (trial_values if trial_values is not None else 1.0)
                    * (y - mu) ** 2
                    / fam.variance(mu)
                )
            )
            / df_resid,
            1e-15,
        )
    )
    derivative = np.asarray(link_obj.dmu_deta(eta), dtype=float)
    working_weights = (
        np.maximum(derivative**2 / fam.variance(mu), 1e-15)
        * obs_weights
        * (trial_values if trial_values is not None else 1.0)
    )
    covariance = dispersion * np.linalg.pinv(x.T @ (working_weights[:, None] * x))
    se = np.sqrt(np.maximum(np.diag(covariance), 0))
    stats = np.divide(beta, se, out=np.full_like(beta, math.inf), where=se > 0)
    pvalues = np.array(
        [2 * min(normal_cdf(float(v)), 1 - normal_cdf(float(v))) for v in stats]
    )
    residuals = y - mu
    null_mu = np.full(
        n,
        np.average(
            y, weights=obs_weights * (trial_values if trial_values is not None else 1.0)
        ),
    )
    if isinstance(fam, BinomialFamily):
        null_mu = np.clip(null_mu, 1e-15, 1 - 1e-15)
    elif isinstance(fam, (PoissonFamily, GammaFamily)):
        null_mu = np.clip(null_mu, 1e-15, None)
    null_deviance = _weighted_deviance(fam, y, null_mu, obs_weights, trial_values)
    ll = _weighted_log_likelihood(fam, y, mu, obs_weights, trial_values)
    aic = math.nan if not math.isfinite(ll) else 2 * p - 2 * ll
    return GLMResult(
        dm,
        beta,
        se,
        stats,
        pvalues,
        covariance,
        mu,
        eta,
        residuals,
        deviance,
        null_deviance,
        dispersion,
        df_resid,
        ll,
        aic,
        fam.name,
        link_obj.name,
        converged,
        iteration,
        obs_weights,
        offset_values,
        trial_values,
        offset if isinstance(offset, str) else None,
    )


__all__ = [
    "BinomialFamily",
    "GLMFamily",
    "GLMResult",
    "GammaFamily",
    "GaussianFamily",
    "IdentityLink",
    "InverseLink",
    "Link",
    "LogLink",
    "LogitLink",
    "NegativeBinomialFamily",
    "PoissonFamily",
    "ProbitLink",
    "QuasiBinomialFamily",
    "QuasiPoissonFamily",
    "glm",
    "resolve_family",
]
