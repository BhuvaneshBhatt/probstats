"""Classical parameter estimation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ._special import normal_ppf, t_ppf
from .distributions import Bernoulli, Exponential, Gamma, Normal, Poisson, Uniform
from .results import StatisticalResult


@dataclass(frozen=True)
class EstimationResult(StatisticalResult):
    distribution: object
    method: str
    n: int
    log_likelihood: float
    parameters: dict
    converged: bool = True

    @property
    def k(self):
        return len(self.parameters)

    @property
    def aic(self):
        return 2 * self.k - 2 * self.log_likelihood

    @property
    def bic(self):
        return math.log(self.n) * self.k - 2 * self.log_likelihood


def _x(data):
    x = np.asarray(data, dtype=float)
    if x.size == 0:
        raise ValueError("data must not be empty")
    return x


def _ll(dist, x):
    return float(sum(float(dist.logpdf(v).evalf()) for v in x))


def maximum_likelihood(family, data):
    """Fit a supported distribution family by closed-form maximum likelihood."""
    x = _x(data)
    cls = family if isinstance(family, type) else type(family)
    if cls is Normal:
        mu = float(x.mean())
        sigma = float(np.sqrt(np.mean((x - mu) ** 2)))
        if sigma <= 0:
            raise ValueError("Normal MLE requires nonzero sample variance")
        d = Normal(mu, sigma)
        p = {"mean": mu, "sigma": sigma}
    elif cls is Bernoulli:
        if np.any((x != 0) & (x != 1)):
            raise ValueError("Bernoulli data must be 0/1")
        q = float(x.mean())
        d = Bernoulli(q)
        p = {"p": q}
    elif cls is Poisson:
        if np.any(x < 0) or np.any(x != np.floor(x)):
            raise ValueError("Poisson data must be nonnegative integers")
        lam = float(x.mean())
        d = Poisson(lam)
        p = {"rate": lam}
    elif cls is Exponential:
        if np.any(x < 0):
            raise ValueError("Exponential data must be nonnegative")
        rate = 1 / float(x.mean())
        d = Exponential(rate)
        p = {"rate": rate}
    elif cls is Uniform:
        lo = float(x.min())
        hi = float(x.max())
        if lo == hi:
            raise ValueError("Uniform MLE requires a nondegenerate sample")
        d = Uniform(lo, hi)
        p = {"low": lo, "high": hi}
    else:
        raise NotImplementedError(f"MLE is not implemented for {cls.__name__}")
    return EstimationResult(d, "maximum_likelihood", x.size, _ll(d, x), p)


def method_of_moments(family, data):
    """Fit a supported distribution family by matching empirical moments."""
    x = _x(data)
    cls = family if isinstance(family, type) else type(family)
    m = float(x.mean())
    v = float(np.var(x, ddof=0))
    if cls is Normal:
        s = math.sqrt(v)
        d = Normal(m, s)
        p = {"mean": m, "sigma": s}
    elif cls is Poisson:
        d = Poisson(m)
        p = {"rate": m}
    elif cls is Exponential:
        r = 1 / m
        d = Exponential(r)
        p = {"rate": r}
    elif cls is Gamma:
        if m <= 0 or v <= 0:
            raise ValueError(
                "Gamma method of moments requires positive mean and variance"
            )
        shape = m * m / v
        scale = v / m
        d = Gamma(shape, scale)
        p = {"shape": shape, "scale": scale}
    else:
        raise NotImplementedError(
            f"method of moments is not implemented for {cls.__name__}"
        )
    return EstimationResult(d, "method_of_moments", x.size, _ll(d, x), p)


__all__ = ["EstimationResult", "maximum_likelihood", "method_of_moments"]


@dataclass(frozen=True)
class ConfidenceInterval:
    low: float
    high: float
    confidence: float
    method: str
    estimate: float
    standard_error: float | None = None

    def contains(self, value):
        return self.low <= value <= self.high


def mean_confidence_interval(data, confidence=0.95, sigma=None):
    """Return a two-sided confidence interval for a population mean."""
    x = _x(data)
    n = x.size
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    alpha = 1 - confidence
    estimate = float(x.mean())
    if sigma is None:
        if n < 2:
            raise ValueError(
                "unknown-variance mean interval requires at least two observations"
            )
        se = float(x.std(ddof=1) / math.sqrt(n))
        critical = t_ppf(1 - alpha / 2, n - 1)
        method = "Student-t mean interval"
    else:
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        se = float(sigma / math.sqrt(n))
        critical = normal_ppf(1 - alpha / 2)
        method = "normal mean interval"
    return ConfidenceInterval(
        estimate - critical * se,
        estimate + critical * se,
        confidence,
        method,
        estimate,
        se,
    )


def proportion_confidence_interval(successes, n, confidence=0.95):
    """Return a Wilson score confidence interval for a binomial proportion."""
    if not (0 <= successes <= n) or n <= 0:
        raise ValueError("invalid successes or n")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    z = normal_ppf(0.5 + confidence / 2)
    p = successes / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z / den * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ConfidenceInterval(
        center - half, center + half, confidence, "Wilson score interval", p, None
    )


__all__ += [
    "ConfidenceInterval",
    "mean_confidence_interval",
    "proportion_confidence_interval",
]
