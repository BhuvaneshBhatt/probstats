"""Common posterior-predictive interface for fitted Bayesian models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .._validation import sample_shape
from ..sampling import as_rng


@dataclass(frozen=True, slots=True)
class EmpiricalPredictive:
    """Posterior-predictive distribution represented by generated draws."""

    samples: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.samples)
        if values.shape[0] == 0:
            raise ValueError("predictive samples must not be empty")
        values = values.copy()
        values.setflags(write=False)
        object.__setattr__(self, "samples", values)

    @property
    def mean(self):
        return np.mean(self.samples, axis=0)

    def quantile(self, probability):
        p = np.asarray(probability, dtype=float)
        if np.any((p < 0) | (p > 1)):
            raise ValueError("probability must lie in [0, 1]")
        return np.quantile(self.samples, p, axis=0)

    def sample(self, size=None, rng=None):
        shape = sample_shape(size)
        generator = as_rng(rng)
        if not shape:
            index = int(generator.integers(self.samples.shape[0]))
            return self.samples[index].copy()
        indices = generator.integers(self.samples.shape[0], size=shape)
        return self.samples[indices].copy()


def _sampled_posterior(fit: Any) -> bool:
    posterior = getattr(fit, "posterior", None)
    return isinstance(posterior, dict) or (
        posterior is not None
        and hasattr(posterior, "items")
        and hasattr(fit, "chains")
        and hasattr(fit, "draws")
    )


def _predictive_draws(fit, x, predictive, *, size, rng):
    if predictive is None:
        raise TypeError(
            "sampled posterior prediction requires predictive(draw, x, rng=...)"
        )
    shape = sample_shape(size)
    if not shape:
        shape = (1,)
    count = int(np.prod(shape))
    generator = as_rng(rng)
    posterior = fit.posterior
    flattened = {
        name: np.asarray(values).reshape((-1,) + np.asarray(values).shape[2:])
        for name, values in posterior.items()
    }
    available = next(iter(flattened.values())).shape[0]
    indices = generator.integers(available, size=count)
    generated = []
    for index in indices:
        draw = {name: values[index] for name, values in flattened.items()}
        generated.append(np.asarray(predictive(draw, x, rng=generator)))
    result = np.asarray(generated)
    return result.reshape(shape + result.shape[1:])


def predict_distribution(
    fit: Any,
    x: Any,
    *,
    observation: bool = True,
    predictive=None,
    size=2000,
    rng=None,
):
    """Return an exact, approximate, or empirical predictive distribution."""
    if _sampled_posterior(fit):
        draws = _predictive_draws(fit, x, predictive, size=size, rng=rng)
        shape = sample_shape(size) or (1,)
        return EmpiricalPredictive(draws.reshape((-1,) + draws.shape[len(shape) :]))
    if hasattr(fit, "predict_distribution"):
        return fit.predict_distribution(x, observation=observation)
    if hasattr(fit, "predict_joint"):
        return fit.predict_joint(x, observation=observation)
    if hasattr(fit, "joint_predictive"):
        return fit.joint_predictive(x, observation=observation)
    if hasattr(fit, "predictive"):
        try:
            return fit.predictive(x, observation=observation)
        except (TypeError, ValueError):
            return tuple(fit.predictive(row, observation=observation) for row in x)
    raise TypeError(f"{type(fit).__name__} does not expose a predictive distribution")


def predict(
    fit: Any,
    x: Any,
    *,
    observation: bool = True,
    predictive=None,
    size=2000,
    rng=None,
):
    """Return posterior-predictive means while preserving event structure."""
    distribution = predict_distribution(
        fit,
        x,
        observation=observation,
        predictive=predictive,
        size=size,
        rng=rng,
    )
    if isinstance(distribution, tuple):
        return np.asarray([value.mean_value for value in distribution], dtype=object)
    if hasattr(distribution, "mean"):
        return np.asarray(distribution.mean)
    if hasattr(distribution, "mean_value"):
        return distribution.mean_value
    raise TypeError("predictive distribution does not expose a mean")


def posterior_predictive(
    fit: Any,
    x: Any,
    *,
    size=1000,
    rng=None,
    observation: bool = True,
    predictive=None,
):
    """Draw posterior-predictive samples from exact or sampled inference results."""
    if _sampled_posterior(fit):
        return _predictive_draws(fit, x, predictive, size=size, rng=rng)
    if hasattr(fit, "sample_posterior_predictive"):
        return fit.sample_posterior_predictive(
            x, size=size, rng=rng, observation=observation
        )
    distribution = predict_distribution(fit, x, observation=observation)
    if isinstance(distribution, tuple):
        draws = [np.asarray(value.sample(size=size, rng=rng)) for value in distribution]
        return np.stack(draws, axis=-1)
    if hasattr(distribution, "sample"):
        return distribution.sample(size=size, rng=rng)
    raise TypeError("predictive distribution does not support sampling")


def predictive_interval(
    fit: Any,
    x: Any,
    *,
    level: float = 0.95,
    observation: bool = True,
    predictive=None,
    size=4000,
    rng=None,
):
    """Return equal-tailed posterior-predictive intervals."""
    if not 0 < level < 1:
        raise ValueError("level must lie strictly between 0 and 1")
    alpha = (1 - level) / 2
    distribution = predict_distribution(
        fit,
        x,
        observation=observation,
        predictive=predictive,
        size=size,
        rng=rng,
    )
    if isinstance(distribution, EmpiricalPredictive):
        bounds = distribution.quantile([alpha, 1 - alpha])
        return bounds[0], bounds[1]

    def interval(value):
        return value.quantile(alpha), value.quantile(1 - alpha)

    if isinstance(distribution, tuple):
        return tuple(interval(value) for value in distribution)
    if hasattr(distribution, "marginal") and hasattr(distribution, "mean"):
        return tuple(
            interval(distribution.marginal(index))
            for index in range(len(distribution.mean))
        )
    if hasattr(distribution, "row_marginal"):
        rows = []
        for row_index in range(distribution.rows):
            row = distribution.row_marginal(row_index)
            rows.append(
                tuple(interval(row.marginal(index)) for index in range(row.dimension))
            )
        return tuple(rows)
    return interval(distribution)


__all__ = [
    "EmpiricalPredictive",
    "posterior_predictive",
    "predict",
    "predict_distribution",
    "predictive_interval",
]
