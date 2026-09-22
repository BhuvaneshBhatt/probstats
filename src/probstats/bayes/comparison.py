"""Bayesian model comparison from marginal likelihoods."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..results import BayesianResult


def _log_evidence(model: Any) -> float:
    value = getattr(model, "log_evidence", None)
    if value is None and hasattr(model, "to_inference_result"):
        value = model.to_inference_result().log_evidence
    if value is None:
        raise ValueError(f"{type(model).__name__} does not provide log evidence")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(value.evalf())


@dataclass(frozen=True, slots=True)
class BayesianModelComparisonResult(BayesianResult):
    names: tuple[str, ...]
    log_evidence: np.ndarray
    posterior_probabilities: np.ndarray
    log_bayes_factors: np.ndarray

    @property
    def best_model(self) -> str:
        return self.names[int(np.argmax(self.posterior_probabilities))]

    def _model_index(self, model: str | int) -> int:
        if isinstance(model, str):
            try:
                return self.names.index(model)
            except ValueError as exc:
                raise KeyError(f"unknown model name: {model!r}") from exc
        index = int(model)
        if index != model or not 0 <= index < len(self.names):
            raise IndexError("model index out of range")
        return index

    def log_bayes_factor(self, numerator: str | int, denominator: str | int) -> float:
        """Return log BF for ``numerator`` relative to ``denominator``."""
        i = self._model_index(numerator)
        j = self._model_index(denominator)
        return float(self.log_bayes_factors[i, j])

    def bayes_factor(self, numerator: str | int, denominator: str | int) -> float:
        """Return BF for ``numerator`` relative to ``denominator``."""
        return float(np.exp(self.log_bayes_factor(numerator, denominator)))

    def posterior_probability(self, model: str | int) -> float:
        """Return the posterior probability assigned to one model."""
        return float(self.posterior_probabilities[self._model_index(model)])

    def summary_data(self) -> dict[str, Any]:
        return {"best_model": self.best_model, "n_models": len(self.names)}

    def to_frame(self):
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "to_frame() requires pandas; install probstats[tabular]"
            ) from exc
        return pd.DataFrame(
            {
                "model": self.names,
                "log_evidence": self.log_evidence,
                "posterior_probability": self.posterior_probabilities,
            }
        ).set_index("model")


def compare_models(
    models: Sequence[Any],
    *,
    names: Sequence[str] | None = None,
    prior_probabilities=None,
) -> BayesianModelComparisonResult:
    """Compare fitted Bayesian models using marginal likelihoods and posterior model probabilities."""
    models = tuple(models)
    if len(models) < 2:
        raise ValueError("at least two models are required")
    labels = (
        tuple(names)
        if names is not None
        else tuple(f"model_{index + 1}" for index in range(len(models)))
    )
    if len(labels) != len(models) or len(set(labels)) != len(labels):
        raise ValueError("names must contain one unique name per model")
    logs = np.asarray([_log_evidence(model) for model in models], dtype=float)
    if not np.all(np.isfinite(logs)):
        raise ValueError("log evidence values must be finite")
    if prior_probabilities is None:
        priors = np.full(len(models), 1 / len(models))
    else:
        priors = np.asarray(prior_probabilities, dtype=float).reshape(-1)
        if (
            priors.shape != (len(models),)
            or np.any(priors <= 0)
            or not np.all(np.isfinite(priors))
        ):
            raise ValueError(
                "prior_probabilities must contain one positive finite value per model"
            )
        priors /= priors.sum()
    scores = logs + np.log(priors)
    weights = np.exp(scores - scores.max())
    weights /= weights.sum()
    factors = logs[:, None] - logs[None, :]
    for array in (logs, weights, factors):
        array.setflags(write=False)
    return BayesianModelComparisonResult(labels, logs, weights, factors)


__all__ = ["BayesianModelComparisonResult", "compare_models"]
