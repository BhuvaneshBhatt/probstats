"""Common result presentation and conversion utilities."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np


class ResultMixin:
    """Consistent conversion and human-readable summaries for result objects."""

    def to_dict(self) -> dict[str, Any]:
        if is_dataclass(self):
            out: dict[str, Any] = {}
            for field in fields(self):
                value = getattr(self, field.name)
                out[field.name] = (
                    value.copy() if isinstance(value, np.ndarray) else value
                )
            return out
        return dict(vars(self))

    def summary_data(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in self.to_dict().items()
            if value is None or isinstance(value, (str, bool, int, float, np.number))
        }

    def summary(self) -> str:
        data = self.summary_data()
        title = type(self).__name__
        if not data:
            return title
        width = max(map(len, data))
        lines = [title]
        lines.extend(f"{key:<{width}} : {value}" for key, value in data.items())
        return "\n".join(lines)

    def to_frame(self):
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "to_frame() requires pandas; install probstats[tabular]"
            ) from exc
        return pd.DataFrame([self.summary_data()])


class StatisticalResult(ResultMixin):
    """Base class for structured statistical results."""


class HypothesisResult(StatisticalResult):
    """Base class for hypothesis-test results."""


class ModelFitResult(StatisticalResult):
    """Base class for fitted-model results."""


class BayesianResult(StatisticalResult):
    """Base class for Bayesian inference and comparison results."""


def coefficient_frame(names, coefficients, standard_errors, statistics, pvalues):
    """Return a labeled coefficient table using pandas."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "coefficient tables require pandas; install probstats[tabular]"
        ) from exc
    return pd.DataFrame(
        {
            "estimate": np.asarray(coefficients),
            "standard_error": np.asarray(standard_errors),
            "statistic": np.asarray(statistics),
            "pvalue": np.asarray(pvalues),
        },
        index=list(names),
    )


__all__ = [
    "BayesianResult",
    "HypothesisResult",
    "ModelFitResult",
    "ResultMixin",
    "StatisticalResult",
    "coefficient_frame",
]
