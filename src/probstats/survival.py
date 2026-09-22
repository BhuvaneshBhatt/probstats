"""Nonparametric survival analysis for right-censored and left-truncated data."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._special import chi2_sf, normal_ppf
from .results import StatisticalResult
from .testing import HypothesisTestResult


def _readonly(values, *, dtype=float):
    array = np.asarray(values, dtype=dtype).reshape(-1).copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class SurvivalData:
    """Observed survival times with censoring and optional delayed entry.

    Parameters
    ----------
    time:
        Exit times (event or right-censoring times).
    event:
        ``True``/1 for an observed event and ``False``/0 for right censoring.
        If omitted, every observation is treated as an event.
    entry:
        Optional delayed-entry (left-truncation) times. An individual is at
        risk at time ``t`` when ``entry <= t <= time``.
    """

    time: np.ndarray
    event: np.ndarray | None = None
    entry: np.ndarray | None = None

    def __post_init__(self):
        time = np.asarray(self.time, dtype=float).reshape(-1)
        if time.size == 0 or not np.all(np.isfinite(time)):
            raise ValueError("time must contain finite observations")
        if np.any(time < 0):
            raise ValueError("survival times must be nonnegative")

        if self.event is None:
            event = np.ones(time.size, dtype=bool)
        else:
            raw_event = np.asarray(self.event).reshape(-1)
            if raw_event.size != time.size:
                raise ValueError("event must have the same length as time")
            if not np.all(np.isin(raw_event, [0, 1, False, True])):
                raise ValueError("event indicators must be boolean or 0/1")
            event = raw_event.astype(bool)

        if self.entry is None:
            entry = np.zeros(time.size, dtype=float)
        else:
            entry = np.asarray(self.entry, dtype=float).reshape(-1)
            if entry.size != time.size:
                raise ValueError("entry must have the same length as time")
            if not np.all(np.isfinite(entry)) or np.any(entry < 0):
                raise ValueError("entry times must be finite and nonnegative")
            if np.any(entry > time):
                raise ValueError("entry times cannot exceed exit times")

        object.__setattr__(self, "time", _readonly(time))
        object.__setattr__(self, "event", _readonly(event, dtype=bool))
        object.__setattr__(self, "entry", _readonly(entry))

    def __len__(self):
        return self.time.size

    @property
    def n(self):
        return len(self)

    @property
    def events(self):
        return int(np.sum(self.event))

    @property
    def censored(self):
        return self.n - self.events

    @classmethod
    def from_event_censor_times(cls, event_times=(), censor_times=()):
        event_times = np.asarray(event_times, dtype=float).reshape(-1)
        censor_times = np.asarray(censor_times, dtype=float).reshape(-1)
        time = np.concatenate([event_times, censor_times])
        event = np.concatenate(
            [
                np.ones(event_times.size, dtype=bool),
                np.zeros(censor_times.size, dtype=bool),
            ]
        )
        return cls(time, event)


@dataclass(frozen=True)
class RiskTable:
    """Counts at each observed exit time."""

    time: np.ndarray
    at_risk: np.ndarray
    events: np.ndarray
    censored: np.ndarray
    entered: np.ndarray

    def __post_init__(self):
        for name in ("time", "at_risk", "events", "censored", "entered"):
            dtype = float if name == "time" else int
            object.__setattr__(self, name, _readonly(getattr(self, name), dtype=dtype))

    def __len__(self):
        return self.time.size


def risk_table(data: SurvivalData | Any, event=None, *, entry=None):
    """Return event, censoring, entry, and risk-set counts by exit time."""
    data = _coerce_survival_data(data, event, entry=entry)
    times = np.unique(data.time)
    at_risk = np.array(
        [np.sum((data.entry <= t) & (data.time >= t)) for t in times], dtype=int
    )
    events = np.array([np.sum((data.time == t) & data.event) for t in times], dtype=int)
    censored = np.array(
        [np.sum((data.time == t) & ~data.event) for t in times], dtype=int
    )
    entered = np.array([np.sum(data.entry == t) for t in times], dtype=int)
    return RiskTable(times, at_risk, events, censored, entered)


def _coerce_survival_data(data, event=None, *, entry=None):
    if isinstance(data, SurvivalData):
        if event is not None or entry is not None:
            raise TypeError("event and entry are not used with SurvivalData")
        return data
    return SurvivalData(data, event, entry)


def _step_value(times, values, query, initial):
    q = np.asarray(query, dtype=float)
    indices = np.searchsorted(times, q, side="right") - 1
    result = np.full(q.shape, initial, dtype=float)
    mask = indices >= 0
    result[mask] = values[indices[mask]]
    if result.ndim == 0:
        return float(result)
    return result


@dataclass(frozen=True)
class KaplanMeierResult(StatisticalResult):
    data: SurvivalData
    table: RiskTable
    event_times: np.ndarray
    survival: np.ndarray
    greenwood_sum: np.ndarray
    variance: np.ndarray
    standard_error: np.ndarray
    confidence_level: float
    lower: np.ndarray
    upper: np.ndarray

    @property
    def median_survival(self):
        indices = np.flatnonzero(self.survival <= 0.5)
        return math.inf if indices.size == 0 else float(self.event_times[indices[0]])

    @property
    def median(self):
        return self.median_survival

    def survival_at(self, time):
        return _step_value(self.event_times, self.survival, time, 1.0)

    def variance_at(self, time):
        return _step_value(self.event_times, self.variance, time, 0.0)

    def confidence_interval_at(self, time):
        return (
            _step_value(self.event_times, self.lower, time, 1.0),
            _step_value(self.event_times, self.upper, time, 1.0),
        )


@dataclass(frozen=True)
class NelsonAalenResult(StatisticalResult):
    data: SurvivalData
    table: RiskTable
    event_times: np.ndarray
    cumulative_hazard: np.ndarray
    variance: np.ndarray
    standard_error: np.ndarray
    confidence_level: float
    lower: np.ndarray
    upper: np.ndarray

    def cumulative_hazard_at(self, time):
        return _step_value(self.event_times, self.cumulative_hazard, time, 0.0)

    def survival_at(self, time):
        return np.exp(-np.asarray(self.cumulative_hazard_at(time)))

    def confidence_interval_at(self, time):
        return (
            _step_value(self.event_times, self.lower, time, 0.0),
            _step_value(self.event_times, self.upper, time, 0.0),
        )


def _validate_confidence(confidence_level):
    confidence_level = float(confidence_level)
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie in (0, 1)")
    return confidence_level


def kaplan_meier(data, event=None, *, entry=None, confidence_level=0.95):
    """Kaplan-Meier product-limit estimator with Greenwood uncertainty.

    Confidence intervals use the standard log-minus-log transform when the
    estimated survival lies strictly between zero and one. This keeps bounds in
    ``[0, 1]`` and uses Greenwood's variance estimate.
    """
    data = _coerce_survival_data(data, event, entry=entry)
    confidence_level = _validate_confidence(confidence_level)
    table = risk_table(data)
    event_mask = table.events > 0
    times = table.time[event_mask]
    n = table.at_risk[event_mask].astype(float)
    d = table.events[event_mask].astype(float)
    if np.any(d > n):
        raise ValueError("event count cannot exceed the risk set")

    factors = 1 - d / n
    survival = np.cumprod(factors)
    terms = np.zeros_like(d)
    regular = n > d
    terms[regular] = d[regular] / (n[regular] * (n[regular] - d[regular]))
    terms[~regular] = math.inf
    greenwood_sum = np.cumsum(terms)
    variance = np.zeros_like(survival)
    positive = survival > 0
    variance[positive] = survival[positive] ** 2 * greenwood_sum[positive]
    standard_error = np.sqrt(variance)

    alpha = 1 - confidence_level
    z = normal_ppf(1 - alpha / 2)
    lower = np.empty_like(survival)
    upper = np.empty_like(survival)
    for i, (s, g) in enumerate(zip(survival, greenwood_sum, strict=True)):
        if s <= 0:
            lower[i] = upper[i] = 0.0
        elif s >= 1 or g <= 0:
            lower[i] = upper[i] = s
        else:
            se_loglog = math.sqrt(g) / abs(math.log(s))
            log_minus_log = math.log(-math.log(s))
            lower[i] = math.exp(-math.exp(log_minus_log + z * se_loglog))
            upper[i] = math.exp(-math.exp(log_minus_log - z * se_loglog))

    return KaplanMeierResult(
        data,
        table,
        _readonly(times),
        _readonly(survival),
        _readonly(greenwood_sum),
        _readonly(variance),
        _readonly(standard_error),
        confidence_level,
        _readonly(lower),
        _readonly(upper),
    )


def nelson_aalen(data, event=None, *, entry=None, confidence_level=0.95):
    """Nelson-Aalen cumulative-hazard estimator."""
    data = _coerce_survival_data(data, event, entry=entry)
    confidence_level = _validate_confidence(confidence_level)
    table = risk_table(data)
    event_mask = table.events > 0
    times = table.time[event_mask]
    n = table.at_risk[event_mask].astype(float)
    d = table.events[event_mask].astype(float)
    increments = d / n
    cumulative = np.cumsum(increments)
    # Aalen variance for grouped/tied failures under the counting-process form.
    variance = np.cumsum(d / n**2)
    standard_error = np.sqrt(variance)
    z = normal_ppf(0.5 + confidence_level / 2)
    lower = np.maximum(0.0, cumulative - z * standard_error)
    upper = cumulative + z * standard_error
    return NelsonAalenResult(
        data,
        table,
        _readonly(times),
        _readonly(cumulative),
        _readonly(variance),
        _readonly(standard_error),
        confidence_level,
        _readonly(lower),
        _readonly(upper),
    )


def median_survival(data, event=None, *, entry=None):
    """Return the Kaplan-Meier median, or ``inf`` if survival never reaches 1/2."""
    return kaplan_meier(data, event, entry=entry).median_survival


def _pooled_previous_survival(total_risk, total_events):
    previous = np.empty(total_events.size, dtype=float)
    survival = 1.0
    for i, (n, d) in enumerate(zip(total_risk, total_events, strict=True)):
        previous[i] = survival
        if n > 0:
            survival *= 1 - d / n
    return previous


def _logrank_weight(method, total_risk, pooled_survival, *, p, q):
    key = method.lower().replace("_", "-")
    if key in {"logrank", "log-rank", "mantel-haenszel"}:
        return np.ones_like(total_risk, dtype=float), "log-rank test"
    if key in {"wilcoxon", "breslow", "gehan-breslow", "generalized-wilcoxon"}:
        return total_risk.astype(float), "Breslow generalized Wilcoxon test"
    if key in {"tarone-ware", "taroneware"}:
        return np.sqrt(total_risk), "Tarone-Ware test"
    if key in {"fleming-harrington", "flemingharrington", "fh"}:
        if p < 0 or q < 0:
            raise ValueError("Fleming-Harrington p and q must be nonnegative")
        weights = pooled_survival**p * (1 - pooled_survival) ** q
        return weights, f"Fleming-Harrington({p:g}, {q:g}) test"
    raise ValueError(
        "method must be 'logrank', 'breslow', 'tarone-ware', or 'fleming-harrington'"
    )


def logrank_test(
    *samples,
    groups=None,
    event=None,
    entry=None,
    method="logrank",
    p=0.0,
    q=0.0,
):
    """Compare two or more survival curves with a weighted log-rank test.

    Supply either two or more :class:`SurvivalData` objects as positional
    arguments, or one survival sample plus a ``groups`` vector. ``method`` may
    be ``"logrank"``, ``"breslow"``, ``"tarone-ware"``, or
    ``"fleming-harrington"``. The latter uses weights
    ``S(t-)**p * (1-S(t-))**q`` from the pooled Kaplan-Meier curve.
    """
    if groups is not None:
        if len(samples) != 1:
            raise TypeError("groups= requires exactly one survival sample")
        data = _coerce_survival_data(samples[0], event, entry=entry)
        labels = np.asarray(groups).reshape(-1)
        if labels.size != data.n:
            raise ValueError("groups must have the same length as the survival data")
        unique = np.unique(labels)
        if unique.size < 2:
            raise ValueError("at least two groups are required")
        datasets = tuple(
            SurvivalData(
                data.time[labels == label],
                data.event[labels == label],
                data.entry[labels == label],
            )
            for label in unique
        )
    else:
        if event is not None or entry is not None:
            raise TypeError("event and entry are only supported with groups=")
        if len(samples) < 2:
            raise ValueError("at least two survival samples are required")
        datasets = tuple(_coerce_survival_data(sample) for sample in samples)

    event_times = (
        np.unique(
            np.concatenate([d.time[d.event] for d in datasets if np.any(d.event)])
        )
        if any(np.any(d.event) for d in datasets)
        else np.array([], dtype=float)
    )
    if event_times.size == 0:
        return HypothesisTestResult(
            0.0,
            1.0,
            "two-sided",
            "log-rank test" if method == "logrank" else str(method),
            tuple(d.n for d in datasets),
            0.0,
            len(datasets) - 1,
        )

    risk = np.array(
        [
            [np.sum((d.entry <= t) & (d.time >= t)) for d in datasets]
            for t in event_times
        ],
        dtype=float,
    )
    events_by_group = np.array(
        [[np.sum((d.time == t) & d.event) for d in datasets] for t in event_times],
        dtype=float,
    )
    total_risk = risk.sum(axis=1)
    total_events = events_by_group.sum(axis=1)
    pooled_previous = _pooled_previous_survival(total_risk, total_events)
    weights, method_name = _logrank_weight(
        method, total_risk, pooled_previous, p=float(p), q=float(q)
    )

    k = len(datasets)
    score = np.zeros(k, dtype=float)
    covariance = np.zeros((k, k), dtype=float)
    for n_groups, d_groups, n, d, weight in zip(
        risk, events_by_group, total_risk, total_events, weights, strict=True
    ):
        if n <= 0 or d <= 0:
            continue
        proportions = n_groups / n
        score += weight * (d_groups - proportions * d)
        if n > 1:
            hypergeom_scale = d * (n - d) / (n - 1)
            covariance += (
                weight**2
                * hypergeom_scale
                * (np.diag(proportions) - np.outer(proportions, proportions))
            )

    # The full covariance is singular because scores sum to zero. Drop the
    # final group to obtain an equivalent (k-1)-dimensional contrast system.
    reduced_score = score[:-1]
    reduced_covariance = covariance[:-1, :-1]
    if np.allclose(reduced_covariance, 0):
        statistic = 0.0 if np.allclose(reduced_score, 0) else math.inf
    else:
        statistic = float(
            reduced_score @ np.linalg.pinv(reduced_covariance) @ reduced_score
        )
        statistic = max(0.0, statistic)
    df = k - 1
    pvalue = 0.0 if math.isinf(statistic) else chi2_sf(statistic, df)
    return HypothesisTestResult(
        statistic,
        pvalue,
        "two-sided",
        method_name,
        tuple(d.n for d in datasets),
        0.0,
        df,
    )


def breslow_test(*samples, **kwargs):
    """Breslow/generalized-Wilcoxon survival-curve test."""
    return logrank_test(*samples, method="breslow", **kwargs)


def tarone_ware_test(*samples, **kwargs):
    """Tarone-Ware survival-curve test."""
    return logrank_test(*samples, method="tarone-ware", **kwargs)


def fleming_harrington_test(*samples, p=0.0, q=0.0, **kwargs):
    """Fleming-Harrington weighted log-rank test."""
    return logrank_test(*samples, method="fleming-harrington", p=p, q=q, **kwargs)


__all__ = [
    "KaplanMeierResult",
    "NelsonAalenResult",
    "RiskTable",
    "SurvivalData",
    "breslow_test",
    "fleming_harrington_test",
    "kaplan_meier",
    "logrank_test",
    "median_survival",
    "nelson_aalen",
    "risk_table",
    "tarone_ware_test",
]
