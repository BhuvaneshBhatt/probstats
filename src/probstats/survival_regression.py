"""Survival regression: Cox proportional hazards and parametric AFT models."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ._special import chi2_sf, normal_cdf, normal_ppf
from ._tabular import array_metadata
from .results import HypothesisResult, ModelFitResult, coefficient_frame
from .survival import SurvivalData, _coerce_survival_data, _readonly, kaplan_meier


def _design_matrix(covariates, n):
    x = np.asarray(covariates, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[0] != n:
        raise ValueError("covariates must have shape (n_observations, n_features)")
    if x.shape[1] == 0 or not np.all(np.isfinite(x)):
        raise ValueError(
            "covariates must contain finite values and at least one feature"
        )
    return x


def _names(names, p):
    if names is None:
        return tuple(f"x{i}" for i in range(p))
    names = tuple(str(name) for name in names)
    if len(names) != p or len(set(names)) != p:
        raise ValueError("covariate_names must contain one unique name per feature")
    return names


def _safe_inverse(matrix):
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(matrix)


def _quadratic_form_pvalue(vector, covariance):
    if vector.size == 0:
        return 0.0, 1.0
    statistic = float(vector @ np.linalg.pinv(covariance) @ vector)
    statistic = max(0.0, statistic)
    return statistic, chi2_sf(statistic, vector.size)


def _cox_terms(beta, data, x, ties):
    eta = x @ beta
    # The shift cancels analytically in the partial likelihood but avoids overflow.
    shift = float(np.max(eta))
    risk_weight = np.exp(eta - shift)
    score = np.zeros(x.shape[1])
    information = np.zeros((x.shape[1], x.shape[1]))
    loglik = 0.0

    event_times = np.unique(data.time[data.event])
    for t in event_times:
        event_mask = (data.time == t) & data.event
        risk_mask = (data.entry <= t) & (data.time >= t)
        d = int(np.sum(event_mask))
        xr = x[risk_mask]
        wr = risk_weight[risk_mask]
        xe = x[event_mask]
        we = risk_weight[event_mask]
        s0 = float(np.sum(wr))
        s1 = np.sum(wr[:, None] * xr, axis=0)
        s2 = np.einsum("i,ij,ik->jk", wr, xr, xr)
        observed = np.sum(xe, axis=0)
        loglik += float(np.sum(eta[event_mask]))
        score += observed

        if ties == "breslow" or d == 1:
            loglik -= d * (math.log(s0) + shift)
            mean = s1 / s0
            score -= d * mean
            information += d * (s2 / s0 - np.outer(mean, mean))
            continue

        e0 = float(np.sum(we))
        e1 = np.sum(we[:, None] * xe, axis=0)
        e2 = np.einsum("i,ij,ik->jk", we, xe, xe)
        for tie_index in range(d):
            fraction = tie_index / d
            denominator = s0 - fraction * e0
            first = s1 - fraction * e1
            second = s2 - fraction * e2
            mean = first / denominator
            loglik -= math.log(denominator) + shift
            score -= mean
            information += second / denominator - np.outer(mean, mean)
    return loglik, score, information


def _fit_cox(beta, data, x, ties, max_iter, tol):
    beta = np.asarray(beta, dtype=float).copy()
    loglik, score, information = _cox_terms(beta, data, x, ties)
    for iteration in range(1, max_iter + 1):
        if np.linalg.norm(score, np.inf) <= tol:
            return beta, loglik, information, True, iteration
        step = np.linalg.pinv(information) @ score
        scale = 1.0
        accepted = False
        while scale >= 2**-20:
            candidate = beta + scale * step
            candidate_loglik, candidate_score, candidate_information = _cox_terms(
                candidate, data, x, ties
            )
            if math.isfinite(candidate_loglik) and candidate_loglik >= loglik:
                beta = candidate
                loglik = candidate_loglik
                score = candidate_score
                information = candidate_information
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            return beta, loglik, information, False, iteration
        if np.linalg.norm(scale * step, np.inf) <= tol * (
            1 + np.linalg.norm(beta, np.inf)
        ):
            return beta, loglik, information, True, iteration
    return beta, loglik, information, False, max_iter


def _baseline_hazard(data, x, beta):
    eta = x @ beta
    event_times = np.unique(data.time[data.event])
    increments = []
    for t in event_times:
        risk = (data.entry <= t) & (data.time >= t)
        d = np.sum((data.time == t) & data.event)
        denominator = np.sum(np.exp(eta[risk]))
        increments.append(float(d / denominator))
    increments = np.asarray(increments)
    return event_times, increments, np.cumsum(increments)


def _step(times, values, query, *, before=False):
    q = np.asarray(query, dtype=float)
    side = "left" if before else "right"
    idx = np.searchsorted(times, q, side=side) - 1
    result = np.zeros(q.shape, dtype=float)
    mask = idx >= 0
    result[mask] = values[idx[mask]]
    return float(result) if result.ndim == 0 else result


@dataclass(frozen=True)
class SchoenfeldResiduals:
    time: np.ndarray
    observation: np.ndarray
    residuals: np.ndarray
    scaled: np.ndarray


@dataclass(frozen=True)
class ProportionalHazardsTestResult(HypothesisResult):
    statistic: float
    pvalue: float
    df: int
    covariate_names: tuple[str, ...]
    per_covariate_statistic: np.ndarray
    per_covariate_pvalue: np.ndarray
    time_transform: str


@dataclass(frozen=True)
class CoxPHResult(ModelFitResult):
    data: SurvivalData
    covariates: np.ndarray
    coefficients: np.ndarray
    covariance: np.ndarray
    standard_error: np.ndarray
    z: np.ndarray
    pvalue: np.ndarray
    confidence_level: float
    confidence_interval: np.ndarray
    covariate_names: tuple[str, ...]
    ties: str
    log_partial_likelihood: float
    baseline_times: np.ndarray
    baseline_hazard: np.ndarray
    baseline_cum_hazard: np.ndarray
    baseline_survival: np.ndarray
    converged: bool
    iterations: int

    @property
    def hazard_ratio(self):
        return np.exp(self.coefficients)

    def coefficient_table(self):
        """Return Cox coefficients, hazard ratios, and inference in a DataFrame."""
        table = coefficient_frame(
            self.covariate_names,
            self.coefficients,
            self.standard_error,
            self.z,
            self.pvalue,
        )
        table.insert(1, "hazard_ratio", self.hazard_ratio)
        return table

    def to_frame(self):
        return self.coefficient_table()

    def linear_predictor(self, covariates):
        x = np.asarray(covariates, dtype=float)
        return x @ self.coefficients

    def cumulative_hazard(self, time, covariates=None):
        base = _step(self.baseline_times, self.baseline_cum_hazard, time)
        if covariates is None:
            return base
        lp = np.asarray(self.linear_predictor(covariates))
        return np.asarray(base) * np.exp(lp)

    def survival(self, time, covariates=None):
        return np.exp(-np.asarray(self.cumulative_hazard(time, covariates)))

    def schoenfeld_residuals(self):
        return schoenfeld_residuals(self)

    def martingale_residuals(self):
        return martingale_residuals(self)

    def deviance_residuals(self):
        return deviance_residuals(self)

    def proportional_hazards_test(self, *, time_transform="rank"):
        return proportional_hazards_test(self, time_transform=time_transform)


def cox_partial_log_likelihood(
    data, covariates, coefficients, event=None, *, entry=None, ties="efron"
):
    """Evaluate the Cox partial log likelihood at fixed coefficients."""
    data = _coerce_survival_data(data, event, entry=entry)
    x = _design_matrix(covariates, data.n)
    beta = np.asarray(coefficients, dtype=float).reshape(-1)
    if beta.size != x.shape[1] or not np.all(np.isfinite(beta)):
        raise ValueError("coefficients must contain one finite value per covariate")
    ties = ties.lower()
    if ties not in {"breslow", "efron"}:
        raise ValueError("ties must be 'breslow' or 'efron'")
    return float(_cox_terms(beta, data, x, ties)[0])


def cox_ph(
    data,
    covariates,
    event=None,
    *,
    entry=None,
    ties="efron",
    covariate_names=None,
    confidence_level=0.95,
    initial=None,
    max_iter=100,
    tol=1e-9,
):
    """Fit a Cox proportional-hazards model by maximum partial likelihood."""
    data = _coerce_survival_data(data, event, entry=entry)
    raw_covariates, _, inferred_names, _ = array_metadata(covariates)
    x = _design_matrix(raw_covariates, data.n)
    if covariate_names is None and inferred_names is not None:
        covariate_names = inferred_names
    if data.events == 0:
        raise ValueError("Cox regression requires at least one observed event")
    ties = ties.lower()
    if ties not in {"breslow", "efron"}:
        raise ValueError("ties must be 'breslow' or 'efron'")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie in (0, 1)")
    p = x.shape[1]
    beta0 = (
        np.zeros(p) if initial is None else np.asarray(initial, dtype=float).reshape(-1)
    )
    if beta0.size != p or not np.all(np.isfinite(beta0)):
        raise ValueError("initial must contain one finite coefficient per covariate")
    beta, loglik, information, converged, iterations = _fit_cox(
        beta0, data, x, ties, int(max_iter), float(tol)
    )
    covariance = _safe_inverse(information)
    se = np.sqrt(np.maximum(0.0, np.diag(covariance)))
    z = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    pvalue = np.array([2 * normal_cdf(-abs(value)) for value in z])
    critical = normal_ppf(0.5 + confidence_level / 2)
    intervals = np.column_stack([beta - critical * se, beta + critical * se])
    times, increments, cumulative = _baseline_hazard(data, x, beta)
    return CoxPHResult(
        data=data,
        covariates=_readonly(x.ravel()).reshape(x.shape),
        coefficients=_readonly(beta),
        covariance=np.asarray(covariance),
        standard_error=_readonly(se),
        z=_readonly(z),
        pvalue=_readonly(pvalue),
        confidence_level=float(confidence_level),
        confidence_interval=np.asarray(intervals),
        covariate_names=_names(covariate_names, p),
        ties=ties,
        log_partial_likelihood=float(loglik),
        baseline_times=_readonly(times),
        baseline_hazard=_readonly(increments),
        baseline_cum_hazard=_readonly(cumulative),
        baseline_survival=_readonly(np.exp(-cumulative)),
        converged=bool(converged),
        iterations=int(iterations),
    )


def schoenfeld_residuals(result: CoxPHResult):
    data = result.data
    x = result.covariates
    beta = result.coefficients
    eta = x @ beta
    rows = []
    times = []
    observations = []
    for t in np.unique(data.time[data.event]):
        event_indices = np.flatnonzero((data.time == t) & data.event)
        risk = (data.entry <= t) & (data.time >= t)
        shift = float(np.max(eta[risk]))
        risk_weights = np.exp(eta[risk] - shift)
        xr = x[risk]
        s0 = float(np.sum(risk_weights))
        s1 = np.sum(risk_weights[:, None] * xr, axis=0)
        if result.ties == "efron" and event_indices.size > 1:
            event_weights = np.exp(eta[event_indices] - shift)
            e0 = float(np.sum(event_weights))
            e1 = np.sum(event_weights[:, None] * x[event_indices], axis=0)
            expected = np.zeros(x.shape[1])
            d = event_indices.size
            for tie_index in range(d):
                fraction = tie_index / d
                expected += (s1 - fraction * e1) / (s0 - fraction * e0)
            expected /= d
        else:
            expected = s1 / s0
        for index in event_indices:
            rows.append(x[index] - expected)
            times.append(t)
            observations.append(index)
    residuals = np.asarray(rows, dtype=float)
    scaled = residuals @ result.covariance
    return SchoenfeldResiduals(
        _readonly(times),
        _readonly(observations, dtype=int),
        residuals,
        scaled,
    )


def martingale_residuals(result: CoxPHResult):
    data = result.data
    lp = result.covariates @ result.coefficients
    exit_hazard = _step(result.baseline_times, result.baseline_cum_hazard, data.time)
    entry_hazard = _step(
        result.baseline_times,
        result.baseline_cum_hazard,
        data.entry,
        before=True,
    )
    expected = np.exp(lp) * (exit_hazard - entry_hazard)
    return data.event.astype(float) - expected


def deviance_residuals(result: CoxPHResult):
    martingale = martingale_residuals(result)
    event = result.data.event.astype(float)
    inside = np.empty_like(martingale)
    censored = event == 0
    inside[censored] = -2 * martingale[censored]
    observed = ~censored
    delta_minus_m = np.maximum(
        event[observed] - martingale[observed], np.finfo(float).tiny
    )
    inside[observed] = -2 * (martingale[observed] + np.log(delta_minus_m))
    inside = np.maximum(inside, 0.0)
    return np.sign(martingale) * np.sqrt(inside)


def proportional_hazards_test(result: CoxPHResult, *, time_transform="rank"):
    """Grambsch-Therneau-style diagnostic based on scaled Schoenfeld residuals.

    Each coefficient is tested for a linear association between its scaled
    Schoenfeld residual and transformed event time. The global statistic is the
    sum of the per-coefficient one-degree-of-freedom chi-square statistics.
    """
    residual = schoenfeld_residuals(result)
    n = residual.time.size
    if n < 3:
        raise ValueError(
            "at least three observed events are required for PH diagnostics"
        )
    key = time_transform.lower()
    if key == "rank":
        order = np.argsort(np.argsort(residual.time, kind="stable"), kind="stable")
        transformed = (order + 1).astype(float) / (n + 1)
    elif key == "identity":
        transformed = residual.time.astype(float)
    elif key == "log":
        if np.any(residual.time <= 0):
            raise ValueError("log time transform requires positive event times")
        transformed = np.log(residual.time)
    elif key == "km":
        km = kaplan_meier(result.data)
        transformed = np.asarray(km.survival_at(residual.time), dtype=float)
    else:
        raise ValueError("time_transform must be 'rank', 'identity', 'log', or 'km'")
    centered = transformed - np.mean(transformed)
    denom = float(centered @ centered)
    stats = np.zeros(result.coefficients.size)
    pvalues = np.ones(result.coefficients.size)
    for j in range(result.coefficients.size):
        y = residual.scaled[:, j]
        slope = float(centered @ y / denom)
        fitted = slope * centered
        rss = float(np.sum((y - np.mean(y) - fitted) ** 2))
        sigma2 = rss / max(1, n - 2)
        variance = sigma2 / denom
        stat = 0.0 if variance <= 0 else slope**2 / variance
        stats[j] = stat
        pvalues[j] = chi2_sf(stat, 1)
    return ProportionalHazardsTestResult(
        statistic=float(np.sum(stats)),
        pvalue=chi2_sf(float(np.sum(stats)), result.coefficients.size),
        df=result.coefficients.size,
        covariate_names=result.covariate_names,
        per_covariate_statistic=_readonly(stats),
        per_covariate_pvalue=_readonly(pvalues),
        time_transform=key,
    )


def _normal_logsf(z):
    # erfc is accurate in the ordinary range; guard exact underflow for very large z.
    sf = 0.5 * np.vectorize(math.erfc)(np.asarray(z, dtype=float) / math.sqrt(2))
    return np.log(np.maximum(sf, np.finfo(float).tiny))


def _aft_log_survival_and_density(family, time, eta, ancillary):
    t = np.asarray(time, dtype=float)
    if np.any(t <= 0):
        raise ValueError(
            "parametric survival regression requires strictly positive exit times"
        )
    logt = np.log(t)
    if family == "exponential":
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            scaled = t * np.exp(-eta)
        return -scaled, -eta - scaled
    if family == "weibull":
        shape = math.exp(ancillary)
        log_scaled = shape * (logt - eta)
        scaled = np.exp(np.clip(log_scaled, -745, 709))
        log_s = -scaled
        log_f = math.log(shape) - eta + (shape - 1) * (logt - eta) - scaled
        return log_s, log_f
    if family == "lognormal":
        sigma = math.exp(ancillary)
        z = (logt - eta) / sigma
        log_s = _normal_logsf(z)
        log_f = -logt - math.log(sigma) - 0.5 * math.log(2 * math.pi) - 0.5 * z**2
        return log_s, log_f
    if family == "loglogistic":
        shape = math.exp(ancillary)
        a = shape * (logt - eta)
        # log(1 + exp(a)) stably.
        logden = np.logaddexp(0.0, a)
        log_s = -logden
        log_f = math.log(shape) - eta + (shape - 1) * (logt - eta) - 2 * logden
        return log_s, log_f
    raise ValueError("unknown parametric survival family")


def _aft_loglik(parameters, family, data, x):
    parameters = np.asarray(parameters, dtype=float)
    if not np.all(np.isfinite(parameters)) or np.any(np.abs(parameters) > 50):
        return -math.inf
    p = x.shape[1]
    beta = parameters[:p]
    ancillary = 0.0 if family == "exponential" else float(parameters[p])
    eta = x @ beta
    log_s, log_f = _aft_log_survival_and_density(family, data.time, eta, ancillary)
    contribution = np.where(data.event, log_f, log_s)
    if not np.all(np.isfinite(contribution)):
        return -math.inf
    limit = np.finfo(float).max / max(1, contribution.size)
    if np.any(np.abs(contribution) > limit):
        return -math.inf
    positive_entry = data.entry > 0
    if np.any(positive_entry):
        entry_s, _ = _aft_log_survival_and_density(
            family, data.entry[positive_entry], eta[positive_entry], ancillary
        )
        contribution[positive_entry] -= entry_s
        if not np.all(np.isfinite(contribution)) or np.any(
            np.abs(contribution) > limit
        ):
            return -math.inf
    return float(np.sum(contribution))


def _numeric_gradient(func, x):
    out = np.empty_like(x)
    for i in range(x.size):
        h = 1e-5 * max(1.0, abs(x[i]))
        step = np.zeros_like(x)
        step[i] = h
        out[i] = (func(x + step) - func(x - step)) / (2 * h)
    return out


def _numeric_hessian(func, x):
    n = x.size
    hessian = np.empty((n, n))
    fx = func(x)
    hs = np.array([1e-4 * max(1.0, abs(v)) for v in x])
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = hs[i]
        hessian[i, i] = (func(x + ei) - 2 * fx + func(x - ei)) / hs[i] ** 2
        for j in range(i):
            ej = np.zeros(n)
            ej[j] = hs[j]
            value = (
                func(x + ei + ej)
                - func(x + ei - ej)
                - func(x - ei + ej)
                + func(x - ei - ej)
            ) / (4 * hs[i] * hs[j])
            hessian[i, j] = hessian[j, i] = value
    return hessian


def _bfgs_maximize(func, initial, max_iter, tol):
    x = np.asarray(initial, dtype=float).copy()
    inv_negative_hessian = np.eye(x.size)
    fx = func(x)
    for iteration in range(1, max_iter + 1):
        grad = _numeric_gradient(func, x)
        if np.linalg.norm(grad, np.inf) < tol:
            return x, fx, True, iteration
        direction = inv_negative_hessian @ grad
        if grad @ direction <= 0:
            direction = grad
            inv_negative_hessian = np.eye(x.size)
        scale = 1.0
        while scale >= 2**-24:
            candidate = x + scale * direction
            candidate_value = func(candidate)
            if math.isfinite(
                candidate_value
            ) and candidate_value >= fx + 1e-4 * scale * float(grad @ direction):
                break
            scale *= 0.5
        else:
            return x, fx, False, iteration
        new_grad = _numeric_gradient(func, candidate)
        s = candidate - x
        # BFGS on the negative log likelihood: y = grad(-l)_new-grad(-l)_old.
        y = grad - new_grad
        ys = float(y @ s)
        if ys > 1e-12:
            rho = 1 / ys
            eye = np.eye(x.size)
            inv_negative_hessian = (
                eye - rho * np.outer(s, y)
            ) @ inv_negative_hessian @ (eye - rho * np.outer(y, s)) + rho * np.outer(
                s, s
            )
        if np.linalg.norm(s, np.inf) <= tol * (1 + np.linalg.norm(x, np.inf)):
            return candidate, candidate_value, True, iteration
        x, fx = candidate, candidate_value
    return x, fx, False, max_iter


@dataclass(frozen=True)
class ParametricSurvivalResult(ModelFitResult):
    family: str
    data: SurvivalData
    covariates: np.ndarray
    coefficients: np.ndarray
    ancillary: float | None
    covariance: np.ndarray
    standard_error: np.ndarray
    z: np.ndarray
    pvalue: np.ndarray
    confidence_interval: np.ndarray
    covariate_names: tuple[str, ...]
    log_likelihood: float
    converged: bool
    iterations: int

    @property
    def parameter_names(self):
        if self.family == "exponential":
            return self.covariate_names
        name = "log_sigma" if self.family == "lognormal" else "log_shape"
        return (*self.covariate_names, name)

    @property
    def parameters(self):
        if self.ancillary is None:
            return self.coefficients.copy()
        raw = math.log(self.ancillary)
        return np.concatenate([self.coefficients, [raw]])

    def linear_predictor(self, covariates):
        return np.asarray(covariates, dtype=float) @ self.coefficients

    def survival(self, time, covariates):
        eta = np.asarray(self.linear_predictor(covariates), dtype=float)
        ancillary_raw = 0.0 if self.ancillary is None else math.log(self.ancillary)
        log_s, _ = _aft_log_survival_and_density(self.family, time, eta, ancillary_raw)
        return np.exp(log_s)

    def cumulative_hazard(self, time, covariates):
        return -np.log(self.survival(time, covariates))

    def median(self, covariates):
        eta = np.asarray(self.linear_predictor(covariates), dtype=float)
        if self.family in {"weibull", "loglogistic"}:
            if self.family == "weibull":
                return np.exp(eta) * math.log(2) ** (1 / self.ancillary)
            return np.exp(eta)
        if self.family == "lognormal":
            return np.exp(eta)
        return np.exp(eta) * math.log(2)


def parametric_survival_regression(
    data,
    covariates,
    family,
    event=None,
    *,
    entry=None,
    covariate_names=None,
    confidence_level=0.95,
    initial=None,
    max_iter=500,
    tol=1e-8,
):
    """Fit an exponential, Weibull, lognormal, or loglogistic AFT model."""
    data = _coerce_survival_data(data, event, entry=entry)
    if np.any(data.time <= 0):
        raise ValueError("parametric survival regression requires positive exit times")
    x = _design_matrix(covariates, data.n)
    family = family.lower()
    if family not in {"exponential", "weibull", "lognormal", "loglogistic"}:
        raise ValueError(
            "family must be exponential, weibull, lognormal, or loglogistic"
        )
    p = x.shape[1]
    size = p if family == "exponential" else p + 1
    if initial is None:
        beta = np.linalg.lstsq(x, np.log(data.time), rcond=None)[0]
        initial = beta if family == "exponential" else np.concatenate([beta, [0.0]])
    else:
        initial = np.asarray(initial, dtype=float).reshape(-1)
    if initial.size != size or not np.all(np.isfinite(initial)):
        raise ValueError(f"initial must contain {size} finite parameters")

    def objective(theta):
        return _aft_loglik(theta, family, data, x)

    estimates, loglik, converged, iterations = _bfgs_maximize(
        objective, initial, int(max_iter), float(tol)
    )
    information = -_numeric_hessian(objective, estimates)
    covariance = _safe_inverse(information)
    se = np.sqrt(np.maximum(0.0, np.diag(covariance)))
    z = np.divide(estimates, se, out=np.zeros_like(estimates), where=se > 0)
    pvalue = np.array([2 * normal_cdf(-abs(value)) for value in z])
    critical = normal_ppf(0.5 + confidence_level / 2)
    intervals = np.column_stack([estimates - critical * se, estimates + critical * se])
    ancillary = None if family == "exponential" else math.exp(estimates[p])
    return ParametricSurvivalResult(
        family=family,
        data=data,
        covariates=np.asarray(x),
        coefficients=_readonly(estimates[:p]),
        ancillary=ancillary,
        covariance=np.asarray(covariance),
        standard_error=_readonly(se),
        z=_readonly(z),
        pvalue=_readonly(pvalue),
        confidence_interval=np.asarray(intervals),
        covariate_names=_names(covariate_names, p),
        log_likelihood=float(loglik),
        converged=bool(converged),
        iterations=int(iterations),
    )


def exponential_regression(data, covariates, event=None, **kwargs):
    return parametric_survival_regression(
        data, covariates, "exponential", event, **kwargs
    )


def weibull_regression(data, covariates, event=None, **kwargs):
    return parametric_survival_regression(data, covariates, "weibull", event, **kwargs)


def lognormal_regression(data, covariates, event=None, **kwargs):
    return parametric_survival_regression(
        data, covariates, "lognormal", event, **kwargs
    )


def loglogistic_regression(data, covariates, event=None, **kwargs):
    return parametric_survival_regression(
        data, covariates, "loglogistic", event, **kwargs
    )


__all__ = [
    "CoxPHResult",
    "ParametricSurvivalResult",
    "ProportionalHazardsTestResult",
    "SchoenfeldResiduals",
    "cox_partial_log_likelihood",
    "cox_ph",
    "deviance_residuals",
    "exponential_regression",
    "loglogistic_regression",
    "lognormal_regression",
    "martingale_residuals",
    "parametric_survival_regression",
    "proportional_hazards_test",
    "schoenfeld_residuals",
    "weibull_regression",
]
