"""Diagnostics for MCMC chains."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..diagnostics.core import effective_sample_size, mcse_mean, split_rhat
from .core import MCMCChain


def _stack(chains: Sequence[MCMCChain]) -> np.ndarray:
    if not chains:
        raise ValueError("At least one chain is required")
    draws = min(c.draws for c in chains)
    ndim = chains[0].ndim
    if any(c.ndim != ndim for c in chains):
        raise ValueError("Chains have incompatible dimensions")
    return np.stack([c.samples[-draws:] for c in chains], axis=0)


def rank_normalized_rhat(values: np.ndarray) -> float:
    """Rank-normalized split R-hat with a folded-tail companion."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2:
        return math.nan
    flat = x.ravel()
    order = np.argsort(flat, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(flat) + 1)
    # Blom-style plotting positions transformed by an accurate erfinv approximation
    p = (ranks - 0.375) / (len(flat) + 0.25)
    try:
        from scipy.special import ndtri

        z = ndtri(p).reshape(x.shape)
    except ImportError:
        # Classical split-Rhat remains a sound fallback when SciPy is absent.
        return split_rhat(x)
    bulk = split_rhat(z)
    folded = np.abs(z - np.median(z))
    tail = split_rhat(folded)
    return float(max(bulk, tail))


def bulk_ess(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    flat = x.ravel()
    order = np.argsort(flat, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(flat) + 1)
    try:
        from scipy.special import ndtri

        z = ndtri((ranks - 0.375) / (len(flat) + 0.25)).reshape(x.shape)
        return effective_sample_size(z)
    except ImportError:
        return effective_sample_size(x)


def tail_ess(values: np.ndarray, probability: float = 0.05) -> float:
    x = np.asarray(values, dtype=float)
    lo, hi = np.quantile(x, [probability, 1 - probability])
    return float(
        min(
            effective_sample_size((x <= lo).astype(float)),
            effective_sample_size((x >= hi).astype(float)),
        )
    )


def geweke(values: np.ndarray, first: float = 0.1, last: float = 0.5) -> float:
    """Geweke early-vs-late z score for a scalar chain."""
    x = np.asarray(values, dtype=float).reshape(-1)
    if len(x) < 20 or not (0 < first < 1 and 0 < last < 1 and first + last <= 1):
        return math.nan
    a = x[: max(2, int(first * len(x)))]
    b = x[-max(2, int(last * len(x))) :]
    va = np.var(a, ddof=1) / len(a)
    vb = np.var(b, ddof=1) / len(b)
    return float((np.mean(a) - np.mean(b)) / math.sqrt(va + vb)) if va + vb > 0 else 0.0


def raftery_lewis(
    values: np.ndarray,
    *,
    quantile: float = 0.025,
    accuracy: float = 0.005,
    probability: float = 0.95,
) -> dict[str, float]:
    """Practical Raftery-Lewis style sample-size estimate.

    The dependence factor is estimated from the ESS of the indicator process, which
    provides a stable approximation without fitting an explicit two-state chain.
    """
    x = np.asarray(values, dtype=float).reshape(-1)
    if len(x) < 20:
        return {
            "minimum": math.nan,
            "dependence_factor": math.nan,
            "recommended": math.nan,
        }
    threshold = np.quantile(x, quantile)
    indicator = (x <= threshold).astype(float)[None, :]
    del probability
    z = 1.959963984540054
    minimum = quantile * (1 - quantile) * (z / accuracy) ** 2
    ess = effective_sample_size(indicator)
    factor = len(x) / ess if ess > 0 else math.inf
    return {
        "minimum": float(math.ceil(minimum)),
        "dependence_factor": float(factor),
        "recommended": float(math.ceil(minimum * factor)),
    }


def heidelberger_welch(
    values: np.ndarray, *, relative_halfwidth: float = 0.1
) -> dict[str, float | bool]:
    """Stationarity/half-width diagnostic with a conservative split-mean test."""
    x = np.asarray(values, dtype=float).reshape(-1)
    if len(x) < 20:
        return {
            "stationary": False,
            "halfwidth_pass": False,
            "relative_halfwidth": math.inf,
        }
    half = len(x) // 2
    a = x[:half]
    b = x[-half:]
    denom = math.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b))
    z = abs(np.mean(a) - np.mean(b)) / denom if denom > 0 else 0.0
    stationary = z < 1.96
    ess = effective_sample_size(x[None, :])
    hw = 1.96 * np.std(x, ddof=1) / math.sqrt(max(ess, 1))
    scale = max(abs(np.mean(x)), np.std(x, ddof=1), 1e-15)
    rel = hw / scale
    return {
        "stationary": bool(stationary),
        "halfwidth_pass": bool(rel <= relative_halfwidth),
        "relative_halfwidth": float(rel),
    }


@dataclass(frozen=True, slots=True)
class MCMCDiagnostics:
    rhat: np.ndarray
    rank_rhat: np.ndarray
    ess: np.ndarray
    bulk_ess: np.ndarray
    tail_ess: np.ndarray
    mcse: np.ndarray
    geweke_z: np.ndarray
    divergence_count: int
    acceptance_rate: np.ndarray

    @property
    def reliable(self) -> bool:
        finite = self.rank_rhat[np.isfinite(self.rank_rhat)]
        return (
            self.divergence_count == 0
            and (not len(finite) or np.all(finite < 1.01))
            and np.all(self.bulk_ess >= 100)
            and np.all(self.tail_ess >= 100)
        )


def diagnose_chains(chains: Sequence[MCMCChain]) -> MCMCDiagnostics:
    x = _stack(chains)
    ndim = x.shape[-1]
    rhat = np.empty(ndim)
    rr = np.empty(ndim)
    ess = np.empty(ndim)
    bess = np.empty(ndim)
    tess = np.empty(ndim)
    mcse = np.empty(ndim)
    gz = np.empty(ndim)
    for j in range(ndim):
        vals = x[:, :, j]
        rhat[j] = split_rhat(vals)
        rr[j] = rank_normalized_rhat(vals)
        ess[j] = effective_sample_size(vals)
        bess[j] = bulk_ess(vals)
        tess[j] = tail_ess(vals)
        mcse[j] = mcse_mean(vals)
        gz[j] = max(abs(geweke(row)) for row in vals)
    divergences = sum(
        int(np.sum(np.asarray(c.sample_stats.get("divergent", []), dtype=bool)))
        for c in chains
    )
    return MCMCDiagnostics(
        rhat,
        rr,
        ess,
        bess,
        tess,
        mcse,
        gz,
        divergences,
        np.asarray([c.acceptance_rate for c in chains], dtype=float),
    )
