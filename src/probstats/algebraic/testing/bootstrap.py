"""Gaussian multiplier bootstrap for SDL semialgebraic tests."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from numbers import Integral

import numpy as np

from ..._arrays import readonly_array
from ._validation import as_generator
from .statistics import SDLStudentizationResult


@dataclass(frozen=True, slots=True)
class SDLBootstrapResult:
    """Conditional Gaussian multiplier bootstrap based on one SDL realization."""

    statistics: np.ndarray
    p_value: float
    replicates: int
    studentization: SDLStudentizationResult
    h_component: np.ndarray | None = None
    g_component: np.ndarray | None = None
    combined: np.ndarray | None = None

    def __post_init__(self):
        if self.statistics is not None:
            object.__setattr__(self, "statistics", readonly_array(self.statistics))
        if self.h_component is not None:
            object.__setattr__(self, "h_component", readonly_array(self.h_component))
        if self.g_component is not None:
            object.__setattr__(self, "g_component", readonly_array(self.g_component))
        if self.combined is not None:
            object.__setattr__(self, "combined", readonly_array(self.combined))


def _replicate_count(value: int) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError("replicates must be an integer")
    value = int(value)
    if value < 1:
        raise ValueError("replicates must be positive")
    return value


def sdl_multiplier_bootstrap(
    studentization: SDLStudentizationResult,
    *,
    replicates: int = 1000,
    rng: np.random.Generator | int | None = None,
    batch_size: int | None = None,
    retain_components: bool = False,
) -> SDLBootstrapResult:
    """Approximate the SDL reference distribution by Gaussian multipliers.

    Conditional on the realized data and Bernoulli subset selectors, this
    implements the two processes in the SDL construction,

    ``U#_h = Nhat**(-1/2) sum_i xi'_i (h_i - U')``

    and

    ``U#_g = n1**(-1/2) sum_i xi_i (G_i - G)``,

    followed by ``U# = m U#_g + sqrt(alpha) U#_h``.  Each bootstrap statistic
    is ``max_j U#_j / sigma_hat_j``.  The returned p-value is the empirical
    fraction of bootstrap statistics at least as large as the observed SDL
    statistic.

    ``batch_size`` limits temporary multiplier arrays. It does not change the
    bootstrap law, but it can change the exact seeded realization because the
    h- and g-multiplier draws are interleaved once per batch. The p-value uses
    the empirical exceedance fraction specified by the SDL procedure rather
    than a finite-Monte-Carlo ``+1`` correction.
    """
    A = _replicate_count(replicates)
    if batch_size is None:
        batch = A
    else:
        if not isinstance(batch_size, Integral) or isinstance(batch_size, bool):
            raise TypeError("batch_size must be an integer")
        batch = int(batch_size)
        if batch < 1:
            raise ValueError("batch_size must be positive")

    u = studentization.u_statistic
    if not u.randomized or u.subset_indices is None:
        raise ValueError("studentization must retain a randomized subset realization")
    n_hat = len(u.subset_indices)
    if n_hat < 1:
        raise ValueError("bootstrap requires at least one selected U-statistic subset")

    # Studentization retains the concrete kernel evaluations so the bootstrap
    # can reuse exactly the same incomplete-U realization.
    h_values = getattr(studentization, "kernel_values", None)
    if h_values is None:
        raise ValueError(
            "studentization does not retain kernel evaluations required by the "
            "multiplier bootstrap; recompute it with sdl_studentize"
        )
    h_values = np.asarray(h_values, dtype=float)
    if h_values.ndim == 1:
        h_values = h_values[:, None]
    g_values = np.asarray(studentization.hajek.values, dtype=float)
    if g_values.ndim == 1:
        g_values = g_values[:, None]
    h_centered = h_values - np.asarray(u.value, dtype=float).reshape(-1)
    g_centered = g_values - np.asarray(studentization.hajek.mean, dtype=float).reshape(
        -1
    )
    n1 = g_centered.shape[0]
    if n1 < 1:
        raise ValueError("bootstrap requires at least one Hájek projection value")

    sigma = np.asarray(studentization.standard_error, dtype=float).reshape(-1)
    if np.any(sigma < 0) or np.any(~np.isfinite(sigma)):
        raise ValueError("studentization contains invalid standard errors")

    generator = as_generator(rng)
    statistics = np.empty(A, dtype=float)
    keep_h = np.empty((A, h_centered.shape[1])) if retain_components else None
    keep_g = np.empty((A, g_centered.shape[1])) if retain_components else None
    keep_combined = np.empty((A, h_centered.shape[1])) if retain_components else None

    m = u.order
    alpha_root = sqrt(studentization.alpha)
    for start in range(0, A, batch):
        stop = min(A, start + batch)
        size = stop - start
        xi_h = generator.standard_normal((size, n_hat))
        xi_g = generator.standard_normal((size, n1))
        uh = (xi_h @ h_centered) / sqrt(n_hat)
        ug = (xi_g @ g_centered) / sqrt(n1)
        combined = m * ug + alpha_root * uh
        with np.errstate(divide="ignore", invalid="ignore"):
            standardized = combined / sigma
        # Degenerate coordinates contribute zero exactly when the bootstrap
        # numerator is zero; otherwise their infinite sign is mathematically
        # retained by NumPy division.
        standardized = np.where((sigma == 0) & (combined == 0), 0.0, standardized)
        statistics[start:stop] = np.max(standardized, axis=1)
        if retain_components:
            keep_h[start:stop] = uh
            keep_g[start:stop] = ug
            keep_combined[start:stop] = combined

    p_value = float(np.mean(statistics >= studentization.statistic))
    return SDLBootstrapResult(
        statistics=statistics,
        p_value=p_value,
        replicates=A,
        studentization=studentization,
        h_component=keep_h,
        g_component=keep_g,
        combined=keep_combined,
    )
