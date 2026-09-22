"""Nested-sampling inference in stable log space.

Evidence quadrature is separated from constrained-prior exploration. Two prior
interfaces are supported:

* ``prior_sampler(rng)`` for arbitrary independent prior draws.  The correctness
  baseline is rejection sampling from that prior.
* ``prior_transform(u)`` with ``u`` uniform on the unit cube.  This is the standard
  nested-sampling parameterization and enables likelihood-constrained slice moves
  that preserve the prior exactly in unit-cube coordinates.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..core import InferenceKind, InferenceResult, InferenceStep

Array = np.ndarray
LogLikelihood = Callable[[Array], float]
PriorSampler = Callable[[np.random.Generator], Array]
PriorTransform = Callable[[Array], Array]


def _logaddexp(a: float, b: float) -> float:
    return float(np.logaddexp(a, b))


def _logdiffexp(a: float, b: float) -> float:
    """Return ``log(exp(a) - exp(b))`` for ``a > b``, stably."""
    if not a > b:
        raise ValueError("logdiffexp requires a > b")
    return float(a + np.log1p(-np.exp(b - a)))


@dataclass(frozen=True, slots=True)
class NestedPoint:
    point: Array
    log_likelihood: float
    log_x: float
    log_weight: float
    log_posterior_weight: float


@dataclass(frozen=True, slots=True)
class EmpiricalPosterior:
    """Weighted empirical posterior reconstructed from nested samples."""

    points: Array
    log_weights: Array
    names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=float)
        weights = np.asarray(self.log_weights, dtype=float)
        if points.ndim != 2:
            raise ValueError("points must be a two-dimensional array")
        if weights.shape != (points.shape[0],):
            raise ValueError("log_weights must have one entry per point")
        norm = float(np.logaddexp.reduce(weights))
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "log_weights", weights - norm)
        object.__setattr__(self, "names", tuple(self.names))

    @property
    def weights(self) -> Array:
        return np.exp(self.log_weights)

    @property
    def mean(self) -> Array:
        return np.sum(self.points * self.weights[:, None], axis=0)

    @property
    def covariance(self) -> Array:
        delta = self.points - self.mean
        return (delta * self.weights[:, None]).T @ delta

    @property
    def effective_sample_size(self) -> float:
        w = self.weights
        return float(1.0 / np.sum(w * w))

    def resample(
        self, size: int, *, rng: np.random.Generator | int | None = None
    ) -> Array:
        generator = np.random.default_rng(rng)
        indices = generator.choice(len(self.points), size=int(size), p=self.weights)
        return self.points[indices].copy()


@dataclass(frozen=True, slots=True)
class ConstrainedDraw:
    point: Array
    log_likelihood: float
    attempts: int
    acceptance_rate: float
    auxiliary: Array | None = None


class ConstrainedSampler(Protocol):
    def draw(
        self,
        *,
        threshold: float,
        live_points: Array,
        live_log_likelihoods: Array,
        log_likelihood: LogLikelihood,
        prior_sampler: PriorSampler | None,
        rng: np.random.Generator,
        live_auxiliary: Array | None = None,
        prior_transform: PriorTransform | None = None,
    ) -> ConstrainedDraw: ...


@dataclass(slots=True)
class RejectionConstrainedSampler:
    """Exact constrained-prior replacement by rejection from independent prior draws."""

    max_attempts: int = 100_000
    ndim: int | None = None

    def draw(
        self,
        *,
        threshold: float,
        live_points: Array,
        live_log_likelihoods: Array,
        log_likelihood: LogLikelihood,
        prior_sampler: PriorSampler | None,
        rng: np.random.Generator,
        live_auxiliary: Array | None = None,
        prior_transform: PriorTransform | None = None,
    ) -> ConstrainedDraw:
        del live_points, live_log_likelihoods, live_auxiliary
        for attempt in range(1, self.max_attempts + 1):
            if prior_sampler is not None:
                point = np.asarray(prior_sampler(rng), dtype=float).reshape(-1)
                auxiliary = None
            elif prior_transform is not None:
                if self.ndim is None:
                    raise RuntimeError("unit-cube dimension was not initialized")
                auxiliary = rng.random(self.ndim)
                point = np.asarray(prior_transform(auxiliary), dtype=float).reshape(-1)
            else:
                raise ValueError("A prior_sampler or prior_transform is required")
            value = float(log_likelihood(point))
            if np.isfinite(value) and value >= threshold:
                return ConstrainedDraw(point, value, attempt, 1.0 / attempt, auxiliary)
        raise RuntimeError(
            f"Could not draw from constrained prior after {self.max_attempts} attempts; "
            "use a slice sampler with a prior_transform for concentrated likelihood contours."
        )


@dataclass(slots=True)
class MCMCConstrainedSampler:
    """Likelihood-constrained adaptive Metropolis in unit-cube prior coordinates.

    This sampler is valid only with ``prior_transform`` because a uniform target in
    the unit cube maps exactly to the prior.  It is useful when callers prefer a
    resumable MCMC-style kernel over directional slice moves.
    """

    steps: int = 32
    proposal_scale: float = 0.05

    def draw(
        self,
        *,
        threshold: float,
        live_points: Array,
        live_log_likelihoods: Array,
        log_likelihood: LogLikelihood,
        prior_sampler: PriorSampler | None,
        rng: np.random.Generator,
        live_auxiliary: Array | None = None,
        prior_transform: PriorTransform | None = None,
    ) -> ConstrainedDraw:
        del prior_sampler
        if prior_transform is None or live_auxiliary is None:
            raise ValueError(
                "MCMCConstrainedSampler requires prior_transform and live unit-cube coordinates"
            )
        from ..mcmc import AdaptiveMetropolis, sample_mcmc

        eligible = np.flatnonzero(live_log_likelihoods >= threshold)
        if not len(eligible):
            raise RuntimeError("No live point satisfies the likelihood constraint")
        seed = int(rng.choice(eligible))
        u0 = np.asarray(live_auxiliary[seed], dtype=float).copy()

        def constrained_log_density(u: Array) -> float:
            u = np.asarray(u, dtype=float)
            if np.any(u < 0.0) or np.any(u > 1.0):
                return -np.inf
            point = np.asarray(prior_transform(u), dtype=float).reshape(-1)
            value = float(log_likelihood(point))
            return 0.0 if np.isfinite(value) and value >= threshold else -np.inf

        chain = sample_mcmc(
            constrained_log_density,
            u0,
            sampler=AdaptiveMetropolis(
                initial_cov=self.proposal_scale**2, adapt_start=max(4, self.steps // 4)
            ),
            draws=1,
            warmup=max(self.steps - 1, 0),
            rng=rng,
        )
        u = chain.samples[-1]
        point = np.asarray(prior_transform(u), dtype=float).reshape(-1)
        value = float(log_likelihood(point))
        if value < threshold or not np.isfinite(value):
            raise RuntimeError(
                "MCMC constrained kernel violated the likelihood constraint"
            )
        return ConstrainedDraw(
            point, value, chain.state.proposed, chain.acceptance_rate, u.copy()
        )


@dataclass(slots=True)
class SliceConstrainedSampler:
    """Likelihood-constrained slice sampling in unit-cube prior coordinates.

    The sampler targets the uniform distribution on
    ``{u in [0,1]^d : log_likelihood(prior_transform(u)) >= threshold}``.
    Mapping ``u`` through ``prior_transform`` therefore preserves the requested
    prior exactly.  Repeated random-direction chord slices mix within the current
    constrained region without the exponentially falling rejection efficiency of
    naive prior rejection.

    ``steps`` controls the number of accepted slice moves per nested replacement.
    ``max_shrink`` bounds likelihood evaluations spent shrinking a one-dimensional
    chord after a rejected proposal.  ``differential_directions`` occasionally uses
    differences between live unit-cube points, which adapts cheaply to anisotropy;
    otherwise an isotropic random direction is used.
    """

    steps: int = 8
    max_shrink: int = 2_000
    differential_directions: bool = True

    @staticmethod
    def _cube_chord(u: Array, direction: Array) -> tuple[float, float]:
        lower = -np.inf
        upper = np.inf
        for value, slope in zip(u, direction):
            if abs(slope) < 1e-15:
                continue
            a = (0.0 - value) / slope
            b = (1.0 - value) / slope
            lower = max(lower, min(a, b))
            upper = min(upper, max(a, b))
        if not lower <= 0.0 <= upper:
            raise RuntimeError("Current unit-cube point is outside its feasible chord")
        return float(lower), float(upper)

    def _direction(self, live_u: Array, rng: np.random.Generator) -> Array:
        ndim = live_u.shape[1]
        if self.differential_directions and len(live_u) >= 3 and rng.random() < 0.5:
            i, j = rng.choice(len(live_u), size=2, replace=False)
            direction = live_u[i] - live_u[j]
        else:
            direction = rng.normal(size=ndim)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-14:
            direction = rng.normal(size=ndim)
            norm = float(np.linalg.norm(direction))
        return direction / norm

    def draw(
        self,
        *,
        threshold: float,
        live_points: Array,
        live_log_likelihoods: Array,
        log_likelihood: LogLikelihood,
        prior_sampler: PriorSampler | None,
        rng: np.random.Generator,
        live_auxiliary: Array | None = None,
        prior_transform: PriorTransform | None = None,
    ) -> ConstrainedDraw:
        del prior_sampler
        if prior_transform is None or live_auxiliary is None:
            raise ValueError(
                "SliceConstrainedSampler requires prior_transform and live unit-cube coordinates. "
                "Use nested_sample(..., prior_transform=...) or the rejection sampler."
            )
        if self.steps < 1:
            raise ValueError("steps must be positive")
        if self.max_shrink < 1:
            raise ValueError("max_shrink must be positive")

        eligible = np.flatnonzero(live_log_likelihoods >= threshold)
        if not len(eligible):
            raise RuntimeError("No live point satisfies the likelihood constraint")
        # Avoid deterministically cloning the worst point on plateaus by choosing a
        # random current live state as the start of the Markov kernel.
        seed = int(rng.choice(eligible))
        u = np.asarray(live_auxiliary[seed], dtype=float).copy()
        point = np.asarray(live_points[seed], dtype=float).copy()
        value = float(live_log_likelihoods[seed])
        evaluations = 0
        accepted_moves = 0

        for _ in range(self.steps):
            direction = self._direction(live_auxiliary, rng)
            left, right = self._cube_chord(u, direction)
            moved = False
            for _ in range(self.max_shrink):
                t = float(rng.uniform(left, right))
                proposal_u = np.clip(u + t * direction, 0.0, 1.0)
                proposal = np.asarray(prior_transform(proposal_u), dtype=float).reshape(
                    -1
                )
                proposal_ll = float(log_likelihood(proposal))
                evaluations += 1
                if np.isfinite(proposal_ll) and proposal_ll >= threshold:
                    u = proposal_u
                    point = proposal
                    value = proposal_ll
                    accepted_moves += 1
                    moved = True
                    break
                if t < 0.0:
                    left = t
                else:
                    right = t
                if right - left <= 1e-14:
                    break
            if not moved:
                # Staying at a valid constrained-prior state leaves the target
                # invariant; subsequent random directions can still make progress.
                continue

        attempts = max(evaluations, 1)
        return ConstrainedDraw(
            point=point,
            log_likelihood=value,
            attempts=attempts,
            acceptance_rate=accepted_moves / attempts,
            auxiliary=u,
        )


@dataclass(frozen=True, slots=True)
class NestedSamplingRun:
    dead_points: tuple[NestedPoint, ...]
    live_points: Array
    live_log_likelihoods: Array
    log_evidence: float
    information: float
    log_evidence_error: float
    iterations: int
    n_live: int
    termination_reason: str
    acceptance_rate: float
    parameter_names: tuple[str, ...] = ()
    constrained_sampler: str = "unknown"

    @property
    def posterior(self) -> EmpiricalPosterior:
        points = np.vstack([sample.point for sample in self.dead_points])
        log_weights = np.asarray(
            [sample.log_posterior_weight for sample in self.dead_points]
        )
        return EmpiricalPosterior(points, log_weights, self.parameter_names)


def _posterior_information(
    log_weights: Array, log_likelihoods: Array, log_z: float
) -> float:
    probs = np.exp(log_weights - log_z)
    return float(np.sum(probs * (log_likelihoods - log_z)))


def _initialize_live_points(
    *,
    prior_sampler: PriorSampler | None,
    prior_transform: PriorTransform | None,
    ndim: int,
    n_live: int,
    rng: np.random.Generator,
) -> tuple[Array, Array | None]:
    if prior_transform is not None:
        live_u = rng.random((n_live, ndim))
        live = np.vstack(
            [np.asarray(prior_transform(u), dtype=float).reshape(-1) for u in live_u]
        )
        return live, live_u
    if prior_sampler is None:
        raise ValueError("Provide either prior_sampler or prior_transform")
    live = np.vstack(
        [np.asarray(prior_sampler(rng), dtype=float).reshape(-1) for _ in range(n_live)]
    )
    return live, None


def nested_sample(
    log_likelihood: LogLikelihood,
    prior_sampler: PriorSampler | None = None,
    *,
    prior_transform: PriorTransform | None = None,
    ndim: int,
    n_live: int = 100,
    max_iterations: int = 10_000,
    min_iterations: int = 100,
    termination_fraction: float = 0.01,
    constrained_sampler: ConstrainedSampler | None = None,
    rng: np.random.Generator | int | None = None,
    parameter_names: Sequence[str] = (),
) -> NestedSamplingRun:
    """Run classical nested sampling with deterministic expected shrinkage.

    If ``prior_transform`` is supplied, initial live points are generated from a
    uniform unit cube and the default constrained sampler is
    :class:`SliceConstrainedSampler`.  This is preferred for production use.
    With only ``prior_sampler``, constrained proposals use exact rejection from
    independent prior draws.
    """
    if n_live < 2:
        raise ValueError("n_live must be at least 2")
    if ndim < 1:
        raise ValueError("ndim must be positive")
    if not 0 < termination_fraction < 1:
        raise ValueError("termination_fraction must lie in (0, 1)")
    if prior_sampler is not None and prior_transform is not None:
        # A transform fully specifies normalized prior sampling and, crucially,
        # exposes the geometry needed by the slice kernel.  Accept both only if
        # callers supplied them; the transform is authoritative.
        prior_sampler_for_draws = None
    else:
        prior_sampler_for_draws = prior_sampler

    generator = np.random.default_rng(rng)
    sampler: ConstrainedSampler
    if constrained_sampler is not None:
        sampler = constrained_sampler
    elif prior_transform is not None:
        sampler = SliceConstrainedSampler()
    else:
        sampler = RejectionConstrainedSampler()

    # Rejection from a unit-cube transform needs to know ndim if explicitly chosen.
    if isinstance(sampler, RejectionConstrainedSampler):
        sampler.ndim = ndim

    live, live_aux = _initialize_live_points(
        prior_sampler=prior_sampler,
        prior_transform=prior_transform,
        ndim=ndim,
        n_live=n_live,
        rng=generator,
    )
    if live.shape != (n_live, ndim):
        raise ValueError(
            f"Prior mapping returned shape incompatible with ndim={ndim}: {live.shape}"
        )
    live_logl = np.asarray(
        [float(log_likelihood(point)) for point in live], dtype=float
    )
    if not np.all(np.isfinite(live_logl)):
        raise ValueError("Initial prior draws produced non-finite log likelihoods")

    dead: list[NestedPoint] = []
    log_z = -np.inf
    acceptance_numer = 0.0
    acceptance_denom = 0.0
    termination_reason = "max_iterations"

    for iteration in range(1, max_iterations + 1):
        worst = int(np.argmin(live_logl))
        threshold = float(live_logl[worst])
        log_x_prev = -(iteration - 1) / n_live
        log_x = -iteration / n_live
        log_w = _logdiffexp(log_x_prev, log_x)
        log_post_w = log_w + threshold
        log_z = _logaddexp(log_z, log_post_w)
        dead.append(
            NestedPoint(live[worst].copy(), threshold, log_x, log_w, log_post_w)
        )

        draw = sampler.draw(
            threshold=threshold,
            live_points=live,
            live_log_likelihoods=live_logl,
            log_likelihood=log_likelihood,
            prior_sampler=prior_sampler_for_draws,
            rng=generator,
            live_auxiliary=live_aux,
            prior_transform=prior_transform,
        )
        if draw.point.shape != (ndim,):
            raise ValueError(
                f"Constrained sampler returned shape {draw.point.shape}, expected {(ndim,)}"
            )
        if draw.log_likelihood < threshold or not np.isfinite(draw.log_likelihood):
            raise RuntimeError("Constrained sampler violated the likelihood constraint")
        live[worst] = draw.point
        live_logl[worst] = draw.log_likelihood
        if live_aux is not None:
            if draw.auxiliary is None:
                raise RuntimeError(
                    "Unit-cube constrained sampler did not return auxiliary coordinates"
                )
            live_aux[worst] = np.asarray(draw.auxiliary, dtype=float)
        acceptance_numer += max(draw.acceptance_rate * draw.attempts, 0.0)
        acceptance_denom += max(draw.attempts, 1)

        log_remaining_upper = log_x + float(np.max(live_logl))
        if iteration >= min_iterations and log_remaining_upper <= log_z + math.log(
            termination_fraction
        ):
            termination_reason = "evidence_fraction"
            break

    iterations = len(dead)
    log_x_final = -iterations / n_live
    log_live_w = log_x_final - math.log(n_live)
    for point, ll in zip(live, live_logl):
        log_post_w = log_live_w + float(ll)
        log_z = _logaddexp(log_z, log_post_w)
        dead.append(
            NestedPoint(point.copy(), float(ll), -np.inf, log_live_w, log_post_w)
        )

    all_logw = np.asarray([p.log_posterior_weight for p in dead])
    all_logl = np.asarray([p.log_likelihood for p in dead])
    information = max(0.0, _posterior_information(all_logw, all_logl, log_z))
    error = math.sqrt(information / n_live)
    acceptance = acceptance_numer / acceptance_denom if acceptance_denom else 0.0

    return NestedSamplingRun(
        dead_points=tuple(dead),
        live_points=live.copy(),
        live_log_likelihoods=live_logl.copy(),
        log_evidence=log_z,
        information=information,
        log_evidence_error=error,
        iterations=iterations,
        n_live=n_live,
        termination_reason=termination_reason,
        acceptance_rate=acceptance,
        parameter_names=tuple(parameter_names),
        constrained_sampler=type(sampler).__name__,
    )


def infer_nested(
    log_likelihood: LogLikelihood,
    prior_sampler: PriorSampler | None = None,
    *,
    prior_transform: PriorTransform | None = None,
    ndim: int,
    n_live: int = 100,
    max_iterations: int = 10_000,
    min_iterations: int = 100,
    termination_fraction: float = 0.01,
    constrained_sampler: ConstrainedSampler | None = None,
    rng: np.random.Generator | int | None = None,
    parameter_names: Sequence[str] = (),
) -> InferenceResult:
    run = nested_sample(
        log_likelihood,
        prior_sampler,
        prior_transform=prior_transform,
        ndim=ndim,
        n_live=n_live,
        max_iterations=max_iterations,
        min_iterations=min_iterations,
        termination_fraction=termination_fraction,
        constrained_sampler=constrained_sampler,
        rng=rng,
        parameter_names=parameter_names,
    )
    return InferenceResult(
        posterior=run.posterior,
        kind=InferenceKind.SAMPLED,
        log_evidence=run.log_evidence,
        steps=(
            InferenceStep(
                method="nested-sampling",
                description=(
                    f"Estimated posterior and evidence with {n_live} live points, "
                    f"{run.iterations} nested replacements, and {run.constrained_sampler}."
                ),
                exact=False,
                metadata={
                    "termination_reason": run.termination_reason,
                    "constrained_sampler": run.constrained_sampler,
                },
            ),
        ),
        diagnostics={
            "information": run.information,
            "log_evidence_error": run.log_evidence_error,
            "acceptance_rate": run.acceptance_rate,
            "iterations": run.iterations,
            "n_live": run.n_live,
            "termination_reason": run.termination_reason,
            "posterior_ess": run.posterior.effective_sample_size,
            "constrained_sampler": run.constrained_sampler,
        },
        metadata={"run": run},
    )


def evidence_resampling(
    run: NestedSamplingRun,
    *,
    samples: int = 500,
    rng: np.random.Generator | int | None = None,
) -> Array:
    """Monte Carlo uncertainty distribution for log evidence.

    Keep the discovered likelihood contour sequence fixed but resample the unknown
    prior-volume shrinkages.  For ``n_live`` live points, each shrinkage factor is
    distributed as ``Beta(n_live, 1)``.  Final live likelihoods share the remaining
    prior volume equally.
    """
    if samples < 1:
        raise ValueError("samples must be positive")
    generator = np.random.default_rng(rng)
    n_dead = run.iterations
    ordered = run.dead_points[:n_dead]
    final_live = run.dead_points[n_dead:]
    output = np.empty(samples, dtype=float)
    for index in range(samples):
        log_x_prev = 0.0
        log_z = -np.inf
        for sample in ordered:
            log_x = log_x_prev + math.log(generator.random()) / run.n_live
            log_w = _logdiffexp(log_x_prev, log_x)
            log_z = _logaddexp(log_z, log_w + sample.log_likelihood)
            log_x_prev = log_x
        log_live_w = log_x_prev - math.log(run.n_live)
        for sample in final_live:
            log_z = _logaddexp(log_z, log_live_w + sample.log_likelihood)
        output[index] = log_z
    return output


def combine_nested_runs(*runs: NestedSamplingRun) -> InferenceResult:
    """Combine independent runs by equal-run mixture of posterior/evidence estimates."""
    if not runs:
        raise ValueError("At least one run is required")
    names = runs[0].parameter_names
    if any(run.parameter_names != names for run in runs):
        raise ValueError("Runs have incompatible parameter names")
    logzs = np.asarray([run.log_evidence for run in runs])
    combined_logz = float(np.logaddexp.reduce(logzs) - math.log(len(runs)))
    pts = np.vstack([run.posterior.points for run in runs])
    logw = np.concatenate(
        [run.posterior.log_weights - math.log(len(runs)) for run in runs]
    )
    posterior = EmpiricalPosterior(pts, logw, names)
    between = (
        float(np.std(logzs, ddof=1)) if len(runs) > 1 else runs[0].log_evidence_error
    )
    return InferenceResult(
        posterior=posterior,
        kind=InferenceKind.SAMPLED,
        log_evidence=combined_logz,
        steps=(
            InferenceStep(
                method="combine-nested-runs",
                description=f"Combined {len(runs)} independent nested-sampling runs.",
                exact=False,
            ),
        ),
        diagnostics={
            "log_evidence_error": between,
            "runs": len(runs),
            "posterior_ess": posterior.effective_sample_size,
        },
        metadata={"runs": runs},
    )
