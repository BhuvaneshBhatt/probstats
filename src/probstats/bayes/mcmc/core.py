"""General-purpose Markov-chain Monte Carlo kernels.

The API is log-density first and keeps enough state to resume sampling without
restarting adaptation. Transition kernels remain independent from the symbolic model
layer.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np

Array = np.ndarray
LogDensity = Callable[[Array], float]
Gradient = Callable[[Array], Array]
ConditionalUpdate = Callable[[Array, np.random.Generator], float]


def _count(value: Any, *, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        qualifier = "positive" if minimum == 1 else "nonnegative"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _position(value: Any) -> Array:
    out = np.asarray(value, dtype=float).reshape(-1)
    if out.size == 0 or not np.all(np.isfinite(out)):
        raise ValueError("MCMC positions must be finite non-empty vectors")
    return out


def _logp(log_density: LogDensity, position: Array) -> float:
    value = float(log_density(position))
    return value if np.isfinite(value) else -math.inf


def finite_difference_gradient(
    log_density: LogDensity, position: Array, *, relative_step: float = 1e-6
) -> Array:
    """Central finite-difference gradient used only when no analytic gradient exists."""
    x = _position(position)
    grad = np.empty_like(x)
    for i in range(len(x)):
        h = relative_step * max(1.0, abs(x[i]))
        plus = x.copy()
        plus[i] += h
        minus = x.copy()
        minus[i] -= h
        fp = _logp(log_density, plus)
        fm = _logp(log_density, minus)
        if not np.isfinite(fp) or not np.isfinite(fm):
            raise ValueError(
                "Finite-difference gradient encountered a non-finite log density"
            )
        grad[i] = (fp - fm) / (2.0 * h)
    return grad


@dataclass(frozen=True, slots=True)
class MCMCState:
    position: Array
    log_prob: float
    iteration: int = 0
    accepted: int = 0
    proposed: int = 0
    sampler_state: Mapping[str, Any] = field(default_factory=dict)
    rng_state: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        position = _position(self.position).copy()
        position.setflags(write=False)
        log_prob = float(self.log_prob)
        if not np.isfinite(log_prob):
            raise ValueError("log_prob must be finite")
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < 0
            for value in (self.iteration, self.accepted, self.proposed)
        ):
            raise ValueError(
                "iteration, accepted, and proposed must be nonnegative integers"
            )
        if self.accepted > self.proposed:
            raise ValueError("accepted cannot exceed proposed")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "log_prob", log_prob)
        object.__setattr__(self, "iteration", int(self.iteration))
        object.__setattr__(self, "accepted", int(self.accepted))
        object.__setattr__(self, "proposed", int(self.proposed))
        object.__setattr__(
            self, "sampler_state", MappingProxyType(dict(self.sampler_state))
        )
        if self.rng_state is not None:
            object.__setattr__(self, "rng_state", copy.deepcopy(dict(self.rng_state)))

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else math.nan


@dataclass(frozen=True, slots=True)
class MCMCChain:
    samples: Array
    log_prob: Array
    state: MCMCState
    sampler: str
    sample_stats: Mapping[str, Array] = field(default_factory=dict)
    parameter_names: tuple[str, ...] = ()
    warmup: int = 0

    def __post_init__(self) -> None:
        samples = np.asarray(self.samples, dtype=float)
        log_prob = np.asarray(self.log_prob, dtype=float)
        if samples.ndim != 2:
            raise ValueError("samples must have shape (draw, parameter)")
        if log_prob.shape != (samples.shape[0],):
            raise ValueError("log_prob must have one entry per draw")
        if not np.all(np.isfinite(samples)) or not np.all(np.isfinite(log_prob)):
            raise ValueError("samples and log_prob must contain only finite values")
        stats = {str(k): np.asarray(v).copy() for k, v in self.sample_stats.items()}
        if any(v.ndim == 0 or v.shape[0] != samples.shape[0] for v in stats.values()):
            raise ValueError("sample_stats must have one leading entry per draw")
        names = tuple(self.parameter_names)
        if names and len(names) != samples.shape[1]:
            raise ValueError("parameter_names length does not match sample dimension")
        samples = samples.copy()
        log_prob = log_prob.copy()
        samples.setflags(write=False)
        log_prob.setflags(write=False)
        for value in stats.values():
            value.setflags(write=False)
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "log_prob", log_prob)
        object.__setattr__(self, "sample_stats", MappingProxyType(stats))
        object.__setattr__(self, "parameter_names", names)

    @property
    def draws(self) -> int:
        return self.samples.shape[0]

    @property
    def ndim(self) -> int:
        return self.samples.shape[1]

    @property
    def acceptance_rate(self) -> float:
        return self.state.acceptance_rate

    def resume(
        self,
        log_density: LogDensity,
        sampler: Sampler,
        draws: int,
        *,
        warmup: int = 0,
        thin: int = 1,
        rng: np.random.Generator | int | None = None,
    ) -> MCMCChain:
        return sample_mcmc(
            log_density,
            self.state.position,
            sampler=sampler,
            draws=draws,
            warmup=warmup,
            thin=thin,
            rng=rng,
            initial_state=self.state,
            parameter_names=self.parameter_names,
        )


@dataclass(frozen=True, slots=True)
class StepResult:
    position: Array
    log_prob: float
    accepted: bool
    sampler_state: Mapping[str, Any] = field(default_factory=dict)
    stats: Mapping[str, Any] = field(default_factory=dict)


class Sampler(Protocol):
    name: str

    def initialize(self, position: Array, log_prob: float) -> Mapping[str, Any]: ...
    def step(
        self, log_density: LogDensity, state: MCMCState, rng: np.random.Generator
    ) -> StepResult: ...


def _normal_logpdf(delta: Array, covariance: Array) -> float:
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        return -math.inf
    solve = np.linalg.solve(covariance, delta)
    return float(-0.5 * (len(delta) * math.log(2 * math.pi) + logdet + delta @ solve))


@dataclass(slots=True)
class MetropolisHastings:
    proposal_cov: Array | float = 1.0
    proposal: Callable[[Array, np.random.Generator], Array] | None = None
    log_proposal: Callable[[Array, Array], float] | None = None
    name: str = "metropolis-hastings"

    def initialize(self, position: Array, log_prob: float) -> Mapping[str, Any]:
        del log_prob
        cov = np.asarray(self.proposal_cov, dtype=float)
        if cov.ndim == 0:
            cov = np.eye(len(position)) * float(cov)
        if cov.shape != (len(position), len(position)):
            raise ValueError("proposal_cov has incompatible shape")
        np.linalg.cholesky(cov)
        return {"proposal_cov": cov}

    def step(
        self, log_density: LogDensity, state: MCMCState, rng: np.random.Generator
    ) -> StepResult:
        cov = np.asarray(state.sampler_state.get("proposal_cov"), dtype=float)
        current = state.position
        proposal = (
            _position(self.proposal(current.copy(), rng))
            if self.proposal is not None
            else rng.multivariate_normal(current, cov)
        )
        proposed_logp = _logp(log_density, proposal)
        log_alpha = proposed_logp - state.log_prob
        if self.log_proposal is not None:
            log_alpha += float(
                self.log_proposal(current, proposal)
                - self.log_proposal(proposal, current)
            )
        accepted = math.log(rng.random()) < min(0.0, log_alpha)
        return StepResult(
            proposal if accepted else current,
            proposed_logp if accepted else state.log_prob,
            accepted,
            state.sampler_state,
            {"log_accept_ratio": log_alpha},
        )


@dataclass(slots=True)
class IndependentMetropolis:
    proposal_sampler: Callable[[np.random.Generator], Array]
    proposal_logpdf: Callable[[Array], float]
    name: str = "independent-metropolis"

    def initialize(self, position: Array, log_prob: float) -> Mapping[str, Any]:
        del position, log_prob
        return {}

    def step(
        self, log_density: LogDensity, state: MCMCState, rng: np.random.Generator
    ) -> StepResult:
        proposal = _position(self.proposal_sampler(rng))
        proposed_logp = _logp(log_density, proposal)
        log_alpha = (
            proposed_logp
            - state.log_prob
            + float(self.proposal_logpdf(state.position))
            - float(self.proposal_logpdf(proposal))
        )
        accepted = math.log(rng.random()) < min(0.0, log_alpha)
        return StepResult(
            proposal if accepted else state.position,
            proposed_logp if accepted else state.log_prob,
            accepted,
            {},
            {"log_accept_ratio": log_alpha},
        )


class Transform(Protocol):
    def forward(self, unconstrained: Array) -> Array: ...
    def inverse(self, constrained: Array) -> Array: ...
    def log_abs_det_jacobian(self, unconstrained: Array) -> float: ...


@dataclass(slots=True)
class TransformedMetropolis:
    transform: Transform
    proposal_cov: Array | float = 1.0
    name: str = "transformed-metropolis"

    def initialize(self, position: Array, log_prob: float) -> Mapping[str, Any]:
        del log_prob
        unconstrained = _position(self.transform.inverse(position))
        cov = np.asarray(self.proposal_cov, dtype=float)
        if cov.ndim == 0:
            cov = np.eye(len(unconstrained)) * float(cov)
        np.linalg.cholesky(cov)
        return {"proposal_cov": cov, "unconstrained": unconstrained}

    def step(
        self, log_density: LogDensity, state: MCMCState, rng: np.random.Generator
    ) -> StepResult:
        u = np.asarray(state.sampler_state["unconstrained"], dtype=float)
        cov = np.asarray(state.sampler_state["proposal_cov"], dtype=float)
        proposal_u = rng.multivariate_normal(u, cov)
        proposal = _position(self.transform.forward(proposal_u))
        proposed_logp = _logp(log_density, proposal)
        current_aug = state.log_prob + float(self.transform.log_abs_det_jacobian(u))
        proposed_aug = proposed_logp + float(
            self.transform.log_abs_det_jacobian(proposal_u)
        )
        log_alpha = proposed_aug - current_aug
        accepted = math.log(rng.random()) < min(0.0, log_alpha)
        new_state = {
            "proposal_cov": cov,
            "unconstrained": proposal_u if accepted else u,
        }
        return StepResult(
            proposal if accepted else state.position,
            proposed_logp if accepted else state.log_prob,
            accepted,
            new_state,
            {"log_accept_ratio": log_alpha},
        )


@dataclass(slots=True)
class AdaptiveMetropolis:
    initial_cov: Array | float = 1.0
    scale: float | None = None
    adapt_start: int = 100
    epsilon: float = 1e-8
    name: str = "adaptive-metropolis"

    def initialize(self, position: Array, log_prob: float) -> Mapping[str, Any]:
        del log_prob
        d = len(position)
        cov = np.asarray(self.initial_cov, dtype=float)
        if cov.ndim == 0:
            cov = np.eye(d) * float(cov)
        if cov.shape != (d, d):
            raise ValueError("initial_cov has incompatible shape")
        np.linalg.cholesky(cov + self.epsilon * np.eye(d))
        return {
            "proposal_cov": cov,
            "mean": position.copy(),
            "m2": np.zeros((d, d)),
            "n": 1,
            "adapt": True,
        }

    def step(
        self, log_density: LogDensity, state: MCMCState, rng: np.random.Generator
    ) -> StepResult:
        ss = state.sampler_state
        cov = np.asarray(ss["proposal_cov"], dtype=float)
        d = len(state.position)
        scale = self.scale if self.scale is not None else 2.38**2 / d
        proposal_cov = scale * cov + self.epsilon * np.eye(d)
        proposal = rng.multivariate_normal(state.position, proposal_cov)
        proposed_logp = _logp(log_density, proposal)
        log_alpha = proposed_logp - state.log_prob
        accepted = math.log(rng.random()) < min(0.0, log_alpha)
        new_position = proposal if accepted else state.position
        n0 = int(ss["n"])
        mean0 = np.asarray(ss["mean"], dtype=float)
        m20 = np.asarray(ss["m2"], dtype=float)
        n = n0 + 1
        delta = new_position - mean0
        mean = mean0 + delta / n
        m2 = m20 + np.outer(delta, new_position - mean)
        learned = m2 / max(n - 1, 1)
        adapting = bool(ss.get("adapt", True))
        next_cov = (
            learned if adapting and state.iteration + 1 >= self.adapt_start else cov
        )
        next_state = {
            "proposal_cov": next_cov,
            "mean": mean,
            "m2": m2,
            "n": n,
            "adapt": adapting,
        }
        return StepResult(
            new_position,
            proposed_logp if accepted else state.log_prob,
            accepted,
            next_state,
            {"log_accept_ratio": log_alpha, "proposal_scale": scale},
        )


def _mass_matrix(mass: Array | float | None, dimension: int) -> tuple[Array, Array]:
    if mass is None:
        matrix = np.eye(dimension)
    else:
        matrix = np.asarray(mass, dtype=float)
        if matrix.ndim == 0:
            matrix = np.eye(dimension) * float(matrix)
    if matrix.shape != (dimension, dimension):
        raise ValueError("mass matrix has incompatible shape")
    np.linalg.cholesky(matrix)
    return matrix, np.linalg.inv(matrix)


def _leapfrog(
    position: Array,
    momentum: Array,
    step_size: float,
    gradient: Gradient,
    inv_mass: Array,
    direction: float = 1.0,
) -> tuple[Array, Array]:
    eps = direction * step_size
    p = momentum + 0.5 * eps * np.asarray(gradient(position), dtype=float)
    q = position + eps * (inv_mass @ p)
    p = p + 0.5 * eps * np.asarray(gradient(q), dtype=float)
    return q, p


def _hamiltonian(log_prob: float, momentum: Array, inv_mass: Array) -> float:
    return -log_prob + 0.5 * float(momentum @ inv_mass @ momentum)


@dataclass(slots=True)
class HamiltonianMonteCarlo:
    gradient: Gradient | None = None
    step_size: float = 0.1
    leapfrog_steps: int = 10
    mass_matrix: Array | float | None = None
    finite_difference_step: float = 1e-6
    target_accept: float = 0.8
    adapt_step_size: bool = True
    name: str = "hmc"

    def initialize(self, position: Array, log_prob: float) -> Mapping[str, Any]:
        del log_prob
        mass, inv_mass = _mass_matrix(self.mass_matrix, len(position))
        return {
            "mass_matrix": mass,
            "inv_mass": inv_mass,
            "step_size": float(self.step_size),
            "step_size_bar": float(self.step_size),
            "hbar": 0.0,
            "mu": math.log(10.0 * self.step_size),
            "adapt_t": 0,
            "adapt": self.adapt_step_size,
        }

    def step(
        self, log_density: LogDensity, state: MCMCState, rng: np.random.Generator
    ) -> StepResult:
        if self.step_size <= 0 or self.leapfrog_steps < 1:
            raise ValueError("HMC requires positive step_size and leapfrog_steps")
        mass = np.asarray(state.sampler_state["mass_matrix"], dtype=float)
        inv_mass = np.asarray(state.sampler_state["inv_mass"], dtype=float)
        grad = self.gradient or (
            lambda x: finite_difference_gradient(
                log_density, x, relative_step=self.finite_difference_step
            )
        )
        momentum0 = rng.multivariate_normal(np.zeros(len(state.position)), mass)
        q = state.position.copy()
        p = momentum0.copy()
        divergent = False
        try:
            for _ in range(self.leapfrog_steps):
                q, p = _leapfrog(
                    q,
                    p,
                    float(state.sampler_state.get("step_size", self.step_size)),
                    grad,
                    inv_mass,
                )
            proposed_logp = _logp(log_density, q)
            current_h = _hamiltonian(state.log_prob, momentum0, inv_mass)
            proposed_h = _hamiltonian(proposed_logp, p, inv_mass)
            energy_error = proposed_h - current_h
            divergent = not np.isfinite(energy_error) or abs(energy_error) > 1000
            log_alpha = -energy_error if not divergent else -math.inf
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            proposed_logp = -math.inf
            log_alpha = -math.inf
            energy_error = math.inf
            divergent = True
        accept_stat = (
            min(1.0, math.exp(min(0.0, log_alpha))) if np.isfinite(log_alpha) else 0.0
        )
        accepted = math.log(rng.random()) < min(0.0, log_alpha)
        ss = dict(state.sampler_state)
        if ss.get("adapt", False):
            t = int(ss.get("adapt_t", 0)) + 1
            eta = 1.0 / (t + 10.0)
            hbar = (1.0 - eta) * float(ss.get("hbar", 0.0)) + eta * (
                self.target_accept - accept_stat
            )
            log_eps = float(ss["mu"]) - math.sqrt(t) / 0.05 * hbar
            weight = t**-0.75
            log_bar = weight * log_eps + (1.0 - weight) * math.log(
                float(ss.get("step_size_bar", self.step_size))
            )
            ss.update(
                step_size=math.exp(log_eps),
                step_size_bar=math.exp(log_bar),
                hbar=hbar,
                adapt_t=t,
            )
        return StepResult(
            q if accepted else state.position,
            proposed_logp if accepted else state.log_prob,
            accepted,
            ss,
            {
                "log_accept_ratio": log_alpha,
                "accept_stat": accept_stat,
                "energy_error": energy_error,
                "divergent": divergent,
                "tree_depth": 0,
            },
        )


@dataclass(slots=True)
class NoUTurnSampler:
    gradient: Gradient | None = None
    step_size: float = 0.25
    max_tree_depth: int = 8
    mass_matrix: Array | float | None = None
    max_energy_error: float = 1000.0
    finite_difference_step: float = 1e-6
    target_accept: float = 0.8
    adapt_step_size: bool = True
    name: str = "nuts"

    def initialize(self, position: Array, log_prob: float) -> Mapping[str, Any]:
        del log_prob
        mass, inv_mass = _mass_matrix(self.mass_matrix, len(position))
        return {
            "mass_matrix": mass,
            "inv_mass": inv_mass,
            "step_size": float(self.step_size),
            "step_size_bar": float(self.step_size),
            "hbar": 0.0,
            "mu": math.log(10.0 * self.step_size),
            "adapt_t": 0,
            "adapt": self.adapt_step_size,
        }

    @staticmethod
    def _stop(
        q_minus: Array, q_plus: Array, p_minus: Array, p_plus: Array, inv_mass: Array
    ) -> bool:
        delta = q_plus - q_minus
        return (
            float(delta @ (inv_mass @ p_minus)) >= 0
            and float(delta @ (inv_mass @ p_plus)) >= 0
        )

    def step(
        self, log_density: LogDensity, state: MCMCState, rng: np.random.Generator
    ) -> StepResult:
        if self.step_size <= 0 or self.max_tree_depth < 1:
            raise ValueError("NUTS requires positive step_size and max_tree_depth")
        mass = np.asarray(state.sampler_state["mass_matrix"], dtype=float)
        inv_mass = np.asarray(state.sampler_state["inv_mass"], dtype=float)
        grad = self.gradient or (
            lambda x: finite_difference_gradient(
                log_density, x, relative_step=self.finite_difference_step
            )
        )
        p0 = rng.multivariate_normal(np.zeros(len(state.position)), mass)
        joint0 = state.log_prob - 0.5 * float(p0 @ inv_mass @ p0)
        log_u = joint0 + math.log(rng.random())
        q_minus = state.position.copy()
        q_plus = state.position.copy()
        p_minus = p0.copy()
        p_plus = p0.copy()
        candidate = state.position.copy()
        candidate_logp = state.log_prob
        n_valid = 1
        keep_going = True
        divergent = False
        accepted_stat = 0.0
        leapfrogs = 0
        depth_reached = 0

        def build(q: Array, p: Array, direction: float, depth: int):
            nonlocal divergent, leapfrogs
            if depth == 0:
                try:
                    q1, p1 = _leapfrog(
                        q,
                        p,
                        float(state.sampler_state.get("step_size", self.step_size)),
                        grad,
                        inv_mass,
                        direction,
                    )
                    leapfrogs += 1
                    lp1 = _logp(log_density, q1)
                    joint = lp1 - 0.5 * float(p1 @ inv_mass @ p1)
                    valid = int(log_u <= joint)
                    cont = (log_u - self.max_energy_error) < joint and np.isfinite(
                        joint
                    )
                    if not cont:
                        divergent = True
                    alpha = (
                        min(1.0, math.exp(min(0.0, joint - joint0)))
                        if np.isfinite(joint)
                        else 0.0
                    )
                    return q1, p1, q1, p1, q1, lp1, valid, cont, alpha, 1
                except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                    divergent = True
                    return q, p, q, p, q, -math.inf, 0, False, 0.0, 1
            qm, pm, qp, pp, qc, lpc, n, cont, alpha, na = build(
                q, p, direction, depth - 1
            )
            if not cont:
                return qm, pm, qp, pp, qc, lpc, n, cont, alpha, na
            if direction < 0:
                qm2, pm2, _, _, qc2, lpc2, n2, cont2, a2, na2 = build(
                    qm, pm, direction, depth - 1
                )
                qm, pm = qm2, pm2
            else:
                _, _, qp2, pp2, qc2, lpc2, n2, cont2, a2, na2 = build(
                    qp, pp, direction, depth - 1
                )
                qp, pp = qp2, pp2
            if n + n2 > 0 and rng.random() < n2 / (n + n2):
                qc, lpc = qc2, lpc2
            n += n2
            cont = cont2 and self._stop(qm, qp, pm, pp, inv_mass)
            return qm, pm, qp, pp, qc, lpc, n, cont, alpha + a2, na + na2

        for depth in range(self.max_tree_depth):
            direction = -1.0 if rng.random() < 0.5 else 1.0
            if direction < 0:
                qm, pm, _, _, qc, lpc, n, cont, a, _na = build(
                    q_minus, p_minus, direction, depth
                )
                q_minus, p_minus = qm, pm
            else:
                _, _, qp, pp, qc, lpc, n, cont, a, _na = build(
                    q_plus, p_plus, direction, depth
                )
                q_plus, p_plus = qp, pp
            if cont and n > 0 and rng.random() < n / max(n_valid + n, 1):
                candidate, candidate_logp = qc, lpc
            n_valid += n
            accepted_stat += a
            depth_reached = depth + 1
            keep_going = cont and self._stop(q_minus, q_plus, p_minus, p_plus, inv_mass)
            if not keep_going:
                break
        accepted = not np.array_equal(candidate, state.position)
        mean_accept = accepted_stat / max(leapfrogs, 1)
        ss = dict(state.sampler_state)
        if ss.get("adapt", False):
            t = int(ss.get("adapt_t", 0)) + 1
            eta = 1.0 / (t + 10.0)
            hbar = (1.0 - eta) * float(ss.get("hbar", 0.0)) + eta * (
                self.target_accept - mean_accept
            )
            log_eps = float(ss["mu"]) - math.sqrt(t) / 0.05 * hbar
            weight = t**-0.75
            log_bar = weight * log_eps + (1.0 - weight) * math.log(
                float(ss.get("step_size_bar", self.step_size))
            )
            ss.update(
                step_size=math.exp(log_eps),
                step_size_bar=math.exp(log_bar),
                hbar=hbar,
                adapt_t=t,
            )
        return StepResult(
            candidate,
            candidate_logp,
            accepted,
            ss,
            {
                "accept_stat": mean_accept,
                "divergent": divergent,
                "tree_depth": depth_reached,
                "leapfrog_steps": leapfrogs,
            },
        )


@dataclass(slots=True)
class GibbsSampler:
    updates: Mapping[int, ConditionalUpdate] | Sequence[ConditionalUpdate]
    random_scan: bool = False
    name: str = "gibbs"

    def initialize(self, position: Array, log_prob: float) -> Mapping[str, Any]:
        del log_prob
        if isinstance(self.updates, Mapping):
            indices = tuple(sorted(int(i) for i in self.updates))
        else:
            indices = tuple(range(len(self.updates)))
        if any(i < 0 or i >= len(position) for i in indices):
            raise ValueError("Gibbs update index is out of range")
        return {"indices": indices}

    def step(
        self, log_density: LogDensity, state: MCMCState, rng: np.random.Generator
    ) -> StepResult:
        q = state.position.copy()
        indices = list(state.sampler_state["indices"])
        if self.random_scan:
            rng.shuffle(indices)
        updates = self.updates
        for i in indices:
            update = updates[i]
            q[i] = float(update(q.copy(), rng))
        lp = _logp(log_density, q)
        if not np.isfinite(lp):
            raise RuntimeError(
                "A Gibbs conditional update produced a state outside the target support"
            )
        return StepResult(
            q, lp, True, state.sampler_state, {"components_updated": len(indices)}
        )


@dataclass(slots=True)
class ComponentMetropolis:
    scales: Array | float = 1.0
    name: str = "component-metropolis"

    def initialize(self, position: Array, log_prob: float) -> Mapping[str, Any]:
        del log_prob
        scales = np.asarray(self.scales, dtype=float)
        if scales.ndim == 0:
            scales = np.full(len(position), float(scales))
        if scales.shape != position.shape or np.any(scales <= 0):
            raise ValueError(
                "component scales must be positive and match the state dimension"
            )
        return {"scales": scales}

    def step(
        self, log_density: LogDensity, state: MCMCState, rng: np.random.Generator
    ) -> StepResult:
        q = state.position.copy()
        lp = state.log_prob
        accepted_components = 0
        scales = np.asarray(state.sampler_state["scales"], dtype=float)
        for i, scale in enumerate(scales):
            proposal = q.copy()
            proposal[i] += rng.normal(scale=scale)
            proposed = _logp(log_density, proposal)
            if math.log(rng.random()) < min(0.0, proposed - lp):
                q, lp = proposal, proposed
                accepted_components += 1
        return StepResult(
            q,
            lp,
            accepted_components > 0,
            state.sampler_state,
            {"accepted_components": accepted_components},
        )


def _generator(
    rng: np.random.Generator | int | None, saved_state: Mapping[str, Any] | None = None
) -> np.random.Generator:
    generator = np.random.default_rng(rng)
    if rng is None and saved_state is not None:
        generator.bit_generator.state = copy.deepcopy(dict(saved_state))
    return generator


def sample_mcmc(
    log_density: LogDensity,
    initial_position: Array,
    *,
    sampler: Sampler,
    draws: int = 1000,
    warmup: int = 500,
    thin: int = 1,
    rng: np.random.Generator | int | None = None,
    initial_state: MCMCState | None = None,
    parameter_names: Sequence[str] = (),
) -> MCMCChain:
    """Sample one resumable Markov chain.

    Warmup transitions update adaptation state but are not retained.  ``thin`` is
    implemented as transition spacing; all acceptance accounting includes the skipped
    transitions so diagnostics remain interpretable.
    """
    draws = _count(draws, name="draws", minimum=1)
    warmup = _count(warmup, name="warmup", minimum=0)
    thin = _count(thin, name="thin", minimum=1)
    position = _position(initial_position)
    if initial_state is None:
        lp = _logp(log_density, position)
        if not np.isfinite(lp):
            raise ValueError("initial_position has non-finite log density")
        sampler_state = sampler.initialize(position, lp)
        state = MCMCState(position, lp, sampler_state=sampler_state)
    else:
        if initial_state.position.shape != position.shape:
            raise ValueError("initial_state has incompatible dimension")
        state = initial_state
    generator = _generator(rng, state.rng_state)

    def transition(current: MCMCState) -> tuple[MCMCState, Mapping[str, Any]]:
        result = sampler.step(log_density, current, generator)
        next_state = MCMCState(
            result.position,
            result.log_prob,
            iteration=current.iteration + 1,
            accepted=current.accepted + int(result.accepted),
            proposed=current.proposed + 1,
            sampler_state=result.sampler_state,
            rng_state=generator.bit_generator.state,
        )
        return next_state, result.stats

    for _ in range(warmup):
        state, _ = transition(state)
    if "adapt" in state.sampler_state:
        frozen = dict(state.sampler_state)
        frozen["adapt"] = False
        if "step_size_bar" in frozen:
            frozen["step_size"] = frozen["step_size_bar"]
        state = replace(state, sampler_state=frozen)

    samples = np.empty((draws, len(position)), dtype=float)
    lps = np.empty(draws, dtype=float)
    stat_rows: list[Mapping[str, Any]] = []
    for draw in range(draws):
        stats: Mapping[str, Any] = {}
        for _ in range(thin):
            state, stats = transition(state)
        samples[draw] = state.position
        lps[draw] = state.log_prob
        stat_rows.append(stats)
    keys = sorted(set().union(*(row.keys() for row in stat_rows))) if stat_rows else []
    packed = {
        key: np.asarray([row.get(key, np.nan) for row in stat_rows]) for key in keys
    }
    return MCMCChain(
        samples, lps, state, sampler.name, packed, tuple(parameter_names), warmup
    )


def run_chains(
    log_density: LogDensity,
    initial_positions: Sequence[Array],
    *,
    sampler_factory: Callable[[], Sampler] | Sampler,
    draws: int = 1000,
    warmup: int = 500,
    thin: int = 1,
    rng: np.random.Generator | int | None = None,
    parameter_names: Sequence[str] = (),
) -> tuple[MCMCChain, ...]:
    """Run independent chains with reproducible child random streams."""
    positions = tuple(_position(p) for p in initial_positions)
    if len(positions) < 1:
        raise ValueError("At least one initial position is required")
    if isinstance(rng, np.random.Generator):
        entropy = rng.integers(0, 2**32, size=4, dtype=np.uint32)
        base = np.random.SeedSequence(entropy)
    elif (
        rng is None or isinstance(rng, (int, np.integer)) and not isinstance(rng, bool)
    ):
        base = np.random.SeedSequence(None if rng is None else int(rng))
    else:
        raise TypeError("rng must be None, an integer seed, or numpy.random.Generator")
    children = base.spawn(len(positions))
    chains = []
    for i, position in enumerate(positions):
        sampler = (
            sampler_factory()
            if callable(sampler_factory) and not hasattr(sampler_factory, "step")
            else copy.deepcopy(sampler_factory)
        )
        chains.append(
            sample_mcmc(
                log_density,
                position,
                sampler=sampler,
                draws=draws,
                warmup=warmup,
                thin=thin,
                rng=np.random.default_rng(children[i]),
                parameter_names=parameter_names,
            )
        )
    return tuple(chains)
