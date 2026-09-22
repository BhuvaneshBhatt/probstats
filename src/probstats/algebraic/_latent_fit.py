"""Numerical fitting for finite latent-class models."""

from __future__ import annotations

from dataclasses import dataclass

from ._tolerances import FITTING_TOLERANCE, validate_tolerance
from ._validation import exact_integer


@dataclass(frozen=True, slots=True)
class LatentClassFitResult:
    """Numerical EM fit of a finite latent-class model."""

    weights: tuple[float, ...]
    conditional_probabilities: tuple[tuple[tuple[float, ...], ...], ...]
    log_likelihood: float
    iterations: int
    converged: bool
    method: str = "em"


def fit_latent_class(
    counts,
    latent_classes: int,
    *,
    max_iter: int = 500,
    tolerance: float = FITTING_TOLERANCE,
    rng=None,
) -> LatentClassFitResult:
    """Fit a latent-class model to a contingency table by EM."""
    classes = exact_integer(latent_classes, name="latent_classes", minimum=1)
    iterations_limit = exact_integer(max_iter, name="max_iter", minimum=1)
    tolerance = validate_tolerance(tolerance)
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("fit_latent_class requires NumPy") from exc
    data = np.asarray(counts, dtype=float)
    if (
        data.ndim < 2
        or np.any(data < 0)
        or not np.isfinite(data).all()
        or data.sum() <= 0
    ):
        raise ValueError(
            "counts must be a finite nonnegative contingency table with positive total"
        )
    generator = np.random.default_rng(rng)
    weights = generator.random(classes)
    weights /= weights.sum()
    factors = []
    for _h in range(classes):
        component = []
        for dim in data.shape:
            vec = generator.random(dim)
            vec /= vec.sum()
            component.append(vec)
        factors.append(component)
    indices = list(np.ndindex(data.shape))
    previous = float("-inf")
    converged = False
    for iteration in range(1, iterations_limit + 1):
        component_mass = np.empty((classes, *data.shape), dtype=float)
        for h in range(classes):
            component_mass[h] = weights[h]
            for axis in range(data.ndim):
                shape = [1] * data.ndim
                shape[axis] = data.shape[axis]
                component_mass[h] *= factors[h][axis].reshape(shape)
        mixture = component_mass.sum(axis=0)
        if np.any(mixture <= 0):
            raise ArithmeticError("EM encountered a zero fitted cell probability")
        responsibilities = component_mass / mixture
        expected = responsibilities * data
        class_counts = expected.reshape(classes, -1).sum(axis=1)
        weights = class_counts / data.sum()
        for h in range(classes):
            for axis in range(data.ndim):
                sum_axes = tuple(i for i in range(data.ndim) if i != axis)
                marginal = expected[h].sum(axis=sum_axes)
                if class_counts[h] > 0:
                    factors[h][axis] = marginal / class_counts[h]
        log_likelihood = float(
            sum(
                data[index] * np.log(mixture[index])
                for index in indices
                if data[index] > 0
            )
        )
        if abs(log_likelihood - previous) <= tolerance * max(1.0, abs(log_likelihood)):
            converged = True
            break
        previous = log_likelihood
    return LatentClassFitResult(
        tuple(float(x) for x in weights),
        tuple(
            tuple(tuple(float(x) for x in vec) for vec in component)
            for component in factors
        ),
        log_likelihood,
        iteration,
        converged,
    )
