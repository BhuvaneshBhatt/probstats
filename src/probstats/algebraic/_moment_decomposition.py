"""Tensor decomposition and mixture recovery for moment statistics."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from ._dependencies import require_moment_tensoratlas
from ._moment_statistics import (
    MomentTensor,
    _indices,
    _tensoratlas_value,
    multi_view_moment,
)
from ._tolerances import (
    DECOMPOSITION_TOLERANCE,
    STOCHASTIC_TOLERANCE,
    validate_tolerance,
)
from ._validation import choice, exact_integer


@dataclass(frozen=True, slots=True)
class MomentDecompositionResult:
    """Tensor decomposition together with rank and uniqueness information."""

    decomposition: object
    rank: object
    identifiability: object | None
    symmetric: bool
    exact: bool
    complete: bool
    method: str


@dataclass(frozen=True, slots=True)
class MultiViewMixtureResult:
    """Recovered finite mixture from a multi-view cross-moment tensor."""

    weights: tuple[object, ...]
    component_means: tuple[tuple[tuple[object, ...], ...], ...]
    reconstructed: object
    residual: object
    components: int
    converged: bool
    exact: bool
    identifiable: bool | None
    certified_identifiable: bool
    method: str


def decompose_moment_tensor(
    tensor,
    components: int | None = None,
    *,
    symmetric: bool = True,
    method: str = "auto",
    tolerance: float = DECOMPOSITION_TOLERANCE,
    rng=None,
) -> MomentDecompositionResult:
    """Decompose a moment tensor using TensorAtlas CP/Waring machinery."""
    target = (
        None
        if components is None
        else exact_integer(components, name="components", minimum=0)
    )
    tolerance = validate_tolerance(tolerance)
    method = choice(method, name="method", allowed=("auto", "exact", "numerical"))
    backend = require_moment_tensoratlas()
    arr = (
        tensor.tensor
        if isinstance(tensor, MomentTensor)
        else _tensoratlas_value(tensor)
    )
    if symmetric:
        decomposition = backend.waring_decompose(
            arr, rank=target, method=method, tolerance=tolerance, rng=rng
        )
        rank_result = backend.symmetric_rank(
            arr,
            method="numerical" if method == "numerical" else "auto",
            tolerance=tolerance,
            rng=rng,
        )
    else:
        decomposition = backend.cp_decompose(
            arr, rank=target, method=method, tolerance=tolerance, rng=rng
        )
        rank_result = backend.tensor_rank(
            arr,
            method="numerical" if method == "numerical" else "auto",
            tolerance=tolerance,
            rng=rng,
        )
    target_rank = decomposition.rank
    shape = tuple(int(dim) for dim in decomposition.reconstructed.dimensions)
    uniqueness = (
        backend.generic_cp_identifiability(shape, target_rank)
        if len(shape) >= 3 and target_rank > 0
        else None
    )
    complete = bool(
        decomposition.exact
        or (decomposition.converged and float(decomposition.residual) <= tolerance)
    )
    return MomentDecompositionResult(
        decomposition,
        rank_result,
        uniqueness,
        symmetric,
        bool(decomposition.exact),
        complete,
        decomposition.method,
    )


def _normalized_component(weight, factors):
    scale = sp.sympify(weight)
    normalized = []
    for vector in factors:
        total = sp.simplify(sum(sp.sympify(value) for value in vector))
        if total.is_zero is not False:
            raise ValueError(
                "a recovered mixture component requires a provably nonzero mode sum"
            )
        scale = sp.simplify(scale * total)
        normalized.append(
            tuple(sp.simplify(sp.sympify(value) / total) for value in vector)
        )
    return sp.simplify(scale), tuple(normalized)


def _certified_or_numerically_nonnegative(value, *, tolerance: float) -> bool:
    item = sp.sympify(value)
    if item.is_nonnegative is True:
        return True
    return bool(item.is_number and item.is_real is True and float(item) >= -tolerance)


def _validate_probability_cross_moment(arr, *, tolerance: float) -> None:
    entries = tuple(
        sp.sympify(arr.component(index)) for index in _indices(arr.dimensions)
    )
    delta = sp.simplify(sum(entries) - 1)
    normalized = delta == 0 or bool(
        delta.is_number and delta.is_real is True and abs(float(delta)) <= tolerance
    )
    if not normalized:
        raise ValueError(
            "mixture recovery requires a cross-probability tensor whose entries sum to one"
        )
    if any(
        not _certified_or_numerically_nonnegative(value, tolerance=tolerance)
        for value in entries
    ):
        raise ValueError(
            "mixture recovery requires entries with certified nonnegative values"
        )


def _validate_stochastic_recovery(weights, components, *, tolerance: float) -> None:
    if any(
        not _certified_or_numerically_nonnegative(weight, tolerance=tolerance)
        for weight in weights
    ):
        raise ValueError(
            "the recovered CP decomposition has a negative or unresolved mixture weight"
        )
    for component in components:
        for vector in component:
            if any(
                not _certified_or_numerically_nonnegative(value, tolerance=tolerance)
                for value in vector
            ):
                raise ValueError(
                    "the recovered CP decomposition has a negative or unresolved component probability"
                )


def recover_multiview_mixture(
    tensor,
    components: int,
    *,
    method: str = "numerical",
    tolerance: float = STOCHASTIC_TOLERANCE,
    rng=None,
    restarts: int = 8,
) -> MultiViewMixtureResult:
    """Recover mixture weights and per-view component means from a cross moment.

    CP scale indeterminacy is converted into statistical normalization by
    making every recovered mode vector sum to one and absorbing those scale
    factors into the component weight.
    """
    target = exact_integer(components, name="components", minimum=1)
    restart_count = exact_integer(restarts, name="restarts", minimum=1)
    tolerance = validate_tolerance(tolerance)
    method = choice(method, name="method", allowed=("auto", "exact", "numerical"))
    backend = require_moment_tensoratlas()
    arr = (
        tensor.tensor
        if isinstance(tensor, MomentTensor)
        else _tensoratlas_value(tensor)
    )
    if not isinstance(arr, backend.TensorArray):
        arr = backend.TensorArray(arr)
    _validate_probability_cross_moment(arr, tolerance=tolerance)
    decomposition = backend.cp_decompose(
        arr,
        rank=target,
        method=method,
        nonnegative=method != "exact",
        tolerance=tolerance,
        rng=rng,
        restarts=restart_count,
    )
    normalized = tuple(
        _normalized_component(weight, factors)
        for weight, factors in zip(
            decomposition.weights, decomposition.factors, strict=True
        )
    )
    weights = tuple(item[0] for item in normalized)
    total = sp.simplify(sum(weights))
    if total.is_zero is not False:
        raise ValueError("recovered mixture weights must have a provably nonzero total")
    weights = tuple(sp.simplify(weight / total) for weight in weights)
    component_means = tuple(item[1] for item in normalized)
    _validate_stochastic_recovery(weights, component_means, tolerance=tolerance)
    # Stable ordering makes exact/smoke outputs reproducible up to CP permutation.
    order = tuple(
        sorted(
            range(target),
            key=lambda i: sp.default_sort_key((weights[i], component_means[i])),
        )
    )
    weights = tuple(weights[i] for i in order)
    component_means = tuple(component_means[i] for i in order)
    uniqueness = (
        backend.generic_cp_identifiability(arr.dimensions, target)
        if arr.rank >= 3
        else None
    )
    identified = None if uniqueness is None else uniqueness.identifiable
    exact = bool(decomposition.exact)
    converged = bool(
        decomposition.converged
        and (exact or float(decomposition.residual) <= tolerance)
    )
    return MultiViewMixtureResult(
        weights,
        component_means,
        decomposition.reconstructed,
        decomposition.residual,
        target,
        converged,
        exact,
        identified,
        bool(uniqueness is not None and uniqueness.certified),
        decomposition.method,
    )


def fit_multiview_mixture(
    views,
    components: int,
    *,
    weights=None,
    method: str = "numerical",
    tolerance: float = STOCHASTIC_TOLERANCE,
    rng=None,
    restarts: int = 8,
) -> MultiViewMixtureResult:
    """Construct the empirical cross moment and recover a multi-view mixture."""
    target = exact_integer(components, name="components", minimum=1)
    restart_count = exact_integer(restarts, name="restarts", minimum=1)
    tolerance = validate_tolerance(tolerance)
    method = choice(method, name="method", allowed=("auto", "exact", "numerical"))
    moment = multi_view_moment(views, weights=weights)
    return recover_multiview_mixture(
        moment,
        target,
        method=method,
        tolerance=tolerance,
        rng=rng,
        restarts=restart_count,
    )
