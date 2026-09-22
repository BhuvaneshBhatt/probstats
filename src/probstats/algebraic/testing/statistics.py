"""Studentization and Hájek-projection estimates for SDL testing."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from probstats.u_statistics import UStatisticResult, incomplete_u_statistic

from ._validation import as_generator


@dataclass(frozen=True, slots=True)
class HajekProjectionEstimate:
    """Divide-and-conquer estimates of the first-order Hájek projection."""

    values: np.ndarray
    mean: np.ndarray
    variance: np.ndarray
    indices: tuple[int, ...]
    blocks_per_observation: int


@dataclass(frozen=True, slots=True)
class SDLStudentizationResult:
    """Incomplete U-statistic and variance terms entering the SDL statistic."""

    statistic: float
    coordinate_statistics: np.ndarray
    u_statistic: UStatisticResult
    hajek: HajekProjectionEstimate
    kernel_variance: np.ndarray
    variance: np.ndarray
    standard_error: np.ndarray
    alpha: float
    kernel_values: np.ndarray


def _vector(value: object) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return array.reshape(1)
    if array.ndim != 1:
        raise ValueError("SDL kernels must return a scalar or one-dimensional vector")
    return array


def _n1(value: int | None, n: int) -> int:
    if value is None:
        return n
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError("n1 must be an integer")
    value = int(value)
    if not 1 <= value <= n:
        raise ValueError("n1 must satisfy 1 <= n1 <= sample size")
    return value


def hajek_projection_estimate(
    sample: Sequence[object],
    kernel: Callable[..., object],
    *,
    order: int | None = None,
    n1: int | None = None,
    indices: Sequence[int] | None = None,
    rng: np.random.Generator | int | None = None,
    shuffle_blocks: bool = False,
) -> HajekProjectionEstimate:
    """Estimate ``g(X_i)=E[h(X_i,X_2,...,X_m)|X_i]`` by divide-and-conquer.

    For each selected observation, the other observations are partitioned into
    ``floor((n-1)/(m-1))`` disjoint blocks of size ``m-1`` and the kernel is
    averaged over those blocks, matching the SDL construction.  For an order-1
    kernel the projection is the kernel itself.
    """
    observations = tuple(sample)
    n = len(observations)
    if order is None:
        order = getattr(kernel, "order", None)
        if order is None:
            raise TypeError("order is required when the kernel has no declared order")
    if not isinstance(order, Integral) or isinstance(order, bool) or int(order) < 1:
        raise ValueError("kernel order must be a positive integer")
    m = int(order)
    if n < m:
        raise ValueError("sample size must be at least the kernel order")
    count = _n1(n1, n)
    if indices is None:
        selected = tuple(range(count))
    else:
        selected = tuple(int(i) for i in indices)
        if len(selected) != count:
            raise ValueError("indices must contain exactly n1 observations")
        if len(set(selected)) != len(selected) or any(
            i < 0 or i >= n for i in selected
        ):
            raise ValueError("indices must be distinct valid sample indices")
    generator = as_generator(rng)
    values: list[np.ndarray] = []
    blocks_per_observation = 1 if m == 1 else (n - 1) // (m - 1)
    if blocks_per_observation < 1:
        raise ValueError("sample is too small to estimate the Hájek projection")
    for i in selected:
        if m == 1:
            values.append(_vector(kernel(observations[i])))
            continue
        others = [j for j in range(n) if j != i]
        if shuffle_blocks:
            generator.shuffle(others)
        evaluations = []
        width = m - 1
        for k in range(blocks_per_observation):
            block = others[k * width : (k + 1) * width]
            evaluations.append(
                _vector(kernel(observations[i], *(observations[j] for j in block)))
            )
        values.append(np.mean(np.stack(evaluations), axis=0))
    matrix = np.stack(values)
    mean = np.mean(matrix, axis=0)
    # Eq. (2.8)/(2.2.5) uses 1/n1, not the sample-variance n1-1 denominator.
    variance = np.mean((matrix - mean) ** 2, axis=0)
    return HajekProjectionEstimate(
        matrix, mean, variance, selected, blocks_per_observation
    )


def sdl_studentize(
    sample: Sequence[object],
    kernel: Callable[..., object],
    *,
    budget: int,
    order: int | None = None,
    n1: int | None = None,
    projection_indices: Sequence[int] | None = None,
    rng: np.random.Generator | int | None = None,
    projection_rng: np.random.Generator | int | None = None,
    shuffle_projection_blocks: bool = False,
    u_result: UStatisticResult | None = None,
) -> SDLStudentizationResult:
    """Compute the studentized SDL maximum statistic and its variance estimates.

    ``alpha = n / budget`` follows the SDL definition, where ``budget`` is the
    Bernoulli selection budget N rather than the realized number of subsets.
    Supplying ``u_result`` reuses an already-realized incomplete U-statistic.
    """
    observations = tuple(sample)
    n = len(observations)
    if u_result is None:
        u_result = incomplete_u_statistic(
            observations,
            kernel,
            order=order,
            budget=budget,
            rng=rng,
            selection="bernoulli",
            return_result=True,
            retain_values=True,
        )
    if not u_result.randomized or u_result.subset_indices is None:
        raise ValueError("u_result must retain a randomized subset realization")
    if u_result.sample_size != n:
        raise ValueError("u_result sample size does not match sample")
    if u_result.requested_budget != budget:
        raise ValueError("u_result requested budget does not match budget")
    m = u_result.order
    u_value = _vector(u_result.value)
    if u_result.kernel_values is None:
        raise ValueError(
            "u_result must retain kernel evaluations; construct it with "
            "retain_values=True"
        )
    selected_values = np.stack([_vector(value) for value in u_result.kernel_values])
    kernel_variance = np.mean((selected_values - u_value) ** 2, axis=0)
    hajek = hajek_projection_estimate(
        observations,
        kernel,
        order=m,
        n1=n1,
        indices=projection_indices,
        rng=projection_rng if projection_rng is not None else rng,
        shuffle_blocks=shuffle_projection_blocks,
    )
    if hajek.variance.shape != kernel_variance.shape:
        raise ValueError("kernel output dimension changed between evaluations")
    alpha = n / budget
    variance = (m * m) * hajek.variance + alpha * kernel_variance
    standard_error = np.sqrt(variance)
    with np.errstate(divide="ignore", invalid="ignore"):
        coordinates = np.sqrt(n) * u_value / standard_error
    zero_error = standard_error == 0
    zero_numerator = u_value == 0
    # 0/0 is the degenerate null coordinate and contributes zero. A nonzero
    # numerator with zero estimated variance is genuinely infinite, with sign.
    coordinates = np.where(zero_error & zero_numerator, 0.0, coordinates)
    statistic = float(np.max(coordinates))
    return SDLStudentizationResult(
        statistic,
        coordinates,
        u_result,
        hajek,
        kernel_variance,
        variance,
        standard_error,
        alpha,
        selected_values,
    )
