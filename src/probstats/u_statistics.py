"""Complete and randomized incomplete U-statistics.

The routines in this module are distribution-agnostic.  Algebraic testing builds
polynomial kernels on top of them, but the U-statistic machinery is useful on its
own.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import combinations
from math import comb
from numbers import Integral

import numpy as np

__all__ = [
    "EmptySubsetSelectionError",
    "UStatisticResult",
    "incomplete_u_statistic",
    "u_statistic",
]


class EmptySubsetSelectionError(RuntimeError):
    """Raised when Bernoulli selection realizes zero U-statistic subsets."""


@dataclass(frozen=True, slots=True)
class UStatisticResult:
    """A computed U-statistic together with its sampling bookkeeping."""

    value: object
    order: int
    sample_size: int
    total_subsets: int
    evaluated_subsets: int
    randomized: bool
    requested_budget: int | None = None
    subset_indices: tuple[tuple[int, ...], ...] | None = None
    kernel_values: tuple[object, ...] | None = None


def _order(value: int) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError("kernel order must be an integer")
    value = int(value)
    if value < 1:
        raise ValueError("kernel order must be positive")
    return value


def _declared_dimension(kernel: Callable[..., object]) -> int | None:
    dimension = getattr(kernel, "dimension", None)
    if dimension is None:
        return None
    if (
        not isinstance(dimension, Integral)
        or isinstance(dimension, bool)
        or dimension < 1
    ):
        raise ValueError("kernel dimension must be a positive integer")
    return int(dimension)


def _evaluation_array(value: object, dimension: int | None) -> np.ndarray | None:
    array = np.asarray(value)
    if array.ndim == 0:
        if dimension not in (None, 1):
            raise ValueError(
                f"kernel declared dimension {dimension} but returned a scalar"
            )
        return None
    if array.ndim != 1:
        raise ValueError("kernel must return a scalar or one-dimensional vector")
    if dimension is not None and array.shape != (dimension,):
        raise ValueError(
            f"kernel declared dimension {dimension} but returned shape {array.shape}"
        )
    return array


def _mean_evaluations(values, *, dimension: int | None = None, retain: bool = False):
    iterator = iter(values)
    try:
        first = next(iterator)
    except StopIteration as exc:
        raise ValueError("at least one kernel evaluation is required") from exc

    kept = [first] if retain else None
    first_array = _evaluation_array(first, dimension)
    count = 1
    if first_array is None:
        total = first
        for value in iterator:
            if _evaluation_array(value, dimension) is not None:
                raise ValueError("kernel output changed from scalar to vector")
            if kept is not None:
                kept.append(value)
            total = total + value
            count += 1
        mean = total / count
    else:
        # Object arrays preserve exact SymPy arithmetic; numeric arrays accumulate
        # in at least floating precision so integer kernels do not truncate means.
        dtype = (
            object
            if first_array.dtype == object
            else np.result_type(first_array.dtype, float)
        )
        total = first_array.astype(dtype, copy=True)
        shape = first_array.shape
        for value in iterator:
            array = _evaluation_array(value, dimension)
            if array is None or array.shape != shape:
                raise ValueError("kernel output shape changed between evaluations")
            if kept is not None:
                kept.append(value)
            total += array
            count += 1
        mean = total / count
    return mean, None if kept is None else tuple(kept)


def u_statistic(
    sample: Sequence[object],
    kernel: Callable[..., object],
    *,
    order: int | None = None,
    return_result: bool = False,
):
    """Compute the complete U-statistic of a symmetric kernel.

    ``kernel`` is called as ``kernel(x1, ..., xm)`` on every ``m``-subset of the
    observations.  The function works with scalar and vector-valued kernels.
    """
    observations = tuple(sample)
    if order is None:
        order = getattr(kernel, "order", None)
        if order is None:
            raise TypeError("order is required when the kernel has no declared order")
    m = _order(order)
    n = len(observations)
    if n < m:
        raise ValueError("sample size must be at least the kernel order")
    total = comb(n, m)
    indices_iter = combinations(range(n), m)
    values = (
        kernel(*(observations[index] for index in indices)) for indices in indices_iter
    )
    value, _ = _mean_evaluations(values, dimension=_declared_dimension(kernel))
    if not return_result:
        return value
    return UStatisticResult(value, m, n, total, total, False)


def _unrank_combination(rank: int, n: int, m: int) -> tuple[int, ...]:
    """Return the lexicographic ``rank``-th m-subset of range(n)."""
    if rank < 0 or rank >= comb(n, m):
        raise ValueError("combination rank is out of range")
    result: list[int] = []
    start = 0
    remaining_rank = rank
    for position in range(m):
        remaining = m - position - 1
        for candidate in range(start, n):
            count = comb(n - candidate - 1, remaining) if remaining else 1
            if remaining_rank < count:
                result.append(candidate)
                start = candidate + 1
                break
            remaining_rank -= count
    return tuple(result)


def _sample_ranks(total: int, count: int, rng: np.random.Generator) -> tuple[int, ...]:
    if count == 0:
        return ()
    if total <= np.iinfo(np.int64).max:
        return tuple(int(x) for x in rng.choice(total, size=count, replace=False))
    # NumPy cannot draw from ranges beyond int64. Rejection sampling retains a
    # uniform sample without replacement and avoids materializing the range.
    selected: set[int] = set()
    nbytes = max(1, (total.bit_length() + 7) // 8)
    ceiling = 1 << (8 * nbytes)
    limit = ceiling - ceiling % total
    while len(selected) < count:
        raw = int.from_bytes(rng.bytes(nbytes), "little")
        if raw < limit:
            selected.add(raw % total)
    return tuple(selected)


def incomplete_u_statistic(
    sample: Sequence[object],
    kernel: Callable[..., object],
    *,
    order: int | None = None,
    budget: int,
    rng: np.random.Generator | int | None = None,
    selection: str = "bernoulli",
    return_result: bool = False,
    retain_values: bool = False,
):
    """Compute a randomized incomplete U-statistic.

    ``selection="bernoulli"`` implements the SDL construction exactly: every
    m-subset is selected independently with probability ``budget / C(n,m)``.
    Rather than enumerate all subsets, the implementation first draws the
    resulting Binomial subset count and then samples that many subset ranks
    uniformly without replacement; this has exactly the same distribution.

    ``selection="fixed"`` is a general-purpose fixed-computational-budget
    variant that samples exactly ``budget`` subsets without replacement.

    Bernoulli selection can realize zero subsets. In that case the statistic is
    undefined and :class:`EmptySubsetSelectionError` is raised; resampling would
    change the stated selection law. Set ``retain_values=True`` with
    ``return_result=True`` when downstream calculations need the exact kernel
    evaluations.
    """
    observations = tuple(sample)
    if order is None:
        order = getattr(kernel, "order", None)
        if order is None:
            raise TypeError("order is required when the kernel has no declared order")
    m = _order(order)
    n = len(observations)
    if n < m:
        raise ValueError("sample size must be at least the kernel order")
    if not isinstance(budget, Integral) or isinstance(budget, bool):
        raise TypeError("budget must be an integer")
    budget = int(budget)
    total = comb(n, m)
    if not 1 <= budget <= total:
        raise ValueError("budget must satisfy 1 <= budget <= C(n, order)")
    generator = np.random.default_rng(rng)
    if selection == "bernoulli":
        selected_count = int(generator.binomial(total, budget / total))
        if selected_count == 0:
            raise EmptySubsetSelectionError(
                "Bernoulli subset selection selected no kernel evaluations; "
                "increase budget or retry with a different random seed"
            )
    elif selection == "fixed":
        selected_count = budget
    else:
        raise ValueError("selection must be 'bernoulli' or 'fixed'")
    ranks = _sample_ranks(total, selected_count, generator)
    selected_indices = tuple(_unrank_combination(rank, n, m) for rank in ranks)
    evaluations = (
        kernel(*(observations[index] for index in indices))
        for indices in selected_indices
    )
    value, kernel_values = _mean_evaluations(
        evaluations, dimension=_declared_dimension(kernel), retain=retain_values
    )
    if not return_result:
        return value
    return UStatisticResult(
        value,
        m,
        n,
        total,
        selected_count,
        True,
        budget,
        selected_indices,
        kernel_values,
    )
