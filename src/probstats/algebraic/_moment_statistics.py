"""Moment tensors and tensor-decomposition methods for multivariate statistics."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import factorial

import sympy as sp

from ._dependencies import require_moment_tensoratlas
from ._validation import cardinalities as _cardinalities
from ._validation import exact_integer


def _matrix_rows(samples) -> tuple[tuple[sp.Expr, ...], ...]:
    rows = tuple(tuple(sp.sympify(value) for value in row) for row in samples)
    if not rows:
        raise ValueError("samples must contain at least one observation")
    width = len(rows[0])
    if width < 1 or any(len(row) != width for row in rows):
        raise ValueError("samples must be a nonempty rectangular observation matrix")
    return rows


def _weights(count: int, weights) -> tuple[sp.Expr, ...]:
    if weights is None:
        return tuple(sp.Rational(1, count) for _ in range(count))
    values = tuple(sp.sympify(value) for value in weights)
    if len(values) != count:
        raise ValueError("weights must match the number of observations")
    if any(value.is_nonnegative is not True for value in values):
        raise ValueError("weights must have certified nonnegative values")
    total = sp.simplify(sum(values))
    if total.is_positive is not True:
        raise ValueError("weights must have certified positive total")
    return tuple(sp.simplify(value / total) for value in values)


def _array_from_entries(entries, shape):
    return sp.ImmutableDenseNDimArray(tuple(entries), tuple(shape))


def _tensoratlas_value(value):
    if isinstance(value, sp.NDimArray):
        return value.tolist()
    return value


def _indices(shape):
    return product(*(range(dim) for dim in shape))


@dataclass(frozen=True, slots=True)
class MomentTensor:
    """Finite-sample tensor statistic with exact symbolic entries."""

    values: sp.ImmutableDenseNDimArray
    order: int
    kind: str
    sample_size: int
    centered: bool = False
    view_dimensions: tuple[int, ...] = ()

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(dim) for dim in self.values.shape)

    @property
    def tensor(self):
        """Return the TensorAtlas representation lazily."""
        backend = require_moment_tensoratlas()
        return backend.TensorArray(
            self.values.tolist(), properties={"kind": self.kind, "order": self.order}
        )


def moment_tensor(samples, order: int, *, weights=None) -> MomentTensor:
    """Return the empirical raw moment tensor ``E[X^{⊗ order}]``.

    Entries are computed with exact SymPy arithmetic when the observations and
    optional weights are exact.
    """
    rows = _matrix_rows(samples)
    degree = exact_integer(order, name="order", minimum=1)
    dim = len(rows[0])
    probs = _weights(len(rows), weights)
    shape = (dim,) * degree
    entries = []
    for index in _indices(shape):
        entries.append(
            sp.simplify(
                sum(
                    weight * sp.prod(row[axis] for axis in index)
                    for row, weight in zip(rows, probs, strict=True)
                )
            )
        )
    return MomentTensor(
        _array_from_entries(entries, shape),
        degree,
        "raw_moment",
        len(rows),
        False,
        shape,
    )


def central_moment_tensor(samples, order: int, *, weights=None) -> MomentTensor:
    """Return ``E[(X-E[X])^{⊗ order}]`` for aligned multivariate samples."""
    rows = _matrix_rows(samples)
    probs = _weights(len(rows), weights)
    means = tuple(
        sp.simplify(
            sum(weight * row[j] for row, weight in zip(rows, probs, strict=True))
        )
        for j in range(len(rows[0]))
    )
    centered = tuple(
        tuple(sp.simplify(value - means[j]) for j, value in enumerate(row))
        for row in rows
    )
    result = moment_tensor(centered, order, weights=probs)
    return MomentTensor(
        result.values,
        result.order,
        "central_moment",
        result.sample_size,
        True,
        result.view_dimensions,
    )


def _set_partitions(items: tuple[int, ...]):
    if not items:
        yield ()
        return
    first, *rest = items
    for partition in _set_partitions(tuple(rest)):
        yield ((first,), *partition)
        for i in range(len(partition)):
            block = tuple(sorted((first, *partition[i])))
            yield (*partition[:i], block, *partition[i + 1 :])


def cumulant_tensor(
    samples, order: int, *, weights=None, max_order: int = 6
) -> MomentTensor:
    """Return the empirical joint cumulant tensor using the partition formula.

    The exact set-partition formula grows combinatorially, so orders above
    ``max_order`` require explicitly increasing the guard.
    """
    rows = _matrix_rows(samples)
    degree = exact_integer(order, name="order", minimum=1)
    order_limit = exact_integer(max_order, name="max_order", minimum=1)
    if degree > order_limit:
        raise ValueError(f"order {degree} exceeds max_order={order_limit}")
    probs = _weights(len(rows), weights)
    dim = len(rows[0])
    shape = (dim,) * degree
    partitions = tuple(_set_partitions(tuple(range(degree))))
    entries = []
    for index in _indices(shape):
        total = 0
        for partition in partitions:
            coefficient = (-1) ** (len(partition) - 1) * factorial(len(partition) - 1)
            term = sp.Integer(coefficient)
            for block in partition:
                term *= sp.simplify(
                    sum(
                        weight * sp.prod(row[index[position]] for position in block)
                        for row, weight in zip(rows, probs, strict=True)
                    )
                )
            total += term
        entries.append(sp.simplify(total))
    return MomentTensor(
        _array_from_entries(entries, shape),
        degree,
        "cumulant",
        len(rows),
        degree > 1,
        shape,
    )


def multi_view_moment(views, *, weights=None) -> MomentTensor:
    """Return ``E[X1 ⊗ ... ⊗ Xk]`` from aligned multi-view observations."""
    matrices = tuple(_matrix_rows(view) for view in views)
    if len(matrices) < 2:
        raise ValueError("multi_view_moment requires at least two views")
    count = len(matrices[0])
    if any(len(matrix) != count for matrix in matrices):
        raise ValueError(
            "all views must contain the same number of aligned observations"
        )
    probs = _weights(count, weights)
    dims = tuple(len(matrix[0]) for matrix in matrices)
    entries = []
    for index in _indices(dims):
        entries.append(
            sp.simplify(
                sum(
                    weight
                    * sp.prod(
                        matrices[mode][sample][index[mode]]
                        for mode in range(len(matrices))
                    )
                    for sample, weight in enumerate(probs)
                )
            )
        )
    return MomentTensor(
        _array_from_entries(entries, dims),
        len(matrices),
        "multi_view_moment",
        count,
        False,
        dims,
    )


def categorical_multi_view_moment(
    observations, *, cardinalities=None, weights=None
) -> MomentTensor:
    """Return the joint one-hot cross moment of aligned categorical views.

    ``observations`` is a sequence of views, each containing integer category
    labels for the same observations.  The resulting tensor is the empirical
    joint probability table and therefore has nonnegative entries summing to
    one.
    """
    raw_views = tuple(tuple(value for value in view) for view in observations)
    if len(raw_views) < 2 or not raw_views[0]:
        raise ValueError(
            "categorical_multi_view_moment requires at least two nonempty views"
        )
    count = len(raw_views[0])
    if any(len(view) != count for view in raw_views):
        raise ValueError(
            "all categorical views must contain the same number of observations"
        )
    encoded: list[tuple[tuple[sp.Integer, ...], ...]] = []
    if cardinalities is None:
        dims = []
        for view in raw_views:
            labels = tuple(
                exact_integer(value, name="categorical label", minimum=0)
                for value in view
            )
            dims.append(max(labels) + 1)
    else:
        dims = list(_cardinalities(cardinalities, minimum_variables=len(raw_views)))
        if len(dims) != len(raw_views) or any(dim < 1 for dim in dims):
            raise ValueError(
                "cardinalities must be positive and match the number of views"
            )
    for view, dim in zip(raw_views, dims, strict=True):
        labels = tuple(
            exact_integer(value, name="categorical label", minimum=0) for value in view
        )
        if any(label < 0 or label >= dim for label in labels):
            raise ValueError("categorical label is outside its declared cardinality")
        encoded.append(
            tuple(tuple(sp.Integer(i == label) for i in range(dim)) for label in labels)
        )
    result = multi_view_moment(tuple(encoded), weights=weights)
    return MomentTensor(
        result.values,
        result.order,
        "categorical_multi_view_moment",
        result.sample_size,
        False,
        result.view_dimensions,
    )
