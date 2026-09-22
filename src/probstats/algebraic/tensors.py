"""Probability tensors and exact tensor-independence tests."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import sympy as sp

from ._dependencies import require_tensoratlas


def _prob_array(value) -> sp.ImmutableDenseNDimArray:
    arr = sp.Array(value)
    if len(arr.shape) < 1:
        raise ValueError("a probability tensor requires at least one variable")
    return sp.ImmutableDenseNDimArray(arr)


@dataclass(frozen=True, slots=True)
class ProbabilityValidationResult:
    """Validation state for a finite probability tensor."""

    normalized: bool | None
    nonnegative: bool | None

    @property
    def valid(self) -> bool | None:
        """Return ``True``/``False`` when validity is decidable, else ``None``."""
        if self.normalized is False or self.nonnegative is False:
            return False
        if self.normalized is True and self.nonnegative is True:
            return True
        return None


@dataclass(frozen=True, slots=True)
class ProbabilityTensor:
    """Joint finite probability distribution represented as a tensor."""

    values: object
    variable_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        values = _prob_array(self.values)
        names = tuple(self.variable_names) or tuple(
            f"X{i}" for i in range(len(values.shape))
        )
        if len(names) != len(values.shape) or len(set(names)) != len(names):
            raise ValueError("variable_names must be distinct and match tensor rank")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "variable_names", names)

    @property
    def cardinalities(self) -> tuple[int, ...]:
        return tuple(int(dim) for dim in self.values.shape)

    @property
    def tensor(self):
        """Return the TensorAtlas representation of this probability tensor."""
        backend = require_tensoratlas()
        return backend.TensorArray(
            self.values.tolist(), properties={"kind": "probability_tensor"}
        )

    def is_normalized(self) -> bool | None:
        """Return whether entries sum to one, or ``None`` if undecidable."""
        total = sp.simplify(sum(self.values[index] for index in self._indices()))
        if total == 1:
            return True
        if total.is_number:
            return False
        equals = sp.simplify(total - 1)
        return True if equals == 0 else None

    def is_nonnegative(self) -> bool | None:
        """Return exact nonnegativity when SymPy can decide every entry."""
        unknown = False
        for index in self._indices():
            value = sp.sympify(self.values[index])
            if value.is_nonnegative is False:
                return False
            if value.is_nonnegative is None:
                unknown = True
        return None if unknown else True

    def validate(self) -> ProbabilityValidationResult:
        """Return normalization and nonnegativity evidence without coercing unknowns."""
        return ProbabilityValidationResult(self.is_normalized(), self.is_nonnegative())

    def require_probability(self) -> ProbabilityTensor:
        """Return ``self`` when it is provably a probability tensor.

        Raises
        ------
        ValueError
            If normalization or nonnegativity is false or undecidable.  Query-style
            methods such as :meth:`is_normalized` retain three-valued semantics; this
            method is for constructive statistical operations requiring a proven
            probability object.
        """
        result = self.validate()
        if result.valid is True:
            return self
        if result.valid is False:
            raise ValueError("tensor entries must be nonnegative and sum to one")
        unresolved = []
        if result.normalized is None:
            unresolved.append("normalization")
        if result.nonnegative is None:
            unresolved.append("nonnegativity")
        raise ValueError(
            "probability validity is undecidable: " + ", ".join(unresolved)
        )

    def marginal(self, *variables: str) -> ProbabilityTensor:
        """Return the marginal probability tensor over requested variables."""
        if not variables:
            raise ValueError(
                "a probability-tensor marginal must retain at least one variable"
            )
        keep = tuple(self.variable_names.index(name) for name in variables)
        if len(set(keep)) != len(keep):
            raise ValueError("marginal variables must be distinct")
        shape = tuple(self.cardinalities[axis] for axis in keep)
        values = []
        for target in product(*(range(dim) for dim in shape)):
            total = 0
            for source in self._indices():
                if all(
                    source[axis] == value
                    for axis, value in zip(keep, target, strict=True)
                ):
                    total += self.values[source]
            values.append(sp.simplify(total))
        array = sp.ImmutableDenseNDimArray(values, shape)
        return ProbabilityTensor(array, tuple(variables))

    def flatten(self, row_variables, col_variables=None) -> sp.Matrix:
        """Return a matrix flattening grouped by statistical variables."""
        backend = require_tensoratlas()
        rows = tuple(self.variable_names.index(name) for name in row_variables)
        cols = None
        if col_variables is not None:
            cols = tuple(self.variable_names.index(name) for name in col_variables)
        return backend.grouped_flatten(self.tensor, rows, cols)

    def _indices(self):
        return product(*(range(dim) for dim in self.cardinalities))


@dataclass(frozen=True, slots=True)
class TensorIndependentResult:
    """Exact result for complete independence of a probability tensor."""

    independent: bool | None
    constraints: tuple[sp.Expr, ...]
    residuals: tuple[sp.Expr, ...]


def probability_tensor(values, *, variables=()) -> ProbabilityTensor:
    """Construct a :class:`ProbabilityTensor` from joint probabilities."""
    return ProbabilityTensor(values, tuple(variables))


def tensor_independent(tensor: ProbabilityTensor) -> TensorIndependentResult:
    """Test complete independence using the Segre rank-one equations."""
    if len(tensor.cardinalities) < 2:
        return TensorIndependentResult(True, (), ())
    backend = require_tensoratlas()
    equations = backend.segre_equations(tensor.cardinalities, prefix="p")
    mapping = {
        sp.Symbol("p_" + "_".join(map(str, index))): tensor.values[index]
        for index in tensor._indices()
    }
    residuals = tuple(sp.simplify(eq.subs(mapping)) for eq in equations)
    if all(value == 0 for value in residuals):
        independent = True
    elif any(value.is_zero is False for value in residuals):
        independent = False
    else:
        independent = None
    return TensorIndependentResult(independent, equations, residuals)
