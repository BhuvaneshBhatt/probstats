"""Shape-aware event and parameter spaces for probability models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import sympy as sp


class MeasureType(str, Enum):
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    MIXED = "mixed"


class Space(ABC):
    """A symbolic set-like object that can emit membership constraints."""

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]: ...

    @property
    def rank(self) -> int:
        return len(self.shape)

    @abstractmethod
    def contains(self, value: Any) -> sp.Basic: ...

    def constraint(self, value: Any) -> sp.Basic:
        """Return the symbolic condition required for ``value`` to be in the space."""
        return sp.sympify(self.contains(value))

    def validate(self, value: Any, *, name: str = "value") -> sp.Basic:
        """Reject provably invalid values while permitting undecidable symbolic values."""
        condition = self.constraint(value)
        try:
            simplified = sp.simplify(condition)
        except (TypeError, ValueError, NotImplementedError, sp.PolynomialError):
            simplified = condition
        if simplified is sp.false or simplified is False:
            raise ValueError(
                f"{name} is outside {type(self).__name__}: required {condition}"
            )
        return simplified


class EventSpace(Space):
    """Shape-aware support of one random draw."""


@dataclass(frozen=True, slots=True)
class ScalarEventSpace(EventSpace):
    domain: sp.Set = sp.S.Reals

    def __post_init__(self) -> None:
        if not isinstance(self.domain, sp.Set):
            raise TypeError("domain must be a SymPy Set")

    @property
    def shape(self) -> tuple[()]:
        return ()

    def contains(self, value: Any) -> sp.Basic:
        return sp.Contains(sp.sympify(value), self.domain)


@dataclass(frozen=True, slots=True)
class RealSpace(ScalarEventSpace):
    def __init__(self) -> None:
        object.__setattr__(self, "domain", sp.S.Reals)


@dataclass(frozen=True, slots=True)
class PositiveRealSpace(ScalarEventSpace):
    def __init__(self) -> None:
        object.__setattr__(self, "domain", sp.Interval.open(0, sp.oo))

    def contains(self, value: Any) -> sp.Basic:
        return sp.sympify(value) > 0


@dataclass(frozen=True, slots=True)
class NonnegativeRealSpace(ScalarEventSpace):
    def __init__(self) -> None:
        object.__setattr__(self, "domain", sp.Interval(0, sp.oo))

    def contains(self, value: Any) -> sp.Basic:
        return sp.sympify(value) >= 0


@dataclass(frozen=True, slots=True)
class UnitIntervalSpace(ScalarEventSpace):
    open_left: bool = False
    open_right: bool = False

    def __init__(self, *, open_left: bool = False, open_right: bool = False) -> None:
        object.__setattr__(self, "open_left", bool(open_left))
        object.__setattr__(self, "open_right", bool(open_right))
        object.__setattr__(
            self,
            "domain",
            sp.Interval(0, 1, left_open=open_left, right_open=open_right),
        )

    def contains(self, value: Any) -> sp.Basic:
        x = sp.sympify(value)
        left = x > 0 if self.open_left else x >= 0
        right = x < 1 if self.open_right else x <= 1
        return sp.And(left, right)


@dataclass(frozen=True, slots=True)
class IntegerSpace(ScalarEventSpace):
    lower: sp.Expr | None = None
    upper: sp.Expr | None = None

    def __init__(self, lower: Any | None = None, upper: Any | None = None) -> None:
        lo = None if lower is None else sp.sympify(lower)
        hi = None if upper is None else sp.sympify(upper)
        object.__setattr__(self, "lower", lo)
        object.__setattr__(self, "upper", hi)
        domain: sp.Set = sp.S.Integers
        if lo is not None or hi is not None:
            domain = sp.Intersection(
                domain,
                sp.Interval(-sp.oo if lo is None else lo, sp.oo if hi is None else hi),
            )
        object.__setattr__(self, "domain", domain)


@dataclass(frozen=True, slots=True)
class NaturalNumberSpace(IntegerSpace):
    include_zero: bool = True

    def __init__(self, *, include_zero: bool = True, upper: Any | None = None) -> None:
        object.__setattr__(self, "include_zero", bool(include_zero))
        lo = sp.Integer(0 if include_zero else 1)
        IntegerSpace.__init__(self, lo, upper)


@dataclass(frozen=True, slots=True)
class VectorEventSpace(EventSpace):
    dimension: int
    coordinate_domain: sp.Set = sp.S.Reals
    constraint_expression: sp.Basic = sp.true
    coordinate_symbols: tuple[sp.Symbol, ...] = ()

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")
        if not isinstance(self.coordinate_domain, sp.Set):
            raise TypeError("coordinate_domain must be a SymPy Set")
        if not self.coordinate_symbols:
            object.__setattr__(
                self,
                "coordinate_symbols",
                sp.symbols(f"_x0:{self.dimension}", real=True),
            )
        elif len(self.coordinate_symbols) != self.dimension:
            raise ValueError("coordinate_symbols length must equal dimension")

    @property
    def shape(self) -> tuple[int]:
        return (self.dimension,)

    def _coerce(self, value: Any) -> tuple[sp.Expr, ...] | None:
        if isinstance(value, sp.MatrixBase):
            if value.rows == 1 or value.cols == 1:
                xs = tuple(value)
            else:
                return None
        else:
            try:
                xs = tuple(sp.sympify(v) for v in value)
            except (TypeError, ValueError):
                return None
        return xs if len(xs) == self.dimension else None

    def contains(self, value: Any) -> sp.Basic:
        xs = self._coerce(value)
        if xs is None:
            return sp.false
        coords = tuple(sp.Contains(x, self.coordinate_domain) for x in xs)
        relation = sp.sympify(self.constraint_expression).subs(
            dict(zip(self.coordinate_symbols, xs))
        )
        return sp.And(*coords, relation)


@dataclass(frozen=True, slots=True)
class PositiveVectorSpace(VectorEventSpace):
    def __init__(self, dimension: int, *, strict: bool = True) -> None:
        domain = sp.Interval.open(0, sp.oo) if strict else sp.Interval(0, sp.oo)
        VectorEventSpace.__init__(self, dimension, domain)


@dataclass(frozen=True, slots=True)
class OrderedVectorSpace(VectorEventSpace):
    strict: bool = True

    def __init__(self, dimension: int, *, strict: bool = True) -> None:
        xs = sp.symbols(f"_o0:{dimension}", real=True)
        relation = sp.And(
            *(
                (xs[i] < xs[i + 1]) if strict else (xs[i] <= xs[i + 1])
                for i in range(dimension - 1)
            )
        )
        object.__setattr__(self, "strict", bool(strict))
        VectorEventSpace.__init__(self, dimension, sp.S.Reals, relation, xs)


@dataclass(frozen=True, slots=True)
class SimplexEventSpace(VectorEventSpace):
    def __init__(self, dimension: int) -> None:
        xs = sp.symbols(f"_p0:{dimension}", real=True)
        VectorEventSpace.__init__(
            self,
            dimension,
            sp.Interval(0, 1),
            sp.Eq(sp.Add(*xs), 1),
            xs,
        )


@dataclass(frozen=True, slots=True)
class CountVectorEventSpace(VectorEventSpace):
    total: sp.Expr | None = None

    def __init__(self, dimension: int, total: Any | None = None) -> None:
        xs = sp.symbols(f"_n0:{dimension}", integer=True, nonnegative=True)
        total_expr = None if total is None else sp.sympify(total)
        relation = sp.true if total_expr is None else sp.Eq(sp.Add(*xs), total_expr)
        object.__setattr__(self, "total", total_expr)
        VectorEventSpace.__init__(self, dimension, sp.S.Naturals0, relation, xs)


@dataclass(frozen=True, slots=True)
class ProductEventSpace(EventSpace):
    """Cartesian product of possibly heterogeneous event spaces."""

    components: tuple[EventSpace, ...]

    def __init__(self, components: Sequence[EventSpace]) -> None:
        comps = tuple(components)
        if not comps:
            raise ValueError("ProductEventSpace requires at least one component")
        if not all(isinstance(space, EventSpace) for space in comps):
            raise TypeError("all components must be EventSpace instances")
        object.__setattr__(self, "components", comps)

    @property
    def shape(self) -> tuple[int, ...]:
        # A product draw is represented as one tuple entry per component; entries
        # may themselves be vectors or matrices.
        return (len(self.components),)

    def contains(self, value: Any) -> sp.Basic:
        try:
            values = tuple(value)
        except TypeError:
            return sp.false
        if len(values) != len(self.components):
            return sp.false
        return sp.And(
            *(space.constraint(item) for space, item in zip(self.components, values))
        )


@dataclass(frozen=True, slots=True)
class MatrixEventSpace(EventSpace):
    rows: int
    cols: int
    symmetric: bool = False
    positive_definite: bool = False
    positive_semidefinite: bool = False

    def __post_init__(self) -> None:
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("matrix dimensions must be positive")
        if (
            self.positive_definite or self.positive_semidefinite
        ) and self.rows != self.cols:
            raise ValueError("definiteness requires a square matrix")
        if self.positive_definite and self.positive_semidefinite:
            raise ValueError(
                "choose positive_definite or positive_semidefinite, not both"
            )

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.cols)

    def _coerce(self, value: Any) -> sp.MatrixExpr | sp.ImmutableDenseMatrix | None:
        if isinstance(value, sp.MatrixExpr):
            return value if value.shape == self.shape else None
        try:
            matrix = sp.ImmutableMatrix(value)
        except (TypeError, ValueError):
            return None
        return matrix if matrix.shape == self.shape else None

    @staticmethod
    def _matrix_predicate(matrix: sp.MatrixBase, *, definite: bool) -> sp.Basic:
        attr = (
            getattr(matrix, "is_positive_definite", None)
            if definite
            else getattr(matrix, "is_positive_semidefinite", None)
        )
        if attr is True:
            return sp.true
        if attr is False:
            return sp.false
        predicate = (
            sp.Q.positive_definite(matrix)
            if definite
            else sp.Q.positive_semidefinite(matrix)
        )
        return predicate

    def contains(self, value: Any) -> sp.Basic:
        matrix = self._coerce(value)
        if matrix is None:
            return sp.false
        conditions: list[sp.Basic] = []
        if self.symmetric:
            if isinstance(matrix, sp.MatrixBase):
                conditions.append(sp.true if matrix == matrix.T else sp.false)
            else:
                conditions.append(sp.Q.symmetric(matrix))
        if self.positive_definite:
            conditions.append(self._matrix_predicate(matrix, definite=True))
        if self.positive_semidefinite:
            conditions.append(self._matrix_predicate(matrix, definite=False))
        return sp.And(*conditions) if conditions else sp.true


@dataclass(frozen=True, slots=True)
class SymmetricMatrixSpace(MatrixEventSpace):
    def __init__(self, dimension: int) -> None:
        MatrixEventSpace.__init__(self, dimension, dimension, symmetric=True)


@dataclass(frozen=True, slots=True)
class PositiveDefiniteMatrixSpace(MatrixEventSpace):
    def __init__(self, dimension: int) -> None:
        MatrixEventSpace.__init__(
            self, dimension, dimension, symmetric=True, positive_definite=True
        )


@dataclass(frozen=True, slots=True)
class PositiveSemidefiniteMatrixSpace(MatrixEventSpace):
    def __init__(self, dimension: int) -> None:
        MatrixEventSpace.__init__(
            self, dimension, dimension, symmetric=True, positive_semidefinite=True
        )


@dataclass(frozen=True, slots=True)
class CorrelationMatrixSpace(MatrixEventSpace):
    dimension: int = 1

    def __init__(self, dimension: int) -> None:
        object.__setattr__(self, "dimension", dimension)
        MatrixEventSpace.__init__(
            self, dimension, dimension, symmetric=True, positive_definite=True
        )

    def contains(self, value: Any) -> sp.Basic:
        matrix = self._coerce(value)
        if matrix is None:
            return sp.false
        base = MatrixEventSpace.contains(self, matrix)
        diagonal = sp.And(*(sp.Eq(matrix[i, i], 1) for i in range(self.dimension)))
        return sp.And(base, diagonal)


@dataclass(frozen=True, slots=True)
class CorrelationCholeskySpace(MatrixEventSpace):
    """Lower-triangular Cholesky factors of positive-definite correlation matrices."""

    dimension: int = 1

    def __init__(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        object.__setattr__(self, "dimension", dimension)
        MatrixEventSpace.__init__(self, dimension, dimension)

    def contains(self, value: Any) -> sp.Basic:
        matrix = self._coerce(value)
        if matrix is None:
            return sp.false
        conditions: list[sp.Basic] = []
        for i in range(self.dimension):
            conditions.append(matrix[i, i] > 0)
            conditions.append(sp.Eq(sum(matrix[i, j] ** 2 for j in range(i + 1)), 1))
            for j in range(i + 1, self.dimension):
                conditions.append(sp.Eq(matrix[i, j], 0))
        return sp.And(*conditions)


ParameterRelation = Callable[[Mapping[str, Any]], sp.Basic]


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    name: str
    space: Space

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("parameter name must be nonempty")
        if not isinstance(self.space, Space):
            raise TypeError("parameter space must be a Space")


@dataclass(frozen=True, slots=True)
class ParameterSpace:
    """Named, possibly non-factorizing domain for distribution parameters."""

    specs: tuple[ParameterSpec, ...]
    relations: tuple[ParameterRelation, ...] = ()

    def __init__(
        self,
        specs: Sequence[ParameterSpec | tuple[str, Space]],
        *,
        relations: Sequence[ParameterRelation] = (),
    ) -> None:
        normalized = tuple(
            s if isinstance(s, ParameterSpec) else ParameterSpec(*s) for s in specs
        )
        names = [s.name for s in normalized]
        if len(names) != len(set(names)):
            raise ValueError("parameter names must be unique")
        object.__setattr__(self, "specs", normalized)
        object.__setattr__(self, "relations", tuple(relations))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.specs)

    def _mapping(self, values: Mapping[str, Any] | Sequence[Any]) -> dict[str, Any]:
        if isinstance(values, Mapping):
            missing = set(self.names) - set(values)
            extra = set(values) - set(self.names)
            if missing or extra:
                raise ValueError(
                    f"parameter keys mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
                )
            return {name: values[name] for name in self.names}
        seq = tuple(values)
        if len(seq) != len(self.specs):
            raise ValueError(f"expected {len(self.specs)} parameters, got {len(seq)}")
        return dict(zip(self.names, seq))

    def constraints(self, values: Mapping[str, Any] | Sequence[Any]) -> sp.Basic:
        mapping = self._mapping(values)
        components = [spec.space.constraint(mapping[spec.name]) for spec in self.specs]
        components.extend(sp.sympify(relation(mapping)) for relation in self.relations)
        return sp.And(*components)

    def contains(self, values: Mapping[str, Any] | Sequence[Any]) -> sp.Basic:
        return self.constraints(values)

    def validate(self, values: Mapping[str, Any] | Sequence[Any]) -> sp.Basic:
        condition = self.constraints(values)
        try:
            simplified = sp.simplify(condition)
        except (TypeError, ValueError, NotImplementedError, sp.PolynomialError):
            simplified = condition
        if simplified is sp.false or simplified is False:
            raise ValueError(f"invalid distribution parameters: required {condition}")
        return simplified


__all__ = [
    "CorrelationCholeskySpace",
    "CorrelationMatrixSpace",
    "CountVectorEventSpace",
    "EventSpace",
    "IntegerSpace",
    "MatrixEventSpace",
    "MeasureType",
    "NaturalNumberSpace",
    "NonnegativeRealSpace",
    "OrderedVectorSpace",
    "ParameterSpace",
    "ParameterSpec",
    "PositiveDefiniteMatrixSpace",
    "PositiveRealSpace",
    "PositiveSemidefiniteMatrixSpace",
    "PositiveVectorSpace",
    "ProductEventSpace",
    "RealSpace",
    "ScalarEventSpace",
    "SimplexEventSpace",
    "Space",
    "SymmetricMatrixSpace",
    "UnitIntervalSpace",
    "VectorEventSpace",
]
