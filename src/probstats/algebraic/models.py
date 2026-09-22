"""Algebraic statistical model objects and exact invariant queries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import sympy as sp

from ._dependencies import require_semialg
from ._validation import cardinalities


def _equation_expr(value) -> sp.Expr:
    expr = sp.sympify(value)
    if isinstance(expr, sp.Equality):
        expr = expr.lhs - expr.rhs
    return sp.expand(expr)


def _relation_expr(value):
    expr = sp.sympify(value)
    if getattr(expr, "is_Relational", False):
        return expr
    return sp.Ge(expr, 0)


@dataclass(frozen=True, slots=True)
class ModelIdeal:
    """Polynomial ideal attached to an algebraic statistical model."""

    variables: tuple[sp.Symbol, ...]
    generators: tuple[sp.Expr, ...]

    def dimension(self) -> int:
        """Return the exact affine dimension using semialg."""
        backend = require_semialg()
        return int(
            backend.analyze_equality_ideal(self.generators, self.variables).dimension
        )

    def degree(self) -> int:
        """Return the exact affine degree using semialg."""
        backend = require_semialg()
        return int(backend.ideal_degree(self.generators, self.variables))

    def singular_locus(self):
        """Return a formula defining the singular locus using semialg."""
        backend = require_semialg()
        return backend.singular_locus(self.generators, self.variables)


@dataclass(frozen=True, slots=True)
class AlgebraicModel:
    """Statistical model defined by polynomial equalities and inequalities.

    ``variety`` describes the Zariski/equality model, while ``region`` also
    enforces probability nonnegativity and any model inequalities.
    """

    probabilities: tuple[sp.Symbol, ...]
    equations: tuple[sp.Expr, ...] = ()
    inequalities: tuple[object, ...] = ()
    normalization: sp.Expr | None = None
    parameters: tuple[sp.Symbol, ...] = ()
    parameterization: Mapping[sp.Symbol, sp.Expr] | None = field(
        default=None, compare=False
    )
    metadata: Mapping[str, object] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        probabilities = tuple(self.probabilities)
        if not probabilities:
            raise ValueError("an algebraic model requires probability coordinates")
        if any(not isinstance(symbol, sp.Symbol) for symbol in probabilities):
            raise TypeError("probability coordinates must be SymPy symbols")
        if len(set(probabilities)) != len(probabilities):
            raise ValueError("probability coordinates must be unique")
        equations = tuple(_equation_expr(eq) for eq in self.equations)
        for equation in equations:
            try:
                sp.Poly(equation, *probabilities)
            except sp.PolynomialError as exc:
                raise ValueError(
                    "model equations must be polynomial in probability coordinates"
                ) from exc
        inequalities = tuple(_relation_expr(expr) for expr in self.inequalities)
        normalization = self.normalization
        if normalization is None:
            normalization = sp.Add(*probabilities) - 1
        else:
            normalization = _equation_expr(normalization)
        params = tuple(self.parameters)
        if any(not isinstance(symbol, sp.Symbol) for symbol in params) or len(
            set(params)
        ) != len(params):
            raise ValueError("parameters must be unique SymPy symbols")
        mapping = None
        if self.parameterization is not None:
            raw_mapping = dict(self.parameterization)
            if set(raw_mapping) != set(probabilities):
                raise ValueError(
                    "parameterization must define every probability coordinate exactly once"
                )
            mapping = MappingProxyType(
                {symbol: sp.sympify(raw_mapping[symbol]) for symbol in probabilities}
            )
        metadata = MappingProxyType(dict(self.metadata))
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "equations", equations)
        object.__setattr__(self, "inequalities", inequalities)
        object.__setattr__(self, "normalization", sp.expand(normalization))
        object.__setattr__(self, "parameters", params)
        object.__setattr__(self, "parameterization", mapping)
        object.__setattr__(self, "metadata", metadata)

    @property
    def variety(self):
        """Return the equality formula defining the affine statistical variety."""
        equations = (*self.equations, self.normalization)
        return sp.And(*(sp.Eq(eq, 0) for eq in equations))

    @property
    def region(self):
        """Return the semialgebraic probability model."""
        nonnegative = tuple(sp.Ge(symbol, 0) for symbol in self.probabilities)
        return sp.And(self.variety, *nonnegative, *self.inequalities)

    def ideal(self) -> ModelIdeal:
        """Return the polynomial equality ideal, including normalization."""
        return ModelIdeal(self.probabilities, (*self.equations, self.normalization))

    def dimension(self) -> int:
        """Return the exact affine model dimension."""
        return self.ideal().dimension()

    def degree(self) -> int:
        """Return the exact affine model degree."""
        return self.ideal().degree()

    def singular_locus(self):
        """Return the algebraic singular locus of the equality model."""
        return self.ideal().singular_locus()


@dataclass(frozen=True, slots=True)
class DiscreteAlgebraicModel(AlgebraicModel):
    """Algebraic model for a finite joint probability table."""

    cardinalities: tuple[int, ...] = ()
    variable_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        AlgebraicModel.__post_init__(self)
        dims = cardinalities(self.cardinalities)
        if sp.prod(dims) != len(self.probabilities):
            raise ValueError("cardinalities do not match the number of probabilities")
        names = tuple(self.variable_names) or tuple(f"X{i}" for i in range(len(dims)))
        if len(names) != len(dims) or len(set(names)) != len(names):
            raise ValueError("variable_names must be distinct and match cardinalities")
        object.__setattr__(self, "cardinalities", dims)
        object.__setattr__(self, "variable_names", names)


@dataclass(frozen=True, slots=True)
class ModelInvariants:
    """Exact algebraic invariants of a statistical model."""

    dimension: int
    degree: int
    singular_locus: object


def model_ideal(model: AlgebraicModel) -> ModelIdeal:
    """Return the polynomial equality ideal of ``model``."""
    return model.ideal()


def model_dimension(model: AlgebraicModel) -> int:
    """Return the exact affine dimension of ``model``."""
    return model.dimension()


def model_degree(model: AlgebraicModel) -> int:
    """Return the exact affine degree of ``model``."""
    return model.degree()


def singular_model_locus(model: AlgebraicModel):
    """Return the algebraic singular locus of ``model``."""
    return model.singular_locus()


def model_invariants(model: AlgebraicModel) -> ModelInvariants:
    """Return core exact algebraic invariants of ``model``."""
    ideal = model.ideal()
    return ModelInvariants(ideal.dimension(), ideal.degree(), ideal.singular_locus())
