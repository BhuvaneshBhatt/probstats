"""Semialgebraic null-hypothesis representations for algebraic statistics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

import sympy as sp

from ..models import AlgebraicModel


def _symbols(values: Sequence[sp.Symbol], *, name: str) -> tuple[sp.Symbol, ...]:
    result = tuple(values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    if any(not isinstance(value, sp.Symbol) for value in result):
        raise TypeError(f"{name} must contain only SymPy symbols")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique symbols")
    return result


def _polynomial(expr, variables: tuple[sp.Symbol, ...], *, name: str) -> sp.Expr:
    value = sp.expand(sp.sympify(expr))
    try:
        sp.Poly(value, *variables)
    except sp.PolynomialError as exc:
        raise ValueError(
            f"{name} must be polynomial in the hypothesis parameters"
        ) from exc
    extra = value.free_symbols.difference(variables)
    if extra:
        names = ", ".join(sorted(str(symbol) for symbol in extra))
        raise ValueError(f"{name} contains undeclared symbols: {names}")
    return value


def _semialgebraic_formula(
    value, variables: tuple[sp.Symbol, ...], *, name: str
) -> sp.Expr:
    formula = sp.sympify(value)
    if formula in (sp.true, sp.false):
        return formula
    if isinstance(formula, (sp.And, sp.Or, sp.Not)):
        for argument in formula.args:
            _semialgebraic_formula(argument, variables, name=name)
        return formula
    if isinstance(formula, sp.Equality):
        _polynomial(formula.lhs - formula.rhs, variables, name=name)
        return formula
    if isinstance(
        formula,
        (sp.LessThan, sp.StrictLessThan, sp.GreaterThan, sp.StrictGreaterThan),
    ):
        _polynomial(formula.lhs - formula.rhs, variables, name=name)
        return formula
    raise TypeError(f"{name} must be a Boolean combination of polynomial relations")


def _equality_polynomial(value, variables: tuple[sp.Symbol, ...]) -> sp.Expr:
    relation = sp.sympify(value)
    if isinstance(relation, sp.Equality):
        relation = relation.lhs - relation.rhs
    elif getattr(relation, "is_Relational", False):
        raise TypeError(
            "equality constraints must be equations or polynomial expressions"
        )
    return _polynomial(relation, variables, name="equality constraint")


def _inequality_polynomial(value, variables: tuple[sp.Symbol, ...]) -> sp.Expr:
    """Return f for a constraint represented canonically as f <= 0."""
    relation = sp.sympify(value)
    if isinstance(relation, (sp.LessThan, sp.StrictLessThan)):
        if isinstance(relation, sp.StrictLessThan):
            raise TypeError(
                "SDL null constraints must be closed non-strict inequalities"
            )
        expr = relation.lhs - relation.rhs
    elif isinstance(relation, (sp.GreaterThan, sp.StrictGreaterThan)):
        if isinstance(relation, sp.StrictGreaterThan):
            raise TypeError(
                "SDL null constraints must be closed non-strict inequalities"
            )
        expr = relation.rhs - relation.lhs
    elif isinstance(relation, sp.Equality):
        raise TypeError("put equality constraints in equalities, not inequalities")
    elif getattr(relation, "is_Relational", False):
        raise TypeError("unsupported relational constraint")
    else:
        # Bare expressions follow the SDL convention f(theta) <= 0.
        expr = relation
    return _polynomial(expr, variables, name="inequality constraint")


@dataclass(frozen=True, slots=True)
class BasicSemialgebraicNull:
    """One basic closed semialgebraic component of a null hypothesis.

    Equalities are retained as equalities for provenance. ``sdl_constraints``
    converts each equality ``g = 0`` to the pair ``g <= 0`` and ``-g <= 0``
    required by the SDL formulation, followed by the native inequalities.
    """

    parameters: tuple[sp.Symbol, ...]
    equalities: tuple[sp.Expr, ...] = ()
    inequalities: tuple[sp.Expr, ...] = ()
    label: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        parameters = _symbols(self.parameters, name="parameters")
        equalities = tuple(
            _equality_polynomial(item, parameters) for item in self.equalities
        )
        inequalities = tuple(
            _inequality_polynomial(item, parameters) for item in self.inequalities
        )
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "equalities", equalities)
        object.__setattr__(self, "inequalities", inequalities)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def formula(self) -> sp.Expr:
        """Return the exact SymPy formula defining this component."""
        atoms = [sp.Eq(expr, 0) for expr in self.equalities]
        atoms.extend(sp.Le(expr, 0) for expr in self.inequalities)
        return sp.And(*atoms)

    @property
    def sdl_constraints(self) -> tuple[sp.Expr, ...]:
        """Return polynomial f_j with the null represented as f_j <= 0."""
        equality_pairs = tuple(
            expr for equality in self.equalities for expr in (equality, -equality)
        )
        return (*equality_pairs, *self.inequalities)

    @property
    def sdl_formula(self) -> sp.Expr:
        """Return the equivalent all-inequality SDL null formula."""
        return sp.And(*(sp.Le(expr, 0) for expr in self.sdl_constraints))


@dataclass(frozen=True, slots=True)
class SemialgebraicHypothesis:
    """A semialgebraic null hypothesis embedded in an ambient parameter space.

    ``components`` represents a finite union of basic semialgebraic sets. The
    core SDL procedure applies to a single basic component; retaining union
    structure supports componentwise/intersection-union procedures while
    use the same hypothesis object without changing the model representation.
    """

    parameters: tuple[sp.Symbol, ...]
    components: tuple[BasicSemialgebraicNull, ...]
    ambient: sp.Expr = sp.true
    estimators: Mapping[sp.Symbol, object] = field(default_factory=dict, compare=False)
    sample_space: object | None = field(default=None, compare=False)
    metadata: Mapping[str, object] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        parameters = _symbols(self.parameters, name="parameters")
        components = tuple(self.components)
        if not components:
            raise ValueError(
                "a semialgebraic hypothesis requires at least one null component"
            )
        if any(component.parameters != parameters for component in components):
            raise ValueError(
                "all null components must use the hypothesis parameters in order"
            )
        ambient = _semialgebraic_formula(
            self.ambient, parameters, name="ambient formula"
        )
        estimators = dict(self.estimators)
        if not set(estimators).issubset(parameters):
            raise ValueError("estimator keys must be hypothesis parameters")
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "ambient", ambient)
        object.__setattr__(self, "estimators", MappingProxyType(estimators))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @classmethod
    def basic(
        cls,
        parameters: Sequence[sp.Symbol],
        *,
        equalities: Sequence[object] = (),
        inequalities: Sequence[object] = (),
        ambient=sp.true,
        estimators: Mapping[sp.Symbol, object] | None = None,
        sample_space=None,
        metadata: Mapping[str, object] | None = None,
    ) -> SemialgebraicHypothesis:
        """Construct a hypothesis with one basic semialgebraic null component."""
        params = tuple(parameters)
        component = BasicSemialgebraicNull(
            params, tuple(equalities), tuple(inequalities)
        )
        return cls(
            params,
            (component,),
            ambient=ambient,
            estimators={} if estimators is None else estimators,
            sample_space=sample_space,
            metadata={} if metadata is None else metadata,
        )

    @classmethod
    def from_model(
        cls,
        model: AlgebraicModel,
        *,
        estimators: Mapping[sp.Symbol, object] | None = None,
        sample_space=None,
        metadata: Mapping[str, object] | None = None,
    ) -> SemialgebraicHypothesis:
        """Construct the probability-model null without duplicating model semantics."""
        probabilities = model.probabilities
        equalities = (*model.equations, model.normalization)
        inequalities = tuple(-probability for probability in probabilities)
        inequalities += tuple(
            _model_inequality_as_le(item) for item in model.inequalities
        )
        combined_metadata = dict(model.metadata)
        if metadata is not None:
            combined_metadata.update(metadata)
        return cls.basic(
            probabilities,
            equalities=equalities,
            inequalities=inequalities,
            estimators=estimators,
            sample_space=sample_space,
            metadata=combined_metadata,
        )

    @property
    def is_basic(self) -> bool:
        """Whether the null is one basic semialgebraic set."""
        return len(self.components) == 1

    @property
    def null_formula(self) -> sp.Expr:
        """Return the null region, including the ambient parameter restriction."""
        return sp.And(
            self.ambient, sp.Or(*(component.formula for component in self.components))
        )

    @property
    def alternative_formula(self) -> sp.Expr:
        """Return the alternative as ambient minus the null region."""
        return sp.And(
            self.ambient, sp.Not(sp.Or(*(c.formula for c in self.components)))
        )

    @property
    def basic_null(self) -> BasicSemialgebraicNull:
        """Return the sole basic null component or reject a reducible union."""
        if not self.is_basic:
            raise ValueError("the hypothesis null is a union of basic components")
        return self.components[0]

    def semialgebraically_equivalent(self, other: SemialgebraicHypothesis) -> bool:
        """Decide exact equality of two null regions using semialg."""
        if self.parameters != other.parameters:
            return False
        try:
            from semialg.decision import equivalent
        except ImportError as exc:
            raise ImportError(
                "semialgebraic hypothesis equivalence requires semialg; "
                "install probstats[algebraic]"
            ) from exc
        return bool(equivalent(self.null_formula, other.null_formula, self.parameters))


def _model_inequality_as_le(value) -> sp.Expr:
    relation = sp.sympify(value)
    if isinstance(relation, sp.GreaterThan):
        return sp.expand(relation.rhs - relation.lhs)
    if isinstance(relation, sp.LessThan):
        return sp.expand(relation.lhs - relation.rhs)
    if isinstance(relation, (sp.StrictGreaterThan, sp.StrictLessThan)):
        raise TypeError("SDL null constraints must be closed non-strict inequalities")
    if getattr(relation, "is_Relational", False):
        raise TypeError("unsupported model inequality")
    # AlgebraicModel normalizes bare inequalities to expr >= 0.
    return -sp.expand(relation)
