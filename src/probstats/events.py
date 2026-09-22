"""First-class probability events.

Events keep probability queries separate from distribution formulas.  Scalar
queries accept SymPy sets or predicates; product events describe one event per
independent component.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import sympy as sp
from sympy.core.relational import Relational
from sympy.logic.boolalg import Boolean


def _is_algebraic_predicate(predicate: sp.Basic, variable: sp.Symbol) -> bool:
    """Return whether SymPy's univariate inequality solver is safe to use.

    Restrict automatic set conversion to rational/algebraic relations.  In
    particular, periodic/transcendental inequalities must not be collapsed to
    a single principal interval.
    """
    if predicate in (sp.true, sp.false):
        return True
    if isinstance(predicate, Relational):
        difference = sp.together(predicate.lhs - predicate.rhs)
        try:
            return bool(difference.is_rational_function(variable))
        except (AttributeError, TypeError):
            return False
    if isinstance(predicate, Boolean):
        return all(_is_algebraic_predicate(arg, variable) for arg in predicate.args)
    return False


class ProbabilityEvent:
    """Protocol-like base for events consumed by :func:`probstats.probability`."""

    def contains(self, value: Any) -> sp.Basic:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class SetEvent(ProbabilityEvent):
    """Event represented by a SymPy set."""

    set: sp.Set

    def __post_init__(self):
        if not isinstance(self.set, sp.Set):
            raise TypeError("set must be a SymPy Set")

    def contains(self, value: Any) -> sp.Basic:
        return sp.Contains(sp.sympify(value), self.set)


@dataclass(frozen=True, slots=True)
class PredicateEvent(ProbabilityEvent):
    """Scalar event represented by a Boolean predicate in one variable."""

    predicate: sp.Basic
    variable: sp.Symbol

    def __post_init__(self):
        pred = sp.sympify(self.predicate)
        var = sp.sympify(self.variable)
        if not isinstance(var, sp.Symbol):
            raise TypeError("variable must be a SymPy Symbol")
        if var not in pred.free_symbols and pred not in (sp.true, sp.false):
            raise ValueError("predicate does not depend on variable")
        object.__setattr__(self, "predicate", pred)
        object.__setattr__(self, "variable", var)

    def contains(self, value: Any) -> sp.Basic:
        return sp.sympify(self.predicate.subs(self.variable, sp.sympify(value)))

    def as_set(self) -> sp.Set:
        """Convert a univariate real predicate to a SymPy set when possible."""

        def convert(predicate):
            if predicate is sp.true:
                return sp.S.Reals
            if predicate is sp.false:
                return sp.S.EmptySet
            if isinstance(predicate, sp.And):
                return sp.Intersection(*(convert(arg) for arg in predicate.args))
            if isinstance(predicate, sp.Or):
                return sp.Union(*(convert(arg) for arg in predicate.args))
            if isinstance(predicate, sp.Not):
                return sp.Complement(sp.S.Reals, convert(predicate.args[0]))
            if not _is_algebraic_predicate(predicate, self.variable):
                return sp.ConditionSet(self.variable, predicate, sp.S.Reals)
            try:
                return sp.solve_univariate_inequality(
                    predicate, self.variable, relational=False
                )
            except (
                NotImplementedError,
                ValueError,
                TypeError,
                AttributeError,
                sp.PolynomialError,
            ):
                return sp.ConditionSet(self.variable, predicate, sp.S.Reals)

        return convert(self.predicate)


@dataclass(frozen=True, slots=True)
class ProductEvent(ProbabilityEvent):
    """Cartesian event with one component event per product component."""

    events: tuple[ProbabilityEvent, ...]

    def __init__(self, events: Sequence[Any]):
        object.__setattr__(self, "events", tuple(as_event(e) for e in events))
        if not self.events:
            raise ValueError("ProductEvent requires at least one component event")

    def contains(self, value: Any) -> sp.Basic:
        values = tuple(value)
        if len(values) != len(self.events):
            raise ValueError("product event value has wrong dimension")
        return sp.And(*(event.contains(x) for event, x in zip(self.events, values)))


def as_event(event: Any, *, variable: sp.Symbol | None = None) -> ProbabilityEvent:
    """Normalize a public event input to a first-class event object."""
    if isinstance(event, ProbabilityEvent):
        return event
    if isinstance(event, sp.Set):
        return SetEvent(event)
    obj = sp.sympify(event)
    if obj in (sp.true, sp.false) or isinstance(obj, (Relational, Boolean)):
        if variable is None:
            symbols = sorted(obj.free_symbols, key=sp.default_sort_key)
            if len(symbols) != 1:
                raise ValueError(
                    "a variable is required when an event predicate does not have exactly one free symbol"
                )
            variable = symbols[0]
        return PredicateEvent(obj, variable)
    raise TypeError("event must be a SymPy Set, Boolean predicate, or ProbabilityEvent")


__all__ = ["PredicateEvent", "ProbabilityEvent", "ProductEvent", "SetEvent", "as_event"]


def event_complement(event, *, variable=None):
    """Return the complement of a scalar probability event."""
    ev = as_event(event, variable=variable)
    if isinstance(ev, SetEvent):
        return SetEvent(sp.Complement(sp.S.Reals, ev.set))
    if isinstance(ev, PredicateEvent):
        return PredicateEvent(sp.Not(ev.predicate), ev.variable)
    raise TypeError("event complement currently requires a scalar event")


def _combine_scalar_events(events, *, operator):
    normalized = tuple(as_event(event) for event in events)
    if len(normalized) < 2:
        raise ValueError("at least two events are required")
    if all(isinstance(event, SetEvent) for event in normalized):
        sets = [event.set for event in normalized]
        combined = sp.Union(*sets) if operator == "union" else sp.Intersection(*sets)
        return SetEvent(combined)
    if all(isinstance(event, PredicateEvent) for event in normalized):
        variables = {event.variable for event in normalized}
        if len(variables) != 1:
            raise ValueError("predicate events must use the same variable")
        variable = next(iter(variables))
        predicates = [event.predicate for event in normalized]
        combined = sp.Or(*predicates) if operator == "union" else sp.And(*predicates)
        return PredicateEvent(combined, variable)
    raise TypeError("events must all be SetEvent or all be PredicateEvent")


def event_union(*events):
    """Return the union of compatible scalar events."""
    return _combine_scalar_events(events, operator="union")


def event_intersection(*events):
    """Return the intersection of compatible scalar events."""
    return _combine_scalar_events(events, operator="intersection")


__all__ += ["event_complement", "event_intersection", "event_union"]
