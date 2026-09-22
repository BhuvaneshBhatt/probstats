"""Statistical relations and explicit assumption contexts."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import sympy as sp

from .random_variables import RandomVariable


def _require_random_variable(value) -> RandomVariable:
    if not isinstance(value, RandomVariable):
        raise TypeError("statistical relations require RandomVariable objects")
    return value


def _canonical_pair(left, right):
    left = _require_random_variable(left)
    right = _require_random_variable(right)
    if left == right:
        raise ValueError("a binary statistical relation requires distinct variables")
    return tuple(sorted((left, right), key=lambda variable: variable.sort_key()))


def _canonical_collection(variables, *, minimum=2):
    variables = tuple(_require_random_variable(value) for value in variables)
    if len(variables) < minimum:
        raise ValueError(f"relation requires at least {minimum} random variables")
    if len(set(variables)) != len(variables):
        raise ValueError("relation variables must be distinct")
    return tuple(sorted(variables, key=lambda variable: variable.sort_key()))


def _canonical_groups(left, right):
    if isinstance(left, RandomVariable):
        left = (left,)
    if isinstance(right, RandomVariable):
        right = (right,)
    left = _canonical_collection(left, minimum=1)
    right = _canonical_collection(right, minimum=1)
    if set(left).intersection(right):
        raise ValueError("independent collections must be disjoint")
    groups = tuple(sorted((left, right), key=lambda g: tuple(v.sort_key() for v in g)))
    return groups


@dataclass(frozen=True, slots=True)
class StatisticalRelation:
    """Base class for immutable statistical propositions."""

    def __invert__(self):
        return NegatedStatisticalRelation(self)


@dataclass(frozen=True, slots=True)
class IndependentRelation(StatisticalRelation):
    variables: tuple[RandomVariable, RandomVariable]

    def __str__(self):
        return f"independent({self.variables[0]}, {self.variables[1]})"


@dataclass(frozen=True, slots=True)
class IndependentCollectionsRelation(StatisticalRelation):
    groups: tuple[tuple[RandomVariable, ...], tuple[RandomVariable, ...]]

    @property
    def variables(self):
        return self.groups[0] + self.groups[1]

    def __str__(self):
        left = ", ".join(map(str, self.groups[0]))
        right = ", ".join(map(str, self.groups[1]))
        return f"independent_collections(({left}), ({right}))"


@dataclass(frozen=True, slots=True)
class ConditionallyIndependentRelation(StatisticalRelation):
    variables: tuple[RandomVariable, RandomVariable]
    given: tuple[RandomVariable, ...]

    def __str__(self):
        given = ", ".join(map(str, self.given))
        return (
            f"conditionally_independent({self.variables[0]}, {self.variables[1]}, "
            f"given=({given}))"
        )


@dataclass(frozen=True, slots=True)
class ConditionallyIndependentCollectionsRelation(StatisticalRelation):
    groups: tuple[tuple[RandomVariable, ...], tuple[RandomVariable, ...]]
    given: tuple[RandomVariable, ...]

    @property
    def variables(self):
        return self.groups[0] + self.groups[1] + self.given

    def __str__(self):
        left = ", ".join(map(str, self.groups[0]))
        right = ", ".join(map(str, self.groups[1]))
        given = ", ".join(map(str, self.given))
        return f"conditionally_independent_collections(({left}), ({right}), given=({given}))"


@dataclass(frozen=True, slots=True)
class PairwiseIndependentRelation(StatisticalRelation):
    variables: tuple[RandomVariable, ...]

    def __str__(self):
        return f"pairwise_independent({', '.join(map(str, self.variables))})"


@dataclass(frozen=True, slots=True)
class MutuallyIndependentRelation(StatisticalRelation):
    variables: tuple[RandomVariable, ...]

    def __str__(self):
        return f"mutually_independent({', '.join(map(str, self.variables))})"


@dataclass(frozen=True, slots=True)
class IdenticallyDistributedRelation(StatisticalRelation):
    variables: tuple[RandomVariable, ...]

    def __str__(self):
        return f"identically_distributed({', '.join(map(str, self.variables))})"


@dataclass(frozen=True, slots=True)
class IIDRelation(StatisticalRelation):
    variables: tuple[RandomVariable, ...]

    def __str__(self):
        return f"iid({', '.join(map(str, self.variables))})"


@dataclass(frozen=True, slots=True)
class UncorrelatedRelation(StatisticalRelation):
    variables: tuple[RandomVariable, RandomVariable]

    def __str__(self):
        return f"uncorrelated({self.variables[0]}, {self.variables[1]})"


@dataclass(frozen=True, slots=True)
class FiniteMomentRelation(StatisticalRelation):
    variable: RandomVariable
    order: int

    @property
    def variables(self):
        return (self.variable,)

    def __post_init__(self):
        _require_random_variable(self.variable)
        if not isinstance(self.order, int):
            raise TypeError("moment order must be an integer")
        if self.order < 1:
            raise ValueError("finite-moment order must be positive")

    def __str__(self):
        return f"finite_moment({self.variable}, {self.order})"


@dataclass(frozen=True, slots=True)
class NegatedStatisticalRelation:
    """Explicitly asserted negaiton of a statistical proposition."""

    relation: StatisticalRelation

    def __post_init__(self):
        if not isinstance(self.relation, StatisticalRelation):
            raise TypeError("only statistical relations can be negated")

    def __invert__(self):
        return self.relation

    def __str__(self):
        return f"not {self.relation}"


def _as_context(value) -> StatisticalAssumptions:
    if isinstance(value, StatisticalAssumptions):
        return value
    if isinstance(value, (StatisticalRelation, NegatedStatisticalRelation)):
        return StatisticalAssumptions(value)
    try:
        return StatisticalAssumptions(*value)
    except TypeError as exc:
        raise TypeError(
            "assumptions must be StatisticalAssumptions, a statistical relation, "
            "or an iterable of relations"
        ) from exc


def _known_laws_compatible(variables) -> bool | None:
    laws = [variable.distribution for variable in variables]
    known = [law for law in laws if law is not None]
    if len(known) < 2:
        return None
    return all(law == known[0] for law in known[1:])


def _positive_implies(stated: StatisticalRelation, query: StatisticalRelation) -> bool:
    if stated == query:
        return True

    if isinstance(query, IndependentRelation):
        pair = frozenset(query.variables)
        if isinstance(
            stated,
            (PairwiseIndependentRelation, MutuallyIndependentRelation, IIDRelation),
        ):
            return pair.issubset(stated.variables)
        if isinstance(stated, IndependentCollectionsRelation):
            left, right = map(set, stated.groups)
            return (query.variables[0] in left and query.variables[1] in right) or (
                query.variables[1] in left and query.variables[0] in right
            )

    if isinstance(query, IndependentCollectionsRelation):
        left, right = map(set, query.groups)
        union = left | right
        if isinstance(stated, (MutuallyIndependentRelation, IIDRelation)):
            return union.issubset(stated.variables)
        if isinstance(stated, IndependentCollectionsRelation):
            sl, sr = map(set, stated.groups)
            return (left.issubset(sl) and right.issubset(sr)) or (
                left.issubset(sr) and right.issubset(sl)
            )

    if isinstance(query, ConditionallyIndependentRelation):
        qleft, qright = map(set, ((query.variables[0],), (query.variables[1],)))
        qgiven = set(query.given)
        if isinstance(stated, ConditionallyIndependentCollectionsRelation):
            sl, sr = map(set, stated.groups)
            sg = set(stated.given)
            same_orientation = qleft.issubset(sl) and qright.issubset(sr)
            reverse_orientation = qleft.issubset(sr) and qright.issubset(sl)
            if sg == qgiven and (same_orientation or reverse_orientation):
                return True
            # Weak union: X ⟂ (Y,W) | Z => X ⟂ Y | (Z,W).
            if sg.issubset(qgiven) and (same_orientation or reverse_orientation):
                moved = qgiven - sg
                source_other = sr if same_orientation else sl
                return moved.issubset(source_other)

    if isinstance(query, ConditionallyIndependentCollectionsRelation):
        ql, qr = map(set, query.groups)
        qg = set(query.given)
        if isinstance(stated, ConditionallyIndependentCollectionsRelation):
            sl, sr = map(set, stated.groups)
            sg = set(stated.given)
            same = ql.issubset(sl) and qr.issubset(sr)
            reverse = ql.issubset(sr) and qr.issubset(sl)
            if sg == qg and (same or reverse):
                return True
            # Weak union for collection-valued queries.
            if sg.issubset(qg) and (same or reverse):
                moved = qg - sg
                source_other = sr if same else sl
                return moved.issubset(source_other)

    if isinstance(query, PairwiseIndependentRelation):
        requested = set(query.variables)
        if isinstance(stated, (MutuallyIndependentRelation, IIDRelation)):
            return requested.issubset(stated.variables)

    if isinstance(query, MutuallyIndependentRelation) and isinstance(
        stated, IIDRelation
    ):
        return set(query.variables).issubset(stated.variables)

    if isinstance(query, IdenticallyDistributedRelation) and isinstance(
        stated, IIDRelation
    ):
        return set(query.variables).issubset(stated.variables)

    if isinstance(query, FiniteMomentRelation) and isinstance(
        stated, FiniteMomentRelation
    ):
        # A finite higher absolute moment implies all lower moments are finite.
        return stated.variable == query.variable and stated.order >= query.order

    return False


@dataclass(frozen=True, slots=True, init=False)
class StatisticalAssumptions:
    """Immutable collection of asserted statistical propositions.

    Queries are three-valued: ``True`` when a proposition is asserted or follows
    from a supported implication, ``False`` when its negation is asserted, and
    ``None`` when the context does not decide it.
    """

    _positive: frozenset[StatisticalRelation]
    _negative: frozenset[StatisticalRelation]

    def __init__(self, *relations):
        positive = set()
        negative = set()
        for item in relations:
            if isinstance(item, NegatedStatisticalRelation):
                negative.add(item.relation)
            elif isinstance(item, StatisticalRelation):
                positive.add(item)
            else:
                raise TypeError("assumptions must contain statistical relations")
        for relation in positive:
            if isinstance(relation, (IIDRelation, IdenticallyDistributedRelation)):
                compatible = _known_laws_compatible(relation.variables)
                if compatible is False:
                    raise ValueError(
                        "identically-distributed assumptions contradict attached laws"
                    )
        object.__setattr__(self, "_positive", frozenset(positive))
        object.__setattr__(self, "_negative", frozenset(negative))
        for relation in self._positive:
            if self._negative_refutes(relation, self._negative):
                raise ValueError(
                    f"contradictory statistical assumptions for {relation}"
                )
        for relation in self._negative:
            if self._positive_entails(relation, self._positive):
                raise ValueError(
                    f"contradictory statistical assumptions for {relation}"
                )

    @property
    def relations(self):
        return self._positive | frozenset(~relation for relation in self._negative)

    def assume(self, *relations):
        return StatisticalAssumptions(*self.relations, *relations)

    @classmethod
    def _positive_entails(cls, query, positive):
        if any(_positive_implies(stated, query) for stated in positive):
            return True
        if isinstance(query, PairwiseIndependentRelation):
            for pair in combinations(query.variables, 2):
                pair_relation = IndependentRelation(_canonical_pair(*pair))
                if not any(
                    _positive_implies(stated, pair_relation) for stated in positive
                ):
                    return False
            return True
        if isinstance(query, ConditionallyIndependentCollectionsRelation):
            qgroups = tuple(map(set, query.groups))
            qg = set(query.given)
            # Contraction: X ⟂ Y | Z and X ⟂ W | (Z,Y) => X ⟂ (Y,W) | Z.
            # Groups are stored canonically, so try both symmetric orientations.
            for ql, qr in (qgroups, tuple(reversed(qgroups))):
                for split_size in range(1, len(qr)):
                    ordered = tuple(sorted(qr, key=lambda v: v.sort_key()))
                    for first_tuple in combinations(ordered, split_size):
                        first = set(first_tuple)
                        second = qr - first
                        r1 = ConditionallyIndependentCollectionsRelation(
                            _canonical_groups(tuple(ql), tuple(first)),
                            tuple(sorted(qg, key=lambda v: v.sort_key())),
                        )
                        r2 = ConditionallyIndependentCollectionsRelation(
                            _canonical_groups(tuple(ql), tuple(second)),
                            tuple(sorted(qg | first, key=lambda v: v.sort_key())),
                        )
                        if cls._positive_entails(
                            r1, positive
                        ) and cls._positive_entails(r2, positive):
                            return True
        return False

    @staticmethod
    def _negative_refutes(query, negative):
        if query in negative:
            return True
        collection_types = (
            PairwiseIndependentRelation,
            MutuallyIndependentRelation,
            IdenticallyDistributedRelation,
            IIDRelation,
        )
        if isinstance(query, collection_types):
            variables = set(query.variables)
            for stated in negative:
                directly_refuting = isinstance(
                    stated, IndependentRelation
                ) and not isinstance(query, IdenticallyDistributedRelation)
                pairwise_refuting = isinstance(
                    stated, PairwiseIndependentRelation
                ) and not isinstance(query, IdenticallyDistributedRelation)
                iid_independence_refuting = isinstance(
                    query, IIDRelation
                ) and isinstance(stated, MutuallyIndependentRelation)
                iid_identity_refuting = isinstance(query, IIDRelation) and isinstance(
                    stated, IdenticallyDistributedRelation
                )
                if (
                    directly_refuting
                    or pairwise_refuting
                    or iid_independence_refuting
                    or iid_identity_refuting
                ) and set(stated.variables).issubset(variables):
                    return True
        return False

    def resolve(self, relation: StatisticalRelation) -> bool | None:
        if not isinstance(relation, StatisticalRelation):
            raise TypeError("resolve requires a statistical relation")
        if self._negative_refutes(relation, self._negative):
            return False
        if self._positive_entails(relation, self._positive):
            return True
        return None

    def __iter__(self):
        return iter(self.relations)

    def __len__(self):
        return len(self.relations)

    def __repr__(self):
        content = ", ".join(sorted(map(str, self.relations)))
        return f"StatisticalAssumptions({content})"


def independent(left, right, *, assumptions=None):
    """Construct or query pairwise independence."""
    relation = IndependentRelation(_canonical_pair(left, right))
    if assumptions is None:
        return relation
    return _as_context(assumptions).resolve(relation)


def independent_collections(left, right, *, assumptions=None):
    """Construct or query independence between two disjoint variable collections."""
    relation = IndependentCollectionsRelation(_canonical_groups(left, right))
    if assumptions is None:
        return relation
    return _as_context(assumptions).resolve(relation)


def conditionally_independent(left, right, *, given, assumptions=None):
    """Construct or query conditional independence."""
    pair = _canonical_pair(left, right)
    if isinstance(given, RandomVariable):
        given = (given,)
    given = _canonical_collection(given, minimum=1)
    if set(pair).intersection(given):
        raise ValueError("conditioned variables must be distinct from left and right")
    relation = ConditionallyIndependentRelation(pair, given)
    if assumptions is None:
        return relation
    return _as_context(assumptions).resolve(relation)


def conditionally_independent_collections(left, right, *, given, assumptions=None):
    """Construct or query conditional independence between variable collections.

    Collection-valued relations support symmetry, decomposition, weak union, and
    contraction (the semigraphoid axioms).
    """
    groups = _canonical_groups(left, right)
    if isinstance(given, RandomVariable):
        given = (given,)
    given = _canonical_collection(given, minimum=1)
    if set(groups[0] + groups[1]).intersection(given):
        raise ValueError("conditioning variables must be disjoint from both groups")
    relation = ConditionallyIndependentCollectionsRelation(groups, given)
    if assumptions is None:
        return relation
    return _as_context(assumptions).resolve(relation)


def pairwise_independent(*variables, assumptions=None):
    """Construct or query pairwise independence for a collection."""
    relation = PairwiseIndependentRelation(_canonical_collection(variables))
    if assumptions is None:
        return relation
    return _as_context(assumptions).resolve(relation)


def mutually_independent(*variables, assumptions=None):
    """Construct or query mutual independence for a collection."""
    relation = MutuallyIndependentRelation(_canonical_collection(variables))
    if assumptions is None:
        return relation
    return _as_context(assumptions).resolve(relation)


def identically_distributed(*variables, assumptions=None):
    """Construct or query equality in distribution for a collection."""
    relation = IdenticallyDistributedRelation(_canonical_collection(variables))
    compatible = _known_laws_compatible(relation.variables)
    if compatible is False:
        return False if assumptions is not None else relation
    if compatible is True and assumptions is not None:
        return True
    if assumptions is None:
        return relation
    return _as_context(assumptions).resolve(relation)


def iid(*variables, assumptions=None):
    """Construct or query an independent-and-identically-distributed relation."""
    relation = IIDRelation(_canonical_collection(variables))
    compatible = _known_laws_compatible(relation.variables)
    if compatible is False:
        if assumptions is None:
            raise ValueError("iid variables with attached laws must have equal laws")
        return False
    if assumptions is None:
        return relation
    return _as_context(assumptions).resolve(relation)


def uncorrelated(left, right, *, assumptions=None):
    """Construct or query zero-correlation structure."""
    relation = UncorrelatedRelation(_canonical_pair(left, right))
    if assumptions is None:
        return relation
    context = _as_context(assumptions)
    resolved = context.resolve(relation)
    if resolved is not None:
        return resolved
    pair = relation.variables
    if independent(*pair, assumptions=context) is True and all(
        finite_moment(v, 2, assumptions=context) is True for v in pair
    ):
        return True
    return None


def finite_moment(variable, order, *, assumptions=None):
    """Construct or query finiteness of a raw moment."""
    variable = _require_random_variable(variable)
    if not isinstance(order, int):
        raise TypeError("moment order must be an integer")
    relation = FiniteMomentRelation(variable, order)
    # Known package laws can often certify finiteness directly.
    if variable.distribution is not None:
        try:
            value = variable.distribution.moment(order)
        except (TypeError, ValueError, NotImplementedError):
            value = None
        if value is not None:
            finite = getattr(value, "is_finite", None)
            if finite is True:
                return relation if assumptions is None else True
            if finite is False or value is sp.nan:
                return relation if assumptions is None else False
    if assumptions is None:
        return relation
    return _as_context(assumptions).resolve(relation)


def finite_mean(variable, *, assumptions=None):
    """Construct or query finiteness of the first moment."""
    return finite_moment(variable, 1, assumptions=assumptions)


def finite_variance(variable, *, assumptions=None):
    """Construct or query finiteness of the second moment."""
    return finite_moment(variable, 2, assumptions=assumptions)


__all__ = [
    "ConditionallyIndependentCollectionsRelation",
    "ConditionallyIndependentRelation",
    "FiniteMomentRelation",
    "IIDRelation",
    "IdenticallyDistributedRelation",
    "IndependentCollectionsRelation",
    "IndependentRelation",
    "MutuallyIndependentRelation",
    "NegatedStatisticalRelation",
    "PairwiseIndependentRelation",
    "StatisticalAssumptions",
    "StatisticalRelation",
    "UncorrelatedRelation",
    "conditionally_independent",
    "conditionally_independent_collections",
    "finite_mean",
    "finite_moment",
    "finite_variance",
    "identically_distributed",
    "iid",
    "independent",
    "independent_collections",
    "mutually_independent",
    "pairwise_independent",
    "uncorrelated",
]
