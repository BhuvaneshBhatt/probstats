"""Symbolic random variables and random-expression inspection."""

from __future__ import annotations

from collections.abc import Iterable

import sympy as sp


class RandomVariable(sp.Symbol):
    __slots__ = ("_distribution",)
    """A symbolic scalar random variable.

    Random variables participate directly in ordinary SymPy expressions while
    retaining a distinct type so statistical operators can distinguish random
    sources from deterministic symbols. Probability laws and dependency
    information are supplied separately through distribution and
    statistical-assumption contexts.
    """

    def __new__(cls, name, distribution=None, **assumptions):
        if isinstance(name, sp.Symbol):
            if assumptions:
                raise TypeError(
                    "assumptions cannot be supplied when constructing from a Symbol"
                )
            assumptions = {
                key: value
                for key, value in name.assumptions0.items()
                if key != "commutative"
            }
            name = name.name
        if not isinstance(name, str) or not name:
            raise TypeError("random-variable name must be a nonempty string or Symbol")
        obj = sp.Symbol.__xnew__(cls, name, **assumptions)
        if distribution is not None:
            from .distributions.base import Distribution

            if not isinstance(distribution, Distribution):
                raise TypeError("distribution must be a probstats Distribution")
        obj._distribution = distribution
        return obj

    @property
    def distribution(self):
        """Return the attached probability law, or ``None`` when unspecified."""
        return self._distribution

    def _hashable_content(self):
        # The declared law is part of random-variable identity. Two variables may
        # share a printed name while representing different stochastic objects.
        return super()._hashable_content() + (self._distribution,)


def random_variables(expression) -> frozenset[RandomVariable]:
    """Return the random variables occurring in a symbolic expression."""
    expression = sp.sympify(expression)
    return frozenset(expression.atoms(RandomVariable))


def depends_on(
    expression, variables: Iterable[RandomVariable] | RandomVariable
) -> bool:
    """Return whether ``expression`` depends on any supplied random variable."""
    if isinstance(variables, RandomVariable):
        variables = (variables,)
    requested = frozenset(variables)
    if not all(isinstance(variable, RandomVariable) for variable in requested):
        raise TypeError("variables must contain only RandomVariable objects")
    return bool(random_variables(expression).intersection(requested))


__all__ = ["RandomVariable", "depends_on", "random_variables"]
