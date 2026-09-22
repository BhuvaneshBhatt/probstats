"""Composable probability distributions.

This module contains structural distributions whose probability laws are built
from other distributions rather than introduced as unrelated named formulas.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Any

import sympy as sp

from ._exact_linear_algebra import exact_solve
from .distributions.base import Distribution
from .distributions.basic import MultivariateNormal
from .functionals import (
    ProbabilityFunctionalError,
    cdf,
    density,
    mean,
    variance,
)
from .spaces import MeasureType, ProductEventSpace, ScalarEventSpace, VectorEventSpace


def _as_weights(weights: Sequence[Any], n: int) -> tuple[sp.Expr, ...]:
    values = tuple(sp.sympify(w) for w in weights)
    if len(values) != n or n == 0:
        raise ValueError("weights must have one entry per component")
    condition = sp.And(*(w >= 0 for w in values), sp.Eq(sp.Add(*values), 1))
    if sp.simplify(condition) is sp.false:
        raise ValueError("mixture weights must be nonnegative and sum to one")
    return values


def _indicator(condition: sp.Basic) -> sp.Expr:
    return sp.Piecewise((sp.S.One, condition), (sp.S.Zero, True))


@dataclass(frozen=True, slots=True)
class ProductDistribution(Distribution):
    """Joint distribution of mutually independent, heterogeneous components."""

    components: tuple[Distribution, ...]

    def __init__(self, components: Sequence[Distribution]):
        comps = tuple(components)
        if not comps:
            raise ValueError("ProductDistribution requires at least one component")
        if not all(isinstance(d, Distribution) for d in comps):
            raise TypeError("all components must be Distribution instances")
        object.__setattr__(self, "components", comps)

    @property
    def support(self):
        supports = [d.support for d in self.components]
        if all(isinstance(s, sp.Set) for s in supports):
            return sp.ProductSet(*supports)
        return sp.S.UniversalSet

    @property
    def event_space(self):
        return ProductEventSpace(tuple(d.event_space for d in self.components))

    @property
    def measure_type(self):
        kinds = {d.measure_type for d in self.components}
        return kinds.pop() if len(kinds) == 1 else MeasureType.MIXED

    @property
    def parameters(self):
        return tuple(p for d in self.components for p in d.parameters)

    @property
    def parameter_constraints(self):
        return sp.And(*(d.parameter_constraints for d in self.components))

    def pdf(self, value):
        values = tuple(value)
        if len(values) != len(self.components):
            raise ValueError("product value has wrong number of components")
        return sp.simplify(
            reduce(
                mul, (density(d, x) for d, x in zip(self.components, values)), sp.S.One
            )
        )

    def logpdf(self, value):
        return sp.log(self.pdf(value))

    def pmf(self, value):
        if self.measure_type is not MeasureType.DISCRETE:
            return super().pmf(value)
        return self.pdf(value)

    def logpmf(self, value):
        if self.measure_type is not MeasureType.DISCRETE:
            return super().logpmf(value)
        return sp.log(self.pmf(value))

    def _mean(self):
        return tuple(mean(d) for d in self.components)

    def marginal(self, indices: int | Sequence[int]):
        return MarginalDistribution(self, indices)

    def condition(self, indices: int | Sequence[int], values: Any):
        return ConditionalDistribution(self, indices, values)


@dataclass(frozen=True, slots=True)
class IndependentDistribution(Distribution):
    """Repeated iid draws from one base distribution."""

    base: Distribution
    count: int

    def __post_init__(self):
        if not isinstance(self.base, Distribution):
            raise TypeError("base must be a Distribution")
        if not isinstance(self.count, int) or self.count <= 0:
            raise ValueError("count must be a positive integer")

    @property
    def components(self):
        return (self.base,) * self.count

    @property
    def support(self):
        return sp.ProductSet(*([self.base.support] * self.count))

    @property
    def event_space(self):
        if isinstance(self.base.event_space, ScalarEventSpace):
            return VectorEventSpace(self.count, self.base.support)
        return ProductEventSpace(self.components_event_spaces)

    @property
    def components_event_spaces(self):
        return tuple(self.base.event_space for _ in range(self.count))

    @property
    def measure_type(self):
        return self.base.measure_type

    @property
    def parameters(self):
        return self.base.parameters

    @property
    def parameter_constraints(self):
        return self.base.parameter_constraints

    def pdf(self, value):
        values = tuple(value)
        if len(values) != self.count:
            raise ValueError("iid value has wrong dimension")
        return sp.simplify(
            reduce(mul, (density(self.base, x) for x in values), sp.S.One)
        )

    def logpdf(self, value):
        return sp.log(self.pdf(value))

    def pmf(self, value):
        if self.measure_type is not MeasureType.DISCRETE:
            return super().pmf(value)
        return self.pdf(value)

    def logpmf(self, value):
        if self.measure_type is not MeasureType.DISCRETE:
            return super().logpmf(value)
        return sp.log(self.pmf(value))

    def _mean(self):
        return sp.ImmutableMatrix([mean(self.base)] * self.count)

    def _variance(self):
        v = variance(self.base)
        return sp.diag(*([v] * self.count))

    def marginal(self, indices: int | Sequence[int]):
        return MarginalDistribution(self, indices)


@dataclass(frozen=True, slots=True)
class TruncatedDistribution(Distribution):
    """A scalar distribution conditioned to a subset of its support."""

    base: Distribution
    truncation: sp.Set

    def __post_init__(self):
        if not isinstance(self.base, Distribution):
            raise TypeError("base must be a Distribution")
        if not isinstance(self.base.event_space, ScalarEventSpace):
            raise TypeError("TruncatedDistribution requires a scalar base")
        if not isinstance(self.truncation, sp.Set):
            raise TypeError("truncation must be a SymPy Set")
        if self.support is sp.S.EmptySet:
            raise ValueError("truncation does not overlap the base support")
        z = self.normalizing_constant
        if z.is_zero is True or z.is_negative is True:
            raise ValueError("truncation event must have positive probability")

    @property
    def support(self):
        return sp.Intersection(self.base.support, self.truncation)

    @property
    def event_space(self):
        return ScalarEventSpace(self.support)

    @property
    def measure_type(self):
        return self.base.measure_type

    @property
    def parameters(self):
        return self.base.parameters

    @property
    def parameter_constraints(self):
        return self.base.parameter_constraints

    @property
    def normalizing_constant(self):
        support = self.support
        x = sp.Dummy("x", real=True)
        if isinstance(support, sp.FiniteSet):
            return sp.simplify(sum(density(self.base, k) for k in support))
        if isinstance(support, sp.Interval):
            lo, hi = support.start, support.end
            if self.measure_type is MeasureType.DISCRETE:
                k = sp.Dummy("k", integer=True)
                return sp.simplify(
                    sp.summation(self.base.pmf(k), (k, sp.ceiling(lo), sp.floor(hi)))
                )
            return sp.simplify(sp.integrate(self.base.pdf(x), (x, lo, hi)))
        raise ProbabilityFunctionalError(
            "truncation normalization requires an interval or finite set"
        )

    def pdf(self, value):
        x = sp.sympify(value)
        return sp.simplify(
            self.base.pdf(x)
            * _indicator(sp.Contains(x, self.support))
            / self.normalizing_constant
        )

    def pmf(self, value):
        if self.measure_type is not MeasureType.DISCRETE:
            return super().pmf(value)
        x = sp.sympify(value)
        return sp.simplify(
            self.base.pmf(x)
            * _indicator(sp.Contains(x, self.support))
            / self.normalizing_constant
        )

    def logpdf(self, value):
        return sp.log(self.pdf(value))

    def logpmf(self, value):
        return sp.log(self.pmf(value))

    def _cdf(self, value):
        z = sp.sympify(value)
        if not isinstance(self.support, sp.Interval):
            return NotImplemented
        lo, hi = self.support.start, self.support.end
        base_lo = cdf(self.base, lo)
        numerator = cdf(self.base, sp.Min(z, hi)) - base_lo
        return sp.Piecewise(
            (0, z < lo),
            (1, z >= hi),
            (sp.simplify(numerator / self.normalizing_constant), True),
        )


@dataclass(frozen=True, slots=True)
class MixtureDistribution(Distribution):
    """Finite mixture of component distributions sharing an event space."""

    components: tuple[Distribution, ...]
    weights: tuple[sp.Expr, ...]

    def __init__(self, components: Sequence[Distribution], weights: Sequence[Any]):
        comps = tuple(components)
        if not comps:
            raise ValueError("MixtureDistribution requires components")
        if not all(isinstance(d, Distribution) for d in comps):
            raise TypeError("all components must be Distribution instances")
        spaces = tuple(d.event_space for d in comps)
        if any(space.shape != spaces[0].shape for space in spaces[1:]):
            raise ValueError("mixture components must have compatible event shapes")
        measures = {d.measure_type for d in comps}
        if len(measures) != 1:
            raise ValueError("mixture components must share one measure type")
        object.__setattr__(self, "components", comps)
        object.__setattr__(self, "weights", _as_weights(weights, len(comps)))

    @property
    def support(self):
        supports = [d.support for d in self.components]
        if all(isinstance(s, sp.Set) for s in supports):
            return sp.Union(*supports)
        return sp.S.UniversalSet

    @property
    def event_space(self):
        first = self.components[0].event_space
        if isinstance(first, ScalarEventSpace):
            return ScalarEventSpace(self.support)
        return first

    @property
    def measure_type(self):
        kinds = {d.measure_type for d in self.components}
        return kinds.pop() if len(kinds) == 1 else MeasureType.MIXED

    @property
    def parameters(self):
        return (*self.weights, *(p for d in self.components for p in d.parameters))

    @property
    def parameter_constraints(self):
        return sp.And(
            *(w >= 0 for w in self.weights),
            sp.Eq(sp.Add(*self.weights), 1),
            *(d.parameter_constraints for d in self.components),
        )

    def pdf(self, value):
        return sp.simplify(
            sp.Add(
                *(w * density(d, value) for w, d in zip(self.weights, self.components))
            )
        )

    def logpdf(self, value):
        return sp.log(self.pdf(value))

    def pmf(self, value):
        if self.measure_type is not MeasureType.DISCRETE:
            return super().pmf(value)
        return self.pdf(value)

    def logpmf(self, value):
        return sp.log(self.pmf(value))

    def _cdf(self, value):
        if self.measure_type is MeasureType.MIXED:
            return NotImplemented
        return sp.simplify(
            sp.Add(*(w * cdf(d, value) for w, d in zip(self.weights, self.components)))
        )

    def _mean(self):
        return sp.simplify(
            sp.Add(*(w * mean(d) for w, d in zip(self.weights, self.components)))
        )

    def _variance(self):
        mu = self._mean()
        return sp.simplify(
            sp.Add(
                *(
                    w * (variance(d) + (mean(d) - mu) ** 2)
                    for w, d in zip(self.weights, self.components)
                )
            )
        )


@dataclass(frozen=True, slots=True)
class ConditionalDistribution(Distribution):
    """Conditional law of independent product components given fixed components.

    For a general dependent symbolic joint law, use ``condition_distribution``;
    this class represents the exact structural conditional of a
    ProductDistribution/IndependentDistribution without inventing dependence.
    """

    joint: Distribution
    conditioned_indices: tuple[int, ...]
    conditioned_values: tuple[Any, ...]

    def __init__(self, joint, indices: int | Sequence[int], values: Any):
        if not isinstance(
            joint, (ProductDistribution, IndependentDistribution, MultivariateNormal)
        ):
            raise TypeError(
                "ConditionalDistribution supports independent products and MultivariateNormal"
            )
        idx = (indices,) if isinstance(indices, int) else tuple(indices)
        vals = (values,) if len(idx) == 1 else tuple(values)
        if len(idx) != len(vals):
            raise ValueError("conditioned indices and values differ in length")
        n = (
            len(joint.components)
            if isinstance(joint, (ProductDistribution, IndependentDistribution))
            else joint.dimension
        )
        if len(set(idx)) != len(idx) or any(i < 0 or i >= n for i in idx):
            raise IndexError("conditioned component index out of range or repeated")
        if isinstance(joint, (ProductDistribution, IndependentDistribution)):
            for i, value in zip(idx, vals):
                joint.components[i].event_space.validate(
                    value, name=f"conditioned value {i}"
                )
        else:
            for i, value in zip(idx, vals):
                ScalarEventSpace(sp.S.Reals).validate(
                    value, name=f"conditioned value {i}"
                )
        object.__setattr__(self, "joint", joint)
        object.__setattr__(self, "conditioned_indices", idx)
        object.__setattr__(self, "conditioned_values", vals)

    @property
    def remaining_indices(self):
        n = (
            len(self.joint.components)
            if isinstance(self.joint, (ProductDistribution, IndependentDistribution))
            else self.joint.dimension
        )
        return tuple(i for i in range(n) if i not in self.conditioned_indices)

    @property
    def distribution(self):
        if isinstance(self.joint, (ProductDistribution, IndependentDistribution)):
            remaining = tuple(self.joint.components[i] for i in self.remaining_indices)
            if len(remaining) == 1:
                return remaining[0]
            return ProductDistribution(remaining)
        # Exact block-Gaussian conditioning.
        r = self.remaining_indices
        c = self.conditioned_indices
        mu = self.joint.mean
        sigma = self.joint.covariance
        mu_r = sp.ImmutableMatrix([mu[i, 0] for i in r])
        mu_c = sp.ImmutableMatrix([mu[i, 0] for i in c])
        observed = sp.ImmutableMatrix(list(map(sp.sympify, self.conditioned_values)))
        s_rr = sp.ImmutableMatrix([[sigma[i, j] for j in r] for i in r])
        s_rc = sp.ImmutableMatrix([[sigma[i, j] for j in c] for i in r])
        s_cr = sp.ImmutableMatrix([[sigma[i, j] for j in r] for i in c])
        s_cc = sp.ImmutableMatrix([[sigma[i, j] for j in c] for i in c])
        cond_mean = sp.ImmutableMatrix(mu_r + s_rc * exact_solve(s_cc, observed - mu_c))
        cond_cov = sp.ImmutableMatrix(s_rr - s_rc * exact_solve(s_cc, s_cr))
        if len(r) == 1:
            from .distributions.exponential_family import Normal

            return Normal(cond_mean[0], sp.sqrt(cond_cov[0, 0]))
        return MultivariateNormal(cond_mean, cond_cov)

    @property
    def support(self):
        return self.distribution.support

    @property
    def event_space(self):
        return self.distribution.event_space

    @property
    def measure_type(self):
        return self.distribution.measure_type

    @property
    def parameters(self):
        return self.distribution.parameters

    @property
    def parameter_constraints(self):
        return self.distribution.parameter_constraints

    def pdf(self, value):
        return self.distribution.pdf(value)

    def logpdf(self, value):
        return self.distribution.logpdf(value)

    def pmf(self, value):
        return self.distribution.pmf(value)

    def logpmf(self, value):
        return self.distribution.logpmf(value)


@dataclass(frozen=True, slots=True)
class MarginalDistribution(Distribution):
    """Marginal of selected components of an independent product joint."""

    joint: Distribution
    indices: tuple[int, ...]

    def __init__(self, joint, indices: int | Sequence[int]):
        if not isinstance(
            joint, (ProductDistribution, IndependentDistribution, MultivariateNormal)
        ):
            raise TypeError(
                "MarginalDistribution supports independent products and MultivariateNormal"
            )
        idx = (indices,) if isinstance(indices, int) else tuple(indices)
        n = (
            len(joint.components)
            if isinstance(joint, (ProductDistribution, IndependentDistribution))
            else joint.dimension
        )
        if not idx or len(set(idx)) != len(idx) or any(i < 0 or i >= n for i in idx):
            raise IndexError("marginal component index out of range or repeated")
        object.__setattr__(self, "joint", joint)
        object.__setattr__(self, "indices", idx)

    @property
    def distribution(self):
        if isinstance(self.joint, (ProductDistribution, IndependentDistribution)):
            selected = tuple(self.joint.components[i] for i in self.indices)
            if len(selected) == 1:
                return selected[0]
            return ProductDistribution(selected)
        idx = self.indices
        mu = sp.ImmutableMatrix([self.joint.mean[i, 0] for i in idx])
        cov = sp.ImmutableMatrix(
            [[self.joint.covariance[i, j] for j in idx] for i in idx]
        )
        if len(idx) == 1:
            from .distributions.exponential_family import Normal

            return Normal(mu[0], sp.sqrt(cov[0, 0]))
        return MultivariateNormal(mu, cov)

    @property
    def support(self):
        return self.distribution.support

    @property
    def event_space(self):
        return self.distribution.event_space

    @property
    def measure_type(self):
        return self.distribution.measure_type

    @property
    def parameters(self):
        return self.distribution.parameters

    @property
    def parameter_constraints(self):
        return self.distribution.parameter_constraints

    def pdf(self, value):
        return self.distribution.pdf(value)

    def logpdf(self, value):
        return self.distribution.logpdf(value)

    def pmf(self, value):
        return self.distribution.pmf(value)

    def logpmf(self, value):
        return self.distribution.logpmf(value)


def marginal_distribution(joint, indices):
    return MarginalDistribution(joint, indices)


def conditional_distribution(joint, indices, values):
    return ConditionalDistribution(joint, indices, values)


__all__ = [
    "ConditionalDistribution",
    "IndependentDistribution",
    "MarginalDistribution",
    "MixtureDistribution",
    "ProductDistribution",
    "TruncatedDistribution",
    "conditional_distribution",
    "marginal_distribution",
]
