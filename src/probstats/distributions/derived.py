"""Derived and user-defined probability distributions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import sympy as sp

from .._symbolic_predicates import TruthValue, certified_equal
from ..events import PredicateEvent, SetEvent
from ..functionals import cdf, density, expectation, probability
from ..spaces import MeasureType, ScalarEventSpace
from .base import Distribution, scalar_batch


def _indicator(condition: sp.Basic) -> sp.Expr:
    return sp.Piecewise((sp.S.One, condition), (sp.S.Zero, True))


def _support_condition(value: sp.Expr, support: sp.Set) -> sp.Basic:
    if isinstance(support, sp.Interval):
        left = value > support.start if support.left_open else value >= support.start
        right = value < support.end if support.right_open else value <= support.end
        return sp.And(left, right)
    if isinstance(support, sp.FiniteSet):
        return sp.Or(*(sp.Eq(value, item) for item in support))
    return sp.Contains(value, support)


def _integrate_over(dist: Distribution, expression: sp.Expr, variable: sp.Symbol):
    support = dist.support
    if dist.measure_type is MeasureType.DISCRETE:
        if isinstance(support, sp.FiniteSet):
            return sp.simplify(sum(expression.subs(variable, k) for k in support))
        if support == sp.S.Naturals0:
            return sp.simplify(sp.summation(expression, (variable, 0, sp.oo)))
        if isinstance(support, sp.Intersection) and support.is_subset(sp.S.Integers):
            bounds = [arg for arg in support.args if isinstance(arg, sp.Interval)]
            if len(bounds) == 1:
                lo = sp.ceiling(bounds[0].start)
                hi = sp.floor(bounds[0].end)
                return sp.simplify(sp.summation(expression, (variable, lo, hi)))
        return sp.Sum(expression, (variable, support))
    if isinstance(support, sp.Interval):
        return sp.simplify(
            sp.integrate(expression, (variable, support.start, support.end))
        )
    if support == sp.S.Reals:
        return sp.simplify(sp.integrate(expression, (variable, -sp.oo, sp.oo)))
    raise ValueError(
        "mixing distribution must have scalar interval or discrete support"
    )


@dataclass(frozen=True, slots=True)
class ProbabilityDistribution(Distribution):
    """A scalar distribution defined by a symbolic density or mass function.

    The expression is interpreted with respect to counting measure when
    ``discrete=True`` and Lebesgue measure otherwise. Optional CDF, quantile,
    and sampler hooks allow a user-defined law to participate in the same
    functional and sampling protocols as built-in distributions.
    """

    variable: sp.Symbol
    density_expression: sp.Expr
    domain: sp.Set = sp.S.Reals
    discrete: bool = False
    cdf_expression: sp.Expr | None = None
    quantile_expression: sp.Expr | None = None
    probability_symbol: sp.Symbol | None = None
    sampler: Callable[..., Any] | None = None
    validate_normalization: bool = True

    def __post_init__(self):
        var = sp.sympify(self.variable)
        expr = sp.sympify(self.density_expression)
        if not isinstance(var, sp.Symbol):
            raise TypeError("variable must be a SymPy Symbol")
        if not isinstance(self.domain, sp.Set):
            raise TypeError("domain must be a SymPy Set")
        if expr.has(sp.nan, sp.zoo):
            raise ValueError(
                "density_expression must be finite on its intended support"
            )
        object.__setattr__(self, "variable", var)
        object.__setattr__(self, "density_expression", expr)
        if self.cdf_expression is not None:
            object.__setattr__(self, "cdf_expression", sp.sympify(self.cdf_expression))
        if self.quantile_expression is not None:
            q = self.probability_symbol
            if q is None:
                q = sp.Symbol("p", real=True)
                object.__setattr__(self, "probability_symbol", q)
            if not isinstance(q, sp.Symbol):
                raise TypeError("probability_symbol must be a SymPy Symbol")
            object.__setattr__(
                self, "quantile_expression", sp.sympify(self.quantile_expression)
            )
        if self.validate_normalization:
            total = self._normalization()
            if (
                total is not None
                and not total.has(sp.Integral, sp.Sum)
                and total.is_number
                and certified_equal(total, 1) is not TruthValue.TRUE
            ):
                raise ValueError("density_expression does not normalize to one")

    def _normalization(self):
        expr = self.density_expression
        if self.discrete:
            if isinstance(self.domain, sp.FiniteSet):
                return sp.simplify(
                    sum(expr.subs(self.variable, k) for k in self.domain)
                )
            if self.domain == sp.S.Naturals0:
                return sp.simplify(sp.summation(expr, (self.variable, 0, sp.oo)))
            return None
        if isinstance(self.domain, sp.Interval):
            return sp.simplify(
                sp.integrate(expr, (self.variable, self.domain.start, self.domain.end))
            )
        if self.domain == sp.S.Reals:
            return sp.simplify(sp.integrate(expr, (self.variable, -sp.oo, sp.oo)))
        return None

    @property
    def support(self):
        return self.domain

    @property
    def measure_type(self):
        return MeasureType.DISCRETE if self.discrete else MeasureType.CONTINUOUS

    @property
    def parameters(self):
        excluded = {self.variable}
        if self.probability_symbol is not None:
            excluded.add(self.probability_symbol)
        expressions = [self.density_expression]
        if self.cdf_expression is not None:
            expressions.append(self.cdf_expression)
        symbols = set().union(*(expr.free_symbols for expr in expressions)) - excluded
        return tuple(sorted(symbols, key=sp.default_sort_key))

    @scalar_batch
    def pdf(self, value):
        x = sp.sympify(value)
        return sp.simplify(
            self.density_expression.subs(self.variable, x)
            * _indicator(_support_condition(x, self.support))
        )

    def pmf(self, value):
        if not self.discrete:
            return super().pmf(value)
        return self.pdf(value)

    @scalar_batch
    def logpdf(self, value):
        return sp.log(self.pdf(value))

    def logpmf(self, value):
        return sp.log(self.pmf(value))

    def _cdf(self, value):
        if self.cdf_expression is None:
            return NotImplemented
        return self.cdf_expression.subs(self.variable, value)

    def _quantile(self, p):
        if self.quantile_expression is None:
            return NotImplemented
        return self.quantile_expression.subs(self.probability_symbol, p)

    def _sample(self, *, size=None, rng=None):
        if self.sampler is None:
            from ..sampling import SamplingError

            raise SamplingError("user-defined distribution has no sampler")
        return self.sampler(size=size, rng=rng)


@dataclass(frozen=True, slots=True)
class CensoredDistribution(Distribution):
    """Distribution obtained by clipping a scalar law to censoring limits.

    For a continuous base law, finite censoring limits become point masses and
    the resulting law therefore has a mixed reference measure.
    """

    base: Distribution
    lower: sp.Expr = -sp.oo
    upper: sp.Expr = sp.oo

    def __post_init__(self):
        if not isinstance(self.base.event_space, ScalarEventSpace):
            raise TypeError("CensoredDistribution requires a scalar base distribution")
        lo, hi = sp.sympify(self.lower), sp.sympify(self.upper)
        if sp.simplify(lo < hi) is sp.false:
            raise ValueError("lower must be less than upper")
        object.__setattr__(self, "lower", lo)
        object.__setattr__(self, "upper", hi)

    @property
    def support(self):
        body = sp.Intersection(self.base.support, sp.Interval(self.lower, self.upper))
        atoms = []
        if self.lower != -sp.oo:
            atoms.append(self.lower)
        if self.upper != sp.oo:
            atoms.append(self.upper)
        return sp.Union(body, sp.FiniteSet(*atoms)) if atoms else body

    @property
    def event_space(self):
        return ScalarEventSpace(self.support)

    @property
    def measure_type(self):
        if self.base.measure_type is MeasureType.DISCRETE:
            return MeasureType.DISCRETE
        finite_limit = self.lower != -sp.oo or self.upper != sp.oo
        return MeasureType.MIXED if finite_limit else self.base.measure_type

    @property
    def parameters(self):
        return (*self.base.parameters, self.lower, self.upper)

    @property
    def parameter_constraints(self):
        return sp.And(self.base.parameter_constraints, self.lower < self.upper)

    @property
    def lower_mass(self):
        return sp.S.Zero if self.lower == -sp.oo else cdf(self.base, self.lower)

    @property
    def upper_mass(self):
        if self.upper == sp.oo:
            return sp.S.Zero
        if self.base.measure_type is MeasureType.DISCRETE:
            return sp.simplify(
                1 - cdf(self.base, self.upper) + self.base.pmf(self.upper)
            )
        return sp.simplify(1 - cdf(self.base, self.upper))

    @scalar_batch
    def pdf(self, value):
        x = sp.sympify(value)
        branches = []
        if self.lower != -sp.oo:
            branches.append((self.lower_mass, sp.Eq(x, self.lower)))
        if self.upper != sp.oo:
            branches.append((self.upper_mass, sp.Eq(x, self.upper)))
        branches.append((self.base.pdf(x), sp.And(x > self.lower, x < self.upper)))
        branches.append((sp.S.Zero, True))
        return sp.Piecewise(*branches)

    def pmf(self, value):
        if self.measure_type is not MeasureType.DISCRETE:
            return super().pmf(value)
        x = sp.sympify(value)
        branches = []
        if self.lower != -sp.oo:
            branches.append((self.lower_mass, sp.Eq(x, self.lower)))
        if self.upper != sp.oo:
            branches.append((self.upper_mass, sp.Eq(x, self.upper)))
        branches.append((self.base.pmf(x), sp.And(x > self.lower, x < self.upper)))
        branches.append((sp.S.Zero, True))
        return sp.Piecewise(*branches)

    @scalar_batch
    def logpdf(self, value):
        return sp.log(self.pdf(value))

    def logpmf(self, value):
        return sp.log(self.pmf(value))

    def _cdf(self, value):
        x = sp.sympify(value)
        return sp.Piecewise(
            (0, x < self.lower),
            (1, x >= self.upper),
            (cdf(self.base, x), True),
        )

    def _probability(self, event):
        if isinstance(event, PredicateEvent):
            event = SetEvent(event.as_set())
        if not isinstance(event, SetEvent):
            return NotImplemented
        event_set = event.set
        interior = sp.Intersection(event_set, sp.Interval.open(self.lower, self.upper))
        value = probability(self.base, interior)
        if self.lower != -sp.oo:
            value += self.lower_mass * _indicator(sp.Contains(self.lower, event_set))
        if self.upper != sp.oo:
            value += self.upper_mass * _indicator(sp.Contains(self.upper, event_set))
        return sp.simplify(value)

    def _expectation(self, expression, variable):
        x = variable if variable is not None else sp.Dummy("x", real=True)
        expr = sp.sympify(expression)
        if self.base.measure_type is MeasureType.DISCRETE:
            value = sp.S.Zero
            if self.lower != -sp.oo:
                value += expr.subs(x, self.lower) * self.lower_mass
            if self.upper != sp.oo:
                value += expr.subs(x, self.upper) * self.upper_mass
            support = self.base.support
            if isinstance(support, sp.FiniteSet):
                for point in support:
                    if (
                        sp.simplify(point > self.lower) is sp.true
                        and sp.simplify(point < self.upper) is sp.true
                    ):
                        value += expr.subs(x, point) * self.base.pmf(point)
                return sp.simplify(value)
            if support == sp.S.Naturals0 and self.upper != sp.oo:
                k = sp.Dummy("k", integer=True, nonnegative=True)
                lo = sp.Max(0, sp.floor(self.lower) + 1) if self.lower != -sp.oo else 0
                hi = sp.ceiling(self.upper) - 1
                body = expr.subs(x, k) * self.base.pmf(k)
                value += sp.summation(body, (k, lo, hi))
                return sp.simplify(value)
            return NotImplemented
        body = expr * self.base.pdf(x)
        value = sp.integrate(body, (x, self.lower, self.upper))
        if self.lower != -sp.oo:
            value += expr.subs(x, self.lower) * self.lower_mass
        if self.upper != sp.oo:
            value += expr.subs(x, self.upper) * self.upper_mass
        return sp.simplify(value)

    def _sample(self, *, size=None, rng=None):
        from ..sampling import sample

        draws = np.asarray(sample(self.base, size=size, rng=rng))
        lo = -np.inf if self.lower == -sp.oo else float(self.lower)
        hi = np.inf if self.upper == sp.oo else float(self.upper)
        return np.clip(draws, lo, hi)


@dataclass(frozen=True, slots=True)
class ParameterMixtureDistribution(Distribution):
    """Mixture formed by integrating a conditional family over a parameter law.

    ``conditional`` is a callable accepting one parameter value and returning a
    scalar :class:`Distribution`. This representation is usable both
    symbolically, by calling it on an internal SymPy symbol, and numerically for
    hierarchical sampling.
    """

    mixing: Distribution
    conditional: Callable[[Any], Distribution]
    parameter: sp.Symbol = field(default_factory=lambda: sp.Symbol("theta", real=True))

    def __post_init__(self):
        if not isinstance(self.mixing.event_space, ScalarEventSpace):
            raise TypeError("mixing distribution must be scalar")
        if not callable(self.conditional):
            raise TypeError("conditional must be callable")
        param = sp.sympify(self.parameter)
        if not isinstance(param, sp.Symbol):
            raise TypeError("parameter must be a SymPy Symbol")
        object.__setattr__(self, "parameter", param)
        component = self.conditional(param)
        if not isinstance(component, Distribution):
            raise TypeError("conditional(parameter) must return a Distribution")
        if not isinstance(component.event_space, ScalarEventSpace):
            raise TypeError(
                "parameter mixtures currently require scalar conditional laws"
            )

    @property
    def component(self):
        return self.conditional(self.parameter)

    @property
    def support(self):
        return self.component.support

    @property
    def event_space(self):
        return self.component.event_space

    @property
    def measure_type(self):
        return self.component.measure_type

    @property
    def parameters(self):
        symbols = (set(self.component.free_symbols) - {self.parameter}) | set(
            self.mixing.free_symbols
        )
        return tuple(sorted(symbols, key=sp.default_sort_key))

    @property
    def parameter_constraints(self):
        condition = sp.sympify(self.component.parameter_constraints)
        if isinstance(condition, sp.And):
            external = [
                arg for arg in condition.args if self.parameter not in arg.free_symbols
            ]
        elif self.parameter in condition.free_symbols:
            external = []
        else:
            external = [condition]
        return sp.And(self.mixing.parameter_constraints, *external)

    @scalar_batch
    def pdf(self, value):
        integrand = density(self.component, value) * density(
            self.mixing, self.parameter
        )
        return _integrate_over(self.mixing, integrand, self.parameter)

    def pmf(self, value):
        if self.measure_type is not MeasureType.DISCRETE:
            return super().pmf(value)
        return self.pdf(value)

    @scalar_batch
    def logpdf(self, value):
        return sp.log(self.pdf(value))

    def logpmf(self, value):
        return sp.log(self.pmf(value))

    def _cdf(self, value):
        integrand = cdf(self.component, value) * density(self.mixing, self.parameter)
        return _integrate_over(self.mixing, integrand, self.parameter)

    def _probability(self, event):
        integrand = probability(self.component, event) * density(
            self.mixing, self.parameter
        )
        return _integrate_over(self.mixing, integrand, self.parameter)

    def _expectation(self, expression, variable):
        inner = expectation(self.component, expression, variable=variable)
        integrand = inner * density(self.mixing, self.parameter)
        return _integrate_over(self.mixing, integrand, self.parameter)

    def _sample(self, *, size=None, rng=None):
        from ..sampling import as_rng, sample

        gen = as_rng(rng)
        theta = sample(self.mixing, size=size, rng=gen)
        if size is None:
            return sample(self.conditional(sp.Float(float(theta))), rng=gen)
        values = np.asarray(theta)
        out = np.empty(values.shape, dtype=float)
        for idx in np.ndindex(values.shape):
            out[idx] = sample(self.conditional(sp.Float(float(values[idx]))), rng=gen)
        return out


@dataclass(frozen=True, slots=True)
class SplicedDistribution(Distribution):
    """Piecewise distribution assembled from region-restricted component laws."""

    components: tuple[Distribution, ...]
    regions: tuple[sp.Set, ...]
    weights: tuple[sp.Expr, ...]

    def __init__(
        self,
        components: Sequence[Distribution],
        regions: Sequence[sp.Set],
        weights: Sequence[Any],
    ):
        from ..composition import MixtureDistribution, TruncatedDistribution

        comps = tuple(components)
        regs = tuple(regions)
        ws = tuple(sp.sympify(w) for w in weights)
        if not comps or len(comps) != len(regs) or len(comps) != len(ws):
            raise ValueError(
                "components, regions, and weights must have the same nonzero length"
            )
        if any(not isinstance(region, sp.Set) for region in regs):
            raise TypeError("regions must be SymPy sets")
        if any(not isinstance(comp.event_space, ScalarEventSpace) for comp in comps):
            raise TypeError("spliced components must be scalar")
        if len({comp.measure_type for comp in comps}) != 1:
            raise ValueError("spliced components must share a measure type")
        for i, left in enumerate(regs):
            for right in regs[i + 1 :]:
                if sp.Intersection(left, right) is not sp.S.EmptySet:
                    raise ValueError("splice regions must be disjoint")
        mixture = MixtureDistribution(
            [TruncatedDistribution(comp, region) for comp, region in zip(comps, regs)],
            ws,
        )
        object.__setattr__(self, "components", comps)
        object.__setattr__(self, "regions", regs)
        object.__setattr__(self, "weights", mixture.weights)

    @property
    def distribution(self):
        from ..composition import MixtureDistribution, TruncatedDistribution

        return MixtureDistribution(
            [
                TruncatedDistribution(comp, region)
                for comp, region in zip(self.components, self.regions)
            ],
            self.weights,
        )

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

    @scalar_batch
    def pdf(self, value):
        return self.distribution.pdf(value)

    def pmf(self, value):
        return self.distribution.pmf(value)

    @scalar_batch
    def logpdf(self, value):
        return self.distribution.logpdf(value)

    def logpmf(self, value):
        return self.distribution.logpmf(value)

    def _cdf(self, value):
        return self.distribution._cdf(value)

    def _probability(self, event):
        return probability(self.distribution, event)

    def _expectation(self, expression, variable):
        return expectation(self.distribution, expression, variable=variable)

    def _sample(self, *, size=None, rng=None):
        from ..sampling import sample

        return sample(self.distribution, size=size, rng=rng)


__all__ = [
    "CensoredDistribution",
    "ParameterMixtureDistribution",
    "ProbabilityDistribution",
    "SplicedDistribution",
]
