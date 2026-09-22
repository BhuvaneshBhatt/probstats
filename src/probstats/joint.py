"""General joint distributions, copulas, and dependent distribution algebra."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

import numpy as np
import sympy as sp

from ._exact_linear_algebra import quadratic_form
from ._integration import integrate_rectangular
from ._symbolic_predicates import TruthValue, certified_equal, certified_zero
from ._validation import integer, sample_shape
from .distributions import MultivariateNormal, Normal
from .distributions.base import Distribution, SymbolicDistribution
from .spaces import (
    MeasureType,
    ProductEventSpace,
    ScalarEventSpace,
)


def _sym_tuple(values):
    return tuple(sp.sympify(v) for v in values)


def _rng(rng):
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def _copula_shape(size):
    return sample_shape(size, none_as_empty=True)


def _eliminate_coordinates(expr, variables, supports, indices, *, discrete):
    """Eliminate selected joint coordinates by exact summation or integration."""
    indices = tuple(indices)
    if not indices:
        return sp.sympify(expr)
    if discrete:
        value = sp.sympify(expr)
        for i in indices:
            variable, support = variables[i], supports[i]
            if isinstance(support, sp.FiniteSet):
                value = sp.Add(*(value.subs(variable, point) for point in support))
            else:
                value = sp.summation(value, (variable, support.inf, support.sup))
        return sp.simplify(value)
    ranges = []
    for i in indices:
        variable, support = variables[i], supports[i]
        if not isinstance(support, sp.Interval):
            return _integrate_sequentially(expr, variables, supports, indices)
        ranges.append((variable, support.start, support.end))
    return integrate_rectangular(expr, ranges)


def _integrate_sequentially(expr, variables, supports, indices):
    value = sp.sympify(expr)
    for i in indices:
        variable, support = variables[i], supports[i]
        try:
            lower, upper = support.inf, support.sup
        except AttributeError as exc:
            raise TypeError(
                f"continuous support for {variable} must provide finite or infinite bounds"
            ) from exc
        value = sp.integrate(value, (variable, lower, upper))
    return sp.simplify(value)


def _indices(indices, dimension):
    try:
        raw_indices = tuple(indices)
    except TypeError:
        raw_indices = (indices,)
    result = tuple(integer(i, name="index", minimum=0) for i in raw_indices)
    if (
        not result
        or len(set(result)) != len(result)
        or any(i < 0 or i >= dimension for i in result)
    ):
        raise ValueError("indices must be distinct valid coordinates")
    return result


@dataclass(frozen=True, slots=True)
class _SymbolicMassDistribution(Distribution):
    value_symbol: sp.Symbol
    mass: sp.Expr
    domain: sp.Set = sp.S.Integers

    @property
    def support(self):
        return self.domain

    @property
    def measure_type(self):
        return MeasureType.DISCRETE

    def pmf(self, value):
        return sp.simplify(self.mass.subs(self.value_symbol, sp.sympify(value)))

    def pdf(self, value):
        return self.pmf(value)

    def logpdf(self, value):
        return sp.log(self.pmf(value))

    def logpmf(self, value):
        return self.logpdf(value)


@dataclass(frozen=True, slots=True)
class JointDistribution(Distribution):
    """Exact joint law defined by a density/ mass expression and named coordinates."""

    variables: tuple[sp.Symbol, ...]
    joint_density: sp.Expr
    supports: tuple[sp.Set, ...]
    discrete: bool = False
    normalized: bool = True

    def __post_init__(self):
        variables = _sym_tuple(self.variables)
        supports = tuple(self.supports)
        if not variables or not all(isinstance(v, sp.Symbol) for v in variables):
            raise TypeError("variables must be a nonempty tuple of SymPy Symbols")
        if len(set(variables)) != len(variables):
            raise ValueError("joint variables must be distinct")
        if len(supports) != len(variables) or not all(
            isinstance(s, sp.Set) for s in supports
        ):
            raise ValueError("one SymPy support set is required per variable")
        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "joint_density", sp.sympify(self.joint_density))
        object.__setattr__(self, "supports", supports)

    @property
    def dimension(self):
        return len(self.variables)

    @property
    def support(self):
        return sp.ProductSet(*self.supports)

    @property
    def event_space(self):
        return ProductEventSpace(tuple(ScalarEventSpace(s) for s in self.supports))

    @property
    def measure_type(self):
        return MeasureType.DISCRETE if self.discrete else MeasureType.CONTINUOUS

    @property
    def parameters(self):
        return (self.joint_density,)

    def pdf(self, value):
        vals = _sym_tuple(value)
        if len(vals) != self.dimension:
            raise ValueError("joint value has wrong dimension")
        return sp.simplify(self.joint_density.subs(dict(zip(self.variables, vals))))

    def logpdf(self, value):
        return sp.log(self.pdf(value))

    def pmf(self, value):
        if not self.discrete:
            return super().pmf(value)
        return self.pdf(value)

    def marginal(self, indices):
        keep = _indices(indices, self.dimension)
        eliminate = tuple(i for i in range(self.dimension) if i not in keep)
        expr = _eliminate_coordinates(
            self.joint_density,
            self.variables,
            self.supports,
            eliminate,
            discrete=self.discrete,
        )
        kept_vars = tuple(self.variables[i] for i in keep)
        kept_supports = tuple(self.supports[i] for i in keep)
        if len(keep) == 1:
            return SymbolicDistribution(
                kept_vars[0],
                sp.log(expr),
                kept_supports[0],
                not expr.has(sp.Integral, sp.Sum),
                self.discrete,
            )
        return JointDistribution(
            kept_vars, expr, kept_supports, self.discrete, self.normalized
        )

    def conditional(self, target_indices, given: dict[int, Any]):
        """Return the conditional law for selected coordinates.

        Coordinates that are neither targets nor observed are marginalized
        before normalization. Continuous nuisance cooordinates are integrated
        and discrete nuisance coordinates are summed over their supports.
        """
        target = _indices(target_indices, self.dimension)
        given = {
            integer(i, name="conditioned index", minimum=0): sp.sympify(v)
            for i, v in given.items()
        }
        if set(target) & set(given):
            raise ValueError("target and conditioned coordinates must be disjoint")
        if any(i < 0 or i >= self.dimension for i in given):
            raise ValueError("conditioned coordinates must be valid indices")
        # Coordinates that are neither targets nor observed are nuisance
        # variables and are marginalized before conditioning.  This makes
        # p(X_A | X_B=b) available without requiring A union B to cover the
        # entire joint vector.
        nuisance = tuple(
            i for i in range(self.dimension) if i not in set(target) | set(given)
        )
        for i, value in given.items():
            contained = self.supports[i].contains(value)
            if contained is sp.S.false:
                raise ValueError(
                    f"conditioned value {value} lies outside support of coordinate {i}"
                )
        expr = _eliminate_coordinates(
            self.joint_density,
            self.variables,
            self.supports,
            nuisance,
            discrete=self.discrete,
        )
        expr = expr.subs({self.variables[i]: value for i, value in given.items()})
        norm = _eliminate_coordinates(
            expr,
            self.variables,
            self.supports,
            target,
            discrete=self.discrete,
        )
        norm = sp.simplify(norm)
        if norm.is_zero is True:
            raise ValueError("conditioning event has zero probability/density")
        density = sp.simplify(expr / norm)
        vars_ = tuple(self.variables[i] for i in target)
        supports = tuple(self.supports[i] for i in target)
        if len(target) == 1:
            return SymbolicDistribution(
                vars_[0],
                sp.log(density),
                supports[0],
                not density.has(sp.Integral, sp.Sum),
                self.discrete,
            )
        return JointDistribution(
            vars_,
            density,
            supports,
            self.discrete,
            not density.has(sp.Integral, sp.Sum),
        )


class Copula(ABC):
    """Copula protocol over the unit hypercube."""

    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @abstractmethod
    def cdf(self, u: Any) -> sp.Expr: ...

    def density(self, u: Any) -> sp.Expr:
        """Copula density, derived from the CDF when not specialized."""
        vals = _sym_tuple(u)
        if len(vals) != self.dimension:
            raise ValueError("copula coordinate has wrong dimension")
        symbols = tuple(sp.Dummy(f"u{i}", real=True) for i in range(self.dimension))
        expr = self.cdf(symbols)
        for symbol in symbols:
            expr = sp.diff(expr, symbol)
        return sp.simplify(expr.subs(dict(zip(symbols, vals))))

    @abstractmethod
    def sample_uniform(self, size=None, rng=None): ...


@dataclass(frozen=True, slots=True)
class IndependenceCopula(Copula):
    dimension: int

    def __post_init__(self):
        dimension = integer(self.dimension, name="dimension", minimum=2)
        object.__setattr__(self, "dimension", dimension)

    def density(self, u):
        vals = _sym_tuple(u)
        if len(vals) != self.dimension:
            raise ValueError("copula coordinate has wrong dimension")
        return sp.Integer(1)

    def cdf(self, u):
        return sp.prod(_sym_tuple(u))

    def sample_uniform(self, size=None, rng=None):
        return _rng(rng).uniform(size=_copula_shape(size) + (self.dimension,))


@dataclass(frozen=True, slots=True)
class GaussianCopula(Copula):
    correlation: sp.ImmutableMatrix

    def __post_init__(self):
        r = sp.ImmutableMatrix(self.correlation)
        if r.rows < 2 or r.rows != r.cols:
            raise ValueError("correlation must be a square matrix of dimension >= 2")
        if any(
            certified_equal(r[i, i], 1) is not TruthValue.TRUE for i in range(r.rows)
        ):
            raise ValueError("correlation matrix must have unit diagonal")
        if r != r.T:
            raise ValueError("correlation matrix must be symmetric")
        if r.is_positive_definite is False:
            raise ValueError("correlation matrix must be positive definite")
        object.__setattr__(self, "correlation", r)

    @property
    def dimension(self):
        return self.correlation.rows

    def cdf(self, u):
        raise NotImplementedError(
            "Gaussian copula CDF has no general exact symbolic form"
        )

    def density(self, u):
        vals = _sym_tuple(u)
        if len(vals) != self.dimension:
            raise ValueError("copula coordinate has wrong dimension")
        z = sp.ImmutableMatrix([sp.sqrt(2) * sp.erfinv(2 * v - 1) for v in vals])
        eye = sp.eye(self.dimension)
        quad = quadratic_form(self.correlation, z) - (z.T * eye * z)[0]
        return sp.simplify(
            self.correlation.det() ** sp.Rational(-1, 2) * sp.exp(-quad / 2)
        )

    def sample_uniform(self, size=None, rng=None):
        r = _rng(rng)
        try:
            corr = np.asarray(self.correlation.tolist(), dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("sampling requires a numeric correlation matrix") from exc
        z = r.multivariate_normal(
            np.zeros(self.dimension), corr, size=_copula_shape(size)
        )
        nd = NormalDist()
        return np.vectorize(nd.cdf, otypes=[float])(z)


@dataclass(frozen=True, slots=True)
class ClaytonCopula(Copula):
    """Bivariate Clayton copula with positive dependence parameter."""

    theta: sp.Expr

    def __post_init__(self):
        theta = sp.sympify(self.theta)
        if theta.is_nonpositive is True:
            raise ValueError("Clayton theta must be positive")
        object.__setattr__(self, "theta", theta)

    @property
    def dimension(self):
        return 2

    def cdf(self, u):
        u1, u2 = _sym_tuple(u)
        return sp.simplify(
            (u1 ** (-self.theta) + u2 ** (-self.theta) - 1) ** (-1 / self.theta)
        )

    def sample_uniform(self, size=None, rng=None):
        r = _rng(rng)
        theta = float(self.theta)
        shape = _copula_shape(size)
        w = r.gamma(shape=1.0 / theta, scale=1.0, size=shape)
        e = r.exponential(size=shape + (2,))
        return (1.0 + e / w[..., None]) ** (-1.0 / theta)


@dataclass(frozen=True, slots=True)
class FrankCopula(Copula):
    """Bivariate Frank copula."""

    theta: sp.Expr

    def __post_init__(self):
        theta = sp.sympify(self.theta)
        if theta.is_zero is True:
            raise ValueError("Frank theta must be nonzero")
        object.__setattr__(self, "theta", theta)

    @property
    def dimension(self):
        return 2

    def cdf(self, u):
        u1, u2 = _sym_tuple(u)
        t = self.theta
        return sp.simplify(
            -sp.log(
                1 + (sp.exp(-t * u1) - 1) * (sp.exp(-t * u2) - 1) / (sp.exp(-t) - 1)
            )
            / t
        )

    def sample_uniform(self, size=None, rng=None):
        r = _rng(rng)
        t = float(self.theta)
        shape = _copula_shape(size)
        u = r.uniform(size=shape)
        w = r.uniform(size=shape)
        a = np.exp(-t * u)
        b = np.exp(-t)
        v = -np.log(1.0 + w * (b - 1.0) / (a - w * (a - 1.0))) / t
        return np.stack((u, v), axis=-1)


@dataclass(frozen=True, slots=True)
class GumbelCopula(Copula):
    """Bivariate Gumbel copula with theta >= 1."""

    theta: sp.Expr

    def __post_init__(self):
        theta = sp.sympify(self.theta)
        if (theta - 1).is_negative is True:
            raise ValueError("Gumbel theta must be at least 1")
        object.__setattr__(self, "theta", theta)

    @property
    def dimension(self):
        return 2

    def cdf(self, u):
        u1, u2 = _sym_tuple(u)
        t = self.theta
        return sp.exp(-(((-sp.log(u1)) ** t + (-sp.log(u2)) ** t) ** (1 / t)))

    def sample_uniform(self, size=None, rng=None):
        # Marshall-Olkin construction using a positive stable frailty.
        r = _rng(rng)
        theta = float(self.theta)
        shape = _copula_shape(size)
        if abs(theta - 1.0) < 1e-15:
            return r.uniform(size=shape + (2,))
        alpha = 1.0 / theta
        v = r.uniform(-np.pi / 2, np.pi / 2, size=shape)
        w = r.exponential(size=shape)
        part1 = np.sin(alpha * (v + np.pi / 2)) / (np.cos(v) ** (1 / alpha))
        part2 = (np.cos(v - alpha * (v + np.pi / 2)) / w) ** ((1 - alpha) / alpha)
        stable = part1 * part2
        e = r.exponential(size=shape + (2,))
        return np.exp(-((e / stable[..., None]) ** alpha))


@dataclass(frozen=True, slots=True)
class CopulaDistribution(Distribution):
    """Joint law formed by a copula and scalar marginals.

    Continuous, discrete, and mixed scalar marginals are supported when the
    copula exposes an exact CDF.  Mixed mass-density values are obtained by
    differentiating in continuous coordinates and taking finite CDF
    differences in discrete coordinates.
    """

    marginals: tuple[Distribution, ...]
    copula: Copula

    def __post_init__(self):
        marginals = tuple(self.marginals)
        if len(marginals) != self.copula.dimension:
            raise ValueError("copula dimension must match number of marginals")
        if any(m.event_space.rank != 0 for m in marginals):
            raise TypeError("copula marginals must be scalar distributions")
        object.__setattr__(self, "marginals", marginals)

    @property
    def dimension(self):
        return len(self.marginals)

    @property
    def support(self):
        return sp.ProductSet(*(m.support for m in self.marginals))

    @property
    def event_space(self):
        return ProductEventSpace(tuple(m.event_space for m in self.marginals))

    @property
    def measure_type(self):
        kinds = {m.measure_type for m in self.marginals}
        if len(kinds) == 1:
            return next(iter(kinds))
        return MeasureType.MIXED

    @property
    def parameters(self):
        return tuple(p for m in self.marginals for p in m.parameters)

    def cdf(self, value):
        vals = _sym_tuple(value)
        if len(vals) != self.dimension:
            raise ValueError("joint value has wrong dimension")
        return sp.simplify(
            self.copula.cdf(tuple(m.cdf(x) for m, x in zip(self.marginals, vals)))
        )

    def _mixed_density(self, vals):
        if all(m.measure_type is MeasureType.CONTINUOUS for m in self.marginals):
            u = tuple(m.cdf(x) for m, x in zip(self.marginals, vals))
            return sp.simplify(
                self.copula.density(u)
                * sp.prod(m.pdf(x) for m, x in zip(self.marginals, vals))
            )
        # Mixed discrete coordinates require an exact copula CDF for finite differences.
        u = tuple(sp.Dummy(f"u{i}", real=True) for i in range(self.dimension))
        expr = self.copula.cdf(u)
        continuous = [
            i
            for i, m in enumerate(self.marginals)
            if m.measure_type is MeasureType.CONTINUOUS
        ]
        discrete = [
            i
            for i, m in enumerate(self.marginals)
            if m.measure_type is MeasureType.DISCRETE
        ]
        for i in continuous:
            expr = sp.diff(expr, u[i])
        # Inclusion-exclusion over upper/lower CDF endpoints of discrete atoms.
        total = sp.Integer(0)
        for mask in range(1 << len(discrete)):
            subs = {}
            sign = 1
            for bit, i in enumerate(discrete):
                m, x = self.marginals[i], vals[i]
                upper = m.cdf(x)
                lower = sp.simplify(upper - m.pmf(x))
                if mask & (1 << bit):
                    subs[u[i]] = lower
                    sign *= -1
                else:
                    subs[u[i]] = upper
            for i in continuous:
                subs[u[i]] = self.marginals[i].cdf(vals[i])
            total += sign * expr.subs(subs)
        jac = sp.prod(self.marginals[i].pdf(vals[i]) for i in continuous)
        return sp.simplify(total * jac)

    def pdf(self, value):
        vals = _sym_tuple(value)
        if len(vals) != self.dimension:
            raise ValueError("joint value has wrong dimension")
        return self._mixed_density(vals)

    def pmf(self, value):
        if self.measure_type is not MeasureType.DISCRETE:
            return super().pmf(value)
        return self.pdf(value)

    def logpdf(self, value):
        return sp.log(self.pdf(value))

    def marginal(self, indices):
        keep = _indices(indices, self.dimension)
        if len(keep) == 1:
            return self.marginals[keep[0]]
        if isinstance(self.copula, IndependenceCopula):
            return CopulaDistribution(
                tuple(self.marginals[i] for i in keep), IndependenceCopula(len(keep))
            )
        if isinstance(self.copula, GaussianCopula):
            r = self.copula.correlation.extract(keep, keep)
            return CopulaDistribution(
                tuple(self.marginals[i] for i in keep), GaussianCopula(r)
            )
        if len(keep) == self.dimension:
            return self
        raise NotImplementedError(
            "this copula family does not provide higher-dimensional subcopulas"
        )


@dataclass(frozen=True, slots=True)
class PushforwardDistribution(Distribution):
    """Exact formal pushforward of a joint law through arbitrary expressions."""

    source: Distribution
    expressions: tuple[sp.Expr, ...]
    variables: tuple[sp.Symbol, ...]

    def __post_init__(self):
        exprs = tuple(sp.sympify(e) for e in self.expressions)
        vars_ = tuple(sp.sympify(v) for v in self.variables)
        if not exprs:
            raise ValueError("at least one output expression is required")
        if not vars_ or not all(isinstance(v, sp.Symbol) for v in vars_):
            raise TypeError("source variables must be SymPy Symbols")
        object.__setattr__(self, "expressions", exprs)
        object.__setattr__(self, "variables", vars_)

    @property
    def support(self):
        return sp.S.Reals if len(self.expressions) == 1 else sp.S.UniversalSet

    @property
    def event_space(self):
        if len(self.expressions) == 1:
            return ScalarEventSpace(sp.S.Reals)
        from .spaces import VectorEventSpace

        return VectorEventSpace(len(self.expressions), sp.S.Reals)

    @property
    def measure_type(self):
        return self.source.measure_type

    @property
    def parameters(self):
        return self.source.parameters

    def _source_expr(self):
        return _joint_pdf_expr(self.source, self.variables)

    def pdf(self, value):
        if self.source.measure_type is MeasureType.MIXED:
            raise NotImplementedError(
                "mixed-measure nonlinear pushforwards require an explicit reference measure"
            )
        vals = (sp.sympify(value),) if len(self.expressions) == 1 else _sym_tuple(value)
        if len(vals) != len(self.expressions):
            raise ValueError("pushforward value has wrong dimension")
        expr = self._source_expr()
        discrete = self.source.measure_type is MeasureType.DISCRETE
        for out_expr, val in zip(self.expressions, vals):
            expr *= (
                sp.KroneckerDelta(out_expr, val)
                if discrete
                else sp.DiracDelta(val - out_expr)
            )
        supports = (
            self.source.support.args
            if isinstance(self.source.support, sp.ProductSet)
            else ()
        )
        if len(supports) != len(self.variables):
            supports = tuple(
                sp.S.Integers if discrete else sp.S.Reals for _ in self.variables
            )
        return _eliminate_coordinates(
            expr,
            self.variables,
            supports,
            range(len(self.variables)),
            discrete=discrete,
        )

    def pmf(self, value):
        if self.measure_type is not MeasureType.DISCRETE:
            return super().pmf(value)
        return self.pdf(value)

    def logpdf(self, value):
        return sp.log(self.pdf(value))


def _affine_normal_pushforward(distribution, expressions, variables):
    """Return the closed affine normal pushforward when all expressions are affine."""
    try:
        rows = []
        offsets = []
        zero_subs = {variable: 0 for variable in variables}
        for expression in expressions:
            polynomial = sp.Poly(expression, *variables)
            if polynomial.total_degree() > 1:
                return None
            offsets.append(sp.simplify(expression.subs(zero_subs)))
            rows.append(
                [sp.simplify(sp.diff(expression, variable)) for variable in variables]
            )
    except (sp.PolynomialError, ValueError):
        return None

    matrix = sp.ImmutableMatrix(rows)
    offset = sp.ImmutableMatrix(offsets)
    mean = sp.ImmutableMatrix(distribution.mean)
    covariance = sp.ImmutableMatrix(distribution.covariance)
    out_mean = sp.simplify(matrix * mean + offset)
    out_cov = sp.simplify(matrix * covariance * matrix.T)
    if len(expressions) == 1:
        return Normal(out_mean[0], sp.sqrt(out_cov[0, 0]))
    return MultivariateNormal(out_mean, out_cov)


def _exact_support_preimage(pivot_support, solved, variable, support):
    """Return an exact support preimage when SymPy can resolve the inequality."""
    try:
        preimage = sp.solve_univariate_inequality(
            pivot_support.as_relational(solved), variable, relational=False
        )
    except (NotImplementedError, TypeError, ValueError):
        return None
    if isinstance(preimage, sp.Interval):
        return sp.Interval(
            sp.Max(support.start, preimage.start),
            sp.Min(support.end, preimage.end),
            left_open=support.left_open or preimage.left_open,
            right_open=support.right_open or preimage.right_open,
        )
    return sp.Intersection(support, preimage)


def dependent_transform(distribution: Distribution, expressions, variables=None):
    """Push a dependent joint distribution through arbitrary expressions.

    Affine maps of multivariate normals are closed analytically; all other
    maps retain an exact formal pushforward using Dirac/Kronecker deltas.
    """
    exprs = (
        (sp.sympify(expressions),)
        if not isinstance(expressions, (tuple, list, sp.MatrixBase))
        else tuple(map(sp.sympify, expressions))
    )
    if variables is None:
        if isinstance(distribution, JointDistribution):
            variables = distribution.variables
        else:
            dim = (
                distribution.event_space.shape[0]
                if distribution.event_space.rank == 1
                else None
            )
            if dim is None:
                raise TypeError("variables are required for a non-vector joint source")
            variables = tuple(sp.Symbol(f"x{i}", real=True) for i in range(dim))
    variables = tuple(map(sp.sympify, variables))
    if isinstance(distribution, MultivariateNormal):
        affine_result = _affine_normal_pushforward(distribution, exprs, variables)
        if affine_result is not None:
            return affine_result
    return PushforwardDistribution(distribution, exprs, variables)


def _joint_pdf_expr(distribution: Distribution, variables):
    if isinstance(distribution, JointDistribution):
        if tuple(variables) == distribution.variables:
            return distribution.joint_density
        return distribution.pdf(variables)
    return distribution.pdf(sp.ImmutableMatrix(variables))


def dependent_linear_combination(
    distribution: Distribution, coefficients=None, shift=0
):
    """Distribution of ``a.T @ X + shift`` under a known joint law."""
    dim = (
        distribution.event_space.shape[0]
        if distribution.event_space.rank == 1
        else None
    )
    if dim is None:
        raise TypeError(
            "dependent linear combinations require a vector-valued joint distribution"
        )
    coeffs = (
        tuple(sp.Integer(1) for _ in range(dim))
        if coefficients is None
        else _sym_tuple(coefficients)
    )
    if len(coeffs) != dim or all(certified_zero(a) is TruthValue.TRUE for a in coeffs):
        raise ValueError("one nonzero coefficient is required per coordinate")
    shift = sp.sympify(shift)
    if isinstance(distribution, MultivariateNormal):
        a = sp.ImmutableMatrix([coeffs])
        mu = sp.ImmutableMatrix(distribution.mean)
        cov = sp.ImmutableMatrix(distribution.covariance)
        return Normal(
            sp.simplify((a * mu)[0] + shift), sp.sqrt(sp.simplify((a * cov * a.T)[0]))
        )

    variables = tuple(sp.Symbol(f"x{i}", real=True) for i in range(dim))
    z = sp.Symbol("z", real=True)
    pivot = next(
        (
            i
            for i in range(dim - 1, -1, -1)
            if certified_zero(coeffs[i]) is TruthValue.FALSE
        ),
        None,
    )
    if pivot is None:
        raise ValueError("at least one coefficient must be provably nonzero")
    solved = sp.simplify(
        (z - shift - sum(coeffs[i] * variables[i] for i in range(dim) if i != pivot))
        / coeffs[pivot]
    )
    density = _joint_pdf_expr(distribution, variables).subs(
        variables[pivot], solved
    ) / sp.Abs(coeffs[pivot])
    # The eliminated coordinate must remain inside its own support after the
    # linear constraint is solved.  Keeping this as a Piecewise indicator lets
    # SymPy derive z-dependent integration/summation bounds instead of using an
    # impossible preimage.
    pivot_support = (
        distribution.support.args[pivot]
        if isinstance(distribution.support, sp.ProductSet)
        else (
            sp.S.Integers
            if distribution.measure_type is MeasureType.DISCRETE
            else sp.S.Reals
        )
    )
    remaining = [i for i in range(dim) if i != pivot]
    exact_preimage = None
    if (
        distribution.measure_type is MeasureType.CONTINUOUS
        and len(remaining) == 1
        and isinstance(pivot_support, sp.Interval)
    ):
        i = remaining[0]
        support = (
            distribution.support.args[i]
            if isinstance(distribution.support, sp.ProductSet)
            else sp.S.Reals
        )
        if isinstance(support, sp.Interval):
            exact_preimage = _exact_support_preimage(
                pivot_support, solved, variables[i], support
            )
    if exact_preimage is None:
        density = sp.Piecewise((density, sp.Contains(solved, pivot_support)), (0, True))
    supports = list(
        distribution.support.args
        if isinstance(distribution.support, sp.ProductSet)
        else (
            sp.S.Integers
            if distribution.measure_type is MeasureType.DISCRETE
            else sp.S.Reals
            for _ in range(dim)
        )
    )
    if exact_preimage is not None and len(remaining) == 1:
        supports[remaining[0]] = exact_preimage
    density = _eliminate_coordinates(
        density,
        variables,
        supports,
        remaining,
        discrete=distribution.measure_type is MeasureType.DISCRETE,
    )
    result = sp.simplify(density)
    if distribution.measure_type is MeasureType.DISCRETE:
        return _SymbolicMassDistribution(z, result, sp.S.Integers)
    output_support = sp.S.Reals
    supports = (
        distribution.support.args
        if isinstance(distribution.support, sp.ProductSet)
        else ()
    )
    if supports and all(isinstance(s, sp.Interval) for s in supports):
        lower = shift
        upper = shift
        for a, s in zip(coeffs, supports):
            if a.is_nonnegative is True:
                lower += a * s.start
                upper += a * s.end
            elif a.is_nonpositive is True:
                lower += a * s.end
                upper += a * s.start
            else:
                lower = -sp.oo
                upper = sp.oo
                break
        output_support = sp.Interval(sp.simplify(lower), sp.simplify(upper))
    return SymbolicDistribution(
        z, sp.log(result), output_support, not result.has(sp.Integral, sp.Sum)
    )


def dependent_convolution(distribution: Distribution):
    """Distribution of the coordinate sum for a dependent joint law."""
    return dependent_linear_combination(distribution)


__all__ = [
    "ClaytonCopula",
    "Copula",
    "CopulaDistribution",
    "FrankCopula",
    "GaussianCopula",
    "GumbelCopula",
    "IndependenceCopula",
    "JointDistribution",
    "PushforwardDistribution",
    "dependent_convolution",
    "dependent_linear_combination",
    "dependent_transform",
]
