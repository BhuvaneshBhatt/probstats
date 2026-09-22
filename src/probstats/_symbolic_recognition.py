"""Distribution algebra and expression-to-law recognition."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import sympy as sp

from ._symbolic_predicates import TruthValue, certified_equal
from ._symbolic_transforms import _s
from ._validation import integer
from .algebra import recognize_affine, recognize_product, recognize_sum
from .distributions import Beta, LogNormal, MultivariateNormal, Normal, Uniform
from .distributions.base import Distribution, SymbolicDistribution
from .spaces import MeasureType
from .transforms import transformed_distribution


def convolution(*distributions: Distribution) -> Distribution:
    """Return the convolution law of independent scalar distributions.

    Closed-family recognition is attempted for the whole collection first;
    otherwise exact pairwise symbolic convolution is retained.
    """
    if not distributions:
        raise ValueError("at least one distribution is required")
    if len(distributions) == 1:
        return distributions[0]
    closed = recognize_sum(*distributions)
    if closed.recognized:
        return closed.distribution

    def pair(left: Distribution, right: Distribution) -> Distribution:
        if left.measure_type != right.measure_type:
            raise TypeError("generic convolution requires a common measure type")
        z = sp.Symbol("z", real=True)
        x = sp.Symbol("x", real=True)
        if left.measure_type is MeasureType.CONTINUOUS:
            lf = sp.Piecewise((left.pdf(x), sp.Contains(x, left.support)), (0, True))
            rf = sp.Piecewise(
                (right.pdf(z - x), sp.Contains(z - x, right.support)), (0, True)
            )
            density = sp.integrate(lf * rf, (x, -sp.oo, sp.oo))
            return SymbolicDistribution(
                z, sp.log(density), sp.S.Reals, density.has(sp.Integral) is False
            )
        k = sp.Symbol("k", integer=True)
        mass = sp.summation(left.pmf(k) * right.pmf(z - k), (k, -sp.oo, sp.oo))
        return SymbolicDistribution(
            z, sp.log(mass), sp.S.Integers, mass.has(sp.Sum) is False, discrete=True
        )

    result = distributions[0]
    for distribution in distributions[1:]:
        result = pair(result, distribution)
    return result


@dataclass(frozen=True, slots=True)
class OrderStatisticDistribution(Distribution):
    """The ``rank``-th order statistic among ``sample_size`` iid scalar draws."""

    base: Distribution
    sample_size: int
    rank: int

    def __post_init__(self):
        if self.base.event_space.rank != 0:
            raise TypeError("order statistics require a scalar base law")
        if self.sample_size < 1 or not 1 <= self.rank <= self.sample_size:
            raise ValueError("require sample_size >= 1 and 1 <= rank <= sample_size")

    @property
    def support(self):
        return self.base.support

    @property
    def measure_type(self):
        return self.base.measure_type

    @property
    def parameters(self):
        return (
            *self.base.parameters,
            sp.Integer(self.sample_size),
            sp.Integer(self.rank),
        )

    def pdf(self, value):
        x = _s(value)
        n = self.sample_size
        r = self.rank
        F = self.base.cdf(x)
        if self.measure_type is MeasureType.DISCRETE:
            return self.pmf(x)
        c = sp.factorial(n) / (sp.factorial(r - 1) * sp.factorial(n - r))
        return sp.simplify(c * F ** (r - 1) * (1 - F) ** (n - r) * self.base.pdf(x))

    def pmf(self, value):
        if self.measure_type is not MeasureType.DISCRETE:
            return super().pmf(value)
        x = _s(value)
        # P(X_(r) <= x) - P(X_(r) <= x - 1)
        return sp.simplify(self._cdf(x) - self._cdf(x - 1))

    def logpdf(self, value):
        return sp.log(self.pdf(value))

    def _cdf(self, value):
        x = _s(value)
        n = self.sample_size
        r = self.rank
        F = self.base.cdf(x)
        return sp.simplify(
            sum(sp.binomial(n, j) * F**j * (1 - F) ** (n - j) for j in range(r, n + 1))
        )


def order_statistic(base: Distribution, sample_size: int, rank: int) -> Distribution:
    """Construct an order-statistic distribution, recognizing selected families."""
    n = integer(sample_size, name="sample_size", minimum=1)
    r = integer(rank, name="rank", minimum=1, maximum=n)
    if isinstance(base, Uniform) and r >= 1 and r <= n:
        # Standard uniform order statistic is Beta; general uniform is affine Beta.
        beta = Beta(r, n - r + 1)
        if (
            certified_equal(base.low, 0) is TruthValue.TRUE
            and certified_equal(base.high, 1) is TruthValue.TRUE
        ):
            return beta
    return OrderStatisticDistribution(base, n, r)


def recognize_affine_matrix(distribution: Distribution, matrix: Any, shift: Any = None):
    """Recognize ``A X + b`` for vector Gaussian laws."""
    if not isinstance(distribution, MultivariateNormal):
        return None
    A = sp.ImmutableMatrix(matrix)
    mu = sp.ImmutableMatrix(distribution.mean)
    cov = sp.ImmutableMatrix(distribution.covariance)
    if A.cols != mu.rows:
        raise ValueError("affine matrix has incompatible input dimension")
    b = sp.zeros(A.rows, 1) if shift is None else sp.ImmutableMatrix(shift)
    if b.shape == (1, A.rows):
        b = b.T
    if b.shape != (A.rows, 1):
        raise ValueError("shift has incompatible output dimension")
    return MultivariateNormal(sp.simplify(A * mu + b), sp.simplify(A * cov * A.T))


def _distribution_of_independent(
    expression: Any, distributions: Mapping[sp.Symbol, Distribution]
) -> Distribution:
    """Infer the law of an expression of independent named random variables.

    Distinct mapping keys are assumed mutually independent.  Exact closed-family
    rules are preferred; unsupported one-variable expressions become exact
    transformed distributions when SymPy can solve their inverse.
    """
    expr = sp.sympify(expression)
    involved = expr.free_symbols & set(distributions)
    if not involved:
        raise ValueError("expression contains no mapped random variable")
    if len(involved) == 1:
        x = next(iter(involved))
        d = distributions[x]
        if expr == x:
            return d
        if expr.is_polynomial(x) and sp.Poly(expr, x).degree() <= 1:
            p = sp.Poly(expr, x)
            a = p.coeff_monomial(x)
            b = p.coeff_monomial(1)
            ans = recognize_affine(d, a, b)
            if ans.recognized:
                return ans.distribution
        if expr == sp.exp(x) and isinstance(d, Normal):
            return LogNormal(d.mean, d.sigma)
    if isinstance(expr, sp.Add):
        laws = []
        for term in expr.args:
            syms = term.free_symbols & set(distributions)
            if not syms:
                continue
            laws.append(_distribution_of_independent(term, distributions))
        closed = recognize_sum(*laws) if laws else None
        const = sp.simplify(expr.subs({s: 0 for s in involved}))
        if closed and closed.recognized:
            if const != 0:
                aff = recognize_affine(closed.distribution, 1, const)
                if aff.recognized:
                    return aff.distribution
            return closed.distribution
    if isinstance(expr, sp.Mul):
        const, rest = expr.as_coeff_Mul()
        if rest != expr and len(rest.free_symbols & set(distributions)) == 1:
            d = _distribution_of_independent(rest, distributions)
            aff = recognize_affine(d, const, 0)
            if aff.recognized:
                return aff.distribution
        laws = []
        for factor in expr.args:
            if factor.free_symbols & set(distributions):
                laws.append(distribution_of(factor, distributions))
        if laws:
            closed = recognize_product(*laws)
            if closed.recognized:
                return closed.distribution
    raise NotImplementedError(f"no exact distribution rule for expression {expr}")


@dataclass(frozen=True, slots=True)
class JointBlock:
    """Named random variables governed by one joint distribution."""

    variables: tuple[sp.Symbol, ...]
    distribution: Distribution

    def __post_init__(self):
        variables = tuple(map(sp.sympify, self.variables))
        if not variables or not all(isinstance(v, sp.Symbol) for v in variables):
            raise TypeError("joint-block variables must be SymPy Symbols")
        if len(set(variables)) != len(variables):
            raise ValueError("joint-block variables must be distinct")
        shape = self.distribution.event_space.shape
        if shape != (len(variables),):
            raise ValueError(
                "joint distribution event shape must match block variables"
            )
        object.__setattr__(self, "variables", variables)


@dataclass(frozen=True, slots=True)
class DistributionContext:
    """Dependency-aware collection of random sources.

    Scalar ``independent`` entries are mutually independent and independent of
    every joint block.  Variables within a :class:`JointBlock` have the
    dependence encoded by that block's joint distribution.
    """

    independent: Mapping[sp.Symbol, Distribution]
    joint_blocks: tuple[JointBlock, ...] = ()

    def __post_init__(self):
        independent = {sp.sympify(k): v for k, v in self.independent.items()}
        if not all(isinstance(k, sp.Symbol) for k in independent):
            raise TypeError("context keys must be SymPy Symbols")
        blocks = tuple(self.joint_blocks)
        seen = set(independent)
        for block in blocks:
            overlap = seen.intersection(block.variables)
            if overlap:
                raise ValueError(
                    f"random variables occur in multiple dependency blocks: {overlap}"
                )
            seen.update(block.variables)
        object.__setattr__(self, "independent", independent)
        object.__setattr__(self, "joint_blocks", blocks)

    @classmethod
    def from_independent(cls, mapping):
        return cls(mapping, ())

    @classmethod
    def from_joint(cls, variables, distribution):
        return cls({}, (JointBlock(tuple(variables), distribution),))

    @property
    def variables(self):
        out = set(self.independent)
        for block in self.joint_blocks:
            out.update(block.variables)
        return frozenset(out)

    def block_for(self, variable):
        variable = sp.sympify(variable)
        if variable in self.independent:
            return (variable,), self.independent[variable]
        for block in self.joint_blocks:
            if variable in block.variables:
                return block.variables, block.distribution
        raise KeyError(variable)


def _linear_coefficients(expr, variables):
    """Return (coefficients, constant) for an affine expression, else None."""
    variables = tuple(variables)
    try:
        poly = sp.Poly(sp.expand(expr), *variables)
    except sp.PolynomialError:
        return None
    if poly.total_degree() > 1:
        return None
    coeffs = [sp.simplify(sp.diff(expr, v)) for v in variables]
    if any(c.free_symbols.intersection(variables) for c in coeffs):
        return None
    constant = sp.simplify(expr - sum(c * v for c, v in zip(coeffs, variables)))
    if constant.free_symbols.intersection(variables):
        return None
    return coeffs, constant


def _distribution_of_joint_affine(expr, block: JointBlock):
    if not isinstance(block.distribution, MultivariateNormal):
        return None
    affine = _linear_coefficients(expr, block.variables)
    if affine is None:
        return None
    coeffs, constant = affine
    a = sp.ImmutableMatrix([coeffs])
    mu = sp.ImmutableMatrix(block.distribution.mean)
    cov = sp.ImmutableMatrix(block.distribution.covariance)
    mean = sp.simplify((a * mu)[0] + constant)
    variance = sp.simplify((a * cov * a.T)[0])
    return Normal(mean, sp.sqrt(variance))


def distribution_of(expression: Any, distributions) -> Distribution:
    """Infer an exact law with explicit dependency semantics.

    A plain mapping treats distinct keys as independent. A
    :class:`DistributionContext` can instead encode correlated joint blocks.
    Linear forms of a multivariate-normal block are recognized exactly,
    including covariance terms.  One-source nonlinear expressions fall back to
    automatic branch-aware symbolic transform inversion when possible.
    """
    expr = sp.sympify(expression)
    if not isinstance(distributions, DistributionContext):
        mapping = distributions
        try:
            return _distribution_of_independent(expr, mapping)
        except NotImplementedError:
            involved = expr.free_symbols & set(mapping)
            if len(involved) == 1:
                x = next(iter(involved))
                return transformed_distribution(mapping[x], expr, x)
            raise

    context = distributions
    involved = expr.free_symbols & set(context.variables)
    if not involved:
        raise ValueError("expression contains no random variable from context")

    # A linear expression contained in one correlated block is exact.
    for block in context.joint_blocks:
        if involved.issubset(set(block.variables)):
            answer = _distribution_of_joint_affine(expr, block)
            if answer is not None:
                return answer

    # Decompose additive expressions by *dependency blocks*, never by raw terms.
    # This preserves covariance when several terms involve variables from one
    # correlated block, while different blocks remain independent.
    if isinstance(expr, sp.Add):
        deterministic = sp.S.Zero
        grouped: dict[tuple[sp.Symbol, ...], sp.Expr] = {}
        distributions_by_group = {}
        for term in expr.args:
            term_vars = term.free_symbols & set(context.variables)
            if not term_vars:
                deterministic += term
                continue
            matching = []
            for block in context.joint_blocks:
                if term_vars.intersection(block.variables):
                    matching.append((block.variables, block.distribution))
            independent_vars = term_vars.intersection(context.independent)
            if matching and independent_vars:
                raise NotImplementedError(
                    "one additive term spans dependent and independent blocks"
                )
            if len(matching) > 1:
                raise NotImplementedError(
                    "one additive term spans multiple dependency blocks"
                )
            if matching:
                key, law = matching[0]
            elif len(independent_vars) == 1:
                v = next(iter(independent_vars))
                key, law = (v,), context.independent[v]
            else:
                raise NotImplementedError("term dependency structure is not separable")
            grouped[key] = sp.simplify(grouped.get(key, 0) + term)
            distributions_by_group[key] = law

        random_laws = []
        for key, subexpr in grouped.items():
            law = distributions_by_group[key]
            if len(key) > 1:
                block = JointBlock(key, law)
                answer = _distribution_of_joint_affine(subexpr, block)
                if answer is None:
                    raise NotImplementedError(
                        f"no exact rule for dependent block expression {subexpr}"
                    )
                random_laws.append(answer)
            else:
                v = key[0]
                try:
                    random_laws.append(_distribution_of_independent(subexpr, {v: law}))
                except NotImplementedError:
                    random_laws.append(transformed_distribution(law, subexpr, v))

        closed = recognize_sum(*random_laws)
        base = closed.distribution if closed.recognized else convolution(*random_laws)
        if deterministic != 0:
            shifted = recognize_affine(base, 1, deterministic)
            if shifted.recognized:
                return shifted.distribution
        return base

    # For a single source, use exact scalar algebra before symbolic inversion. For one
    # coordinate of an MVN joint block, derive its exact Normal marginal first.
    if len(involved) == 1:
        x = next(iter(involved))
        if x in context.independent:
            base = context.independent[x]
        else:
            block_vars, joint = context.block_for(x)
            marginal = _distribution_of_joint_affine(x, JointBlock(block_vars, joint))
            if marginal is None:
                raise NotImplementedError("joint marginal is not recognized")
            base = marginal
        try:
            return _distribution_of_independent(expr, {x: base})
        except NotImplementedError:
            return transformed_distribution(base, expr, x)

    raise NotImplementedError(f"no exact dependency-aware rule for expression {expr}")


__all__ = [
    "DistributionContext",
    "JointBlock",
    "OrderStatisticDistribution",
    "convolution",
    "distribution_of",
    "order_statistic",
    "recognize_affine_matrix",
]
