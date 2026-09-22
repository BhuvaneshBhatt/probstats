"""Distribution transforms.

Transforms are first-class composition objects.  Scalar transforms may be
monotone (increasing or decreasing) or provide finitely many inverse branches.
Vector transforms use an explicit inverse map and Jacobian determinant.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import sympy as sp

from ._symbolic_predicates import TruthValue, certified_equal
from .distributions.base import Distribution
from .spaces import EventSpace, MeasureType, ScalarEventSpace, VectorEventSpace


@dataclass(frozen=True, slots=True)
class InverseBranch:
    """One inverse branch of a scalar transform."""

    inverse: sp.Expr
    inverse_variable: sp.Symbol
    domain: sp.Set = sp.S.Reals

    def __post_init__(self):
        object.__setattr__(self, "inverse", sp.sympify(self.inverse))
        object.__setattr__(self, "inverse_variable", sp.sympify(self.inverse_variable))
        if not isinstance(self.inverse_variable, sp.Symbol):
            raise TypeError("inverse_variable must be a SymPy Symbol")
        if not isinstance(self.domain, sp.Set):
            raise TypeError("branch domain must be a SymPy Set")

    def apply(self, value):
        return self.inverse.subs(self.inverse_variable, sp.sympify(value))

    def jacobian_abs(self, value):
        y = sp.sympify(value)
        derivative = sp.diff(self.inverse, self.inverse_variable)
        return sp.simplify(sp.Abs(derivative.subs(self.inverse_variable, y)))


@dataclass(frozen=True, slots=True)
class Transform:
    """Scalar transform with an explicit inverse.

    ``orientation`` is +1 for increasing, -1 for decreasing, and 0 when no
    global monotonicity is asserted.  ``inverse_branches`` enables finite
    many-to-one transforms such as ``x -> x**2``.
    """

    forward: sp.Expr
    variable: sp.Symbol
    inverse: sp.Expr
    inverse_variable: sp.Symbol
    domain: sp.Set = sp.S.Reals
    codomain: sp.Set = sp.S.Reals
    orientation: int = 1
    inverse_branches: tuple[InverseBranch, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "forward", sp.sympify(self.forward))
        object.__setattr__(self, "variable", sp.sympify(self.variable))
        object.__setattr__(self, "inverse", sp.sympify(self.inverse))
        object.__setattr__(self, "inverse_variable", sp.sympify(self.inverse_variable))
        if not isinstance(self.variable, sp.Symbol) or not isinstance(
            self.inverse_variable, sp.Symbol
        ):
            raise TypeError("transform variables must be SymPy Symbols")
        if not isinstance(self.domain, sp.Set) or not isinstance(self.codomain, sp.Set):
            raise TypeError("transform domain/codomain must be SymPy Sets")
        if self.orientation not in (-1, 0, 1):
            raise ValueError("orientation must be -1, 0, or 1")
        object.__setattr__(self, "inverse_branches", tuple(self.inverse_branches))

    @classmethod
    def from_expressions(
        cls,
        forward,
        variable,
        inverse,
        inverse_variable,
        *,
        domain=sp.S.Reals,
        codomain=sp.S.Reals,
        orientation=1,
    ):
        return cls(
            sp.sympify(forward),
            sp.sympify(variable),
            sp.sympify(inverse),
            sp.sympify(inverse_variable),
            domain,
            codomain,
            orientation,
        )

    @classmethod
    def from_branches(
        cls,
        forward,
        variable,
        inverse_variable,
        branches: Sequence[InverseBranch],
        *,
        domain=sp.S.Reals,
        codomain=sp.S.Reals,
    ):
        branches = tuple(branches)
        if not branches:
            raise ValueError("at least one inverse branch is required")
        # Scalar inverse access uses the first verified branch.
        return cls(
            sp.sympify(forward),
            sp.sympify(variable),
            branches[0].inverse,
            sp.sympify(inverse_variable),
            domain,
            codomain,
            0,
            branches,
        )

    def apply(self, value):
        return self.forward.subs(self.variable, sp.sympify(value))

    def invert(self, value):
        if self.inverse_branches:
            if len(self.inverse_branches) != 1:
                raise ValueError("transform has multiple inverse branches")
            return self.inverse_branches[0].apply(value)
        return self.inverse.subs(self.inverse_variable, sp.sympify(value))

    def inverse_jacobian_abs(self, value):
        y = sp.sympify(value)
        derivative = sp.diff(self.inverse, self.inverse_variable)
        return sp.simplify(sp.Abs(derivative.subs(self.inverse_variable, y)))

    @property
    def branches(self):
        if self.inverse_branches:
            return self.inverse_branches
        return (InverseBranch(self.inverse, self.inverse_variable, self.codomain),)


@dataclass(frozen=True, slots=True)
class MultivariateTransform:
    """Bijective vector transform with explicit inverse map."""

    forward: sp.ImmutableDenseMatrix
    variables: tuple[sp.Symbol, ...]
    inverse: sp.ImmutableDenseMatrix
    inverse_variables: tuple[sp.Symbol, ...]
    domain: EventSpace
    codomain: EventSpace

    def __init__(
        self,
        forward,
        variables,
        inverse,
        inverse_variables,
        *,
        domain=None,
        codomain=None,
    ):
        vars_ = tuple(map(sp.sympify, variables))
        inv_vars = tuple(map(sp.sympify, inverse_variables))
        fwd = sp.ImmutableMatrix(forward)
        inv = sp.ImmutableMatrix(inverse)
        if (
            not vars_
            or len(vars_) != len(inv_vars)
            or fwd.rows != len(vars_)
            or inv.rows != len(vars_)
        ):
            raise ValueError("forward/inverse maps and variable dimensions must agree")
        if fwd.cols != 1 or inv.cols != 1:
            raise ValueError("forward and inverse maps must be column vectors")
        if not all(isinstance(v, sp.Symbol) for v in (*vars_, *inv_vars)):
            raise TypeError("transform variables must be SymPy Symbols")
        d = domain or VectorEventSpace(len(vars_), sp.S.Reals)
        c = codomain or VectorEventSpace(len(vars_), sp.S.Reals)
        object.__setattr__(self, "forward", fwd)
        object.__setattr__(self, "variables", vars_)
        object.__setattr__(self, "inverse", inv)
        object.__setattr__(self, "inverse_variables", inv_vars)
        object.__setattr__(self, "domain", d)
        object.__setattr__(self, "codomain", c)

    def apply(self, value):
        vals = tuple(map(sp.sympify, value))
        if len(vals) != len(self.variables):
            raise ValueError("transform value has wrong dimension")
        return sp.ImmutableMatrix(
            [expr.subs(dict(zip(self.variables, vals))) for expr in self.forward]
        )

    def invert(self, value):
        vals = tuple(map(sp.sympify, value))
        if len(vals) != len(self.inverse_variables):
            raise ValueError("inverse-transform value has wrong dimension")
        return sp.ImmutableMatrix(
            [
                expr.subs(dict(zip(self.inverse_variables, vals)))
                for expr in self.inverse
            ]
        )

    def inverse_jacobian_abs(self, value):
        vals = tuple(map(sp.sympify, value))
        subs = dict(zip(self.inverse_variables, vals))
        jac = self.inverse.jacobian(self.inverse_variables)
        return sp.simplify(sp.Abs(jac.det()).subs(subs))


@dataclass(frozen=True, slots=True)
class TransformedDistribution(Distribution):
    base: Distribution
    transform: Transform | MultivariateTransform

    def __post_init__(self):
        if not isinstance(self.base, Distribution):
            raise TypeError("base must be a Distribution")
        if isinstance(self.transform, Transform):
            if not isinstance(self.base.event_space, ScalarEventSpace):
                raise TypeError("scalar Transform requires a scalar base distribution")
        elif isinstance(self.transform, MultivariateTransform):
            if self.base.event_space.shape != self.transform.domain.shape:
                raise ValueError(
                    "base event shape and transform domain are incompatible"
                )
        else:
            raise TypeError("transform must be Transform or MultivariateTransform")

    @property
    def support(self):
        if isinstance(self.transform, Transform):
            return self.transform.codomain
        return sp.S.UniversalSet

    @property
    def event_space(self):
        if isinstance(self.transform, Transform):
            return ScalarEventSpace(self.support)
        return self.transform.codomain

    @property
    def measure_type(self):
        return self.base.measure_type

    def _entropy(self):
        """Return entropy from a certified bijective change of variables when available."""
        from ._transformation_certification import transformed_entropy_value

        value = transformed_entropy_value(self)
        return NotImplemented if value is None else value

    @property
    def parameters(self):
        return self.base.parameters

    @property
    def parameter_constraints(self):
        return self.base.parameter_constraints

    def pdf(self, value):
        if self.measure_type is MeasureType.DISCRETE:
            return self.pmf(value)
        if isinstance(self.transform, MultivariateTransform):
            y = tuple(value)
            x = self.transform.invert(y)
            return sp.simplify(
                self.base.pdf(x) * self.transform.inverse_jacobian_abs(y)
            )
        y = sp.sympify(value)
        terms = []
        for branch in self.transform.branches:
            x = branch.apply(y)
            valid = sp.And(
                sp.Contains(y, self.transform.codomain),
                sp.Contains(y, branch.domain),
                sp.Contains(x, self.base.support),
            )
            terms.append(
                self.base.pdf(x)
                * branch.jacobian_abs(y)
                * sp.Piecewise((1, valid), (0, True))
            )
        return sp.simplify(sp.Add(*terms))

    def pmf(self, value):
        if self.measure_type is not MeasureType.DISCRETE:
            return super().pmf(value)
        if isinstance(self.transform, MultivariateTransform):
            return sp.simplify(self.base.pmf(self.transform.invert(value)))
        y = sp.sympify(value)
        terms = []
        for branch in self.transform.branches:
            x = branch.apply(y)
            valid = sp.And(
                sp.Contains(y, self.transform.codomain),
                sp.Contains(y, branch.domain),
                sp.Contains(x, self.base.support),
            )
            terms.append(self.base.pmf(x) * sp.Piecewise((1, valid), (0, True)))
        return sp.simplify(sp.Add(*terms))

    def logpdf(self, value):
        return sp.log(self.pdf(value))

    def logpmf(self, value):
        return sp.log(self.pmf(value))

    def _mean(self):
        if isinstance(self.transform, MultivariateTransform):
            return NotImplemented
        from .functionals import expectation

        x = sp.Dummy("x", real=True)
        result = expectation(self.base, self.transform.apply(x), variable=x)
        return sp.simplify(result.rewrite(sp.erf))

    def _variance(self):
        if isinstance(self.transform, MultivariateTransform):
            return NotImplemented
        from .functionals import expectation

        x = sp.Dummy("x", real=True)
        y = self.transform.apply(x)
        mu = self._mean()
        result = expectation(self.base, (y - mu) ** 2, variable=x)
        return sp.simplify(result.rewrite(sp.erf))

    def _cdf(self, value):
        from .functionals import cdf

        if (
            not isinstance(self.transform, Transform)
            or self.transform.orientation == 0
            or self.transform.inverse_branches
        ):
            return NotImplemented
        y = sp.sympify(value)
        x = self.transform.invert(y)
        if self.transform.orientation > 0:
            return cdf(self.base, x)
        return sp.simplify(1 - cdf(self.base, x))

    def _quantile(self, p):
        from .functionals import quantile

        if (
            not isinstance(self.transform, Transform)
            or self.transform.orientation == 0
            or self.transform.inverse_branches
        ):
            return NotImplemented
        q = quantile(self.base, p if self.transform.orientation > 0 else 1 - p)
        return self.transform.apply(q)

    def _expectation(self, expr, variable):
        from .functionals import expectation

        if isinstance(self.transform, MultivariateTransform):
            return NotImplemented
        y = sp.sympify(variable) if variable is not None else sp.Dummy("y", real=True)
        x = sp.Dummy("x", real=True)
        return expectation(self.base, expr.subs(y, self.transform.apply(x)), variable=x)


__all__ = [
    "InverseBranch",
    "MultivariateTransform",
    "Transform",
    "TransformedDistribution",
]


@dataclass(frozen=True, slots=True)
class CorrelationCholeskyTransform:
    """Biject unconstrained vectors to correlation Cholesky factors.

    The unconstrained coordinates are mapped to partial correlations with ``tanh``
    and then assembled row-by-row using the standard hyperspherical construction.
    """

    dimension: int

    def __post_init__(self):
        if not isinstance(self.dimension, int) or self.dimension < 2:
            raise ValueError("dimension must be an integer >= 2")

    @property
    def unconstrained_dimension(self):
        return self.dimension * (self.dimension - 1) // 2

    @property
    def domain(self):
        return VectorEventSpace(self.unconstrained_dimension, sp.S.Reals)

    @property
    def codomain(self):
        from .spaces import CorrelationCholeskySpace

        return CorrelationCholeskySpace(self.dimension)

    def apply(self, value):
        ys = tuple(map(sp.sympify, value))
        if len(ys) != self.unconstrained_dimension:
            raise ValueError("unconstrained vector has wrong dimension")
        out = sp.zeros(self.dimension)
        out[0, 0] = 1
        k = 0
        for i in range(1, self.dimension):
            remaining = sp.S.One
            for j in range(i):
                z = sp.tanh(ys[k])
                out[i, j] = sp.simplify(z * sp.sqrt(remaining))
                remaining = sp.simplify(remaining * (1 - z**2))
                k += 1
            out[i, i] = sp.sqrt(remaining)
        return sp.ImmutableMatrix(out)

    def invert(self, value):
        chol = sp.ImmutableMatrix(value)
        if chol.shape != (self.dimension, self.dimension):
            raise ValueError("Cholesky factor has wrong shape")
        ys = []
        for i in range(1, self.dimension):
            remaining = sp.S.One
            for j in range(i):
                z = sp.simplify(chol[i, j] / sp.sqrt(remaining))
                ys.append(sp.atanh(z))
                remaining = sp.simplify(remaining * (1 - z**2))
        return sp.ImmutableMatrix(ys)

    def apply_numeric(self, value):
        import numpy as np

        ys = np.asarray(value, dtype=float).reshape(-1)
        if ys.size != self.unconstrained_dimension:
            raise ValueError("unconstrained vector has wrong dimension")
        out = np.zeros((self.dimension, self.dimension), dtype=float)
        out[0, 0] = 1.0
        k = 0
        for i in range(1, self.dimension):
            remaining = 1.0
            for j in range(i):
                z = np.tanh(ys[k])
                out[i, j] = z * np.sqrt(remaining)
                remaining *= 1.0 - z * z
                k += 1
            out[i, i] = np.sqrt(max(remaining, 0.0))
        return out

    def invert_numeric(self, value):
        import numpy as np

        chol = np.asarray(value, dtype=float)
        if chol.shape != (self.dimension, self.dimension):
            raise ValueError("Cholesky factor has wrong shape")
        ys = []
        for i in range(1, self.dimension):
            remaining = 1.0
            for j in range(i):
                z = chol[i, j] / np.sqrt(remaining)
                z = np.clip(z, -1 + 1e-15, 1 - 1e-15)
                ys.append(np.arctanh(z))
                remaining *= 1.0 - z * z
        return np.asarray(ys)

    def log_abs_det_jacobian(self, value):
        ys = tuple(map(sp.sympify, value))
        if len(ys) != self.unconstrained_dimension:
            raise ValueError("unconstrained vector has wrong dimension")
        total = sp.S.Zero
        k = 0
        for i in range(1, self.dimension):
            for j in range(i):
                z = sp.tanh(ys[k])
                exponent = sp.Rational(i - j + 1, 2)
                total += exponent * sp.log(1 - z**2)
                k += 1
        return sp.simplify(total)


@dataclass(frozen=True, slots=True)
class CorrelationMatrixTransform:
    """Biject unconstrained vectors to positive-definite correlation matrices."""

    dimension: int

    def __post_init__(self):
        if not isinstance(self.dimension, int) or self.dimension < 2:
            raise ValueError("dimension must be an integer >= 2")

    @property
    def cholesky_transform(self):
        return CorrelationCholeskyTransform(self.dimension)

    @property
    def unconstrained_dimension(self):
        return self.cholesky_transform.unconstrained_dimension

    @property
    def domain(self):
        return self.cholesky_transform.domain

    @property
    def codomain(self):
        from .spaces import CorrelationMatrixSpace

        return CorrelationMatrixSpace(self.dimension)

    def apply(self, value):
        chol = self.cholesky_transform.apply(value)
        return sp.simplify(chol * chol.T)

    def invert(self, value):
        r = sp.ImmutableMatrix(value)
        if r.shape != (self.dimension, self.dimension):
            raise ValueError("correlation matrix has wrong shape")
        return self.cholesky_transform.invert(r.cholesky())

    def apply_numeric(self, value):
        chol = self.cholesky_transform.apply_numeric(value)
        return chol @ chol.T

    def invert_numeric(self, value):
        import numpy as np

        r = np.asarray(value, dtype=float)
        return self.cholesky_transform.invert_numeric(np.linalg.cholesky(r))

    def log_abs_det_jacobian(self, value):
        ys = tuple(map(sp.sympify, value))
        chol = self.cholesky_transform.apply(ys)
        extra = sp.S.Zero
        # Row-wise L -> R mapping is triangular; each new correlation row
        # contributes the determinant of the preceding Cholesky block.
        for i in range(1, self.dimension):
            for k in range(1, i):
                extra += sp.log(chol[k, k])
        return sp.simplify(self.cholesky_transform.log_abs_det_jacobian(ys) + extra)


__all__.extend(["CorrelationCholeskyTransform", "CorrelationMatrixTransform"])


def _monotonic_orientation(expression, variable, domain):
    """Return the proven monotonic orientation, or zero when it is unresolved."""
    from sympy.calculus.singularities import is_decreasing, is_increasing

    derivative = sp.diff(expression, variable)
    if derivative.is_positive is True:
        return 1
    if derivative.is_negative is True:
        return -1
    try:
        if is_increasing(expression, domain, variable) is True:
            return 1
        if is_decreasing(expression, domain, variable) is True:
            return -1
    except (NotImplementedError, TypeError, ValueError):
        return 0
    return 0


def invert_transform(
    forward,
    variable,
    *,
    inverse_variable=None,
    domain=sp.S.Reals,
    codomain=None,
):
    """Construct a scalar :class:`Transform` by symbolically inverting ``forward``.

    Finite inverse sets become branch-aware transforms automatically.  The
    codomain is inferred with SymPy's ``function_range`` when possible.  This
    is exact: unresolved/implicit inverse sets raise  ``NotImplementedError``
    rather than inventing a local inverse.
    """
    from sympy.calculus.util import function_range

    x = sp.sympify(variable)
    if not isinstance(x, sp.Symbol):
        raise TypeError("variable must be a SymPy Symbol")
    y = (
        sp.Symbol("y", real=True)
        if inverse_variable is None
        else sp.sympify(inverse_variable)
    )
    if not isinstance(y, sp.Symbol):
        raise TypeError("inverse_variable must be a SymPy Symbol")
    f = sp.sympify(forward)
    if codomain is None:
        try:
            codomain = function_range(f, x, domain)
        except (ValueError, NotImplementedError):
            codomain = sp.S.Reals
    if not isinstance(codomain, sp.Set):
        raise TypeError("codomain must be a SymPy Set")

    # ``solve`` is preferable here for real symbolic inversion: ``solveset``
    # over Complexes may encode periodic complex branches (e.g. exp/log) even
    # though the declared source domain is real.  Every candidate is verified.
    try:
        raw_solutions = sp.solve(sp.Eq(f, y), x)
    except (NotImplementedError, ValueError):
        raw_solutions = []
    solutions = tuple(
        sp.simplify(sol)
        for sol in raw_solutions
        if certified_equal(f.subs(x, sol), y) is TruthValue.TRUE
    )
    if not solutions:
        solved = sp.solveset(sp.Eq(f, y), x, domain=sp.S.Reals)
        if isinstance(solved, sp.FiniteSet):
            solutions = tuple(sp.simplify(sol) for sol in solved)
    if not solutions:
        raise NotImplementedError(
            "transform inverse is not a finite explicit branch set"
        )

    # Single-valued inverses retain monotonic orientation when SymPy can prove it.
    if len(solutions) == 1:
        orientation = _monotonic_orientation(f, x, domain)
        return Transform.from_expressions(
            f,
            x,
            solutions[0],
            y,
            domain=domain,
            codomain=codomain,
            orientation=orientation,
        )

    branches = tuple(InverseBranch(sol, y, codomain) for sol in solutions)
    return Transform.from_branches(
        f,
        x,
        y,
        branches,
        domain=domain,
        codomain=codomain,
    )


def transformed_distribution(
    base: Distribution, forward, variable, *, domain=None, codomain=None
):
    """Create a transformed law by automatically solving inverse branches."""
    dom = base.support if domain is None else domain
    transform = invert_transform(
        forward,
        variable,
        domain=dom,
        codomain=codomain,
    )
    return TransformedDistribution(base, transform)


__all__.extend(["invert_transform", "transformed_distribution"])
