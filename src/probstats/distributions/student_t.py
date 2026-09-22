"""Student-t distributions used by conjugate posterior predictives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sympy as sp

from .._exact_linear_algebra import quadratic_form
from .._special import t_cdf, t_ppf
from ..spaces import (
    MeasureType,
    ParameterSpace,
    PositiveDefiniteMatrixSpace,
    PositiveRealSpace,
    RealSpace,
    VectorEventSpace,
)
from .base import Distribution, scalar_batch


@dataclass(frozen=True, slots=True)
class StudentT(Distribution):
    location: sp.Expr
    scale: sp.Expr
    df: sp.Expr

    def __post_init__(self):
        object.__setattr__(self, "location", sp.sympify(self.location))
        object.__setattr__(self, "scale", sp.sympify(self.scale))
        object.__setattr__(self, "df", sp.sympify(self.df))
        if self.scale.is_nonpositive is True:
            raise ValueError("StudentT scale must be positive.")
        if self.df.is_nonpositive is True:
            raise ValueError("StudentT degrees of freedom must be positive.")

    @property
    def support(self):
        return sp.S.Reals

    @property
    def parameters(self):
        return (self.location, self.scale, self.df)

    @property
    def parameter_constraints(self):
        return sp.And(self.scale > 0, self.df > 0)

    @property
    def parameter_space(self):
        return ParameterSpace(
            (
                ("location", RealSpace()),
                ("scale", PositiveRealSpace()),
                ("df", PositiveRealSpace()),
            )
        )

    @scalar_batch
    def logpdf(self, value: Any):
        x = sp.sympify(value)
        v = self.df
        s = self.scale
        z = (x - self.location) / s
        return sp.simplify(
            sp.loggamma((v + 1) / 2)
            - sp.loggamma(v / 2)
            - sp.log(sp.sqrt(v * sp.pi) * s)
            - (v + 1) / 2 * sp.log(1 + z**2 / v)
        )

    def _cdf(self, value):
        if all(parameter.is_number for parameter in self.parameters):
            z = float((sp.sympify(value) - self.location) / self.scale)
            return sp.Float(t_cdf(z, float(self.df)))
        return NotImplemented

    def _quantile(self, probability):
        if (
            all(parameter.is_number for parameter in self.parameters)
            and sp.sympify(probability).is_number
        ):
            p = float(probability)
            if not 0 <= p <= 1:
                raise ValueError("probability must lie in [0, 1]")
            return sp.Float(
                float(self.location) + float(self.scale) * t_ppf(p, float(self.df))
            )
        return NotImplemented

    def _mean(self):
        return sp.Piecewise((self.location, self.df > 1), (sp.nan, True))

    def _variance(self):
        return sp.Piecewise(
            (self.scale**2 * self.df / (self.df - 2), self.df > 2),
            (sp.oo, sp.And(self.df > 1, self.df <= 2)),
            (sp.nan, True),
        )

    def _entropy(self):
        v = self.df
        return sp.log(sp.sqrt(v) * sp.beta(v / 2, sp.Rational(1, 2)) * self.scale) + (
            v + 1
        ) / 2 * (sp.polygamma(0, (v + 1) / 2) - sp.polygamma(0, v / 2))


@dataclass(frozen=True, slots=True)
class MultivariateStudentT(Distribution):
    """Multivariate Student-t with scale matrix (not covariance matrix)."""

    location: sp.ImmutableDenseMatrix
    scale: sp.ImmutableDenseMatrix
    df: sp.Expr

    def __init__(self, location, scale, df):
        loc = sp.ImmutableMatrix(location)
        if loc.cols != 1:
            loc = sp.ImmutableMatrix(list(location))
        sc = sp.ImmutableMatrix(scale)
        v = sp.sympify(df)
        if sc.rows != sc.cols or loc.cols != 1 or loc.rows != sc.rows:
            raise ValueError("location and scale dimensions are incompatible")
        if v.is_nonpositive is True:
            raise ValueError(
                "MultivariateStudentT degrees of freedom must be positive."
            )
        if sc != sc.T:
            raise ValueError("MultivariateStudentT scale matrix must be symmetric.")
        if not sc.free_symbols and sc.is_positive_definite is not True:
            raise ValueError(
                "MultivariateStudentT scale matrix must be positive definite."
            )
        object.__setattr__(self, "location", loc)
        object.__setattr__(self, "scale", sc)
        object.__setattr__(self, "df", v)

    @property
    def dimension(self):
        return self.location.rows

    def marginal(self, index: int) -> StudentT:
        """Return the univariate Student-t marginal for one component."""
        if not 0 <= index < self.dimension:
            raise IndexError("component index out of range")
        return StudentT(
            self.location[index],
            sp.sqrt(self.scale[index, index]),
            self.df,
        )

    @property
    def support(self):
        return sp.ProductSet(*([sp.S.Reals] * self.dimension))

    @property
    def event_space(self):
        return VectorEventSpace(self.dimension, sp.S.Reals)

    @property
    def measure_type(self):
        return MeasureType.CONTINUOUS

    @property
    def parameters(self):
        return tuple(self.location) + tuple(self.scale) + (self.df,)

    @property
    def parameter_constraints(self):
        return self.df > 0

    @property
    def parameter_space(self):
        return ParameterSpace(
            (
                ("location", VectorEventSpace(self.dimension, sp.S.Reals)),
                ("scale", PositiveDefiniteMatrixSpace(self.dimension)),
                ("df", PositiveRealSpace()),
            )
        )

    @property
    def parameter_space_constraints(self):
        return self.parameter_space.constraints((self.location, self.scale, self.df))

    @scalar_batch
    def logpdf(self, value):
        x = sp.ImmutableMatrix(value)
        if x.cols != 1:
            x = sp.ImmutableMatrix(list(value))
        if x.rows != self.dimension:
            raise ValueError("value has wrong dimension")
        d = self.dimension
        delta = x - self.location
        quad = quadratic_form(self.scale, delta)
        return sp.simplify(
            sp.loggamma((self.df + d) / 2)
            - sp.loggamma(self.df / 2)
            - sp.Rational(d, 2) * sp.log(self.df * sp.pi)
            - sp.Rational(1, 2) * sp.log(self.scale.det())
            - (self.df + d) / 2 * sp.log(1 + quad / self.df)
        )

    def _cdf(self, value):
        if all(parameter.is_number for parameter in self.parameters):
            z = float((sp.sympify(value) - self.location) / self.scale)
            return sp.Float(t_cdf(z, float(self.df)))
        return NotImplemented

    def _quantile(self, probability):
        if (
            all(parameter.is_number for parameter in self.parameters)
            and sp.sympify(probability).is_number
        ):
            p = float(probability)
            if not 0 <= p <= 1:
                raise ValueError("probability must lie in [0, 1]")
            return sp.Float(
                float(self.location) + float(self.scale) * t_ppf(p, float(self.df))
            )
        return NotImplemented

    def _mean(self):
        return sp.Piecewise((self.location, self.df > 1), (sp.nan, True))

    def _variance(self):
        return sp.Piecewise(
            (self.df / (self.df - 2) * self.scale, self.df > 2), (sp.nan, True)
        )
