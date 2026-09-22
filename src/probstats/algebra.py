"""Symbolic distribution canonicalization and closed-family recognition.

The rules in this module reason structurally about probability laws without
sampling or integration. Public recognizers return ``RecognitionResult`` values
that record the rule and side conditions supporting each result.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import sympy as sp

from ._symbolic_predicates import TruthValue, certified_equal
from .composition import MixtureDistribution, ProductDistribution, TruncatedDistribution
from .distributions import (
    Bernoulli,
    BetaPrime,
    Binomial,
    Cauchy,
    ChiSquared,
    ExponentialFamily,
    FDistribution,
    Gamma,
    Geometric,
    Laplace,
    Logistic,
    LogNormal,
    MultivariateNormal,
    NegativeBinomial,
    NoncentralChiSquared,
    NoncentralF,
    NoncentralT,
    Normal,
    Poisson,
    StudentT,
    Uniform,
)
from .distributions.base import Distribution
from .spaces import MeasureType
from .transforms import Transform, TransformedDistribution


class RecognitionKind(str, Enum):
    IDENTITY = "identity"
    CANONICALIZATION = "canonicalization"
    AFFINE = "affine"
    TRANSFORM = "transform"
    SUM = "sum"
    PRODUCT = "product"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class RecognitionResult:
    """Result of a distribution-recognition rule."""

    distribution: Distribution | None
    kind: RecognitionKind
    rule: str
    exact: bool = True
    conditions: sp.Basic = sp.true

    @property
    def recognized(self) -> bool:
        return self.distribution is not None


@dataclass(frozen=True, slots=True)
class ExponentialFamilyMetadata:
    """Canonical exponential-family information suitable for planners."""

    family: str
    natural_parameters: tuple[sp.Expr, ...]
    natural_constraints: sp.Basic
    sufficient_statistics: tuple[sp.Expr, ...]
    base_measure: sp.Expr
    log_partition: sp.Expr
    parameter_constraints: sp.Basic
    measure_type: MeasureType
    event_shape: tuple[int, ...]

    @property
    def natural_dimension(self) -> int:
        return len(self.natural_parameters)


@dataclass(frozen=True, slots=True)
class ConjugacySignature:
    """Structural signature used by conjugacy registries and inference planners."""

    likelihood_family: str
    conjugate_prior_family: str | None
    statistic_names: tuple[str, ...]
    natural_dimension: int | None


@dataclass(frozen=True, slots=True)
class DistributionMetadata:
    """Stable high-level traits for planner dispatch."""

    family: str
    measure_type: MeasureType
    event_shape: tuple[int, ...]
    is_exponential_family: bool
    is_location_scale_family: bool
    is_compound: bool
    is_mixture: bool
    is_truncated: bool
    positive_support: bool | None
    conjugacy: ConjugacySignature


def _simp(expr: Any) -> sp.Expr:
    return sp.simplify(sp.sympify(expr))


def _same(a: Any, b: Any) -> bool:
    try:
        return certified_equal(a, b) is TruthValue.TRUE
    except (TypeError, ValueError):
        return a == b


def _event_shape(distribution: Distribution) -> tuple[int, ...]:
    shape = distribution.event_space.shape
    return tuple(shape) if shape else ()


def _scale_positive(scale: sp.Expr) -> bool | None:
    s = sp.sympify(scale)
    if s.is_positive is True:
        return True
    if s.is_negative is True:
        return False
    if s.is_zero is True:
        return False
    return None


def canonicalize_distribution(distribution: Distribution) -> Distribution:
    """Return a deterministic canonical representation when an exact rule exists."""
    if not isinstance(distribution, Distribution):
        raise TypeError("distribution must be a Distribution")

    if isinstance(distribution, ProductDistribution):
        components = tuple(
            canonicalize_distribution(d) for d in distribution.components
        )
        return ProductDistribution(components)
    if isinstance(distribution, MixtureDistribution):
        components = tuple(
            canonicalize_distribution(d) for d in distribution.components
        )
        return MixtureDistribution(components, tuple(map(_simp, distribution.weights)))
    if isinstance(distribution, TruncatedDistribution):
        return TruncatedDistribution(
            canonicalize_distribution(distribution.base), distribution.truncation
        )

    if isinstance(distribution, ChiSquared):
        return Gamma(_simp(distribution.df / 2), sp.Integer(2))
    if isinstance(distribution, FDistribution):
        return BetaPrime(
            _simp(distribution.df1 / 2),
            _simp(distribution.df2 / 2),
            _simp(distribution.df2 / distribution.df1),
        )
    if isinstance(distribution, NoncentralChiSquared) and _same(
        distribution.noncentrality, 0
    ):
        return Gamma(_simp(distribution.df / 2), sp.Integer(2))
    if isinstance(distribution, NoncentralF) and _same(distribution.noncentrality, 0):
        return BetaPrime(
            _simp(distribution.df1 / 2),
            _simp(distribution.df2 / 2),
            _simp(distribution.df2 / distribution.df1),
        )
    if isinstance(distribution, NoncentralT) and _same(distribution.noncentrality, 0):
        return StudentT(sp.S.Zero, sp.S.One, _simp(distribution.df))
    if isinstance(distribution, Geometric):
        # Our NB counts failures; Geometric counts trials, so it is an affine NB,
        # not canonicalized to NB to avoid changing support coordinates.
        return Geometric(_simp(distribution.p))

    params = tuple(_simp(p) for p in distribution.parameters)
    try:
        return type(distribution)(*params)
    except (TypeError, ValueError):
        return distribution


def equivalent_distributions(left: Distribution, right: Distribution) -> bool:
    """Prove equality using canonical family/parameter identities when possible."""
    a = canonicalize_distribution(left)
    b = canonicalize_distribution(right)
    if type(a) is not type(b):
        return False
    if len(a.parameters) != len(b.parameters):
        return False
    return all(_same(x, y) for x, y in zip(a.parameters, b.parameters))


def recognize_affine(
    distribution: Distribution, scale: Any, shift: Any = 0
) -> RecognitionResult:
    """Recognize the law of ``scale * X + shift`` exactly when closed."""
    a, b = _simp(scale), _simp(shift)
    if a.is_zero is True:
        return RecognitionResult(
            None,
            RecognitionKind.NONE,
            "degenerate-affine-not-represented",
            conditions=sp.Eq(a, 0),
        )

    if isinstance(distribution, Normal):
        return RecognitionResult(
            Normal(
                _simp(a * distribution.mean + b), _simp(sp.Abs(a) * distribution.sigma)
            ),
            RecognitionKind.AFFINE,
            "normal-affine",
        )
    if isinstance(distribution, Cauchy):
        return RecognitionResult(
            Cauchy(
                _simp(a * distribution.location + b),
                _simp(sp.Abs(a) * distribution.scale),
            ),
            RecognitionKind.AFFINE,
            "cauchy-affine",
        )
    if isinstance(distribution, Laplace):
        return RecognitionResult(
            Laplace(
                _simp(a * distribution.location + b),
                _simp(sp.Abs(a) * distribution.scale),
            ),
            RecognitionKind.AFFINE,
            "laplace-affine",
        )
    if isinstance(distribution, Logistic):
        return RecognitionResult(
            Logistic(
                _simp(a * distribution.location + b),
                _simp(sp.Abs(a) * distribution.scale),
            ),
            RecognitionKind.AFFINE,
            "logistic-affine",
        )
    if isinstance(distribution, StudentT):
        return RecognitionResult(
            StudentT(
                _simp(a * distribution.location + b),
                _simp(sp.Abs(a) * distribution.scale),
                distribution.df,
            ),
            RecognitionKind.AFFINE,
            "student-t-affine",
        )
    if isinstance(distribution, Uniform):
        if _scale_positive(a) is True:
            lo, hi = _simp(a * distribution.low + b), _simp(a * distribution.high + b)
        elif a.is_negative is True:
            lo, hi = _simp(a * distribution.high + b), _simp(a * distribution.low + b)
        else:
            return RecognitionResult(
                None,
                RecognitionKind.NONE,
                "uniform-affine-sign-undecidable",
                conditions=sp.Ne(a, 0),
            )
        return RecognitionResult(
            Uniform(lo, hi), RecognitionKind.AFFINE, "uniform-affine"
        )
    if (
        isinstance(distribution, LogNormal)
        and _same(b, 0)
        and _scale_positive(a) is True
    ):
        return RecognitionResult(
            LogNormal(_simp(distribution.mean + sp.log(a)), distribution.sigma),
            RecognitionKind.AFFINE,
            "lognormal-positive-scale",
        )
    if isinstance(distribution, MultivariateNormal):
        # Scalar affine map applied to the whole vector.
        mean = sp.ImmutableMatrix(distribution.mean)
        cov = sp.ImmutableMatrix(distribution.covariance)
        shifted = mean.applyfunc(lambda x: _simp(a * x + b))
        return RecognitionResult(
            MultivariateNormal(shifted, _simp(a**2) * cov),
            RecognitionKind.AFFINE,
            "multivariate-normal-scalar-affine",
        )
    if isinstance(distribution, Gamma) and _same(b, 0) and _scale_positive(a) is True:
        # Positive scaling changes Gamma scale but not shape.
        return RecognitionResult(
            Gamma(distribution.shape, _simp(distribution.scale * a)),
            RecognitionKind.AFFINE,
            "gamma-positive-scale",
        )

    return RecognitionResult(None, RecognitionKind.NONE, "no-closed-affine-family")


def recognize_transform(
    distribution: Distribution, transform: Transform
) -> RecognitionResult:
    """Recognize selected transformed-family identities, else retain a transformed law."""
    x = transform.variable
    f = sp.expand_log(sp.sympify(transform.forward), force=True)

    # exp(Normal) = LogNormal is imported lazily to avoid a large top-level list.
    from .distributions import LogNormal

    if isinstance(distribution, Normal) and _same(f, sp.exp(x)):
        return RecognitionResult(
            LogNormal(distribution.mean, distribution.sigma),
            RecognitionKind.TRANSFORM,
            "exp-normal-lognormal",
        )

    poly = sp.Poly(sp.expand(f), x) if sp.sympify(f).is_polynomial(x) else None
    if poly is not None and poly.degree() <= 1:
        a = poly.coeff_monomial(x)
        b = poly.coeff_monomial(1)
        result = recognize_affine(distribution, a, b)
        if result.recognized:
            return result

    return RecognitionResult(
        TransformedDistribution(distribution, transform),
        RecognitionKind.TRANSFORM,
        "generic-transformed-distribution",
    )


def _all_type(distributions: tuple[Distribution, ...], cls: type) -> bool:
    return bool(distributions) and all(isinstance(d, cls) for d in distributions)


def recognize_sum(*distributions: Distribution) -> RecognitionResult:
    """Recognize the sum of mutually independent random variables."""
    ds = tuple(canonicalize_distribution(d) for d in distributions)
    if not ds:
        raise ValueError("at least one distribution is required")
    if len(ds) == 1:
        return RecognitionResult(ds[0], RecognitionKind.IDENTITY, "single-summand")

    if _all_type(ds, MultivariateNormal):
        dim = ds[0].dimension
        if not all(d.dimension == dim for d in ds):
            return RecognitionResult(
                None, RecognitionKind.NONE, "multivariate-normal-dimension-mismatch"
            )
        mean = sp.ImmutableMatrix(
            [sp.Add(*(d.mean[i] for d in ds)) for i in range(dim)]
        )
        cov = sp.ImmutableMatrix.zeros(dim)
        for d in ds:
            cov += d.covariance
        return RecognitionResult(
            MultivariateNormal(mean, cov),
            RecognitionKind.SUM,
            "independent-multivariate-normal-sum",
        )
    if _all_type(ds, Normal):
        return RecognitionResult(
            Normal(
                _simp(sum(d.mean for d in ds)),
                sp.sqrt(_simp(sum(d.sigma**2 for d in ds))),
            ),
            RecognitionKind.SUM,
            "independent-normal-sum",
        )
    if _all_type(ds, Poisson):
        return RecognitionResult(
            Poisson(_simp(sum(d.rate for d in ds))),
            RecognitionKind.SUM,
            "independent-poisson-sum",
        )
    if _all_type(ds, Bernoulli) and all(_same(d.p, ds[0].p) for d in ds[1:]):
        return RecognitionResult(
            Binomial(len(ds), ds[0].p), RecognitionKind.SUM, "common-p-bernoulli-sum"
        )
    if _all_type(ds, Binomial) and all(_same(d.p, ds[0].p) for d in ds[1:]):
        return RecognitionResult(
            Binomial(_simp(sum(d.n for d in ds)), ds[0].p),
            RecognitionKind.SUM,
            "common-p-binomial-sum",
        )
    if _all_type(ds, NegativeBinomial) and all(_same(d.p, ds[0].p) for d in ds[1:]):
        return RecognitionResult(
            NegativeBinomial(_simp(sum(d.r for d in ds)), ds[0].p),
            RecognitionKind.SUM,
            "common-p-negative-binomial-sum",
        )
    if _all_type(ds, Gamma) and all(_same(d.scale, ds[0].scale) for d in ds[1:]):
        return RecognitionResult(
            Gamma(_simp(sum(d.shape for d in ds)), ds[0].scale),
            RecognitionKind.SUM,
            "common-scale-gamma-sum",
        )
    if _all_type(ds, Cauchy):
        return RecognitionResult(
            Cauchy(_simp(sum(d.location for d in ds)), _simp(sum(d.scale for d in ds))),
            RecognitionKind.SUM,
            "independent-cauchy-sum",
        )

    # Chi-square canonicalizes to Gamma(df/2, 2), so the Gamma rule covers it.
    return RecognitionResult(None, RecognitionKind.NONE, "no-closed-sum-family")


def recognize_product(*distributions: Distribution) -> RecognitionResult:
    """Recognize the scalar product of mutually independent random variables."""
    ds = tuple(canonicalize_distribution(d) for d in distributions)
    if not ds:
        raise ValueError("at least one distribution is required")
    if len(ds) == 1:
        return RecognitionResult(ds[0], RecognitionKind.IDENTITY, "single-factor")

    from .distributions import LogNormal

    if _all_type(ds, LogNormal):
        return RecognitionResult(
            LogNormal(
                _simp(sum(d.mean for d in ds)),
                sp.sqrt(_simp(sum(d.sigma**2 for d in ds))),
            ),
            RecognitionKind.PRODUCT,
            "independent-lognormal-product",
        )
    if _all_type(ds, Bernoulli):
        p = sp.prod(d.p for d in ds)
        return RecognitionResult(
            Bernoulli(_simp(p)),
            RecognitionKind.PRODUCT,
            "independent-bernoulli-product",
        )

    return RecognitionResult(None, RecognitionKind.NONE, "no-closed-product-family")


def exponential_family_metadata(
    distribution: Distribution,
) -> ExponentialFamilyMetadata | None:
    """Return canonical exponential-family metadata, if exposed by the law."""
    if not isinstance(distribution, ExponentialFamily):
        return None
    x = sp.Symbol("x", real=True)
    return ExponentialFamilyMetadata(
        family=type(distribution).__name__,
        natural_parameters=tuple(map(_simp, distribution.natural_parameters)),
        natural_constraints=sp.sympify(distribution.natural_parameter_constraints()),
        sufficient_statistics=tuple(map(_simp, distribution.sufficient_statistics(x))),
        base_measure=_simp(distribution.base_measure(x)),
        log_partition=_simp(distribution.log_partition),
        parameter_constraints=sp.sympify(distribution.parameter_constraints),
        measure_type=distribution.measure_type,
        event_shape=_event_shape(distribution),
    )


def conjugacy_signature(distribution: Distribution) -> ConjugacySignature:
    """Return stable structural keys for conjugacy dispatch.

    The prior family names are descriptive keys, not imports of Bayesian inference
    prior classes.  This keeps probstats independent while allowing a registry to
    match on a stable protocol.
    """
    family = type(distribution).__name__
    table = {
        "Bernoulli": ("Beta", ("successes", "trials")),
        "Binomial": ("Beta", ("successes", "trials")),
        "Poisson": ("Gamma", ("count_sum", "exposure")),
        "Categorical": ("Dirichlet", ("category_counts",)),
        "Multinomial": ("Dirichlet", ("category_counts", "trials")),
        "Normal": ("NormalInverseGamma", ("count", "sum", "sum_squares")),
        "MultivariateNormal": ("NormalInverseWishart", ("count", "sum", "scatter")),
    }
    prior, stats = table.get(family, (None, ()))
    ef = exponential_family_metadata(distribution)
    dimensions = {
        "Categorical": max(len(getattr(distribution, "probabilities", ())) - 1, 0),
        "Multinomial": max(len(getattr(distribution, "probabilities", ())) - 1, 0),
        "Dirichlet": len(getattr(distribution, "concentration", ())),
        "MultivariateNormal": None,
        "Wishart": None,
    }
    dim = ef.natural_dimension if ef else dimensions.get(family)
    return ConjugacySignature(family, prior, stats, dim)


def distribution_metadata(distribution: Distribution) -> DistributionMetadata:
    """Return planner-oriented structural metadata for a distribution."""
    family = type(distribution).__name__
    location_scale = family in {
        "Normal",
        "LogNormal",
        "Cauchy",
        "Laplace",
        "Logistic",
        "StudentT",
        "Gumbel",
        "MultivariateNormal",
        "MultivariateStudentT",
        "MatrixNormal",
    }
    compound = family in {"BetaBinomial", "DirichletMultinomial"}
    support = getattr(distribution, "support", None)
    positive: bool | None = None
    if isinstance(support, sp.Set):
        subset = support.is_subset(sp.Interval(0, sp.oo))
        if subset in (True, False):
            positive = bool(subset)
    known_exponential = isinstance(distribution, ExponentialFamily) or family in {
        "Categorical",
        "Multinomial",
        "Dirichlet",
        "MultivariateNormal",
        "Wishart",
    }
    return DistributionMetadata(
        family=family,
        measure_type=distribution.measure_type,
        event_shape=_event_shape(distribution),
        is_exponential_family=known_exponential,
        is_location_scale_family=location_scale,
        is_compound=compound,
        is_mixture=isinstance(distribution, MixtureDistribution),
        is_truncated=isinstance(distribution, TruncatedDistribution),
        positive_support=positive,
        conjugacy=conjugacy_signature(distribution),
    )


def recognize_distribution(distribution: Distribution) -> DistributionMetadata:
    """Alias for the stable planner-facing metadata protocol."""
    return distribution_metadata(distribution)


__all__ = [
    "ConjugacySignature",
    "DistributionMetadata",
    "ExponentialFamilyMetadata",
    "RecognitionKind",
    "RecognitionResult",
    "canonicalize_distribution",
    "conjugacy_signature",
    "distribution_metadata",
    "equivalent_distributions",
    "exponential_family_metadata",
    "recognize_affine",
    "recognize_distribution",
    "recognize_product",
    "recognize_sum",
    "recognize_transform",
]
