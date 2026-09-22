"""Shared certification of scalar transformations used by probability theorems."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from ._symbolic_predicates import TruthValue, certified_equal


def _load_funcprops():
    try:
        import funcprops
    except ImportError:
        return None
    return funcprops


def _range_as_set(result, image_variable):
    if isinstance(result, sp.Set):
        return result
    if isinstance(result, sp.Basic):
        try:
            value = result.as_set()
        except (AttributeError, NotImplementedError, TypeError, ValueError):
            return None
        if isinstance(value, sp.Set) and not value.has(image_variable):
            return value
    return None


@dataclass(frozen=True, slots=True)
class ScalarChangeOfVariablesCertification:
    """Verified scalar Jacobian factor for a globally nonvanishing derivative."""

    expression: sp.Expr
    variable: sp.Symbol
    domain: sp.Set
    derivative: sp.Expr
    volume_factor: sp.Expr

    def verify(self):
        """Recheck the symbolic derivative and its global nonvanishing fact."""
        derivative = sp.diff(self.expression, self.variable)
        return (
            certified_equal(derivative, self.derivative) is TruthValue.TRUE
            and self.derivative.is_nonzero is True
            and certified_equal(sp.Abs(self.derivative), self.volume_factor)
            is TruthValue.TRUE
        )


@dataclass(frozen=True, slots=True)
class ScalarTransformationCertification:
    """Exact certification data for a scalar transformation on a stated domain."""

    expression: sp.Expr
    variable: sp.Symbol
    domain: sp.Set
    codomain: sp.Set
    bijective: bool
    change_of_variables: ScalarChangeOfVariablesCertification | None = None

    @property
    def volume_factor(self):
        certificate = self.change_of_variables
        return None if certificate is None else certificate.volume_factor


def certify_scalar_transformation(
    expression,
    variable,
    *,
    domain,
    codomain=None,
    require_change_of_variables=False,
):
    """Certify a scalar bijection and, when requested, its Jacobian factor.

    ``None`` means that the optional reasoning dependency is unavailable or the
    requested theorem hypotheses could not be certified. A returned object is
    globally bijective on ``domain``. Change-of-variables certification also
    requires a symbolically nonvanishing scalar derivative.
    """
    funcprops = _load_funcprops()
    if funcprops is None or not isinstance(domain, sp.Set):
        return None

    expression = sp.sympify(expression)
    variable = sp.sympify(variable)
    if not isinstance(variable, sp.Symbol):
        raise TypeError("transformation variable must be a SymPy Symbol")

    image_variable = sp.Dummy("image", real=True)
    try:
        if codomain is None:
            range_result = funcprops.function_range(
                expression,
                variable,
                image_variable,
                constraints=domain,
            )
            codomain = _range_as_set(range_result, image_variable)
        if not isinstance(codomain, sp.Set):
            return None
        verdict = funcprops.function_bijective(
            expression,
            variable,
            domain=domain,
            codomain=codomain,
        )
        if verdict is not funcprops.TruthValue.TRUE:
            return None

        change = None
        if require_change_of_variables:
            derivative = sp.diff(expression, variable)
            if derivative.is_nonzero is not True:
                return None
            change = ScalarChangeOfVariablesCertification(
                expression=expression,
                variable=variable,
                domain=domain,
                derivative=derivative,
                volume_factor=sp.Abs(derivative),
            )
            if not change.verify():
                return None
    except (ImportError, NotImplementedError, TypeError, ValueError):
        return None

    return ScalarTransformationCertification(
        expression,
        variable,
        domain,
        codomain,
        True,
        change,
    )


def certify_random_expression_bijection(expression):
    """Return the source random variable for a certified scalar bijective transform."""
    from ._random_variable_semantics import stochastic_variables

    expression = sp.sympify(expression)
    variables = stochastic_variables(expression)
    if len(variables) != 1:
        return None
    source = next(iter(variables))
    if expression == source:
        return source

    if source.distribution is not None and isinstance(
        source.distribution.support, sp.Set
    ):
        domain = source.distribution.support
    elif source.is_real is True:
        domain = sp.S.Reals
    else:
        return None

    symbol = sp.Dummy(source.name, real=True)
    transform = expression.xreplace({source: symbol})
    certification = certify_scalar_transformation(transform, symbol, domain=domain)
    return source if certification is not None else None


def _normalized_transform(transform, symbol):
    return sp.simplify(transform.forward.xreplace({transform.variable: symbol}))


def certify_common_pushforward(left, right):
    """Certify that two transformed laws use one bijection on both base supports.

    The common map is certified on the union of the supports. This is stronger
    than certifying each restriction separately and is the theorem hypothesis
    needed for KL invariance under a common measurable bijection.
    """
    from .transforms import Transform, TransformedDistribution

    if not isinstance(left, TransformedDistribution) or not isinstance(
        right, TransformedDistribution
    ):
        return None
    if not isinstance(left.transform, Transform) or not isinstance(
        right.transform, Transform
    ):
        return None
    if not isinstance(left.base.support, sp.Set) or not isinstance(
        right.base.support, sp.Set
    ):
        return None

    symbol = sp.Dummy("x", real=True)
    left_expr = _normalized_transform(left.transform, symbol)
    right_expr = _normalized_transform(right.transform, symbol)
    if certified_equal(left_expr, right_expr) is not TruthValue.TRUE:
        return None

    domain = sp.Union(left.base.support, right.base.support)
    return certify_scalar_transformation(left_expr, symbol, domain=domain)


def transformed_entropy_value(distribution):
    """Return entropy of a certified scalar bijective pushforward, else ``None``.

    Discrete Shannon entropy is invariant under a bijection. For continuous
    scalar laws, the differential-entropy theorem is
    ``h(g(X)) = h(X) + E[log |g'(X)|]`` and requires both global bijectivity and
    a verified nonvanishing scalar Jacobian factor.
    """
    from .functionals import entropy, expectation
    from .spaces import MeasureType
    from .transforms import Transform, TransformedDistribution

    if not isinstance(distribution, TransformedDistribution) or not isinstance(
        distribution.transform, Transform
    ):
        return None
    if not isinstance(distribution.base.support, sp.Set):
        return None

    transform = distribution.transform
    symbol = sp.Dummy(transform.variable.name, real=True)
    expression = _normalized_transform(transform, symbol)

    if distribution.measure_type is MeasureType.DISCRETE:
        certification = certify_scalar_transformation(
            expression,
            symbol,
            domain=distribution.base.support,
        )
        if certification is None:
            return None
        return sp.simplify(entropy(distribution.base))

    if distribution.measure_type is not MeasureType.CONTINUOUS:
        return None

    certification = certify_scalar_transformation(
        expression,
        symbol,
        domain=distribution.base.support,
        require_change_of_variables=True,
    )
    if certification is None:
        return None

    volume_factor = certification.volume_factor
    if volume_factor is None:
        return None
    correction = expectation(
        distribution.base,
        sp.log(volume_factor),
        variable=symbol,
    )
    return sp.simplify(entropy(distribution.base) + correction)
