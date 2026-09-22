"""Information-theoretic measures for probability distributions.

The implementation is exact-first.  It verifies support containment before
relative-information calculations, attempts symbolic summation/integration,
and can fall back to Monte Carlo for concrete distributions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import sympy as sp

from ._integration import integrate_rectangular
from .distributions.base import Distribution
from .information_registry import (
    get_information_formula,
    register_information_formula,
    registered_information_formulas,
)
from .results import StatisticalResult
from .spaces import MeasureType, ScalarEventSpace, VectorEventSpace


class InformationMeasureError(ValueError):
    """Raised when an information measure cannot be evaluated safely."""


class SupportRelation(str, Enum):
    PROVEN = "proven"
    DISPROVEN = "disproven"
    UNKNOWN = "unknown"


class InformationMethod(str, Enum):
    CLOSED_FORM = "closed_form"
    SYMBOLIC = "symbolic"
    MONTE_CARLO = "monte_carlo"


@dataclass(frozen=True, slots=True)
class InformationResult(StatisticalResult):
    value: Any
    method: InformationMethod
    support_relation: SupportRelation = SupportRelation.PROVEN
    exact: bool = True


def support_subset(p: Distribution, q: Distribution) -> SupportRelation:
    """Determine whether ``support(p)`` is contained in ``support(q)``.

    The result is three-valued.  ``UNKNOWN`` means that SymPy
    could not prove or disprove containment; it is never interpreted as false.
    Event-space dimensionality is checked before symbolic set containment.
    """
    if not isinstance(p, Distribution) or not isinstance(q, Distribution):
        raise TypeError("p and q must be Distribution instances")

    pe = p.event_space
    qe = q.event_space
    if isinstance(pe, VectorEventSpace) or isinstance(qe, VectorEventSpace):
        if not isinstance(pe, VectorEventSpace) or not isinstance(qe, VectorEventSpace):
            return SupportRelation.DISPROVEN
        if pe.dimension != qe.dimension:
            return SupportRelation.DISPROVEN
    elif not isinstance(pe, ScalarEventSpace) or not isinstance(qe, ScalarEventSpace):
        if type(pe) is not type(qe):
            return SupportRelation.DISPROVEN

    try:
        relation = p.support.is_subset(q.support)
    except (AttributeError, TypeError, ValueError, NotImplementedError):
        relation = None
    if relation is True:
        return SupportRelation.PROVEN
    if relation is False:
        return SupportRelation.DISPROVEN

    try:
        difference = p.support - q.support
    except (TypeError, ValueError, NotImplementedError):
        return SupportRelation.UNKNOWN
    if difference is sp.S.EmptySet or difference == sp.S.EmptySet:
        return SupportRelation.PROVEN
    if getattr(difference, "is_empty", None) is False:
        return SupportRelation.DISPROVEN
    return SupportRelation.UNKNOWN


def _check_measure_compatibility(p: Distribution, q: Distribution) -> bool:
    """Return whether p can be absolutely continuous with respect to q's measure."""
    if p.measure_type is q.measure_type:
        return True
    # A discrete law and an absolutely continuous law are mutually singular in
    # the package's current measure model even if their set-valued supports overlap.
    return {p.measure_type, q.measure_type} != {
        MeasureType.DISCRETE,
        MeasureType.CONTINUOUS,
    }


def _variables(dist: Distribution):
    event = dist.event_space
    if isinstance(event, ScalarEventSpace):
        integer = dist.measure_type is MeasureType.DISCRETE
        return sp.Dummy("x", integer=integer, real=True)
    if isinstance(event, VectorEventSpace):
        return tuple(sp.Dummy(f"x{i}", real=True) for i in range(event.dimension))
    raise InformationMeasureError(
        f"unsupported event space for generic information measure: {type(event).__name__}"
    )


def _log_density_at(dist: Distribution, value):
    if dist.measure_type is MeasureType.DISCRETE:
        return dist.logpmf(value)
    return dist.logpdf(value)


def _density_at(dist: Distribution, value):
    if dist.measure_type is MeasureType.DISCRETE:
        return dist.pmf(value)
    return dist.pdf(value)


def _support_indicator(dist: Distribution, value):
    support = dist.support
    if isinstance(value, tuple):
        if isinstance(support, sp.ProductSet) and len(support.args) == len(value):
            conditions = []
            for component, x in zip(support.args, value):
                conditions.append(component.contains(x))
            return sp.And(*conditions)
        return support.contains(sp.Tuple(*value))
    return support.contains(value)


def _supported_density(dist: Distribution, value):
    indicator = _support_indicator(dist, value)
    if indicator is sp.S.true:
        return _density_at(dist, value)
    if indicator is sp.S.false:
        return sp.S.Zero
    return sp.Piecewise((_density_at(dist, value), indicator), (sp.S.Zero, True))


def _integrate_over_support(dist: Distribution, expression, variables):
    support = dist.support
    if isinstance(variables, tuple):
        if not isinstance(support, sp.ProductSet) or len(support.args) != len(
            variables
        ):
            raise InformationMeasureError(
                "symbolic multivariate evaluation requires a ProductSet support"
            )
        ranges = []
        pairs = reversed(tuple(zip(variables, support.args)))
        for variable, component in pairs:
            if not isinstance(component, sp.Interval):
                raise InformationMeasureError(
                    "symbolic multivariate evaluation currently requires interval components"
                )
            ranges.append((variable, component.start, component.end))
        return integrate_rectangular(expression, ranges)

    if dist.measure_type is MeasureType.DISCRETE:
        if isinstance(support, sp.FiniteSet):
            return sp.simplify(sum(expression.subs(variables, x) for x in support))
        if support == sp.S.Integers:
            return sp.simplify(sp.summation(expression, (variables, -sp.oo, sp.oo)))
        if support == sp.S.Naturals0:
            return sp.simplify(sp.summation(expression, (variables, 0, sp.oo)))
        if isinstance(support, sp.Range):
            stop = support.stop - support.step
            return sp.simplify(
                sp.summation(expression, (variables, support.start, stop, support.step))
            )
        raise InformationMeasureError(
            f"unsupported discrete support for symbolic information measure: {support}"
        )

    if isinstance(support, sp.Interval):
        return sp.simplify(
            sp.integrate(expression, (variables, support.start, support.end))
        )
    if support == sp.S.Reals:
        return sp.simplify(sp.integrate(expression, (variables, -sp.oo, sp.oo)))
    raise InformationMeasureError(
        f"unsupported continuous support for symbolic information measure: {support}"
    )


def _symbolic_expectation_under(p: Distribution, function):
    """Evaluate a symbolic expectation with one explicit fallback boundary."""
    variables = _variables(p)
    point = variables
    try:
        integrand = sp.simplify(_density_at(p, point) * function(point))
        return _integrate_over_support(p, integrand, variables)
    except InformationMeasureError:
        raise
    except (TypeError, ValueError, NotImplementedError, sp.PolynomialError) as exc:
        raise InformationMeasureError(
            "symbolic information evaluation is unavailable"
        ) from exc


def _numeric_log_density_function(dist: Distribution, *, support_aware: bool = False):
    variables = _variables(dist)
    point = variables
    if support_aware:
        expression = sp.log(_supported_density(dist, point))
    else:
        expression = _log_density_at(dist, point)
    if expression.free_symbols - set(
        variables if isinstance(variables, tuple) else (variables,)
    ):
        raise InformationMeasureError("numerical fallback requires concrete parameters")
    args = variables if isinstance(variables, tuple) else (variables,)
    fn = sp.lambdify(args, expression, modules="numpy")

    def evaluate(value):
        arr = np.asarray(value, dtype=float)
        if isinstance(variables, tuple):
            if arr.ndim == 0 or arr.shape[-1] != len(variables):
                raise InformationMeasureError("sample has the wrong event dimension")
            args_values = np.moveaxis(arr, -1, 0)
            result = np.asarray(fn(*args_values), dtype=float)
            expected_shape = arr.shape[:-1]
        else:
            result = np.asarray(fn(arr), dtype=float)
            expected_shape = arr.shape
        if result.ndim == 0 and expected_shape:
            result = np.full(expected_shape, float(result))
        return float(result) if result.ndim == 0 else result

    return evaluate


def _mc_expectation(p: Distribution, function, *, samples: int, rng=None):
    if not isinstance(samples, int) or samples <= 0:
        raise ValueError("samples must be a positive integer")
    draws = np.asarray(p.sample(size=samples, rng=rng))
    try:
        values = np.asarray(function(draws), dtype=float)
        if values.shape != (samples,):
            raise ValueError
    except (TypeError, ValueError):
        values = np.asarray([function(draws[i]) for i in range(samples)], dtype=float)
    if np.any(np.isnan(values)):
        raise InformationMeasureError("numerical information measure produced NaN")
    return float(np.mean(values))


def _finish(value, method, support_relation, *, return_result):
    result = InformationResult(
        value=value,
        method=method,
        support_relation=support_relation,
        exact=method is not InformationMethod.MONTE_CARLO,
    )
    return result if return_result else value


def _same_distribution(p: Distribution, q: Distribution) -> bool:
    if p == q:
        return True
    try:
        return bool(p.equivalent_to(q))
    except (AttributeError, TypeError, ValueError, NotImplementedError):
        return False


def _common_pushforward_kl(p, q, **kwargs):
    """Return base-law KL when a common bijective pushforward is certified."""
    from ._transformation_certification import certify_common_pushforward
    from .transforms import TransformedDistribution

    if not isinstance(p, TransformedDistribution) or not isinstance(
        q, TransformedDistribution
    ):
        return None
    if certify_common_pushforward(p, q) is None:
        return None
    return kl_divergence(p.base, q.base, **kwargs)


def kl_divergence(
    p: object,
    q: object,
    *,
    numerical_fallback: bool = False,
    samples: int = 100_000,
    rng=None,
    return_result: bool = False,
):
    """Return ``D_KL(p || q)``.

    A disproven support-containment or measure-compatibility condition returns
    ``oo``.  If containment is unknown, symbolic evaluation is attempted rather
    than incorrectly declaring the divergence finite or infinite.
    """
    pushforward_value = _common_pushforward_kl(
        p,
        q,
        numerical_fallback=numerical_fallback,
        samples=samples,
        rng=rng,
        return_result=return_result,
    )
    if pushforward_value is not None:
        return pushforward_value

    formula = get_information_formula("kl", p, q)
    if not isinstance(p, Distribution) or not isinstance(q, Distribution):
        if formula is None:
            raise TypeError(
                "p and q must be Distribution instances or have a registered "
                "closed-form information formula"
            )
        if p == q:
            return _finish(
                sp.S.Zero,
                InformationMethod.CLOSED_FORM,
                SupportRelation.PROVEN,
                return_result=return_result,
            )
        try:
            value = formula(p, q)
        except NotImplementedError as exc:
            raise InformationMeasureError(str(exc)) from exc
        relation = (
            SupportRelation.DISPROVEN if value is sp.oo else SupportRelation.PROVEN
        )
        return _finish(
            sp.simplify(value),
            InformationMethod.CLOSED_FORM,
            relation,
            return_result=return_result,
        )
    if _same_distribution(p, q):
        return _finish(
            sp.S.Zero,
            InformationMethod.CLOSED_FORM,
            SupportRelation.PROVEN,
            return_result=return_result,
        )

    relation = support_subset(p, q)
    if relation is SupportRelation.DISPROVEN or not _check_measure_compatibility(p, q):
        return _finish(
            sp.oo,
            InformationMethod.CLOSED_FORM,
            SupportRelation.DISPROVEN,
            return_result=return_result,
        )

    if formula is not None:
        try:
            value = formula(p, q)
        except NotImplementedError:
            value = None
        if value is not None:
            return _finish(
                sp.simplify(value),
                InformationMethod.CLOSED_FORM,
                relation,
                return_result=return_result,
            )

    try:
        value = _symbolic_expectation_under(
            p, lambda x: sp.simplify(_log_density_at(p, x) - _log_density_at(q, x))
        )
        if value.has(sp.Integral, sp.Sum):
            raise InformationMeasureError("symbolic KL divergence remained unevaluated")
        return _finish(
            sp.simplify(value),
            InformationMethod.SYMBOLIC,
            relation,
            return_result=return_result,
        )
    except (InformationMeasureError, NotImplementedError):
        if not numerical_fallback:
            raise

    logp = _numeric_log_density_function(p, support_aware=True)
    logq = _numeric_log_density_function(q, support_aware=True)
    value = _mc_expectation(p, lambda x: logp(x) - logq(x), samples=samples, rng=rng)
    return _finish(
        value,
        InformationMethod.MONTE_CARLO,
        relation,
        return_result=return_result,
    )


def cross_entropy(
    p: object,
    q: object,
    *,
    numerical_fallback: bool = False,
    samples: int = 100_000,
    rng=None,
    return_result: bool = False,
):
    """Return the cross-entropy ``H(p, q) = -E_p[log q(X)]``."""
    formula = get_information_formula("kl", p, q)
    entropy_formula = get_information_formula("entropy", p, p)
    if not isinstance(p, Distribution) or not isinstance(q, Distribution):
        if formula is None or entropy_formula is None:
            raise TypeError(
                "non-Distribution objects require registered KL and entropy formulas"
            )
        try:
            value = sp.simplify(entropy_formula(p, p) + formula(p, q))
        except NotImplementedError as exc:
            raise InformationMeasureError(str(exc)) from exc
        return _finish(
            value,
            InformationMethod.CLOSED_FORM,
            SupportRelation.PROVEN,
            return_result=return_result,
        )
    relation = support_subset(p, q)
    if relation is SupportRelation.DISPROVEN or not _check_measure_compatibility(p, q):
        return _finish(
            sp.oo,
            InformationMethod.CLOSED_FORM,
            SupportRelation.DISPROVEN,
            return_result=return_result,
        )
    if formula is not None and entropy_formula is not None:
        try:
            value = sp.simplify(entropy_formula(p, p) + formula(p, q))
        except NotImplementedError:
            value = None
        if value is not None:
            return _finish(
                value,
                InformationMethod.CLOSED_FORM,
                relation,
                return_result=return_result,
            )
    try:
        value = _symbolic_expectation_under(p, lambda x: -_log_density_at(q, x))
        if value.has(sp.Integral, sp.Sum):
            raise InformationMeasureError("symbolic cross-entropy remained unevaluated")
        return _finish(
            sp.simplify(value),
            InformationMethod.SYMBOLIC,
            relation,
            return_result=return_result,
        )
    except (InformationMeasureError, NotImplementedError):
        if not numerical_fallback:
            raise
    logq = _numeric_log_density_function(q, support_aware=True)
    value = _mc_expectation(p, lambda x: -logq(x), samples=samples, rng=rng)
    return _finish(
        value,
        InformationMethod.MONTE_CARLO,
        relation,
        return_result=return_result,
    )


def information_entropy(
    p: object,
    *,
    numerical_fallback: bool = False,
    samples: int = 100_000,
    rng=None,
    return_result: bool = False,
):
    """Return Shannon entropy with the same exact/numerical result protocol."""
    formula = get_information_formula("entropy", p, p)
    if formula is not None:
        return _finish(
            sp.simplify(formula(p, p)),
            InformationMethod.CLOSED_FORM,
            SupportRelation.PROVEN,
            return_result=return_result,
        )
    try:
        if hasattr(p, "_entropy"):
            value = p._entropy()
            if value is not NotImplemented:
                return _finish(
                    sp.simplify(value),
                    InformationMethod.CLOSED_FORM,
                    SupportRelation.PROVEN,
                    return_result=return_result,
                )
        value = _symbolic_expectation_under(p, lambda x: -_log_density_at(p, x))
        if value.has(sp.Integral, sp.Sum):
            raise InformationMeasureError("symbolic entropy remained unevaluated")
        return _finish(
            sp.simplify(value),
            InformationMethod.SYMBOLIC,
            SupportRelation.PROVEN,
            return_result=return_result,
        )
    except (InformationMeasureError, NotImplementedError):
        if not numerical_fallback:
            raise
    logp = _numeric_log_density_function(p, support_aware=True)
    value = _mc_expectation(p, lambda x: -logp(x), samples=samples, rng=rng)
    return _finish(
        value,
        InformationMethod.MONTE_CARLO,
        SupportRelation.PROVEN,
        return_result=return_result,
    )


def renyi_divergence(
    p: object,
    q: object,
    alpha,
    *,
    numerical_fallback: bool = False,
    samples: int = 100_000,
    rng=None,
    return_result: bool = False,
):
    """Return Rényi divergence of order ``alpha``.

    The order must be positive.  ``alpha == 1`` is defined by continuity and
    dispatches to KL divergence.
    """
    a = sp.sympify(alpha)
    if a.is_positive is False or a == 0:
        raise ValueError("alpha must be positive")
    if a == 1:
        return kl_divergence(
            p,
            q,
            numerical_fallback=numerical_fallback,
            samples=samples,
            rng=rng,
            return_result=return_result,
        )
    formula = get_information_formula("renyi", p, q)
    if not isinstance(p, Distribution) or not isinstance(q, Distribution):
        if formula is None:
            raise TypeError(
                "non-Distribution objects require a registered Rényi formula"
            )
        try:
            value = formula(p, q, a)
        except NotImplementedError as exc:
            raise InformationMeasureError(str(exc)) from exc
        relation = (
            SupportRelation.DISPROVEN if value is sp.oo else SupportRelation.PROVEN
        )
        return _finish(
            sp.simplify(value),
            InformationMethod.CLOSED_FORM,
            relation,
            return_result=return_result,
        )
    relation = support_subset(p, q)
    # For alpha > 1, lack of absolute continuity makes the divergence infinite.
    if (
        (
            relation is SupportRelation.DISPROVEN
            or not _check_measure_compatibility(p, q)
        )
        and a.is_number
        and a > 1
    ):
        return _finish(
            sp.oo,
            InformationMethod.CLOSED_FORM,
            SupportRelation.DISPROVEN,
            return_result=return_result,
        )

    if formula is not None:
        try:
            value = formula(p, q, a)
        except NotImplementedError:
            value = None
        if value is not None:
            return _finish(
                sp.simplify(value),
                InformationMethod.CLOSED_FORM,
                relation,
                return_result=return_result,
            )

    try:
        variables = _variables(p)
        point = variables
        integrand = sp.simplify(
            _supported_density(p, point) ** a * _supported_density(q, point) ** (1 - a)
        )
        integral = _integrate_over_support(p, integrand, variables)
        if integral.has(sp.Integral, sp.Sum):
            raise InformationMeasureError(
                "symbolic Rényi integral remained unevaluated"
            )
        value = sp.simplify(sp.log(integral) / (a - 1))
        return _finish(
            value,
            InformationMethod.SYMBOLIC,
            relation,
            return_result=return_result,
        )
    except (InformationMeasureError, NotImplementedError):
        if not numerical_fallback:
            raise

    af = float(a)
    logp = _numeric_log_density_function(p, support_aware=True)
    logq = _numeric_log_density_function(q, support_aware=True)
    moment = _mc_expectation(
        p,
        lambda x: np.exp((af - 1.0) * (logp(x) - logq(x))),
        samples=samples,
        rng=rng,
    )
    value = float(np.log(moment) / (af - 1.0))
    return _finish(
        value,
        InformationMethod.MONTE_CARLO,
        relation,
        return_result=return_result,
    )


def bhattacharyya_coefficient(
    p: object,
    q: object,
    *,
    numerical_fallback: bool = False,
    samples: int = 100_000,
    rng=None,
    return_result: bool = False,
):
    """Return the Bhattacharyya coefficient ``∫ sqrt(p q)``."""
    formula = get_information_formula("renyi", p, q)
    if formula is not None:
        rhalf = formula(p, q, sp.Rational(1, 2))
        relation = (
            support_subset(p, q)
            if isinstance(p, Distribution) and isinstance(q, Distribution)
            else SupportRelation.PROVEN
        )
        return _finish(
            sp.simplify(sp.exp(-rhalf / 2)),
            InformationMethod.CLOSED_FORM,
            relation,
            return_result=return_result,
        )
    try:
        variables = _variables(p)
        point = variables
        integrand = sp.sqrt(_supported_density(p, point) * _supported_density(q, point))
        value = _integrate_over_support(p, integrand, variables)
        if value.has(sp.Integral, sp.Sum):
            raise InformationMeasureError(
                "symbolic Bhattacharyya coefficient remained unevaluated"
            )
        return _finish(
            sp.simplify(value),
            InformationMethod.SYMBOLIC,
            support_subset(p, q),
            return_result=return_result,
        )
    except (InformationMeasureError, NotImplementedError):
        if not numerical_fallback:
            raise
    logp = _numeric_log_density_function(p, support_aware=True)
    logq = _numeric_log_density_function(q, support_aware=True)
    value = _mc_expectation(
        p,
        lambda x: np.exp(0.5 * (logq(x) - logp(x))),
        samples=samples,
        rng=rng,
    )
    return _finish(
        value,
        InformationMethod.MONTE_CARLO,
        support_subset(p, q),
        return_result=return_result,
    )


def bhattacharyya_distance(p: Distribution, q: Distribution, **kwargs):
    """Return ``-log`` of the Bhattacharyya coefficient."""
    return_result = bool(kwargs.pop("return_result", False))
    result = bhattacharyya_coefficient(p, q, return_result=True, **kwargs)
    value = (
        sp.simplify(-sp.log(result.value)) if result.exact else -np.log(result.value)
    )
    return _finish(
        value,
        result.method,
        result.support_relation,
        return_result=return_result,
    )


def hellinger_distance(p: Distribution, q: Distribution, **kwargs):
    """Return the Hellinger distance ``sqrt(1 - BC(p, q))``."""
    return_result = bool(kwargs.pop("return_result", False))
    result = bhattacharyya_coefficient(p, q, return_result=True, **kwargs)
    if result.exact:
        value = sp.simplify(sp.sqrt(1 - result.value))
    else:
        value = float(np.sqrt(max(0.0, 1.0 - result.value)))
    return _finish(
        value,
        result.method,
        result.support_relation,
        return_result=return_result,
    )


def hellinger_squared(p: Distribution, q: Distribution, **kwargs):
    """Return squared Hellinger distance ``1 - BC(p, q)``."""
    return_result = bool(kwargs.pop("return_result", False))
    result = bhattacharyya_coefficient(p, q, return_result=True, **kwargs)
    value = sp.simplify(1 - result.value) if result.exact else 1.0 - result.value
    return _finish(
        value,
        result.method,
        result.support_relation,
        return_result=return_result,
    )


def jensen_shannon_divergence(
    p: Distribution,
    q: Distribution,
    *,
    weight=None,
    numerical_fallback: bool = False,
    samples: int = 100_000,
    rng=None,
    return_result: bool = False,
):
    """Return weighted Jensen-Shannon divergence.

    This is evaluated directly as expectations under ``p`` and ``q`` so no
    artificial mixture-distribution class is required.
    """
    w = sp.Rational(1, 2) if weight is None else sp.sympify(weight)
    if w.is_number and not (0 <= w <= 1):
        raise ValueError("weight must lie in [0, 1]")
    if w == 0 or w == 1:
        return _finish(
            sp.S.Zero,
            InformationMethod.CLOSED_FORM,
            SupportRelation.PROVEN,
            return_result=return_result,
        )

    symbolic_js = (
        p.measure_type is MeasureType.DISCRETE
        and q.measure_type is MeasureType.DISCRETE
        and isinstance(p.support, sp.FiniteSet)
        and isinstance(q.support, sp.FiniteSet)
    )
    if not symbolic_js and not numerical_fallback:
        raise InformationMeasureError(
            "generic symbolic Jensen-Shannon evaluation is limited to finite "
            "discrete supports; enable numerical_fallback for continuous laws"
        )

    try:
        if not symbolic_js:
            raise InformationMeasureError("use numerical Jensen-Shannon fallback")

        def log_mix(x):
            return sp.log(
                w * _supported_density(p, x) + (1 - w) * _supported_density(q, x)
            )

        left = _symbolic_expectation_under(
            p, lambda x: _log_density_at(p, x) - log_mix(x)
        )
        right = _symbolic_expectation_under(
            q, lambda x: _log_density_at(q, x) - log_mix(x)
        )
        value = sp.simplify(w * left + (1 - w) * right)
        if value.has(sp.Integral, sp.Sum):
            raise InformationMeasureError(
                "symbolic Jensen-Shannon divergence remained unevaluated"
            )
        return _finish(
            value,
            InformationMethod.SYMBOLIC,
            SupportRelation.PROVEN,
            return_result=return_result,
        )
    except (InformationMeasureError, NotImplementedError):
        if not numerical_fallback:
            raise

    wf = float(w)
    rp = np.random.default_rng(rng)
    seed_p = int(rp.integers(0, 2**63 - 1))
    seed_q = int(rp.integers(0, 2**63 - 1))
    logp = _numeric_log_density_function(p, support_aware=True)
    logq = _numeric_log_density_function(q, support_aware=True)

    def one_side(source, *, seed):
        def term(x):
            lp = logp(x)
            lq = logq(x)
            m = np.logaddexp(np.log(wf) + lp, np.log1p(-wf) + lq)
            ls = lp if source is p else lq
            return ls - m

        return _mc_expectation(source, term, samples=samples, rng=seed)

    value = wf * one_side(p, seed=seed_p) + (1 - wf) * one_side(q, seed=seed_q)
    return _finish(
        value,
        InformationMethod.MONTE_CARLO,
        SupportRelation.PROVEN,
        return_result=return_result,
    )


__all__ = [
    "InformationMeasureError",
    "InformationMethod",
    "InformationResult",
    "SupportRelation",
    "bhattacharyya_coefficient",
    "bhattacharyya_distance",
    "cross_entropy",
    "hellinger_distance",
    "hellinger_squared",
    "information_entropy",
    "jensen_shannon_divergence",
    "kl_divergence",
    "register_information_formula",
    "registered_information_formulas",
    "renyi_divergence",
    "support_subset",
]
