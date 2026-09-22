"""Generic probability/statistics functionals and exact-first solver dispatch."""

from __future__ import annotations

from functools import reduce
from operator import mul

import numpy as np
import sympy as sp

from .data import WeightedData
from .events import PredicateEvent, ProbabilityEvent, ProductEvent, SetEvent, as_event
from .solver import (
    EvaluationMethod,
    FunctionalResult,
    monte_carlo_expectation,
    monte_carlo_probability,
)
from .spaces import MeasureType, ScalarEventSpace


def _array_map(function, value):
    array = np.asarray(value)
    if array.ndim == 0:
        return None
    out = np.empty(array.shape, dtype=object)
    for index in np.ndindex(array.shape):
        out[index] = function(
            array[index].item() if hasattr(array[index], "item") else array[index]
        )
    return out


class ProbabilityFunctionalError(ValueError):
    pass


def _x():
    return sp.Dummy("x", real=True)


def _wrap(value, method: EvaluationMethod, *, return_result: bool):
    result = FunctionalResult(value, method, method is not EvaluationMethod.MONTE_CARLO)
    return result if return_result else value


def density(dist, value):
    """Return density or mass, preserving the shape of array-like input."""
    mapped = _array_map(lambda item: density(dist, item), value)
    if mapped is not None:
        return mapped
    return (
        dist.pmf(value)
        if dist.measure_type is MeasureType.DISCRETE
        else dist.pdf(value)
    )


def log_density(dist, value):
    mapped = _array_map(lambda item: log_density(dist, item), value)
    if mapped is not None:
        return mapped
    return (
        dist.logpmf(value)
        if dist.measure_type is MeasureType.DISCRETE
        else dist.logpdf(value)
    )


def cdf(dist, value):
    """Cumulative distribution function, preserving array-like input shape."""
    mapped = _array_map(lambda item: cdf(dist, item), value)
    if mapped is not None:
        return mapped
    if hasattr(dist, "_cdf"):
        result = dist._cdf(sp.sympify(value))
        if result is not NotImplemented:
            return sp.simplify(result)
    if not isinstance(dist.event_space, ScalarEventSpace):
        raise ProbabilityFunctionalError("cdf requires a scalar event space")
    x = _x()
    z = sp.sympify(value)
    support = dist.support
    if dist.measure_type is MeasureType.DISCRETE:
        if isinstance(support, sp.FiniteSet):
            return sp.simplify(
                sum(dist.pmf(k) * sp.Piecewise((1, k <= z), (0, True)) for k in support)
            )
        if support == sp.S.Naturals0:
            k = sp.Dummy("k", integer=True, nonnegative=True)
            return sp.simplify(sp.summation(dist.pmf(k), (k, 0, sp.floor(z))))
        raise ProbabilityFunctionalError(
            "generic discrete cdf requires a finite support or Naturals0"
        )
    if isinstance(support, sp.Interval):
        lo, hi = support.start, support.end
        interior = sp.simplify(sp.integrate(dist.pdf(x), (x, lo, z)))
        if lo is -sp.oo and hi is sp.oo:
            return interior
        branches = []
        if lo is not -sp.oo:
            branches.append((sp.S.Zero, z < lo))
        if hi is not sp.oo:
            branches.append((sp.S.One, z >= hi))
        branches.append((interior, True))
        return sp.Piecewise(*branches)
    if support == sp.S.Reals:
        return sp.simplify(sp.integrate(dist.pdf(x), (x, -sp.oo, z)))
    raise ProbabilityFunctionalError(f"unsupported scalar support for cdf: {support}")


def survival(dist, value):
    """Survival function ``P(X > x)``, preserving array-like input shape."""
    mapped = _array_map(lambda item: survival(dist, item), value)
    if mapped is not None:
        return mapped
    if hasattr(dist, "_survival"):
        result = dist._survival(sp.sympify(value))
        if result is not NotImplemented:
            return sp.simplify(result)
    return sp.simplify(1 - cdf(dist, value))


def quantile(dist, probability):
    """Quantile, preserving the shape of array-like probability input."""
    mapped = _array_map(lambda item: quantile(dist, item), probability)
    if mapped is not None:
        return mapped
    if isinstance(dist, WeightedData):
        return dist.quantile(float(probability))
    p = sp.sympify(probability)
    if p.is_number and (p < 0 or p > 1):
        raise ValueError("probability must lie in [0, 1]")
    if hasattr(dist, "_quantile"):
        result = dist._quantile(p)
        if result is not NotImplemented:
            return sp.simplify(result)
    x = _x()
    sol = sp.solve(sp.Eq(cdf(dist, x), p), x)
    if len(sol) == 1:
        return sp.simplify(sol[0])
    raise ProbabilityFunctionalError("quantile could not be inverted symbolically")


def inverse_survival(dist, probability):
    """Return the inverse survival function for a scalar distribution.

    The result is the smallest quantile whose upper-tail probability does not
    exceed ``probability``. For continuous laws this is exactly
    ``quantile(dist, 1 - probability)``.
    """
    mapped = _array_map(lambda item: inverse_survival(dist, item), probability)
    if mapped is not None:
        return mapped
    p = sp.sympify(probability)
    if p.is_number and (p < 0 or p > 1):
        raise ValueError("probability must lie in [0, 1]")
    if hasattr(dist, "_inverse_survival"):
        result = dist._inverse_survival(p)
        if result is not NotImplemented:
            return sp.simplify(result)
    return quantile(dist, sp.simplify(1 - p))


def hazard(dist, value):
    """Return the hazard at ``value`` using the distribution's measure.

    Continuous hazards are ``f(x) / P(X > x)``. Discrete hazards use
    ``P(X=x | X>=x)``, so their denominator includes the mass at ``x``.
    """
    mapped = _array_map(lambda item: hazard(dist, item), value)
    if mapped is not None:
        return mapped
    x = sp.sympify(value)
    if not isinstance(dist.event_space, ScalarEventSpace):
        raise ProbabilityFunctionalError("hazard requires a scalar event space")
    if hasattr(dist, "_hazard"):
        result = dist._hazard(x)
        if result is not NotImplemented:
            return sp.simplify(result)
    tail = survival(dist, x)
    if dist.measure_type is MeasureType.DISCRETE:
        return sp.simplify(dist.pmf(x) / (tail + dist.pmf(x)))
    if dist.measure_type is MeasureType.CONTINUOUS:
        return sp.simplify(dist.pdf(x) / tail)
    raise ProbabilityFunctionalError(
        "hazard is not defined generically for mixed measures"
    )


def cumulative_hazard(dist, value):
    """Return ``-log(P(X > x))`` for a scalar distribution."""
    mapped = _array_map(lambda item: cumulative_hazard(dist, item), value)
    if mapped is not None:
        return mapped
    if not isinstance(dist.event_space, ScalarEventSpace):
        raise ProbabilityFunctionalError(
            "cumulative hazard requires a scalar event space"
        )
    if hasattr(dist, "_cumulative_hazard"):
        result = dist._cumulative_hazard(sp.sympify(value))
        if result is not NotImplemented:
            return sp.simplify(result)
    return sp.simplify(-sp.log(survival(dist, value)))


def _factorial_moment_from_pgf(dist, order):
    from .symbolic import probability_generating_function

    z = sp.Dummy("z")
    try:
        pgf = probability_generating_function(dist, z)
        value = sp.diff(pgf, z, order).subs(z, 1)
    except (TypeError, ValueError, NotImplementedError, ProbabilityFunctionalError):
        return None
    if value.has(sp.Derivative):
        return None
    return sp.simplify(value)


def _cumulant_from_cgf(dist, order):
    from .symbolic import cumulant_generating_function

    t = sp.Dummy("t", real=True)
    try:
        cgf = cumulant_generating_function(dist, t)
    except (TypeError, ValueError, NotImplementedError, ProbabilityFunctionalError):
        return None
    irregular = cgf.has(sp.Piecewise, sp.Integral, sp.Sum, sp.oo, -sp.oo, sp.zoo)
    if irregular:
        return None
    value = sp.diff(cgf, t, order).subs(t, 0)
    if value.is_finite is False:
        return None
    return sp.simplify(value)


def factorial_moment(dist, order: int):
    """Return the falling-factorial moment ``E[(X)_order]``."""
    if not isinstance(order, int) or order < 0:
        raise ValueError("factorial moment order must be a nonnegative integer")
    if dist.measure_type is not MeasureType.DISCRETE:
        raise ProbabilityFunctionalError(
            "factorial moments require a discrete distribution"
        )
    if hasattr(dist, "_factorial_moment"):
        result = dist._factorial_moment(order)
        if result is not NotImplemented:
            return sp.simplify(result)
    if order == 0:
        return sp.S.One
    generated = _factorial_moment_from_pgf(dist, order)
    if generated is not None:
        return generated
    x = sp.Dummy("x", integer=True, nonnegative=True)
    return sp.simplify(expectation(dist, sp.ff(x, order), variable=x))


def cumulant(dist, order: int, *, assumptions=None):
    """Return a distribution cumulant or symbolic random-expression cumulant.

    For random-variable expressions, algebraic simplification uses explicit
    statistical assumptions and never guesses independent relationships.


    Orders one and above are computed from raw moments by the standard
    triangular recurrence. This avoids differentiating Piecewise MGFs at their
    convergence boundaries. The zeroth cumulant is zero.
    """
    from .random_variable_algebra import is_random_expression

    if isinstance(dist, sp.Basic) and is_random_expression(dist):
        from .random_variable_algebra import cumulant as random_cumulant

        return random_cumulant(dist, order, assumptions=assumptions)
    if assumptions is not None:
        raise TypeError("assumptions apply only to random-variable expressions")
    if not isinstance(order, int) or order < 0:
        raise ValueError("cumulant order must be a nonnegative integer")
    if order == 0:
        return sp.S.Zero
    if hasattr(dist, "_cumulant"):
        result = dist._cumulant(order)
        if result is not NotImplemented:
            return sp.simplify(result)
    generated = _cumulant_from_cgf(dist, order)
    if generated is not None:
        return generated
    cumulants = [sp.S.Zero]
    moments = [sp.S.One] + [raw_moment(dist, n) for n in range(1, order + 1)]
    for n in range(1, order + 1):
        correction = sp.Add(
            *(
                sp.binomial(n - 1, k - 1) * cumulants[k] * moments[n - k]
                for k in range(1, n)
            )
        )
        cumulants.append(sp.simplify(moments[n] - correction))
    return cumulants[order]


def likelihood_value(dist, observations):
    """Return the joint likelihood of iid observations under ``dist``.

    ``observations`` may be any finite iterable. The function preserves exact
    symbolic arithmetic and uses the distribution's density or mass according
    to its reference measure.
    """
    values = tuple(observations)
    if not values:
        return sp.S.One
    return sp.simplify(sp.prod(density(dist, value) for value in values))


def log_likelihood(dist, observations):
    """Return the iid sample log likelihood under ``dist``."""
    values = tuple(observations)
    if not values:
        return sp.S.Zero
    return sp.simplify(sp.Add(*(log_density(dist, value) for value in values)))


def _discrete_set_probability(dist, event_set: sp.Set):
    support = dist.support
    intersection = sp.Intersection(support, event_set)
    if intersection is sp.S.EmptySet:
        return sp.S.Zero
    if isinstance(intersection, sp.FiniteSet):
        return sp.simplify(sum(dist.pmf(k) for k in intersection))
    if support == sp.S.Naturals0 and isinstance(event_set, sp.Interval):
        lo = max(sp.S.Zero, sp.ceiling(event_set.start))
        hi = sp.floor(event_set.end)
        if event_set.left_open and event_set.start.is_integer is True:
            lo += 1
        if event_set.right_open and event_set.end.is_integer is True:
            hi -= 1
        k = sp.Dummy("k", integer=True, nonnegative=True)
        return sp.simplify(sp.summation(dist.pmf(k), (k, lo, hi)))
    if isinstance(event_set, sp.Interval):
        # Difference of CDFs, correcting the left endpoint for discrete laws.
        upper = cdf(dist, event_set.end)
        left = event_set.start
        if event_set.left_open:
            lower = cdf(dist, left)
        else:
            lower = sp.simplify(cdf(dist, left) - dist.pmf(left))
        if event_set.right_open:
            upper = sp.simplify(upper - dist.pmf(event_set.end))
        return sp.simplify(upper - lower)
    raise ProbabilityFunctionalError(
        f"unsupported discrete probability event: {event_set}"
    )


def _continuous_set_probability(dist, event_set: sp.Set):
    intersection = sp.Intersection(dist.support, event_set)
    if intersection is sp.S.EmptySet:
        return sp.S.Zero
    if intersection == dist.support:
        return sp.S.One
    if isinstance(intersection, sp.Interval):
        return sp.simplify(cdf(dist, intersection.end) - cdf(dist, intersection.start))
    if isinstance(intersection, sp.FiniteSet):
        return sp.S.Zero
    if isinstance(intersection, sp.Union):
        return sp.simplify(
            sum(_continuous_set_probability(dist, part) for part in intersection.args)
        )
    # Do not ask SymPy to integrate arbitrary ConditionSet/Imageset indicators:
    # such reductions can leak the predicate variable or collapse periodic sets
    # to a principal branch. Leave these events to explicit numerical fallback.
    raise ProbabilityFunctionalError(
        f"unsupported continuous probability event: {event_set}"
    )


def _scalar_probability(dist, event: ProbabilityEvent):
    if isinstance(event, PredicateEvent):
        # Finite discrete supports are best evaluated directly, preserving equalities.
        if dist.measure_type is MeasureType.DISCRETE and isinstance(
            dist.support, sp.FiniteSet
        ):
            return sp.simplify(
                sum(
                    dist.pmf(k) * sp.Piecewise((1, event.contains(k)), (0, True))
                    for k in dist.support
                )
            )
        event = SetEvent(event.as_set())
    if not isinstance(event, SetEvent):
        raise ProbabilityFunctionalError(
            "scalar probability requires a set or predicate event"
        )
    if dist.measure_type is MeasureType.DISCRETE:
        return _discrete_set_probability(dist, event.set)
    if dist.measure_type is MeasureType.CONTINUOUS:
        return _continuous_set_probability(dist, event.set)
    raise ProbabilityFunctionalError(
        "generic scalar probability does not support mixed measures"
    )


def _structural_probability(dist, event: ProbabilityEvent):
    # Import here because composition imports this module.
    from .composition import (
        ConditionalDistribution,
        IndependentDistribution,
        MarginalDistribution,
        MixtureDistribution,
        ProductDistribution,
        TruncatedDistribution,
    )

    if isinstance(dist, (MarginalDistribution, ConditionalDistribution)):
        return probability(dist.distribution, event, return_result=True)
    if isinstance(dist, MixtureDistribution):
        parts = [
            probability(component, event, return_result=True)
            for component in dist.components
        ]
        value = sp.simplify(
            sp.Add(*(w * part.value for w, part in zip(dist.weights, parts)))
        )
        return FunctionalResult(
            value, EvaluationMethod.STRUCTURAL, all(part.exact for part in parts)
        )
    if isinstance(dist, TruncatedDistribution) and isinstance(event, SetEvent):
        if sp.Intersection(event.set, dist.support) == dist.support:
            return FunctionalResult(sp.S.One, EvaluationMethod.STRUCTURAL, True)
        restricted = SetEvent(sp.Intersection(event.set, dist.support))
        numerator = probability(dist.base, restricted, return_result=True)
        value = sp.simplify(numerator.value / dist.normalizing_constant)
        return FunctionalResult(value, EvaluationMethod.STRUCTURAL, numerator.exact)
    if isinstance(dist, ProductDistribution) and isinstance(event, ProductEvent):
        if len(event.events) != len(dist.components):
            raise ProbabilityFunctionalError(
                "product event has wrong number of components"
            )
        parts = [
            probability(component, ev, return_result=True)
            for component, ev in zip(dist.components, event.events)
        ]
        value = sp.simplify(reduce(mul, (part.value for part in parts), sp.S.One))
        return FunctionalResult(
            value, EvaluationMethod.STRUCTURAL, all(part.exact for part in parts)
        )
    if isinstance(dist, IndependentDistribution) and isinstance(event, ProductEvent):
        if len(event.events) != dist.count:
            raise ProbabilityFunctionalError(
                "iid product event has wrong number of components"
            )
        parts = [probability(dist.base, ev, return_result=True) for ev in event.events]
        value = sp.simplify(reduce(mul, (part.value for part in parts), sp.S.One))
        return FunctionalResult(
            value, EvaluationMethod.STRUCTURAL, all(part.exact for part in parts)
        )
    return None


def probability(
    dist,
    event,
    *,
    variable=None,
    numerical_fallback: bool = False,
    samples: int = 100_000,
    rng=None,
    return_result: bool = False,
):
    """Compute ``P(event)`` under ``dist`` using exact-first dispatch.

    Events may be SymPy sets, scalar Boolean predicates, or first-class
    :class:`ProbabilityEvent` objects.  Numerical Monte Carlo fallback is never
    implicit; set ``numerical_fallback=True`` to opt in.
    """
    ev = (
        as_event(event, variable=variable)
        if not isinstance(event, ProductEvent)
        else event
    )
    if hasattr(dist, "_probability"):
        direct = dist._probability(ev)
        if direct is not NotImplemented:
            return _wrap(
                sp.simplify(direct),
                EvaluationMethod.STRUCTURAL,
                return_result=return_result,
            )
    structural = _structural_probability(dist, ev)
    if structural is not None:
        return structural if return_result else structural.value
    try:
        if not isinstance(dist.event_space, ScalarEventSpace):
            raise ProbabilityFunctionalError(
                "generic probability requires a scalar distribution or structural product event"
            )
        value = _scalar_probability(dist, ev)
        unevaluated = value.has(sp.Integral, sp.Sum)
        if unevaluated and numerical_fallback:
            result = monte_carlo_probability(
                dist, ev, variable=variable, samples=samples, rng=rng
            )
            return result if return_result else result.value
        method = (
            EvaluationMethod.SYMBOLIC if unevaluated else EvaluationMethod.CLOSED_FORM
        )
        return _wrap(value, method, return_result=return_result)
    except (
        ProbabilityFunctionalError,
        NotImplementedError,
        ValueError,
        TypeError,
    ) as exc:
        if not numerical_fallback:
            raise ProbabilityFunctionalError(str(exc)) from exc
        result = monte_carlo_probability(
            dist, ev, variable=variable, samples=samples, rng=rng
        )
        return result if return_result else result.value


def _infer_scalar_variable(dist, expr, variable):
    if variable is not None:
        return sp.sympify(variable)
    candidates = sorted(expr.free_symbols - dist.free_symbols, key=sp.default_sort_key)
    if len(candidates) == 1:
        return candidates[0]
    return _x()


def _closed_raw_moment(dist, power: int):
    if power == 0:
        return sp.S.One
    if hasattr(dist, "_raw_moment"):
        result = dist._raw_moment(power)
        if result is not NotImplemented:
            return sp.simplify(result)
    if power == 1 and hasattr(dist, "_mean"):
        result = dist._mean()
        if result is not NotImplemented:
            return sp.simplify(result)
    if power == 2 and hasattr(dist, "_mean") and hasattr(dist, "_variance"):
        mu = dist._mean()
        var = dist._variance()
        if mu is not NotImplemented and var is not NotImplemented:
            return sp.simplify(var + mu**2)
    return None


def _polynomial_expectation(dist, expr, variable):
    try:
        poly = sp.Poly(sp.expand(expr), variable)
    except (sp.PolynomialError, TypeError, ValueError):
        return None
    terms = []
    for (power,), coeff in poly.terms():
        moment_value = _closed_raw_moment(dist, power)
        if moment_value is None:
            return None
        terms.append(coeff * moment_value)
    return sp.simplify(sp.Add(*terms))


def _scalar_expectation_exact(dist, expr, variable):
    # Constants do not need integration and this also handles parameter-only expressions.
    if variable not in expr.free_symbols:
        return sp.simplify(expr), EvaluationMethod.STRUCTURAL
    poly = _polynomial_expectation(dist, expr, variable)
    if poly is not None:
        return poly, EvaluationMethod.STRUCTURAL
    support = dist.support
    if dist.measure_type is MeasureType.DISCRETE:
        if isinstance(support, sp.FiniteSet):
            return sp.simplify(
                sum(expr.subs(variable, k) * dist.pmf(k) for k in support)
            ), EvaluationMethod.SYMBOLIC
        if support == sp.S.Naturals0:
            k = sp.Dummy("k", integer=True, nonnegative=True)
            value = sp.summation(expr.subs(variable, k) * dist.pmf(k), (k, 0, sp.oo))
            return sp.simplify(value), EvaluationMethod.SYMBOLIC
        raise ProbabilityFunctionalError("unsupported discrete support for expectation")
    integrand = expr * dist.pdf(variable)
    if support == sp.S.Reals:
        value = sp.integrate(integrand, (variable, -sp.oo, sp.oo))
        return sp.simplify(value), EvaluationMethod.SYMBOLIC
    if isinstance(support, sp.Interval):
        value = sp.integrate(integrand, (variable, support.start, support.end))
        return sp.simplify(value), EvaluationMethod.SYMBOLIC
    raise ProbabilityFunctionalError("unsupported continuous support for expectation")


def _product_expectation(dist, expr, variables):
    from .composition import IndependentDistribution, ProductDistribution

    if isinstance(dist, (ProductDistribution, IndependentDistribution)):
        components = dist.components
    else:
        return None
    vars_ = tuple(map(sp.sympify, variables))
    if len(vars_) != len(components):
        raise ProbabilityFunctionalError(
            "variables must have one symbol per product component"
        )
    value = sp.sympify(expr)
    # Iterated expectation is exact for independent product laws and handles
    # arbitrary cross-products, not just separable expressions.
    for component, var in zip(components, vars_):
        value = expectation(component, value, variable=var)
    return sp.simplify(value)


def expectation(
    dist,
    expression=None,
    *,
    variable=None,
    variables=None,
    numerical_fallback: bool = False,
    samples: int = 100_000,
    rng=None,
    return_result: bool = False,
    assumptions=None,
):
    """Expectation under a distribution or of a symbolic random expression.

    For product distributions, pass ``variables=(x1, x2, ...)`` to evaluate a
    joint expression by iterated expectation.  Monte Carlo is opt-in.
    """
    if (
        expression is None
        and isinstance(dist, sp.Basic)
        and not hasattr(dist, "event_space")
    ):
        from .random_variable_algebra import expectation as random_expectation

        if (
            variable is not None
            or variables is not None
            or numerical_fallback
            or return_result
        ):
            raise TypeError(
                "distribution-evaluation options do not apply to symbolic expectation"
            )
        return random_expectation(dist, assumptions=assumptions)
    if assumptions is not None:
        raise TypeError(
            "assumptions apply only to symbolic random-variable expressions"
        )
    if expression is None:
        value = mean(dist)
        return _wrap(value, EvaluationMethod.CLOSED_FORM, return_result=return_result)
    if callable(expression):
        if variables is not None:
            expression = expression(*variables)
        else:
            variable = variable or (
                _x() if isinstance(dist.event_space, ScalarEventSpace) else None
            )
            if variable is None:
                if numerical_fallback:
                    result = monte_carlo_expectation(
                        dist, expression, samples=samples, rng=rng
                    )
                    return result if return_result else result.value
                raise ProbabilityFunctionalError(
                    "a variable is required for multivariate callable expectations"
                )
            expression = expression(variable)
    expr = sp.sympify(expression)

    if variables is not None:
        try:
            value = _product_expectation(dist, expr, variables)
            if value is not None:
                return _wrap(
                    value, EvaluationMethod.STRUCTURAL, return_result=return_result
                )
        except (
            ProbabilityFunctionalError,
            NotImplementedError,
            ValueError,
            TypeError,
        ) as exc:
            if numerical_fallback:
                result = monte_carlo_expectation(
                    dist, expr, variables=variables, samples=samples, rng=rng
                )
                return result if return_result else result.value
            raise ProbabilityFunctionalError(str(exc)) from exc

    if hasattr(dist, "_expectation"):
        result = dist._expectation(expr, variable)
        if result is not NotImplemented:
            result = sp.simplify(result)
            return _wrap(
                result, EvaluationMethod.STRUCTURAL, return_result=return_result
            )
    if not isinstance(dist.event_space, ScalarEventSpace):
        if numerical_fallback:
            result = monte_carlo_expectation(
                dist, expr, variable=variable, samples=samples, rng=rng
            )
            return result if return_result else result.value
        raise ProbabilityFunctionalError(
            "generic expectation requires a scalar event space or product variables"
        )
    x = _infer_scalar_variable(dist, expr, variable)
    try:
        value, method = _scalar_expectation_exact(dist, expr, x)
        if value.has(sp.Integral, sp.Sum) and numerical_fallback:
            result = monte_carlo_expectation(
                dist, expr, variable=x, samples=samples, rng=rng
            )
            return result if return_result else result.value
        return _wrap(value, method, return_result=return_result)
    except (
        ProbabilityFunctionalError,
        NotImplementedError,
        ValueError,
        TypeError,
    ) as exc:
        if not numerical_fallback:
            raise ProbabilityFunctionalError(str(exc)) from exc
        result = monte_carlo_expectation(
            dist, expr, variable=x, samples=samples, rng=rng
        )
        return result if return_result else result.value


def raw_moment(dist, order: int, *, assumptions=None):
    """Return a distribution raw moment or symbolic random-expression moment."""
    if isinstance(dist, sp.Basic):
        from .random_variable_algebra import is_random_expression
        from .random_variable_algebra import raw_moment as random_raw_moment

        if is_random_expression(dist):
            return random_raw_moment(dist, order, assumptions=assumptions)
    if assumptions is not None:
        raise TypeError(
            "assumptions apply only to symbolic random-variable expressions"
        )
    if order < 0:
        raise ValueError("moment order must be nonnegative")
    if hasattr(dist, "_raw_moment"):
        result = dist._raw_moment(order)
        if result is not NotImplemented:
            return sp.simplify(result)
    x = _x()
    return expectation(dist, x**order, variable=x)


def moment(dist, order: int, *, central: bool = False, assumptions=None):
    """Return a raw or central moment for a distribution or random expression."""
    if isinstance(dist, sp.Basic):
        from .random_variable_algebra import is_random_expression
        from .random_variable_algebra import moment as random_moment

        if is_random_expression(dist):
            return random_moment(dist, order, central=central, assumptions=assumptions)
    if assumptions is not None:
        raise TypeError(
            "assumptions apply only to symbolic random-variable expressions"
        )
    if not central:
        return raw_moment(dist, order)
    x = _x()
    mu = mean(dist)
    return expectation(dist, (x - mu) ** order, variable=x)


def central_moment(dist, order: int, *, assumptions=None):
    """Return a central moment for a distribution or symbolic random expression."""
    return moment(dist, order, central=True, assumptions=assumptions)


def mean(dist, *, assumptions=None):
    """Return a distribution mean or symbolic random-expression expectation."""
    if isinstance(dist, sp.Basic):
        from .random_variable_algebra import is_random_expression

        if is_random_expression(dist):
            return expectation(dist, assumptions=assumptions)
    if assumptions is not None:
        raise TypeError(
            "assumptions apply only to symbolic random-variable expressions"
        )
    if isinstance(dist, WeightedData):
        result = dist.mean()
        return float(result) if getattr(result, "ndim", 0) == 0 else result
    if hasattr(dist, "_mean"):
        result = dist._mean()
        if result is not NotImplemented:
            return result
    return raw_moment(dist, 1)


def variance(dist, ddof=None, *, assumptions=None):
    """Return distribution variance or symbolic random-expression variance."""
    if isinstance(dist, sp.Basic):
        from .random_variable_algebra import is_random_expression
        from .random_variable_algebra import variance as random_variance

        if is_random_expression(dist):
            if ddof not in (None, 0):
                raise TypeError("ddof applies only to observed data")
            return random_variance(dist, assumptions=assumptions)
    if assumptions is not None:
        raise TypeError(
            "assumptions apply only to symbolic random-variable expressions"
        )
    if isinstance(dist, WeightedData):
        effective_ddof = 1 if ddof is None else ddof
        result = dist.variance(ddof=effective_ddof)
        return float(result) if getattr(result, "ndim", 0) == 0 else result
    if ddof not in (None, 0):
        raise ValueError("ddof applies only to sample/WeightedData variance")
    if hasattr(dist, "_variance"):
        result = dist._variance()
        if result is not NotImplemented:
            return result
    m = mean(dist)
    return sp.simplify(raw_moment(dist, 2) - m**2)


def covariance(left, right, *, assumptions=None):
    """Return covariance of two symbolic random expressions."""
    from .random_variable_algebra import (
        covariance as random_covariance,
    )
    from .random_variable_algebra import (
        is_random_expression,
    )

    if not (
        isinstance(left, sp.Basic)
        and isinstance(right, sp.Basic)
        and is_random_expression(left)
        and is_random_expression(right)
    ):
        raise TypeError("functionals.covariance requires two random expressions")
    return random_covariance(left, right, assumptions=assumptions)


def entropy(dist):
    if hasattr(dist, "_entropy"):
        result = dist._entropy()
        if result is not NotImplemented:
            return sp.simplify(result)
    if not isinstance(dist.event_space, ScalarEventSpace):
        raise ProbabilityFunctionalError(
            "generic entropy requires a scalar event space"
        )
    x = _x()
    d = density(dist, x)
    return sp.simplify(-expectation(dist, sp.log(d), variable=x))


__all__ = [
    "ProbabilityFunctionalError",
    "cdf",
    "central_moment",
    "covariance",
    "cumulant",
    "cumulative_hazard",
    "density",
    "entropy",
    "expectation",
    "factorial_moment",
    "hazard",
    "inverse_survival",
    "likelihood_value",
    "log_density",
    "log_likelihood",
    "mean",
    "moment",
    "probability",
    "quantile",
    "raw_moment",
    "survival",
    "variance",
]
