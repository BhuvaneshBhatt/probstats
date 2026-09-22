"""Optimization backends used by Laplace inference."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Protocol

import numpy as np
import sympy as sp
from sympy.core.relational import Relational


class OptimizationError(RuntimeError):
    """Raised when a MAP optimization backend cannot produce a usable optimum."""


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """Backend-independent MAP optimization result."""

    point: tuple[float, ...]
    value: float
    success: bool = True
    message: str = ""
    raw: Any = None
    certified: bool | None = None
    attained: bool | None = None
    status: str | None = None
    global_value: Any = None

    def as_substitutions(
        self, variables: Sequence[sp.Symbol]
    ) -> dict[sp.Symbol, float]:
        return dict(zip(variables, self.point, strict=True))


class OptimizationBackend(Protocol):
    """Protocol implemented by Laplace MAP optimization backends."""

    name: str

    def optimize(
        self,
        log_density: sp.Expr,
        variables: Sequence[sp.Symbol],
        *,
        supports: Sequence[sp.Set] | None = None,
        initial_guess: Sequence[float] | Mapping[sp.Symbol | str, float] | None = None,
        assumptions: sp.Expr = sp.true,
        options: Mapping[str, Any] | None = None,
    ) -> OptimizationResult: ...


def _guess_vector(
    variables: Sequence[sp.Symbol],
    initial_guess: Sequence[float] | Mapping[sp.Symbol | str, float] | None,
) -> np.ndarray | None:
    if initial_guess is None:
        return None
    if isinstance(initial_guess, Mapping):
        vals = []
        for var in variables:
            if var in initial_guess:
                vals.append(initial_guess[var])
            elif var.name in initial_guess:
                vals.append(initial_guess[var.name])
            else:
                raise ValueError(f"Missing initial guess for {var}.")
        return np.asarray(vals, dtype=float)
    arr = np.asarray(initial_guess, dtype=float)
    if arr.shape != (len(variables),):
        raise ValueError(
            f"Expected {len(variables)} initial values, got shape {arr.shape}."
        )
    return arr


def _contains(support: sp.Set, value: float) -> bool:
    contains = support.contains(sp.Float(value))
    return contains is not sp.false


def _assumption_conditions(assumptions: Any) -> tuple[sp.Basic, ...]:
    if assumptions in (None, True, sp.true):
        return ()
    asm = sp.sympify(assumptions)
    if asm in (True, sp.true):
        return ()
    if asm in (False, sp.false):
        raise OptimizationError("Optimization assumptions are inconsistent (False).")
    return tuple(asm.args) if asm.func is sp.And else (asm,)


def _supports_with_simple_assumptions(
    variables: Sequence[sp.Symbol],
    supports: Sequence[sp.Set],
    assumptions: Any,
) -> tuple[sp.Set, ...]:
    """Intersect supports with simple one-variable relational assumptions.

    SymPy/SciPy backends reject target-variable constraints they
    cannot enforce instead of silently optimizing the wrong problem.
    """
    variables = tuple(variables)
    variable_set = set(variables)
    out = list(supports)
    for condition in _assumption_conditions(assumptions):
        if not isinstance(condition, Relational):
            if condition.free_symbols & variable_set:
                raise OptimizationError(
                    f"Unsupported target-variable optimization assumption: {condition}. "
                    "Use symbopt for general symbolic constraints."
                )
            continue
        involved = condition.free_symbols & variable_set
        if not involved:
            continue
        if len(involved) != 1:
            raise OptimizationError(
                f"Coupled optimization constraint {condition} is unsupported by this backend; use symbopt."
            )
        var = next(iter(involved))
        other = None
        relation = condition
        if condition.lhs == var and not (condition.rhs.free_symbols & variable_set):
            other = condition.rhs
            left_side = True
        elif condition.rhs == var and not (condition.lhs.free_symbols & variable_set):
            other = condition.lhs
            left_side = False
        else:
            raise OptimizationError(
                f"Constraint {condition} is not a simple bound on {var}; use symbopt."
            )
        if other.free_symbols:
            raise OptimizationError(
                f"Constraint bound {other} for {var} is symbolic; SymPy/SciPy cannot enforce it numerically. Use symbopt."
            )
        try:
            float(sp.N(other))
        except (TypeError, ValueError):
            raise OptimizationError(
                f"Constraint bound {other} for {var} is not numeric."
            )
        if isinstance(relation, sp.StrictGreaterThan):
            interval = (
                sp.Interval.open(other, sp.oo)
                if left_side
                else sp.Interval.open(-sp.oo, other)
            )
        elif isinstance(relation, sp.GreaterThan):
            interval = (
                sp.Interval(other, sp.oo) if left_side else sp.Interval(-sp.oo, other)
            )
        elif isinstance(relation, sp.StrictLessThan):
            interval = (
                sp.Interval.open(-sp.oo, other)
                if left_side
                else sp.Interval.open(other, sp.oo)
            )
        elif isinstance(relation, sp.LessThan):
            interval = (
                sp.Interval(-sp.oo, other) if left_side else sp.Interval(other, sp.oo)
            )
        else:
            raise OptimizationError(
                f"Equality/inequality constraint {condition} is unsupported by this backend; use symbopt."
            )
        index = variables.index(var)
        out[index] = sp.Intersection(out[index], interval)
        if out[index] is sp.S.EmptySet:
            raise OptimizationError(f"Assumptions make support of {var} empty.")
    return tuple(out)


class SymPyOptimizationBackend:
    """Symbolic stationary-point optimizer with an ``nsolve`` fallback.

    The symbolic path solves the gradient equations and selects the feasible stationary
    point with the largest objective value. If symbolic solving does not yield a usable
    candidate and an initial guess is supplied, ``sympy.nsolve`` is attempted.
    """

    name = "sympy"

    def optimize(
        self,
        log_density: sp.Expr,
        variables: Sequence[sp.Symbol],
        *,
        supports: Sequence[sp.Set] | None = None,
        initial_guess: Sequence[float] | Mapping[sp.Symbol | str, float] | None = None,
        assumptions: sp.Expr = sp.true,
        options: Mapping[str, Any] | None = None,
    ) -> OptimizationResult:
        del options
        variables = tuple(variables)
        supports = tuple(supports or (sp.S.Reals,) * len(variables))
        supports = _supports_with_simple_assumptions(variables, supports, assumptions)
        gradient = [sp.diff(log_density, var) for var in variables]
        candidates: list[tuple[tuple[float, ...], float]] = []

        try:
            solved = sp.solve(gradient, variables, dict=True)
        except (TypeError, ValueError, NotImplementedError, sp.PolynomialError):
            solved = []
        for solution in solved:
            if any(var not in solution for var in variables):
                continue
            try:
                point = tuple(float(sp.N(solution[var])) for var in variables)
            except (TypeError, ValueError):
                continue
            if not all(np.isfinite(point)):
                continue
            if not all(
                _contains(support, val)
                for support, val in zip(supports, point, strict=True)
            ):
                continue
            value_expr = log_density.subs(dict(zip(variables, point, strict=True)))
            try:
                value = float(sp.N(value_expr))
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                candidates.append((point, value))

        if candidates:
            point, value = max(candidates, key=lambda item: item[1])
            return OptimizationResult(
                point=point, value=value, message="symbolic stationary point"
            )

        guess = _guess_vector(variables, initial_guess)
        if guess is not None:
            try:
                root = sp.nsolve(gradient, variables, tuple(guess))
                root_arr = np.asarray(root, dtype=float).reshape(-1)
                point = tuple(float(v) for v in root_arr)
                if not all(
                    _contains(support, val)
                    for support, val in zip(supports, point, strict=True)
                ):
                    raise OptimizationError(
                        "nsolve stationary point lies outside parameter support."
                    )
                value = float(
                    sp.N(log_density.subs(dict(zip(variables, point, strict=True))))
                )
                return OptimizationResult(
                    point=point, value=value, message="sympy.nsolve"
                )
            except (
                TypeError,
                ValueError,
                NotImplementedError,
                ZeroDivisionError,
                sp.PolynomialError,
            ) as exc:
                raise OptimizationError(f"SymPy optimization failed: {exc}") from exc

        raise OptimizationError(
            "SymPy could not identify a feasible stationary point; provide an initial guess "
            "or use the SciPy backend."
        )


class SciPyOptimizationBackend:
    """Numerical MAP optimization using :func:`scipy.optimize.minimize`."""

    name = "scipy"

    def optimize(
        self,
        log_density: sp.Expr,
        variables: Sequence[sp.Symbol],
        *,
        supports: Sequence[sp.Set] | None = None,
        initial_guess: Sequence[float] | Mapping[sp.Symbol | str, float] | None = None,
        assumptions: sp.Expr = sp.true,
        options: Mapping[str, Any] | None = None,
    ) -> OptimizationResult:
        try:
            from scipy.optimize import minimize
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise OptimizationError("SciPy is not installed.") from exc

        variables = tuple(variables)
        supports = tuple(supports or (sp.S.Reals,) * len(variables))
        supports = _supports_with_simple_assumptions(variables, supports, assumptions)
        guess = _guess_vector(variables, initial_guess)
        if guess is None:
            guess = np.asarray(
                [_default_initial_value(s) for s in supports], dtype=float
            )

        fun_raw = sp.lambdify(variables, log_density, modules="numpy")
        grad_expr = [sp.diff(log_density, var) for var in variables]
        grad_raw = sp.lambdify(variables, grad_expr, modules="numpy")

        def objective(x: np.ndarray) -> float:
            try:
                value = float(np.asarray(fun_raw(*x)))
            except (
                TypeError,
                ValueError,
                OverflowError,
                ZeroDivisionError,
                FloatingPointError,
            ):
                return np.inf
            return -value if np.isfinite(value) else np.inf

        def jac(x: np.ndarray) -> np.ndarray:
            try:
                value = np.asarray(grad_raw(*x), dtype=float).reshape(-1)
                return -value
            except (
                TypeError,
                ValueError,
                OverflowError,
                ZeroDivisionError,
                FloatingPointError,
            ):
                return np.full(len(variables), np.nan)

        bounds = [_scipy_bound(s) for s in supports]
        opts = dict(options or {})
        method = opts.pop(
            "method", "L-BFGS-B" if any(b != (None, None) for b in bounds) else "BFGS"
        )
        result = minimize(
            objective,
            guess,
            jac=jac,
            bounds=bounds
            if method.upper() in {"L-BFGS-B", "TNC", "SLSQP", "POWELL", "NELDER-MEAD"}
            else None,
            method=method,
            options=opts or None,
        )
        if not result.success or not np.isfinite(result.fun):
            raise OptimizationError(f"SciPy optimization failed: {result.message}")
        point = tuple(float(v) for v in result.x)
        if not all(
            _contains(support, val)
            for support, val in zip(supports, point, strict=True)
        ):
            raise OptimizationError(
                "SciPy optimum lies outside parameter support/assumption bounds."
            )
        return OptimizationResult(
            point=point,
            value=float(-result.fun),
            success=True,
            message=str(result.message),
            raw=result,
        )


def _default_initial_value(support: sp.Set) -> float:
    if isinstance(support, sp.Interval):
        left = float(support.start) if support.start.is_finite else None
        right = float(support.end) if support.end.is_finite else None
        if left is not None and right is not None:
            return (left + right) / 2.0
        if left is not None:
            return left + max(1.0, abs(left) * 0.1)
        if right is not None:
            return right - max(1.0, abs(right) * 0.1)
    return 0.0


def _scipy_bound(support: sp.Set) -> tuple[float | None, float | None]:
    if not isinstance(support, sp.Interval):
        return (None, None)
    lower = None if support.start is sp.S.NegativeInfinity else float(support.start)
    upper = None if support.end is sp.S.Infinity else float(support.end)
    # scipy bounds are inclusive. Nudge open finite endpoints inward.
    eps = np.finfo(float).eps ** 0.5
    if lower is not None and support.left_open:
        lower += eps * max(1.0, abs(lower))
    if upper is not None and support.right_open:
        upper -= eps * max(1.0, abs(upper))
    return lower, upper


class SymboptOptimizationBackend:
    """Certified MAP optimization through :mod:`symbopt`.

    The backend translates continuous SymPy supports and explicit relational
    assumptions into ``symbopt.maximize`` constraints.  It then preserves the
    distinction made by symbopt between a represented feasible optimizer and a
    certified global optimum.  Laplace approximation requires an *attained*
    optimizer; an exact but unattained supremum is therefore rejected.

    Parameters passed in ``options`` are forwarded to ``symbopt.maximize``.
    This includes symbopt controls such as ``working_precision``,
    ``semialg_certification``, interval/SOS settings, and solver budgets.
    """

    name = "symbopt"

    def __init__(self, *, loader: Callable[[], Any] | None = None) -> None:
        self._loader = loader

    def _load(self) -> Any:
        return import_module("symbopt") if self._loader is None else self._loader()

    def optimize(
        self,
        log_density: sp.Expr,
        variables: Sequence[sp.Symbol],
        *,
        supports: Sequence[sp.Set] | None = None,
        initial_guess: Sequence[float] | Mapping[sp.Symbol | str, float] | None = None,
        assumptions: sp.Expr = sp.true,
        options: Mapping[str, Any] | None = None,
    ) -> OptimizationResult:
        """Maximize ``log_density`` with symbopt and return a MAP result.

        ``supports`` are converted to exact relational constraints.  A finite
        ``initial_guess`` is validated for dimensional consistency but is not
        forwarded because symbopt's public optimization pipeline is global and
        does not require a local starting point.
        """
        opts = dict(options or {})
        adapter = opts.pop("optimizer", None)
        if adapter is not None:
            try:
                raw = adapter(sp.sympify(log_density), tuple(variables), **opts)
            except (ArithmeticError, RuntimeError, TypeError, ValueError) as exc:
                raise OptimizationError(f"symbopt adapter failed: {exc}") from exc
            return _coerce_external_result(raw)

        try:
            module = self._load()
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise OptimizationError(
                "symbopt is not installed; install probstats[symbopt] "
                "or select backend='sympy'/'scipy'."
            ) from exc

        maximize = getattr(module, "maximize", None)
        if maximize is None:
            raise OptimizationError(
                "Installed symbopt does not expose its supported maximize() API."
            )

        variables = tuple(variables)
        supports = tuple(supports or (sp.S.Reals,) * len(variables))
        if len(supports) != len(variables):
            raise ValueError("supports must match the number of variables.")
        if initial_guess is not None:
            _guess_vector(variables, initial_guess)  # validate only

        constraints = _symbopt_constraints(variables, supports, assumptions)
        # Prevent accidental collision with arguments owned by this adapter.
        for reserved in ("variables", "constraints", "objective"):
            if reserved in opts:
                raise ValueError(f"optimizer_options must not contain {reserved!r}.")

        try:
            raw = maximize(
                sp.sympify(log_density),
                constraints,
                variables=list(variables),
                **opts,
            )
        except (ArithmeticError, RuntimeError, TypeError, ValueError) as exc:
            raise OptimizationError(f"symbopt optimization failed: {exc}") from exc
        return _coerce_symbopt_result(raw, variables, log_density)


def _symbopt_constraints(
    variables: Sequence[sp.Symbol],
    supports: Sequence[sp.Set],
    assumptions: sp.Expr,
) -> list[Relational]:
    """Translate Laplace supports/assumptions into symbopt constraints."""
    constraints: list[Relational] = []
    for var, support in zip(variables, supports, strict=True):
        if support is sp.S.Reals:
            continue
        if not isinstance(support, sp.Interval):
            raise OptimizationError(
                f"symbopt Laplace backend requires continuous real/interval support; "
                f"{var} has {support}."
            )
        if support.start is not sp.S.NegativeInfinity:
            constraints.append(
                sp.Gt(var, support.start)
                if support.left_open
                else sp.Ge(var, support.start)
            )
        if support.end is not sp.S.Infinity:
            constraints.append(
                sp.Lt(var, support.end)
                if support.right_open
                else sp.Le(var, support.end)
            )

    assumption_expr = sp.sympify(assumptions)
    if assumption_expr not in (sp.true, True):
        pieces = sp.And.make_args(assumption_expr)
        for piece in pieces:
            if isinstance(piece, Relational):
                constraints.append(piece)
            elif piece not in (sp.true, True):
                raise OptimizationError(
                    "symbopt assumptions must be relational constraints or an And of them."
                )
    return constraints


def _coerce_symbopt_result(
    raw: Any, variables: Sequence[sp.Symbol], log_density: sp.Expr
) -> OptimizationResult:
    """Convert a symbopt ``PipelineResult`` without discarding certification."""
    attained = getattr(raw, "attained", None)
    certified = bool(getattr(raw, "certified", False))
    status = getattr(raw, "status", None)
    global_value = getattr(raw, "optimum_value", None)
    candidate = getattr(raw, "best_candidate", None)

    if attained is False and candidate is None:
        raise OptimizationError(
            "symbopt found a supremum that is not attained; Laplace approximation "
            "requires a finite MAP point."
        )
    if candidate is None:
        raise OptimizationError(
            "symbopt did not return a represented feasible optimizer for the MAP."
        )

    mapping = getattr(candidate, "original_variable_values", None)
    if mapping is None:
        mapping = getattr(candidate, "primal_values", None)
    if mapping is None:
        raise OptimizationError(
            "symbopt best_candidate does not expose variable values."
        )

    try:
        point = tuple(float(sp.N(mapping[var])) for var in variables)
    except (KeyError, TypeError, ValueError) as exc:
        raise OptimizationError(
            "symbopt optimizer could not be projected to all Laplace variables."
        ) from exc
    if not np.all(np.isfinite(point)):
        raise OptimizationError("symbopt returned a non-finite MAP point.")

    candidate_value = getattr(candidate, "objective_value", None)
    if candidate_value is None:
        candidate_value = log_density.subs(dict(zip(variables, point, strict=True)))
    try:
        value = float(sp.N(candidate_value))
    except (TypeError, ValueError) as exc:
        raise OptimizationError("symbopt returned a non-numeric MAP value.") from exc

    return OptimizationResult(
        point=point,
        value=value,
        success=True,
        message=(
            f"symbopt status={status or 'unknown'}; "
            f"certified={certified}; attained={attained}"
        ),
        raw=raw,
        certified=certified,
        attained=attained,
        status=str(status) if status is not None else None,
        global_value=global_value,
    )


def _coerce_external_result(raw: Any) -> OptimizationResult:
    if isinstance(raw, OptimizationResult):
        return raw
    if isinstance(raw, tuple) and len(raw) == 2:
        point, value = raw
        return OptimizationResult(tuple(float(v) for v in point), float(value), raw=raw)
    point = getattr(raw, "point", None)
    if point is None:
        point = getattr(raw, "x", None)
    value = getattr(raw, "value", None)
    if value is None and hasattr(raw, "fun"):
        # scipy-style result convention minimizes; external adapter can override with value.
        value = -float(raw.fun)
    if point is None or value is None:
        raise OptimizationError(
            "Unsupported result returned by external optimization backend."
        )
    success = bool(getattr(raw, "success", True))
    if not success:
        raise OptimizationError(
            str(getattr(raw, "message", "external optimizer failed"))
        )
    return OptimizationResult(
        tuple(float(v) for v in point),
        float(value),
        success=success,
        message=str(getattr(raw, "message", "")),
        raw=raw,
    )


def get_optimization_backend(backend: str | OptimizationBackend) -> OptimizationBackend:
    """Resolve a backend name or return an already constructed backend.

    ``"auto"`` prefers symbopt, then SciPy, then the built-in SymPy backend.
    Optional dependencies are imported lazily.
    """
    if not isinstance(backend, str):
        return backend
    name = backend.lower().replace("_", "-")
    if name == "sympy":
        return SymPyOptimizationBackend()
    if name == "scipy":
        return SciPyOptimizationBackend()
    if name == "symbopt":
        return SymboptOptimizationBackend()
    if name == "auto":
        try:
            import_module("symbopt")
        except ImportError:
            try:
                import_module("scipy")
            except ImportError:
                return SymPyOptimizationBackend()
            return SciPyOptimizationBackend()
        return SymboptOptimizationBackend()
    raise ValueError(f"Unknown optimization backend {backend!r}.")
