"""Second-order Laplace posterior and evidence approximation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import sympy as sp

from ..._exact_linear_algebra import exact_determinant
from ..certification import certify_zero
from ..core import (
    InferenceKind,
    InferenceResult,
    InferenceStep,
    Model,
    Variable,
)
from ..reasoning import (
    analyze_posterior_geometry,
    certify_positive_definite,
)
from .asymptotic import (
    AsymptoticCorrector,
    AsymptoticIntegrationError,
    AsymptoticLaplaceAnalysis,
)
from .backends import (
    OptimizationBackend,
    OptimizationError,
    OptimizationResult,
    get_optimization_backend,
)


class LaplaceApproximationError(RuntimeError):
    """Raised when a valid local Gaussian approximation cannot be constructed."""


class HigherOrderCorrector(Protocol):
    """Hook used by optional asymptotic engines to refine second-order Laplace."""

    name: str

    def correction(
        self,
        *,
        log_density: sp.Expr,
        variables: tuple[sp.Symbol, ...],
        point: tuple[float, ...],
        precision: np.ndarray,
        order: int,
    ) -> Any:
        """Return an additive correction to the second-order log evidence."""
        ...


@dataclass(frozen=True, slots=True)
class GaussianLaplacePosterior:
    """Gaussian posterior approximation centered at the MAP."""

    variables: tuple[sp.Symbol, ...]
    mean: tuple[float, ...]
    covariance: np.ndarray
    precision: np.ndarray

    def __post_init__(self) -> None:
        cov = np.asarray(self.covariance, dtype=float)
        precision = np.asarray(self.precision, dtype=float)
        n = len(self.variables)
        if cov.shape != (n, n) or precision.shape != (n, n):
            raise ValueError(
                "Covariance and precision matrices must match variable count."
            )
        if not np.all(np.isfinite(cov)) or not np.all(np.isfinite(precision)):
            raise ValueError(
                "Covariance and precision matrices must contain only finite values."
            )
        if not np.allclose(cov, cov.T) or not np.allclose(precision, precision.T):
            raise ValueError("Covariance and precision matrices must be symmetric.")
        if np.linalg.slogdet(cov)[0] <= 0 or np.linalg.slogdet(precision)[0] <= 0:
            raise ValueError(
                "Covariance and precision matrices must be positive definite."
            )
        if not np.allclose(cov @ precision, np.eye(n), rtol=1e-7, atol=1e-9):
            raise ValueError(
                "Covariance and precision matrices must be numerical inverses."
            )
        object.__setattr__(self, "covariance", cov)
        object.__setattr__(self, "precision", precision)
        object.__setattr__(self, "mean", tuple(float(v) for v in self.mean))

    @property
    def dimension(self) -> int:
        return len(self.variables)

    def logpdf(self, values: Sequence[Any] | Mapping[sp.Symbol | str, Any]) -> sp.Expr:
        vals = _value_vector(self.variables, values)
        delta = sp.Matrix(vals) - sp.Matrix(self.mean)
        precision = sp.Matrix(self.precision)
        sign, logdet_value = np.linalg.slogdet(self.covariance)
        if sign <= 0:
            raise ValueError("Covariance matrix must be positive definite.")
        logdet = sp.Float(logdet_value)
        return sp.simplify(
            -sp.Rational(self.dimension, 2) * sp.log(2 * sp.pi)
            - logdet / 2
            - (delta.T * precision * delta)[0] / 2
        )

    def pdf(self, values: Sequence[Any] | Mapping[sp.Symbol | str, Any]) -> sp.Expr:
        return sp.exp(self.logpdf(values))


@dataclass(frozen=True, slots=True)
class SingularLaplacePosterior:
    """Leading separable posterior approximation at a degenerate MAP.

    Each coordinate is represented by its first nonzero even local decay term,
    ``exp(-a_i |x_i-mode_i|**m_i)``.  The object is used only when ordinary
    Gaussian Laplace fails because one or more Hessian eigenvalues vanish.
    """

    variables: tuple[sp.Symbol, ...]
    mode: tuple[float, ...]
    local_orders: tuple[int, ...]
    local_coefficients: tuple[sp.Expr, ...]

    def __post_init__(self) -> None:
        n = len(self.variables)
        if not (
            len(self.mode)
            == len(self.local_orders)
            == len(self.local_coefficients)
            == n
        ):
            raise ValueError("Singular posterior fields must have matching dimensions.")
        if any(order < 2 or order % 2 for order in self.local_orders):
            raise ValueError("Local singular orders must be even integers >= 2.")
        object.__setattr__(self, "mode", tuple(float(v) for v in self.mode))
        object.__setattr__(
            self, "local_coefficients", tuple(map(sp.sympify, self.local_coefficients))
        )

    @property
    def dimension(self) -> int:
        """Number of approximated posterior coordinates."""
        return len(self.variables)

    @staticmethod
    def _log_normalizer(order: int, coefficient: sp.Expr) -> sp.Expr:
        return sp.log(
            2
            * sp.gamma(sp.Rational(1, order))
            / (order * coefficient ** sp.Rational(1, order))
        )

    def logpdf(self, values: Sequence[Any] | Mapping[sp.Symbol | str, Any]) -> sp.Expr:
        """Evaluate the normalized leading singular posterior log density."""
        vals = _value_vector(self.variables, values)
        terms = []
        for value, center, order, coefficient in zip(
            vals, self.mode, self.local_orders, self.local_coefficients, strict=True
        ):
            delta = sp.sympify(value) - sp.Float(center)
            terms.append(
                -coefficient * delta**order - self._log_normalizer(order, coefficient)
            )
        return sp.simplify(sp.Add(*terms))

    def pdf(self, values: Sequence[Any] | Mapping[sp.Symbol | str, Any]) -> sp.Expr:
        """Evaluate the normalized leading singular posterior density."""
        return sp.exp(self.logpdf(values))


def _value_vector(
    variables: tuple[sp.Symbol, ...],
    values: Sequence[Any] | Mapping[sp.Symbol | str, Any],
) -> tuple[sp.Expr, ...]:
    if isinstance(values, Mapping):
        out = []
        for var in variables:
            if var in values:
                out.append(sp.sympify(values[var]))
            elif var.name in values:
                out.append(sp.sympify(values[var.name]))
            else:
                raise KeyError(f"Missing value for {var}.")
        return tuple(out)
    vals = tuple(sp.sympify(v) for v in values)
    if len(vals) != len(variables):
        raise ValueError("Value vector length does not match posterior dimension.")
    return vals


def symbolic_hessian(expression: sp.Expr, variables: Sequence[sp.Symbol]) -> sp.Matrix:
    """Return the exact symbolic Hessian matrix."""
    vars_tuple = tuple(variables)
    return sp.hessian(sp.sympify(expression), vars_tuple)


def precision_at(
    log_density: sp.Expr,
    variables: Sequence[sp.Symbol],
    point: Sequence[float] | Mapping[sp.Symbol | str, float],
) -> np.ndarray:
    """Evaluate ``-H(log density)`` at a candidate posterior maximum."""
    vars_tuple = tuple(variables)
    if isinstance(point, Mapping):
        subs = {}
        for var in vars_tuple:
            if var in point:
                subs[var] = point[var]
            elif var.name in point:
                subs[var] = point[var.name]
            else:
                raise KeyError(f"Missing point value for {var}.")
    else:
        vals = tuple(point)
        if len(vals) != len(vars_tuple):
            raise ValueError("Point dimension does not match variables.")
        subs = dict(zip(vars_tuple, vals, strict=True))
    hess = symbolic_hessian(log_density, vars_tuple)
    evaluated = np.asarray(hess.subs(subs).evalf(), dtype=float)
    if not np.all(np.isfinite(evaluated)):
        raise LaplaceApproximationError(
            "Hessian evaluation produced non-finite values."
        )
    return -0.5 * (evaluated + evaluated.T)


def laplace_log_evidence(log_density_at_mode: Any, precision: np.ndarray) -> sp.Expr:
    """Standard second-order Laplace approximation to a log integral."""
    precision = np.asarray(precision, dtype=float)
    if precision.ndim != 2 or precision.shape[0] != precision.shape[1]:
        raise ValueError("precision must be a square matrix.")
    if not np.all(np.isfinite(precision)):
        raise ValueError("precision must contain only finite values.")
    sign, logdet = np.linalg.slogdet(precision)
    if sign <= 0:
        raise LaplaceApproximationError("Posterior precision is not positive definite.")
    d = precision.shape[0]
    return (
        sp.sympify(log_density_at_mode)
        + sp.Rational(d, 2) * sp.log(2 * sp.pi)
        - sp.Float(logdet) / 2
    )


def _target_variables(
    model: Model, targets: Sequence[str | Variable] | None
) -> tuple[Variable, ...]:
    if targets is None:
        return tuple(model.latent_variables)
    result: list[Variable] = []
    for target in targets:
        name = target.name if isinstance(target, Variable) else target
        try:
            result.append(model.variable_map[name])
        except KeyError as exc:
            raise KeyError(f"Unknown target variable {name!r}.") from exc
    return tuple(result)


def infer_laplace(
    model: Model,
    *,
    targets: Sequence[str | Variable] | None = None,
    backend: str | OptimizationBackend = "auto",
    initial_guess: Sequence[float] | Mapping[sp.Symbol | str, float] | None = None,
    assumptions: sp.Expr = sp.true,
    optimizer_options: Mapping[str, Any] | None = None,
    geometry: Any | None = None,
    corrector: HigherOrderCorrector | None = None,
    order: int = 2,
    exprtest_loader: Callable[[], Any] | None = None,
    semialg_loader: Callable[[], Any] | None = None,
    funcprops_loader: Callable[[], Any] | None = None,
) -> InferenceResult:
    """Approximate a model posterior and evidence by Laplace's method.

    Ordinary positive-definite modes produce a Gaussian posterior and the
    standard second-order determinant formula.  For ``order > 2`` a higher-order
    corrector refines the evidence.  A concrete :class:`AsymptoticCorrector`
    additionally handles even-order singular modes (zero Hessian curvature) for
    one-dimensional or coordinate-separable models by delegating the integral
    expansion to ``asymptotic.laplace_asymptotic_integral``.
    """
    if order < 2:
        raise ValueError("Laplace order must be at least 2.")
    target_vars = _target_variables(model, targets)
    if not target_vars:
        raise LaplaceApproximationError(
            "Laplace inference requires at least one latent target."
        )
    target_symbols = tuple(v.symbol for v in target_vars)
    target_set = set(target_symbols)
    unresolved_latents = tuple(
        v for v in model.latent_variables if v.symbol not in target_set
    )
    if unresolved_latents:
        names = ", ".join(v.name for v in unresolved_latents)
        raise LaplaceApproximationError(
            "Partial-target Laplace would leave latent variables unresolved: "
            + names
            + ". "
            "Marginalize them exactly first or include them among the Laplace targets."
        )
    log_density = sp.simplify(model.joint_log_density(substitute_observations=True))
    if geometry is None:
        geometry = analyze_posterior_geometry(
            log_density,
            target_symbols,
            domain=assumptions if assumptions is not None else True,
            funcprops_loader=funcprops_loader,
            semialg_loader=semialg_loader,
        )

    optimizer = get_optimization_backend(backend)
    try:
        optimum: OptimizationResult = optimizer.optimize(
            log_density,
            target_symbols,
            supports=tuple(model.effective_support(v) for v in target_vars),
            initial_guess=initial_guess,
            assumptions=assumptions,
            options=optimizer_options,
        )
    except OptimizationError as exc:
        raise LaplaceApproximationError(str(exc)) from exc

    # Retain an exact determinant certificate alongside the floating Hessian.
    point_subs = dict(zip(target_symbols, optimum.point, strict=True))
    exact_precision = -symbolic_hessian(log_density, target_symbols).subs(point_subs)
    precision_determinant = sp.simplify(exact_determinant(exact_precision))
    determinant_certificate = certify_zero(
        precision_determinant, exprtest_loader=exprtest_loader
    )
    definiteness_certificate = certify_positive_definite(
        exact_precision,
        assumptions=assumptions if assumptions is not None else True,
        semialg_loader=semialg_loader,
    )

    precision = precision_at(log_density, target_symbols, optimum.point)
    eigvals = np.linalg.eigvalsh(precision)
    finite_eigs = np.all(np.isfinite(eigvals))
    tolerance = 1e-10 * max(1.0, float(np.max(np.abs(eigvals))) if finite_eigs else 1.0)
    negative = (not finite_eigs) or np.min(eigvals) < -tolerance
    singular_numeric = bool(finite_eigs and np.min(eigvals) <= tolerance)
    # A proved zero determinant is stronger evidence of degeneracy than a
    # floating threshold; a proved nonzero determinant prevents tiny-but-real
    # curvature from being mislabeled singular solely by scale.
    singular = (
        True
        if determinant_certificate.is_zero is True
        else False
        if determinant_certificate.is_zero is False and not negative
        else singular_numeric
    )
    supports = tuple(v.support for v in target_vars)

    if negative:
        raise LaplaceApproximationError(
            "Negative Hessian at MAP is not positive definite: it has a negative/nonfinite "
            "eigenvalue, so the point is not a valid local maximum for Laplace inference."
        )

    asymptotic_analysis: AsymptoticLaplaceAnalysis | None = None
    correction = sp.S.Zero
    if singular:
        if not isinstance(corrector, AsymptoticCorrector):
            raise LaplaceApproximationError(
                "Negative Hessian at MAP is singular; use AsymptoticCorrector() for "
                "even-order singular Laplace inference."
            )
        try:
            asymptotic_analysis = corrector.analyze(
                log_density=log_density,
                variables=target_symbols,
                point=optimum.point,
                supports=supports,
                precision=precision,
                order=order,
                base_log_evidence=None,
            )
        except AsymptoticIntegrationError as exc:
            raise LaplaceApproximationError(str(exc)) from exc
        posterior: GaussianLaplacePosterior | SingularLaplacePosterior = (
            SingularLaplacePosterior(
                target_symbols,
                optimum.point,
                asymptotic_analysis.local_orders,
                asymptotic_analysis.local_coefficients,
            )
        )
        covariance = None
        log_z_2 = None
        log_evidence = asymptotic_analysis.log_evidence
        steps = [
            InferenceStep(
                method=f"singular-laplace-{optimizer.name}",
                description=(
                    "Optimized the observed joint log density and detected degenerate "
                    "curvature at the MAP."
                ),
                exact=False,
                metadata={
                    "map": optimum.point,
                    "optimizer_certified": optimum.certified,
                    "optimizer_attained": optimum.attained,
                    "optimizer_status": optimum.status,
                },
            ),
            InferenceStep(
                method=corrector.name,
                description=(
                    "Applied an even-order singular Laplace expansion using the optional "
                    "asymptotic integral engine."
                ),
                exact=False,
                metadata={
                    "status": asymptotic_analysis.status,
                    "certified": asymptotic_analysis.certified,
                    "local_orders": asymptotic_analysis.local_orders,
                },
            ),
        ]
    else:
        covariance = np.linalg.inv(precision)
        posterior = GaussianLaplacePosterior(
            target_symbols, optimum.point, covariance, precision
        )
        log_z_2 = laplace_log_evidence(optimum.value, precision)
        log_evidence = log_z_2
        steps = [
            InferenceStep(
                method=f"laplace-{optimizer.name}",
                description=(
                    f"Optimized the observed joint log density over {len(target_symbols)} parameter(s) "
                    "and constructed the local Gaussian from the negative Hessian."
                ),
                exact=False,
                metadata={
                    "map": optimum.point,
                    "optimizer_message": optimum.message,
                    "optimizer_certified": optimum.certified,
                    "optimizer_attained": optimum.attained,
                    "optimizer_status": optimum.status,
                },
            )
        ]
        if order > 2:
            if corrector is None:
                raise LaplaceApproximationError(
                    "order > 2 requires a higher-order asymptotic corrector."
                )
            if (
                isinstance(corrector, AsymptoticCorrector)
                and corrector._correction_function is None
            ):
                try:
                    asymptotic_analysis = corrector.analyze(
                        log_density=log_density,
                        variables=target_symbols,
                        point=optimum.point,
                        supports=supports,
                        precision=precision,
                        order=order,
                        base_log_evidence=log_z_2,
                    )
                except AsymptoticIntegrationError as exc:
                    raise LaplaceApproximationError(str(exc)) from exc
                correction = sp.sympify(asymptotic_analysis.correction)
                log_evidence = asymptotic_analysis.log_evidence
            else:
                correction = sp.sympify(
                    corrector.correction(
                        log_density=log_density,
                        variables=target_symbols,
                        point=optimum.point,
                        precision=precision,
                        order=order,
                    )
                )
                log_evidence = sp.simplify(log_z_2 + correction)
            steps.append(
                InferenceStep(
                    method=getattr(corrector, "name", type(corrector).__name__),
                    description=f"Applied an order-{order} higher-order log-evidence correction.",
                    exact=False,
                    metadata={
                        "correction": correction,
                        "order": order,
                        "status": getattr(asymptotic_analysis, "status", None),
                        "certified": getattr(asymptotic_analysis, "certified", False),
                    },
                )
            )

    metadata = {
        "log_density": log_density,
        "map_log_density": optimum.value,
        "precision": precision,
        "covariance": covariance,
        "base_log_evidence": log_z_2,
        "higher_order_correction": correction,
        "optimization_backend": optimizer.name,
        "optimization_certified": optimum.certified,
        "optimization_attained": optimum.attained,
        "optimization_status": optimum.status,
        "optimization_global_value": optimum.global_value,
        "unresolved_symbols": (),
        "singular_laplace": singular,
        "precision_determinant": precision_determinant,
        "precision_determinant_certificate": determinant_certificate,
        "precision_determinant_certified_nonzero": determinant_certificate.is_zero
        is False,
        "precision_determinant_certified_zero": determinant_certificate.is_zero is True,
        "precision_definiteness_certificate": definiteness_certificate,
        "precision_certified_positive_definite": definiteness_certificate.positive_definite
        is True,
        "posterior_geometry": geometry,
    }
    if asymptotic_analysis is not None:
        metadata.update(
            {
                "asymptotic_expression": asymptotic_analysis.expression,
                "asymptotic_parameter": asymptotic_analysis.parameter,
                "asymptotic_evaluation_parameter": asymptotic_analysis.evaluation_parameter,
                "asymptotic_status": asymptotic_analysis.status,
                "asymptotic_certified": asymptotic_analysis.certified,
                "asymptotic_remainder": asymptotic_analysis.remainder,
                "asymptotic_certificate": asymptotic_analysis.certificate,
                "local_orders": asymptotic_analysis.local_orders,
                "local_coefficients": asymptotic_analysis.local_coefficients,
            }
        )

    return InferenceResult(
        posterior=posterior,
        kind=InferenceKind.APPROXIMATE,
        log_evidence=sp.simplify(log_evidence),
        steps=tuple(steps),
        diagnostics={
            "map": optimum.point,
            "precision_eigenvalues": tuple(float(v) for v in eigvals),
            "singular_laplace": singular,
            "asymptotic_certified": bool(
                asymptotic_analysis and asymptotic_analysis.certified
            ),
            "precision_determinant": precision_determinant,
            "precision_determinant_certificate": determinant_certificate,
            "precision_determinant_certified_nonzero": determinant_certificate.is_zero
            is False,
            "precision_determinant_certified_zero": determinant_certificate.is_zero
            is True,
            "precision_definiteness_certificate": definiteness_certificate,
            "precision_certified_positive_definite": definiteness_certificate.positive_definite
            is True,
            "posterior_geometry": geometry,
        },
        metadata=metadata,
    )


def fit_precision_at_max(
    points: Sequence[Sequence[float]], values: Sequence[float]
) -> np.ndarray:
    """Fit a local quadratic precision matrix from sampled log-density/evidence values.

    The largest supplied value is treated as the local maximum and a symmetric quadratic
    ``delta_value = -1/2 delta.T @ P @ delta`` is fit by linear least squares.
    """
    x = np.asarray(points, dtype=float)
    y = np.asarray(values, dtype=float)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError("points must be (n, d) and values must be length n.")
    n, d = x.shape
    n_terms = d * (d + 1) // 2
    if n < n_terms + 2:
        raise ValueError(
            f"At least {n_terms + 2} samples are required to fit a {d}D precision matrix."
        )
    best = int(np.argmax(y))
    delta = x - x[best]
    dy = y - y[best]
    columns = []
    ij: list[tuple[int, int]] = []
    for i in range(d):
        for j in range(i, d):
            # -1/2 * P_ii dx_i^2; cross pair appears twice -> -P_ij dx_i dx_j.
            columns.append(
                -0.5 * delta[:, i] ** 2 if i == j else -delta[:, i] * delta[:, j]
            )
            ij.append((i, j))
    design = np.column_stack(columns)
    coeff, _, rank, _ = np.linalg.lstsq(design, dy, rcond=None)
    if rank < n_terms:
        raise LaplaceApproximationError(
            "Sample geometry is insufficient to identify precision."
        )
    precision = np.zeros((d, d), dtype=float)
    for value, (i, j) in zip(coeff, ij, strict=True):
        precision[i, j] = value
        precision[j, i] = value
    return precision
