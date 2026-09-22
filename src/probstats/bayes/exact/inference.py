"""Exact posterior normalization, marginalization, and evidence calculation."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import sympy as sp

from ..conjugacy import DEFAULT_REGISTRY, ConjugacyRegistry, infer_conjugate
from ..core import InferenceKind, InferenceResult, InferenceStep, Model, Variable
from .backends import ExactIntegrationBackend
from .symbolic import (
    ExactIntegrationError,
    SymbolicJointDistribution,
    _integrate_over_with_trace,
    _validate_normalizer,
    integrate_over,
)


def evidence(
    model: Model,
    *,
    backend: str | ExactIntegrationBackend = "auto",
    assumptions: Any | None = None,
) -> sp.Expr:
    """Return the exact marginal likelihood of a model's observations."""
    unnormalized = model.joint_density(substitute_observations=True)
    latent = model.latent_variables
    if not latent:
        return sp.simplify(unnormalized)
    value = integrate_over(
        unnormalized, latent, backend=backend, assumptions=assumptions
    )
    _validate_normalizer(value, assumptions=assumptions)
    return sp.simplify(value)


def normalize_posterior(
    model: Model,
    *,
    targets: Iterable[str | Variable] | None = None,
    backend: str | ExactIntegrationBackend = "auto",
    assumptions: Any | None = None,
) -> tuple[SymbolicJointDistribution, sp.Expr]:
    """Normalize the observed joint density and optionally eliminate nuisance variables."""
    latent = model.latent_variables
    if not latent:
        raise ValueError("Posterior inference requires at least one latent variable.")
    joint = model.joint_density(substitute_observations=True)
    z = integrate_over(joint, latent, backend=backend, assumptions=assumptions)
    _validate_normalizer(z, assumptions=assumptions)
    normalized = sp.simplify(joint / z)
    posterior = SymbolicJointDistribution(latent, normalized, z)
    if targets is not None:
        names = tuple(t.name if isinstance(t, Variable) else t for t in targets)
        posterior = posterior.marginal(*names, backend=backend, assumptions=assumptions)
    return posterior, z


def marginalize(
    model: Model,
    variables: Iterable[str | Variable],
    *,
    backend: str | ExactIntegrationBackend = "auto",
    assumptions: Any | None = None,
) -> sp.Expr:
    """Eliminate selected latent variables from the observed joint density."""
    names = {v.name if isinstance(v, Variable) else v for v in variables}
    by_name = model.variable_map
    unknown = names - set(by_name)
    if unknown:
        raise KeyError(f"Unknown variables to marginalize: {sorted(unknown)}.")
    selected = tuple(by_name[name] for name in names)
    observed = {v.name for v in model.observed_variables}
    if names & observed:
        raise ValueError(
            "Observed variables have already been substituted and cannot be marginalized."
        )
    return integrate_over(
        model.joint_density(substitute_observations=True),
        selected,
        backend=backend,
        assumptions=assumptions,
    )


def infer_exact(
    model: Model | None = None,
    *,
    targets: Iterable[str | Variable] | None = None,
    likelihood: Any | None = None,
    data: Iterable[Any] | None = None,
    prior: Any | None = None,
    registry: ConjugacyRegistry = DEFAULT_REGISTRY,
    fallback_to_conjugacy: bool = True,
    backend: str | ExactIntegrationBackend = "auto",
    assumptions: Any | None = None,
    exprtest_loader: Callable[[], Any] | None = None,
    semialg_loader: Callable[[], Any] | None = None,
) -> InferenceResult:
    """Perform exact inference with structured integration and conjugacy fallback.

    Direct normalization uses ``backend``.  The default ``"auto"`` route tries
    ``multiple-integrate`` for continuous structured multiple integrals and
    native SymPy otherwise.  If direct exact integration fails and a
    likelihood/data/prior conjugate context was supplied, registered conjugacy is used.
    """
    if model is not None:
        try:
            latent = model.latent_variables
            if not latent:
                raise ValueError(
                    "Posterior inference requires at least one latent variable."
                )
            joint = model.joint_density(substitute_observations=True)
            z, trace = _integrate_over_with_trace(
                joint, latent, backend=backend, assumptions=assumptions
            )
            normalizer_validation = _validate_normalizer(
                z,
                assumptions=assumptions,
                exprtest_loader=exprtest_loader,
                semialg_loader=semialg_loader,
            )
            normalizer_certificate = normalizer_validation.zero_certificate
            positive_certificate = normalizer_validation.positive_certificate
            posterior = SymbolicJointDistribution(latent, sp.simplify(joint / z), z)
            if targets is not None:
                names = tuple(t.name if isinstance(t, Variable) else t for t in targets)
                posterior = posterior.marginal(
                    *names, backend=backend, assumptions=assumptions
                )
            return InferenceResult(
                posterior=posterior,
                kind=InferenceKind.EXACT,
                log_evidence=sp.simplify(sp.log(z)),
                steps=(
                    InferenceStep(
                        "symbolic-normalization",
                        f"Integrated and normalized the joint density over {len(latent)} latent variable(s) using {trace.backend}.",
                        True,
                        {
                            "evidence": z,
                            "engine": trace.backend,
                            "attempts": trace.attempts,
                            "normalizer_certificate": normalizer_certificate,
                            "positive_certificate": positive_certificate,
                        },
                    ),
                ),
                metadata={
                    "evidence": z,
                    "engine": trace.backend,
                    "integration_attempts": trace.attempts,
                    "normalizer_certificate": normalizer_certificate,
                    "normalizer_certified_nonzero": normalizer_certificate.status.value
                    == "nonzero"
                    and normalizer_certificate.proven,
                    "positive_certificate": positive_certificate,
                    "normalizer_certified_positive": positive_certificate.value is True,
                },
            )
        except ExactIntegrationError as exc:
            direct_error = exc
    else:
        direct_error = ExactIntegrationError(
            "No Model was supplied for direct symbolic integration."
        )

    if fallback_to_conjugacy and likelihood is not None and data is not None:
        try:
            result = infer_conjugate(likelihood, data, prior, registry=registry)
        except (LookupError, ValueError) as fallback_error:
            raise ExactIntegrationError(
                f"Direct exact inference failed ({direct_error}); conjugate fallback also failed ({fallback_error})."
            ) from fallback_error
        step = InferenceStep(
            "conjugacy-fallback",
            f"Direct symbolic integration was unavailable; {result.steps[0].method} supplied an exact posterior.",
            True,
            {"direct_error": str(direct_error)},
        )
        return InferenceResult(
            posterior=result.posterior,
            kind=result.kind,
            log_evidence=result.log_evidence,
            steps=(step, *result.steps),
            diagnostics=result.diagnostics,
            metadata={
                **dict(result.metadata),
                "fallback_from": "symbolic-normalization",
            },
        )

    raise direct_error
