"""Cost-aware Bayesian inference planning and dispatch."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import sympy as sp

from ..conjugacy import DEFAULT_REGISTRY, ConjugacyRegistry, infer_conjugate
from ..core import InferenceResult, InferenceStep, Model
from ..diagnostics import (
    PosteriorConversionError,
    assess_quality,
    to_posterior_data,
    with_quality,
)
from ..exact import ExactIntegrationError, infer_exact
from ..gp import GaussianProcessRegressor
from ..laplace import LaplaceApproximationError, OptimizationError, infer_laplace
from ..mcmc import (
    AdaptiveMetropolis,
    NoUTurnSampler,
    infer_mcmc_model,
    model_log_posterior_gradient,
)
from ..models import BayesianLinearRegression, BayesianMultivariateLinearRegression
from ..nested import infer_nested_model
from ..neural import JAXRegressionNetwork
from ..reasoning import analyze_posterior_geometry
from .assessment import (
    _all_continuous,
    _conjugacy_assessment,
    _model_complexity,
    _model_distribution_metadata,
)
from .types import (
    CandidateAssessment,
    InferencePlan,
    InferencePlanningError,
    PlannerConfig,
)


class InferencePlanner:
    """Inspect a model/problem and dispatch to the cheapest suitable inference engine."""

    def __init__(
        self,
        *,
        config: PlannerConfig | None = None,
        registry: ConjugacyRegistry = DEFAULT_REGISTRY,
        geometry_analyzer: Callable[..., Any] = analyze_posterior_geometry,
    ):
        self.config = config or PlannerConfig()
        self.registry = registry
        self.geometry_analyzer = geometry_analyzer

    def plan(self, subject: Any, **context: Any) -> InferencePlan:
        candidates: list[CandidateAssessment] = []

        # Analytic specialized objects are already mathematical eliminations of their
        # latent Gaussian structure, so they receive very low exact costs.
        if isinstance(subject, BayesianMultivariateLinearRegression):
            if "x" in context and "y" in context:
                try:
                    n = len(context["y"])
                    p = subject.design_dimension(context["x"])
                    m = sp.ImmutableMatrix(context["y"]).cols
                    cost = 0.6 + (n * p * p + p**3 + m**3) / 1_000_000.0
                except (
                    TypeError,
                    ValueError,
                    AttributeError,
                    RuntimeError,
                    sp.PolynomialError,
                ):
                    cost = 1.1
                candidates.append(
                    CandidateAssessment(
                        "analytic-multivariate-linear-regression",
                        True,
                        True,
                        cost,
                        "Matrix-Normal/Inverse-Wishart regression has a closed-form posterior, predictive law, and evidence.",
                    )
                )
            else:
                candidates.append(
                    CandidateAssessment(
                        "analytic-multivariate-linear-regression",
                        False,
                        True,
                        1.1,
                        "Multivariate regression dispatch requires x and y.",
                    )
                )
            return InferencePlan(tuple(candidates))

        if isinstance(subject, BayesianLinearRegression):
            if "x" in context and "y" in context:
                try:
                    n = len(context["y"])
                    p = subject.design_dimension(context["x"])
                    cost = 0.5 + (n * p * p + p**3) / 1_000_000.0
                except (
                    TypeError,
                    ValueError,
                    AttributeError,
                    RuntimeError,
                    sp.PolynomialError,
                ):
                    cost = 1.0
                candidates.append(
                    CandidateAssessment(
                        "analytic-linear-regression",
                        True,
                        True,
                        cost,
                        "Normal-Inverse-Gamma regression has a closed-form posterior and evidence.",
                    )
                )
            else:
                candidates.append(
                    CandidateAssessment(
                        "analytic-linear-regression",
                        False,
                        True,
                        1.0,
                        "Regression dispatch requires x and y.",
                    )
                )
            return InferencePlan(tuple(candidates))

        if isinstance(subject, GaussianProcessRegressor):
            if "x" in context and "y" in context:
                try:
                    n = len(context["y"])
                    cost = 0.75 + n**3 / 1_000_000.0
                except (
                    TypeError,
                    ValueError,
                    AttributeError,
                    RuntimeError,
                    sp.PolynomialError,
                ):
                    cost = 1.5
                candidates.append(
                    CandidateAssessment(
                        "analytic-gaussian-process",
                        True,
                        True,
                        cost,
                        "Fixed-hyperparameter Gaussian-process conditioning and evidence are analytic.",
                    )
                )
            else:
                candidates.append(
                    CandidateAssessment(
                        "analytic-gaussian-process",
                        False,
                        True,
                        1.5,
                        "Gaussian-process dispatch requires x and y.",
                    )
                )
            return InferencePlan(tuple(candidates))

        if isinstance(subject, JAXRegressionNetwork):
            if "x" in context and "y" in context:
                try:
                    n = len(context["y"])
                    parameter_scale = (
                        sum(subject.config.widths) + subject.config.input_dim
                    )
                    cost = 25.0 + n * max(parameter_scale, 1) / 10_000.0
                except (
                    TypeError,
                    ValueError,
                    AttributeError,
                    RuntimeError,
                    sp.PolynomialError,
                ):
                    cost = 30.0
                candidates.append(
                    CandidateAssessment(
                        "jax-mc-dropout-regression",
                        True,
                        False,
                        cost,
                        "JAX regression network supports stochastic dropout training and predictive uncertainty.",
                    )
                )
            else:
                candidates.append(
                    CandidateAssessment(
                        "jax-mc-dropout-regression",
                        False,
                        False,
                        30.0,
                        "Neural-network dispatch requires x and y.",
                    )
                )
            return InferencePlan(tuple(candidates))

        if not isinstance(subject, Model):
            return InferencePlan(
                (
                    CandidateAssessment(
                        "model-inference",
                        False,
                        False,
                        float("inf"),
                        f"Unsupported inference subject type {type(subject).__name__}.",
                    ),
                )
            )

        model = subject
        latent, operations = _model_complexity(model)
        exact_ok = (
            latent > 0
            and latent <= self.config.max_exact_latents
            and operations <= self.config.max_exact_operations
        )
        exact_cost = 5.0 + latent * 5.0 + operations / 25.0
        candidates.append(
            CandidateAssessment(
                "symbolic-exact",
                exact_ok,
                True,
                exact_cost,
                (
                    f"Symbolic integration is within planner limits ({latent} latent, {operations} operations)."
                    if exact_ok
                    else f"Symbolic integration exceeds planner limits or has no latent variables ({latent} latent, {operations} operations)."
                ),
                {
                    "latent_variables": latent,
                    "operations": operations,
                    "distribution_metadata": _model_distribution_metadata(model),
                },
            )
        )
        candidates.append(_conjugacy_assessment(context, self.registry))

        continuous = _all_continuous(model)
        geometry = None
        if continuous and operations <= 1000:
            try:
                geometry = self.geometry_analyzer(
                    model.joint_log_density(substitute_observations=True),
                    tuple(v.symbol for v in model.latent_variables),
                    domain=context.get("assumptions", True),
                )
            except (
                TypeError,
                ValueError,
                AttributeError,
                RuntimeError,
                sp.PolynomialError,
            ):
                geometry = None
        laplace_ok = self.config.allow_laplace and continuous
        laplace_cost = 50.0 + latent * 5.0 + operations / 100.0
        laplace_reason = (
            "All latent variables have continuous real support."
            if laplace_ok
            else "Laplace is disabled or at least one latent variable is non-continuous."
        )
        geometry_meta = {}
        if geometry is not None and geometry.source == "funcprops":
            geometry_meta = {
                "convexity": geometry.convexity,
                "globally_concave": geometry.globally_concave,
                "continuity": geometry.continuity,
                "has_singularities": geometry.has_singularities,
            }
            if geometry.laplace_favorable:
                laplace_cost = max(1.0, laplace_cost - 20.0)
                laplace_reason += " funcprops certified globally concave, nonsingular-compatible geometry."
            if geometry.has_singularities is True:
                laplace_cost += 80.0
                laplace_reason += (
                    " funcprops found singularities, increasing local-Laplace risk."
                )
            if geometry.continuity is False:
                laplace_cost += 100.0
                laplace_reason += (
                    " funcprops found discontinuity on the queried domain."
                )
        candidates.append(
            CandidateAssessment(
                "laplace",
                laplace_ok,
                False,
                laplace_cost,
                laplace_reason,
                geometry_meta,
            )
        )

        has_prior_geometry = context.get("prior_transform") is not None
        has_prior_draws = context.get("prior_sampler") is not None
        nested_ok = (
            self.config.allow_nested
            and (has_prior_geometry or has_prior_draws)
            and latent > 0
        )
        nested_reason = (
            "A prior_transform was supplied; unit-cube slice constrained sampling is available."
            if has_prior_geometry
            else "A prior_sampler was supplied for likelihood-constrained prior sampling."
            if has_prior_draws
            else "Nested sampling requires a prior_sampler or prior_transform and at least one latent variable."
        )
        nested_cost = 100.0 + 10.0 * latent
        if geometry is not None and geometry.has_singularities is True:
            nested_cost = max(1.0, nested_cost - 20.0)
            nested_reason += (
                " Structural singularities make global sampling relatively preferable."
            )
        candidates.append(
            CandidateAssessment(
                "nested-sampling",
                nested_ok,
                False,
                nested_cost,
                nested_reason,
            )
        )

        initial_positions = context.get("initial_positions")
        mcmc_ok = (
            self.config.allow_mcmc
            and continuous
            and latent > 0
            and initial_positions is not None
        )
        gradient_available = False
        if mcmc_ok:
            try:
                gradient_available = (
                    model_log_posterior_gradient(model, context.get("targets"))
                    is not None
                )
            except (
                TypeError,
                ValueError,
                AttributeError,
                RuntimeError,
                sp.PolynomialError,
            ):
                gradient_available = False
        mcmc_method = "NUTS" if gradient_available else "Adaptive Metropolis"
        mcmc_reason = (
            f"Continuous latent model with supplied initial_positions; {mcmc_method} is available."
            if mcmc_ok
            else "MCMC requires continuous latent variables and explicit initial_positions."
        )
        mcmc_cost = 75.0 + 8.0 * latent + operations / 200.0
        candidates.append(
            CandidateAssessment(
                "mcmc",
                mcmc_ok,
                False,
                mcmc_cost,
                mcmc_reason,
                {
                    "gradient_available": gradient_available,
                    "preferred_sampler": mcmc_method,
                },
            )
        )

        # Exactness is a policy preference, but cost still orders exact methods among
        # themselves.  Approximate methods follow after exact methods when enabled.
        if self.config.prefer_exact:
            candidates.sort(
                key=lambda c: (not c.applicable, not c.exact, c.estimated_cost)
            )
        else:
            candidates.sort(
                key=lambda c: (not c.applicable, c.estimated_cost, not c.exact)
            )
        return InferencePlan(tuple(candidates))

    def infer(self, subject: Any, **context: Any) -> InferenceResult:
        plan = self.plan(subject, **context)
        applicable = plan.applicable
        if not applicable:
            raise InferencePlanningError(plan.explain())
        failures: list[tuple[str, str]] = []
        for candidate in applicable:
            try:
                result = self._execute(candidate.method, subject, context)
            except (
                ExactIntegrationError,
                LaplaceApproximationError,
                OptimizationError,
                LookupError,
                ValueError,
                RuntimeError,
            ) as exc:
                failures.append((candidate.method, str(exc)))
                continue
            if self.config.assess_diagnostics:
                pdata = None
                if result.kind.value == "sampled":
                    try:
                        pdata = to_posterior_data(
                            result,
                            draws=int(context.get("diagnostic_draws", 500)),
                            chains=int(context.get("diagnostic_chains", 4)),
                            rng=context.get("diagnostic_rng", context.get("rng")),
                        )
                    except (PosteriorConversionError, ValueError):
                        pdata = None
                quality = assess_quality(result, posterior_data=pdata)
                result = with_quality(result, quality)
                if self.config.fallback_on_poor_quality and not quality.trustworthy:
                    failures.append(
                        (
                            candidate.method,
                            "diagnostic quality: " + "; ".join(quality.reasons),
                        )
                    )
                    continue
            planner_step = InferenceStep(
                "planner-dispatch",
                f"Selected {candidate.method} after applicability/cost analysis and diagnostic assessment.",
                result.is_exact,
                {
                    "selected": candidate.method,
                    "estimated_cost": candidate.estimated_cost,
                    "failed_before_selection": tuple(failures),
                    "quality": getattr(result.quality, "level", None),
                    "candidate_metadata": dict(candidate.metadata),
                },
            )
            return InferenceResult(
                posterior=result.posterior,
                kind=result.kind,
                log_evidence=result.log_evidence,
                steps=(planner_step, *result.steps),
                diagnostics=result.diagnostics,
                metadata={
                    **dict(result.metadata),
                    "planner": {
                        "selected": candidate.method,
                        "candidates": tuple(
                            {
                                "method": c.method,
                                "applicable": c.applicable,
                                "exact": c.exact,
                                "estimated_cost": c.estimated_cost,
                                "reason": c.reason,
                                "metadata": dict(c.metadata),
                            }
                            for c in plan.candidates
                        ),
                        "failures": tuple(failures),
                        "quality_level": getattr(result.quality, "level", None),
                        "trustworthy": getattr(result.quality, "trustworthy", None),
                    },
                },
            )
        detail = "; ".join(f"{method}: {message}" for method, message in failures)
        raise InferencePlanningError(
            f"All applicable inference routes failed. {detail}"
        )

    def _execute(
        self, method: str, subject: Any, context: Mapping[str, Any]
    ) -> InferenceResult:
        if method == "analytic-multivariate-linear-regression":
            return subject.infer(context["x"], context["y"])
        if method == "analytic-linear-regression":
            return subject.infer(context["x"], context["y"])
        if method == "analytic-gaussian-process":
            return subject.infer(context["x"], context["y"])
        if method == "jax-mc-dropout-regression":
            return subject.infer(
                context["x"],
                context["y"],
                state=context.get("network_state"),
                epochs=context.get("epochs", 200),
                learning_rate=context.get("learning_rate", 1e-3),
                batch_size=context.get("batch_size"),
                lambda_=context.get("network_regularization", 1e-4),
                p=context.get("network_regularization_p", 2.0),
                alpha=context.get("alpha", 0.0),
                sample_number=context.get("network_samples", 1),
                rng=context.get("rng", 0),
            )
        if method == "symbolic-exact":
            return infer_exact(
                subject,
                targets=context.get("targets"),
                fallback_to_conjugacy=False,
                assumptions=context.get("assumptions"),
            )
        if method == "conjugacy":
            return infer_conjugate(
                context["likelihood"],
                context["data"],
                context.get("prior"),
                registry=self.registry,
            )
        if method == "laplace":
            return infer_laplace(
                subject,
                targets=context.get("targets"),
                backend=context.get("laplace_backend", "auto"),
                initial_guess=context.get("initial_guess"),
                assumptions=context.get("assumptions", sp.true),
                optimizer_options=context.get("optimizer_options"),
                corrector=context.get("corrector"),
                order=context.get("laplace_order", 2),
            )
        if method == "mcmc":
            sampler = context.get("mcmc_sampler")
            if sampler is None:
                gradient = model_log_posterior_gradient(subject, context.get("targets"))
                sampler = (
                    NoUTurnSampler(gradient=gradient)
                    if gradient is not None
                    else AdaptiveMetropolis()
                )
            return infer_mcmc_model(
                subject,
                initial_positions=context["initial_positions"],
                sampler=sampler,
                variables=context.get("targets"),
                draws=context.get("mcmc_draws", 1000),
                warmup=context.get("mcmc_warmup", 500),
                thin=context.get("mcmc_thin", 1),
                rng=context.get("rng"),
            )
        if method == "nested-sampling":
            return infer_nested_model(
                subject,
                context.get("prior_sampler"),
                prior_transform=context.get("prior_transform"),
                variables=context.get("targets"),
                n_live=context.get("n_live", 100),
                max_iterations=context.get("max_iterations", 10_000),
                min_iterations=context.get("min_iterations", 100),
                termination_fraction=context.get("termination_fraction", 0.01),
                constrained_sampler=context.get("constrained_sampler"),
                rng=context.get("rng"),
            )
        raise InferencePlanningError(f"Unknown planner method {method!r}.")


def plan_inference(
    subject: Any,
    *,
    config: PlannerConfig | None = None,
    registry: ConjugacyRegistry = DEFAULT_REGISTRY,
    **context: Any,
) -> InferencePlan:
    """Return the ranked inference plan without executing it."""
    return InferencePlanner(config=config, registry=registry).plan(subject, **context)


def infer(
    subject: Any,
    *,
    config: PlannerConfig | None = None,
    registry: ConjugacyRegistry = DEFAULT_REGISTRY,
    **context: Any,
) -> InferenceResult:
    """Automatically choose and execute an inference method."""
    return InferencePlanner(config=config, registry=registry).infer(subject, **context)
