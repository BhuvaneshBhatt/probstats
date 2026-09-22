"""Applicability and cost assessment for Bayesian inference routes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import sympy as sp

from ...algebra import conjugacy_signature, distribution_metadata
from ...spaces import MeasureType
from ..conjugacy import ConjugacyRegistry, default_normal_prior
from ..core import Model
from ..distributions import Normal
from .types import CandidateAssessment


def _model_complexity(model: Model) -> tuple[int, int]:
    latent = len(model.latent_variables)
    try:
        operations = int(
            sp.count_ops(model.joint_log_density(substitute_observations=True))
        )
    except (TypeError, ValueError, AttributeError, RuntimeError, sp.PolynomialError):
        operations = 10_000
    return latent, operations


def _factor_metadata(distribution):
    try:
        return distribution_metadata(distribution)
    except (TypeError, ValueError, AttributeError, RuntimeError):
        return None


def _model_distribution_metadata(model: Model) -> tuple[Any, ...]:
    items = []
    latent_symbols = {v.symbol for v in model.latent_variables}
    for factor in model.factors:
        if factor.target.symbol in latent_symbols and factor.distribution is not None:
            metadata = _factor_metadata(factor.distribution)
            if metadata is not None:
                items.append(metadata)
    return tuple(items)


def _all_continuous(model: Model) -> bool:
    metadata = _model_distribution_metadata(model)
    if metadata and len(metadata) == len(model.latent_variables):
        return all(item.measure_type is MeasureType.CONTINUOUS for item in metadata)
    for variable in model.latent_variables:
        try:
            subset = variable.support.is_subset(sp.S.Reals)
        except (
            TypeError,
            ValueError,
            AttributeError,
            RuntimeError,
            sp.PolynomialError,
        ):
            subset = None
        if subset is not True:
            return False
    return bool(model.latent_variables)


def _conjugacy_assessment(
    context: Mapping[str, Any], registry: ConjugacyRegistry
) -> CandidateAssessment:
    likelihood = context.get("likelihood")
    data = context.get("data")
    prior = context.get("prior")
    if likelihood is None or data is None:
        return CandidateAssessment(
            "conjugacy",
            False,
            True,
            2.0,
            "No likelihood/data conjugacy context was supplied.",
        )
    resolved_prior = prior
    if resolved_prior is None and isinstance(likelihood, Normal):
        resolved_prior = default_normal_prior()
    if resolved_prior is None:
        return CandidateAssessment(
            "conjugacy",
            False,
            True,
            2.0,
            "This likelihood has no default conjugate prior.",
        )
    signature = conjugacy_signature(likelihood)
    metadata = distribution_metadata(likelihood)
    rule = registry.resolve(likelihood, resolved_prior)
    if rule is None:
        return CandidateAssessment(
            "conjugacy",
            False,
            True,
            2.0,
            f"No registered rule matches probstats signature {signature.likelihood_family}/{type(resolved_prior).__name__}.",
            {"conjugacy_signature": signature, "distribution_metadata": metadata},
        )
    try:
        n = len(data)
    except TypeError:
        n = 1
    return CandidateAssessment(
        "conjugacy",
        True,
        True,
        1.0 + min(float(n), 1000.0) / 1000.0,
        f"Registered rule {rule.name!r} provides a closed-form update.",
        {
            "rule": rule.name,
            "conjugacy_signature": signature,
            "distribution_metadata": metadata,
        },
    )
