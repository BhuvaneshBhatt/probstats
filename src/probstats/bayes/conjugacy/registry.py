"""Extensible conjugacy registry driven by :mod:`probstats` signatures."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from ...algebra import conjugacy_signature, distribution_metadata
from ..core import InferenceKind, InferenceResult, InferenceStep
from ..distributions import (
    Bernoulli,
    Beta,
    Binomial,
    Categorical,
    Dirichlet,
    ExponentialFamily,
    Gamma,
    Multinomial,
    MultivariateNormal,
    Normal,
    Poisson,
)
from .base import NaturalConjugatePrior, update_natural_conjugate
from .multivariate import NormalInverseWishart
from .normal import NormalInverseGamma, default_normal_prior, update_normal
from .standard import (
    update_beta_bernoulli,
    update_beta_binomial,
    update_dirichlet_categorical,
    update_dirichlet_multinomial,
    update_gamma_poisson,
    update_multivariate_normal_niw,
)


@dataclass(frozen=True, slots=True)
class ConjugacyRule:
    likelihood_type: type
    prior_type: type
    updater: Callable[..., Any]
    name: str
    passes_likelihood: bool = False
    likelihood_family: str | None = None
    prior_family: str | None = None

    @property
    def signature_key(self) -> tuple[str, str]:
        return (
            self.likelihood_family or self.likelihood_type.__name__,
            self.prior_family or self.prior_type.__name__,
        )


class ConjugacyRegistry:
    """Resolve conjugacy using stable probstats family signatures.

    Class-based matching is also available for custom
    subclasses and the generic natural-conjugate exponential-family rule.
    """

    def __init__(self):
        self._rules: list[ConjugacyRule] = []

    def register(self, rule: ConjugacyRule):
        key = rule.signature_key
        self._rules = [r for r in self._rules if r.signature_key != key]
        self._rules.append(rule)

    def resolve(self, likelihood, prior):
        signature = conjugacy_signature(likelihood)
        prior_family = type(prior).__name__
        desired = (signature.likelihood_family, prior_family)
        for rule in reversed(self._rules):
            if rule.signature_key == desired:
                return rule
        # Generic natural-conjugate rules describe a protocol,
        # not one named prior family, so retain the structural fallback.
        for rule in reversed(self._rules):
            if isinstance(likelihood, rule.likelihood_type) and isinstance(
                prior, rule.prior_type
            ):
                if isinstance(prior, NaturalConjugatePrior) and type(
                    prior.family
                ) is not type(likelihood):
                    continue
                return rule
        return None

    def signature_for(self, likelihood):
        return conjugacy_signature(likelihood)

    @property
    def rules(self):
        return tuple(self._rules)


DEFAULT_REGISTRY = ConjugacyRegistry()
DEFAULT_REGISTRY.register(
    ConjugacyRule(
        ExponentialFamily,
        NaturalConjugatePrior,
        update_natural_conjugate,
        "natural-exponential-family",
    )
)
DEFAULT_REGISTRY.register(
    ConjugacyRule(
        Normal, NormalInverseGamma, update_normal, "normal-normal-inverse-gamma"
    )
)
DEFAULT_REGISTRY.register(
    ConjugacyRule(Bernoulli, Beta, update_beta_bernoulli, "beta-bernoulli", True)
)
DEFAULT_REGISTRY.register(
    ConjugacyRule(Binomial, Beta, update_beta_binomial, "beta-binomial", True)
)
DEFAULT_REGISTRY.register(
    ConjugacyRule(Poisson, Gamma, update_gamma_poisson, "gamma-poisson", True)
)
DEFAULT_REGISTRY.register(
    ConjugacyRule(
        Categorical,
        Dirichlet,
        update_dirichlet_categorical,
        "dirichlet-categorical",
        True,
    )
)
DEFAULT_REGISTRY.register(
    ConjugacyRule(
        Multinomial,
        Dirichlet,
        update_dirichlet_multinomial,
        "dirichlet-multinomial",
        True,
    )
)
DEFAULT_REGISTRY.register(
    ConjugacyRule(
        MultivariateNormal,
        NormalInverseWishart,
        update_multivariate_normal_niw,
        "multivariate-normal-normal-inverse-wishart",
        True,
    )
)


def conjugate_update(
    likelihood, data: Iterable, prior, *, registry: ConjugacyRegistry = DEFAULT_REGISTRY
):
    """Apply a conjugate Bayesian update and return its structured result."""
    rule = registry.resolve(likelihood, prior)
    if rule is None:
        sig = conjugacy_signature(likelihood)
        raise LookupError(
            f"No conjugate update registered for {sig.likelihood_family} with prior {type(prior).__name__}."
        )
    return (
        rule.updater(likelihood, data, prior)
        if rule.passes_likelihood
        else rule.updater(data, prior)
    )


def infer_conjugate(
    likelihood,
    data: Iterable,
    prior=None,
    *,
    registry: ConjugacyRegistry = DEFAULT_REGISTRY,
):
    """Infer a conjugate posterior using the supplied or default conjugacy registry."""
    if prior is None and isinstance(likelihood, Normal):
        prior = default_normal_prior()
    if prior is None:
        sig = conjugacy_signature(likelihood)
        raise LookupError(
            f"No default conjugate prior is defined for {sig.likelihood_family}."
        )
    rule = registry.resolve(likelihood, prior)
    if rule is None:
        raise LookupError(
            f"No conjugate update registered for {type(likelihood).__name__} with {type(prior).__name__}."
        )
    updated = (
        rule.updater(likelihood, data, prior)
        if rule.passes_likelihood
        else rule.updater(data, prior)
    )
    if isinstance(updated, dict):
        posterior = updated["posterior"]
        log_evidence = updated.get("log_evidence")
        metadata = {
            k: v for k, v in updated.items() if k not in {"posterior", "log_evidence"}
        }
    else:
        posterior = updated.posterior
        log_evidence = updated.log_evidence
        metadata = {
            "prior": updated.prior,
            "prior_predictive": updated.prior_predictive,
            "posterior_predictive": updated.posterior_predictive,
            "n_observations": updated.n_observations,
            "sufficient_statistics": updated.sufficient_statistics,
        }
    sig = conjugacy_signature(likelihood)
    metadata = {
        **metadata,
        "conjugacy_signature": sig,
        "distribution_metadata": distribution_metadata(likelihood),
    }
    return InferenceResult(
        posterior=posterior,
        kind=InferenceKind.EXACT,
        log_evidence=log_evidence,
        steps=(
            InferenceStep(
                rule.name,
                f"Applied exact conjugate update to {metadata['n_observations']} observations.",
                True,
            ),
        ),
        metadata=metadata,
    )
