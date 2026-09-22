"""Conjugate-prior rules and exact posterior updates."""

from .base import NaturalConjugatePrior, update_natural_conjugate
from .multivariate import (
    NormalInverseWishart,
    default_multivariate_normal_prior,
    update_multivariate_normal,
)
from .normal import (
    ConjugateUpdateResult,
    NormalInverseGamma,
    default_normal_prior,
    update_normal,
)
from .registry import (
    DEFAULT_REGISTRY,
    ConjugacyRegistry,
    ConjugacyRule,
    conjugate_update,
    infer_conjugate,
)
from .standard import (
    update_beta_bernoulli,
    update_beta_binomial,
    update_dirichlet_categorical,
    update_dirichlet_multinomial,
    update_gamma_poisson,
    update_multivariate_normal_niw,
)

__all__ = [
    "DEFAULT_REGISTRY",
    "ConjugacyRegistry",
    "ConjugacyRule",
    "ConjugateUpdateResult",
    "NaturalConjugatePrior",
    "NormalInverseGamma",
    "NormalInverseWishart",
    "conjugate_update",
    "default_multivariate_normal_prior",
    "default_normal_prior",
    "infer_conjugate",
    "update_beta_bernoulli",
    "update_beta_binomial",
    "update_dirichlet_categorical",
    "update_dirichlet_multinomial",
    "update_gamma_poisson",
    "update_multivariate_normal",
    "update_multivariate_normal_niw",
    "update_natural_conjugate",
    "update_normal",
]
