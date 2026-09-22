"""Generic exponential-family conjugate-prior algebra."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import sympy as sp

from ..distributions import Exponential, ExponentialFamily, Normal


@dataclass(frozen=True, slots=True)
class NaturalConjugatePrior:
    """Natural conjugate prior ``p(eta|chi,nu) ∝ exp(eta·chi - nu A(eta))``."""

    family: ExponentialFamily
    chi: tuple[sp.Expr, ...]
    nu: sp.Expr

    def __init__(self, family: ExponentialFamily, chi, nu):
        chi = tuple(map(sp.sympify, chi))
        nu = sp.sympify(nu)
        if len(chi) != family.natural_parameters_count:
            raise ValueError("chi dimension must match natural parameter dimension")
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "chi", chi)
        object.__setattr__(self, "nu", nu)

    def kernel(self, eta=None):
        eta = tuple(
            self.family.natural_parameters if eta is None else map(sp.sympify, eta)
        )
        if len(eta) != len(self.chi):
            raise ValueError("eta dimension mismatch")
        return sp.exp(
            sum(e * c for e, c in zip(eta, self.chi))
            - self.nu * self.family.log_partition_from_natural(eta)
        )

    @property
    def normalizer(self):
        """Return a closed-form multiplicative normalizer when available."""
        if isinstance(self.family, Exponential):
            (c,) = self.chi
            return sp.simplify(c ** (self.nu + 1) / sp.gamma(self.nu + 1))
        if isinstance(self.family, Normal):
            c1, c2 = self.chi
            n = self.nu
            # Closed form for the Normal natural-parameter conjugate family.
            return sp.simplify(
                2 ** (sp.Rational(3, 2) - n / 2)
                * sp.sqrt(n)
                * (n / (-(c1**2) + n * c2)) ** (1 - n / 2)
                / (sp.sqrt(sp.pi) * sp.gamma(-1 + n / 2))
            )
        return None

    @property
    def normalizer_conditions(self):
        if isinstance(self.family, Exponential):
            return sp.And(self.nu > 0, self.chi[0] > 0)
        if isinstance(self.family, Normal):
            return sp.And(self.nu > 2, sp.Q.real(self.chi[0]), sp.Q.real(self.chi[1]))
        return self.nu > 0

    def pdf(self, eta=None):
        z = self.normalizer
        if z is None:
            raise NotImplementedError(
                f"No closed-form conjugate normalizer registered for {type(self.family).__name__}"
            )
        return sp.simplify(z * self.kernel(eta))

    def update(self, data: Iterable[Any]):
        values = tuple(data)
        sums = [sp.S.Zero] * len(self.chi)
        for value in values:
            stats = self.family.sufficient_statistics(value)
            sums = [sp.simplify(a + b) for a, b in zip(sums, stats)]
        return NaturalConjugatePrior(
            self.family,
            tuple(sp.simplify(c + s) for c, s in zip(self.chi, sums)),
            sp.simplify(self.nu + len(values)),
        )

    def predictive_pdf(self, value: Any):
        """Exact one-step predictive from a known conjugate normalizer ratio."""
        old = self.normalizer
        updated = self.update([value]).normalizer
        if old is None or updated is None:
            raise NotImplementedError("Predictive normalizer ratio unavailable")
        return sp.simplify(self.family.base_measure(value) * old / updated)


def update_natural_conjugate(data: Iterable[Any], prior: NaturalConjugatePrior):
    """Update a natural conjugate prior from IID observations.

    The posterior hyperparameters are ``chi + sum T(x_i)`` and ``nu + n``.
    When both normalizers are available, the exact marginal likelihood is
    computed from their ratio and the product of base measures.
    """
    values = tuple(data)
    posterior = prior.update(values)
    old = prior.normalizer
    new = posterior.normalizer
    log_evidence = None
    if old is not None and new is not None:
        log_evidence = sp.simplify(
            sum(sp.log(prior.family.base_measure(x)) for x in values)
            + sp.log(old)
            - sp.log(new)
        )
    prior_predictive = None
    posterior_predictive = None
    return {
        "prior": prior,
        "posterior": posterior,
        "log_evidence": log_evidence,
        "prior_predictive": prior_predictive,
        "posterior_predictive": posterior_predictive,
        "n_observations": len(values),
    }
