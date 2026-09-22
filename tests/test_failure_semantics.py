"""Failure categories must remain observably distinct to callers."""

from __future__ import annotations

import pytest
import sympy as sp

from probstats import MultivariateNormal, Normal
from probstats._integration import StructuredIntegrationUnavailable
from probstats.bayes.certification.core import CertificationStatus, certify_zero
from probstats.bayes.core.variable import Variable
from probstats.bayes.exact.backends import (
    ExactBackendFailure,
    ExactBackendUnavailableError,
    MultipleIntegrateBackend,
    SymPyExactIntegrationBackend,
)
from probstats.functionals import ProbabilityFunctionalError, cdf, quantile


def test_unsupported_operation_is_not_reported_as_backend_failure():
    dist = MultivariateNormal([0, 0], [[1, 0], [0, 1]])
    with pytest.raises(ProbabilityFunctionalError, match="scalar"):
        cdf(dist, 0)


def test_backend_unavailable_is_distinct_from_backend_evaluation_failure():
    backend = MultipleIntegrateBackend(
        loader=lambda: (_ for _ in ()).throw(
            StructuredIntegrationUnavailable("missing")
        )
    )
    with pytest.raises(ExactBackendUnavailableError):
        backend._load()
    x = sp.Symbol("x", integer=True)
    bad = Variable("x", sp.ImageSet(sp.Lambda(x, 2 * x), sp.S.Integers))
    with pytest.raises(ExactBackendFailure):
        SymPyExactIntegrationBackend().integrate(sp.exp(-(bad.symbol**2)), [bad])
    assert issubclass(ExactBackendUnavailableError, ImportError)
    assert not issubclass(ExactBackendFailure, ImportError)


def test_mathematically_invalid_input_raises_value_error_not_unknown():
    with pytest.raises(ValueError):
        Normal(0, -1)
    with pytest.raises(ValueError):
        quantile(Normal(0, 1), 2)


def test_unknown_certification_is_a_result_state_not_an_exception():
    result = certify_zero(sp.Symbol("x", real=True), use_exprtest=False)
    assert result.status is CertificationStatus.UNKNOWN
    assert result.proven is False
