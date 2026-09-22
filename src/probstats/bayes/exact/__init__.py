"""Exact symbolic Bayesian inference."""

from .backends import (
    ExactBackendFailure,
    ExactBackendUnavailableError,
    ExactIntegrationBackend,
    ExactIntegrationTrace,
    MultipleIntegrateBackend,
    SymPyExactIntegrationBackend,
)
from .inference import evidence, infer_exact, marginalize, normalize_posterior
from .routing import AutoExactIntegrationBackend, get_exact_integration_backend
from .symbolic import ExactIntegrationError, SymbolicJointDistribution, integrate_over

__all__ = [
    "AutoExactIntegrationBackend",
    "ExactBackendFailure",
    "ExactBackendUnavailableError",
    "ExactIntegrationBackend",
    "ExactIntegrationError",
    "ExactIntegrationTrace",
    "MultipleIntegrateBackend",
    "SymPyExactIntegrationBackend",
    "SymbolicJointDistribution",
    "evidence",
    "get_exact_integration_backend",
    "infer_exact",
    "integrate_over",
    "marginalize",
    "normalize_posterior",
]
