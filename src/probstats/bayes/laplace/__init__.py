"""Laplace approximation and optimization backends."""

from .asymptotic import (
    AsymptoticCorrector,
    AsymptoticIntegrationError,
    AsymptoticLaplaceAnalysis,
)
from .backends import (
    OptimizationBackend,
    OptimizationError,
    OptimizationResult,
    SciPyOptimizationBackend,
    SymboptOptimizationBackend,
    SymPyOptimizationBackend,
    get_optimization_backend,
)
from .core import (
    GaussianLaplacePosterior,
    HigherOrderCorrector,
    LaplaceApproximationError,
    SingularLaplacePosterior,
    fit_precision_at_max,
    infer_laplace,
    laplace_log_evidence,
    precision_at,
    symbolic_hessian,
)

__all__ = [
    "AsymptoticCorrector",
    "AsymptoticIntegrationError",
    "AsymptoticLaplaceAnalysis",
    "GaussianLaplacePosterior",
    "HigherOrderCorrector",
    "LaplaceApproximationError",
    "OptimizationBackend",
    "OptimizationError",
    "OptimizationResult",
    "SciPyOptimizationBackend",
    "SingularLaplacePosterior",
    "SymPyOptimizationBackend",
    "SymboptOptimizationBackend",
    "fit_precision_at_max",
    "get_optimization_backend",
    "infer_laplace",
    "laplace_log_evidence",
    "precision_at",
    "symbolic_hessian",
]
