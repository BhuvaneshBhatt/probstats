"""Empirical-Bayes evidence optimization."""

from .core import (
    EvidenceEvaluation,
    EvidenceObjective,
    EvidenceOptimizationError,
    EvidenceOptimizationResult,
    Hyperparameter,
    LaplaceEvidenceObjective,
    MacKayFixedPointResult,
    mackay_fixed_point,
    mackay_laplace,
    mackay_precision_noise_update,
    mackay_precision_update,
    optimize_evidence,
    optimize_laplace_evidence,
)

__all__ = [
    "EvidenceEvaluation",
    "EvidenceObjective",
    "EvidenceOptimizationError",
    "EvidenceOptimizationResult",
    "Hyperparameter",
    "LaplaceEvidenceObjective",
    "MacKayFixedPointResult",
    "mackay_fixed_point",
    "mackay_laplace",
    "mackay_precision_noise_update",
    "mackay_precision_update",
    "optimize_evidence",
    "optimize_laplace_evidence",
]
