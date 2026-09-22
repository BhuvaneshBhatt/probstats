"""Semialgebraic hypothesis testing and unbiased polynomial kernels."""

from .bootstrap import SDLBootstrapResult, sdl_multiplier_bootstrap
from .constraints import (
    ConstraintAugmentation,
    augment_constraints,
    augment_hypothesis_constraints,
)
from .diagnostics import (
    SDLCalibrationResult,
    SDLDiagnostics,
    sdl_calibrate,
    sdl_diagnostics,
)
from .hypotheses import BasicSemialgebraicNull, SemialgebraicHypothesis
from .kernels import (
    Kernel,
    UnbiasedParameterEstimator,
    hypothesis_kernel,
    polynomial_constraint_kernel,
)
from .reducible import SDLUnionTestResult, sdl_union_test
from .sdl import SDLTestResult, sdl_test
from .statistics import (
    HajekProjectionEstimate,
    SDLStudentizationResult,
    hajek_projection_estimate,
    sdl_studentize,
)
from .trinomial import (
    TRINOMIAL_PAPER_CONFIGURATION,
    SDLReferenceConfiguration,
    multinomial_point_on_model,
    trinomial_model4_components,
    trinomial_reference_hypothesis,
    trinomial_sampler,
)

__all__ = [
    "TRINOMIAL_PAPER_CONFIGURATION",
    "BasicSemialgebraicNull",
    "ConstraintAugmentation",
    "HajekProjectionEstimate",
    "Kernel",
    "SDLBootstrapResult",
    "SDLCalibrationResult",
    "SDLDiagnostics",
    "SDLReferenceConfiguration",
    "SDLStudentizationResult",
    "SDLTestResult",
    "SDLUnionTestResult",
    "SemialgebraicHypothesis",
    "UnbiasedParameterEstimator",
    "augment_constraints",
    "augment_hypothesis_constraints",
    "hajek_projection_estimate",
    "hypothesis_kernel",
    "multinomial_point_on_model",
    "polynomial_constraint_kernel",
    "sdl_calibrate",
    "sdl_diagnostics",
    "sdl_multiplier_bootstrap",
    "sdl_studentize",
    "sdl_test",
    "sdl_union_test",
    "trinomial_model4_components",
    "trinomial_reference_hypothesis",
    "trinomial_sampler",
]
