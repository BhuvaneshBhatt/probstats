"""Symbolic distribution algebra and transform-based reasoning."""

from ._symbolic_inversion import (
    TransformInversionDistribution,
    distribution_from_characteristic_function,
    distribution_from_mgf,
)
from ._symbolic_multivariate import (
    ProofStatus,
    TransformEquivalenceProof,
    multivariate_characteristic_function,
    multivariate_moment_generating_function,
    prove_equivalent_by_transform,
)
from ._symbolic_recognition import (
    DistributionContext,
    JointBlock,
    OrderStatisticDistribution,
    convolution,
    distribution_of,
    order_statistic,
    recognize_affine_matrix,
)
from ._symbolic_transforms import (
    central_moment_generating_function,
    characteristic_function,
    cumulant_generating_function,
    factorial_moment_generating_function,
    moment_generating_function,
    probability_generating_function,
)

__all__ = [
    "DistributionContext",
    "JointBlock",
    "OrderStatisticDistribution",
    "ProofStatus",
    "TransformEquivalenceProof",
    "TransformInversionDistribution",
    "central_moment_generating_function",
    "characteristic_function",
    "convolution",
    "cumulant_generating_function",
    "distribution_from_characteristic_function",
    "distribution_from_mgf",
    "distribution_of",
    "factorial_moment_generating_function",
    "moment_generating_function",
    "multivariate_characteristic_function",
    "multivariate_moment_generating_function",
    "order_statistic",
    "probability_generating_function",
    "prove_equivalent_by_transform",
    "recognize_affine_matrix",
]
