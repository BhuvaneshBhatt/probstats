"""End-to-end SDL test for basic semialgebraic hypotheses."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ._validation import (
    generator_from_seed_or_rng,
    positive_integer,
    spawn_named_generators,
)
from .bootstrap import SDLBootstrapResult, sdl_multiplier_bootstrap
from .constraints import ConstraintAugmentation, augment_hypothesis_constraints
from .hypotheses import SemialgebraicHypothesis
from .kernels import Kernel, hypothesis_kernel
from .statistics import SDLStudentizationResult, sdl_studentize


@dataclass(frozen=True, slots=True)
class SDLTestResult:
    """Complete result of an SDL test for one basic semialgebraic null."""

    statistic: float
    p_value: float
    hypothesis: SemialgebraicHypothesis
    kernel: Kernel
    studentization: SDLStudentizationResult
    bootstrap: SDLBootstrapResult
    sample_size: int
    kernel_order: int
    budget: int
    projection_size: int
    bootstrap_replicates: int
    constraint_count: int
    seed: int | None
    augmentation: ConstraintAugmentation | None = None
    symmetrization: str = "exact"
    symmetrization_permutations: int | None = None

    @property
    def coordinate_statistics(self) -> np.ndarray:
        """Studentized statistics for the individual polynomial constraints."""
        return self.studentization.coordinate_statistics

    @property
    def standard_error(self) -> np.ndarray:
        """Estimated coordinate standard deviations used for studentization."""
        return self.studentization.standard_error

    @property
    def realized_subsets(self) -> int:
        """Number of Bernoulli-selected subsets in the incomplete U-statistic."""
        return len(self.studentization.u_statistic.subset_indices or ())


def sdl_test(
    sample: Sequence[object],
    hypothesis: SemialgebraicHypothesis,
    *,
    budget: int,
    bootstrap_replicates: int = 1000,
    kernel_order: int | None = None,
    n1: int | None = None,
    projection_indices: Sequence[int] | None = None,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    shuffle_projection_blocks: bool = False,
    bootstrap_batch_size: int | None = None,
    retain_bootstrap_components: bool = False,
    augment_constraints_count: int = 0,
    augmentation_concentration: float = 1.0,
    kernel_symmetrization: str = "exact",
    symmetrization_permutations: int | None = None,
) -> SDLTestResult:
    """Run the core SDL procedure for one basic semialgebraic null.

    The function composes the semialgebraic testing primitives: construct the
    unbiased symmetric polynomial kernel, realize the
    Bernoulli incomplete U-statistic, estimate the Hájek and kernel variance
    terms, studentize the maximum statistic, and obtain its conditional
    Gaussian-multiplier bootstrap p-value.

    ``seed`` provides reproducibility for the whole procedure. Alternatively,
    pass a NumPy ``Generator`` in ``rng`` as the entropy source. Independent
    child streams isolate augmentation, symmetrization, subset selection,
    projection shuffling, and bootstrap randomness. The arguments are mutually
    exclusive. A finite-union null is rejected
    here; use ``sdl_union_test`` for componentwise intersection-union testing.
    """
    observations = tuple(sample)
    if not observations:
        raise ValueError("sample must contain at least one observation")
    if not isinstance(hypothesis, SemialgebraicHypothesis):
        raise TypeError("hypothesis must be a SemialgebraicHypothesis")
    # Accessing basic_null gives one stable error boundary for finite unions.
    component = hypothesis.basic_null
    N = positive_integer(budget, name="budget")
    A = positive_integer(bootstrap_replicates, name="bootstrap_replicates")
    generator, seed = generator_from_seed_or_rng(seed, rng)
    streams, _ = spawn_named_generators(
        generator,
        ("augmentation", "symmetrization", "subsets", "projection", "bootstrap"),
    )
    augmentation_rng = streams["augmentation"]
    symmetrization_rng = streams["symmetrization"]
    subset_rng = streams["subsets"]
    projection_rng = streams["projection"]
    bootstrap_rng = streams["bootstrap"]

    augmentation = None
    if augment_constraints_count:
        hypothesis, augmentation = augment_hypothesis_constraints(
            hypothesis,
            count=augment_constraints_count,
            concentration=augmentation_concentration,
            rng=augmentation_rng,
        )
        component = hypothesis.basic_null
    kernel = hypothesis_kernel(
        hypothesis,
        order=kernel_order,
        symmetrization=kernel_symmetrization,
        permutation_count=symmetrization_permutations,
        rng=symmetrization_rng,
    )
    studentization = sdl_studentize(
        observations,
        kernel,
        budget=N,
        n1=n1,
        projection_indices=projection_indices,
        rng=subset_rng,
        projection_rng=projection_rng,
        shuffle_projection_blocks=shuffle_projection_blocks,
    )
    bootstrap = sdl_multiplier_bootstrap(
        studentization,
        replicates=A,
        rng=bootstrap_rng,
        batch_size=bootstrap_batch_size,
        retain_components=retain_bootstrap_components,
    )
    return SDLTestResult(
        statistic=studentization.statistic,
        p_value=bootstrap.p_value,
        hypothesis=hypothesis,
        kernel=kernel,
        studentization=studentization,
        bootstrap=bootstrap,
        sample_size=len(observations),
        kernel_order=kernel.order,
        budget=N,
        projection_size=len(studentization.hajek.indices),
        bootstrap_replicates=A,
        constraint_count=len(component.sdl_constraints),
        seed=seed,
        augmentation=augmentation,
        symmetrization=kernel.symmetrization,
        symmetrization_permutations=kernel.permutation_count,
    )
