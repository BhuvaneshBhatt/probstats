"""Intersection-union SDL testing for reducible semialgebraic nulls."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ._validation import generator_from_seed_or_rng, spawn_named_generators
from .hypotheses import SemialgebraicHypothesis
from .sdl import SDLTestResult, sdl_test


@dataclass(frozen=True, slots=True)
class SDLUnionTestResult:
    """Componentwise SDL tests for a finite union null.

    For ``H0 = union_k H0k``, rejection requires rejecting every component.
    The intersection-union p-value is therefore ``max_k p_k``.
    """

    p_value: float
    component_results: tuple[SDLTestResult, ...]
    hypothesis: SemialgebraicHypothesis
    component_seeds: tuple[int, ...]
    seed: int | None

    @property
    def component_p_values(self) -> np.ndarray:
        return np.asarray(
            [result.p_value for result in self.component_results], dtype=float
        )

    @property
    def component_statistics(self) -> np.ndarray:
        return np.asarray(
            [result.statistic for result in self.component_results], dtype=float
        )

    @property
    def component_count(self) -> int:
        return len(self.component_results)


def _component_hypothesis(
    hypothesis: SemialgebraicHypothesis, index: int
) -> SemialgebraicHypothesis:
    component = hypothesis.components[index]
    return SemialgebraicHypothesis(
        hypothesis.parameters,
        (component,),
        ambient=hypothesis.ambient,
        estimators=hypothesis.estimators,
        sample_space=hypothesis.sample_space,
        metadata={
            **dict(hypothesis.metadata),
            "union_component_index": index,
            "union_component_label": component.label,
        },
    )


def sdl_union_test(
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
) -> SDLUnionTestResult:
    """Test a finite union null by componentwise SDL intersection-union testing.

    Each basic component is tested with the ordinary ``sdl_test`` procedure.
    Since the null is a union, the global null is rejected only when every
    component null is rejected; the valid combined p-value is the maximum of
    the component p-values.

    Independent child streams are derived for component procedures whether the
    entropy source is ``seed`` or ``rng``. Derived component seeds are recorded
    so every component test can be replayed independently.
    """
    if not isinstance(hypothesis, SemialgebraicHypothesis):
        raise TypeError("hypothesis must be a SemialgebraicHypothesis")
    generator, seed = generator_from_seed_or_rng(seed, rng)
    names = tuple(f"component_{index}" for index in range(len(hypothesis.components)))
    component_streams, component_seed_map = spawn_named_generators(generator, names)
    component_generators = tuple(component_streams[name] for name in names)
    child_seeds = tuple(component_seed_map[name] for name in names)

    results = []
    for index in range(len(hypothesis.components)):
        component_hypothesis = _component_hypothesis(hypothesis, index)
        component_kwargs = {
            "budget": budget,
            "bootstrap_replicates": bootstrap_replicates,
            "kernel_order": kernel_order,
            "n1": n1,
            "projection_indices": projection_indices,
            "shuffle_projection_blocks": shuffle_projection_blocks,
            "bootstrap_batch_size": bootstrap_batch_size,
            "retain_bootstrap_components": retain_bootstrap_components,
            "augment_constraints_count": augment_constraints_count,
            "augmentation_concentration": augmentation_concentration,
            "kernel_symmetrization": kernel_symmetrization,
            "symmetrization_permutations": symmetrization_permutations,
        }
        component_kwargs["rng"] = component_generators[index]
        results.append(sdl_test(sample, component_hypothesis, **component_kwargs))

    component_results = tuple(results)
    p_value = max(result.p_value for result in component_results)
    return SDLUnionTestResult(
        p_value=p_value,
        component_results=component_results,
        hypothesis=hypothesis,
        component_seeds=child_seeds,
        seed=seed,
    )
