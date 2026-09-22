"""Inference result types shared by exact and numerical engines."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

from ...results import BayesianResult


class InferenceKind(str, Enum):
    EXACT = "exact"
    APPROXIMATE = "approximate"
    SAMPLED = "sampled"


@dataclass(frozen=True, slots=True)
class InferenceStep:
    """One auditable step performed by an inference engine/planner."""

    method: str
    description: str
    exact: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class InferenceResult(BayesianResult):
    """Backend-independent result container.

    ``posterior`` is unconstrained: exact engines may return a
    symbolic distribution, while sampling engines may return an ArviZ-compatible
    object. ``diagnostics`` and ``metadata`` are immutable mappings.
    """

    posterior: Any
    kind: InferenceKind
    log_evidence: Any | None = None
    steps: tuple[InferenceStep, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(
            self, "diagnostics", MappingProxyType(dict(self.diagnostics))
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def is_exact(self) -> bool:
        return self.kind is InferenceKind.EXACT

    def to_posterior_data(
        self, *, draws: int = 1000, chains: int = 4, rng: Any = None
    ) -> Any:
        """Convert this result to the common ``PosteriorData`` representation."""
        from ..diagnostics import to_posterior_data

        return to_posterior_data(self, draws=draws, chains=chains, rng=rng)

    def to_arviz(self, *, draws: int = 1000, chains: int = 4, rng: Any = None) -> Any:
        """Convert this result to ``arviz.InferenceData`` when ArviZ is installed."""
        from ..diagnostics import to_arviz

        return to_arviz(self, draws=draws, chains=chains, rng=rng)

    def predict_distribution(
        self, x, *, observation: bool = True, predictive=None, size=2000, rng=None
    ):
        from ..predictive import predict_distribution

        return predict_distribution(
            self.posterior,
            x,
            observation=observation,
            predictive=predictive,
            size=size,
            rng=rng,
        )

    def predict(
        self, x, *, observation: bool = True, predictive=None, size=2000, rng=None
    ):
        from ..predictive import predict

        return predict(
            self.posterior,
            x,
            observation=observation,
            predictive=predictive,
            size=size,
            rng=rng,
        )

    def posterior_predictive(
        self,
        x,
        *,
        size=1000,
        rng=None,
        observation: bool = True,
        predictive=None,
    ):
        from ..predictive import posterior_predictive

        return posterior_predictive(
            self.posterior,
            x,
            size=size,
            rng=rng,
            observation=observation,
            predictive=predictive,
        )

    def predictive_interval(
        self,
        x,
        *,
        level: float = 0.95,
        observation: bool = True,
        predictive=None,
        size=4000,
        rng=None,
    ):
        from ..predictive import predictive_interval

        return predictive_interval(
            self.posterior,
            x,
            level=level,
            observation=observation,
            predictive=predictive,
            size=size,
            rng=rng,
        )

    @property
    def quality(self) -> Any | None:
        """Diagnostic quality report attached by the planner, if available."""
        return self.diagnostics.get("quality")

    def explain(self) -> str:
        if not self.steps:
            return "No inference provenance was recorded."
        lines = ["Inference plan"]
        for index, step in enumerate(self.steps, start=1):
            qualifier = "exact" if step.exact else "approximate"
            lines.append(f"{index}. {step.method} [{qualifier}]: {step.description}")
        return "\n".join(lines)
