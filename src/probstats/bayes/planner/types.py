"""Planning configuration and result objects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


class InferencePlanningError(RuntimeError):
    """Raised when no inference route is applicable or every applicable route fails."""


@dataclass(frozen=True, slots=True)
class PlannerConfig:
    """Heuristics controlling automatic dispatch.

    Costs are relative policy scores, not wall-clock predictions.  Lower is preferred.
    Exact symbolic integration is guarded by latent dimension and symbolic operation
    count so an apparently attractive exact route does not dominate obviously harder
    problems.
    """

    max_exact_latents: int = 2
    max_exact_operations: int = 250
    prefer_exact: bool = True
    allow_laplace: bool = True
    allow_nested: bool = True
    allow_mcmc: bool = True
    assess_diagnostics: bool = True
    fallback_on_poor_quality: bool = False


@dataclass(frozen=True, slots=True)
class CandidateAssessment:
    method: str
    applicable: bool
    exact: bool
    estimated_cost: float
    reason: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class InferencePlan:
    candidates: tuple[CandidateAssessment, ...]

    @property
    def applicable(self) -> tuple[CandidateAssessment, ...]:
        return tuple(c for c in self.candidates if c.applicable)

    @property
    def selected(self) -> CandidateAssessment | None:
        applicable = self.applicable
        return applicable[0] if applicable else None

    def explain(self) -> str:
        lines = ["Candidate inference routes"]
        for candidate in self.candidates:
            state = "applicable" if candidate.applicable else "rejected"
            qualifier = "exact" if candidate.exact else "approximate"
            lines.append(
                f"- {candidate.method}: {state}, {qualifier}, cost={candidate.estimated_cost:g}: "
                f"{candidate.reason}"
            )
        return "\n".join(lines)


__all__ = [
    "CandidateAssessment",
    "InferencePlan",
    "InferencePlanningError",
    "PlannerConfig",
]
