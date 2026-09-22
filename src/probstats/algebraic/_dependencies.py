"""Lazy capability loaders for optional algebraic and tensor backends."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SemialgCapabilities:
    analyze_equality_ideal: object
    ideal_degree: object
    singular_locus: object


@dataclass(frozen=True, slots=True)
class TensorAtlasCapabilities:
    TensorArray: object
    grouped_flatten: object
    segre_equations: object
    segre_parameterization: object


@dataclass(frozen=True, slots=True)
class ToricSemialgCapabilities:
    toric_ideal: object
    markov_basis: object


@dataclass(frozen=True, slots=True)
class LikelihoodSemialgCapabilities:
    elimination_ideal_qq: object
    saturate_ideal_qq: object
    solve_zero_dimensional_system: object
    semialgebraic_maximize: object


@dataclass(frozen=True, slots=True)
class LatentCapabilities:
    elimination_ideal_qq: object
    generic_cp_identifiability: object
    secant_expected_dimension: object
    secant_parameterization: object


@dataclass(frozen=True, slots=True)
class MomentTensorCapabilities:
    TensorArray: object
    cp_decompose: object
    generic_cp_identifiability: object
    symmetric_rank: object
    tensor_rank: object
    waring_decompose: object


def require_semialg():
    """Return named semialg capabilities used by model invariants."""
    try:
        from semialg import singular_locus
        from semialg.algebraic import analyze_equality_ideal, ideal_degree
    except ImportError as exc:
        raise ImportError(
            "algebraic-model invariants require semialg; install probstats[algebraic]"
        ) from exc
    return SemialgCapabilities(analyze_equality_ideal, ideal_degree, singular_locus)


def require_tensoratlas():
    """Return named TensorAtlas capabilities used by tensor-valued models."""
    try:
        from tensoratlas.algebraic import (
            grouped_flatten,
            segre_equations,
            segre_parameterization,
        )
        from tensoratlas.core import TensorArray
    except ImportError as exc:
        raise ImportError(
            "tensor-statistics operations require tensoratlas; install probstats[algebraic-tensor]"
        ) from exc
    return TensorAtlasCapabilities(
        TensorArray, grouped_flatten, segre_equations, segre_parameterization
    )


def require_toric_semialg():
    """Return named semialg toric-model capabilities."""
    try:
        from semialg.algebraic import markov_basis, toric_ideal
    except ImportError as exc:
        raise ImportError(
            "toric-model operations require semialg; install probstats[algebraic]"
        ) from exc
    return ToricSemialgCapabilities(toric_ideal, markov_basis)


def require_likelihood_semialg():
    """Return named exact semialg capabilities used by likelihood geometry."""
    try:
        from semialg.algebraic import elimination_ideal_qq, saturate_ideal_qq
        from semialg.optimization import semialgebraic_maximize
        from semialg.solve import solve_zero_dimensional_system
    except ImportError as exc:
        raise ImportError(
            "algebraic likelihood operations require semialg; install probstats[algebraic]"
        ) from exc
    return LikelihoodSemialgCapabilities(
        elimination_ideal_qq,
        saturate_ideal_qq,
        solve_zero_dimensional_system,
        semialgebraic_maximize,
    )


def require_latent_backends():
    """Return named exact algebra/tensor capabilities for latent geometry."""
    try:
        from semialg.algebraic import elimination_ideal_qq
        from tensoratlas.algebraic import (
            generic_cp_identifiability,
            secant_expected_dimension,
            secant_parameterization,
        )
    except ImportError as exc:
        raise ImportError(
            "latent algebraic models require semialg and tensoratlas; install probstats[algebraic-tensor]"
        ) from exc
    return LatentCapabilities(
        elimination_ideal_qq,
        generic_cp_identifiability,
        secant_expected_dimension,
        secant_parameterization,
    )


def require_moment_tensoratlas():
    """Return named TensorAtlas decomposition capabilities for moment statistics."""
    try:
        from tensoratlas.algebraic import (
            cp_decompose,
            generic_cp_identifiability,
            symmetric_rank,
            tensor_rank,
            waring_decompose,
        )
        from tensoratlas.core import TensorArray
    except ImportError as exc:
        raise ImportError(
            "moment-tensor operations require tensoratlas; install probstats[algebraic-tensor]"
        ) from exc
    return MomentTensorCapabilities(
        TensorArray,
        cp_decompose,
        generic_cp_identifiability,
        symmetric_rank,
        tensor_rank,
        waring_decompose,
    )
