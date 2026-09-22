"""Shared validation for semialgebraic testing internals."""

from numbers import Integral

import numpy as np


def positive_integer(value: int, *, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def as_generator(rng: np.random.Generator | int | None) -> np.random.Generator:
    """Normalize a low-level NumPy generator-or-seed argument."""
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def generator_from_seed_or_rng(
    seed: int | None, rng: np.random.Generator | None
) -> tuple[np.random.Generator, int | None]:
    if seed is not None and rng is not None:
        raise ValueError("pass either seed or rng, not both")
    if seed is not None:
        if not isinstance(seed, Integral) or isinstance(seed, bool):
            raise TypeError("seed must be an integer")
        normalized_seed = int(seed)
        return np.random.default_rng(normalized_seed), normalized_seed
    if rng is None:
        return np.random.default_rng(), None
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    return rng, None


def spawn_generators(
    generator: np.random.Generator, count: int
) -> tuple[tuple[np.random.Generator, ...], tuple[int, ...]]:
    """Derive independent child streams while consuming fixed parent entropy."""
    seeds = tuple(
        int(value)
        for value in generator.integers(
            0, np.iinfo(np.uint64).max, size=count, dtype=np.uint64
        )
    )
    return tuple(np.random.default_rng(seed) for seed in seeds), seeds


def spawn_named_generators(
    generator: np.random.Generator, names: tuple[str, ...]
) -> tuple[dict[str, np.random.Generator], dict[str, int]]:
    """Derive independent child streams with explicit semantic ownership."""
    if len(set(names)) != len(names):
        raise ValueError("stream names must be unique")
    generators, seeds = spawn_generators(generator, len(names))
    return dict(zip(names, generators, strict=True)), dict(
        zip(names, seeds, strict=True)
    )
