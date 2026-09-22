"""Small repeatable benchmarks for SDL bootstrap and Hájek projection paths."""

from __future__ import annotations

from statistics import median
from time import perf_counter

import numpy as np
import sympy as sp

from probstats.algebraic.testing import (
    SemialgebraicHypothesis,
    hajek_projection_estimate,
    sdl_multiplier_bootstrap,
    sdl_test,
)


def _median_seconds(function, repeats: int = 5) -> float:
    timings = []
    for _ in range(repeats):
        start = perf_counter()
        function()
        timings.append(perf_counter() - start)
    return median(timings)


def main() -> None:
    rng = np.random.default_rng(1)
    sample = rng.normal(size=300)
    theta = sp.Symbol("theta")
    hypothesis = SemialgebraicHypothesis.basic(
        (theta,), inequalities=(theta,), estimators={theta: lambda x: x}
    )
    result = sdl_test(sample, hypothesis, budget=250, bootstrap_replicates=20, seed=2)
    bootstrap = _median_seconds(
        lambda: sdl_multiplier_bootstrap(
            result.studentization, replicates=1000, rng=3, batch_size=250
        )
    )
    hajek = _median_seconds(
        lambda: hajek_projection_estimate(
            sample,
            lambda x, y: np.array([x + y, x * y]),
            order=2,
            n1=300,
        )
    )
    print(f"bootstrap A=1000: {bootstrap:.6f} s")
    print(f"Hajek n=300, m=2, d=2: {hajek:.6f} s")


if __name__ == "__main__":
    main()
