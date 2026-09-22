"""Small manual benchmarks for combinatorial statistical kernels.

Run with ``python benchmarks/u_statistics.py``. These are not correctness tests and
are kept outside pytest so release tests do not acquire timing-dependent failures.
"""

from time import perf_counter

from probstats.u_statistics import incomplete_u_statistic, u_statistic


def time_call(label, function):
    start = perf_counter()
    function()
    print(f"{label}: {perf_counter() - start:.3f}s")


def main():
    sample = tuple(range(40))
    time_call(
        "complete order-3",
        lambda: u_statistic(sample, lambda x, y, z: x + y + z, order=3),
    )
    time_call(
        "incomplete order-4 N=5000",
        lambda: incomplete_u_statistic(
            sample,
            lambda a, b, c, d: a + b + c + d,
            order=4,
            budget=5000,
            selection="fixed",
            rng=0,
        ),
    )


if __name__ == "__main__":
    main()
