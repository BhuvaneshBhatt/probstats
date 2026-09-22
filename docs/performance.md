# Performance

`probstats` keeps symbolic and numerical workloads separate where possible. Numerical sample operations use NumPy arrays and batched linear algebra; exact formulas remain SymPy expressions.

## Numerical workloads

Prefer `size=` sampling instead of Python loops when drawing many observations, matrices, or eigenvalue sets. Information-measure Monte Carlo fallbacks evaluate NumPy-compatible lambdified log densities in batches. Rolling sum and mean use NumPy sliding-window reductions, and random-matrix built-in spectral statistics use stacked linear algebra where that reduces overhead.

For custom functions that cannot accept arrays, numerical information-measure evaluation falls back to scalar calls. This keeps the public API flexible while retaining the fast path for vectorized functions.

## Exact symbolic workloads

Exact determinants used in Bayesian definiteness certification, matrix-distribution formulas, information measures, and multivariate Bayesian regression are routed through SymPy polynomial domains when the entries admit an exact domain. Tiny matrices remain on the direct SymPy path because domain conversion costs more than it saves there; larger polynomial and rational matrices use the domain backend to avoid expression swell.

SymPy can use `python-flint` for supported ground-domain arithmetic when Flint support is available in the Python environment. `probstats` does not expose Flint objects in its public API, so enabling such acceleration does not change returned expression types or package contracts. General expression-domain matrices still use the ordinary SymPy determinant path.

Direct conversion of arbitrary symbolic probability expressions to Flint objects is usually not helpful: transcendental expressions, assumptions, piecewise conditions, and symbolic integration remain SymPy-level operations. Flint is most promising for exact integer/rational/polynomial kernels rather than as a replacement symbolic backend.

## Memory and algorithm choices

Not every Python loop should be vectorized. Running quantiles use a coordinate-compressed Fenwick tree for the common NumPy interpolation methods, giving logarithmic-time order-statistic updates without an ``n x n`` intermediate. Kendall tau-b uses a related Fenwick inversion count rather than quadratic pair enumeration.

LOWESS sorts the predictor once and restricts each fit to the contiguous nearest-neighbor window; compact-kernel local-polynomial smoothing uses ``searchsorted`` to exclude observations outside the kernel support before solving the local weighted least-squares problem. The local solver remains NumPy ``lstsq`` rather than normal equations, preserving numerical conditioning while reducing the rows processed per target.

Marchenko--Pastur CDF evaluation uses an angular change of variables that removes the square-root spectral-edge factors. Array inputs are sorted and integrated incrementally between successive query points in bounded chunks, avoiding repeated integration from the lower edge while keeping memory usage controlled.
