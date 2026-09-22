# Random-matrix statistics

`probstats.random_matrix` separates finite random-matrix ensembles and spectral
statistics from matrix-valued probability distributions such as `Wishart`,
`InverseWishart`, `LKJ`, and `MatrixNormal`.

## Gaussian ensembles

`GOE(n)`, `GUE(n)`, and `GSE(n)` use Wigner scaling, so their empirical spectral
distribution converges to the radius-two Wigner semicircle law. `sample_matrix()`
returns dense invariant GOE/GUE matrices. For GSE it returns the exact
Dumitriu--Edelman beta=4 tridiagonal spectral representative; its eigenvalues
have the GSE joint eigenvalue law without pretending that a real tridiagonal
matrix is a quaternion-Hermitian matrix representation.

`sample_eigenvalues()` uses the beta-Hermite tridiagonal model for beta 1, 2,
and 4 and therefore avoids forming a dense matrix when only eigenvalues are
needed.

## Wishart ensembles

`WishartEnsemble(p, df, beta=1|2, normalized=True)` samples real or complex
Wishart matrices. With `normalized=True`, the matrix is `X*X / df`, which is the
normalization used by the Marchenko--Pastur limit.

For real identity/scalar covariance Wishart matrices, `trace_distribution()`
returns the exact Gamma law. `determinant_law()` returns Bartlett's exact product
representation, including exact chi-square degrees of freedom, sampling, mean,
and log-determinant moments. General condition-number laws do not have a simple
closed form, so `sample_spectral_statistic(..., "condition_number")` provides an
explicit Monte-Carlo route rather than claiming one.

## Wigner semicircle and Marchenko--Pastur

`WignerSemicircle` implements PDF, CDF, exact mean/variance, and exact sampling.

`MarchenkoPastur(ratio=p/n, scale=1)` implements the continuous density, spectral
edges, atom at zero when `p/n > 1`, numerical CDF, and first two moments. The
atom is represented explicitly rather than hidden inside a continuous density.

## Largest eigenvalues and Tracy--Widom

`gaussian_largest_eigenvalue_scaling()` and
`wishart_largest_eigenvalue_scaling()` return soft-edge centering/scaling maps.
The corresponding `*_approximation()` functions compose those maps with a
`TracyWidom(beta=1|2|4)` object.

The core package does not ship a low-accuracy Tracy--Widom
approximation. Numerical PDF/CDF/PPF evaluation is delegated to a backend that
implements the `TracyWidomBackend` protocol and is registered with
`register_tracy_widom_backend()`. Registrations reject duplicate names by
default and can be removed with `unregister_tracy_widom_backend()`. Eigenvalue sampling and soft-edge
standardization themselves require no Tracy--Widom backend.

## Spectral statistics

`eigenvalue_statistics(matrix)` reports minimum/maximum eigenvalue, trace,
determinant, spectral radius, and spectral condition number.
`sample_spectral_statistic()` samples finite-ensemble distributions of largest
eigenvalue, trace, determinant, condition number, or a caller-supplied statistic.

## Batch sampling

Matrix ensembles accept `size=` when matrices as well as eigenvalues are needed. NumPy's stacked matrix operations are used where they are beneficial:

```python
from probstats.random_matrix import (
    GOE,
    WishartEnsemble,
)

matrices = GOE(40).sample_matrix(size=200, rng=123)
eigenvalues = GOE(40).sample_eigenvalues(size=200, rng=123)

wishart = WishartEnsemble(20, 50)
matrices = wishart.sample_matrix(size=(4, 25), rng=123)
```

The sample dimensions precede the matrix dimensions, so the last two axes always identify a matrix. Large Wishart eigenvalue requests use a memory-conscious path for larger matrix dimensions rather than forcing every sample into one stacked allocation.
