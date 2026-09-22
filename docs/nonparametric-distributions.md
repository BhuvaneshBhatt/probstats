# Nonparametric distributions

The nonparametric layer represents data-driven probability laws as ordinary
`probstats.Distribution` objects. They therefore work with generic density,
likelihood, expectation, sampling, and information-measure machinery.

## Histogram distributions

`HistogramDistribution(data, bins=...)` converts a sample to a piecewise-uniform
probability law. It also accepts explicit `bin_edges=` and `probabilities=`. Bin
probabilities are normalized, so unequal-width bins correctly have density equal
to probability divided by bin width. Its CDF, mean, and variance are available in
closed form.

## Gaussian kernel mixtures

`KernelMixtureDistribution(locations, bandwidths, weights=None)` represents a
finite mixture of univariate Gaussian kernels. Bandwidths may be common or
component-specific. This is useful independently of KDE, for example for smooth
empirical priors or approximate posterior representations.

## Univariate KDE

`KernelDensityDistribution(data, bandwidth="scott", weights=None)` is a weighted
Gaussian KDE. The bandwidth may be positive numeric
or one of:

- `"scott"`: weighted Scott rule;
- `"silverman"`: robust Silverman normal-reference rule;
- `"plugin"`: one-stage Gaussian plug-in estimate using an exact pilot-mixture
  curvature integral;
- `"cv"` / `"lscv"`: least-squares cross-validation over a logarithmic candidate
  grid.

The selectors are also public functions: `scott_bandwidth`,
`silverman_bandwidth`, `plugin_bandwidth`, `cv_bandwidth`, and
`select_bandwidth`.

Weighted bandwidth rules use Kish effective sample size, `1/sum(w**2)`, after
normalization.

## Multivariate KDE

`MultivariateKernelDensityDistribution(data, bandwidth="scott", weights=None)`
fits a Gaussian KDE to an `(n, d)` sample. Scott
and Silverman rules scale the weighted sample covariance to obtain a full
positive-definite bandwidth matrix. A user can instead supply an explicit `d x d`
positive-definite bandwidth matrix.

Sampling chooses a kernel center according to its weight and adds Gaussian noise
with the bandwidth covariance. `size` describes batch dimensions, so a 2-D KDE
sampled with `size=(5, 3)` returns shape `(5, 3, 2)`.

## Scope

Density estimation is exposed as probability distributions rather
than plotting or dataframe-specific wrappers. Boundary-corrected kernels,
adaptive/local bandwidth KDEs, and FFT acceleration are possible later extensions;
they are not silently substituted for the Gaussian estimators documented here.
