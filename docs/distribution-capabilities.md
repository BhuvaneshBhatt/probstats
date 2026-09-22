# Distribution capability matrix
This generated matrix distinguishes native symbolic implementations from generic fallbacks. `generic fallback` means the protocol exists but may remain unevaluated or fail for parameterizations that SymPy cannot solve; it is not a promise of a closed form. SciPy-oracle availability records whether the test registry has a compatible direct differential reference.

| Distribution | Category | Event | Density/PMF | CDF | Quantile | Sampling | Moments | Entropy | Symbolic parameters | SciPy oracle |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `LKJ` | catalogue law | vector/matrix | generic from log-density | n/a (non-scalar) | n/a (non-scalar) | generic/derived | native | native | yes | no/direct mapping absent |
| `Bernoulli` | catalogue law | scalar | native symbolic | native symbolic | native symbolic | generic/derived | native | native | yes | yes |
| `Beta` | catalogue law | scalar | native symbolic | generic fallback | generic fallback | generic/derived | native | native | yes | yes |
| `BetaBinomial` | catalogue law | scalar | native symbolic | generic fallback | generic fallback | native | native | generic exact/numerical | yes | yes |
| `BetaPrime` | catalogue law | scalar | generic from log-density | native symbolic | generic fallback | native | native | generic exact/numerical | yes | yes |
| `Binomial` | catalogue law | scalar | native symbolic | generic fallback | generic fallback | generic/derived | native | generic exact/numerical | yes | yes |
| `Categorical` | catalogue law | scalar | native symbolic | generic fallback | generic fallback | generic/derived | native | native | yes | no/direct mapping absent |
| `Cauchy` | catalogue law | scalar | generic from log-density | native symbolic | native symbolic | native | native | native | yes | yes |
| `CensoredDistribution` | wrapper/data-driven | scalar | native symbolic | native symbolic | generic fallback | native | generic exact | generic exact/numerical | yes | no/direct mapping absent |
| `ChiSquared` | catalogue law | scalar | generic from log-density | native symbolic | generic fallback | native | native | native | yes | yes |
| `Dirichlet` | catalogue law | vector/matrix | native symbolic | n/a (non-scalar) | n/a (non-scalar) | generic/derived | native | generic exact/numerical | yes | no/direct mapping absent |
| `DirichletMultinomial` | catalogue law | vector/matrix | generic from log-density | n/a (non-scalar) | n/a (non-scalar) | native | native | generic exact/numerical | yes | no/direct mapping absent |
| `DiscreteUniform` | catalogue law | scalar | native symbolic | native symbolic | native symbolic | native | native | generic exact/numerical | yes | yes |
| `Distribution` | catalogue law | abstract | abstract | abstract | abstract | generic/derived | generic exact | generic exact/numerical | yes | no/direct mapping absent |
| `Exponential` | catalogue law | scalar | native symbolic | native symbolic | native symbolic | generic/derived | native | native | yes | yes |
| `ExponentialFamily` | catalogue law | abstract | abstract | abstract | abstract | generic/derived | generic exact | generic exact/numerical | yes | no/direct mapping absent |
| `FDistribution` | catalogue law | scalar | generic from log-density | native symbolic | generic fallback | native | native | generic exact/numerical | yes | yes |
| `Frechet` | catalogue law | scalar | generic from log-density | native symbolic | native symbolic | native | native | generic exact/numerical | yes | yes |
| `Gamma` | catalogue law | scalar | native symbolic | generic fallback | generic fallback | generic/derived | native | native | yes | yes |
| `Geometric` | catalogue law | scalar | native symbolic | native symbolic | native symbolic | native | native | native | yes | yes |
| `Gumbel` | catalogue law | scalar | generic from log-density | native symbolic | native symbolic | native | native | generic exact/numerical | yes | yes |
| `HalfCauchy` | catalogue law | scalar | generic from log-density | native symbolic | native symbolic | native | native | native | yes | yes |
| `HalfNormal` | catalogue law | scalar | generic from log-density | native symbolic | native symbolic | native | native | native | yes | yes |
| `HistogramDistribution` | wrapper/data-driven | scalar | native symbolic | native symbolic | generic fallback | generic/derived | native | generic exact/numerical | data-driven | no/direct mapping absent |
| `Hypergeometric` | catalogue law | scalar | native symbolic | generic fallback | generic fallback | native | native | generic exact/numerical | yes | yes |
| `InverseGamma` | catalogue law | scalar | native symbolic | generic fallback | generic fallback | generic/derived | generic exact | generic exact/numerical | yes | yes |
| `InverseGaussian` | catalogue law | scalar | generic from log-density | native symbolic | generic fallback | native | native | generic exact/numerical | yes | no/direct mapping absent |
| `InverseWishart` | catalogue law | vector/matrix | generic from log-density | n/a (non-scalar) | n/a (non-scalar) | generic/derived | generic exact | generic exact/numerical | yes | no/direct mapping absent |
| `KernelDensityDistribution` | wrapper/data-driven | scalar | native symbolic | generic fallback | generic fallback | generic/derived | native | generic exact/numerical | data-driven | no/direct mapping absent |
| `KernelMixtureDistribution` | wrapper/data-driven | scalar | native symbolic | native symbolic | generic fallback | generic/derived | native | generic exact/numerical | data-driven | no/direct mapping absent |
| `Kumaraswamy` | catalogue law | scalar | generic from log-density | native symbolic | native symbolic | native | native | generic exact/numerical | yes | no/direct mapping absent |
| `LKJCholesky` | catalogue law | vector/matrix | generic from log-density | n/a (non-scalar) | n/a (non-scalar) | generic/derived | native | native | yes | no/direct mapping absent |
| `Laplace` | catalogue law | scalar | generic from log-density | native symbolic | native symbolic | native | native | native | yes | yes |
| `LogNormal` | catalogue law | scalar | native symbolic | generic fallback | generic fallback | generic/derived | generic exact | generic exact/numerical | yes | yes |
| `Logistic` | catalogue law | scalar | generic from log-density | native symbolic | native symbolic | native | native | native | yes | yes |
| `MatrixNormal` | catalogue law | vector/matrix | generic from log-density | n/a (non-scalar) | n/a (non-scalar) | generic/derived | native | native | yes | no/direct mapping absent |
| `Maxwell` | catalogue law | scalar | generic from log-density | native symbolic | generic fallback | native | native | generic exact/numerical | yes | yes |
| `Multinomial` | catalogue law | vector/matrix | native symbolic | n/a (non-scalar) | n/a (non-scalar) | generic/derived | native | generic exact/numerical | yes | no/direct mapping absent |
| `MultivariateKernelDensityDistribution` | catalogue law | vector/matrix | native symbolic | n/a (non-scalar) | n/a (non-scalar) | generic/derived | native | generic exact/numerical | data-driven | no/direct mapping absent |
| `MultivariateNormal` | catalogue law | vector/matrix | native symbolic | n/a (non-scalar) | n/a (non-scalar) | generic/derived | native | generic exact/numerical | yes | no/direct mapping absent |
| `MultivariateStudentT` | catalogue law | vector/matrix | generic from log-density | n/a (non-scalar) | n/a (non-scalar) | generic/derived | native | generic exact/numerical | yes | no/direct mapping absent |
| `Nakagami` | catalogue law | scalar | generic from log-density | generic fallback | generic fallback | native | native | generic exact/numerical | yes | yes |
| `NegativeBinomial` | catalogue law | scalar | native symbolic | generic fallback | generic fallback | native | native | generic exact/numerical | yes | yes |
| `NoncentralChiSquared` | catalogue law | scalar | native symbolic | generic fallback | generic fallback | native | native | generic exact/numerical | yes | yes |
| `NoncentralF` | catalogue law | scalar | native symbolic | generic fallback | generic fallback | native | native | generic exact/numerical | yes | yes |
| `NoncentralT` | catalogue law | scalar | native symbolic | generic fallback | generic fallback | native | native | generic exact/numerical | yes | yes |
| `Normal` | catalogue law | scalar | native symbolic | native symbolic | native symbolic | generic/derived | native | native | yes | yes |
| `ParameterMixtureDistribution` | wrapper/data-driven | scalar | native symbolic | native symbolic | generic fallback | native | generic exact | generic exact/numerical | yes | no/direct mapping absent |
| `Pareto` | catalogue law | scalar | generic from log-density | native symbolic | native symbolic | native | native | native | yes | yes |
| `Poisson` | catalogue law | scalar | native symbolic | generic fallback | generic fallback | generic/derived | native | generic exact/numerical | yes | yes |
| `PowerDistribution` | catalogue law | scalar | generic from log-density | native symbolic | native symbolic | native | native | generic exact/numerical | yes | yes |
| `ProbabilityDistribution` | wrapper/data-driven | scalar | native symbolic | native symbolic | native symbolic | native | generic exact | generic exact/numerical | yes | no/direct mapping absent |
| `Rayleigh` | catalogue law | scalar | generic from log-density | native symbolic | native symbolic | native | native | generic exact/numerical | yes | yes |
| `Rice` | catalogue law | scalar | generic from log-density | generic fallback | generic fallback | native | native | generic exact/numerical | yes | yes |
| `Skellam` | catalogue law | scalar | native symbolic | native symbolic | generic fallback | native | native | generic exact/numerical | yes | yes |
| `SplicedDistribution` | wrapper/data-driven | scalar | native symbolic | native symbolic | generic fallback | native | generic exact | generic exact/numerical | yes | no/direct mapping absent |
| `StudentT` | catalogue law | scalar | generic from log-density | native symbolic | native symbolic | generic/derived | native | native | yes | yes |
| `SymbolicDistribution` | wrapper/data-driven | scalar | generic from log-density | generic fallback | generic fallback | generic/derived | generic exact | generic exact/numerical | yes | no/direct mapping absent |
| `Triangular` | catalogue law | scalar | native symbolic | native symbolic | generic fallback | native | native | generic exact/numerical | yes | yes |
| `Uniform` | catalogue law | scalar | generic from log-density | native symbolic | native symbolic | native | native | native | yes | yes |
| `VonMises` | catalogue law | scalar | generic from log-density | generic fallback | generic fallback | native | native | generic exact/numerical | yes | yes |
| `Weibull` | catalogue law | scalar | generic from log-density | native symbolic | native symbolic | native | native | native | yes | yes |
| `Wishart` | catalogue law | vector/matrix | generic from log-density | n/a (non-scalar) | n/a (non-scalar) | generic/derived | generic exact | generic exact/numerical | yes | no/direct mapping absent |
| `Zipf` | catalogue law | scalar | native symbolic | native symbolic | generic fallback | native | native | generic exact/numerical | yes | yes |
