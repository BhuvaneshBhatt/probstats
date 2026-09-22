# Probability distributions

`probstats` provides symbolic-first scalar, vector, simplex, and matrix-valued probability distributions. Every distribution exposes structural support and parameter constraints, exact density or mass functions, generic probability functionals, and numerical sampling when all required parameters are concrete.

## Scalar families

The scalar catalogue includes Bernoulli, Binomial, Beta, Categorical, Cauchy, Chi-squared, discrete uniform, Exponential, F, Gamma, Geometric, Gumbel, hypergeometric, inverse Gamma, inverse Gaussian, Kumaraswamy, Laplace, Logistic, LogNormal, Maxwell, Nakagami, negative binomial, noncentral Chi-squared, noncentral F, noncentral t, Normal, Pareto, Poisson, power, Rayleigh, Rice, Skellam, Student-t, triangular, Uniform, von Mises, Weibull, Zipf, and beta-prime distributions.

Compound count distributions include beta-binomial and Dirichlet-multinomial laws.

## Parameter conventions

Parameterizations are explicit in constructor names and attributes. `Gamma(shape, scale)` uses a scale parameter. `InverseGaussian(mean, shape)` uses mean \(\mu\) and shape \(\lambda\). `FDistribution(df1, df2)` uses numerator and denominator degrees of freedom. Noncentral families use a nonnegative `noncentrality` parameter. `Nakagami(shape, spread)` uses \(m\) and \(\Omega\), so \(E[X^2]=\Omega\).

`Hypergeometric(successes, failures, draws)` describes sampling without replacement from a population containing `successes` marked and `failures` unmarked objects. `Geometric(p)` counts trials through the first success, while `NegativeBinomial(r, p)` counts failures before `r` successes.

## Algebraic identities

Exact distribution algebra canonicalizes identities when doing so preserves the random variable's coordinates. In particular,

\[
F_{d_1,d_2}\equiv \operatorname{BetaPrime}\left(\frac{d_1}{2},\frac{d_2}{2},\frac{d_2}{d_1}\right),
\]

where the third beta-prime parameter is its scale.

Several transform functions are exposed directly when closed forms are useful. The inverse-Gaussian and Gumbel laws have direct MGFs, Skellam has a direct characteristic function, Zipf has a polylogarithmic PGF, and discrete uniform has closed finite geometric transform forms.

## Noncentral distributions

`NoncentralChiSquared` has a Bessel-form density. `NoncentralF` retains the exact Poisson-mixture series. `NoncentralT` retains an exact normal/Chi-squared integral for its density. Sampling uses the defining stochastic constructions, so no SciPy dependency is required.

## Circular distributions

`VonMises(location, concentration)` uses the principal interval `[location - pi, location + pi]` as its scalar representation. Samples are returned on that same interval. This representation makes the ordinary density and support protocol well defined while retaining the usual circular von Mises law.
