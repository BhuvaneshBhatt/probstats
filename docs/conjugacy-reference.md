# Conjugacy reference

The default conjugacy registry is driven by stable `probstats` distribution-family
signatures. The table below is the complete named registry in this release.

| Likelihood | Prior | Sufficient statistics | Posterior update | Evidence | Predictive |
| --- | --- | --- | --- | --- | --- |
| Bernoulli | Beta | successes, failures | `alpha += successes`, `beta += failures` | exact beta-function ratio | distribution-level predictive not yet materialized by the named updater |
| Binomial | Beta | total successes, total failures | same Beta update | exact binomial coefficients × beta-function ratio | not yet materialized |
| Poisson | Gamma | sum of counts, sample size | shape increases by count sum; rate increases by sample size | exact gamma/rate expression | not yet materialized |
| Categorical | Dirichlet | category counts | concentration vector plus counts | exact multibeta ratio | not yet materialized |
| Multinomial | Dirichlet | category counts | concentration vector plus counts | exact multinomial coefficients × multibeta ratio | not yet materialized |
| Normal | Normal-Inverse-Gamma | `n`, sample mean, centered sum of squares | exact NIG update | exact closed form | Student-t prior and posterior predictive |
| MultivariateNormal | Normal-Inverse-Wishart | sample mean/scatter algebra | exact NIW update | not yet implemented in the named wrapper | multivariate Student-t predictive |
| Generic `ExponentialFamily` | `NaturalConjugatePrior` | family-defined `T(x)` | `chi += sum(T(x))`, `nu += n` | exact when both conjugate normalizers are registered | one-step normalizer-ratio predictive when normalizers are registered |

## Example: Beta–Bernoulli

```python
import sympy as sp
from probstats import (
    Bernoulli,
    Beta,
)
from probstats.bayes import infer_conjugate

p = sp.Symbol("p", real=True)
result = infer_conjugate(Bernoulli(p), [1, 0, 1, 1], Beta(2, 3))
assert result.posterior == Beta(5, 4)
```

The sufficient-statistic representation means permutations of the observations produce
exactly the same posterior. Batch and sequential updates also agree algebraically.

## Example: Gamma–Poisson

For `Gamma(shape, scale)`, the update is most easily described in rate form. If
`rate0 = 1 / scale`, then after `n` observations with count sum `s`,

\[
\text{shape}_1=\text{shape}_0+s,\qquad
\text{rate}_1=\text{rate}_0+n.
\]

The returned posterior converts this back to the package's scale parameterization.

## Example: Normal–Inverse-Gamma

```python
from probstats import Normal
from probstats.bayes import NormalInverseGamma, infer_conjugate

prior = NormalInverseGamma(mu=0, lambda_=2, beta=3, nu=4)
result = infer_conjugate(Normal(0, 1), [1, 2, 4], prior)
predictive = result.metadata["posterior_predictive"]
```

The `Normal` object's numeric parameters are used to identify the likelihood family; the
unknown mean/variance algebra is represented by the NIG updater's sufficient statistics
and hyperparameters.

## Registry extension

Custom pairs can be added through `ConjugacyRule` and `ConjugacyRegistry`. A rule should
return a posterior plus enough metadata to state the number of observations and, where
available, evidence and predictive quantities. Named registry matching uses the stable
`probstats.algebra.ConjugacySignature`, with structural matching available for generic
natural-conjugate exponential families.
