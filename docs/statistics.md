# Empirical, descriptive, and classical statistics

## Descriptive data API

Distribution-level functionals (`mean(dist)`, `variance(dist)`) remain unchanged.
For raw samples use `mean`, `median`, `quantile`, `variance`, and `standard_deviation`.

`describe(data)` returns a `DescriptiveSummary` with count, mean, sample standard
deviation, minimum, quartiles, maximum, IQR, MAD, skewness, and excess kurtosis.
`EmpiricalDistribution` provides an ECDF, empirical quantiles, moments, and
bootstrap-style resampling from the observed empirical measure.

## Estimation

`maximum_likelihood(family, data)` and `method_of_moments(family, data)` return
`EstimationResult`.  The result records the fitted `probstats` distribution,
parameter mapping, log likelihood, AIC, BIC, sample size, method, and convergence
flag.  Closed-form estimators are used where the package can certify them; an
unsupported family raises `NotImplementedError` instead of invoking an opaque
numerical optimizer.

`mean_confidence_interval` returns Student-t intervals when sigma is unknown and
normal intervals when sigma is supplied.  `proportion_confidence_interval` uses
the Wilson score interval.

## Hypothesis testing

All tests return `HypothesisTestResult`, with statistic, p-value, alternative,
method, sample sizes, null value, degrees of freedom, estimate, and standard
error where applicable.  `reject_05` is a convenience property only; callers
should choose their own significance threshold when appropriate.

Current procedures:

- `one_sample_t_test`
- `welch_t_test`
- `paired_t_test`
- `one_sample_z_test`
- `one_proportion_z_test`
- `two_proportion_z_test`
- `exact_binomial_test`
- `chi_square_goodness_of_fit`
- `chi_square_independence`
- `pearson_correlation_test`

These procedures implement their standard textbook assumptions; they do not
silently test normality, equal variance, independence, or minimum expected-count
conditions on the user's behalf.

## Dependence measures and rank-correlation tests

The classical testing layer includes Pearson, Spearman, Kendall tau-b, distance
correlation, and Cramer's V. Spearman uses average ranks for ties. Kendall uses
tau-b, so ties in either variable are reflected in the denominator, and its test
uses the large-sample tie-corrected variance of the Kendall score.

```python
from probstats.stats import (
    spearman_correlation_test,
    kendall_tau_test,
)
from probstats.testing import (
    distance_correlation_test,
    cramers_v,
)

spearman_correlation_test(x, y)
kendall_tau_test(x, y)
distance_correlation_test(x, y, permutations=999, rng=0)
cramers_v(contingency_table)
```

Distance correlation accepts scalar or vector-valued observations and can detect
nonlinear dependence that ordinary linear correlation misses. Its hypothesis test
is permutation based and therefore makes the approximation explicit through the
requested number of permutations and random-number generator.

## Variance and scale tests

`one_sample_variance_test` implements the Normal-theory chi-square variance test,
and `variance_ratio_test` implements the classical two-sample F test. For several
groups, `bartlett_test` provides the Normal-sensitive likelihood-ratio-style test,
while `levene_test` is more robust. The default Levene center is the median, which
is the Brown-Forsythe test; `brown_forsythe_test` names that test directly.

```python
from probstats.stats import bartlett_test
from probstats.testing import (
    brown_forsythe_test,
    variance_ratio_test,
)

variance_ratio_test(group_a, group_b)
bartlett_test(group_a, group_b, group_c)
brown_forsythe_test(group_a, group_b, group_c)
```

Use Bartlett when approximate Normality is scientifically defensible and
Brown-Forsythe when robustness to non-Normal tails is more important.

## Equivalence and noninferiority

Mean equivalence is implemented with the two-one-sided-tests (TOST) procedure.
`one_sample_tost` and `welch_tost` return an `EquivalenceTestResult` containing
both one-sided statistics and p-values; its combined `pvalue` is their maximum.
The null is rejected only when both one-sided tests reject.

```python
from probstats.testing import (
    one_sample_tost,
    welch_tost,
    noninferiority_t_test,
)

one_sample_tost(data, lower=-0.2, upper=0.2, mu=0)
welch_tost(treatment, control, lower=-0.2, upper=0.2)
noninferiority_t_test(treatment, control, margin=0.1, direction="greater")
```

`two_proportion_tost` and `proportion_noninferiority_z_test` provide the analogous
large-sample procedures for differences of independent proportions.

The sign convention for noninferiority is explicit. With `direction="greater"`,
the null boundary is `-margin`, so the test asks whether the estimand is greater
than the allowed loss. With `direction="less"`, the boundary is `+margin`.

## Autocorrelation tests

`autocorrelation` computes the conventional sample autocorrelation. `box_pierce_test`
and `ljung_box_test` test a collection of lags jointly against white noise; the
latter applies the finite-sample Ljung-Box correction. `durbin_watson` returns the
standard first-order residual-autocorrelation diagnostic.

```python
from probstats.testing import (
    autocorrelation,
    ljung_box_test,
    durbin_watson,
)

autocorrelation(series, lag=1)
ljung_box_test(series, lags=10)
durbin_watson(regression_residuals)
```

These portmanteau p-values use the usual chi-square asymptotic reference. If model
parameters have been estimated from the same series, callers should choose the
lag degrees of freedom with that modeling context in mind rather than interpreting
the default result as an exact finite-sample test.

## Multivariate hypothesis tests

`hotelling_t2_test` tests a single multivariate mean vector and
`two_sample_hotelling_t2_test` compares two multivariate means under the common
covariance assumption. Both use their exact finite-sample F transformation when
the covariance matrix is nonsingular and the required sample-size conditions hold.

`box_m_test` provides Box's M chi-square approximation for equality of covariance
matrices across groups. `mardia_normality_test` reports Mardia's multivariate
skewness and kurtosis diagnostics separately, including the skewness chi-square
and kurtosis Normal-reference p-values.

```python
from probstats.testing import (
    hotelling_t2_test,
    two_sample_hotelling_t2_test,
    box_m_test,
    mardia_normality_test,
)

hotelling_t2_test(X, mean=[0, 0])
two_sample_hotelling_t2_test(X1, X2)
box_m_test(X1, X2, X3)
mardia_normality_test(X)
```

For one-dimensional input, the one-sample Hotelling statistic reduces exactly to
the square of the ordinary one-sample Student-t statistic; this identity is used
as a regression test.

## Assumptions and failure-mode reference

| Procedure family | Null/reference quantity | Principal assumptions | Exact/asymptotic status | Edge cases and failure modes |
| --- | --- | --- | --- | --- |
| One-/two-sample t procedures | mean or mean difference | independent observations within the design; Normality for exact small-sample reference, with Welch avoiding equal-variance assumption | exact under Normal model; otherwise often asymptotic/robust in large samples | zero variance, too-small sample, nonfinite data |
| Paired t | mean paired difference | meaningful pairing; independent pairs; approximately Normal differences for exact small-sample reference | exact under Normal difference model | unequal pair lengths, zero variance of differences |
| Proportion z tests | one/two proportions | independent Bernoulli trials and adequate Normal approximation | asymptotic | sparse counts; use exact binomial where appropriate |
| Exact binomial | Bernoulli success probability | independent Bernoulli trials with fixed null probability | exact finite-sample | invalid counts/probability |
| Chi-square GOF/independence | category probabilities/independence | independent counts and adequate expected counts for chi-square calibration | asymptotic | zero/very small expected counts, incompatible margins |
| Pearson correlation | zero linear correlation | paired independent observations; bivariate Normal model for exact classical calibration | exact under bivariate Normal null; otherwise large-sample interpretation | constant vectors/nonfinite data |
| Spearman/Kendall | rank association | independent paired observations; tie handling as documented | exact/approximation depends on procedure and sample structure | all ties/degenerate ranks |
| Variance ratio/Bartlett | equal Normal-theory variances | independence and Normality | exact F for two-sample variance ratio; Bartlett asymptotic | zero variance, strong non-Normality |
| Brown-Forsythe/Levene | equal scale | independent groups; robustness depends on center choice | asymptotic F-style calibration | groups too small/degenerate |
| TOST/noninferiority | effect outside equivalence/noninferiority region | assumptions of underlying t/z procedure plus margins chosen before inference | same calibration as component one-sided tests | invalid/reversed margins, sign-convention mistakes |
| Ljung-Box/Box-Pierce | zero autocorrelations through chosen lags | weak stationarity/appropriate residual context | asymptotic chi-square | excessive lags, degrees-of-freedom not adjusted after model fitting |
| Hotelling T-squared | multivariate mean vector/difference | multivariate Normality and nonsingular covariance; common covariance for two-sample form | exact F transformation under assumptions | singular covariance, dimension too large for sample size |
| Box's M | covariance equality | independent multivariate Normal groups | chi-square approximation | singular covariances, sensitivity to non-Normality |
| Mardia diagnostics | multivariate Normal shape | independent multivariate observations and nonsingular covariance | asymptotic references | singular covariance, small sample/high dimension |

Every test result reports the method/calibration it actually used where the result type supports that metadata. Tests do not silently diagnose these assumptions on the caller's behalf.
