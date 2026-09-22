# Survival analysis

`probstats` provides a nonparametric survival-analysis core for right-censored
observations, with optional delayed entry (left truncation). The estimators and
tests share one risk-set implementation so censoring, ties, and entry semantics
remain consistent across the API.

## Survival data

```python
from probstats.survival import SurvivalData

data = SurvivalData(
    time=[2, 4, 5, 7, 9],
    event=[1, 1, 0, 1, 0],
)
```

`event=True` denotes an observed failure and `event=False` denotes right
censoring. With delayed entry, supply an `entry` time for every observation:

```python
data = SurvivalData(
    time=[4, 6, 8],
    event=[1, 0, 1],
    entry=[0, 2, 3],
)
```

An observation is in the risk set at time `t` when `entry <= t <= time`. Events
and censoring at the same exit time are therefore both included in that time's
risk set; failures are applied to the estimator before subjects censored at that
same time leave the risk set.

`SurvivalData.from_event_censor_times(events, censored)` is a convenience
constructor when event and censoring times are stored separately.

## Risk tables

```python
from probstats.survival import risk_table

table = risk_table(data)
table.time
table.at_risk
table.events
table.censored
```

The table is evaluated at every distinct observed exit time and includes delayed
entry when calculating `at_risk`.

## Kaplan-Meier

```python
from probstats.survival import kaplan_meier

fit = kaplan_meier(data, confidence_level=0.95)
fit.event_times
fit.survival
fit.standard_error
fit.lower, fit.upper
fit.median_survival
fit.survival_at(5.0)
```

At each failure time, the product-limit estimator is

\[
\widehat S(t)=\prod_{t_i\le t}\left(1-\frac{d_i}{n_i}\right),
\]

where `n_i` is the number at risk and `d_i` the number of failures. Greenwood's
variance is

\[
\widehat{\operatorname{Var}}\{\widehat S(t)\}
=\widehat S(t)^2
\sum_{t_i\le t}\frac{d_i}{n_i(n_i-d_i)}.
\]

Confidence intervals use the log-minus-log transform, which respects the
`[0, 1]` range. The Kaplan-Meier median is the earliest failure time at which
the estimated survival is at most one half; it is `math.inf` when the fitted
curve never reaches one half.

## Nelson-Aalen

```python
from probstats.survival import nelson_aalen

fit = nelson_aalen(data)
fit.cumulative_hazard
fit.cumulative_hazard_at(5.0)
fit.survival_at(5.0)
```

The cumulative-hazard estimator is

\[
\widehat H(t)=\sum_{t_i\le t}\frac{d_i}{n_i},
\]

with counting-process variance estimate `sum(d_i / n_i**2)`. The corresponding
survival approximation is `exp(-H(t))`.

## Comparing survival curves

The default `logrank_test` is the Mantel-Haenszel log-rank test. Two or more
`SurvivalData` objects can be compared directly:

```python
from probstats.survival import logrank_test

result = logrank_test(treatment, control)
result.statistic
result.pvalue
```

A single sample plus group labels is also accepted:

```python
result = logrank_test(data, groups=labels)
```

Weighted members of the log-rank family are available through either `method=`
or named wrappers:

```python
from probstats.survival import (
    breslow_test,
    fleming_harrington_test,
    tarone_ware_test,
)

breslow_test(treatment, control)
tarone_ware_test(treatment, control)
fleming_harrington_test(treatment, control, p=1, q=0)
```

The weights are:

- log-rank: `1`;
- Breslow/generalized Wilcoxon: `n_i`;
- Tarone-Ware: `sqrt(n_i)`;
- Fleming-Harrington: `S_pool(t_i-)**p * (1-S_pool(t_i-))**q`.

`Fleming-Harrington(0, 0)` is exactly the ordinary log-rank test. The test
statistic uses the tied-event hypergeometric covariance and has asymptotic
chi-square degrees of freedom equal to `number_of_groups - 1`.

## Scope

This module is the nonparametric survival core. Cox proportional
hazards, parametric survival regression, residual diagnostics, and tie handling
inside regression partial likelihood belong to the subsequent survival-modeling
layer rather than being hidden inside these estimators.

## Survival regression

`probstats` provides semiparametric Cox proportional-hazards regression and four
parametric accelerated-failure-time (AFT) models. All regression routines use
`SurvivalData`, so right censoring and delayed entry use the same conventions as
Kaplan-Meier and Nelson-Aalen estimation.

### Cox proportional hazards

```python
from probstats.survival import SurvivalData
from probstats.survival_regression import cox_ph

data = SurvivalData(time, event, entry=entry)
fit = cox_ph(data, X, ties="efron", covariate_names=["age", "treatment"])

fit.coefficients
fit.hazard_ratio
fit.standard_error
fit.pvalue
fit.baseline_cum_hazard
fit.baseline_survival
```

The model is

\[
h(t\mid x)=h_0(t)\exp(x^T\beta).
\]

Both Breslow and Efron approximations are available for tied failures. Efron is
the default. `cox_partial_log_likelihood(...)` evaluates the same objective at
fixed coefficients. Coefficient covariance is the inverse observed information
from the partial likelihood. Baseline cumulative hazard uses the usual Breslow
estimator after fitting, including for coefficients fitted with Efron ties.

`CoxPHResult` also provides subject-specific cumulative hazard and survival,
Schoenfeld residuals, martingale residuals, deviance residuals, and a
scaled-Schoenfeld proportional-hazards diagnostic. The diagnostic supports
`rank`, `identity`, `log`, and Kaplan-Meier (`km`) transforms of event time and
reports both per-covariate and global chi-square tests.

### Parametric AFT regression

```python
from probstats.survival_regression import (
    exponential_regression,
    weibull_regression,
    lognormal_regression,
    loglogistic_regression,
)

fit = weibull_regression(data, X)
fit.coefficients
fit.ancillary  # Weibull shape
fit.survival(5.0, x)
fit.median(x)
```

The four models use a common AFT convention. For exponential, Weibull, and
log-logistic models, `exp(x @ beta)` is the scale. For log-normal regression,
`x @ beta` is the Normal location of `log(T)`. Weibull and log-logistic fit a
positive shape parameter, and log-normal fits a positive `sigma`; optimization
is performed on its logarithm.

The right-censored likelihood contribution is `log f(t)` for an event and
`log S(t)` for a censored observation. With delayed entry at `a`, every
contribution is additionally conditioned on survival to entry by subtracting
`log S(a)`. Standard errors and Wald intervals use the observed Hessian of the
full censored-data log likelihood.
