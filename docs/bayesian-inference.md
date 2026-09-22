# Bayesian inference

Bayesian inference is part of `probstats` under the `probstats.bayes` namespace.
The probability and distribution layers remain independent of Bayesian inference;
the Bayesian layer depends on those lower-level primitives, not the reverse.

```python
import sympy as sp
from probstats import (
    Beta,
    Binomial,
)
from probstats.bayes import conjugate_update

prior = Beta(2, 3)
likelihood = Binomial(10, sp.Symbol("p", positive=True))
```

## Subsystems

- `probstats.bayes.core`: variables, factors, models, observations, and inference results.
- `probstats.bayes.conjugacy`: exact conjugate-family recognition and updates.
- `probstats.bayes.exact`: exact symbolic marginalization, evidence, and posterior normalization.
- `probstats.bayes.laplace`: symbolic/numerical Laplace approximations and evidence.
- `probstats.bayes.nested`: nested sampling and empirical posterior construction.
- `probstats.bayes.models`: higher-level Bayesian models such as Bayesian linear regression.
- `probstats.bayes.gp`: Gaussian-process regression and predictive distributions.
- `probstats.bayes.neural`: regression-network fitting and evidence utilities.
- `probstats.bayes.reasoning` and `probstats.bayes.certification`: proof-oriented posterior geometry and expression certification.
- `probstats.bayes.diagnostics`: sampled-posterior diagnostics and ArviZ conversion.
- `probstats.bayes.planner`: applicability/cost-based inference-method selection.

The Bayesian namespace is not star-exported from `probstats`. This avoids
collisions between general statistics objects and Bayesian implementation objects with
similar names, while keeping shared distributions available from the package root.

## Prediction and model comparison

Fitted Bayesian models use one predictive vocabulary:

```python
fit.predict(x)
fit.predict_distribution(x)
fit.posterior_predictive(x, size=1000, rng=123)
fit.predictive_interval(x, level=0.95)
```

Conjugate regression and Gaussian-process fits expose exact predictive laws where available. For sampled `InferenceResult` objects, the same interface accepts a model-specific `predictive(draw, x, rng=...)` callback and represents the resulting Monte Carlo law as `EmpiricalPredictive`.

`compare_models()` accepts fitted models that provide log marginal likelihoods. `BayesianModelComparisonResult` reports log evidence, posterior model probabilities, pairwise log Bayes factors, `bayes_factor()`, `posterior_probability()`, and the highest-probability `best_model`.

## Optional integrations

The base Bayesian model and conjugacy APIs require only the normal `probstats`
dependencies. Optional extras enable SciPy Laplace optimization, `symbopt`,
`asymptotic`, exact/certified reasoning backends, ArviZ, and JAX.

Start with the [Bayesian tutorial](bayesian-tutorial.md), then see the
[inference method guide](bayesian-methods.md) and [conjugacy reference](conjugacy-reference.md).
Detailed API notes in `docs/api/` cover exact inference, Laplace inference, certification,
and reasoning.


For general posterior sampling, see [General MCMC framework](mcmc.md).
