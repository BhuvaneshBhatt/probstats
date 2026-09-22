# Bayesian inference tutorial

This tutorial follows a Gaussian location problem from a closed-form symbolic model to
approximate and sampling-oriented inference. The purpose is to show one model through the
common `probstats.bayes` interfaces rather than present unrelated API fragments.

## 1. Build the model once

Suppose

\[
\theta \sim \mathcal N(0,1), \qquad y\mid\theta \sim \mathcal N(\theta,1),
\]

and observe \(y=1\).

```python
import sympy as sp
from probstats import Normal
from probstats.bayes import Factor, Model, Parameter, RandomVariable

theta = Parameter("theta", sp.S.Reals)
y = RandomVariable("y", sp.S.Reals)
model = Model(
    (theta, y),
    (
        Factor.from_distribution(theta, Normal(0, 1)),
        Factor.from_distribution(y, Normal(theta.symbol, 1)),
    ),
).observe(y=1)
```

The model object is independent of an inference engine. Its symbolic joint density is
therefore available before choosing exact integration, Laplace approximation, or nested
sampling.

## 2. Exact symbolic posterior and evidence

```python
from probstats.bayes.exact import infer_exact

exact = infer_exact(model)
sp.simplify(exact.posterior.density)
sp.simplify(exact.log_evidence)
```

For this model the posterior is exactly

\[
\theta\mid y=1 \sim \mathcal N\!\left(\frac12,\frac12\right),
\]

where the second argument here denotes variance; the corresponding `Normal` scale is
\(1/\sqrt2\). The marginal likelihood is

\[
p(y=1)=\frac{e^{-1/4}}{2\sqrt\pi}.
\]

`infer_exact` records the integration route and normalization evidence in the returned
`InferenceResult`.

## 3. Automatic planning

```python
from probstats.bayes import plan_inference

plan = plan_inference(model)
print(plan.explain())
```

The planner ranks only applicable methods. For small symbolic models, direct exact
normalization is normally preferred. Increasing latent dimension or symbolic complexity
can make Laplace or nested sampling preferable; explicit conjugacy context can make a
registered closed-form update cheaper than generic integration.

## 4. Conjugate updating with unknown mean and variance

A richer Gaussian model may treat both mean and variance as unknown. `probstats.bayes`
represents the corresponding Normal-Inverse-Gamma conjugate prior directly:

```python
from probstats.bayes import NormalInverseGamma, infer_conjugate

mu = sp.Symbol("mu", real=True)
sigma = sp.Symbol("sigma", positive=True)
prior = NormalInverseGamma(mu=0, lambda_=2, beta=3, nu=4)
fit = infer_conjugate(Normal(mu, sigma), [1, 2, 4], prior)

fit.posterior
fit.metadata["posterior_predictive"]
fit.log_evidence
```

The posterior predictive is a Student-t distribution. Batch and sequential conjugate
updates are algebraically identical; this identity is exercised by the property-based
test suite.

## 5. Laplace approximation

For a smooth continuous posterior, Laplace inference expands the log posterior around
its mode and uses the inverse negative Hessian as a local covariance approximation.

```python
from probstats.bayes.laplace import infer_laplace

laplace = infer_laplace(model, backend="sympy", initial_guess=[0])
laplace.posterior.mean
laplace.posterior.covariance
laplace.log_evidence
```

Because this tutorial model has an exactly quadratic log posterior, Laplace inference
recovers the exact posterior mean, covariance, and evidence (up to numerical rounding in
an optimization backend). For nonquadratic models it is an approximation and its result
kind and provenance reflect that distinction.

## 6. Nested sampling when analytic structure is insufficient

Nested sampling requires a numerical likelihood and either a normalized-prior sampler or,
preferably, a unit-cube prior transform. It simultaneously estimates evidence and a
weighted empirical posterior.

```python
import numpy as np
from probstats.bayes.nested import infer_nested

result = infer_nested(
    lambda value: -0.5 * (1.0 - value[0]) ** 2,
    prior_sampler=lambda rng: np.array([rng.normal()]),
    ndim=1,
    n_live=100,
    rng=123,
)
```

Use nested sampling when exact/conjugate routes are unavailable and a global numerical
evidence calculation is needed. For production use with structured priors, prefer
`prior_transform` and the slice constrained sampler.

## 7. Choosing a result to trust

Exact and conjugate results carry `InferenceKind.EXACT`. Laplace results are approximate;
nested-sampling results are sampled numerical results with evidence uncertainty and
sampling diagnostics. Do not treat agreement between two approximate methods as a proof:
where an exact route exists, the test suite uses it as an independent mathematical oracle.

See [Inference method guide](bayesian-methods.md) and
[Conjugacy reference](conjugacy-reference.md) for the supported routes and closed-form
families.
