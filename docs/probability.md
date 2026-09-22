# Probability functionals

`probstats` exposes distribution-independent probability functionals alongside
methods on each `Distribution` instance. Closed forms supplied by a distribution
are used first; generic symbolic integration, summation, differentiation, or
algebra is used when the result can be represented exactly.

## Tails and hazards

For a scalar distribution `d`:

```python
from probstats import Exponential
from probstats.functionals import (
    cumulative_hazard,
    hazard,
    inverse_survival,
)

d = Exponential(2)
hazard(d, 3)
cumulative_hazard(d, 3)
inverse_survival(d, 0.25)
```

Continuous hazard is `f(x) / P(X > x)`. For discrete distributions the hazard
is `P(X=x | X>=x)`, so the current point mass is included in the denominator.
`cumulative_hazard(d, x)` uses `-log(P(X > x))`.

## Moments and cumulants

```python
from probstats import Normal
from probstats.functionals import cumulant

d = Normal(2, 3)
cumulant(d, 1)
cumulant(d, 2)
cumulant(d, 3)
```

Cumulants are differentiated from a closed, regular cumulant-generating
function when one is available. Otherwise they are computed from raw moments
using the triangular moment-cumulant recurrence, avoiding differentiation of
piecewise MGFs at convergence boundaries.

The symbolic transform API includes:

```python
from probstats.symbolic import (
    central_moment_generating_function,
    cumulant_generating_function,
    factorial_moment_generating_function,
)
```

The factorial-moment generating function is `E[(1+t)**X]`; its derivatives at
zero are falling-factorial moments.

## Factorial moments

```python
from probstats import Poisson
from probstats.functionals import factorial_moment

factorial_moment(Poisson(4), 3)
# 64
```

Factorial moments are defined for discrete distributions and use derivatives of
the PGF when available, with exact expectation as a fallback.

## Likelihood values

For a fully specified distribution, iid likelihoods remain symbolic:

```python
from probstats import Bernoulli
from probstats.functionals import log_likelihood
from probstats.stats import likelihood

p = ...
d = Bernoulli(p)
likelihood(d, [1, 0, 1])
log_likelihood(d, [1, 0, 1])
```

`likelihood()` also remains the constructor for parametric numerical inference
when passed a distribution class and parameter names:

```python
from probstats import Normal
from probstats.stats import likelihood

model = likelihood(Normal, data, ("mean", "sigma"))
```

This keeps exact likelihood evaluation and numerical parameter inference under
one public concept while preserving distinct result types.
