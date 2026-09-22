# Random-variable algebra

`probstats.RandomVariable` is a SymPy symbol subtype for scalar random quantities. Ordinary symbolic composition remains ordinary SymPy algebra; statistical operators add only mathematically valid statistical rewrites.

```python
import sympy as sp
from probstats import RandomVariable, expectation

X = RandomVariable("X")
Y = RandomVariable("Y")
a, b = sp.symbols("a b")

expectation(a * X + b * Y + 2)
```

The result is linear in the expectations of `X` and `Y`. No distribution is required.

## Explicit assumptions

Relationships are supplied through immutable `StatisticalAssumptions` contexts. Queries remain three-valued: true, false, or unresolved.

```python
from probstats import StatisticalAssumptions, independent

ctx = StatisticalAssumptions(independent(X, Y))
independent(X, Y, assumptions=ctx)  # True
```

A product is not factored without the required evidence:

```python
expectation(X * Y)
expectation(X * Y, assumptions=ctx)
```

The first expression remains an unevaluated joint expectation. The second becomes the product of the marginal expectations.

For three or more variables, product factorization and higher-order cumulant additivity require a mutually `independent` relationship; pairwise `independent` assertions alone are not promoted to a stronger statement.

## Variance and covariance

Covariance is bilinear and symmetric, and variance expands sums through covariance terms:

```python
from probstats import covariance, variance

variance(2 * X + 3)
covariance(2 * X + 1, 5 * Y - 4)
variance(X + Y)
variance(X + Y, assumptions=ctx)
```

An asserted `independent(X, Y)` or `uncorrelated(X, Y)` is sufficient to reduce `covariance(X, Y)` to zero. The reverse implication is never inferred.

## Moments

Raw, central, and mixed moments use the same symbolic expectation engine:

```python
from probstats import central_moment, mixed_moment, moment, raw_moment

raw_moment(X, 3)
moment(2 * X + 1, 2)
central_moment(2 * X + 1, 3)
mixed_moment(X, Y, orders=(2, 3), assumptions=ctx)
```

Orders must be exact nonnegative integers. The identities

```text
raw_moment(X, 0) = 1
central_moment(X, 1) = 0
central_moment(X, 2) = variance(X)
```

are canonicalized directly.

## Cumulants

Cumulants preserve affine scaling and add over sums only when the required variables are certified `independent`:

```python
from probstats import cumulant

cumulant(2 * X + 3, 4)
cumulant(X + Y, 3, assumptions=ctx)
```

The first cumulant is expectation and the second cumulant is variance. Higher-order additivity is not inferred from pairwise relationships among three or more variables.

## Deterministic statistical summaries

An expression such as `expectation(X)` contains `X` syntactically but is a deterministic statistical quantity. The random-variable algebra therefore distinguishes random variables mentioned inside a statistical summary from stochastic inputs to an outer expression. This prevents nested simplification from treating `E[X]`, `Var(X)`, or a moment as another random variable.

## Attached laws and derived distributions

A `RandomVariable` may carry a probability law without requiring every symbolic variable to have one:

```python
from probstats import Normal, RandomVariable, distribution

X = RandomVariable("X", distribution=Normal(0, 1))
distribution(2 * X + 3)
```

Named closure rules are used when they are mathematically exact. Affine normal transformations remain normal; sums of certified-independent normal variables remain normal; certified-independent Poisson sums remain Poisson; and binomial sums with the same success probability remain binomial. When every source law is known but no named closure rule is available, `distribution(...)` returns a `RandomExpressionLaw` that records the exact symbolic construction. If a required source law is unknown, it returns `None` rather than inventing a distribution.

The declared law is part of a random variable's identity. This avoids ambiguity between two same-named symbolic variables that represent different stochastic quantities.

## Generating functions

Random expressions support MGF, CGF, characteristic-function, and PGF algebra:

```python
from probstats import (
    characteristic_function,
    cumulant_generating_function,
    moment_generating_function,
    probability_generating_function,
)

moment_generating_function(2 * X + 3)
cumulant_generating_function(2 * X + 3)
characteristic_function(2 * X + 3)
```

Affine transformations apply the exact substitution rules. Certified-independent sums factor MGFs, characteristic functions, and PGFs and add CGFs. As elsewhere in the random-variable algebra, three or more variables require mutual `independent` evidence rather than pairwise evidence alone.

When a variable has an attached distribution, the existing distribution transform engine supplies its closed form. Otherwise an unevaluated symbolic generating-function object is retained.

## Conditional expectation and variance

Conditional operators use explicit conditioning variables and the same assumptions context:

```python
from probstats import (
    conditional_expectation,
    conditional_variance,
    total_expectation,
    total_variance,
)

conditional_expectation(X, given=Y)
conditional_variance(X, given=Y)
total_expectation(X, given=Y)
total_variance(X, given=Y)
```

The implementation canonicalizes deterministic quantities, variables already present in the conditioning set, affine deterministic factors, and targets certified independent of the conditioning variables. A `conditionally_independent(X, Y, given=Z)` assertion permits the exact reduction

```text
E[X | Y, Z] = E[X | Z].
```

The tower identity is represented by `total_expectation`, while `total_variance(..., expanded=True)` returns the decomposition

```text
E[Var(X | Y)] + Var(E[X | Y]).
```

The default `total_variance(X, given=Y)` returns the mathematically equal canonical quantity `Var(X)`.

## Moment-Taylor approximations

Nonlinear functions of one random variable can be approximated directly from
central moments. `taylor_expectation` expands about the symbolic mean and keeps
terms through the requested order, while `taylor_variance` takes the variance
of the corresponding truncated Taylor polynomial.

```python
from probstats import delta_method, taylor_expectation, taylor_variance

taylor_expectation(sp.exp(X), order=4)
delta_method(sp.exp(X))
taylor_variance(X**2, order=2)
```

The first-order variance approximation is exposed as `delta_method`. Higher
orders retain central moments explicitly instead of substituting distributional
assumptions that were not supplied. The current Taylor API is by design
univariate; expressions involving more than one random variable must not be
silently reduced to a one-dimensional approximation.

## Complex random variables

Complex-valued random variables use explicit Hermitian covariance operations so
the established real-valued `covariance` API keeps its existing symmetric
bilinear semantics.

```python
from probstats import complex_covariance, complex_variance, pseudo_covariance

complex_covariance(X, Y)
complex_variance(X)
pseudo_covariance(X, Y)
```

`complex_covariance(X, Y)` represents
`E[(X - E[X]) * conjugate(Y - E[Y])]`; consequently it is linear in its first
argument and conjugate-linear in its second. Complex variance scales by squared
modulus, so `complex_variance(a * X)` becomes
`a * conjugate(a) * complex_variance(X)`. `pseudo_covariance` omits conjugation
and is therefore bilinear and symmetric. Certified `independent` expressions
have zero Hermitian covariance and zero pseudo-covariance. Expectation also
commutes with complex conjugation.


## Extended probability algebra and certification

The probability-algebra layer centralizes assumption normalization, stochastic dependency
inspection, collection independence, affine decomposition, and conditioning
measurability. Pairwise independence is never promoted to independence from a
joint conditioning set: simplifications such as `E[X | Y, Z] = E[X]` require
independence between the relevant collections or mutual independence of their
union.

Attached `RandomVariable.distribution` metadata is authoritative throughout the
statistical algebra. Known laws therefore evaluate means, variances, raw and
central moments, cumulants, and generating functions before the algebra falls
back to formal symbolic operators. Reproductive distribution rules are exposed
through affine and independent-sum closure registries rather than a monolithic
family switch.

The assumptions vocabulary additionally supports `identically_distributed`,
`independent_collections`, and finite-moment predicates. `iid(...)` entails both
mutual independence and identical distribution and rejects declarations whose
attached laws are already known to disagree.

Conditional algebra includes measurable-factor extraction, conditional moments,
conditional covariance, and the law of total covariance. The ordinary real
variance/covariance API applies scalar homogeneity only when deterministic
coefficients are certified real; complex-valued calculations use the Hermitian
`complex_variance` and `complex_covariance` APIs, whose covariance obeys
conjugate symmetry.

Standardized statistics include symbolic `correlation`, `standardized_moment`,
`skewness`, `kurtosis`, `coefficient_of_variation`, and `factorial_moment`.
First-class scalar events support complements, unions, and intersections, with
probability helpers for conditional probability, Bayes' rule, total probability,
inclusion-exclusion, and event-independence checks. Information-theoretic random
variable algebra adds joint and conditional entropy, mutual information, chain
rules, product-KL additivity, and explicit bijection-invariance theorem helpers.

Taylor approximations can return `TaylorApproximationResult` metadata recording
the expansion point, truncation order, and required moments. `delta_variance`
provides the first-order propagated variance directly, while `delta_method(...,
return_result=True)` returns the transformed center, derivative, asymptotic
variance, and implied normal limit law.

### Shared transformation certification

Mutual-information invariance, KL invariance for common pushforwards, and
transformed differential-entropy identities share the same internal
`funcprops` certification path. `probstats` owns the probability support and
the information-theoretic theorem; `funcprops` owns range, global bijectivity,
Jacobian, and local change-of-variables certification. An `UNKNOWN` mapping
result never authorizes an invariance simplification.
