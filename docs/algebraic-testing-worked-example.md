# Worked semialgebraic test

This example shows how the objects in `probstats.algebraic.testing` fit together. The purpose is to expose the calculation, not to hide it behind a single convenience call.

Suppose a scalar parameter `theta` is constrained by

\[
H_0:\quad \theta^2 - 1 \le 0,
\]

and one observation `X` is an unbiased estimator of `theta`. The constraint has degree two, so the default kernel has order two. Before symmetrization its quadratic term uses independent observations; exact permutation averaging produces the symmetric kernel used by the U-statistic.

```python
import sympy as sp
from probstats.algebraic.testing import SemialgebraicHypothesis, sdl_test

theta = sp.Symbol("theta")
hypothesis = SemialgebraicHypothesis.basic(
    (theta,),
    inequalities=(theta**2 - 1,),
    estimators={theta: lambda x: x},
)

result = sdl_test(
    [-0.8, -0.2, 0.1, 0.4, 0.7, 0.9],
    hypothesis,
    budget=10,
    bootstrap_replicates=1000,
    seed=7,
)

result.statistic
result.p_value
result.studentization.variance
```

The computation can be read as

\[
\widehat\sigma_j^2
= m^2\widehat\sigma_{g,j}^2
+ \alpha_n\widehat\sigma_{h,j}^2,
\qquad
\alpha_n=\frac{n}{N},
\]

followed by

\[
T=\max_j \frac{\sqrt n\,U'_{n,N,j}}{\widehat\sigma_j}.
\]

The multiplier bootstrap combines its two centered Gaussian processes as

\[
U^\# = mU_g^\# + \sqrt{\alpha_n}\,U_h^\#,
\]

and the reported Monte Carlo p-value is the fraction of bootstrap maxima at least as large as `T`.

## Computational scaling

| Quantity | Main scaling | Practical implication |
| --- | --- | --- |
| Exact kernel symmetrization | `m!` evaluations | Factorial time; permutations are generated lazily, so construction does not store `m!` tuples |
| Complete U-statistic | `C(n, m)` kernel evaluations | Useful for small problems and reference checks |
| Incomplete U-statistic | about `N` evaluations | Main computational control for SDL |
| Hájek projection | controlled by `n1` and available blocks | Larger `n1` uses more observations in variance estimation |
| Multiplier bootstrap | `A` replicates | Monte Carlo resolution and cost grow with `A` |
| Constraint count | approximately linear in coordinates per kernel evaluation | Redundant augmentation can improve geometry while increasing computation |

## Interpretation and limitations

Exact and random kernel symmetrization are not interchangeable mathematical guarantees. Random permutation averaging is a computational approximation and is fixed once the kernel is constructed. Bernoulli incomplete-U selection can realize zero subsets; `probstats` reports that event instead of resampling because resampling would change the selection law. With `A` bootstrap replicates the empirical p-value uses the SDL exceedance fraction and can equal zero.

For finite-union nulls use `sdl_union_test()`, which performs the component tests and combines them by the intersection-union rule. Reference trinomial helpers reproduce the published model descriptions and common simulation settings; they are fixtures for methodological comparison, not a promise of bit-for-bit agreement with another implementation.
