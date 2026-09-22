# Derived and user-defined distributions

`probstats` can construct probability laws from other laws or from symbolic density definitions.

## User-defined distributions

`ProbabilityDistribution` defines a scalar continuous or discrete law from a SymPy expression and support. The constructor checks normalization when it can evaluate the total mass exactly. Optional CDF, quantile, and sampler hooks integrate the law with the ordinary `cdf`, `quantile`, and `sample` protocols.

```python
import sympy as sp
from probstats.distributions import ProbabilityDistribution

x, p = sp.symbols("x p", real=True)
d = ProbabilityDistribution(
    x,
    2 * x,
    sp.Interval(0, 1),
    cdf_expression=x**2,
    quantile_expression=sp.sqrt(p),
    probability_symbol=p,
)
```

Set `discrete=True` when the expression is a probability mass function. If symbolic normalization is outside SymPy's reach, `validate_normalization=False` disables the constructor check.

## Censoring

`CensoredDistribution(base, lower, upper)` represents the observed value

\[
Y=\min(\max(X,\ell),u).
\]

Censoring is distinct from truncation. A continuous base law develops point masses at finite censoring limits, so the resulting law uses a mixed reference measure. Exact probabilities, expectations, likelihood values, CDFs, and sampling preserve those atoms.

## Parameter mixtures

`ParameterMixtureDistribution` represents a hierarchical law

\[
p(x)=\int p(x\mid\theta)p(\theta)\,d\theta.
\]

The conditional family is supplied as a callable so the same construction works for symbolic integration and numerical hierarchical sampling.

```python
from probstats import (
    Gamma,
    Poisson,
)
from probstats.distributions import ParameterMixtureDistribution

mixed = ParameterMixtureDistribution(
    Gamma(2, 3),
    lambda rate: Poisson(rate),
)
```

Scalar conditional laws are currently supported. Exact density/mass, CDF, probability, and expectation calculations integrate over the mixing law.

## Spliced distributions

`SplicedDistribution` combines region-restricted component distributions with explicit segment weights. Each component is normalized on its assigned region before the segment weights are applied.

```python
from probstats import Normal
from probstats.distributions import SplicedDistribution

spliced = SplicedDistribution(
    [Normal(-1, 1), Normal(2, 1)],
    [sp.Interval.open(-sp.oo, 0), sp.Interval(0, sp.oo)],
    [sp.Rational(2, 5), sp.Rational(3, 5)],
)
```

Regions must be disjoint and components must share a scalar reference measure.
