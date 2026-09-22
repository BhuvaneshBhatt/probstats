# Results, namespaces, arrays, and tabular data

`probstats` separates numerical computation from presentation and labeling. NumPy and SymPy remain the core numerical and symbolic engines. Pandas is optional and is used at tabular API boundaries when row labels, column names, or DataFrame output improve the workflow.

## Namespaces

The package root contains only core distribution constructors, generic probability/evaluation functions, and major domain namespaces. Specialized workflows belong to `probstats.distributions`, `probstats.stats`, `probstats.smoothing`, `probstats.survival`, `probstats.information`, `probstats.random_matrix`, or `probstats.bayes`. This ownership is part of the public API.

## Result objects

Statistical results share a lightweight protocol:

```python
result.summary()
result.summary_data()
result.to_dict()
result.to_frame()
```

`summary()` is dependency-free. `to_frame()` requires the `tabular` extra. Regression, GLM, formula-model, and Cox results provide coefficient-oriented tables rather than flattening an entire fit into one row.

## Pandas interoperability

Install Pandas support with:

```bash
pip install "probstats[tabular]"
```

DataFrame/Series input is converted with `to_numpy(copy=False)` where possible. Names and indexes are retained separately from the dense arrays used by numerical routines. For example, direct regression records feature/response names and returns a Series when predicting from a DataFrame:

```python
import pandas as pd
from probstats.stats import linear_regression

x = pd.DataFrame({"x1": [1, 2, 3], "x2": [0, 1, 0]}, index=["a", "b", "c"])
y = pd.Series([2, 5, 6], index=x.index, name="response")
fit = linear_regression(x, y)

fit.feature_names
fit.coefficient_table()
fit.predict(pd.DataFrame({"x1": [4], "x2": [1]}, index=["new"]))
```

Formula models accept DataFrames directly and preserve prediction indexes. Cox regression also derives covariate names from DataFrame columns when names are not supplied explicitly.

Pandas is not used for matrix algebra, likelihood loops, smoothing kernels, random-matrix operations, or Monte Carlo inner loops. NumPy is faster and has more predictable broadcasting semantics for those operations.

## Array semantics

Scalar-event distributions accept scalar or array-like input for `pdf`, `logpdf`, `cdf`, `survival`, `hazard`, and quantiles. Array input preserves its sample shape. Multivariate and matrix-valued distributions treat their event dimensions as part of one observation rather than mistakenly mapping over coordinates.

```python
import numpy as np
from probstats import Normal

d = Normal(0, 1)
d.pdf(np.array([[-1.0, 0.0], [1.0, 2.0]])).shape
# (2, 2)
```

This convention also underlies the package sampling API: `size` describes sample dimensions, while vector or matrix event dimensions are appended by the distribution.
