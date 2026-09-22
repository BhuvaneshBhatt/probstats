# Formula models, generalized linear models, and ANOVA

Ordinary least squares, generalized linear models, ANCOVA, and fixed-effects factorial ANOVA share one formula parser and design-matrix compiler. A compiled `DesignMatrix` retains individual columns, higher-level model terms, categorical encodings, interactions, transformed predictors, and the fitted factor levels needed for prediction on new data.

## Formula grammar

`design_matrix(formula, data)` accepts a mapping of column names to equally long arrays or a Pandas DataFrame. DataFrame column names and prediction indexes are preserved by fitted formula models. Supported syntax includes:

- `y ~ x + z` for main effects;
- `y ~ x:z` for an explicit interaction;
- `y ~ x * z` for hierarchical main effects plus interaction;
- `0` or `-1` to remove the intercept;
- `C(group)` to mark a predictor categorical explicitly;
- `log(x)`, `exp(x)`, and `sqrt(x)` numeric transforms;
- `I(x**k)` for an integer power;
- `poly(x, d)` for the raw basis `x, x**2, ..., x**d`.

String, object, and Boolean columns are treated as categorical automatically. Treatment/reference coding and sum-to-zero coding are available through `TreatmentContrast`, `SumContrast`, or the `contrasts=` argument.

## Linear models

`linear_model(formula, data, weights=...)` compiles the formula and delegates numerical fitting to the ordinary least-squares engine. Positive weights produce weighted least squares. `FormulaRegressionResult` retains coefficient names and the fitted design specification while exposing coefficient covariance, standard errors, t tests, fitted values, residuals, R-squared measures, and the overall F test. `coefficient_table()` returns a labeled Pandas table when Pandas is installed, and prediction from a DataFrame returns a Series with the input index.

HC0-HC4 heteroskedasticity-consistent covariance estimators are available through the regression-inference API.

## Generalized linear models

`glm(formula, data, family=..., link=..., weights=..., offset=..., trials=...)` uses iteratively reweighted least squares with domain checks and step halving.

Built-in families include:

- Gaussian;
- binomial;
- Poisson;
- Gamma;
- Negative Binomial NB2 with supplied `alpha`;
- quasi-Poisson;
- quasi-binomial.

Built-in links are identity, log, logit, probit, and inverse. Offsets enter the linear predictor with fixed coefficient one. Observation/frequency weights feed both fitting and inference. For grouped binomial data, pass successes as the response and trial counts through `trials=`; trial counts scale information, deviance, and the grouped-binomial likelihood.

Quasi families estimate dispersion from Pearson residuals and do not report ordinary log likelihood or AIC because they do not define a full probability likelihood.

`GLMResult` records coefficients, covariance, standard errors, coefficient statistics, fitted means, linear predictors, residuals, deviance, null deviance, dispersion, likelihood information when defined, and convergence diagnostics.

## Coefficient hypotheses

`coefficient_contrast()` tests a scalar linear combination of coefficients. Contrasts may be supplied as coefficient-name mappings or numeric vectors. Ordinary least squares uses a Student-t reference distribution; GLMs use a large-sample normal reference.

`joint_wald_test()` tests several linear restrictions at once using the Wald chi-square statistic.

## Estimated marginal means

`estimated_marginal_means()` builds a balanced reference grid. The requested categorical factor varies across its fitted levels, other categorical factors are averaged equally over their combinations, and continuous predictors are fixed at their training means.

For GLMs, `scale="response"` applies the inverse link and delta-method standard errors. `scale="linear"` retains the linear-predictor scale.

## ANOVA and ANCOVA

`factorial_anova()` uses the same compiled model terms as the regression and GLM layers. Type I, II, and III sums of squares are supported for fixed-effects models.

- Type I tests terms sequentially and therefore depends on formula order.
- Type II tests a term after terms that do not contain it while excluding higher-order interactions containing that term.
- Type III compares the full model with the model obtained by dropping only the tested term.

Sum-to-zero contrasts are recommended for Type III inference, particularly in unbalanced designs.

`ancova()` uses the same term-comparison machinery and validates that the model contains both numeric and categorical predictors. Interactions such as `baseline * C(treatment)` can be used to test heterogeneous slopes.

## Model comparison

`compare_models()` expects nested models ordered from reduced to fuller. Ordinary least-squares models use partial F tests, full-likelihood GLMs use deviance chi-square tests, and quasi-GLMs use scaled-deviance F comparisons.

## Scope

The current model layer is fixed-effects only. Mixed effects, repeated-measures covariance, random effects, spline bases, cluster-robust GLM covariance, and joint estimation of Negative Binomial dispersion are not yet implemented.
