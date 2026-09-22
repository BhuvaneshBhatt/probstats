# Statistical inference

The inference API complements symbolic probability calculations with likelihood-based estimation, resampling, regression inference, and hypothesis testing. Numerical optimization is implemented internally with finite-difference BFGS; SciPy is not a required dependency. Tail probabilities and critical values use mpmath-based calculations where closed forms are not convenient.

## Generic likelihoods and numerical estimation

`likelihood(factory, data, parameter_names, fixed=...)` builds a `Likelihood` from a scalar distribution factory. `numerical_mle()` maximizes the log likelihood, and `numerical_map()` adds either a callable log prior or one prior distribution per free parameter.

Both return `OptimizationResult`, which records estimates, convergence state, iteration count, objective value, log likelihood, covariance, standard errors, and local information when available.

Automatic constraints can be requested with `constraints="auto"`. Scalar `RealSpace`, `PositiveRealSpace`, `NonnegativeRealSpace`, and `UnitIntervalSpace` parameters are mapped to unconstrained optimizer coordinates. Cross-parameter relations are checked on every candidate. Unsupported integer or structured spaces raise `NotImplementedError` instead of being silently relaxed.

Positive and nonnegative transforms parameterize the interior of their domains, so exact boundary optima are approached asymptotically rather than represented by finite unconstrained coordinates.

## Fisher information and standard errors

`observed_fisher_information()` uses the negative Hessian of the full-sample log likelihood. `expected_fisher_information()` estimates the expectation of one-observation information by Monte Carlo and scales it to the observed sample size. `estimator_covariance()` converts information to a covariance estimate, using a pseudoinverse only when explicitly required by the numerical linear algebra path.

These are local, asymptotic uncertainty summaries. Weak identification, singular information, or boundary parameters can make them unstable.

## Likelihood tests and profile likelihood

`likelihood_ratio_test()`, `wald_test()`, and `score_test()` return `HypothesisTestResult`. Their default calibration uses the standard asymptotic chi-square reference law.

`profile_likelihood()` fixes a parameter of interest over a grid and reoptimizes all nuisance parameters. `likelihood_confidence_region()` profiles nuisance parameters not included in the requested parameter vector and applies the corresponding likelihood-ratio cutoff.

Boundary problems, nonregular models, weak identification, and multimodality can invalidate ordinary chi-square calibration.

## Resampling

`bootstrap()` supports percentile, BCa, and bootstrap-t intervals. BCa uses bootstrap bias correction and jackknife acceleration. Bootstrap-t accepts a caller-supplied standard-error estimator or computes inner-bootstrap standard errors when none is provided.

`permutation_test()` enumerates all distinct label assignments when the exact permutation space is below the configured threshold. Larger problems use Monte Carlo permutations with a finite-simulation correction.

## Multiple testing

`adjust_pvalues()` preserves input order and returns adjusted p-values and rejection decisions. Supported methods include Bonferroni, Sidak, Holm, Holm-Sidak, Hochberg, Benjamini-Hochberg, and Benjamini-Yekutieli.

The procedures have different dependence assumptions; callers should select a method appropriate to the family of hypotheses being tested.

## Regression covariance

`robust_regression_covariance()` implements HC0, HC1, HC2, HC3, and HC4 sandwich covariance estimators for ordinary least squares. `RegressionResult.with_covariance()` returns a result with standard errors and coefficient tests recomputed from the selected covariance.

These covariance estimators address heteroskedasticity in the variance estimate; they do not make the underlying observations independent or repair model misspecification.

## Nonparametric tests

Mann-Whitney U and Wilcoxon signed-rank tests use exact conditional enumeration when the state space is small enough and `exact="auto"` or `exact=True` is used. Ties are retained through the observed averaged ranks. Larger problems use tie-corrected normal approximations.

`exact=True` raises when the required enumeration exceeds the configured threshold rather than silently changing method.

## Goodness of fit

`kolmogorov_smirnov_test(data, distribution)` performs a one-sample KS test against a fully specified continuous distribution. When a family is fitted from the same data, the fitted-model form uses a parametric bootstrap: every simulated replicate is refitted before its KS statistic is calculated. This avoids applying the fully specified-null KS reference distribution after parameter estimation.

## Inference assumptions and failure-mode reference

| Workflow | Assumptions | Returned quantities to inspect | Important failure modes |
| --- | --- | --- | --- |
| Closed-form MLE/MoM | family-specific identifiability and parameter domain | fitted distribution/parameters, log likelihood, information criteria | unsupported family, invalid sample/domain |
| Numerical MLE/MAP | sufficiently regular objective and identifiable parameterization | convergence flag, iterations, objective/log likelihood, covariance/information | nonconvergence, invalid transformed candidate, weak/singular curvature, local optimum |
| Fisher information | local differentiability and regular asymptotic regime | information matrix, inverse/pseudoinverse covariance, standard errors | singular/indefinite matrix, boundary or weak identification |
| LR/Wald/score tests | regular nested-model asymptotics | statistic, degrees of freedom, p-value, estimates/SEs | boundary/nonregular null, weak identification, poor quadratic approximation |
| Profile likelihood | meaningful grid and successful nuisance reoptimization | profile values, optimizer status at each point, LR cutoff | local optima, failed nuisance fits, disconnected confidence set |
| Bootstrap | resampling scheme represents the data-generating design | interval type, replicates, standard-error estimates | too few replicates, degenerate resamples, invalid iid resampling assumption |
| Permutation | exchangeability under the null | exact/enumerated versus Monte Carlo method, permutation count | nonexchangeable design, too-small Monte Carlo budget |
| Robust covariance | correctly specified estimating equation/mean model; independent clusters unless a clustered method is used | covariance variant and recomputed coefficient tests | leverage instability, dependence not handled by HC estimators |
| Exact rank tests | exchangeability/symmetry assumptions of the selected test | exact/approximate method and statistic | enumeration threshold exceeded when `exact=True`, ties/zeros outside supported exact state space |
| Fitted-model KS | bootstrap model adequately represents the fitted null | refitted replicate distribution and Monte Carlo p-value | failed replicate fits, insufficient bootstrap replications |

See [Failure semantics](failure-semantics.md) for the package-wide distinction between invalid input, unsupported operations, unavailable optional backends, backend failure, nonconvergence, and unresolved symbolic truth.
