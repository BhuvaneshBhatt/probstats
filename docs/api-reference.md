# Generated public API reference
This file is generated from each namespace's `__all__`. Run `python tools/generate_api_reference.py` after public-API changes.

## Package root
| Name | Kind | Signature | Summary |
| --- | --- | --- | --- |
| `Bernoulli` | class | `(p: 'sp.Expr') -> None` | Bernoulli distribution on ``{0, 1}`` with success probability ``p``. |
| `Beta` | class | `(alpha: 'sp.Expr', beta: 'sp.Expr') -> None` | Beta distribution with positive shape parameters ``alpha`` and ``beta``. |
| `Binomial` | class | `(n: 'sp.Expr', p: 'sp.Expr') -> None` | Binomial distribution with ``n`` trials and success probability ``p``. |
| `DeltaMethodResult` | class | `(transformed_center: 'sp.Expr', derivative: 'sp.Expr', asymptotic_variance: 'sp.Expr', limit_distribution: 'object') -> None` | DeltaMethodResult(transformed_center: 'sp.Expr', derivative: 'sp.Expr', asymptotic_variance: 'sp.Expr', limit_distribution: 'object') |
| `Distribution` | class | `()` | Helper class that provides a standard way to create an ABC using |
| `Exponential` | class | `(rate: 'sp.Expr') -> None` | Exponential distribution with rate ``rate``. |
| `Gamma` | class | `(shape: 'sp.Expr', scale: 'sp.Expr') -> None` | Gamma distribution with shape ``shape`` and scale ``scale``. |
| `JointDistribution` | class | `(variables: 'tuple[sp.Symbol, ...]', joint_density: 'sp.Expr', supports: 'tuple[sp.Set, ...]', discrete: 'bool' = False, normalized: 'bool' = True) -> None` | Exact joint law defined by a density/ mass expression and named coordinates. |
| `MultivariateNormal` | class | `(mean: 'Sequence[Any]', covariance: 'Any')` | Multivariate normal distribution with mean vector and covariance matrix. |
| `Normal` | class | `(mean: 'sp.Expr', sigma: 'sp.Expr') -> None` | Univariate normal distribution with mean ``mean`` and std. dev. ``sigma``. |
| `Poisson` | class | `(rate: 'sp.Expr') -> None` | Poisson distribution with positive rate ``rate``. |
| `RandomExpressionLaw` | class | `(expression: 'sp.Expr', component_laws: 'tuple[tuple[RandomVariable, Distribution], ...]', exact: 'bool' = True) -> None` | RandomExpressionLaw(expression: 'sp.Expr', component_laws: 'tuple[tuple[RandomVariable, Distribution], ...]', exact: 'bool' = True) |
| `RandomVariable` | class | `(name, distribution=None, **assumptions)` | Symbol class is used to create symbolic variables. |
| `StatisticalAssumptions` | class | `(*relations)` | Immutable collection of asserted statistical propositions. |
| `StudentT` | class | `(location: 'sp.Expr', scale: 'sp.Expr', df: 'sp.Expr') -> None` | StudentT(location: 'sp.Expr', scale: 'sp.Expr', df: 'sp.Expr') |
| `TaylorApproximationResult` | class | `(expression: 'sp.Expr', order: 'int', expansion_point: 'sp.Expr', required_moments: 'tuple[int, ...]', exact: 'bool') -> None` | TaylorApproximationResult(expression: 'sp.Expr', order: 'int', expansion_point: 'sp.Expr', required_moments: 'tuple[int, ...]', exact: 'bool') |
| `Uniform` | class | `(low: 'sp.Expr', high: 'sp.Expr') -> None` | Uniform(low: 'sp.Expr', high: 'sp.Expr') |
| `__version__` | object | `` | str(object='') -> str |
| `algebraic` | module | `` | Algebraic statistics for polynomial probability models. |
| `bayes` | module | `` | High-level Bayesian modeling and inference workflows. |
| `bayes_probability` | function | `(dist, event, *, given, variable=None)` | Evaluate Bayes' rule for two events under ``dist``. |
| `cdf` | function | `(dist, value)` | Cumulative distribution function, preserving array-like input shape. |
| `central_moment` | function | `(dist, order: 'int', *, assumptions=None)` | Return a central moment for a distribution or symbolic random expression. |
| `characteristic_function` | function | `(expr, t=None, *, assumptions=None)` | Return the characteristic function of a distribution or random expression. |
| `coefficient_of_variation` | function | `(expression, *, assumptions=None)` | Return standard deviation divided by mean. |
| `complex_covariance` | function | `(left, right, *, assumptions=None)` | Normalize Hermitian covariance for complex-valued random expressions. |
| `complex_variance` | function | `(expression, *, assumptions=None)` | Normalize the nonnegative variance of a complex-valued random expression. |
| `conditional_covariance` | function | `(left, right, *, given, assumptions=None)` | Normalize conditional covariance using bilinearity and conditional independence. |
| `conditional_entropy` | function | `(expression, *, given, assumptions=None)` | Return conditional entropy with independence simplification. |
| `conditional_expectation` | function | `(expression, *, given, assumptions=None)` | Normalize conditional expectation using measurability and independence rules. |
| `conditional_moment` | function | `(expression, order, *, given, central=False, assumptions=None)` | Return a raw or central conditional moment. |
| `conditional_probability` | function | `(dist, event, *, given, variable=None)` | Return ``P(event \| given)`` from the defining ratio. |
| `conditional_variance` | function | `(expression, *, given, assumptions=None)` | Normalize conditional variance using measurability and covariance rules. |
| `conditionally_independent` | function | `(left, right, *, given, assumptions=None)` | Construct or query conditional independence. |
| `conditionally_independent_collections` | function | `(left, right, *, given, assumptions=None)` | Construct or query conditional independence between variable collections. |
| `correlation` | function | `(left, right, *, assumptions=None)` | Return symbolic Pearson correlation. |
| `covariance` | function | `(left, right=None, *, ddof=None, assumptions=None)` | Return covariance for random expressions or paired observations. |
| `cumulant` | function | `(dist, order: 'int', *, assumptions=None)` | Return a distribution cumulant or symbolic random-expression cumulant. |
| `cumulant_generating_function` | function | `(expr, t=None, *, assumptions=None)` | Return the CGF of a distribution or symbolic random expression. |
| `delta_method` | function | `(expression, *, variable=None, center=None, asymptotic_variance=None, assumptions=None)` | Apply the scalar first-order delta method. |
| `delta_variance` | function | `(expression, *, variable=None, assumptions=None)` | Return the first-order propagated variance approximation. |
| `density` | function | `(dist, value)` | Return density or mass, preserving the shape of array-like input. |
| `depends_on` | function | `(expression, variables: 'Iterable[RandomVariable] \| RandomVariable') -> 'bool'` | Return whether ``expression`` depends on any supplied random variable. |
| `distribution` | function | `(expression, *, assumptions=None)` | Return the exact or symbolic law of a random expression when derivable. |
| `distributions` | module | `` | Probability distribution classes, families, and nonparametric laws. |
| `entropy_chain_rule` | function | `(*expressions, assumptions=None)` | Return the entropy chain-rule decomposition for an ordered tuple. |
| `event_complement` | function | `(event, *, variable=None)` | Return the complement of a scalar probability event. |
| `event_intersection` | function | `(*events)` | Return the intersection of compatible scalar events. |
| `event_union` | function | `(*events)` | Return the union of compatible scalar events. |
| `events_independent` | function | `(dist, left, right, *, variable=None)` | Return exact event-independence status when the probabilities decide it. |
| `expectation` | function | `(dist, expression=None, *, variable=None, variables=None, numerical_fallback: 'bool' = False, samples: 'int' = 100000, rng=None, return_result: 'bool' = False, assumptions=None)` | Expectation under a distribution or of a symbolic random expression. |
| `factorial_moment` | function | `(expression, order, *, assumptions=None)` | Return a falling-factorial moment. |
| `finite_mean` | function | `(variable, *, assumptions=None)` | Construct or query finiteness of the first moment. |
| `finite_moment` | function | `(variable, order, *, assumptions=None)` | Construct or query finiteness of a raw moment. |
| `finite_variance` | function | `(variable, *, assumptions=None)` | Construct or query finiteness of the second moment. |
| `functionals` | module | `` | Generic probability/statistics functionals and exact solver dispatch. |
| `identically_distributed` | function | `(*variables, assumptions=None)` | Construct or query equality in distribution for a collection. |
| `iid` | function | `(*variables, assumptions=None)` | Construct or query an independent-and-identically-distributed relation. |
| `inclusion_exclusion` | function | `(dist, *events, variable=None)` | Return the finite inclusion-exclusion probability of an event union. |
| `independent` | function | `(left, right, *, assumptions=None)` | Construct or query pairwise independence. |
| `independent_collections` | function | `(left, right, *, assumptions=None)` | Construct or query independence between two disjoint variable collections. |
| `information` | module | `` | Information-theoretic measures for probability distributions. |
| `joint_entropy` | function | `(*expressions, assumptions=None)` | Return joint entropy, simplifying independent collections additively. |
| `kurtosis` | function | `(expression, *, excess=False, assumptions=None)` | Return ordinary or excess kurtosis. |
| `mean` | function | `(value, *, weights=None, assumptions=None)` | Return the arithmetic mean of data or the mean of a distribution. |
| `median` | function | `(value)` | Return the sample median or the median of a distribution. |
| `mixed_moment` | function | `(*expressions, orders=None, assumptions=None)` | Return a mixed raw moment of one or more random expressions. |
| `moment` | function | `(dist, order: 'int', *, central: 'bool' = False, assumptions=None)` | Return a raw or central moment for a distribution or random expression. |
| `moment_generating_function` | function | `(expr, t=None, *, assumptions=None)` | Return the MGF of a distribution or symbolic random expression. |
| `mutual_information` | function | `(left, right, *, assumptions=None)` | Return symbolic mutual information with independence and bijection simplification. |
| `mutual_information_chain_rule` | function | `(left, *right, assumptions=None)` | Return ``I(X;Y1)+I(X;Y2\|Y1)+...`` as a symbolic chain rule. |
| `mutually_independent` | function | `(*variables, assumptions=None)` | Construct or query mutual independence for a collection. |
| `pairwise_independent` | function | `(*variables, assumptions=None)` | Construct or query pairwise independence for a collection. |
| `probability` | function | `(dist, event, *, variable=None, numerical_fallback: 'bool' = False, samples: 'int' = 100000, rng=None, return_result: 'bool' = False)` | Compute ``P(event)`` under ``dist`` using exact-first dispatch. |
| `probability_generating_function` | function | `(expr, z=None, *, assumptions=None)` | Return the PGF of a distribution or symbolic random expression. |
| `product_kl_divergence` | function | `(pairs, **kwargs)` | Apply KL additivity to a finite product of independent component laws. |
| `pseudo_covariance` | function | `(left, right=None, *, assumptions=None)` | Normalize pseudo-covariance for complex-valued random expressions. |
| `quantile` | function | `(value, q, *, method='linear')` | Return sample or distribution quantiles. |
| `random_entropy` | function | `(expression, *, assumptions=None)` | Return entropy of a random variable when its law is known, else formally. |
| `random_matrix` | module | `` | Random-matrix ensembles and spectral statistics. |
| `random_variables` | function | `(expression) -> 'frozenset[RandomVariable]'` | Return the random variables occurring in a symbolic expression. |
| `raw_moment` | function | `(dist, order: 'int', *, assumptions=None)` | Return a distribution raw moment or symbolic random-expression moment. |
| `register_affine_closure` | function | `(distribution_type)` | Register a process-global affine pushforward rule. |
| `register_sum_closure` | function | `(function)` | Register a process-global independent-sum closure rule. |
| `sample` | function | `(distribution: 'Distribution', size=None, rng=None, *, max_attempts: 'int' = 100000)` | Draw random variates from ``distribution``. |
| `skewness` | function | `(expression, *, assumptions=None)` | Return the third standardized moment. |
| `smoothing` | module | `` | Smoothing, rolling, weighted-data, and binned-data namespace. |
| `spaces` | module | `` | Shape-aware event and parameter spaces for probability models. |
| `standard_deviation` | function | `(value, *, ddof=None, assumptions=None)` | Return sample or distribution standard deviation. |
| `standardized_moment` | function | `(expression, order, *, assumptions=None)` | Return a standardized central moment. |
| `stats` | module | `` | User-facing statistical estimation, testing, and modeling namespace. |
| `survival` | module | `` | Nonparametric survival analysis for right-censored and left-truncated data. |
| `symbolic` | module | `` | Symbolic distribution algebra and transform-based reasoning. |
| `taylor_expectation` | function | `(expression, *, variable=None, order=2, assumptions=None, return_result=False)` | Approximate an expectation by a central-moment Taylor expansion. |
| `taylor_variance` | function | `(expression, *, variable=None, order=1, assumptions=None, return_result=False)` | Approximate a variance from a truncated Taylor polynomial. |
| `total_covariance` | function | `(left, right, *, given, assumptions=None, expanded=False)` | Return the law of total covariance, optionally in expanded form. |
| `total_expectation` | function | `(expression, *, given, assumptions=None, expanded=False)` | Return the tower-property identity, optionally in expanded form. |
| `total_probability` | function | `(dist, event, *, partition, variable=None)` | Apply the law of total probability over a supplied finite partition. |
| `total_variance` | function | `(expression, *, given, assumptions=None, expanded=False)` | Return the law of total variance, optionally in expanded form. |
| `u_statistics` | module | `` | Complete and randomized incomplete U-statistics. |
| `uncorrelated` | function | `(left, right, *, assumptions=None)` | Construct or query zero-correlation structure. |
| `union_probability` | function | `(dist, *events, variable=None)` | Return ``P(union(events))``; exact evaluation is delegated to probability(). |
| `variance` | function | `(value, *, ddof=None, assumptions=None)` | Return sample variance or distribution variance. |

## Algebraic statistics
| Name | Kind | Signature | Summary |
| --- | --- | --- | --- |
| `TRINOMIAL_PAPER_CONFIGURATION` | object | `` | Published SDL configuration attached to a reference experiment. |
| `AlgebraicMLEResult` | class | `(mle: 'Mapping[sp.Symbol, sp.Expr] \| None', mle_points: 'tuple[Mapping[sp.Symbol, sp.Expr], ...]', likelihood_value: 'sp.Expr', critical_points: 'CriticalPointResult \| None', exact: 'bool', complete: 'bool', attained: 'bool', method: 'str', certificate: 'object \| None' = None) -> None` | Exact global maximum-likelihood result on a semialgebraic model. |
| `AlgebraicModel` | class | `(probabilities: 'tuple[sp.Symbol, ...]', equations: 'tuple[sp.Expr, ...]' = (), inequalities: 'tuple[object, ...]' = (), normalization: 'sp.Expr \| None' = None, parameters: 'tuple[sp.Symbol, ...]' = (), parameterization: 'Mapping[sp.Symbol, sp.Expr] \| None' = None, metadata: 'Mapping[str, object]' = <factory>) -> None` | Statistical model defined by polynomial equalities and inequalities. |
| `BasicSemialgebraicNull` | class | `(parameters: 'tuple[sp.Symbol, ...]', equalities: 'tuple[sp.Expr, ...]' = (), inequalities: 'tuple[sp.Expr, ...]' = (), label: 'str \| None' = None, metadata: 'Mapping[str, object]' = <factory>) -> None` | One basic closed semialgebraic component of a null hypothesis. |
| `ConstraintAugmentation` | class | `(component: 'BasicSemialgebraicNull', original_constraints: 'tuple[sp.Expr, ...]', added_constraints: 'tuple[sp.Expr, ...]', weights: 'np.ndarray', method: 'str', seed: 'int \| None') -> None` | A basic null together with generated redundant SDL constraints. |
| `ContingencyTable` | class | `(counts: 'object', variable_names: 'tuple[str, ...]' = ()) -> None` | Immutable finite table of nonnegative integer counts. |
| `CriticalPointResult` | class | `(points: 'tuple[Mapping[sp.Symbol, sp.Expr], ...]', likelihood_values: 'tuple[sp.Expr, ...]', equations: 'LikelihoodEquationSystem', real: 'bool', positive: 'bool', exact: 'bool', complete: 'bool', backend: 'str', notes: 'tuple[str, ...]' = ()) -> None` | Exact finite critical points of a likelihood function. |
| `DiscreteAlgebraicModel` | class | `(probabilities: 'tuple[sp.Symbol, ...]', equations: 'tuple[sp.Expr, ...]' = (), inequalities: 'tuple[object, ...]' = (), normalization: 'sp.Expr \| None' = None, parameters: 'tuple[sp.Symbol, ...]' = (), parameterization: 'Mapping[sp.Symbol, sp.Expr] \| None' = None, metadata: 'Mapping[str, object]' = <factory>, cardinalities: 'tuple[int, ...]' = (), variable_names: 'tuple[str, ...]' = ()) -> None` | Algebraic model for a finite joint probability table. |
| `ExactConditionalTestResult` | class | `(pvalue: 'sp.Rational', observed_score: 'object', observed_probability: 'sp.Rational', fiber_size: 'int', method: 'str', exact: 'bool' = True) -> None` | Result of exact inference after conditioning on sufficient statistics. |
| `Fiber` | class | `(design_matrix: 'tuple[tuple[int, ...], ...]', statistic: 'tuple[int, ...]') -> None` | Nonnegative integer solutions of ``A x = b``. |
| `FiberSample` | class | `(samples: 'tuple[tuple[int, ...], ...]', accepted: 'int', proposed: 'int') -> None` | Samples from a Markov-basis random walk on a fiber. |
| `HajekProjectionEstimate` | class | `(values: 'np.ndarray', mean: 'np.ndarray', variance: 'np.ndarray', indices: 'tuple[int, ...]', blocks_per_observation: 'int') -> None` | Divide-and-conquer estimates of the first-order Hájek projection. |
| `IdentifiabilityResult` | class | `(identifiable: 'bool \| None', up_to_label_swapping: 'bool \| None', generic: 'bool', parameter_dimension: 'int', image_dimension: 'int', fiber_dimension: 'int', expected_label_orbit: 'int', locally_finite_to_one: 'bool', certified: 'bool', method: 'str', reason: 'str') -> None` | Local/global information about a latent parameterization. |
| `Kernel` | class | `(order: 'int', function: 'Callable[..., object]', dimension: 'int' = 1, symmetrization: 'str' = 'exact', permutation_count: 'int \| None' = None) -> None` | A scalar or vector-valued kernel with declared order and symmetrization metadata. |
| `LatentClassFitResult` | class | `(weights: 'tuple[float, ...]', conditional_probabilities: 'tuple[tuple[tuple[float, ...], ...], ...]', log_likelihood: 'float', iterations: 'int', converged: 'bool', method: 'str' = 'em') -> None` | Numerical EM fit of a finite latent-class model. |
| `LatentClassModel` | class | `(probabilities: 'tuple[sp.Symbol, ...]', equations: 'tuple[sp.Expr, ...]' = (), inequalities: 'tuple[object, ...]' = (), normalization: 'sp.Expr \| None' = None, parameters: 'tuple[sp.Symbol, ...]' = (), parameterization: 'Mapping[sp.Symbol, sp.Expr] \| None' = None, metadata: 'Mapping[str, object]' = <factory>, cardinalities: 'tuple[int, ...]' = (), variable_names: 'tuple[str, ...]' = (), latent_classes: 'int' = 1, mixing_weights: 'tuple[sp.Expr, ...]' = (), conditional_probabilities: 'tuple[tuple[tuple[sp.Expr, ...], ...], ...]' = ()) -> None` | Finite latent-class/naive-Bayes model with one hidden class variable. |
| `LikelihoodEquationSystem` | class | `(probabilities: 'tuple[sp.Symbol, ...]', counts: 'tuple[int, ...]', likelihood: 'sp.Expr', equations: 'tuple[sp.Expr, ...]', multipliers: 'tuple[sp.Symbol, ...]', lagrange_equations: 'tuple[sp.Expr, ...]', saturated: 'bool' = True) -> None` | Polynomial likelihood critical equations in probability coordinates. |
| `MLDegreeResult` | class | `(degree: 'int', generic: 'bool', generic_certified: 'bool', witness_counts: 'tuple[tuple[int, ...], ...]', witness_degrees: 'tuple[int, ...]', exact: 'bool', complete: 'bool') -> None` | Likelihood critical-point count for generic-data witness samples. |
| `ModelIdeal` | class | `(variables: 'tuple[sp.Symbol, ...]', generators: 'tuple[sp.Expr, ...]') -> None` | Polynomial ideal attached to an algebraic statistical model. |
| `ModelInvariants` | class | `(dimension: 'int', degree: 'int', singular_locus: 'object') -> None` | Exact algebraic invariants of a statistical model. |
| `MomentDecompositionResult` | class | `(decomposition: 'object', rank: 'object', identifiability: 'object \| None', symmetric: 'bool', exact: 'bool', complete: 'bool', method: 'str') -> None` | Tensor decomposition together with rank and uniqueness information. |
| `MomentTensor` | class | `(values: 'sp.ImmutableDenseNDimArray', order: 'int', kind: 'str', sample_size: 'int', centered: 'bool' = False, view_dimensions: 'tuple[int, ...]' = ()) -> None` | Finite-sample tensor statistic with exact symbolic entries. |
| `MultiViewMixtureResult` | class | `(weights: 'tuple[object, ...]', component_means: 'tuple[tuple[tuple[object, ...], ...], ...]', reconstructed: 'object', residual: 'object', components: 'int', converged: 'bool', exact: 'bool', identifiable: 'bool \| None', certified_identifiable: 'bool', method: 'str') -> None` | Recovered finite mixture from a multi-view cross-moment tensor. |
| `ProbabilityTensor` | class | `(values: 'object', variable_names: 'tuple[str, ...]' = ()) -> None` | Joint finite probability distribution represented as a tensor. |
| `ProbabilityValidationResult` | class | `(normalized: 'bool \| None', nonnegative: 'bool \| None') -> None` | Validation state for a finite probability tensor. |
| `SDLBootstrapResult` | class | `(statistics: 'np.ndarray', p_value: 'float', replicates: 'int', studentization: 'SDLStudentizationResult', h_component: 'np.ndarray \| None' = None, g_component: 'np.ndarray \| None' = None, combined: 'np.ndarray \| None' = None) -> None` | Conditional Gaussian multiplier bootstrap based on one SDL realization. |
| `SDLCalibrationResult` | class | `(alpha: 'float', repetitions: 'int', rejection_rate: 'float', monte_carlo_standard_error: 'float', p_values: 'np.ndarray', statistics: 'np.ndarray', realized_budgets: 'np.ndarray', requested_budget: 'int', kernel_orders: 'np.ndarray', projection_sizes: 'np.ndarray', seed: 'int \| None') -> None` | Monte Carlo null calibration summary for a fixed SDL configuration. |
| `SDLDiagnostics` | class | `(realized_budget: 'int', requested_budget: 'int', realized_budget_ratio: 'float', alpha_n: 'float', projection_size: 'int', blocks_per_observation: 'int', kernel_order: 'int', constraint_count: 'int', hajek_variance_share: 'np.ndarray', kernel_variance_share: 'np.ndarray', zero_standard_error: 'np.ndarray', zero_over_zero: 'np.ndarray', nonzero_over_zero: 'np.ndarray', bootstrap_standard_error: 'float', bootstrap_critical_values: 'Mapping[float, float]') -> None` | Numerical diagnostics for one completed basic-null SDL test. |
| `SDLReferenceConfiguration` | class | `(sample_size: 'int', budget: 'int', bootstrap_replicates: 'int', projection_size: 'int') -> None` | Published SDL configuration attached to a reference experiment. |
| `SDLStudentizationResult` | class | `(statistic: 'float', coordinate_statistics: 'np.ndarray', u_statistic: 'UStatisticResult', hajek: 'HajekProjectionEstimate', kernel_variance: 'np.ndarray', variance: 'np.ndarray', standard_error: 'np.ndarray', alpha: 'float', kernel_values: 'np.ndarray') -> None` | Incomplete U-statistic and variance terms entering the SDL statistic. |
| `SDLTestResult` | class | `(statistic: 'float', p_value: 'float', hypothesis: 'SemialgebraicHypothesis', kernel: 'Kernel', studentization: 'SDLStudentizationResult', bootstrap: 'SDLBootstrapResult', sample_size: 'int', kernel_order: 'int', budget: 'int', projection_size: 'int', bootstrap_replicates: 'int', constraint_count: 'int', seed: 'int \| None', augmentation: 'ConstraintAugmentation \| None' = None, symmetrization: 'str' = 'exact', symmetrization_permutations: 'int \| None' = None) -> None` | Complete result of an SDL test for one basic semialgebraic null. |
| `SDLUnionTestResult` | class | `(p_value: 'float', component_results: 'tuple[SDLTestResult, ...]', hypothesis: 'SemialgebraicHypothesis', component_seeds: 'tuple[int, ...]', seed: 'int \| None') -> None` | Componentwise SDL tests for a finite union null. |
| `SemialgebraicHypothesis` | class | `(parameters: 'tuple[sp.Symbol, ...]', components: 'tuple[BasicSemialgebraicNull, ...]', ambient: 'sp.Expr' = True, estimators: 'Mapping[sp.Symbol, object]' = <factory>, sample_space: 'object \| None' = None, metadata: 'Mapping[str, object]' = <factory>) -> None` | A semialgebraic null hypothesis embedded in an ambient parameter space. |
| `TensorIndependentResult` | class | `(independent: 'bool \| None', constraints: 'tuple[sp.Expr, ...]', residuals: 'tuple[sp.Expr, ...]') -> None` | Exact result for complete independence of a probability tensor. |
| `ToricModel` | class | `(design_matrix, probabilities: 'Sequence[sp.Symbol] \| None' = None, *, inequalities: 'Iterable[object]' = (), metadata: 'Mapping[str, object] \| None' = None) -> 'None'` | Normalized toric probability model defined by an integer design matrix. |
| `UnbiasedParameterEstimator` | class | `(function: 'Callable[..., object]', arity: 'int' = 1) -> None` | An unbiased estimator of one parameter from ``arity`` i.i.d. observations. |
| `algebraic_mle` | function | `(model: 'AlgebraicModel', data, *, include_critical_points: 'bool' = True, max_faces: 'int' = 64) -> 'AlgebraicMLEResult'` | Return the exact global MLE by solving every relevant simplex face. |
| `augment_constraints` | function | `(component: 'BasicSemialgebraicNull', *, count: 'int', method: 'str' = 'convex', concentration: 'float' = 1.0, seed: 'int \| None' = None, rng: 'np.random.Generator \| None' = None) -> 'ConstraintAugmentation'` | Add redundant random convex combinations of canonical SDL constraints. |
| `augment_hypothesis_constraints` | function | `(hypothesis: 'SemialgebraicHypothesis', *, count: 'int', method: 'str' = 'convex', concentration: 'float' = 1.0, seed: 'int \| None' = None, rng: 'np.random.Generator \| None' = None) -> 'tuple[SemialgebraicHypothesis, ConstraintAugmentation]'` | Augment the single basic component of a hypothesis. |
| `categorical_multi_view_moment` | function | `(observations, *, cardinalities=None, weights=None) -> 'MomentTensor'` | Return the joint one-hot cross moment of aligned categorical views. |
| `central_moment_tensor` | function | `(samples, order: 'int', *, weights=None) -> 'MomentTensor'` | Return ``E[(X-E[X])^{⊗ order}]`` for aligned multivariate samples. |
| `conditional_test` | function | `(observed, model, *, statistic: 'str \| Callable[[tuple[int, ...]], object]' = 'probability', alternative: 'str' = 'greater', max_candidates: 'int' = 1000000) -> 'ExactConditionalTestResult'` | Perform an exact test on the sufficient-statistic fiber of a toric model. |
| `conditionally_independent_model` | function | `(variables, statements) -> 'DiscreteAlgebraicModel'` | Return a discrete conditional-independence model. |
| `critical_points` | function | `(model: 'AlgebraicModel', data, *, real: 'bool' = True, positive: 'bool' = True) -> 'CriticalPointResult'` | Solve the finite likelihood critical-point system exactly with semialg. |
| `cumulant_tensor` | function | `(samples, order: 'int', *, weights=None, max_order: 'int' = 6) -> 'MomentTensor'` | Return the empirical joint cumulant tensor using the partition formula. |
| `decompose_moment_tensor` | function | `(tensor, components: 'int \| None' = None, *, symmetric: 'bool' = True, method: 'str' = 'auto', tolerance: 'float' = 1e-10, rng=None) -> 'MomentDecompositionResult'` | Decompose a moment tensor using TensorAtlas CP/Waring machinery. |
| `fit_latent_class` | function | `(counts, latent_classes: 'int', *, max_iter: 'int' = 500, tolerance: 'float' = 1e-10, rng=None) -> 'LatentClassFitResult'` | Fit a latent-class model to a contingency table by EM. |
| `fit_multiview_mixture` | function | `(views, components: 'int', *, weights=None, method: 'str' = 'numerical', tolerance: 'float' = 1e-08, rng=None, restarts: 'int' = 8) -> 'MultiViewMixtureResult'` | Construct the empirical cross moment and recover a multi-view mixture. |
| `generic_identifiability` | function | `(model: 'LatentClassModel') -> 'IdentifiabilityResult'` | Certify generic identifiability where dimension and Kruskal theory allow. |
| `hajek_projection_estimate` | function | `(sample: 'Sequence[object]', kernel: 'Callable[..., object]', *, order: 'int \| None' = None, n1: 'int \| None' = None, indices: 'Sequence[int] \| None' = None, rng: 'np.random.Generator \| int \| None' = None, shuffle_blocks: 'bool' = False) -> 'HajekProjectionEstimate'` | Estimate ``g(X_i)=E[h(X_i,X_2,...,X_m)\|X_i]`` by divide-and-conquer. |
| `hypothesis_kernel` | function | `(hypothesis: 'SemialgebraicHypothesis', *, order: 'int \| None' = None, symmetrization: 'str' = 'exact', permutation_count: 'int \| None' = None, rng: 'np.random.Generator \| int \| None' = None) -> 'Kernel'` | Construct the core SDL kernel for a basic semialgebraic hypothesis. |
| `identifiability` | function | `(model: 'LatentClassModel', parameters: 'Mapping[sp.Symbol, object] \| None' = None) -> 'IdentifiabilityResult'` | Return generic or pointwise local identifiability information. |
| `independent_model` | function | `(cardinalities, *, variables=()) -> 'DiscreteAlgebraicModel'` | Return the complete-independence model for a finite probability table. |
| `latent_class_model` | function | `(cardinalities, latent_classes: 'int', *, variables=()) -> 'LatentClassModel'` | Construct a finite latent-class model for observed categorical variables. |
| `likelihood_equations` | function | `(model: 'AlgebraicModel', data) -> 'LikelihoodEquationSystem'` | Return the multiplier-eliminated, very-affine likelihood equations. |
| `maximum_likelihood_degree` | function | `(model: 'AlgebraicModel', *, samples: 'int' = 2) -> 'MLDegreeResult'` | Return the stable exact complex critical-point count at generic witnesses. |
| `model_degree` | function | `(model: 'AlgebraicModel') -> 'int'` | Return the exact affine degree of ``model``. |
| `model_dimension` | function | `(model: 'AlgebraicModel') -> 'int'` | Return the exact affine dimension of ``model``. |
| `model_ideal` | function | `(model: 'AlgebraicModel') -> 'ModelIdeal'` | Return the polynomial equality ideal of ``model``. |
| `model_invariants` | function | `(model: 'AlgebraicModel') -> 'ModelInvariants'` | Return core exact algebraic invariants of ``model``. |
| `moment_tensor` | function | `(samples, order: 'int', *, weights=None) -> 'MomentTensor'` | Return the empirical raw moment tensor ``E[X^{⊗ order}]``. |
| `multi_view_moment` | function | `(views, *, weights=None) -> 'MomentTensor'` | Return ``E[X1 ⊗ ... ⊗ Xk]`` from aligned multi-view observations. |
| `multinomial_point_on_model` | function | `(model: 'int', probabilities: 'Sequence[float]') -> 'bool'` | Numerically check whether a trinomial point satisfies a reference null. |
| `polynomial_constraint_kernel` | function | `(component: 'BasicSemialgebraicNull', estimators: 'Mapping[sp.Symbol, object]', *, order: 'int \| None' = None, symmetrization: 'str' = 'exact', permutation_count: 'int \| None' = None, rng: 'np.random.Generator \| int \| None' = None) -> 'Kernel'` | Construct an unbiased SDL polynomial kernel for a basic null. |
| `probability_tensor` | function | `(values, *, variables=()) -> 'ProbabilityTensor'` | Construct a :class:`ProbabilityTensor` from joint probabilities. |
| `recover_multiview_mixture` | function | `(tensor, components: 'int', *, method: 'str' = 'numerical', tolerance: 'float' = 1e-08, rng=None, restarts: 'int' = 8) -> 'MultiViewMixtureResult'` | Recover mixture weights and per-view component means from a cross moment. |
| `sample_fiber` | function | `(observed, *, model=None, fiber: 'Fiber \| None' = None, size: 'int' = 1000, burnin: 'int' = 100, thin: 'int' = 1, rng=None, measure: 'str' = 'conditional') -> 'FiberSample'` | Sample a fiber with a Markov-basis Metropolis walk. |
| `sdl_calibrate` | function | `(sampler: 'Callable[[np.random.Generator], Sequence[object]]', hypothesis: 'SemialgebraicHypothesis', *, repetitions: 'int', alpha: 'float' = 0.05, budget: 'int', bootstrap_replicates: 'int' = 1000, seed: 'int \| None' = None, rng: 'np.random.Generator \| None' = None, **test_options) -> 'SDLCalibrationResult'` | Estimate null rejection frequency for one SDL configuration by simulation. |
| `sdl_diagnostics` | function | `(result: 'SDLTestResult', *, critical_levels: 'Sequence[float]' = (0.9, 0.95, 0.99)) -> 'SDLDiagnostics'` | Summarize the numerical regime of a completed SDL test. |
| `sdl_multiplier_bootstrap` | function | `(studentization: 'SDLStudentizationResult', *, replicates: 'int' = 1000, rng: 'np.random.Generator \| int \| None' = None, batch_size: 'int \| None' = None, retain_components: 'bool' = False) -> 'SDLBootstrapResult'` | Approximate the SDL reference distribution by Gaussian multipliers. |
| `sdl_studentize` | function | `(sample: 'Sequence[object]', kernel: 'Callable[..., object]', *, budget: 'int', order: 'int \| None' = None, n1: 'int \| None' = None, projection_indices: 'Sequence[int] \| None' = None, rng: 'np.random.Generator \| int \| None' = None, projection_rng: 'np.random.Generator \| int \| None' = None, shuffle_projection_blocks: 'bool' = False, u_result: 'UStatisticResult \| None' = None) -> 'SDLStudentizationResult'` | Compute the studentized SDL maximum statistic and its variance estimates. |
| `sdl_test` | function | `(sample: 'Sequence[object]', hypothesis: 'SemialgebraicHypothesis', *, budget: 'int', bootstrap_replicates: 'int' = 1000, kernel_order: 'int \| None' = None, n1: 'int \| None' = None, projection_indices: 'Sequence[int] \| None' = None, seed: 'int \| None' = None, rng: 'np.random.Generator \| None' = None, shuffle_projection_blocks: 'bool' = False, bootstrap_batch_size: 'int \| None' = None, retain_bootstrap_components: 'bool' = False, augment_constraints_count: 'int' = 0, augmentation_concentration: 'float' = 1.0, kernel_symmetrization: 'str' = 'exact', symmetrization_permutations: 'int \| None' = None) -> 'SDLTestResult'` | Run the core SDL procedure for one basic semialgebraic null. |
| `sdl_union_test` | function | `(sample: 'Sequence[object]', hypothesis: 'SemialgebraicHypothesis', *, budget: 'int', bootstrap_replicates: 'int' = 1000, kernel_order: 'int \| None' = None, n1: 'int \| None' = None, projection_indices: 'Sequence[int] \| None' = None, seed: 'int \| None' = None, rng: 'np.random.Generator \| None' = None, shuffle_projection_blocks: 'bool' = False, bootstrap_batch_size: 'int \| None' = None, retain_bootstrap_components: 'bool' = False, augment_constraints_count: 'int' = 0, augmentation_concentration: 'float' = 1.0, kernel_symmetrization: 'str' = 'exact', symmetrization_permutations: 'int \| None' = None) -> 'SDLUnionTestResult'` | Test a finite union null by componentwise SDL intersection-union testing. |
| `singular_model_locus` | function | `(model: 'AlgebraicModel')` | Return the algebraic singular locus of ``model``. |
| `tensor_independent` | function | `(tensor: 'ProbabilityTensor') -> 'TensorIndependentResult'` | Test complete independence using the Segre rank-one equations. |
| `toric_model` | function | `(design_matrix, probabilities=None, **kwargs) -> 'ToricModel'` | Construct a :class:`ToricModel`. |
| `trinomial_model4_components` | function | `() -> 'SemialgebraicHypothesis'` | Return Model 4 decomposed into its three Model-2-like components. |
| `trinomial_reference_hypothesis` | function | `(model: 'int') -> 'SemialgebraicHypothesis'` | Return one of the paper's four trinomial semialgebraic null models. |
| `trinomial_sampler` | function | `(probabilities: 'Sequence[float]', sample_size: 'int')` | Return a sampler producing standard-basis trinomial observations. |

## Statistics
| Name | Kind | Signature | Summary |
| --- | --- | --- | --- |
| `ANOVAResult` | class | `(statistic: 'float', pvalue: 'float', df_between: 'int', df_within: 'int', ss_between: 'float', ss_within: 'float', ms_between: 'float', ms_within: 'float', group_means: 'tuple[float, ...]') -> None` | ANOVAResult(statistic: 'float', pvalue: 'float', df_between: 'int', df_within: 'int', ss_between: 'float', ss_within: 'float', ms_between: 'float', ms_within: 'float', group_means: 'tuple[float, ...]') |
| `DescriptiveSummary` | class | `(count: 'int', mean: 'float', std: 'float', minimum: 'float', q1: 'float', median: 'float', q3: 'float', maximum: 'float', iqr: 'float', mad: 'float', skewness: 'float', kurtosis: 'float') -> None` | DescriptiveSummary(count: 'int', mean: 'float', std: 'float', minimum: 'float', q1: 'float', median: 'float', q3: 'float', maximum: 'float', iqr: 'float', mad: 'float', skewness: 'float', kurtosis: 'float') |
| `EmpiricalDistribution` | class | `(data)` | EmpiricalDistribution(data) |
| `EquivalenceTestResult` | class | `(estimate: 'float', lower_bound: 'float', upper_bound: 'float', lower_statistic: 'float', upper_statistic: 'float', lower_pvalue: 'float', upper_pvalue: 'float', standard_error: 'float', degrees_of_freedom: 'float', method: 'str', sample_sizes: 'tuple[int, ...]') -> None` | Result of an equivalence test based on two one-sided tests (TOST). |
| `EstimationResult` | class | `(distribution: 'object', method: 'str', n: 'int', log_likelihood: 'float', parameters: 'dict', converged: 'bool' = True) -> None` | EstimationResult(distribution: 'object', method: 'str', n: 'int', log_likelihood: 'float', parameters: 'dict', converged: 'bool' = True) |
| `FormulaRegressionResult` | class | `(design_matrix: 'DesignMatrix', regression: 'Any') -> None` | FormulaRegressionResult(design_matrix: 'DesignMatrix', regression: 'Any') |
| `GLMResult` | class | `(design_matrix: 'DesignMatrix', coefficients: 'np.ndarray', standard_errors: 'np.ndarray', statistics: 'np.ndarray', pvalues: 'np.ndarray', covariance: 'np.ndarray', fitted_mean: 'np.ndarray', linear_predictor: 'np.ndarray', residuals: 'np.ndarray', deviance: 'float', null_deviance: 'float', dispersion: 'float', df_resid: 'int', log_likelihood: 'float', aic: 'float', family: 'str', link: 'str', converged: 'bool', iterations: 'int', observation_weights: 'np.ndarray \| None' = None, offset: 'np.ndarray \| None' = None, trials: 'np.ndarray \| None' = None, offset_name: 'str \| None' = None) -> None` | GLMResult(design_matrix: 'DesignMatrix', coefficients: 'np.ndarray', standard_errors: 'np.ndarray', statistics: 'np.ndarray', pvalues: 'np.ndarray', covariance: 'np.ndarray', fitted_mean: 'np.ndarray', linear_predictor: 'np.ndarray', residuals: 'np.ndarray', deviance: 'float', null_deviance: 'float', dispersion: 'float', df_resid: 'int', log_likelihood: 'float', aic: 'float', family: 'str', link: 'str', converged: 'bool', iterations: 'int', observation_weights: 'np.ndarray \| None' = None, offset: 'np.ndarray \| None' = None, trials: 'np.ndarray \| None' = None, offset_name: 'str \| None' = None) |
| `HypothesisTestResult` | class | `(statistic: 'float', pvalue: 'float', alternative: 'str', method: 'str', sample_sizes: 'tuple[int \| float, ...]', null_value: 'float \| None' = None, degrees_of_freedom: 'float \| None' = None, estimate: 'float \| None' = None, standard_error: 'float \| None' = None) -> None` | HypothesisTestResult(statistic: 'float', pvalue: 'float', alternative: 'str', method: 'str', sample_sizes: 'tuple[int \| float, ...]', null_value: 'float \| None' = None, degrees_of_freedom: 'float \| None' = None, estimate: 'float \| None' = None, standard_error: 'float \| None' = None) |
| `MultipleTestingResult` | class | `(pvalues: 'np.ndarray', adjusted_pvalues: 'np.ndarray', rejected: 'np.ndarray', alpha: 'float', method: 'str') -> None` | MultipleTestingResult(pvalues: 'np.ndarray', adjusted_pvalues: 'np.ndarray', rejected: 'np.ndarray', alpha: 'float', method: 'str') |
| `RegressionResult` | class | `(coefficients: 'np.ndarray', standard_errors: 'np.ndarray', t_statistics: 'np.ndarray', pvalues: 'np.ndarray', covariance: 'np.ndarray', fitted: 'np.ndarray', residuals: 'np.ndarray', r_squared: 'float', adjusted_r_squared: 'float', sigma2: 'float', df_resid: 'int', f_statistic: 'float', f_pvalue: 'float', intercept: 'bool', design: 'np.ndarray \| None' = None, response: 'np.ndarray \| None' = None, covariance_type: 'str' = 'classical', feature_names: 'tuple[str, ...] \| None' = None, response_name: 'str \| None' = None, row_index: 'Any' = None) -> None` | RegressionResult(coefficients: 'np.ndarray', standard_errors: 'np.ndarray', t_statistics: 'np.ndarray', pvalues: 'np.ndarray', covariance: 'np.ndarray', fitted: 'np.ndarray', residuals: 'np.ndarray', r_squared: 'float', adjusted_r_squared: 'float', sigma2: 'float', df_resid: 'int', f_statistic: 'float', f_pvalue: 'float', intercept: 'bool', design: 'np.ndarray \| None' = None, response: 'np.ndarray \| None' = None, covariance_type: 'str' = 'classical', feature_names: 'tuple[str, ...] \| None' = None, response_name: 'str \| None' = None, row_index: 'Any' = None) |
| `adjust_pvalues` | function | `(pvalues, *, method='holm', alpha=0.05)` | Adjust a family of p-values for FWER or FDR control. |
| `ancova` | function | `(formula: 'str', data: 'Mapping[str, Any]', *, type=2, contrasts=None) -> 'FactorialANOVAResult'` | Fixed-effects ANCOVA using the common formula/term infrastructure. |
| `bartlett_test` | function | `(*samples)` | Bartlett test of equality of variances across two or more groups. |
| `bootstrap` | function | `(data, statistic=<function mean at 0x…>, *, iterations=2000, confidence=0.95, rng=None, method='percentile', standard_error=None, studentized_iterations=200)` | Bootstrap a statistic and optionally construct a confidence interval. |
| `chi_square_goodness_of_fit` | function | `(observed, expected=None)` | Perform a Pearson chi-square goodness-of-fit test. |
| `chi_square_independence` | function | `(table)` | Pearson chi-square test of independence for a contingency table. |
| `coefficient_contrast` | function | `(model, contrast, *, value=0.0) -> 'CoefficientContrastResult'` | Test a scalar linear coefficient contrast c' beta = value. |
| `compare_models` | function | `(*models, names=None) -> 'ModelComparisonResult'` | Compare nested OLS models by F tests or GLMs by deviance tests. |
| `correlation` | function | `(x, y=None)` | Return the Pearson correlation for paired observations or weighted data. |
| `covariance` | function | `(x, y=None, ddof=1)` | Return covariance for paired observations or weighted data. |
| `describe` | function | `(data)` | Return a compact descriptive summary of a one-dimensional sample. |
| `descriptive` | module | `` | Descriptive and empirical statistics. |
| `estimated_marginal_means` | function | `(model, factor: 'str', *, scale='response') -> 'EstimatedMarginalMeansResult'` | Balanced reference-grid estimated marginal means for a categorical factor. |
| `estimation` | module | `` | Classical parameter estimation. |
| `exact_binomial_test` | function | `(successes, n, p=0.5, alternative='two-sided')` | Perform an exact test for a binomial success probability. |
| `factorial_anova` | function | `(formula: 'str', data: 'Mapping[str, Any]', *, type=2, contrasts=None) -> 'FactorialANOVAResult'` | Multi-factor fixed-effects ANOVA using the shared formula/design infrastructure. |
| `glm` | function | `(formula: 'str', data: 'Mapping[str, Any]', *, family='gaussian', link=None, contrasts=None, offset=None, weights=None, trials=None, max_iter=100, tol=1e-08) -> 'GLMResult'` | Fit a generalized linear model by iteratively reweighted least squares. |
| `inference` | module | `` | General likelihood inference, resampling, regression, and nonparametric tests. |
| `interquartile_range` | function | `(data)` | Return the difference between the empirical 75th and 25th percentiles. |
| `joint_wald_test` | function | `(model, contrast_matrix, *, value=None) -> 'JointWaldResult'` | Joint Wald chi-square test R beta = q. |
| `kendall_tau_test` | function | `(x, y, alternative='two-sided')` | Kendall tau-b test using the large-sample tie-corrected Normal approximation. |
| `kruskal_wallis_test` | function | `(*groups)` | Perform the Kruskal–Wallis rank test for independent groups. |
| `kurtosis` | function | `(data, fisher=True, bias=False)` | Return sample kurtosis, optionally on the Fisher excess-kurtosis scale. |
| `levene_test` | function | `(*samples, center='median')` | Levene variance-homogeneity test; ``center='median'`` gives Brown-Forsythe. |
| `likelihood` | function | `(factory, data, parameter_names=None, *, fixed=None)` | Construct a parametric likelihood or evaluate an iid sample likelihood. |
| `linear_model` | function | `(formula: 'str', data: 'Mapping[str, Any]', *, contrasts=None, covariance='classical', weights=None) -> 'FormulaRegressionResult'` | Fit OLS/WLS from a formula using the shared design-matrix compiler. |
| `linear_regression` | function | `(x, y, *, intercept=True, covariance='classical')` | Fit ordinary least squares with coefficient and model-level inference. |
| `mann_whitney_u_test` | function | `(x, y, alternative='two-sided', *, exact='auto', exact_threshold=200000)` | Compare two independent samples using the Mann-Whitney rank statistic. |
| `maximum_likelihood` | function | `(family, data)` | Fit a supported distribution family by closed-form maximum likelihood. |
| `mean` | function | `(data, weights=None)` | Return the arithmetic mean, with optional explicit weights. |
| `median` | function | `(data)` | Return the sample median. |
| `median_absolute_deviation` | function | `(data, scale=1.0)` | Return the median absolute deviation about the sample median. |
| `method_of_moments` | function | `(family, data)` | Fit a supported distribution family by matching empirical moments. |
| `modeling` | module | `` | Formula models, generalized linear models, ANOVA, and post-fit analysis. |
| `one_proportion_z_test` | function | `(successes, n, p0=0.5, alternative='two-sided')` | Perform a large-sample z test for one binomial proportion. |
| `one_sample_t_test` | function | `(data, mu=0, alternative='two-sided')` | Perform a one-sample Student t test for a population mean. |
| `one_sample_z_test` | function | `(data, mu=0, sigma=None, alternative='two-sided')` | Perform a one-sample z test for a population mean. |
| `paired_t_test` | function | `(x, y, alternative='two-sided')` | Paired Student-t test, implemented as a one-sample test of differences. |
| `pearson_correlation_test` | function | `(x, y, alternative='two-sided')` | Test the null hypothesis of zero Pearson correlation. |
| `permutation_test` | function | `(x, y, statistic=None, *, alternative='two-sided', iterations=5000, rng=None, exact_threshold=100000)` | Test exchangeability of two samples by label permutation. |
| `profile_likelihood` | function | `(lik, fit, parameter, *, values=None, confidence=0.95, points=81, span=4.0, max_iter=300, tol=1e-08)` | Profile one likelihood parameter while optimizing all nuisance values. |
| `quantile` | function | `(data, q, method='linear')` | Return empirical quantiles using the requested interpolation method. |
| `robust_regression_covariance` | function | `(result, kind='HC3')` | White/MacKinnon heteroskedasticity-consistent OLS covariance. |
| `skewness` | function | `(data, bias=False)` | Return the standardized third central moment with optional bias correction. |
| `spearman_correlation_test` | function | `(x, y, alternative='two-sided')` | Test zero Spearman rank correlation using the usual t approximation. |
| `standard_deviation` | function | `(data, ddof=1)` | Return the sample standard deviation. |
| `testing` | module | `` | Classical hypothesis tests with structured results. |
| `trimmed_mean` | function | `(data, proportion=0.1)` | Return the mean after symmetrically trimming each tail. |
| `two_proportion_z_test` | function | `(successes1, n1, successes2, n2, alternative='two-sided')` | Pooled two-sample z test for equality of proportions. |
| `variance` | function | `(data, ddof=1)` | Return the sample variance using the requested degrees-of-freedom correction. |
| `welch_t_test` | function | `(x, y, alternative='two-sided')` | Perform Welch’s unequal-variance two-sample t test. |
| `wilcoxon_signed_rank_test` | function | `(x, y=None, alternative='two-sided', *, exact='auto', exact_threshold=200000)` | Test paired or one-sample location differences with signed ranks. |
| `winsorize` | function | `(data, proportion=0.1)` | Return observations with each tail winsorized to boundary order statistics. |

## Bayesian inference
| Name | Kind | Signature | Summary |
| --- | --- | --- | --- |
| `BayesianLinearRegression` | class | `(*, prior: 'LinearRegressionPrior \| None' = None, include_intercept: 'bool' = False)` | Bayesian linear regression with exact Normal-Inverse-Gamma updating. |
| `BayesianLinearRegressionFit` | class | `(prior: 'LinearRegressionPrior', posterior: 'LinearRegressionPrior', design_matrix: 'sp.ImmutableDenseMatrix', response: 'sp.ImmutableDenseMatrix', log_evidence: 'sp.Expr') -> None` | Closed-form posterior and prediction object for Bayesian linear regression. |
| `BayesianModelComparisonResult` | class | `(names: 'tuple[str, ...]', log_evidence: 'np.ndarray', posterior_probabilities: 'np.ndarray', log_bayes_factors: 'np.ndarray') -> None` | BayesianModelComparisonResult(names: 'tuple[str, ...]', log_evidence: 'np.ndarray', posterior_probabilities: 'np.ndarray', log_bayes_factors: 'np.ndarray') |
| `BayesianMultivariateLinearRegression` | class | `(*, prior: 'MatrixNormalInverseWishartPrior \| None' = None, include_intercept: 'bool' = False)` | Exact multivariate Gaussian linear regression with MNIW conjugacy. |
| `BayesianMultivariateLinearRegressionFit` | class | `(prior: 'MatrixNormalInverseWishartPrior', posterior: 'MatrixNormalInverseWishartPrior', design_matrix: 'sp.ImmutableDenseMatrix', response: 'sp.ImmutableDenseMatrix', log_evidence: 'sp.Expr') -> None` | Exact posterior, evidence and prediction for multivariate regression. |
| `ConjugateUpdateResult` | class | `(prior: 'Any', posterior: 'Any', prior_predictive: 'Any', posterior_predictive: 'Any', log_evidence: 'sp.Expr \| None', n_observations: 'int', sufficient_statistics: 'dict[str, Any]') -> None` | ConjugateUpdateResult(prior: 'Any', posterior: 'Any', prior_predictive: 'Any', posterior_predictive: 'Any', log_evidence: 'sp.Expr \| None', n_observations: 'int', sufficient_statistics: 'dict[str, Any]') |
| `Factor` | class | `(target: 'Variable', distribution: 'Distribution \| None' = None, log_density: 'sp.Expr \| None' = None, name: 'str \| None' = None) -> None` | A probability factor associated with a target variable. |
| `GaussianProcessFit` | class | `(x_train: 'np.ndarray', y_train: 'np.ndarray', kernel: 'KernelBase', noise_variance: 'NoiseFunction \| float', mean_function: 'MeanFunction \| float \| None', covariance: 'np.ndarray', cholesky: 'np.ndarray', alpha: 'np.ndarray', log_evidence: 'float', jitter: 'float') -> None` | A fitted GP with cached Cholesky factorization. |
| `GaussianProcessRegressor` | class | `(kernel: 'KernelBase \| Callable[[np.ndarray, np.ndarray], float]', *, noise_variance: 'NoiseFunction \| float' = 0.0, mean_function: 'MeanFunction \| float \| None' = None, jitter: 'float' = 1e-10) -> 'None'` | Exact GP regression for a fixed covariance kernel and Gaussian observation noise. |
| `InferenceKind` | class | `` | str(object='') -> str |
| `InferencePlan` | class | `(candidates: 'tuple[CandidateAssessment, ...]') -> None` | InferencePlan(candidates: 'tuple[CandidateAssessment, ...]') |
| `InferenceResult` | class | `(posterior: 'Any', kind: 'InferenceKind', log_evidence: 'Any \| None' = None, steps: 'tuple[InferenceStep, ...]' = (), diagnostics: 'Mapping[str, Any]' = <factory>, metadata: 'Mapping[str, Any]' = <factory>) -> None` | Backend-independent result container. |
| `Model` | class | `(variables: 'tuple[Variable, ...]', factors: 'tuple[Factor, ...]' = (), observations: 'Mapping[str, Any]' = <factory>, name: 'str \| None' = None) -> None` | Immutable probabilistic model. |
| `NormalInverseGamma` | class | `(mu: 'sp.Expr', lambda_: 'sp.Expr', beta: 'sp.Expr', nu: 'sp.Expr') -> None` | Normal-Inverse-Gamma in the source package's parameterization. |
| `NormalInverseWishart` | class | `(mu, lambda_, psi, nu)` | NormalInverseWishart(mu, lambda_, psi, nu) |
| `Observation` | class | `(variable: 'Variable', value: 'Any') -> None` | An observed value attached to a model variable. |
| `Parameter` | class | `(name: 'str \| sp.Symbol', support: 'sp.Set' = Reals) -> None` | An unknown model variable, usually assigned a prior factor. |
| `PosteriorData` | class | `(posterior: 'Mapping[str, np.ndarray]', sample_stats: 'Mapping[str, np.ndarray]' = <factory>, attrs: 'Mapping[str, Any]' = <factory>) -> None` | Small ArviZ-shaped posterior container. |
| `RBFKernel` | class | `(length_scale: 'float \| Sequence[float]' = 1.0, variance: 'float' = 1.0) -> None` | RBFKernel(length_scale: 'float \| Sequence[float]' = 1.0, variance: 'float' = 1.0) |
| `RandomVariable` | class | `(name: 'str \| sp.Symbol', support: 'sp.Set' = Reals) -> None` | A stochastic variable whose probability law is represented by model factors. |
| `certification` | module | `` | Expression certification helpers. |
| `compare_models` | function | `(models: 'Sequence[Any]', *, names: 'Sequence[str] \| None' = None, prior_probabilities=None) -> 'BayesianModelComparisonResult'` | Compare fitted Bayesian models using marginal likelihoods and posterior model probabilities. |
| `comparison` | module | `` | Bayesian model comparison from marginal likelihoods. |
| `conjugacy` | module | `` | Conjugate-prior rules and exact posterior updates. |
| `conjugate_update` | function | `(likelihood, data: 'Iterable', prior, *, registry: 'ConjugacyRegistry' = <probstats.bayes.conjugacy.registry.ConjugacyRegistry object at 0x…>)` | Apply a conjugate Bayesian update and return its structured result. |
| `diagnostics` | module | `` | Posterior-sample diagnostics and conversion utilities. |
| `empirical_bayes` | module | `` | Empirical-Bayes evidence optimization. |
| `exact` | module | `` | Exact symbolic Bayesian inference. |
| `gp` | module | `` | Gaussian-process regression. |
| `infer` | function | `(subject: 'Any', *, config: 'PlannerConfig \| None' = None, registry: 'ConjugacyRegistry' = <probstats.bayes.conjugacy.registry.ConjugacyRegistry object at 0x…>, **context: 'Any') -> 'InferenceResult'` | Automatically choose and execute an inference method. |
| `infer_conjugate` | function | `(likelihood, data: 'Iterable', prior=None, *, registry: 'ConjugacyRegistry' = <probstats.bayes.conjugacy.registry.ConjugacyRegistry object at 0x…>)` | Infer a conjugate posterior using the supplied or default conjugacy registry. |
| `laplace` | module | `` | Laplace approximation and optimization backends. |
| `mcmc` | module | `` | General MCMC framework. |
| `models` | module | `` | Ready-to-use Bayesian model families. |
| `nested` | module | `` | Nested-sampling evidence and posterior inference. |
| `neural` | module | `` | Optional JAX Bayesian neural-network regression layer. |
| `plan_inference` | function | `(subject: 'Any', *, config: 'PlannerConfig \| None' = None, registry: 'ConjugacyRegistry' = <probstats.bayes.conjugacy.registry.ConjugacyRegistry object at 0x…>, **context: 'Any') -> 'InferencePlan'` | Return the ranked inference plan without executing it. |
| `planner` | module | `` | Inference-method planning and high-level Bayesian dispatch. |
| `posterior_predictive` | function | `(fit: 'Any', x: 'Any', *, size=1000, rng=None, observation: 'bool' = True, predictive=None)` | Draw posterior-predictive samples from exact or sampled inference results. |
| `predict` | function | `(fit: 'Any', x: 'Any', *, observation: 'bool' = True, predictive=None, size=2000, rng=None)` | Return posterior-predictive means while preserving event structure. |
| `predict_distribution` | function | `(fit: 'Any', x: 'Any', *, observation: 'bool' = True, predictive=None, size=2000, rng=None)` | Return an exact, approximate, or empirical predictive distribution. |
| `predictive` | module | `` | Common posterior-predictive interface for fitted Bayesian models. |
| `predictive_interval` | function | `(fit: 'Any', x: 'Any', *, level: 'float' = 0.95, observation: 'bool' = True, predictive=None, size=4000, rng=None)` | Return equal-tailed posterior-predictive intervals. |
| `reasoning` | module | `` | Certified semialgebraic and function-property reasoning. |

## Distributions
| Name | Kind | Signature | Summary |
| --- | --- | --- | --- |
| `LKJ` | class | `(dimension: 'int', eta: 'Any' = 1)` | LKJ prior over ``dimension`` x ``dimension`` correlation matrices. |
| `Bernoulli` | class | `(p: 'sp.Expr') -> None` | Bernoulli distribution on ``{0, 1}`` with success probability ``p``. |
| `Beta` | class | `(alpha: 'sp.Expr', beta: 'sp.Expr') -> None` | Beta distribution with positive shape parameters ``alpha`` and ``beta``. |
| `BetaBinomial` | class | `(n: 'sp.Expr', alpha: 'sp.Expr', beta: 'sp.Expr') -> None` | BetaBinomial(n: 'sp.Expr', alpha: 'sp.Expr', beta: 'sp.Expr') |
| `BetaPrime` | class | `(alpha: 'sp.Expr', beta: 'sp.Expr', scale: 'sp.Expr' = 1) -> None` | BetaPrime(alpha: 'sp.Expr', beta: 'sp.Expr', scale: 'sp.Expr' = 1) |
| `Binomial` | class | `(n: 'sp.Expr', p: 'sp.Expr') -> None` | Binomial distribution with ``n`` trials and success probability ``p``. |
| `Categorical` | class | `(probabilities: 'Sequence[Any]')` | Categorical distribution over integer labels ``0, ..., k-1``. |
| `Cauchy` | class | `(location: 'sp.Expr' = 0, scale: 'sp.Expr' = 1) -> None` | Cauchy(location: 'sp.Expr' = 0, scale: 'sp.Expr' = 1) |
| `CensoredDistribution` | class | `(base: 'Distribution', lower: 'sp.Expr' = -oo, upper: 'sp.Expr' = oo) -> None` | Distribution obtained by clipping a scalar law to censoring limits. |
| `ChiSquared` | class | `(df: 'sp.Expr') -> None` | ChiSquared(df: 'sp.Expr') |
| `Dirichlet` | class | `(concentration: 'Sequence[Any]')` | Dirichlet distribution over a probability simplex. |
| `DirichletMultinomial` | class | `(n: 'Any', concentration: 'Sequence[Any]')` | DirichletMultinomial(n: 'Any', concentration: 'Sequence[Any]') |
| `DiscreteUniform` | class | `(low: 'sp.Expr', high: 'sp.Expr') -> None` | DiscreteUniform(low: 'sp.Expr', high: 'sp.Expr') |
| `Distribution` | class | `()` | Helper class that provides a standard way to create an ABC using |
| `Exponential` | class | `(rate: 'sp.Expr') -> None` | Exponential distribution with rate ``rate``. |
| `ExponentialFamily` | class | `()` | Base class for regular scalar exponential-family distributions. |
| `FDistribution` | class | `(df1: 'sp.Expr', df2: 'sp.Expr') -> None` | FDistribution(df1: 'sp.Expr', df2: 'sp.Expr') |
| `Frechet` | class | `(shape: 'sp.Expr', scale: 'sp.Expr' = 1, location: 'sp.Expr' = 0) -> None` | Frechet(shape: 'sp.Expr', scale: 'sp.Expr' = 1, location: 'sp.Expr' = 0) |
| `Gamma` | class | `(shape: 'sp.Expr', scale: 'sp.Expr') -> None` | Gamma distribution with shape ``shape`` and scale ``scale``. |
| `Geometric` | class | `(p: 'sp.Expr') -> None` | Number of trials until the first success (support 1, 2, ...). |
| `Gumbel` | class | `(location: 'sp.Expr' = 0, scale: 'sp.Expr' = 1) -> None` | Gumbel(location: 'sp.Expr' = 0, scale: 'sp.Expr' = 1) |
| `HalfCauchy` | class | `(scale: 'sp.Expr' = 1) -> None` | HalfCauchy(scale: 'sp.Expr' = 1) |
| `HalfNormal` | class | `(sigma: 'sp.Expr' = 1) -> None` | HalfNormal(sigma: 'sp.Expr' = 1) |
| `HistogramDistribution` | class | `(data=None, bins='auto', *, weights=None, bin_edges=None, probabilities=None)` | Piecewise-uniform histogram probability distribution. |
| `Hypergeometric` | class | `(successes: 'sp.Expr', failures: 'sp.Expr', draws: 'sp.Expr') -> None` | Hypergeometric(successes: 'sp.Expr', failures: 'sp.Expr', draws: 'sp.Expr') |
| `InverseGamma` | class | `(shape: 'sp.Expr', scale: 'sp.Expr') -> None` | Inverse-gamma distribution with shape ``shape`` and scale ``scale``. |
| `InverseGaussian` | class | `(mean: 'sp.Expr', shape: 'sp.Expr') -> None` | InverseGaussian(mean: 'sp.Expr', shape: 'sp.Expr') |
| `InverseWishart` | class | `(df: 'Any', scale: 'Any')` | Inverse-Wishart distribution with degrees of freedom ``df`` and scale matrix ``scale``. |
| `KernelDensityDistribution` | class | `(data, bandwidth='scott', *, weights=None)` | Weighted univariate Gaussian kernel density estimate. |
| `KernelMixtureDistribution` | class | `(locations, bandwidths, weights=None)` | Finite univariate Gaussian-kernel mixture. |
| `Kumaraswamy` | class | `(alpha: 'sp.Expr', beta: 'sp.Expr') -> None` | Kumaraswamy(alpha: 'sp.Expr', beta: 'sp.Expr') |
| `LKJCholesky` | class | `(dimension: 'int', eta: 'Any' = 1)` | LKJ law expressed over correlation-matrix Cholesky factors. |
| `Laplace` | class | `(location: 'sp.Expr' = 0, scale: 'sp.Expr' = 1) -> None` | Laplace(location: 'sp.Expr' = 0, scale: 'sp.Expr' = 1) |
| `LogNormal` | class | `(mean: 'sp.Expr', sigma: 'sp.Expr') -> None` | Log-normal distribution, where log(X) ~ Normal(mean, sigma). |
| `Logistic` | class | `(location: 'sp.Expr' = 0, scale: 'sp.Expr' = 1) -> None` | Logistic(location: 'sp.Expr' = 0, scale: 'sp.Expr' = 1) |
| `MatrixNormal` | class | `(mean: 'Any', row_covariance: 'Any', column_covariance: 'Any')` | Matrix normal ``MN(mean, row_covariance, column_covariance)``. |
| `Maxwell` | class | `(scale: 'sp.Expr' = 1) -> None` | Maxwell(scale: 'sp.Expr' = 1) |
| `Multinomial` | class | `(n: 'Any', probabilities: 'Sequence[Any]')` | Multinomial count-vector distribution. |
| `MultivariateKernelDensityDistribution` | class | `(data, bandwidth='scott', *, weights=None)` | Multivariate Gaussian KDE with a full positive-definite bandwidth matrix. |
| `MultivariateNormal` | class | `(mean: 'Sequence[Any]', covariance: 'Any')` | Multivariate normal distribution with mean vector and covariance matrix. |
| `MultivariateStudentT` | class | `(location, scale, df)` | Multivariate Student-t with scale matrix (not covariance matrix). |
| `Nakagami` | class | `(shape: 'sp.Expr', spread: 'sp.Expr' = 1) -> None` | Nakagami(shape: 'sp.Expr', spread: 'sp.Expr' = 1) |
| `NegativeBinomial` | class | `(r: 'sp.Expr', p: 'sp.Expr') -> None` | Failures before ``r`` successes, with success probability ``p``. |
| `NoncentralChiSquared` | class | `(df: 'sp.Expr', noncentrality: 'sp.Expr') -> None` | NoncentralChiSquared(df: 'sp.Expr', noncentrality: 'sp.Expr') |
| `NoncentralF` | class | `(df1: 'sp.Expr', df2: 'sp.Expr', noncentrality: 'sp.Expr') -> None` | NoncentralF(df1: 'sp.Expr', df2: 'sp.Expr', noncentrality: 'sp.Expr') |
| `NoncentralT` | class | `(df: 'sp.Expr', noncentrality: 'sp.Expr' = 0) -> None` | NoncentralT(df: 'sp.Expr', noncentrality: 'sp.Expr' = 0) |
| `Normal` | class | `(mean: 'sp.Expr', sigma: 'sp.Expr') -> None` | Univariate normal distribution with mean ``mean`` and std. dev. ``sigma``. |
| `ParameterMixtureDistribution` | class | `(mixing: 'Distribution', conditional: 'Callable[[Any], Distribution]', parameter: 'sp.Symbol' = <factory>) -> None` | Mixture formed by integrating a conditional family over a parameter law. |
| `Pareto` | class | `(shape: 'sp.Expr', scale: 'sp.Expr' = 1) -> None` | Pareto(shape: 'sp.Expr', scale: 'sp.Expr' = 1) |
| `Poisson` | class | `(rate: 'sp.Expr') -> None` | Poisson distribution with positive rate ``rate``. |
| `PowerDistribution` | class | `(shape: 'sp.Expr') -> None` | PowerDistribution(shape: 'sp.Expr') |
| `ProbabilityDistribution` | class | `(variable: 'sp.Symbol', density_expression: 'sp.Expr', domain: 'sp.Set' = Reals, discrete: 'bool' = False, cdf_expression: 'sp.Expr \| None' = None, quantile_expression: 'sp.Expr \| None' = None, probability_symbol: 'sp.Symbol \| None' = None, sampler: 'Callable[..., Any] \| None' = None, validate_normalization: 'bool' = True) -> None` | A scalar distribution defined by a symbolic density or mass function. |
| `Rayleigh` | class | `(scale: 'sp.Expr' = 1) -> None` | Rayleigh(scale: 'sp.Expr' = 1) |
| `Rice` | class | `(noncentrality: 'sp.Expr', scale: 'sp.Expr' = 1) -> None` | Rice(noncentrality: 'sp.Expr', scale: 'sp.Expr' = 1) |
| `Skellam` | class | `(rate1: 'sp.Expr', rate2: 'sp.Expr') -> None` | Skellam(rate1: 'sp.Expr', rate2: 'sp.Expr') |
| `SplicedDistribution` | class | `(components: 'Sequence[Distribution]', regions: 'Sequence[sp.Set]', weights: 'Sequence[Any]')` | Piecewise distribution assembled from region-restricted component laws. |
| `StudentT` | class | `(location: 'sp.Expr', scale: 'sp.Expr', df: 'sp.Expr') -> None` | StudentT(location: 'sp.Expr', scale: 'sp.Expr', df: 'sp.Expr') |
| `SymbolicDistribution` | class | `(value_symbol: 'sp.Symbol', log_density: 'sp.Expr', domain: 'sp.Set' = Reals, normalizer_known: 'bool' = False, discrete: 'bool' = False) -> None` | SymbolicDistribution(value_symbol: 'sp.Symbol', log_density: 'sp.Expr', domain: 'sp.Set' = Reals, normalizer_known: 'bool' = False, discrete: 'bool' = False) |
| `Triangular` | class | `(low: 'sp.Expr', mode: 'sp.Expr', high: 'sp.Expr') -> None` | Triangular(low: 'sp.Expr', mode: 'sp.Expr', high: 'sp.Expr') |
| `Uniform` | class | `(low: 'sp.Expr', high: 'sp.Expr') -> None` | Uniform(low: 'sp.Expr', high: 'sp.Expr') |
| `VonMises` | class | `(location: 'sp.Expr' = 0, concentration: 'sp.Expr' = 1) -> None` | VonMises(location: 'sp.Expr' = 0, concentration: 'sp.Expr' = 1) |
| `Weibull` | class | `(shape: 'sp.Expr', scale: 'sp.Expr' = 1) -> None` | Weibull(shape: 'sp.Expr', scale: 'sp.Expr' = 1) |
| `Wishart` | class | `(df: 'Any', scale: 'Any')` | Wishart distribution with degrees of freedom ``df`` and scale matrix ``scale``. |
| `Zipf` | class | `(exponent: 'sp.Expr') -> None` | Zipf(exponent: 'sp.Expr') |
| `cv_bandwidth` | function | `(data: 'Any', weights: 'Any \| None' = None, *, candidates: 'Any \| None' = None, grid_size: 'int' = 31) -> 'float'` | Least-squares cross-validation bandwidth for a Gaussian KDE. |
| `plugin_bandwidth` | function | `(data: 'Any', weights: 'Any \| None' = None) -> 'float'` | One-stage Gaussian plug-in bandwidth using a pilot curvature estimate. |
| `scott_bandwidth` | function | `(data: 'Any', weights: 'Any \| None' = None) -> 'float'` | Scott's rule-of-thumb bandwidth for a univariate Gaussian KDE. |
| `select_bandwidth` | function | `(data: 'Any', method: 'str' = 'scott', weights: 'Any \| None' = None) -> 'float'` | Select a univariate KDE bandwidth by name. |
| `silverman_bandwidth` | function | `(data: 'Any', weights: 'Any \| None' = None) -> 'float'` | Silverman's robust normal-reference bandwidth. |
| `validate_distribution_parameters` | function | `(distribution: 'Distribution') -> 'sp.Basic'` |  |

## Probability functionals
| Name | Kind | Signature | Summary |
| --- | --- | --- | --- |
| `ProbabilityFunctionalError` | class | `` | Inappropriate argument value (of correct type). |
| `cdf` | function | `(dist, value)` | Cumulative distribution function, preserving array-like input shape. |
| `central_moment` | function | `(dist, order: 'int', *, assumptions=None)` | Return a central moment for a distribution or symbolic random expression. |
| `covariance` | function | `(left, right, *, assumptions=None)` | Return covariance of two symbolic random expressions. |
| `cumulant` | function | `(dist, order: 'int', *, assumptions=None)` | Return a distribution cumulant or symbolic random-expression cumulant. |
| `cumulative_hazard` | function | `(dist, value)` | Return ``-log(P(X > x))`` for a scalar distribution. |
| `density` | function | `(dist, value)` | Return density or mass, preserving the shape of array-like input. |
| `entropy` | function | `(dist)` |  |
| `expectation` | function | `(dist, expression=None, *, variable=None, variables=None, numerical_fallback: 'bool' = False, samples: 'int' = 100000, rng=None, return_result: 'bool' = False, assumptions=None)` | Expectation under a distribution or of a symbolic random expression. |
| `factorial_moment` | function | `(dist, order: 'int')` | Return the falling-factorial moment ``E[(X)_order]``. |
| `hazard` | function | `(dist, value)` | Return the hazard at ``value`` using the distribution's measure. |
| `inverse_survival` | function | `(dist, probability)` | Return the inverse survival function for a scalar distribution. |
| `likelihood_value` | function | `(dist, observations)` | Return the joint likelihood of iid observations under ``dist``. |
| `log_density` | function | `(dist, value)` |  |
| `log_likelihood` | function | `(dist, observations)` | Return the iid sample log likelihood under ``dist``. |
| `mean` | function | `(dist, *, assumptions=None)` | Return a distribution mean or symbolic random-expression expectation. |
| `moment` | function | `(dist, order: 'int', *, central: 'bool' = False, assumptions=None)` | Return a raw or central moment for a distribution or random expression. |
| `probability` | function | `(dist, event, *, variable=None, numerical_fallback: 'bool' = False, samples: 'int' = 100000, rng=None, return_result: 'bool' = False)` | Compute ``P(event)`` under ``dist`` using exact-first dispatch. |
| `quantile` | function | `(dist, probability)` | Quantile, preserving the shape of array-like probability input. |
| `raw_moment` | function | `(dist, order: 'int', *, assumptions=None)` | Return a distribution raw moment or symbolic random-expression moment. |
| `survival` | function | `(dist, value)` | Survival function ``P(X > x)``, preserving array-like input shape. |
| `variance` | function | `(dist, ddof=None, *, assumptions=None)` | Return distribution variance or symbolic random-expression variance. |

## Information measures
| Name | Kind | Signature | Summary |
| --- | --- | --- | --- |
| `InformationMeasureError` | class | `` | Raised when an information measure cannot be evaluated safely. |
| `InformationMethod` | class | `` | str(object='') -> str |
| `InformationResult` | class | `(value: 'Any', method: 'InformationMethod', support_relation: 'SupportRelation' = <SupportRelation.PROVEN: 'proven'>, exact: 'bool' = True) -> None` | InformationResult(value: 'Any', method: 'InformationMethod', support_relation: 'SupportRelation' = <SupportRelation.PROVEN: 'proven'>, exact: 'bool' = True) |
| `SupportRelation` | class | `` | str(object='') -> str |
| `bhattacharyya_coefficient` | function | `(p: 'object', q: 'object', *, numerical_fallback: 'bool' = False, samples: 'int' = 100000, rng=None, return_result: 'bool' = False)` | Return the Bhattacharyya coefficient ``∫ sqrt(p q)``. |
| `bhattacharyya_distance` | function | `(p: 'Distribution', q: 'Distribution', **kwargs)` | Return ``-log`` of the Bhattacharyya coefficient. |
| `cross_entropy` | function | `(p: 'object', q: 'object', *, numerical_fallback: 'bool' = False, samples: 'int' = 100000, rng=None, return_result: 'bool' = False)` | Return the cross-entropy ``H(p, q) = -E_p[log q(X)]``. |
| `hellinger_distance` | function | `(p: 'Distribution', q: 'Distribution', **kwargs)` | Return the Hellinger distance ``sqrt(1 - BC(p, q))``. |
| `hellinger_squared` | function | `(p: 'Distribution', q: 'Distribution', **kwargs)` | Return squared Hellinger distance ``1 - BC(p, q)``. |
| `information_entropy` | function | `(p: 'object', *, numerical_fallback: 'bool' = False, samples: 'int' = 100000, rng=None, return_result: 'bool' = False)` | Return Shannon entropy with the same exact/numerical result protocol. |
| `jensen_shannon_divergence` | function | `(p: 'Distribution', q: 'Distribution', *, weight=None, numerical_fallback: 'bool' = False, samples: 'int' = 100000, rng=None, return_result: 'bool' = False)` | Return weighted Jensen-Shannon divergence. |
| `kl_divergence` | function | `(p: 'object', q: 'object', *, numerical_fallback: 'bool' = False, samples: 'int' = 100000, rng=None, return_result: 'bool' = False)` | Return ``D_KL(p \|\| q)``. |
| `register_information_formula` | function | `(measure: 'str', p_type: 'type', q_type: 'type')` | Register a closed-form formula for a distribution pair. |
| `registered_information_formulas` | function | `() -> 'tuple[tuple[str, type, type], ...]'` | Return the registered measure/type keys in deterministic order. |
| `renyi_divergence` | function | `(p: 'object', q: 'object', alpha, *, numerical_fallback: 'bool' = False, samples: 'int' = 100000, rng=None, return_result: 'bool' = False)` | Return Rényi divergence of order ``alpha``. |
| `support_subset` | function | `(p: 'Distribution', q: 'Distribution') -> 'SupportRelation'` | Determine whether ``support(p)`` is contained in ``support(q)``. |

## Survival
| Name | Kind | Signature | Summary |
| --- | --- | --- | --- |
| `KaplanMeierResult` | class | `(data: 'SurvivalData', table: 'RiskTable', event_times: 'np.ndarray', survival: 'np.ndarray', greenwood_sum: 'np.ndarray', variance: 'np.ndarray', standard_error: 'np.ndarray', confidence_level: 'float', lower: 'np.ndarray', upper: 'np.ndarray') -> None` | KaplanMeierResult(data: 'SurvivalData', table: 'RiskTable', event_times: 'np.ndarray', survival: 'np.ndarray', greenwood_sum: 'np.ndarray', variance: 'np.ndarray', standard_error: 'np.ndarray', confidence_level: 'float', lower: 'np.ndarray', upper: 'np.ndarray') |
| `NelsonAalenResult` | class | `(data: 'SurvivalData', table: 'RiskTable', event_times: 'np.ndarray', cumulative_hazard: 'np.ndarray', variance: 'np.ndarray', standard_error: 'np.ndarray', confidence_level: 'float', lower: 'np.ndarray', upper: 'np.ndarray') -> None` | NelsonAalenResult(data: 'SurvivalData', table: 'RiskTable', event_times: 'np.ndarray', cumulative_hazard: 'np.ndarray', variance: 'np.ndarray', standard_error: 'np.ndarray', confidence_level: 'float', lower: 'np.ndarray', upper: 'np.ndarray') |
| `RiskTable` | class | `(time: 'np.ndarray', at_risk: 'np.ndarray', events: 'np.ndarray', censored: 'np.ndarray', entered: 'np.ndarray') -> None` | Counts at each observed exit time. |
| `SurvivalData` | class | `(time: 'np.ndarray', event: 'np.ndarray \| None' = None, entry: 'np.ndarray \| None' = None) -> None` | Observed survival times with censoring and optional delayed entry. |
| `breslow_test` | function | `(*samples, **kwargs)` | Breslow/generalized-Wilcoxon survival-curve test. |
| `fleming_harrington_test` | function | `(*samples, p=0.0, q=0.0, **kwargs)` | Fleming-Harrington weighted log-rank test. |
| `kaplan_meier` | function | `(data, event=None, *, entry=None, confidence_level=0.95)` | Kaplan-Meier product-limit estimator with Greenwood uncertainty. |
| `logrank_test` | function | `(*samples, groups=None, event=None, entry=None, method='logrank', p=0.0, q=0.0)` | Compare two or more survival curves with a weighted log-rank test. |
| `median_survival` | function | `(data, event=None, *, entry=None)` | Return the Kaplan-Meier median, or ``inf`` if survival never reaches 1/2. |
| `nelson_aalen` | function | `(data, event=None, *, entry=None, confidence_level=0.95)` | Nelson-Aalen cumulative-hazard estimator. |
| `risk_table` | function | `(data: 'SurvivalData \| Any', event=None, *, entry=None)` | Return event, censoring, entry, and risk-set counts by exit time. |
| `tarone_ware_test` | function | `(*samples, **kwargs)` | Tarone-Ware survival-curve test. |
