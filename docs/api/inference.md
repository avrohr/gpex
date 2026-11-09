# Inference

`GaussianProcess.condition(dataset)` returns a `ConditionedPosterior`.
For a complete example, start with [First posterior](../tutorials/first-posterior.ipynb).
The posterior accepts a `FunctionalQuery` for prediction and sampling.

## Prediction methods

| Method | Result |
| --- | --- |
| `predict_mean(query)` | Named means |
| `predict_variance(query)` | Named marginal variances |
| `predict_marginals(query)` | Means and marginal variances together |
| `predict_covariance(query)` | Full joint covariance |
| `predict(query)` | Means and full joint covariance together |
| `sample(key, query, sample_count=...)` | Joint Gaussian draws at the query |

The marginal methods avoid constructing a full query covariance. Use
`predict_marginals` for pointwise uncertainty bands. Use a full covariance
when correlations between query outputs matter.

## Named results and shapes

For a query block named `"target"` with public shape `s`:

| Access | Shape |
| --- | --- |
| `prediction.mean["target"]` | `s` |
| `marginals.variance["target"]` | `s` |
| `prediction.covariance.variance["target"]` | `s` |
| `prediction.covariance.block("left", "right")` | `(*left_shape, *right_shape)` |
| `samples["target"]` | `(sample_count, *s)` |

A block at `n` locations has shape `(n, *operator_output_shape)`.
For a joint query with a scalar block and a gradient block in `d` dimensions,
their cross-covariance has shape `(n, n, d)`. Sample rows correspond to the
same joint draw across blocks. Query predictions describe latent functionals.
To obtain a covariance for future noisy readings, add their measurement-error
variances to the appropriate diagonal entries.

Prediction results can cross an outer `jit`. Under `vmap`, named arrays retain
the leading batch axes. Dataset and query input dimensions must agree.

## Conditioning and numerical options

`gp.condition(dataset, jitter=1e-6)` adds jitter to the observation covariance
diagonal before solving. Jitter is an absolute covariance quantity and must be
finite and nonnegative. Choose it in the units of your model and keep it
consistent between fitting and conditioning. Observation noise is specified
by the dataset's design. Sampling has its own `jitter` argument for stabilizing
the query covariance factorization.

The default dense solver supports first-order forward and reverse derivatives
through conditioning and updates. Higher-order derivatives through its
factorization and updates are unsupported. Failed Cholesky factorizations can
produce non-finite results, including under JIT. Check outputs before using
them. Use consistent [floating precision](../getting-started.md#floating-precision),
sensible units, and explicit regularization for redundant measurements.
A jitter too small relative to the covariance diagonal can round away.

`GaussianProcess(kernel, covariance_tile_size=1024)` evaluates observation
covariances in smaller tiles. Tile size counts scalar outputs, with at least
one location's complete output per tile. Smaller tiles reduce temporary memory
at the cost of more loop overhead. The final dense covariance and its
factorization still have to fit in memory. The default is `None`.
See [numerical backends](backends.md) for solver configuration and interfaces.

## Proposed measurements and posterior updates

`gp.prepare_conditioning(design)` returns a `ConditioningState` for covariance
calculations before outcomes are available. `PreparedQuery(conditioning, query)`
reuses a fixed target when comparing proposed measurements. Preparation stores
the full target covariance, including for subsequent variance-only requests.
Rebuild it after changing the GP, existing design, jitter, or target.

`posterior.append(locations, readings, block="name")` updates an existing
posterior within a declared block's capacity. `posterior.conditioning.append`
performs a covariance update without readings. See
[Experimental design](../tutorials/design-of-experiments.ipynb) and
[Sequential observations](../tutorials/sequential-streaming.ipynb) for their use,
and [observation capacity](../concepts/jax-and-capacity.md) for compilation and
model-reuse conditions.

## Classes and methods

::: gpex.gp
    options:
      members:
        - GaussianProcess
        - ConditioningState
        - ConditionedPosterior

::: gpex.results
    options:
      members:
        - Prediction
        - MarginalPrediction
        - CovarianceResult

::: gpex.prepared_query.PreparedQuery

Feature-based callable draws require a `FeatureKernel`. RBF and Matérn families
provide Fourier features, while custom feature kernels can use other bases.
`GaussianProcess`, `ConditioningState`, and `ConditionedPosterior` preserve the
kernel type so this requirement is visible to static type checkers.

::: gpex.feature_paths
    options:
      members:
        - FeaturePosteriorPath
