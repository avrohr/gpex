# Parameters and priors

## Parameter values

Only explicit `Parameter` leaves are fitted. Plain numbers and arrays remain
fixed. Parameters store unconstrained optimization coordinates and expose the
model value through `.value`.

| Constructor | Initial value | Constraint |
| --- | --- | --- |
| `Parameter(unconstrained=...)` | Optimization coordinates | Identity transform by default |
| `PositiveParameter(value)` | Positive model value | Softplus |
| `CholeskyParameter(value)` | Lower-triangular model matrix | Positive diagonal and zero upper triangle |

`PositiveParameter` is used for trainable kernel variance, lengthscales, and
scalar observation-noise variance. `CholeskyParameter` is used for a full
input precision factor. A custom transform supplies `forward` and `inverse`.
Pass `transform.inverse(value)` to the base constructor's `unconstrained`
argument when starting from a constrained value.

See [Fitting hyperparameters](../tutorials/mle-map.ipynb) for a complete example.

## Priors

A prior's `log_prob` receives the constrained model value and returns one scalar
joint log density. Scalar prior families sum independent contributions for
array-valued parameters. `MultivariateLogNormalPrior` describes the joint normal
distribution of the log-values. A correlated prior on ARD lengthscales still
uses an axis-aligned kernel metric.

MAP requires a prior on every trainable parameter. MLE ignores attached priors.
The optimization transform adds no Jacobian term to the MAP objective, which
is defined on the constrained model values.

For a full precision factor `L`, `IndependentCholeskyPrior` uses separate
priors for positive diagonal and unconstrained strict-lower entries.
`ScaledLKJCholeskyPrior` describes `L = diag(scales) @ C`, with positive scales
and a correlation Cholesky factor `C`. These correlations describe the input
precision metric. The prior includes the density correction from scales and
correlation coordinates to the free entries of `L`.

Custom priors provide a scalar `log_prob(value)`. Built-in transforms and priors
implement the `Transform` and `Prior` interfaces below.

## Classes and functions

::: gpex.parameters
    options:
      members:
        - Transform
        - Parameter
        - PositiveParameter
        - CholeskyParameter
        - IdentityTransform
        - ExpTransform
        - SoftplusTransform
        - CholeskyTransform
        - log_prior

::: gpex.parameters.priors
    options:
      members:
        - Prior
        - NormalPrior
        - LogNormalPrior
        - MultivariateLogNormalPrior
        - GammaPrior
        - IndependentCholeskyPrior
        - ScaledLKJCholeskyPrior
