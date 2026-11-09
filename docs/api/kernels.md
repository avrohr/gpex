# Kernels and means

## Built-in kernel families

Standard kernels accept `variance=1.0` and `lengthscale=1.0` by default.
`variance` is the covariance amplitude. A scalar lengthscale gives isotropic
scaling, and a vector supplies one lengthscale per input coordinate (ARD).

```python
import jax.numpy as jnp
from gpex.kernels import RBFKernel, Matern52Kernel

isotropic = RBFKernel(variance=1.0, lengthscale=0.7)
ard = Matern52Kernel(variance=1.0, lengthscale=jnp.array([0.7, 1.2]))
```

| Class | Kernel family | Supported GP derivative order |
| --- | --- | --- |
| `Matern12Kernel` | Matérn, ν = 1/2 | 0 |
| `Matern32Kernel` | Matérn, ν = 3/2 | 1 |
| `Matern52Kernel` | Matérn, ν = 5/2 | 2 |
| `RBFKernel` | Squared exponential | Any finite order |

All four families support [callable posterior paths](../tutorials/posterior-paths.ipynb).
Kernel sums and products, constructed with `+` and `*`, support conditioning,
prediction, and finite-query sampling. They do not currently provide features
for callable paths.

## Full input metrics

The `Generalized...Kernel` variants of these four families accept a complete
lower-triangular factor `L` with a positive diagonal. It defines the precision
metric `M = L @ L.T`, with distance `norm(L.T @ (x - y))`:

```python
from gpex.kernels import GeneralizedMatern52Kernel

L = jnp.array([[1.4, 0.0], [0.3, 0.8]])
kernel = GeneralizedMatern52Kernel(variance=1.0, L=L)
```

For a diagonal `L`, its diagonal entries are reciprocal lengthscales.
`L = jnp.diag(1 / jnp.array([0.7, 1.2]))` reproduces the corresponding ARD metric.
Generalized kernels also support callable paths.

Plain numbers and arrays remain fixed during fitting. Use `PositiveParameter`
for variance and lengthscales, or `CholeskyParameter` for `L`, to make them
trainable. Read values through `kernel.variance`, `kernel.lengthscales.lengthscale`,
or `kernel.lengthscales.L`. See [Fitting hyperparameters](../tutorials/mle-map.ipynb)
and [parameters and priors](parameters.md).

## Compose kernels and means

Use `+` and `*` to compose kernels or means. For example, this two-dimensional
model combines a trainable full input metric with a product of two fixed RBF
kernels and a linear mean:

```python
from gpex import GaussianProcess
from gpex.kernels import GeneralizedRBFKernel
from gpex.means import ConstantMean, LinearMean
from gpex.parameters import CholeskyParameter
from gpex.parameters.priors import IndependentCholeskyPrior, LogNormalPrior, NormalPrior

factor = CholeskyParameter(
    jnp.array([[1.0, 0.0], [0.2, 1.4]]),
    prior=IndependentCholeskyPrior(
        diagonal_prior=LogNormalPrior(log_standard_deviation=0.5),
        off_diagonal_prior=NormalPrior(standard_deviation=0.4),
    ),
    name="precision_factor",
)
local = GeneralizedRBFKernel(variance=1.0, L=factor)
broad = RBFKernel(variance=0.3, lengthscale=2.0)
modulation = RBFKernel(lengthscale=jnp.array([1.5, 0.8]))
mean = LinearMean(weights=jnp.array([0.15, -0.1])) + ConstantMean(0.2)
gp = GaussianProcess(kernel=local + broad * modulation, mean=mean)
```

Here only the precision factor is trainable. The kernel variances, fixed
lengthscales, and mean coefficients remain fixed.

## Extension interfaces

A `Kernel` maps two points of shape `(d,)` to a scalar covariance. A `Mean` maps
one point to a scalar mean. Both are Equinox modules. Custom kernels must be
positive semidefinite and declare `max_derivative_order` when the latent GP has
finite smoothness. Evaluation must support the JAX transformations required by
the chosen operators, including derivatives at coincident inputs.

`RadialKernel` supplies input scaling and evaluation from squared distance.
Subclasses define `_correlation` and `_derivative_order`. `RadialFeatureKernel`
also requires `_sample_frequencies` for its unit-scale correlation.

`SeparableKernel.factorization(input_dimension)` returns a
`KernelFactorization(coefficient, factors)`. Each factor is a scalar-coordinate
kernel, and their product times the coefficient must equal the kernel and its
derivatives. Derive factors from current parameter values. Factors with array
state should be Equinox modules. This interface enables
[factored tensor quadrature](../concepts/linear-operators.md#factored-tensor-quadrature).
Pass the kernel object directly to `GaussianProcess` to expose this capability.

`FeatureKernel.random_features` returns a fixed callable mapping one point to
a feature vector. Feature inner products approximate the complete kernel,
including its variance. Store random draws as array fields so repeated calls
use the same features and the map can cross JIT boundaries. Exact finite feature
maps are supported as well. See the class contracts below.

## Classes

::: gpex.kernels
    options:
      members:
        - Kernel
        - FeatureKernel
        - RadialKernel
        - RadialFeatureKernel
        - Lengthscales
        - GeneralizedLengthscales
        - SeparableKernel
        - KernelFactorization
        - RBFKernel
        - Matern12Kernel
        - Matern32Kernel
        - Matern52Kernel
        - GeneralizedRBFKernel
        - GeneralizedMatern12Kernel
        - GeneralizedMatern32Kernel
        - GeneralizedMatern52Kernel
        - SumKernel
        - ProductKernel

`ZeroMean()` has no configuration. Mean composition uses
`SumMean(left: Mean, right: Mean)` and `ProductMean(left: Mean, right: Mean)`,
also available through `+` and `*`.

::: gpex.means
    options:
      members:
        - Mean
        - ZeroMean
        - ConstantMean
        - LinearMean
        - SumMean
        - ProductMean
