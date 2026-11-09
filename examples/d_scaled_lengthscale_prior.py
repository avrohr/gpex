"""Fit an ARD RBF with a dimension-scaled lengthscale prior.

Reference: Carl Hvarfner, Erik O. Hellsten, and Luigi Nardi (ICML 2024),
"Vanilla Bayesian Optimization Performs Great in High Dimensions",
Sections 5.1 and 6: https://proceedings.mlr.press/v235/hvarfner24a.html

Adding 0.5 * log(d) to the log-normal location scales lengthscales by sqrt(d).
This example illustrates the prior in a small regression fit.
"""

# Run with JAX_ENABLE_X64=1 for the recommended precision.

import math

import jax
import jax.numpy as jnp
import optax

from gpex import (
    FunctionalBlock,
    FunctionalDataset,
    GaussianProcess,
    ObservationBlock,
    ObservationDesignBlock,
)
from gpex.blocks import Locations
from gpex.fitting import fit, negative_log_posterior
from gpex.kernels import RBFKernel
from gpex.operators import Identity
from gpex.parameters import PositiveParameter
from gpex.parameters.priors import MultivariateLogNormalPrior

dimension = 4
log_mean = math.sqrt(2.0) + 0.5 * math.log(dimension)
log_standard_deviation = math.sqrt(3.0)
prior_mode = math.exp(log_mean - log_standard_deviation**2)

# A diagonal log-covariance gives independent priors for the ARD lengthscales.
lengthscale_prior = MultivariateLogNormalPrior(
    log_mean=jnp.full((dimension,), log_mean),
    log_covariance=jnp.eye(dimension) * log_standard_deviation**2,
)
lengthscales = PositiveParameter(
    jnp.full((dimension,), prior_mode),
    prior=lengthscale_prior,
    name="ard_lengthscales",
)

# Use inputs in [0, 1]^d and standardized outputs, with kernel variance fixed
# at one as in the paper's model setup.
points = jax.random.uniform(jax.random.key(7), (32, dimension))
raw_values = (
    jnp.sin(2.0 * jnp.pi * points[:, 0])
    + 0.5 * points[:, 1]
    - 0.25 * points[:, 2] * points[:, 3]
)
values = (raw_values - raw_values.mean()) / raw_values.std()
measured_values = FunctionalBlock(Locations(points), Identity(), "values")
measurement_design = ObservationDesignBlock(
    measured_values, variance=0.01, max_size=points.shape[0]
)
dataset = FunctionalDataset([ObservationBlock(measurement_design, y=values)])
gp = GaussianProcess(
    RBFKernel(
        variance=1.0,
        lengthscale=lengthscales,
    )
)

result = fit(
    gp,
    lambda candidate: negative_log_posterior(candidate, dataset),
    optax.adam(0.03),
    step_count=100,
)

print("prior: dimension-scaled independent log-normal lengthscales")
print("log mean:", lengthscale_prior.log_mean)
print("log covariance:", lengthscale_prior.log_covariance)
print("prior mode:", prior_mode)
print("initial ARD lengthscales:", gp.kernel.lengthscales.lengthscale)
print("fitted ARD lengthscales:", result.model.kernel.lengthscales.lengthscale)
print("initial/final MAP loss:", result.losses[0], result.losses[-1])
