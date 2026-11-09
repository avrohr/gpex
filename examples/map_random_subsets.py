"""Approximate MAP fitting with a fresh random subset at every update."""

# Run with JAX_ENABLE_X64=1 for the recommended precision.

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
from gpex.fitting import (
    fit_random_subsets,
    negative_log_marginal_likelihood,
    negative_log_posterior,
)
from gpex.kernels import RBFKernel
from gpex.operators import Identity
from gpex.parameters import PositiveParameter, log_prior
from gpex.parameters.priors import GammaPrior, LogNormalPrior

points = jnp.linspace(-3.0, 3.0, 48)[:, None]
values = jnp.sin(1.7 * points[:, 0])

gp = GaussianProcess(
    RBFKernel(
        variance=PositiveParameter(0.7, prior=GammaPrior(2.0, 2.0)),
        lengthscale=PositiveParameter(1.2, prior=LogNormalPrior(-0.2, 0.5)),
    )
)
measured_values = FunctionalBlock(Locations(points), Identity(), "values")
variance = PositiveParameter(0.05, prior=GammaPrior(2.0, 40.0))
measurement_design = ObservationDesignBlock(
    measured_values, variance=variance, max_size=points.shape[0]
)
dataset = FunctionalDataset([ObservationBlock(measurement_design, y=values)])

subset_size = 12
likelihood_scale = points.shape[0] / subset_size


def exact_map_objective(candidate):
    candidate_gp, candidate_dataset = candidate
    return negative_log_posterior(candidate_gp, candidate_dataset)


def subset_map_objective(candidate):
    candidate_gp, selected_dataset = candidate
    approximate_likelihood = likelihood_scale * negative_log_marginal_likelihood(
        candidate_gp,
        selected_dataset,
    )
    return approximate_likelihood - log_prior(
        (candidate_gp, selected_dataset.design),
        require_all=True,
    )


result = fit_random_subsets(
    (gp, dataset),
    subset_map_objective,
    subset_size=subset_size,
    reference_objective=exact_map_objective,
    key=jax.random.key(7),
    optimizer=optax.adam(0.02),
    step_count=100,
)

fitted_gp, fitted_dataset = result.model
print(f"reference loss: {float(result.reference_loss):.3f}")
print(f"variance:       {float(fitted_gp.kernel.variance):.3f}")
print(f"lengthscale:    {float(fitted_gp.kernel.lengthscales.lengthscale):.3f}")
print(f"noise variance: {float(fitted_dataset.blocks[0].design.variances[0]):.3f}")
