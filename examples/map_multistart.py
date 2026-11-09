"""Fit kernel and observation-noise parameters from informed MAP starts."""

# Run with JAX_ENABLE_X64=1 for the recommended precision.

import jax.numpy as jnp

from gpex import (
    FunctionalBlock,
    FunctionalDataset,
    GaussianProcess,
    ObservationBlock,
    ObservationDesignBlock,
)
from gpex.blocks import Locations
from gpex.fitting import fit_lbfgs, fit_multistart, negative_log_posterior
from gpex.kernels import RBFKernel
from gpex.operators import Identity
from gpex.parameters import PositiveParameter
from gpex.parameters.priors import GammaPrior, LogNormalPrior

# Exact synthetic readings drive the fitted noise toward zero. Use explicit
# regularization and a moderate gradient target so this example is well posed
# in both precisions. Jitter changes the fitted system and is separate from noise.
JITTER = 0.01
GRADIENT_TOLERANCE = 1e-3
points = jnp.linspace(-1.0, 1.0, 12)[:, None]
observations = jnp.sin(2 * points[:, 0])


def make_model(
    variance: float,
    lengthscale: float,
    observation_variance: float,
) -> tuple[GaussianProcess, FunctionalDataset]:
    gp = GaussianProcess(
        RBFKernel(
            variance=PositiveParameter(
                variance,
                prior=GammaPrior(concentration=2.0, rate=1.0),
                name="kernel_variance",
            ),
            lengthscale=PositiveParameter(
                lengthscale,
                prior=LogNormalPrior(
                    log_mean=0.0,
                    log_standard_deviation=0.7,
                ),
                name="kernel_lengthscale",
            ),
        )
    )
    measured_values = FunctionalBlock(Locations(points), Identity(), "values")
    variance_parameter = PositiveParameter(
        observation_variance,
        prior=GammaPrior(concentration=2.0, rate=20.0),
        name="observation_variance",
    )
    measurement_design = ObservationDesignBlock(
        measured_values, variance=variance_parameter, max_size=points.shape[0]
    )
    dataset = FunctionalDataset([ObservationBlock(measurement_design, y=observations)])
    return gp, dataset


def objective(
    candidate: tuple[GaussianProcess, FunctionalDataset],
) -> jnp.ndarray:
    candidate_gp, candidate_dataset = candidate
    return negative_log_posterior(candidate_gp, candidate_dataset, jitter=JITTER)


starts = (
    make_model(variance=0.2, lengthscale=0.2, observation_variance=0.01),
    make_model(variance=1.0, lengthscale=0.8, observation_variance=0.05),
    make_model(variance=4.0, lengthscale=2.0, observation_variance=0.2),
)
result = fit_multistart(
    starts,
    lambda model: fit_lbfgs(model, objective, gradient_tolerance=GRADIENT_TOLERANCE),
)
assert jnp.all(result.converged)
fitted_gp, fitted_dataset = result.model

print("loss by start:", result.final_losses)
print("steps by start:", result.step_counts)
print("gradient norm by start:", result.gradient_norms)
print("objective evaluations by start:", result.evaluation_counts)
print("best start:", result.best_index)
print("variance:", fitted_gp.kernel.variance)
print("lengthscale:", fitted_gp.kernel.lengthscales.lengthscale)
print("noise variance:", fitted_dataset.variance_flat[0])
