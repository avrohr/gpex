"""Differentiable exact GP objectives."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from ..blocks import FunctionalDataset
from ..covariance_systems import CovarianceSystem
from ..gp import GaussianProcess
from ..moments.assembly import assemble_mean
from ..moments.measurements import prepare_functionals
from ..parameters import log_prior

ScalarObjective = Float[Array, ""]


def negative_log_marginal_likelihood(
    gp: GaussianProcess,
    dataset: FunctionalDataset,
    *,
    jitter: float = 1e-6,
) -> ScalarObjective:
    """Return the exact Gaussian negative log marginal likelihood.

    Invalid factorizations produce non-finite trials for optimizers to reject.
    Padding contributes neither observations nor likelihood normalization.
    """
    measurements = prepare_functionals(tuple(block.design for block in dataset.blocks))
    variance = dataset.variance_flat
    system = CovarianceSystem(measurements, gp.kernel, variance, jitter)
    prepared = gp.solver.prepare(
        system,
        materialization_tile_size=gp.covariance_tile_size,
        active=measurements.active,
    )
    residual = dataset.values_flat - assemble_mean(measurements, gp.mean)
    quadratic = prepared.quadratic_form(residual)
    logdet = prepared.logdet()
    normalizer = dataset.active_size * jnp.log(2 * jnp.pi)
    return 0.5 * (quadratic + logdet + normalizer)


def negative_log_posterior(
    gp: GaussianProcess,
    dataset: FunctionalDataset,
    *,
    jitter: float = 1e-6,
) -> ScalarObjective:
    """Add explicitly declared GP and noise priors to the exact likelihood."""
    likelihood = negative_log_marginal_likelihood(gp, dataset, jitter=jitter)
    return likelihood - log_prior((gp, dataset), require_all=True)


__all__ = ["negative_log_marginal_likelihood", "negative_log_posterior"]
