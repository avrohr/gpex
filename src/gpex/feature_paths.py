"""Callable posterior draws using kernel-provided finite features."""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from .blocks import FunctionalBlock, Locations
from .gp import ConditionedPosterior, ObservationVector
from .kernels.features import FeatureKernel
from .moments.assembly import ScalarFunction, assemble_covariance, assemble_mean
from .moments.measurements import PreparedFunctionals
from .operators.local import Identity

Point = Float[Array, "input_dimension"]
Scalar = Float[Array, ""]


class _FeaturePath(eqx.Module):
    features: Callable[[Point], Array]
    weights: Array

    def __call__(self, point: Point) -> Scalar:
        return jnp.dot(
            self.weights, self.features(point), precision=jax.lax.Precision.HIGHEST
        )


class FeaturePosteriorPath[KernelType: FeatureKernel](eqx.Module):
    """A callable posterior draw built from a finite kernel feature map.

    The path uses the kernel's random features for the prior and a pathwise
    conditioning correction for the observations. Repeated calls evaluate the
    same draw, so the object can be vectorized, differentiated, or passed to a
    [`LinearOperator`][gpex.operators.LinearOperator]. Construct with
    ``FeaturePosteriorPath(posterior, key)``. For exact samples at a fixed finite query,
    use ``posterior.sample(key, query)``. Differentiating a finite feature
    expansion beyond the GP's smoothness does not define a valid GP derivative.

    Args:
        posterior: Conditioned GP whose kernel implements ``FeatureKernel``.
            RBF and Matérn families provide Fourier features. Custom feature
            kernels may provide other finite bases.
        key: JAX random key for this draw.
        feature_count: Number of features. Defaults to 1024. For the built-in
            kernels, larger maps improve the prior approximation statistically.
    """

    prior_path: _FeaturePath
    mean: ScalarFunction
    kernel: KernelType
    measurements: PreparedFunctionals
    correction_weights: Array
    covariance_tile_size: int | None = eqx.field(static=True)
    input_dimension: int = eqx.field(static=True)

    def __init__(
        self,
        posterior: ConditionedPosterior[KernelType],
        key: PRNGKeyArray,
        *,
        feature_count: int = 1024,
    ) -> None:
        conditioning = posterior.conditioning
        kernel = conditioning.gp.kernel
        if not isinstance(kernel, FeatureKernel):
            raise TypeError(
                "FeaturePosteriorPath requires a FeatureKernel; "
                f"got {type(kernel).__name__}. "
                "Use posterior.sample(key, query) for finite-query samples."
            )
        input_dimension = conditioning.measurements.input_dimension
        feature_key, weight_key, noise_key = jax.random.split(key, 3)
        features = kernel.random_features(
            feature_key, input_dimension, feature_count=feature_count
        )
        prior_path = _FeaturePath(
            features=features,
            weights=jax.random.normal(weight_key, (feature_count,)),
        )

        self.prior_path = prior_path
        self.mean = conditioning.gp.mean
        self.kernel = kernel
        self.measurements = conditioning.measurements
        self.correction_weights = _conditioning_weights(
            posterior, prior_path, noise_key
        )
        self.covariance_tile_size = conditioning.gp.covariance_tile_size
        self.input_dimension = input_dimension

    def __call__(self, point: Point) -> Scalar:
        point = jnp.asarray(point)
        if point.shape != (self.input_dimension,):
            raise ValueError(f"path points must have shape ({self.input_dimension},)")
        value = FunctionalBlock(
            Locations(point[None, :]), Identity(), "posterior_path_value"
        )
        cross_covariance = assemble_covariance(
            (value,),
            self.measurements,
            self.kernel,
            tile_size=self.covariance_tile_size,
        )[0]
        correction = jnp.dot(
            cross_covariance,
            self.correction_weights,
            precision=jax.lax.Precision.HIGHEST,
        )
        return self.mean(point) + self.prior_path(point) + correction


def _conditioning_weights(
    posterior: ConditionedPosterior,
    prior_path: ScalarFunction,
    noise_key: PRNGKeyArray,
) -> ObservationVector:
    """Condition a zero-mean prior draw independently of its representation."""
    conditioning = posterior.conditioning
    prior_observations = assemble_mean(conditioning.measurements, prior_path)
    observation_mean = assemble_mean(conditioning.measurements, conditioning.gp.mean)
    effective_variance = conditioning.variance_flat + conditioning.jitter
    noise_draw = jnp.sqrt(effective_variance) * jax.random.normal(
        noise_key, effective_variance.shape
    )
    return conditioning.solve(
        conditioning.measurements.mask(
            posterior.y_flat - observation_mean - prior_observations - noise_draw
        )
    )


__all__ = ["FeaturePosteriorPath"]
