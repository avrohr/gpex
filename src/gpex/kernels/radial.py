"""Shared radial evaluation and random Fourier features."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from math import inf
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from ..parameters import PositiveParameter
from ..validation.checks import error_if
from .base import Kernel, Point, Scalar
from .features import FeatureKernel
from .lengthscales import GeneralizedLengthscales, Lengthscales


class _FourierFeatures(eqx.Module):
    frequencies: Array
    phases: Array
    scale: Scalar

    def __call__(self, point: Point) -> Array:
        if point.shape != (self.frequencies.shape[1],):
            raise ValueError("feature points must match the input dimension")
        return self.scale * jnp.cos(
            jnp.matmul(self.frequencies, point, precision=jax.lax.Precision.HIGHEST)
            + self.phases
        )


class RadialKernel(Kernel):
    """Kernel evaluated from squared distance after lengthscale transformation.

    Args:
        lengthscales: ``Lengthscales`` or ``GeneralizedLengthscales`` owning the input
            scaling and its parameters.
        variance: Finite positive scalar amplitude or a scalar ``PositiveParameter``.

    Subclasses supply ``_correlation(squared_distance)`` as a static callable
    and set ``_derivative_order`` to the largest supported latent derivative
    order. The default assumes unlimited smoothness. Subclass
    ``RadialFeatureKernel`` to additionally provide random Fourier features.

    Radiality is with respect to the transformed coordinates and does not imply
    a product over the original coordinates. Evaluation and numerical state support
    JIT, vmap, and differentiation.
    """

    variance_parameter: PositiveParameter | Scalar
    lengthscales: Lengthscales | GeneralizedLengthscales
    _correlation: ClassVar[Callable[[Scalar], Scalar]]
    _derivative_order: ClassVar[float] = inf

    def __init__(
        self,
        lengthscales: Lengthscales | GeneralizedLengthscales,
        *,
        variance: PositiveParameter | Scalar | float,
    ) -> None:
        value = (
            variance.value
            if isinstance(variance, PositiveParameter)
            else jnp.asarray(variance)
        )
        if value.ndim != 0:
            raise ValueError("variance must be scalar")
        self.variance_parameter = (
            variance
            if isinstance(variance, PositiveParameter)
            else error_if(
                value,
                jnp.any(~jnp.isfinite(value) | (value <= 0)),
                "variance must be finite and positive",
            )
        )
        self.lengthscales = lengthscales

    @property
    def variance(self) -> Scalar:
        if isinstance(self.variance_parameter, PositiveParameter):
            return self.variance_parameter.value
        return self.variance_parameter

    @property
    def max_derivative_order(self) -> float:
        return self._derivative_order

    def __call__(self, left: Point, right: Point) -> Scalar:
        if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape:
            raise ValueError("kernel points must have the same one-dimensional shape")
        self.lengthscales.validate_dimension(left.shape[0])
        displacement = self.lengthscales.transform_displacement(left - right)
        return self.variance * self._correlation(jnp.sum(displacement**2))


class RadialFeatureKernel(RadialKernel, FeatureKernel):
    """A radial kernel with a sampler for its normalized spectral distribution.

    Subclasses supply ``_sample_frequencies(key, shape)`` with
    ``shape = (feature_count, input_dimension)``. The shared implementation
    applies the same lengthscale convention to distances and frequency draws.
    """

    # Reuse the radial constructor, including its variance validation.
    __init__ = RadialKernel.__init__

    @staticmethod
    @abstractmethod
    def _sample_frequencies(key: PRNGKeyArray, shape: tuple[int, int]) -> Array:
        """Draw frequencies for the unit-variance, unit-lengthscale correlation."""
        ...

    def random_features(
        self,
        key: PRNGKeyArray,
        input_dimension: int,
        *,
        feature_count: int = 1024,
    ) -> Callable[[Point], Array]:
        """Construct Fourier features; see
        [`FeatureKernel.random_features`][gpex.kernels.FeatureKernel.random_features].
        """
        if (
            isinstance(feature_count, bool)
            or not isinstance(feature_count, int)
            or feature_count < 1
        ):
            raise ValueError("feature_count must be a positive integer")
        if (
            isinstance(input_dimension, bool)
            or not isinstance(input_dimension, int)
            or input_dimension < 1
        ):
            raise ValueError("input_dimension must be a positive integer")
        self.lengthscales.validate_dimension(input_dimension)
        frequency_key, phase_key = jax.random.split(key)
        frequencies = self.lengthscales.transform_frequencies(
            self._sample_frequencies(frequency_key, (feature_count, input_dimension))
        )
        phases = jax.random.uniform(
            phase_key, (feature_count,), minval=0.0, maxval=2.0 * jnp.pi
        )
        return _FourierFeatures(
            frequencies, phases, jnp.sqrt(2.0 * self.variance / feature_count)
        )


__all__ = ["RadialKernel", "RadialFeatureKernel"]
