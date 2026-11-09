"""Matérn kernels, analytic radial derivatives, and Fourier frequency sampling.

The radial functions take squared radii including the factor ``2 * nu``.
Their derivative identities avoid cancellation and ``0 * inf`` terms at
coincident points. They preserve spatial derivatives through order two for
Matérn 3/2 and order four for Matérn 5/2, as required for their supported
derivative-observation covariances.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from ..parameters import CholeskyParameter, PositiveParameter
from .base import Scalar
from .lengthscales import (
    CholeskyFactor,
    GeneralizedLengthscales,
    Lengthscale,
    Lengthscales,
)
from .radial import RadialFeatureKernel


def _radius(squared_radius: Scalar) -> Scalar:
    # The exact value at zero is kept separate from sqrt's singular derivative.
    # No epsilon is added: nonzero distances retain the original kernel.
    zero = squared_radius == 0
    return jnp.where(zero, 0.0, jnp.sqrt(jnp.where(zero, 1.0, squared_radius)))


def matern12(squared_radius: Scalar) -> Scalar:
    return jnp.exp(-_radius(squared_radius))


@jax.custom_jvp
def matern32(squared_radius: Scalar) -> Scalar:
    radius = _radius(squared_radius)
    return (1.0 + radius) * jnp.exp(-radius)


@matern32.defjvp
def _matern32_jvp(primals, tangents):
    (squared_radius,), (tangent,) = primals, tangents
    return matern32(squared_radius), -0.5 * matern12(squared_radius) * tangent


@jax.custom_jvp
def matern52(squared_radius: Scalar) -> Scalar:
    radius = _radius(squared_radius)
    return (1.0 + radius + squared_radius / 3.0) * jnp.exp(-radius)


@matern52.defjvp
def _matern52_jvp(primals, tangents):
    (squared_radius,), (tangent,) = primals, tangents
    return matern52(squared_radius), -matern32(squared_radius) * tangent / 6.0


def _matern32_correlation(squared_distance: Scalar) -> Scalar:
    return matern32(3.0 * squared_distance)


def _matern52_correlation(squared_distance: Scalar) -> Scalar:
    return matern52(5.0 * squared_distance)


def _sample_matern(key: PRNGKeyArray, shape: tuple[int, int], *, degrees: int) -> Array:
    normal_key, scale_key = jax.random.split(key)
    normal = jax.random.normal(normal_key, shape)
    # One chi-squared scale per vector gives a multivariate Student-t draw.
    chi_squared = 2.0 * jax.random.gamma(scale_key, degrees / 2.0, (shape[0], 1))
    return normal * jnp.sqrt(degrees / chi_squared)


class Matern12Kernel(RadialFeatureKernel):
    """Matérn-1/2 kernel with scalar or ARD lengthscales.

    PositiveParameter shapes, meanings, and defaults match
    [`RBFKernel`][gpex.kernels.RBFKernel].
    Supports value observations, with no derivative observations.
    Fourier features use multivariate Student-t frequencies with 1
    degree of freedom.
    """

    lengthscales: Lengthscales
    _correlation = staticmethod(matern12)
    _sample_frequencies = staticmethod(partial(_sample_matern, degrees=1))
    _derivative_order = 0

    def __init__(
        self,
        *,
        variance: PositiveParameter | Scalar | float = 1.0,
        lengthscale: PositiveParameter | Lengthscale | float = 1.0,
    ) -> None:
        super().__init__(Lengthscales(lengthscale), variance=variance)


class GeneralizedMatern12Kernel(RadialFeatureKernel):
    """Matérn-1/2 kernel with a full precision metric.

    PositiveParameter shapes, meanings, and defaults match
    [`GeneralizedRBFKernel`][gpex.kernels.GeneralizedRBFKernel].
    Supports value observations, with no derivative observations.
    Fourier features use multivariate Student-t frequencies with 1
    degree of freedom.
    """

    lengthscales: GeneralizedLengthscales
    _correlation = staticmethod(matern12)
    _sample_frequencies = staticmethod(partial(_sample_matern, degrees=1))
    _derivative_order = 0

    def __init__(
        self,
        *,
        L: CholeskyParameter | CholeskyFactor,
        variance: PositiveParameter | Scalar | float = 1.0,
    ) -> None:
        super().__init__(GeneralizedLengthscales(L), variance=variance)


class Matern32Kernel(RadialFeatureKernel):
    """Matérn-3/2 kernel with scalar or ARD lengthscales.

    PositiveParameter shapes, meanings, and defaults match
    [`RBFKernel`][gpex.kernels.RBFKernel].
    Supports values and first derivatives.
    Fourier features use multivariate Student-t frequencies with 3
    degrees of freedom.
    """

    lengthscales: Lengthscales
    _correlation = staticmethod(_matern32_correlation)
    _sample_frequencies = staticmethod(partial(_sample_matern, degrees=3))
    _derivative_order = 1

    def __init__(
        self,
        *,
        variance: PositiveParameter | Scalar | float = 1.0,
        lengthscale: PositiveParameter | Lengthscale | float = 1.0,
    ) -> None:
        super().__init__(Lengthscales(lengthscale), variance=variance)


class GeneralizedMatern32Kernel(RadialFeatureKernel):
    """Matérn-3/2 kernel with a full precision metric.

    PositiveParameter shapes, meanings, and defaults match
    [`GeneralizedRBFKernel`][gpex.kernels.GeneralizedRBFKernel].
    Supports values and first derivatives.
    Fourier features use multivariate Student-t frequencies with 3
    degrees of freedom.
    """

    lengthscales: GeneralizedLengthscales
    _correlation = staticmethod(_matern32_correlation)
    _sample_frequencies = staticmethod(partial(_sample_matern, degrees=3))
    _derivative_order = 1

    def __init__(
        self,
        *,
        L: CholeskyParameter | CholeskyFactor,
        variance: PositiveParameter | Scalar | float = 1.0,
    ) -> None:
        super().__init__(GeneralizedLengthscales(L), variance=variance)


class Matern52Kernel(RadialFeatureKernel):
    """Matérn-5/2 kernel with scalar or ARD lengthscales.

    PositiveParameter shapes, meanings, and defaults match
    [`RBFKernel`][gpex.kernels.RBFKernel].
    Supports values, first derivatives, and second derivatives.
    Fourier features use multivariate Student-t frequencies with 5
    degrees of freedom.
    """

    lengthscales: Lengthscales
    _correlation = staticmethod(_matern52_correlation)
    _sample_frequencies = staticmethod(partial(_sample_matern, degrees=5))
    _derivative_order = 2

    def __init__(
        self,
        *,
        variance: PositiveParameter | Scalar | float = 1.0,
        lengthscale: PositiveParameter | Lengthscale | float = 1.0,
    ) -> None:
        super().__init__(Lengthscales(lengthscale), variance=variance)


class GeneralizedMatern52Kernel(RadialFeatureKernel):
    """Matérn-5/2 kernel with a full precision metric.

    PositiveParameter shapes, meanings, and defaults match
    [`GeneralizedRBFKernel`][gpex.kernels.GeneralizedRBFKernel].
    Supports values, first derivatives, and second derivatives.
    Fourier features use multivariate Student-t frequencies with 5
    degrees of freedom.
    """

    lengthscales: GeneralizedLengthscales
    _correlation = staticmethod(_matern52_correlation)
    _sample_frequencies = staticmethod(partial(_sample_matern, degrees=5))
    _derivative_order = 2

    def __init__(
        self,
        *,
        L: CholeskyParameter | CholeskyFactor,
        variance: PositiveParameter | Scalar | float = 1.0,
    ) -> None:
        super().__init__(GeneralizedLengthscales(L), variance=variance)


__all__ = [
    "Matern12Kernel",
    "GeneralizedMatern12Kernel",
    "Matern32Kernel",
    "GeneralizedMatern32Kernel",
    "Matern52Kernel",
    "GeneralizedMatern52Kernel",
]
