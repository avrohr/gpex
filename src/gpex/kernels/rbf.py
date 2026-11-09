"""Squared-exponential kernels with ordinary or generalized lengthscales."""

from __future__ import annotations

from math import inf

import equinox as eqx
import jax
import jax.numpy as jnp

from ..parameters import CholeskyParameter, PositiveParameter
from .base import Scalar
from .lengthscales import (
    CholeskyFactor,
    GeneralizedLengthscales,
    Lengthscale,
    Lengthscales,
)
from .radial import RadialFeatureKernel
from .separable import KernelFactorization, SeparableKernel


def _rbf_correlation(squared_distance: Scalar) -> Scalar:
    return jnp.exp(-0.5 * squared_distance)


class _RBFFactor(eqx.Module):
    lengthscale: Scalar

    def __call__(self, left: Scalar, right: Scalar) -> Scalar:
        return _rbf_correlation(((left - right) / self.lengthscale) ** 2)


class RBFKernel(RadialFeatureKernel, SeparableKernel):
    """Squared-exponential kernel with isotropic or ARD lengthscales.

    Args:
        variance: Positive scalar covariance amplitude or scalar ``PositiveParameter``.
            Defaults to 1.
        lengthscale: Positive scalar for isotropic distances or vector of shape
            ``(d,)`` for ARD. Use a ``PositiveParameter`` to make it trainable.
            Defaults to 1; values are ordinary lengthscales, not their reciprocals.

    Uses squared distance ``sum(((x - y) / lengthscale)**2)``.
    Supports derivatives of every order. Provides random Fourier features.
    """

    lengthscales: Lengthscales
    _correlation = staticmethod(_rbf_correlation)
    _sample_frequencies = staticmethod(jax.random.normal)
    _derivative_order = inf

    def __init__(
        self,
        *,
        variance: PositiveParameter | Scalar | float = 1.0,
        lengthscale: PositiveParameter | Lengthscale | float = 1.0,
    ) -> None:
        super().__init__(Lengthscales(lengthscale), variance=variance)

    def factorization(self, input_dimension: int) -> KernelFactorization:
        """Expose scalar RBF factors using the current scalar or ARD lengths."""
        self.lengthscales.validate_dimension(input_dimension)
        lengths = jnp.broadcast_to(self.lengthscales.lengthscale, (input_dimension,))
        return KernelFactorization(
            self.variance, tuple(_RBFFactor(length) for length in lengths)
        )


class GeneralizedRBFKernel(RadialFeatureKernel):
    """Squared-exponential kernel with a full Mahalanobis metric.

    Args:
        variance: Positive scalar covariance amplitude or scalar ``PositiveParameter``.
            Defaults to 1.
        L: Complete lower-triangular factor of shape ``(d, d)`` with positive
            diagonal. Use a ``CholeskyParameter`` to make it trainable.

    Uses squared distance ``||(L.T @ (x - y))||²``, with precision matrix
    ``M = L @ L.T``. A diagonal ``L`` contains reciprocal lengthscales.
    Supports derivatives of every order. Provides random Fourier features.
    """

    lengthscales: GeneralizedLengthscales
    _correlation = staticmethod(_rbf_correlation)
    _sample_frequencies = staticmethod(jax.random.normal)
    _derivative_order = inf

    def __init__(
        self,
        *,
        L: CholeskyParameter | CholeskyFactor,
        variance: PositiveParameter | Scalar | float = 1.0,
    ) -> None:
        super().__init__(GeneralizedLengthscales(L), variance=variance)


__all__ = ["RBFKernel", "GeneralizedRBFKernel"]
