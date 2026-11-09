"""Ordinary and generalized lengthscale representations for radial kernels."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..parameters import CholeskyParameter, Parameter, PositiveParameter
from ..validation.checks import error_if
from ..validation.cholesky import validate_cholesky_factor
from .base import Point

CholeskyFactor = Float[Array, "dimension dimension"]
Lengthscale = Float[Array, "#input_dimension"]


class Lengthscales(eqx.Module):
    """Ordinary scalar or ARD lengthscales, stored once as model parameters.

    Args:
        lengthscale: Finite positive scalar or vector of shape ``(d,)``. Use a
            ``PositiveParameter`` to make it trainable. A scalar applies to
            every input coordinate, and a vector supplies one scale per axis.

    Displacements and rows of Fourier frequencies are divided by these values.
    Construction and numerical operations support JIT, vmap, and differentiation.
    """

    lengthscale_parameter: PositiveParameter | Lengthscale

    def __init__(self, lengthscale: PositiveParameter | Lengthscale | float) -> None:
        value = (
            lengthscale.value
            if isinstance(lengthscale, PositiveParameter)
            else jnp.asarray(lengthscale)
        )
        if value.ndim > 1 or value.size == 0:
            raise ValueError("lengthscale must be scalar or a nonempty vector")
        self.lengthscale_parameter = (
            lengthscale
            if isinstance(lengthscale, PositiveParameter)
            else error_if(
                value,
                jnp.any(~jnp.isfinite(value) | (value <= 0)),
                "lengthscale must be finite and positive",
            )
        )

    @property
    def lengthscale(self) -> Lengthscale:
        if isinstance(self.lengthscale_parameter, PositiveParameter):
            return self.lengthscale_parameter.value
        return self.lengthscale_parameter

    def validate_dimension(self, input_dimension: int) -> None:
        lengthscale = self.lengthscale
        if lengthscale.ndim != 0 and lengthscale.shape != (input_dimension,):
            raise ValueError("ARD lengthscale shape must match the input dimension")

    def transform_displacement(self, displacement: Point) -> Point:
        return displacement / self.lengthscale

    def transform_frequencies(self, frequencies: Array) -> Array:
        return frequencies / self.lengthscale


class GeneralizedLengthscales(eqx.Module):
    """Generalized lengthscales represented by a lower-triangular precision factor.

    Args:
        L: Finite lower-triangular array of shape ``(d, d)`` with positive
            diagonal, or a ``CholeskyParameter`` for a trainable factor.

    The metric is ``L @ L.T``. Displacements are transformed by ``L.T``,
    and rows of Fourier frequencies by right-multiplication with ``L.T``.
    A diagonal ``L`` contains reciprocal ordinary lengthscales. Off-diagonal
    entries allow directions of variation that are not aligned with the axes.
    Construction and numerical operations support JIT, vmap, and differentiation.
    """

    L_parameter: CholeskyParameter | CholeskyFactor

    def __init__(self, L: CholeskyParameter | CholeskyFactor) -> None:
        if isinstance(L, Parameter) and not isinstance(L, CholeskyParameter):
            raise TypeError("trainable L must be a CholeskyParameter")
        # A CholeskyParameter already owns the matrix invariants.
        self.L_parameter = (
            L if isinstance(L, CholeskyParameter) else validate_cholesky_factor(L)
        )

    @property
    def L(self) -> CholeskyFactor:
        if isinstance(self.L_parameter, CholeskyParameter):
            return self.L_parameter.value
        return self.L_parameter

    def validate_dimension(self, input_dimension: int) -> None:
        if self.L.shape[0] != input_dimension:
            raise ValueError("points must match the L dimension")

    def transform_displacement(self, displacement: Point) -> Point:
        return jnp.matmul(self.L.T, displacement, precision=jax.lax.Precision.HIGHEST)

    def transform_frequencies(self, frequencies: Array) -> Array:
        return jnp.matmul(frequencies, self.L.T, precision=jax.lax.Precision.HIGHEST)


__all__ = ["Lengthscales", "GeneralizedLengthscales"]
