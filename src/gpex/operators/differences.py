"""Differences of function values at displaced locations."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from .base import FunctionalValue, LinearOperator, Point, PointFunction

Displacement = Float[Array, "input_dimension"]


class Increment(LinearOperator):
    """Return the unnormalized increment ``f(x + displacement) - f(x)``."""

    displacement: Displacement

    def __init__(self, displacement: Displacement) -> None:
        displacement = jnp.asarray(displacement)
        if displacement.ndim != 1 or displacement.size == 0:
            raise ValueError("displacement must be a non-empty one-dimensional array")
        self.displacement = displacement

    def output_shape(self, input_dimension: int) -> tuple[int, ...]:
        if self.displacement.shape != (input_dimension,):
            raise ValueError(
                "displacement shape must match the locations' input dimension"
            )
        return ()

    def apply(self, function: PointFunction, point: Point) -> FunctionalValue:
        self.output_shape(point.shape[-1])
        return function(point + self.displacement) - function(point)


__all__ = ["Increment"]
