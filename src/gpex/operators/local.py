"""Function values and differential operators evaluated at one location."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .base import FunctionalValue, LinearOperator, Point, PointFunction

Direction = Float[Array, "input_dimension"]


class Identity(LinearOperator):
    def output_shape(self, input_dimension: int) -> tuple[int, ...]:
        return ()

    def apply(self, function: PointFunction, point: Point) -> FunctionalValue:
        return function(point)


class PartialDerivative(LinearOperator):
    axis: int = eqx.field(static=True)

    @property
    def derivative_order(self) -> int:
        return 1

    def output_shape(self, input_dimension: int) -> tuple[int, ...]:
        if not 0 <= self.axis < input_dimension:
            raise ValueError(
                f"axis {self.axis} is invalid for dimension {input_dimension}"
            )
        return ()

    def apply(self, function: PointFunction, point: Point) -> FunctionalValue:
        self.output_shape(point.shape[-1])
        tangent = jnp.zeros_like(point).at[self.axis].set(1)
        return jax.jvp(function, (point,), (tangent,))[1]


class Gradient(LinearOperator):
    @property
    def derivative_order(self) -> int:
        return 1

    def output_shape(self, input_dimension: int) -> tuple[int, ...]:
        return (input_dimension,)

    def apply(self, function: PointFunction, point: Point) -> FunctionalValue:
        return jax.jacfwd(function)(point)


class Hessian(LinearOperator):
    @property
    def derivative_order(self) -> int:
        return 2

    def output_shape(self, input_dimension: int) -> tuple[int, ...]:
        return (input_dimension, input_dimension)

    def apply(self, function: PointFunction, point: Point) -> FunctionalValue:
        return jax.jacfwd(jax.jacfwd(function))(point)


class Laplacian(LinearOperator):
    """Return the trace of the input Hessian at each location.

    Sum second coordinate derivatives without constructing the full Hessian.
    """

    @property
    def derivative_order(self) -> int:
        return 2

    def output_shape(self, input_dimension: int) -> tuple[int, ...]:
        return ()

    def apply(self, function: PointFunction, point: Point) -> FunctionalValue:
        def second_derivative(axis: int) -> FunctionalValue:
            tangent = jax.nn.one_hot(axis, point.shape[0])

            def first_derivative(variable: Point) -> FunctionalValue:
                return jax.jvp(function, (variable,), (tangent,))[1]

            return jax.jvp(first_derivative, (point,), (tangent,))[1]

        def accumulate(axis: int, total: FunctionalValue) -> FunctionalValue:
            return total + second_derivative(axis)

        # Preserve array-valued intermediate functions from another operator.
        initial = jnp.zeros_like(function(point))
        return jax.lax.fori_loop(0, point.shape[0], accumulate, initial)


class DirectionalDerivative(LinearOperator):
    direction: Direction

    def __init__(self, direction: Direction) -> None:
        direction = jnp.asarray(direction)
        if direction.ndim != 1:
            raise ValueError("direction must be a one-dimensional array")
        self.direction = direction

    @property
    def derivative_order(self) -> int:
        return 1

    def output_shape(self, input_dimension: int) -> tuple[int, ...]:
        if self.direction.shape != (input_dimension,):
            raise ValueError(
                "direction shape must match the locations' input dimension"
            )
        return ()

    def apply(self, function: PointFunction, point: Point) -> FunctionalValue:
        return jax.jvp(function, (point,), (self.direction,))[1]


__all__ = [
    "Identity",
    "PartialDerivative",
    "Gradient",
    "Hessian",
    "Laplacian",
    "DirectionalDerivative",
]
