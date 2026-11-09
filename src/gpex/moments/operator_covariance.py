"""Type-directed covariance evaluation for linear measurement pairs."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array
from plum import dispatch

from ..kernels.base import Scalar, SumKernel
from ..kernels.separable import SeparableKernel
from ..operators.base import LinearOperator, Point
from ..operators.quadrature import TensorQuadratureOperator


@dispatch
def operator_covariance(
    kernel: Callable,
    left: LinearOperator,
    right: LinearOperator,
    *,
    x: Point,
    y: Point,
) -> Array:
    """Apply both operators, with output axes ordered left then right.

    Only the kernel and operator classes participate in multiple dispatch.
    Numerical locations are keyword-only and remain dynamic under JAX.

    Return annotations use plain ``Array`` because Plum checks them at runtime.
    Binding named output axes would couple unrelated pairs in a jaxtyping context.
    """
    return right.apply(lambda v: left.apply(lambda u: kernel(u, v), x), y)


@operator_covariance.dispatch
def _tensor_left(
    kernel: SeparableKernel,
    left: TensorQuadratureOperator,
    right: LinearOperator,
    *,
    x: Point,
    y: Point,
) -> Array:
    representation = kernel.factorization(x.shape[0])
    rules = left.coordinate_quadrature(x)

    def function(v: Point) -> Scalar:
        value = representation.coefficient
        for factor, (nodes, weights), coordinate in zip(
            representation.factors, rules, v, strict=True
        ):
            values = jax.vmap(factor, in_axes=(0, None))(nodes, coordinate)
            value = value * jnp.dot(
                weights, values, precision=jax.lax.Precision.HIGHEST
            )
        return value

    return right.apply(function, y)


@operator_covariance.dispatch
def _tensor_right(
    kernel: SeparableKernel,
    left: LinearOperator,
    right: TensorQuadratureOperator,
    *,
    x: Point,
    y: Point,
) -> Array:
    representation = kernel.factorization(x.shape[0])
    rules = right.coordinate_quadrature(y)

    def function(u: Point) -> Scalar:
        value = representation.coefficient
        for factor, (nodes, weights), coordinate in zip(
            representation.factors, rules, u, strict=True
        ):
            values = jax.vmap(factor, in_axes=(None, 0))(coordinate, nodes)
            value = value * jnp.dot(
                weights, values, precision=jax.lax.Precision.HIGHEST
            )
        return value

    return left.apply(function, x)


@operator_covariance.dispatch
def _tensor_both(
    kernel: SeparableKernel,
    left: TensorQuadratureOperator,
    right: TensorQuadratureOperator,
    *,
    x: Point,
    y: Point,
) -> Array:
    representation = kernel.factorization(x.shape[0])
    left_rules = left.coordinate_quadrature(x)
    right_rules = right.coordinate_quadrature(y)
    value = representation.coefficient
    for factor, (left_nodes, left_weights), (right_nodes, right_weights) in zip(
        representation.factors, left_rules, right_rules, strict=True
    ):
        matrix = jax.vmap(jax.vmap(factor, in_axes=(None, 0)), in_axes=(0, None))(
            left_nodes, right_nodes
        )
        value = value * jnp.dot(
            left_weights,
            jnp.dot(matrix, right_weights, precision=jax.lax.Precision.HIGHEST),
            precision=jax.lax.Precision.HIGHEST,
        )
    return value


@operator_covariance.dispatch
def _sum(
    kernel: SumKernel,
    left: LinearOperator,
    right: LinearOperator,
    *,
    x: Point,
    y: Point,
) -> Array:
    return operator_covariance(
        kernel.left, left, right, x=x, y=y
    ) + operator_covariance(kernel.right, left, right, x=x, y=y)
