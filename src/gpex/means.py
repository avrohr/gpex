"""Scalar mean functions and their composition."""

from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .parameters import Parameter

Scalar = Float[Array, ""]
Point = Float[Array, "input_dimension"]
MeanWeights = Float[Array, "input_dimension"]


class Mean(eqx.Module):
    """Base class providing composition for scalar mean functions."""

    def __add__(self, other: Mean) -> SumMean:
        return SumMean(self, other)

    def __mul__(self, other: Mean) -> ProductMean:
        return ProductMean(self, other)

    @abstractmethod
    def __call__(self, point: Point) -> Scalar:
        """Evaluate the scalar mean at a point."""
        ...


class SumMean(Mean):
    left: Mean
    right: Mean

    def __call__(self, point: Point) -> Scalar:
        return self.left(point) + self.right(point)


class ProductMean(Mean):
    left: Mean
    right: Mean

    def __call__(self, point: Point) -> Scalar:
        return self.left(point) * self.right(point)


class ZeroMean(Mean):
    def __call__(self, point: Point) -> Scalar:
        return jnp.zeros(())


class ConstantMean(Mean):
    """A scalar real floating mean, fixed or explicitly trainable.

    Supply values in JAX's default floating precision, such as ``0.0``.
    """

    constant: Parameter | Scalar

    def __init__(self, value: Parameter | Scalar | float = 0.0) -> None:
        constant = value if isinstance(value, Parameter) else jnp.asarray(value)
        constrained = constant.value if isinstance(constant, Parameter) else constant
        if constrained.ndim != 0:
            raise ValueError("constant mean must be scalar")
        self.constant = constant

    @property
    def value(self) -> Scalar:
        if isinstance(self.constant, Parameter):
            return self.constant.value
        return self.constant

    def __call__(self, point: Point) -> Scalar:
        del point
        return self.value


class LinearMean(Mean):
    """Linear mean ``weights @ point + intercept`` with explicit parameters."""

    weights_parameter: Parameter | MeanWeights
    intercept_parameter: Parameter | Scalar

    def __init__(
        self,
        weights: Parameter | MeanWeights,
        intercept: Parameter | Scalar | float = 0.0,
    ) -> None:
        weights = weights if isinstance(weights, Parameter) else jnp.asarray(weights)
        weights_value = weights.value if isinstance(weights, Parameter) else weights
        if weights_value.ndim != 1 or weights_value.size == 0:
            raise ValueError("linear-mean weights must be a nonempty vector")

        intercept = (
            intercept if isinstance(intercept, Parameter) else jnp.asarray(intercept)
        )
        intercept_value = (
            intercept.value if isinstance(intercept, Parameter) else intercept
        )
        if intercept_value.ndim != 0:
            raise ValueError("linear-mean intercept must be scalar")
        self.weights_parameter = weights
        self.intercept_parameter = intercept

    @property
    def weights(self) -> MeanWeights:
        if isinstance(self.weights_parameter, Parameter):
            return self.weights_parameter.value
        return self.weights_parameter

    @property
    def intercept(self) -> Scalar:
        if isinstance(self.intercept_parameter, Parameter):
            return self.intercept_parameter.value
        return self.intercept_parameter

    def __call__(self, point: Point) -> Scalar:
        if point.shape != self.weights.shape:
            raise ValueError("point shape must match linear-mean weights")
        return (
            jnp.dot(self.weights, point, precision=jax.lax.Precision.HIGHEST)
            + self.intercept
        )


__all__ = ["Mean", "SumMean", "ProductMean", "ZeroMean", "ConstantMean", "LinearMean"]
