"""Scalar covariance interface and composition."""

from __future__ import annotations

from abc import abstractmethod
from math import inf

import equinox as eqx
from jaxtyping import Array, Float

Scalar = Float[Array, ""]
Point = Float[Array, "input_dimension"]


class Kernel(eqx.Module):
    """Base class providing composition for scalar covariance kernels."""

    def __add__(self, other: Kernel) -> SumKernel:
        return SumKernel(self, other)

    def __mul__(self, other: Kernel) -> ProductKernel:
        return ProductKernel(self, other)

    @abstractmethod
    def __call__(self, left: Point, right: Point) -> Scalar:
        """Evaluate the scalar covariance between two points."""
        ...

    @property
    def max_derivative_order(self) -> float:
        """Largest mean-square derivative order; override for finite smoothness."""
        return inf


class SumKernel(Kernel):
    left: Kernel
    right: Kernel

    def __call__(self, left: Point, right: Point) -> Scalar:
        return self.left(left, right) + self.right(left, right)

    @property
    def max_derivative_order(self) -> float:
        return min(self.left.max_derivative_order, self.right.max_derivative_order)


class ProductKernel(Kernel):
    left: Kernel
    right: Kernel

    def __call__(self, left: Point, right: Point) -> Scalar:
        return self.left(left, right) * self.right(left, right)

    @property
    def max_derivative_order(self) -> float:
        return min(self.left.max_derivative_order, self.right.max_derivative_order)


__all__ = ["Kernel", "SumKernel", "ProductKernel"]
