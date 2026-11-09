"""Exact coordinate-product representations of scalar covariance kernels."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable

import equinox as eqx

from .base import Kernel, Point, Scalar

CoordinateKernel = Callable[[Scalar, Scalar], Scalar]


class KernelFactorization(eqx.Module):
    """Represent ``coefficient * product(factor_j(x_j, y_j))``.

    ``coefficient`` is a scalar array, and ``factors`` contains one scalar
    two-argument callable per input coordinate, in coordinate order. Factors
    need not be stationary. Their numerical state must remain dynamic PyTree
    leaves and their evaluation must support JAX transformations.

    Construct this representation from the kernel's current values; it does
    not own additional trainable parameters or approximate the kernel.
    """

    coefficient: Scalar
    factors: tuple[CoordinateKernel, ...]


class SeparableKernel(Kernel):
    """A kernel with an exact product of factors in the input coordinates.

    Subclasses implement ``factorization(input_dimension)``. The default
    point evaluation multiplies the factors; subclasses may supply an
    equivalent direct implementation. A sum of separable kernels is generally
    not itself separable and remains a ``SumKernel``.
    """

    @abstractmethod
    def factorization(self, input_dimension: int) -> KernelFactorization:
        """Return one factor per coordinate and a scalar coefficient.

        Validate the requested dimension against kernel parameter shapes.
        Derive factors from current numerical leaves so differentiation
        through this representation agrees with point evaluation.
        """

    def __call__(self, left: Point, right: Point) -> Scalar:
        if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape:
            raise ValueError("kernel points must have the same one-dimensional shape")
        representation = self.factorization(left.shape[0])
        value = representation.coefficient
        for factor, x, y in zip(representation.factors, left, right, strict=True):
            value = value * factor(x, y)
        return value


__all__ = ["KernelFactorization", "SeparableKernel"]
