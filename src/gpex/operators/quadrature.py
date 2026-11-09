"""Explicit quadrature rules for scalar linear measurements."""

from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from ..validation.checks import error_if
from .base import FunctionalValue, LinearOperator, Point, PointFunction

QuadraturePoints = Float[Array, "nodes input_dimension"]
QuadratureWeights = Float[Array, "nodes"]
QuadratureRule = tuple[QuadraturePoints, QuadratureWeights]
CoordinateNodes = Float[Array, "nodes"]
CoordinateRule = tuple[CoordinateNodes, QuadratureWeights]
TensorQuadratureRule = tuple[CoordinateRule, ...]
BoxRadius = Float[Array, "#input_dimension"]
GaussianScale = Float[Array, "#input_dimension"]
WaveletScale = Float[Array, ""]
IntegralBase = Float[Array, ""]


class QuadratureOperator(LinearOperator):
    """A scalar linear measurement defined by weighted function evaluations.

    Subclasses implement ``quadrature(point)`` to return evaluation points of
    shape ``(q, d)`` and weights of shape ``(q,)``. Both may depend on the query
    location and operator parameters, but not on the function being evaluated.
    Weights may be signed and are not normalized by this class.

    The rule must use JAX-compatible operations and keep ``q`` fixed under
    tracing. ``apply`` supports JIT, batching, and differentiation through the
    points and weights. It also preserves trailing function-output axes when
    operators are nested during covariance compilation.
    """

    def output_shape(self, input_dimension: int) -> tuple[int, ...]:
        """Return scalar output; subclasses may validate the input dimension."""
        return ()

    @abstractmethod
    def quadrature(self, point: Point) -> QuadratureRule:
        """Return absolute evaluation points ``(q, d)`` and weights ``(q,)``.

        Include the rule's normalization and any change-of-variable factors
        in the weights. The point count and input dimension are static.
        """

    def apply(self, function: PointFunction, point: Point) -> FunctionalValue:
        self.output_shape(point.shape[-1])
        points, weights = self.quadrature(point)
        values = jax.vmap(function)(points)
        return jnp.tensordot(
            weights, values, axes=(0, 0), precision=jax.lax.Precision.HIGHEST
        )


class TensorQuadratureOperator(QuadratureOperator):
    """A scalar quadrature rule that factors over input coordinates.

    Implement ``coordinate_quadrature(point)`` to return one pair of node and
    weight vectors per coordinate. Nodes are absolute coordinate values, and
    full weights are products of coordinate weights. Both may depend on the
    entire query point and operator parameters, independently of the function.
    Node counts are static under tracing and may differ between coordinates.

    The inherited ``apply`` evaluates the Cartesian-product rule. Covariance
    evaluation with a ``SeparableKernel`` contracts coordinate rules directly,
    preserving the same approximation without constructing the full grid.
    Overrides of ``apply`` or ``quadrature`` must represent that same rule.
    """

    @abstractmethod
    def coordinate_quadrature(self, point: Point) -> TensorQuadratureRule:
        """Return ``((nodes_0, weights_0), ...)`` in input-coordinate order.

        Each pair contains equally sized one-dimensional arrays. Include
        normalization and change-of-variable factors in the weights; signed
        and unnormalized rules are supported.
        """

    def quadrature(self, point: Point) -> QuadratureRule:
        rules = self.coordinate_quadrature(point)
        points = jnp.stack(
            jnp.meshgrid(*(nodes for nodes, _ in rules), indexing="ij"), axis=-1
        ).reshape(-1, point.shape[-1])
        weights = jnp.prod(
            jnp.stack(
                jnp.meshgrid(*(weights for _, weights in rules), indexing="ij"),
                axis=-1,
            ),
            axis=-1,
        ).reshape(-1)
        return points, weights


class BoxAverage(TensorQuadratureOperator):
    """Average a scalar function over a box centered at each input location.

    A scalar radius defines a square or hypercube. A radius vector defines one
    half-width per input dimension. The integral is approximated by a symmetric
    tensor-product midpoint rule with ``quadrature_order ** dimension`` points.

    Construction and evaluation support JIT, batching, and differentiation of
    finite positive ``radius`` values. Quadrature order remains static.
    """

    radius: BoxRadius
    quadrature_order: int = eqx.field(static=True)

    def __init__(self, radius: BoxRadius | float, *, quadrature_order: int = 8) -> None:
        radius = jnp.asarray(radius)
        if radius.ndim > 1 or radius.size == 0:
            raise ValueError("box-average radius must be scalar or one-dimensional")
        if quadrature_order < 1:
            raise ValueError("quadrature order must be positive")
        self.radius = error_if(
            radius,
            jnp.any(~jnp.isfinite(radius) | (radius <= 0)),
            "box-average radius must be finite and strictly positive",
        )
        self.quadrature_order = quadrature_order

    def output_shape(self, input_dimension: int) -> tuple[int, ...]:
        if self.radius.ndim == 1 and self.radius.shape != (input_dimension,):
            raise ValueError("radius shape must match the locations' input dimension")
        return ()

    def coordinate_quadrature(self, point: Point) -> TensorQuadratureRule:
        input_dimension = point.shape[-1]
        radius = jnp.broadcast_to(self.radius, (input_dimension,))
        normalized_nodes = (
            2.0 * (jnp.arange(self.quadrature_order) + 0.5) / self.quadrature_order
            - 1.0
        )
        weights = jnp.full((self.quadrature_order,), 1.0 / self.quadrature_order)
        return tuple(
            (point[axis] + normalized_nodes * radius[axis], weights)
            for axis in range(input_dimension)
        )


class GaussianSmoothing(TensorQuadratureOperator):
    """Average a scalar function under a centered Gaussian density.

    A scalar scale defines an isotropic Gaussian. A scale vector defines one
    standard deviation per input dimension. The expectation is approximated
    with tensor-product Gauss--Hermite quadrature using
    ``quadrature_order ** dimension`` points.

    Construction and evaluation support JIT, batching, and differentiation of
    finite positive ``scale`` values. Quadrature order remains static.
    """

    scale: GaussianScale
    quadrature_order: int = eqx.field(static=True)
    _nodes: tuple[float, ...] = eqx.field(static=True)
    _weights: tuple[float, ...] = eqx.field(static=True)

    def __init__(
        self, scale: GaussianScale | float, *, quadrature_order: int = 8
    ) -> None:
        scale = jnp.asarray(scale)
        if scale.ndim > 1 or scale.size == 0:
            raise ValueError(
                "Gaussian-smoothing scale must be scalar or one-dimensional"
            )
        nodes, weights = np.polynomial.hermite.hermgauss(quadrature_order)
        self.scale = error_if(
            scale,
            jnp.any(~jnp.isfinite(scale) | (scale <= 0)),
            "Gaussian-smoothing scale must be finite and strictly positive",
        )
        self.quadrature_order = quadrature_order
        self._nodes = tuple(float(node * np.sqrt(2.0)) for node in nodes)
        self._weights = tuple(float(weight / np.sqrt(np.pi)) for weight in weights)

    def output_shape(self, input_dimension: int) -> tuple[int, ...]:
        if self.scale.ndim == 1 and self.scale.shape != (input_dimension,):
            raise ValueError("scale shape must match the locations' input dimension")
        return ()

    def coordinate_quadrature(self, point: Point) -> TensorQuadratureRule:
        input_dimension = point.shape[-1]
        scale = jnp.broadcast_to(self.scale, (input_dimension,))
        nodes = jnp.asarray(self._nodes)
        weights = jnp.asarray(self._weights)

        return tuple(
            (point[axis] + nodes * scale[axis], weights)
            for axis in range(input_dimension)
        )


class CumulativeIntegral1D(QuadratureOperator):
    """Integrate a one-dimensional function from ``base`` to each location.

    Gauss--Legendre quadrature is mapped from ``[-1, 1]`` to the oriented
    interval from ``base`` to ``x``. Locations below ``base`` therefore produce
    the correctly signed integral, and the value at ``base`` is exactly zero.
    """

    base: IntegralBase
    quadrature_order: int = eqx.field(static=True)
    _nodes: tuple[float, ...] = eqx.field(static=True)
    _weights: tuple[float, ...] = eqx.field(static=True)

    def __init__(
        self, base: IntegralBase | float, *, quadrature_order: int = 8
    ) -> None:
        base = jnp.asarray(base)
        if base.ndim != 0:
            raise ValueError("integration base must be scalar")
        nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
        self.base = base
        self.quadrature_order = quadrature_order
        self._nodes = tuple(float(node) for node in nodes)
        self._weights = tuple(float(weight) for weight in weights)

    def output_shape(self, input_dimension: int) -> tuple[int, ...]:
        if input_dimension != 1:
            raise ValueError(
                "CumulativeIntegral1D only supports one-dimensional inputs"
            )
        return ()

    def quadrature(self, point: Point) -> QuadratureRule:
        nodes = jnp.asarray(self._nodes)
        weights = jnp.asarray(self._weights)
        half_interval = (point[0] - self.base) / 2.0
        midpoint = (point[0] + self.base) / 2.0
        quadrature_points = midpoint + half_interval * nodes
        return quadrature_points[:, None], half_interval * weights


class HaarWavelet1D(QuadratureOperator):
    """A translated, fixed-scale Haar-wavelet coefficient in one dimension.

    ``scale`` is the full width of the wavelet support. The analyzing wavelet
    is positive on the left half, negative on the right half, has unit
    $L^2$ norm, and therefore annihilates constants. Each half-integral
    is approximated with a symmetric midpoint rule.

    Construction and evaluation support JIT, batching, and differentiation of
    finite positive ``scale`` values. Quadrature order remains static.
    """

    scale: WaveletScale
    quadrature_order: int = eqx.field(static=True)

    def __init__(
        self, scale: WaveletScale | float, *, quadrature_order: int = 8
    ) -> None:
        scale = jnp.asarray(scale)
        if scale.ndim != 0:
            raise ValueError("Haar-wavelet scale must be scalar")
        if quadrature_order < 1:
            raise ValueError("quadrature order must be positive")
        self.scale = error_if(
            scale,
            ~jnp.isfinite(scale) | (scale <= 0),
            "Haar-wavelet scale must be finite and strictly positive",
        )
        self.quadrature_order = quadrature_order

    def output_shape(self, input_dimension: int) -> tuple[int, ...]:
        if input_dimension != 1:
            raise ValueError("HaarWavelet1D currently supports one-dimensional inputs")
        return ()

    def quadrature(self, point: Point) -> QuadratureRule:
        nodes = (jnp.arange(self.quadrature_order) + 0.5) / self.quadrature_order
        half_width = self.scale / 2.0
        left_offsets = -half_width + nodes * half_width
        right_offsets = nodes * half_width
        offsets = jnp.concatenate((left_offsets, right_offsets))[:, None]
        signs = jnp.concatenate(
            (jnp.ones(self.quadrature_order), -jnp.ones(self.quadrature_order))
        )
        weights = signs * jnp.sqrt(self.scale) / (2 * self.quadrature_order)
        return point + offsets, weights


__all__ = [
    "BoxAverage",
    "CumulativeIntegral1D",
    "GaussianSmoothing",
    "HaarWavelet1D",
    "QuadratureOperator",
    "TensorQuadratureOperator",
]
