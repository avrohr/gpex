"""Locations and named linear functionals."""

from __future__ import annotations

from collections.abc import Sequence
from math import prod

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..operators.base import LinearOperator

Points = Float[Array, "points input_dimension"]
Point = Float[Array, "input_dimension"]
FunctionalValues = Float[Array, "points *output_shape"]


class Locations(eqx.Module):
    """Non-empty real locations in JAX's default floating precision.

    Args:
        points: Floating array of shape ``(n, d)``: ``n`` locations, each with
            ``d`` coordinates. Callers supply floating values. For a one-dimensional
            vector ``x`` of shape ``(n,)``, use ``x[:, None]`` to add the input axis.
    """

    points: Points

    def __init__(self, points: Points) -> None:
        points = jnp.asarray(points)
        if points.ndim != 2:
            raise ValueError("locations must have shape (n, d)")
        if points.shape[0] == 0 or points.shape[1] == 0:
            raise ValueError("locations cannot have empty axes")
        self.points = points

    @property
    def count(self) -> int:
        return self.points.shape[0]

    @property
    def input_dimension(self) -> int:
        return self.points.shape[1]



class FunctionalBlock(eqx.Module):
    """Named linear outputs at a set of locations, without observed values.

    Args:
        locations: Floating coordinates with shape ``(n, d)``, or ``Locations``.
        operator: Linear operator. Its output shape determines each location's
            outputs: ``()`` for values, ``(d,)`` for gradients, for example.
        name: Nonempty block name used to access results.
    """

    locations: Locations
    operator: LinearOperator
    name: str = eqx.field(static=True)

    def __init__(
        self, locations: Locations | Points, operator: LinearOperator, name: str
    ) -> None:
        locations = (
            locations if isinstance(locations, Locations) else Locations(locations)
        )
        if not name:
            raise ValueError("block names cannot be empty")
        operator.output_shape(locations.input_dimension)
        self.locations = locations
        self.operator = operator
        self.name = name

    @property
    def shape(self) -> tuple[int, ...]:
        return (
            self.locations.count,
            *self.operator.output_shape(self.locations.input_dimension),
        )

    @property
    def size(self) -> int:
        return prod(self.shape)

    def append(self, locations: Locations) -> FunctionalBlock:
        """Return this block extended by locations, retaining its operator and name.

        The new locations must have the same input dimension. Supports JIT and
        differentiation; the original block is unchanged.
        """
        if locations.input_dimension != self.locations.input_dimension:
            raise ValueError("appended locations must have the same input dimension")
        return FunctionalBlock(
            Locations(jnp.concatenate((self.locations.points, locations.points))),
            self.operator,
            self.name,
        )



def _validate_collection(
    blocks: Sequence[FunctionalBlock],
) -> None:
    if not blocks:
        raise ValueError("a query, design, or dataset must contain at least one block")
    dimensions = {block.locations.input_dimension for block in blocks}
    if len(dimensions) != 1:
        raise ValueError("all blocks must use the same input dimension")
