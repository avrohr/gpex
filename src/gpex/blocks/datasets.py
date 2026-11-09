"""Observed outcomes attached to individual blocks and measurement designs."""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp

from ..operators.base import LinearOperator
from ..operators.local import Identity
from ..parameters import PositiveParameter
from ..validation.checks import error_if
from .designs import Count, ObservationDesign, ObservationDesignBlock, Scalar
from .functionals import (
    FunctionalBlock,
    FunctionalValues,
    Locations,
    Point,
    Points,
    _validate_collection,
)
from .indexing import IndexMap, ObservationVector
from .queries import FunctionalQuery


class ObservationBlock(eqx.Module):
    """A measurement design and its observed outputs.

    Args:
        design: Functional outputs and their independent Gaussian noise variances.
        y: Observed values with shape ``(active_count, *output_shape)``:
            ``(n,)`` for values or ``(n, d)`` for gradients, for example.
            A full capacity buffer is also accepted. Inactive values are zeroed.
    """

    design: ObservationDesignBlock
    y: FunctionalValues

    def __init__(self, design: ObservationDesignBlock, y: FunctionalValues) -> None:
        y = jnp.asarray(y)
        functional = design.functional
        shape = functional.shape
        if y.ndim != len(shape) or y.shape[1:] != shape[1:] or y.shape[0] > shape[0]:
            raise ValueError(
                f"values for {functional.name!r} have shape {y.shape}; expected {shape}"
            )
        if y.shape[0] < design.capacity:
            y = error_if(
                y,
                design.count != y.shape[0],
                "values must match the active location count",
            )
        y = jnp.pad(y, ((0, design.capacity - y.shape[0]), *((0, 0),) * (y.ndim - 1)))
        self.design = design
        self.y = jnp.where(design.active.reshape((-1,) + (1,) * (y.ndim - 1)), y, 0)

    @classmethod
    def from_arrays(
        cls,
        points: Points,
        y: FunctionalValues,
        *,
        variance: PositiveParameter | Scalar | float,
        name: str,
        operator: LinearOperator | None = None,
        max_size: int | None = None,
        padding_point: Point | None = None,
    ) -> ObservationBlock:
        """Construct observations directly from coordinates and outcomes.

        ``points`` has shape ``(n, d)``. The default identity operator expects
        ``y`` of shape ``(n,)``; otherwise use ``(n, *operator.output_shape(d))``.
        ``variance`` is one scalar measurement-error variance shared by the block.
        Numerical values may be traced; the name and shapes must remain static.

        ``max_size`` reserves locations, defaulting to ``max(1024, n)``.
        Unused slots repeat the first point. For ``n=0``, supply a
        ``padding_point`` valid for the kernel, mean and operator, and an array
        ``points`` with shape ``(0, d)``. Padding is excluded from inference.
        """
        operator = Identity() if operator is None else operator
        points = jnp.asarray(points)
        if points.ndim != 2:
            raise ValueError("locations must have shape (n, d)")
        count = points.shape[0]
        if count == 0:
            if padding_point is None:
                raise ValueError("empty observations require a valid padding_point")
            padding_point = jnp.asarray(padding_point)
            if padding_point.shape != (points.shape[1],):
                raise ValueError("padding_point must match the input dimension")
            points = padding_point[None, :]
        return cls(
            ObservationDesignBlock(
                FunctionalBlock(points, operator, name),
                variance,
                max_size=max_size,
                count=count,
            ),
            y,
        )

    @property
    def functional(self) -> FunctionalBlock:
        """The stored functional description, without noise or outcomes."""
        return self.design.functional

    @property
    def capacity(self) -> int:
        return self.design.capacity

    @property
    def count(self) -> Count:
        return self.design.count

    def replace(self, points: Points, y: FunctionalValues) -> ObservationBlock:
        """Replace contents within capacity, retaining operator, name and variance."""
        return ObservationBlock(self.design.replace(points), y)

    def append(self, locations: Locations, y: FunctionalValues) -> ObservationBlock:
        """Append outcomes using this block's operator and shared variance.

        Locations have shape ``(m, d)`` with the existing input dimension ``d``;
        ``y`` must have shape ``(m, *output_shape)``. Returns a new block and
        supports JIT and differentiation, including the shared variance parameter.
        """
        design = self.design.append(locations)
        y = jnp.asarray(y)
        expected_shape = (locations.count, *self.functional.shape[1:])
        if y.shape != expected_shape:
            raise ValueError(
                f"appended values have shape {y.shape}; expected {expected_shape}"
            )
        values = jax.lax.dynamic_update_slice_in_dim(self.y, y, self.count, axis=0)
        return ObservationBlock(design, values)



class FunctionalDataset(eqx.Module):
    """An ordered collection of observed measurement blocks.

    Args:
        blocks: Nonempty sequence of observation blocks with unique names and a
            common input dimension. Each block retains its own variance parameter.
    """

    blocks: tuple[ObservationBlock, ...]
    index_map: IndexMap = eqx.field(static=True)

    def __init__(self, blocks: Sequence[ObservationBlock]) -> None:
        blocks = tuple(blocks)
        self.blocks = blocks
        _validate_collection(self.functional_blocks)
        self.index_map = IndexMap.from_blocks(self.functional_blocks)

    @property
    def functional_blocks(self) -> tuple[FunctionalBlock, ...]:
        return tuple(block.functional for block in self.blocks)

    @property
    def query(self) -> FunctionalQuery:
        return FunctionalQuery(self.functional_blocks)

    @property
    def design(self) -> ObservationDesign:
        return ObservationDesign([block.design for block in self.blocks])

    @property
    def values_flat(self) -> ObservationVector:
        return jnp.concatenate(
            [
                jnp.where(block.design.active_flat, block.y.reshape(-1), 0)
                for block in self.blocks
            ]
        )

    @property
    def active_size(self) -> Count:
        return sum((block.design.active_size for block in self.blocks), jnp.asarray(0))

    @property
    def variance_flat(self) -> ObservationVector:
        """Measurement-error variances in canonical scalar-output order."""
        return jnp.concatenate(
            [block.design.variances.reshape(-1) for block in self.blocks]
        )

    def __add__(self, other: FunctionalDataset) -> FunctionalDataset:
        if not isinstance(other, FunctionalDataset):
            return NotImplemented
        return FunctionalDataset((*self.blocks, *other.blocks))
