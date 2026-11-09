"""Measurement designs with noise and reserved capacity, without outcomes."""

from __future__ import annotations

from collections.abc import Sequence
from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from ..parameters import PositiveParameter
from ..validation.checks import error_if
from .functionals import FunctionalBlock, Locations, Points, _validate_collection
from .indexing import IndexMap, ObservationVector
from .queries import FunctionalQuery

VarianceValues = Float[Array, "points *output_shape"]
Scalar = Float[Array, ""]
Count = Int[Array, ""]
Activity = Bool[Array, "capacity"]
FlatActivity = Bool[Array, "scalar_capacity"]


class ObservationDesignBlock(eqx.Module):
    """A functional block with measurement noise, but no observed outcome.

    Args:
        functional: Named outputs to measure, including their locations and operator.
        variance: One finite, nonnegative scalar measurement-error variance,
            shared by every location and output component. A scalar array or
            float is fixed. A scalar ``PositiveParameter`` is trainable and
            preserves positivity during fitting. Observation errors are independent.
        max_size: Reserved number of locations, at least the supplied number.
            Defaults to ``max(1024, supplied locations)``. Set it explicitly to
            the supplied count for compact storage. Operator output components
            each occupy a scalar slot per reserved location.
        count: Active prefix length. Defaults to all supplied locations. May be
            traced, unlike capacity. Inactive locations repeat the first supplied
            point, which must be valid for the operator and kernel even when the
            active count is zero.
    """

    functional: FunctionalBlock
    variance: PositiveParameter | Scalar
    count: Count

    def __init__(
        self,
        functional: FunctionalBlock,
        variance: PositiveParameter | Scalar | float,
        *,
        max_size: int | None = None,
        count: Count | int | None = None,
    ) -> None:
        if isinstance(variance, PositiveParameter):
            value = variance.value
        else:
            variance = jnp.asarray(variance)
            variance = error_if(
                variance,
                jnp.any(~jnp.isfinite(variance) | (variance < 0)),
                "noise variances must be finite and nonnegative",
            )
            value = variance
        if value.ndim != 0:
            raise ValueError("noise variance must be scalar")
        supplied = functional.locations.count
        capacity = max(1024, supplied) if max_size is None else max_size
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError("max_size must be an integer")
        if capacity < supplied:
            raise ValueError("max_size must be at least the supplied location count")
        count = jnp.asarray(supplied if count is None else count)
        if count.ndim != 0 or not jnp.issubdtype(count.dtype, jnp.integer):
            raise ValueError("active count must be an integer scalar")
        count = error_if(
            count,
            (count < 0) | (count > supplied),
            "active count must lie within the supplied locations",
        )
        points = jnp.pad(
            functional.locations.points,
            ((0, capacity - supplied), (0, 0)),
            mode="edge",
        )
        points = jnp.where((jnp.arange(capacity) < count)[:, None], points, points[0])
        self.functional = eqx.tree_at(lambda f: f.locations.points, functional, points)
        self.variance = variance
        self.count = count

    @property
    def capacity(self) -> int:
        """Reserved locations, independent of the active count."""
        return self.functional.locations.count

    @property
    def active(self) -> Activity:
        return jnp.arange(self.capacity) < self.count

    @property
    def active_flat(self) -> FlatActivity:
        """Activity in canonical scalar-output order."""
        return jnp.repeat(self.active, prod(self.functional.shape[1:]))

    @property
    def active_size(self) -> Count:
        """Number of active scalar observations, including operator components."""
        return self.count * prod(self.functional.shape[1:])

    @property
    def variances(self) -> VarianceValues:
        """The shared variance repeated over the functional output shape."""
        value = (
            self.variance.value
            if isinstance(self.variance, PositiveParameter)
            else self.variance
        )
        return jnp.broadcast_to(value, self.functional.shape)

    def append(self, locations: Locations) -> ObservationDesignBlock:
        """Fill unused locations, retaining operator, name, variance and capacity.

        Locations have shape ``(m, d)`` with the existing input dimension ``d``.
        The returned design supports JIT and differentiation.
        """
        if locations.input_dimension != self.functional.locations.input_dimension:
            raise ValueError("appended locations must have the same input dimension")
        if locations.count > self.capacity:
            raise ValueError("appended locations exceed max_size")
        points, count = error_if(
            (self.functional.locations.points, self.count),
            self.count + locations.count > self.capacity,
            "appended locations exceed max_size",
        )
        points = jax.lax.dynamic_update_slice_in_dim(
            points, locations.points, count, axis=0
        )
        count = count + locations.count
        points = jnp.where(
            (jnp.arange(self.capacity) < count)[:, None], points, points[0]
        )
        return eqx.tree_at(
            lambda b: (b.functional.locations.points, b.count), self, (points, count)
        )

    def replace(self, points: Points) -> ObservationDesignBlock:
        """Replace all active locations, preserving capacity and shared variance.

        Empty contents retain the previous valid padding point. Supports JIT
        and differentiation. Different input shapes specialize this operation,
        but its result always has the original reserved shapes.
        """
        points = jnp.asarray(points)
        expected = self.functional.locations.input_dimension
        if points.ndim != 2 or points.shape[1] != expected:
            raise ValueError(f"replacement locations must have shape (n, {expected})")
        count = points.shape[0]
        if count > self.capacity:
            raise ValueError("replacement locations exceed max_size")
        if count == 0:
            points = self.functional.locations.points[:1]
        functional = FunctionalBlock(
            points, self.functional.operator, self.functional.name
        )
        return ObservationDesignBlock(
            functional, self.variance, max_size=self.capacity, count=count
        )



class ObservationDesign(eqx.Module):
    """An ordered measurement design without observed outcomes.

    Args:
        blocks: Nonempty sequence of design blocks with unique names and a common
            input dimension. Variances are flattened in the same order as outputs.
    """

    blocks: tuple[ObservationDesignBlock, ...]
    index_map: IndexMap = eqx.field(static=True)

    def __init__(self, blocks: Sequence[ObservationDesignBlock]) -> None:
        blocks = tuple(blocks)
        self.blocks = blocks
        _validate_collection(self.functional_blocks)
        self.index_map = IndexMap.from_blocks(self.functional_blocks)

    @property
    def query(self) -> FunctionalQuery:
        return FunctionalQuery(self.functional_blocks)

    @property
    def functional_blocks(self) -> tuple[FunctionalBlock, ...]:
        return tuple(block.functional for block in self.blocks)

    @property
    def variance_flat(self) -> ObservationVector:
        """Measurement-error variances in canonical scalar-output order."""
        return jnp.concatenate([block.variances.reshape(-1) for block in self.blocks])

    @property
    def active_size(self) -> Count:
        return sum((block.active_size for block in self.blocks), jnp.asarray(0))
