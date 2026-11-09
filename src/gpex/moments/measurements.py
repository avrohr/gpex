"""Numerical measurement groups, independent of public names and solvers."""

from __future__ import annotations

from collections.abc import Sequence
from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int

from ..blocks import FunctionalBlock, Locations, ObservationDesignBlock
from ..operators.base import LinearOperator
from ..validation.checks import error_if

PointMatrix = Float[Array, "locations dimension"]
Permutation = Int[Array, "functionals"]
Activity = Bool[Array, "functionals"]


def _signature(operator: LinearOperator, dimension: int) -> tuple:
    arrays, static = eqx.partition(operator, eqx.is_array)
    return (
        jax.tree.structure(static),
        tuple(jax.tree.leaves(static)),
        tuple(leaf.shape for leaf in jax.tree.leaves(arrays)),
        operator.output_shape(dimension),
    )


class MeasurementGroup(eqx.Module):
    """One operator program, with its numerical leaves batched over locations."""

    points: PointMatrix
    operator_arrays: LinearOperator
    operator_static: LinearOperator = eqx.field(static=True)
    output_shape: tuple[int, ...] = eqx.field(static=True)

    @property
    def operators(self) -> LinearOperator:
        return eqx.combine(self.operator_arrays, self.operator_static)

    @property
    def size(self) -> int:
        return self.points.shape[0] * prod(self.output_shape)

    @classmethod
    def from_blocks(cls, blocks: Sequence[FunctionalBlock]) -> MeasurementGroup:
        first = blocks[0]
        static = eqx.filter(first.operator, eqx.is_array, inverse=True)
        operator_arrays = []
        for block in blocks:
            arrays = eqx.filter(block.operator, eqx.is_array)
            repeated = jax.tree.map(
                lambda value, count=block.locations.count: jnp.broadcast_to(
                    value, (count, *value.shape)
                ),
                arrays,
            )
            operator_arrays.append(repeated)
        arrays = jax.tree.map(lambda *parts: jnp.concatenate(parts), *operator_arrays)
        return cls(
            jnp.concatenate([block.locations.points for block in blocks]),
            arrays,
            static,
            first.operator.output_shape(first.locations.input_dimension),
        )


class PreparedFunctionals(eqx.Module):
    """Numerical inputs in operator-group order with canonical scalar activity.

    Array parameters remain dynamic and differentiable. Group membership depends
    only on operator structure, static settings and leaf shapes, never values.
    An integer permutation connects grouped storage to canonical observation
    order. Activity stays in canonical order, independent of grouping. Public
    names and index maps stay outside this numerical PyTree.
    """

    groups: tuple[MeasurementGroup, ...]
    restore: Permutation | None
    active: Activity

    @property
    def size(self) -> int:
        return sum(group.size for group in self.groups)

    @property
    def input_dimension(self) -> int:
        return self.groups[0].points.shape[1]

    @classmethod
    def from_blocks(
        cls, blocks: Sequence[FunctionalBlock | ObservationDesignBlock]
    ) -> PreparedFunctionals:
        active = jnp.concatenate(
            [
                block.active_flat
                if isinstance(block, ObservationDesignBlock)
                else jnp.ones(block.size, dtype=bool)
                for block in blocks
            ]
        )
        blocks = tuple(
            block.functional if isinstance(block, ObservationDesignBlock) else block
            for block in blocks
        )
        if len({block.locations.input_dimension for block in blocks}) > 1:
            raise ValueError("covariance blocks must use the same input dimension")
        members: dict[tuple, list[FunctionalBlock]] = {}
        offsets: dict[tuple, list[int]] = {}
        start = 0
        for block in blocks:
            key = _signature(block.operator, block.locations.input_dimension)
            members.setdefault(key, []).append(block)
            offsets.setdefault(key, []).extend(range(start, start + block.size))
            start += block.size
        groups = tuple(
            MeasurementGroup.from_blocks(group) for group in members.values()
        )
        order = [index for key in members for index in offsets[key]]
        if order == list(range(start)):
            return cls(groups, None, active)
        return cls(
            groups,
            jnp.asarray(np.argsort(order)),
            active,
        )

    def append(
        self,
        locations: Locations,
        *,
        group_index: int,
        location_start: int,
        capacity: int,
        scalar_start: int,
    ) -> tuple[PreparedFunctionals, PreparedFunctionals, Permutation]:
        """Activate rows within one block, returning its compact new measurements.

        The block index and batch shape are static. The active count and returned
        canonical scalar indices are dynamic. Existing group shapes stay fixed.
        """
        start = location_start
        group = self.groups[group_index]
        if locations.input_dimension != self.input_dimension:
            raise ValueError("appended locations must have the same input dimension")
        added = locations.count
        if added > capacity:
            raise ValueError("appended locations exceed max_size")
        components = prod(group.output_shape)
        active = self.active[scalar_start : scalar_start + capacity * components]
        count = jnp.sum(active) // components
        count = error_if(
            count, count + added > capacity, "appended locations exceed max_size"
        )
        points = jax.lax.dynamic_update_slice_in_dim(
            group.points[start : start + capacity], locations.points, count, axis=0
        )
        points = jnp.where(
            (jnp.arange(capacity) < count + added)[:, None], points, points[0]
        )
        groups = list(self.groups)
        groups[group_index] = eqx.tree_at(
            lambda g: g.points,
            group,
            group.points.at[start : start + capacity].set(points),
        )
        indices = scalar_start + count * components + jnp.arange(added * components)
        updated = eqx.tree_at(
            lambda m: (m.groups, m.active),
            self,
            (tuple(groups), self.active.at[indices].set(True)),
        )
        # A block shares one operator. Its batched leaves already contain valid
        # operator parameters in unused rows, including for an empty block.
        arrays = jax.tree.map(
            lambda a: jnp.broadcast_to(a[start], (added, *a.shape[1:])),
            group.operator_arrays,
        )
        new_group = MeasurementGroup(
            locations.points, arrays, group.operator_static, group.output_shape
        )
        appended = PreparedFunctionals(
            (new_group,),
            None,
            jnp.ones(added * components, dtype=bool),
        )
        return updated, appended, indices

    def mask(self, values: Array) -> Array:
        """Zero inactive canonical rows, preserving any trailing output axes."""
        return jnp.where(
            self.active.reshape((-1,) + (1,) * (values.ndim - 1)), values, 0
        )

    def to_canonical(self, values: Array) -> Array:
        return values if self.restore is None else values[self.restore]

    def to_grouped(self, values: Array) -> Array:
        """Reorder canonical rows into numerical operator groups."""
        if self.restore is None:
            return values
        return jnp.zeros_like(values).at[self.restore].set(values)


def prepare_functionals(
    blocks: Sequence[FunctionalBlock | ObservationDesignBlock] | PreparedFunctionals,
) -> PreparedFunctionals:
    """Accept public blocks at moment-assembly boundaries, or reuse prepared inputs."""
    if isinstance(blocks, PreparedFunctionals):
        return blocks
    return PreparedFunctionals.from_blocks(blocks)


def measurement_block_layout(
    blocks: Sequence[ObservationDesignBlock],
) -> tuple[tuple[int, int], ...]:
    """Locate public blocks in numerical groups without changing assembly inputs.

    Conditioning state retains this static (group index, first location) mapping beside
    its public index map. Numerical grouping stays independent of block names
    and boundaries, so equivalent grouped inputs share compiled covariance code.
    """
    offsets: dict[tuple, int] = {}
    layout = []
    for block in blocks:
        functional = block.functional
        key = _signature(functional.operator, functional.locations.input_dimension)
        if key not in offsets:
            offsets[key] = 0
        layout.append((tuple(offsets).index(key), offsets[key]))
        offsets[key] += functional.locations.count
    return tuple(layout)
