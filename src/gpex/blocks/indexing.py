"""Canonical flattening, static index maps, and named array views."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from .functionals import FunctionalBlock, FunctionalValues

FlatValues = Float[Array, "*leading flat_size"]
ObservationVector = Float[Array, "observation_size"]


@dataclass(frozen=True)
class BlockIndex:
    name: str
    shape: tuple[int, ...]
    start: int
    stop: int

    @property
    def slice(self) -> slice:
        return slice(self.start, self.stop)



@dataclass(frozen=True)
class IndexMap:
    blocks: tuple[BlockIndex, ...]
    size: int

    def __post_init__(self) -> None:
        names = [block.name for block in self.blocks]
        if len(set(names)) != len(names):
            raise ValueError("index-map block names must be unique")

    @classmethod
    def from_blocks(
        cls,
        blocks: Sequence[FunctionalBlock],
    ) -> IndexMap:
        indices: list[BlockIndex] = []
        start = 0
        for block in blocks:
            stop = start + block.size
            indices.append(BlockIndex(block.name, block.shape, start, stop))
            start = stop
        return cls(tuple(indices), start)

    def append(self, block: FunctionalBlock) -> IndexMap:
        """Extend the public layout, requiring a new block name."""
        stop = self.size + block.size
        index = BlockIndex(block.name, block.shape, self.size, stop)
        return IndexMap((*self.blocks, index), stop)

    def block(self, name: str) -> BlockIndex:
        for block in self.blocks:
            if block.name == name:
                return block
        raise KeyError(name)

    def unpack(self, flat: FlatValues) -> dict[str, FunctionalValues]:
        flat = jnp.asarray(flat)
        if flat.ndim == 0 or flat.shape[-1] != self.size:
            raise ValueError(f"last axis must have size {self.size}")
        leading_shape = flat.shape[:-1]
        return {
            block.name: flat[..., block.slice].reshape((*leading_shape, *block.shape))
            for block in self.blocks
        }

    def pack(self, values: Mapping[str, FunctionalValues]) -> FlatValues:
        extra = set(values) - {block.name for block in self.blocks}
        if extra:
            raise ValueError(f"unexpected blocks: {sorted(extra)}")
        parts = []
        for block in self.blocks:
            value = jnp.asarray(values[block.name])
            if value.shape != block.shape:
                raise ValueError(
                    f"block {block.name!r} has shape {value.shape}; "
                    f"expected {block.shape}"
                )
            parts.append(value.reshape(-1))
        return jnp.concatenate(parts)



class StructuredValues(eqx.Module):
    """Flat and named views of means, variances, or samples.

    Args:
        flat: Numerical values with shape ``(*leading, index_map.size)``.
        index_map: Static mapping from names to flattened slices and shapes.

    Attributes:
        flat: Array of shape ``(*leading, q)``. Samples have a leading draw axis;
            means and variances have no leading axes.
        by_name: Views derived from ``flat``, of shape ``(*leading, *block.shape)``.
            The same arrays are available as ``values[name]``.
        index_map: Static mapping from names to flattened slices and shapes.
    """

    flat: FlatValues
    index_map: IndexMap = eqx.field(static=True)

    def __init__(self, flat: FlatValues, index_map: IndexMap) -> None:
        flat = jnp.asarray(flat)
        if flat.ndim == 0 or flat.shape[-1] != index_map.size:
            raise ValueError(f"last axis must have size {index_map.size}")
        self.flat = flat
        self.index_map = index_map

    @property
    def by_name(self) -> Mapping[str, FunctionalValues]:
        return self.index_map.unpack(self.flat)

    def __getitem__(self, name: str) -> FunctionalValues:
        block = self.index_map.block(name)
        return self.flat[..., block.slice].reshape(
            (*self.flat.shape[:-1], *block.shape)
        )
