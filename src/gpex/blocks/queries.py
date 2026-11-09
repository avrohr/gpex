"""Collections of named functionals to predict or sample."""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx

from .functionals import FunctionalBlock, _validate_collection
from .indexing import IndexMap


class FunctionalQuery(eqx.Module):
    """An ordered collection of named outputs to predict or sample.

    Args:
        blocks: Nonempty sequence of ``FunctionalBlock`` objects with unique
            names and a common input dimension. Ordering defines the flat output.
    """

    blocks: tuple[FunctionalBlock, ...]
    index_map: IndexMap = eqx.field(static=True)

    def __init__(self, blocks: Sequence[FunctionalBlock]) -> None:
        blocks = tuple(blocks)
        _validate_collection(blocks)
        self.blocks = blocks
        self.index_map = IndexMap.from_blocks(blocks)

    @property
    def functional_blocks(self) -> tuple[FunctionalBlock, ...]:
        return self.blocks
