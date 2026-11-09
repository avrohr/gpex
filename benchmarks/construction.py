"""Construction workload. Run this file with --help for options."""

from __future__ import annotations

import jax.numpy as jnp
from _data import data

from gpex import (
    FunctionalBlock,
    FunctionalDataset,
    ObservationBlock,
    ObservationDesignBlock,
)
from gpex.blocks import Locations
from gpex.operators import Identity


def prepare(case):
    x, y = data(case["n"])

    def construct(points, values):
        result = FunctionalDataset(
            [
                ObservationBlock(
                    ObservationDesignBlock(
                        FunctionalBlock(Locations(points), Identity(), "values"),
                        variance=0.05,
                        max_size=points.shape[0],
                    ),
                    y=values,
                )
            ]
        )
        # Exercise metadata reads as well as construction.
        return result.values_flat, result.index_map.unpack(result.values_flat)

    inputs = ((jnp.asarray(x), jnp.asarray(y)),)

    return construct, inputs, None, {}


if __name__ == "__main__":
    from run import main

    main(default_workloads=["construction"])
