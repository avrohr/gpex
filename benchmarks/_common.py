"""Repeated value-data and query setup. Timed operations live in workload scripts."""

from __future__ import annotations

import jax.numpy as jnp
from _data import data, query_points

from gpex import (
    FunctionalBlock,
    FunctionalDataset,
    FunctionalQuery,
    ObservationBlock,
)
from gpex.blocks import Locations
from gpex.operators import Identity


def prediction(gp, dataset, query):
    p = gp.condition(dataset).predict(query)
    return p.mean.flat, p.covariance.full


def value_dataset(n):
    """Compact function values with the common data and noise variance."""
    x, y = data(n)
    block = ObservationBlock.from_arrays(
        jnp.asarray(x), jnp.asarray(y), variance=0.05, name="values", max_size=n
    )
    return FunctionalDataset([block])


def prediction_queries(q):
    """Two query batches with matching shapes for warm-call measurement."""
    return tuple(
        FunctionalQuery(
            [
                FunctionalBlock(
                    Locations(jnp.asarray(query_points(q, s))), Identity(), "prediction"
                )
            ]
        )
        for s in (0.0, 0.037)
    )
