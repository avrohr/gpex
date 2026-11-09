"""Mixed values, slopes and averages workload. Run this file with --help for options."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from _common import prediction, prediction_queries
from _data import data
from checks import check_variance

from gpex import (
    FunctionalBlock,
    FunctionalDataset,
    GaussianProcess,
    ObservationBlock,
    ObservationDesignBlock,
)
from gpex.blocks import Locations
from gpex.kernels import RBFKernel
from gpex.operators import BoxAverage, Gradient, Identity


def prepare(case):
    n = case["n"]
    x, y = data(n)
    gp = GaussianProcess(
        RBFKernel(variance=1.0, lengthscale=0.8),
        covariance_tile_size=case["covariance_tile"],
    )
    queries = prediction_queries(case["queries"])
    evidence = {}
    # n means scalar observations: split between values, slopes and averages.
    blocks = []
    for name, op, indices in zip(
        ("values", "slopes", "averages"),
        (Identity(), Gradient(), BoxAverage(0.15)),
        np.array_split(np.arange(n), 3),
        strict=True,
    ):
        points = jnp.asarray(x[indices])
        values = jnp.asarray(y[indices])
        if name == "slopes":
            values = 1.7 * jnp.cos(1.7 * points) - 0.31 * jnp.sin(3.1 * points)
        elif name == "averages":
            values = jnp.sin(1.7 * points[:, 0]) * jnp.sinc(
                1.7 * 0.15 / jnp.pi
            ) + 0.1 * jnp.cos(3.1 * points[:, 0]) * jnp.sinc(3.1 * 0.15 / jnp.pi)
        blocks.append(
            ObservationBlock(
                ObservationDesignBlock(
                    FunctionalBlock(Locations(points), op, name),
                    variance=0.05,
                    max_size=points.shape[0],
                ),
                y=values,
            )
        )
    dataset = FunctionalDataset(blocks)

    def validate(result, args):
        check_variance(np.diag(np.asarray(result[1])), case, evidence)

    return prediction, tuple((gp, dataset, q) for q in queries), validate, evidence


if __name__ == "__main__":
    from run import main

    main(default_workloads=["mixed"])
