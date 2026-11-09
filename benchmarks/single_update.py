"""Single update workload. Run this file with --help for options."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from _common import prediction, prediction_queries
from _data import data
from checks import check_variance

from gpex import (
    FunctionalDataset,
    GaussianProcess,
    ObservationBlock,
)
from gpex.blocks import Locations
from gpex.kernels import RBFKernel


def prepare(case):
    n, workload = case["n"], case["workload"]
    gp = GaussianProcess(
        RBFKernel(variance=1.0, lengthscale=0.8),
        covariance_tile_size=case["covariance_tile"],
    )
    x, y = data(n)
    queries = prediction_queries(case["queries"])
    evidence = {}
    block = ObservationBlock.from_arrays(
        jnp.asarray(x), jnp.asarray(y), variance=0.05, name="values", max_size=n + 1
    )
    dataset = FunctionalDataset([block])
    posterior = gp.condition(dataset)
    jax.block_until_ready(posterior)
    new_x = jnp.array([[4.125]])
    new_y = jnp.sin(1.7 * new_x[:, 0]) + 0.1 * jnp.cos(3.1 * new_x[:, 0])
    complete = FunctionalDataset([block.append(Locations(new_x), new_y)])

    def append(current, points, values, query):
        p = current.append(Locations(points), values, block="values").predict(query)
        return p.mean.flat, p.covariance.full

    function = append if workload == "append" else prediction
    inputs = tuple(
        (posterior, new_x, new_y, query)
        if workload == "append"
        else (gp, complete, query)
        for query in queries
    )
    evidence["reserved_observations"] = n + 1

    def validate(result, args):
        check_variance(np.diag(np.asarray(result[1])), case, evidence)

    return function, inputs, validate, evidence


if __name__ == "__main__":
    from run import main

    main(default_workloads=["append", "rebuild"])
