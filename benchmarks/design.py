"""Design workload. Run this file with --help for options."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from checks import check_variance

from gpex import (
    FunctionalBlock,
    FunctionalQuery,
    GaussianProcess,
    ObservationDesign,
    ObservationDesignBlock,
    PreparedQuery,
)
from gpex.blocks import Locations
from gpex.kernels import RBFKernel
from gpex.operators import Gradient, Identity


def design(points, conditioning, query):
    return jnp.trace(
        conditioning.append(Locations(points), block="values")
        .predict_covariance(query)
        .full
    )


def design_prepared(points, prepared, variance):
    future = ObservationDesignBlock(
        FunctionalBlock(Locations(points), Identity(), "future"),
        variance=variance,
        max_size=points.shape[0],
    )
    return jnp.sum(prepared.predict_variance_after(future).flat)


def prepare(case):
    n, workload = case["n"], case["workload"]
    gp = GaussianProcess(
        RBFKernel(variance=1.0, lengthscale=0.8),
        covariance_tile_size=case["covariance_tile"],
    )
    evidence = {}
    points = jnp.asarray(np.random.default_rng(0).uniform(-4.0, 4.0, (n, 2)))
    conditioning = gp.prepare_conditioning(
        ObservationDesign(
            [
                ObservationDesignBlock(
                    FunctionalBlock(Locations(points), Identity(), "values"),
                    variance=0.05,
                    max_size=n + 4 if workload == "design" else n,
                )
            ]
        )
    )
    query = FunctionalQuery(
        [FunctionalBlock(Locations(jnp.zeros((1, 2))), Gradient(), "gradient")]
    )
    candidates = jnp.asarray(np.random.default_rng(1).uniform(-1.0, 1.0, (4, 2)))
    inputs = tuple((candidates + s, conditioning, query) for s in (0.0, 0.037))
    function = design
    if workload == "design-prepared":
        prepared = PreparedQuery(conditioning, query)
        inputs = tuple((args[0], prepared, 0.05) for args in inputs)
        function = design_prepared
        evidence["query_preparation_included"] = False

    def validate(result, args):
        check_variance(result[0], case, evidence, prior_variance=2 / 0.8**2)

    return jax.value_and_grad(function), inputs, validate, evidence


if __name__ == "__main__":
    from run import main

    main(default_workloads=["design-prepared"])
