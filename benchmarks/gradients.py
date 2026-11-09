"""Gradients workload. Run this file with --help for options."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from _common import prediction, prediction_queries, value_dataset

from gpex import (
    GaussianProcess,
)
from gpex.kernels import RBFKernel


def prepare(case):
    dataset = value_dataset(case["n"])
    queries = prediction_queries(case["queries"])
    gp = GaussianProcess(
        RBFKernel(variance=1.0, lengthscale=0.8),
        covariance_tile_size=case["covariance_tile"],
    )

    def prediction_objective(lengthscale, model, observations, query):
        model = eqx.tree_at(
            lambda m: m.kernel.lengthscales.lengthscale_parameter, model, lengthscale
        )
        mean, covariance = prediction(model, observations, query)
        return jnp.sum(mean) + jnp.trace(covariance)

    function = jax.value_and_grad(prediction_objective)
    inputs = tuple((jnp.asarray(0.8), gp, dataset, query) for query in queries)

    return function, inputs, None, {}


if __name__ == "__main__":
    from run import main

    main(default_workloads=["kernel-gradient"])
