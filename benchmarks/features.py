"""Features workload. Run this file with --help for options."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from _data import query_points

from gpex.kernels import RBFKernel


def prepare(case):
    kernel = RBFKernel(variance=1.0, lengthscale=0.8)
    q = case["queries"]

    def feature_objective(multiplier, kernel, points):
        scaled_kernel = jax.tree.map(lambda value: multiplier * value, kernel)
        features = scaled_kernel.random_features(
            jax.random.key(42), points.shape[1], feature_count=256
        )
        return jnp.sum(jax.vmap(features)(points) ** 2)

    function = jax.value_and_grad(feature_objective)
    inputs = tuple(
        (jnp.asarray(1.0), kernel, jnp.asarray(query_points(q, shift)))
        for shift in (0.0, 0.037)
    )

    evidence = {"feature_count": 256}
    return function, inputs, None, evidence


if __name__ == "__main__":
    from run import main

    main(default_workloads=["feature-gradient"])
