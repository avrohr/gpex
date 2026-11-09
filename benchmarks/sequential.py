"""Grow 512 observations to 1,024, adding and predicting one value at a time.

Initial conditioning/cache preparation and every update are timed. gpex
compares fixed-capacity append against fixed-capacity reconditioning. GPJax
reconditions. GPyTorch uses get_fantasy_model and retains its prediction caches.
Run: python benchmarks/sequential.py --output results/sequential.json

Optional multiblock variants distribute the same readings across two reserved
blocks, alternating the update target. Request them explicitly with --workloads.
They are gpex-specific diagnostics, excluded from the default comparison.
"""

from typing import Any

import numpy as np
from _runtime import initialize
from _sequence import Sequence

METHODS = [
    "sequential",
    "sequential-rebuild",
    "sequential-gpjax",
    "sequential-gpytorch",
]


def prepare(case):
    initialize(case)
    from _data import query_points

    start, end = case["start_size"], case["n"]
    x = np.random.default_rng(2026).uniform(-4.0, 4.0, (end, 1))
    y = np.sin(1.7 * x[:, 0]) + 0.1 * np.cos(3.1 * x[:, 0])
    q = query_points(case["queries"])
    counts = list(range(start + 1, end + 1))
    method = case["workload"]
    if method in (
        "sequential",
        "sequential-rebuild",
        "sequential-multiblock",
        "sequential-rebuild-multiblock",
    ):
        import jax.numpy as jnp

        from gpex import (
            FunctionalBlock,
            FunctionalDataset,
            FunctionalQuery,
            GaussianProcess,
            ObservationBlock,
        )
        from gpex.blocks import Locations
        from gpex.kernels import RBFKernel
        from gpex.operators import Identity

        gp = GaussianProcess(
            RBFKernel(variance=1.0, lengthscale=0.8),
            covariance_tile_size=case.get("covariance_tile"),
        )
        block_count = 2 if method.endswith("multiblock") else 1
        use_append = method in ("sequential", "sequential-multiblock")
        blocks = tuple(
            ObservationBlock.from_arrays(
                jnp.asarray(x[j:start:block_count]),
                jnp.asarray(y[j:start:block_count]),
                variance=0.05,
                name=f"values{j}",
                max_size=len(range(j, end, block_count)),
            )
            for j in range(block_count)
        )
        query = FunctionalQuery(
            [FunctionalBlock(jnp.asarray(q), Identity(), "prediction")]
        )

        def initial_gpex(gp, blocks, query):
            posterior = gp.condition(FunctionalDataset(blocks))
            p = posterior.predict(query)
            state = posterior if use_append else blocks
            return state, (p.mean.flat, p.covariance.full)

        def step_gpex(state, gp, x, y, query, block_index):
            if use_append:
                state = state.append(Locations(x), y, block=f"values{block_index}")
                p = state.predict(query)
            else:
                state = tuple(
                    b.append(Locations(x), y) if j == block_index else b
                    for j, b in enumerate(state)
                )
                p = gp.condition(FunctionalDataset(state)).predict(query)
            return state, (p.mean.flat, p.covariance.full)

        inputs = [
            (
                gp,
                jnp.asarray(x[i : i + 1]),
                jnp.asarray(y[i : i + 1]),
                query,
                i % block_count,
            )
            for i in range(start, end)
        ]
        return Sequence(
            step_gpex, inputs, counts, "gpex", initial_gpex, (gp, blocks, query)
        )

    if method == "sequential-gpjax":
        import gpjax as gpx
        import jax.numpy as jnp
        from compare import gpjax_model

        model = gpjax_model()
        query = jnp.asarray(q)

        def predict(model, x, y, query):
            p = model.predict(
                query, train_data=gpx.Dataset(X=x, y=y), covariance="dense"
            )
            return p.mean, p.covariance()

        def initial_gpjax(model, x, y, query):
            return (x, y), predict(model, x, y, query)

        def step_gpjax(state, model, x, y, query):
            x = jnp.concatenate((state[0], x))
            y = jnp.concatenate((state[1], y))
            return (x, y), predict(model, x, y, query)

        inputs = [
            (model, jnp.asarray(x[i : i + 1]), jnp.asarray(y[i : i + 1, None]), query)
            for i in range(start, end)
        ]
        args = (model, jnp.asarray(x[:start]), jnp.asarray(y[:start, None]), query)
        return Sequence(
            step_gpjax,
            inputs,
            counts,
            "gpjax",
            initial_gpjax,
            args,
            prior_variance=1.0 + 1e-6,
        )

    from compare import torch_model

    model, array = torch_model(case, x[:start], y[:start])
    query = array(q)

    def initial_torch():
        # A prediction initializes the caches required by get_fantasy_model.
        p: Any = model(query)
        return model, (p.mean, p.covariance_matrix)

    def step_torch(model, x, y):
        model = model.get_fantasy_model(x, y)
        p: Any = model(query)
        return model, (p.mean, p.covariance_matrix)

    inputs = [(array(x[i : i + 1]), array(y[i : i + 1])) for i in range(start, end)]
    return Sequence(step_torch, inputs, counts, "torch", initial_torch)


if __name__ == "__main__":
    from run import main

    main(default_workloads=METHODS, default_sizes=[1024])
