"""Condition on 100 independent datasets, at sizes 16..512, and predict once.

Compare gpex capacity 512, compact gpex, GPJax and GPyTorch in this file.
Inputs are prepared before timing. Every call changes locations AND values.
Each size is visited once, in a fixed shuffled order.
Run: python benchmarks/datasets.py --output results/datasets.json
"""

from typing import Any

import numpy as np
from _runtime import initialize
from _sequence import Sequence

METHODS = ["datasets", "datasets-compact", "datasets-gpjax", "datasets-gpytorch"]


def dataset_sizes(case):
    count, maximum, minimum = case["sequence_length"], case["n"], case["min_size"]
    if minimum < 1 or maximum < minimum or count < 1:
        raise ValueError(
            "dataset sizes and sequence length must be positive and ordered"
        )
    if case.get("pattern", "distinct") != "distinct":
        raise ValueError("dataset sequences require distinct sizes")
    if count > maximum - minimum + 1:
        raise ValueError(
            "sequence length cannot exceed the number of distinct sizes in the range"
        )
    sizes = np.rint(np.linspace(minimum, maximum, count)).astype(int)
    rng = np.random.default_rng(2026)
    rng.shuffle(sizes)
    return sizes.tolist()


def datasets(case):
    sizes = dataset_sizes(case)
    rng = np.random.default_rng(42)
    result = []
    for n in sizes:
        x = rng.uniform(-4.0, 4.0, (n, 1))
        y = np.sin(1.7 * x[:, 0]) + 0.1 * np.cos(3.1 * x[:, 0])
        result.append((x, y))
    return sizes, result


def prepare(case):
    initialize(case)
    from _data import query_points

    counts, raw = datasets(case)
    method = case["workload"]
    q = query_points(case["queries"])
    if method in ("datasets", "datasets-compact"):
        import jax.numpy as jnp

        from gpex import (
            FunctionalBlock,
            FunctionalDataset,
            FunctionalQuery,
            GaussianProcess,
            ObservationBlock,
        )
        from gpex.kernels import RBFKernel
        from gpex.operators import Identity

        gp = GaussianProcess(
            RBFKernel(variance=1.0, lengthscale=0.8),
            covariance_tile_size=case.get("covariance_tile"),
        )
        query = FunctionalQuery(
            [FunctionalBlock(jnp.asarray(q), Identity(), "prediction")]
        )
        inputs = [
            (
                gp,
                FunctionalDataset(
                    [
                        ObservationBlock.from_arrays(
                            jnp.asarray(x),
                            jnp.asarray(y),
                            variance=0.05,
                            name="values",
                            max_size=case["n"] if method == "datasets" else len(x),
                        )
                    ]
                ),
                query,
            )
            for x, y in raw
        ]

        def condition_gpex(_, gp, data, query):
            p = gp.condition(data).predict(query)
            return None, (p.mean.flat, p.covariance.full)

        return Sequence(condition_gpex, inputs, counts, "gpex")

    if method == "datasets-gpjax":
        import gpjax as gpx
        import jax.numpy as jnp
        from compare import gpjax_model

        model = gpjax_model()
        inputs = [
            (model, jnp.asarray(x), jnp.asarray(y[:, None]), jnp.asarray(q))
            for x, y in raw
        ]

        def condition_gpjax(_, model, x, y, query):
            p = model.predict(
                query, train_data=gpx.Dataset(X=x, y=y), covariance="dense"
            )
            return None, (p.mean, p.covariance())

        return Sequence(
            condition_gpjax, inputs, counts, "gpjax", prior_variance=1.0 + 1e-6
        )

    from compare import torch_model

    model, array = torch_model(case, *raw[0])
    inputs = [(array(x), array(y), array(q)) for x, y in raw]

    def condition_torch(_, x, y, query):
        # set_train_data invalidates the old posterior caches for the new data.
        model.set_train_data(inputs=x, targets=y, strict=False)
        p: Any = model(query)
        return None, (p.mean, p.covariance_matrix)

    return Sequence(condition_torch, inputs, counts, "torch")


if __name__ == "__main__":
    from run import main

    main(default_workloads=METHODS, default_sizes=[512])
