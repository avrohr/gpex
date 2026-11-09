"""Condition on N values and predict one query: gpex, GPJax and GPyTorch.

Each case uses compact training storage. First-call time includes compilation.
Run: python benchmarks/dense.py --sizes 16 128 512 --output results/dense.json
"""

import numpy as np
from checks import check_variance


def prepare(case):
    from collections.abc import Callable
    from typing import Any

    from _data import data, query_points

    operation: Callable[..., Any]
    evidence = {}
    method = case["workload"]
    if method == "dense":
        from _common import prediction_queries, value_dataset

        from gpex import GaussianProcess
        from gpex.kernels import RBFKernel

        gp = GaussianProcess(
            RBFKernel(variance=1.0, lengthscale=0.8),
            covariance_tile_size=case["covariance_tile"],
        )
        dataset = value_dataset(case["n"])
        queries = prediction_queries(case["queries"])

        def infer_gpex(gp, dataset, query):
            result = gp.condition(dataset).predict(query)
            return result.mean.flat, result.covariance.full

        operation = infer_gpex
        arguments = tuple((gp, dataset, query) for query in queries)
    elif method == "gpjax":
        import gpjax as gpx
        import jax.numpy as jnp
        from compare import gpjax_model

        x, y = data(case["n"])
        model = gpjax_model()

        def infer_gpjax(model, x, y, query):
            dataset = gpx.Dataset(X=x, y=y)
            result = model.predict(query, train_data=dataset, covariance="dense")
            return result.mean, result.covariance()

        operation = infer_gpjax
        arguments = tuple(
            (
                model,
                jnp.asarray(x),
                jnp.asarray(y[:, None]),
                jnp.asarray(query_points(case["queries"], shift)),
            )
            for shift in (0.0, 0.037)
        )
    else:
        from compare import torch_model

        model, array = torch_model(case, *data(case["n"]))

        def infer_torch(query):
            # This workload reconditions every time, including warm calls.
            model.prediction_strategy = None
            result: Any = model(query)
            return result.mean, result.covariance_matrix

        operation = infer_torch
        arguments = tuple(
            (array(query_points(case["queries"], shift)),) for shift in (0.0, 0.037)
        )

    def validate(result, args):
        check_variance(
            np.diag(np.asarray(result[1])),
            case,
            evidence,
            prior_variance=1.0 + (1e-6 if method == "gpjax" else 0.0),
        )

    return operation, arguments, validate, evidence


if __name__ == "__main__":
    from run import main

    main(default_workloads=["dense", "gpjax", "gpytorch"])
