"""Prediction workload. Run this file with --help for options."""

from __future__ import annotations

import jax
import numpy as np
from _common import prediction_queries, value_dataset
from checks import check_variance

from gpex import GaussianProcess
from gpex.kernels import RBFKernel


def prepare(case):
    gp = GaussianProcess(
        RBFKernel(variance=1.0, lengthscale=0.8),
        covariance_tile_size=case["covariance_tile"],
    )
    dataset = value_dataset(case["n"])
    queries = prediction_queries(case["queries"])
    evidence = {}
    workload = case["workload"]
    posterior = gp.condition(dataset)
    jax.block_until_ready(posterior)

    def predict(current, query):
        if workload == "predict-marginals":
            result = current.predict_marginals(query)
            return result.mean.flat, result.variance.flat
        result = current.predict(query)
        return result.mean.flat, result.covariance.full

    def validate(result, args):
        variance = np.asarray(result[1])
        if workload == "predict":
            variance = np.diag(variance)
        check_variance(variance, case, evidence)

    evidence["conditioning_included"] = False
    return (
        predict,
        tuple((posterior, query) for query in queries),
        validate,
        evidence,
    )


if __name__ == "__main__":
    from run import main

    main(default_workloads=["predict", "predict-marginals"])
