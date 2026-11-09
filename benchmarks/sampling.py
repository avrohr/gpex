"""Sampling workload. Run this file with --help for options."""

from __future__ import annotations

import jax
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

    def operation(model, observations, query):
        posterior = model.condition(observations)
        if workload == "marginals":
            result = posterior.predict_marginals(query)
            return result.mean.flat, result.variance.flat
        return posterior.sample(jax.random.key(42), query, sample_count=8).flat

    def validate(result, args):
        if workload == "marginals":
            check_variance(result[1], case, evidence)

    return (
        operation,
        tuple((gp, dataset, query) for query in queries),
        validate,
        evidence,
    )


if __name__ == "__main__":
    from run import main

    main(default_workloads=["marginals", "sample"])
