"""Fitting workload. Run this file with --help for options."""

from __future__ import annotations

import numpy as np
import optax
from _common import value_dataset

from gpex import (
    GaussianProcess,
)
from gpex.fitting import fit, negative_log_marginal_likelihood, negative_log_posterior
from gpex.kernels import RBFKernel
from gpex.parameters import PositiveParameter
from gpex.parameters.priors import GammaPrior


def prepare(case):
    dataset = value_dataset(case["n"])
    evidence = {}
    workload = case["workload"]
    gp = GaussianProcess(
        RBFKernel(
            variance=PositiveParameter(
                1.0, prior=GammaPrior(concentration=2.0, rate=2.0)
            ),
            lengthscale=PositiveParameter(
                0.8, prior=GammaPrior(concentration=2.0, rate=2.0)
            ),
        ),
        covariance_tile_size=case["covariance_tile"],
    )

    def objective(model):
        return (
            negative_log_posterior(*model)
            if workload == "map"
            else negative_log_marginal_likelihood(*model)
        )

    def fitting(model):
        fitted = fit(model, objective, optax.adam(0.02), step_count=case["steps"])
        return fitted.losses, fitted.model

    inputs = (((gp, dataset),),)

    def validate(result, args):
        # The fit already returns the initial and final objective. Avoid
        # extra factorizations just to report optimizer progress.
        losses = np.asarray(result[0])
        evidence.update(
            initial_objective=float(losses[0]),
            final_objective=float(losses[-1]),
        )

    return fitting, inputs, validate, evidence


if __name__ == "__main__":
    from run import main

    main(default_workloads=["mle", "map"])
