"""Pyright fixtures for preserving kernel capabilities through GP operations."""

# pyright: reportUnnecessaryTypeIgnoreComment=true

from typing import assert_type

import jax

from gpex import (
    ConditionedPosterior,
    ConditioningState,
    FeaturePosteriorPath,
    FunctionalDataset,
    FunctionalQuery,
    GaussianProcess,
)
from gpex.blocks import FunctionalValues, Locations
from gpex.kernels import RBFKernel, SumKernel


def feature_kernel_workflow(
    dataset: FunctionalDataset, locations: Locations, values: FunctionalValues
) -> None:
    gp = GaussianProcess(RBFKernel())
    assert_type(gp, GaussianProcess[RBFKernel])
    conditioning = gp.prepare_conditioning(dataset)
    assert_type(conditioning, ConditioningState[RBFKernel])
    assert_type(
        conditioning.append(locations, block="values"), ConditioningState[RBFKernel]
    )
    updated, _ = conditioning.append_solve(
        locations, dataset.values_flat, dataset.values_flat, block="values"
    )
    assert_type(updated, ConditioningState[RBFKernel])
    posterior = gp.condition(dataset)
    assert_type(posterior, ConditionedPosterior[RBFKernel])
    assert_type(
        posterior.append(locations, values, block="values"),
        ConditionedPosterior[RBFKernel],
    )
    path = FeaturePosteriorPath(posterior, jax.random.key(0))
    assert_type(path, FeaturePosteriorPath[RBFKernel])
    assert_type(path.kernel, RBFKernel)


def covariance_only_workflow(
    dataset: FunctionalDataset, query: FunctionalQuery
) -> None:
    posterior = GaussianProcess(RBFKernel() + RBFKernel()).condition(dataset)
    assert_type(posterior, ConditionedPosterior[SumKernel])
    posterior.sample(jax.random.key(0), query)
    FeaturePosteriorPath(posterior, jax.random.key(0))  # pyright: ignore[reportArgumentType]
