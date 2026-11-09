"""Batch independent fitting calls and select by the final objective."""

from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import PRNGKeyArray

from ._common import Model, Objective, partition_trainable
from .results import (
    FitResult,
    MultiStartFitResult,
    MultiStartRandomSubsetFitResult,
)
from .subsets import GPModel, SubsetSize, fit_random_subsets


def fit_multistart(
    starts: Sequence[Model],
    fit_one: Callable[[Model], FitResult[Model]],
) -> MultiStartFitResult[Model]:
    """Vectorize a fitting callable over informed parameter starts.

    For example, ``fit_multistart(starts, lambda m: fit_lbfgs(m, objective))``.
    Starts may differ only in their explicit parameter arrays: structure,
    shapes, parameter metadata, and fixed leaves must agree. Each run owns its
    optimizer state. Any failed run raises; finite final objectives determine
    the selected model. This function selects ``best_index`` on the host; use
    the single-fit functions when nesting fitting inside another JIT.
    """
    runs, models = _map_starts(starts, fit_one)
    return MultiStartFitResult(
        models=models,
        losses=runs.losses,
        optimizer_state=runs.optimizer_state,
        final_losses=runs.loss,
        gradient_norms=runs.gradient_norm,
        step_counts=runs.step_count,
        evaluation_counts=runs.evaluation_count,
        converged=runs.converged,
        best_index=int(jax.device_get(jnp.argmin(runs.loss))),
    )


def fit_multistart_random_subsets(
    starts: Sequence[GPModel],
    subset_objective: Objective[GPModel],
    optimizer: optax.GradientTransformation,
    *,
    step_count: int,
    subset_size: SubsetSize,
    reference_objective: Objective[GPModel],
    key: PRNGKeyArray,
) -> MultiStartRandomSubsetFitResult[GPModel]:
    """Run ``fit_random_subsets`` with an independent key per start.

    Arguments and failure behavior match ``fit_random_subsets``. Starts must
    satisfy the same structure and fixed-leaf requirements as ``fit_multistart``.
    Select the final model with the lowest full-data reference objective.
    """
    keys = jax.random.split(key, len(starts))

    def fit_one(model: GPModel, start_key: PRNGKeyArray):
        return fit_random_subsets(
            model,
            subset_objective,
            optimizer,
            step_count=step_count,
            subset_size=subset_size,
            reference_objective=reference_objective,
            key=start_key,
        )

    runs, models = _map_starts(starts, fit_one, keys)
    return MultiStartRandomSubsetFitResult(
        models=models,
        subset_losses=runs.subset_losses,
        optimizer_state=runs.optimizer_state,
        reference_losses=runs.reference_loss,
        reference_gradient_norms=runs.reference_gradient_norm,
        step_counts=runs.step_count,
        evaluation_counts=runs.evaluation_count,
        best_index=int(jax.device_get(jnp.argmin(runs.reference_loss))),
    )


def _prepare_starts(starts: Sequence[Any]) -> tuple[Any, Any]:
    if not starts:
        raise ValueError("at least one start is required")
    first, fixed = partition_trainable(starts[0])
    trainables = [first]
    structure = jax.tree_util.tree_structure(first)
    shapes = [x.shape for x in jax.tree_util.tree_leaves(first)]
    for start in starts[1:]:
        params, current_fixed = partition_trainable(start)
        if (
            jax.tree_util.tree_structure(params) != structure
            or [x.shape for x in jax.tree_util.tree_leaves(params)] != shapes
            or not eqx.tree_equal(current_fixed, fixed)
        ):
            raise ValueError(
                "starts must have identical structure, parameter shapes, "
                "parameter metadata, and fixed leaves"
            )
        trainables.append(params)
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *trainables), fixed


@eqx.filter_jit
def _run_batched(params: Any, fixed: Any, fit_one: Callable, *mapped_args: Any):
    def run_one(candidate: Any, *args: Any):
        result = fit_one(eqx.combine(candidate, fixed), *args)
        # Keep fixed data shared rather than broadcasting it into every result.
        fitted_params, _ = partition_trainable(result.model)
        return eqx.tree_at(lambda r: r.model, result, fitted_params)

    return eqx.filter_vmap(run_one)(params, *mapped_args)


def _map_starts(starts: Sequence[Any], fit_one: Callable, *mapped_args: Any):
    params, fixed = _prepare_starts(starts)
    runs = _run_batched(params, fixed, fit_one, *mapped_args)
    models = tuple(
        eqx.combine(
            jax.tree_util.tree_map(lambda x, index=i: x[index], runs.model), fixed
        )
        for i in range(len(starts))
    )
    return runs, models


__all__ = [
    "fit_multistart",
    "fit_multistart_random_subsets",
]
