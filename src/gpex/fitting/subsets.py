"""Fixed-step Optax fitting on freshly sampled observation subsets."""

from collections.abc import Mapping
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import PRNGKeyArray

from ..blocks import (
    FunctionalBlock,
    FunctionalDataset,
    Locations,
    ObservationBlock,
    ObservationDesignBlock,
)
from ..gp import GaussianProcess
from ..validation.checks import error_if
from ._common import (
    Objective,
    Scalar,
    check_finite,
    partition_trainable,
    tree_all_finite,
    tree_norm,
    validate_positive_integer,
)
from .results import RandomSubsetFitResult

type GPModel = tuple[GaussianProcess, FunctionalDataset]
type SubsetSize = int | Mapping[str, int]


class _State(NamedTuple):
    params: Any
    optimizer_state: Any
    key: PRNGKeyArray


def fit_random_subsets(
    model: GPModel,
    subset_objective: Objective[GPModel],
    optimizer: optax.GradientTransformation,
    *,
    step_count: int,
    subset_size: SubsetSize,
    reference_objective: Objective[GPModel],
    key: PRNGKeyArray,
) -> RandomSubsetFitResult[GPModel]:
    """Fit ``(gp, dataset)`` with a fresh subset per Optax update.

    Supply a first-order optimizer such as ``optax.adam(0.01)``. ``subset_size``
    is an active location count per block, or a mapping from every block name to
    its count. Sampling is uniform without replacement over active locations and
    reproducible from ``key``. Every block needs at least that many active rows.
    A subset GP likelihood is an approximation, not an unbiased estimate of the
    full likelihood. The caller defines any scaling in ``subset_objective``.

    Run exactly ``step_count`` updates, then evaluate ``reference_objective``
    and its gradient once on the full data. Return the final model; non-finite
    iterates, subset diagnostics, or reference diagnostics raise an error.
    """
    validate_positive_integer("step_count", step_count)
    _validate_gp_model(model)
    sizes = _resolve_subset_sizes(model[1], subset_size)
    return _fit_random_subsets(
        model,
        subset_objective,
        optimizer,
        step_count=step_count,
        subset_sizes=sizes,
        reference_objective=reference_objective,
        key=key,
    )


@eqx.filter_jit
def _fit_random_subsets(
    model: GPModel,
    subset_objective: Objective[GPModel],
    optimizer: optax.GradientTransformation,
    *,
    step_count: int,
    subset_sizes: tuple[int, ...],
    reference_objective: Objective[GPModel],
    key: PRNGKeyArray,
) -> RandomSubsetFitResult[GPModel]:
    params, fixed = partition_trainable(model)

    def subset_value_fn(candidate: Any, subset_key: PRNGKeyArray) -> Scalar:
        candidate_model = eqx.combine(candidate, fixed)
        return subset_objective(
            _select_random_subset(candidate_model, subset_sizes, subset_key)
        )

    def reference_value_fn(candidate: Any) -> Scalar:
        return reference_objective(eqx.combine(candidate, fixed))

    def step(state: _State, _) -> tuple[_State, Scalar]:
        next_key, subset_key = jax.random.split(state.key)
        value, gradient = jax.value_and_grad(subset_value_fn)(state.params, subset_key)
        params, value, gradient = check_finite(state.params, value, gradient)
        updates, optimizer_state = optimizer.update(
            gradient, state.optimizer_state, params
        )
        params = optax.apply_updates(params, updates)
        # Only parameters changed. Attach this check to the recorded loss too,
        # so it remains active when a caller returns just the loss history.
        params, value = eqx.error_if(
            (params, value),
            ~tree_all_finite(params),
            "Fitting encountered non-finite parameters after an optimizer update.",
        )
        return _State(params, optimizer_state, next_key), value

    initial = _State(params, optimizer.init(params), key)
    final, losses = jax.lax.scan(step, initial, xs=None, length=step_count)
    value, gradient = jax.value_and_grad(reference_value_fn)(final.params)
    result = RandomSubsetFitResult(
        model=eqx.combine(final.params, fixed),
        subset_losses=losses,
        optimizer_state=final.optimizer_state,
        reference_loss=value,
        reference_gradient_norm=tree_norm(gradient),
        step_count=jnp.asarray(step_count),
        evaluation_count=jnp.asarray(step_count + 1),
    )
    # Every numerical output must depend on the final reference validation,
    # including histories and counters selected by an outer compiled caller.
    return eqx.error_if(
        result,
        ~tree_all_finite((final.params, value, gradient)),
        "Fitting encountered non-finite parameters, loss, or gradient.",
    )


def _validate_gp_model(model: Any) -> None:
    if (
        not isinstance(model, tuple)
        or len(model) != 2
        or not isinstance(model[0], GaussianProcess)
        or not isinstance(model[1], FunctionalDataset)
    ):
        raise TypeError("random-subset fitting requires a (gp, dataset) model")


def _resolve_subset_sizes(
    dataset: FunctionalDataset,
    subset_size: SubsetSize,
) -> tuple[int, ...]:
    names = tuple(block.functional.name for block in dataset.blocks)
    if isinstance(subset_size, int):
        requested = {name: subset_size for name in names}
    elif isinstance(subset_size, Mapping):
        if set(subset_size) != set(names):
            raise ValueError("subset_size must specify every observation block")
        requested = dict(subset_size)
    else:
        raise TypeError("subset_size must be an integer or block-name mapping")

    sizes = []
    for block in dataset.blocks:
        functional = block.functional
        size = requested[functional.name]
        if isinstance(size, bool) or not isinstance(size, int):
            raise TypeError("subset_size must contain integers")
        if size < 1 or size > functional.locations.count:
            raise ValueError(
                f"subset size for {functional.name!r} must lie between 1 and "
                f"{functional.locations.count}"
            )
        sizes.append(size)
    return tuple(sizes)


def _select_random_subset(
    model: GPModel,
    subset_sizes: tuple[int, ...],
    key: PRNGKeyArray,
) -> GPModel:
    gp, dataset = model
    keys = jax.random.split(key, len(dataset.blocks))
    selected_blocks = []
    for block, size, block_key in zip(dataset.blocks, subset_sizes, keys, strict=True):
        functional = block.functional
        active = error_if(
            block.design.active,
            block.count < size,
            "subset size exceeds the active location count",
        )
        indices = jnp.sort(
            jax.random.choice(
                block_key,
                functional.locations.count,
                shape=(size,),
                replace=False,
                p=active / jnp.sum(active),
            )
        )
        selected_functional = FunctionalBlock(
            Locations(functional.locations.points[indices]),
            functional.operator,
            functional.name,
        )
        selected_design = ObservationDesignBlock(
            selected_functional, block.design.variance, max_size=size
        )
        selected_blocks.append(ObservationBlock(selected_design, block.y[indices]))
    return gp, FunctionalDataset(selected_blocks)


__all__ = [
    "fit_random_subsets",
]
