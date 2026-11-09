"""Fixed-step fitting with a caller-supplied Optax optimizer."""

from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from ._common import (
    Model,
    Objective,
    Scalar,
    partition_trainable,
    tree_all_finite,
    tree_norm,
    validate_positive_integer,
)
from .results import FitResult


class _State(NamedTuple):
    params: Any
    optimizer_state: Any
    value: Scalar
    gradient: Any
    all_finite: jax.Array


def fit(
    model: Model,
    objective: Objective[Model],
    optimizer: optax.GradientTransformation,
    *,
    step_count: int,
) -> FitResult[Model]:
    """Minimize a scalar objective for exactly ``step_count`` updates.

    Only explicit ``Parameter`` leaves are trained. Supply an Optax optimizer
    such as ``optax.adam(0.01)`` or a composed transformation whose update needs
    only gradients, state, and parameters. Use ``fit_lbfgs`` for L-BFGS.

    The result contains the final iterate, its optimizer state, and a loss trace
    of length ``step_count + 1``. There is no early stopping. The compiled scan
    supports reverse-mode differentiation through the optimization trajectory.
    Every iterate is checked for finite parameters, objective, and gradient.
    Any failure raises after the loop completes, even if later iterates are finite.
    """
    validate_positive_integer("step_count", step_count)
    return _fit(model, objective, optimizer, step_count=step_count)


@eqx.filter_jit
def _fit(
    model: Model,
    objective: Objective[Model],
    optimizer: optax.GradientTransformation,
    *,
    step_count: int,
) -> FitResult[Model]:
    params, fixed = partition_trainable(model)

    def value_fn(candidate: Any) -> Scalar:
        return objective(eqx.combine(candidate, fixed))

    value_and_grad = jax.value_and_grad(value_fn)
    initial_value, gradient = value_and_grad(params)
    initial = _State(
        params,
        optimizer.init(params),
        initial_value,
        gradient,
        tree_all_finite((params, initial_value, gradient)),
    )

    def step(state: _State, _) -> tuple[_State, Scalar]:
        updates, optimizer_state = optimizer.update(
            state.gradient, state.optimizer_state, state.params
        )
        params = optax.apply_updates(state.params, updates)
        value, gradient = value_and_grad(params)
        all_finite = state.all_finite & tree_all_finite((params, value, gradient))
        return _State(params, optimizer_state, value, gradient, all_finite), value

    final, losses = jax.lax.scan(step, initial, xs=None, length=step_count)
    # An error branch in every scan iteration adds GPU control-flow overhead.
    # Accumulate failures above and guard computed results once, so consuming
    # only a fitted parameter or loss still detects any invalid iterate.
    params, optimizer_state, value, gradient, losses = eqx.error_if(
        (
            final.params,
            final.optimizer_state,
            final.value,
            final.gradient,
            jnp.concatenate((initial_value[None], losses)),
        ),
        ~final.all_finite,
        "Fitting encountered non-finite parameters, loss, or gradient.",
    )
    return FitResult(
        model=eqx.combine(params, fixed),
        losses=losses,
        optimizer_state=optimizer_state,
        loss=value,
        gradient_norm=tree_norm(gradient),
        step_count=jnp.asarray(step_count),
        evaluation_count=jnp.asarray(step_count + 1),
        converged=jnp.asarray(False),
    )


__all__ = [
    "fit",
]
