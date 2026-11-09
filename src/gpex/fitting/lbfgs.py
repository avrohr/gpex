"""L-BFGS fitting with cached line-search evaluations and gradient stopping."""

from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from ._common import (
    Model,
    Objective,
    Scalar,
    check_finite,
    partition_trainable,
    tree_norm,
    validate_positive_integer,
    validate_tolerance,
)
from .results import FitResult, LossHistory, ScalarInteger


class _State(NamedTuple):
    params: Any
    optimizer_state: Any
    value: Scalar
    gradient: Any
    losses: LossHistory
    step_count: ScalarInteger
    evaluation_count: ScalarInteger


def fit_lbfgs(
    model: Model,
    objective: Objective[Model],
    *,
    max_steps: int = 100,
    gradient_tolerance: float = 1e-5,
    memory_size: int = 10,
    max_linesearch_steps: int = 20,
) -> FitResult[Model]:
    """Fit a deterministic objective with Optax L-BFGS and a zoom line search.

    Stop when the gradient norm is at most ``gradient_tolerance`` or after
    ``max_steps`` updates. Only explicit ``Parameter`` leaves are trained.
    Cached line-search values and gradients avoid reevaluating accepted points.
    Non-finite accepted iterates and failed line searches raise an error.

    The result contains the final iterate. Its loss trace has length
    ``max_steps + 1``, with NaN after ``step_count + 1`` valid entries. Check
    ``converged`` to distinguish gradient convergence from the iteration limit.
    This dynamic loop cannot be reverse-mode differentiated through; the
    objective itself is differentiated normally.
    """
    validate_positive_integer("max_steps", max_steps)
    validate_positive_integer("memory_size", memory_size)
    validate_positive_integer("max_linesearch_steps", max_linesearch_steps)
    validate_tolerance(gradient_tolerance)
    return _fit_lbfgs(
        model,
        objective,
        max_steps=max_steps,
        gradient_tolerance=gradient_tolerance,
        memory_size=memory_size,
        max_linesearch_steps=max_linesearch_steps,
    )


@eqx.filter_jit
def _fit_lbfgs(
    model: Model,
    objective: Objective[Model],
    *,
    max_steps: int,
    gradient_tolerance: float,
    memory_size: int,
    max_linesearch_steps: int,
) -> FitResult[Model]:
    params, fixed = partition_trainable(model)

    def value_fn(candidate: Any) -> Scalar:
        return objective(eqx.combine(candidate, fixed))

    optimizer = optax.lbfgs(
        memory_size=memory_size,
        linesearch=optax.scale_by_zoom_linesearch(
            max_linesearch_steps=max_linesearch_steps,
            initial_guess_strategy="one",
        ),
    )
    value, gradient = jax.value_and_grad(value_fn)(params)
    params, value, gradient = check_finite(params, value, gradient)
    initial = _State(
        params=params,
        optimizer_state=optimizer.init(params),
        value=value,
        gradient=gradient,
        losses=jnp.full((max_steps + 1,), jnp.nan).at[0].set(value),
        step_count=jnp.asarray(0),
        evaluation_count=jnp.asarray(1),
    )

    def continuing(state: _State):
        return (state.step_count < max_steps) & (
            tree_norm(state.gradient) > gradient_tolerance
        )

    def step(state: _State) -> _State:
        updates, optimizer_state = optimizer.update(
            state.gradient,
            state.optimizer_state,
            state.params,
            value=state.value,
            grad=state.gradient,
            value_fn=value_fn,
        )
        params = optax.apply_updates(state.params, updates)
        # These fields belong to the zoom line search constructed above.
        value = optax.tree.get(optimizer_state, "value")
        gradient = optax.tree.get(optimizer_state, "grad")
        info = optax.tree.get(optimizer_state, "info")
        assert value is not None and gradient is not None and info is not None
        # A vmapped while loop also evaluates lanes that have already finished.
        # Their proposals are discarded, so only check still-active lanes.
        params, value, gradient = eqx.error_if(
            (params, value, gradient),
            continuing(state)
            & ((info.decrease_error > 0) | (info.curvature_error > 0)),
            "L-BFGS line search failed to satisfy its decrease/curvature criteria. "
            "Check the objective, gradient_tolerance, max_linesearch_steps, "
            "and numerical precision.",
        )
        params, value, gradient = check_finite(
            params, value, gradient, active=continuing(state)
        )
        count = state.step_count + 1
        return _State(
            params=params,
            optimizer_state=optimizer_state,
            value=value,
            gradient=gradient,
            losses=state.losses.at[count].set(value),
            step_count=count,
            evaluation_count=state.evaluation_count + info.num_linesearch_steps,
        )

    final = jax.lax.while_loop(continuing, step, initial)
    gradient_norm = tree_norm(final.gradient)
    return FitResult(
        model=eqx.combine(final.params, fixed),
        losses=final.losses,
        optimizer_state=final.optimizer_state,
        loss=final.value,
        gradient_norm=gradient_norm,
        step_count=final.step_count,
        evaluation_count=final.evaluation_count,
        converged=gradient_norm <= gradient_tolerance,
    )


__all__ = [
    "fit_lbfgs",
]
