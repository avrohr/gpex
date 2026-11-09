"""Parameter selection, numerical checks, and argument validation."""

from collections.abc import Callable
from math import isfinite
from typing import Any, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..parameters import Parameter

Model = TypeVar("Model")
Scalar = Float[Array, ""]
Objective = Callable[[Model], Scalar]


def partition_trainable(model: Any) -> tuple[Any, Any]:
    """Train only explicit parameters; ordinary model arrays remain fixed."""
    filter_spec = jax.tree_util.tree_map(
        lambda item: isinstance(item, Parameter),
        model,
        is_leaf=lambda item: isinstance(item, Parameter),
    )
    trainable, fixed = eqx.partition(model, filter_spec)
    if not any(eqx.is_inexact_array(x) for x in jax.tree_util.tree_leaves(trainable)):
        raise ValueError("model does not contain any explicit Parameter objects")
    return trainable, fixed


def tree_norm(tree: Any) -> Scalar:
    """Compute the Euclidean norm without squaring unscaled gradient entries."""
    leaves = jax.tree_util.tree_leaves(tree)
    scale = jnp.asarray(0.0)
    for leaf in leaves:
        scale = jnp.maximum(scale, jnp.max(jnp.abs(leaf), initial=0.0))
    # Scaling cancels algebraically; differentiating it adds unnecessary work.
    # A zero tree uses unit scale to avoid division by zero.
    scale = jax.lax.stop_gradient(jnp.where(scale == 0.0, 1.0, scale))
    scaled_leaves = [leaf / scale for leaf in leaves]
    return scale * jnp.sqrt(
        sum(
            jnp.vdot(x, x, precision=jax.lax.Precision.HIGHEST).real
            for x in scaled_leaves
        )
    )


def tree_all_finite(tree: Any) -> jax.Array:
    finite = jnp.asarray(True)
    for leaf in jax.tree_util.tree_leaves(tree):
        finite = finite & jnp.all(jnp.isfinite(leaf))
    return finite


def check_finite(
    params: Any, value: Scalar, gradient: Any, *, active: bool | Array = True
) -> tuple[Any, Scalar, Any]:
    """Raise at runtime, including under JIT, for an invalid accepted iterate."""
    finite = tree_all_finite(params) & jnp.isfinite(value) & tree_all_finite(gradient)
    return eqx.error_if(
        (params, value, gradient),
        active & ~finite,
        "Fitting encountered non-finite parameters, loss, or gradient.",
    )


def validate_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")


def validate_tolerance(value: float) -> None:
    if not isfinite(value) or value < 0:
        raise ValueError("gradient_tolerance must be finite and nonnegative")
