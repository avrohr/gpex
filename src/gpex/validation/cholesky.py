"""Shared structure and support checks for Cholesky factors."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from .checks import error_if

ParameterValue = Float[Array, "*parameter_shape"]


def validate_factor_shape(value: ParameterValue) -> ParameterValue:
    """Return a real array of nonempty square shape, or raise ValueError."""
    value = jnp.asarray(value)
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[0] != value.shape[1]:
        raise ValueError("L must be a nonempty square matrix")
    if jnp.issubdtype(value.dtype, jnp.complexfloating):
        raise ValueError("L must be real")
    return value


def is_cholesky_factor(value: ParameterValue) -> Array:
    """Test finite lower-triangular support with positive diagonal under JIT.

    The input must already have real, nonempty square matrix structure.
    """
    return (
        jnp.all(jnp.isfinite(value))
        & jnp.all(jnp.triu(value, k=1) == 0)
        & jnp.all(jnp.diag(value) > 0)
    )


def validate_cholesky_factor(value: ParameterValue) -> ParameterValue:
    """Return a valid factor, attaching numerical checks under JIT and vmap."""
    value = validate_factor_shape(value)
    return error_if(
        value,
        ~is_cholesky_factor(value),
        "L must be finite and lower triangular with positive diagonal",
    )
