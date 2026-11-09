"""Numerical validation that remains active under JAX transformations."""

from math import isfinite

import equinox as eqx
import jax
from jax.core import Tracer


def validate_jitter(jitter: float) -> None:
    """Validate the static diagonal stabilization used by numerical systems."""
    if not isfinite(jitter) or jitter < 0:
        raise ValueError("jitter must be finite and nonnegative")


def error_if[T](value: T, predicate: bool | jax.Array, message: str) -> T:
    """Raise ValueError eagerly, or attach a runtime check to traced values."""
    if isinstance(predicate, Tracer):
        return eqx.error_if(value, predicate, message)
    if bool(predicate):
        raise ValueError(message)
    return value
