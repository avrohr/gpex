"""Explicit trainable parameters and prior traversal."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..validation.checks import error_if
from ..validation.cholesky import validate_cholesky_factor
from .priors import Prior
from .transforms import (
    CholeskyTransform,
    IdentityTransform,
    ParameterValue,
    SoftplusTransform,
    Transform,
)

Scalar = Float[Array, ""]


class Parameter(eqx.Module):
    """Trainable optimization coordinates with a transform and optional prior.

    Args:
        unconstrained: Initial raw scalar or array, stored directly. The model
            value is ``transform.forward(unconstrained)`` and may have a different
            shape, for example a matrix represented by packed coordinates.
        transform: Map from optimization coordinates to constrained values.
            Defaults to ``IdentityTransform``.
        prior: Distribution with ``log_prob(value)`` returning one scalar joint
            log density. Defaults to ``None``; MAP requires an explicit prior.
        name: Optional diagnostic name. Defaults to ``None``.

    Construction supports JIT. Use ``PositiveParameter(value)`` or
    ``CholeskyParameter(value)`` to initialize from a constrained model value.
    """

    unconstrained: ParameterValue
    transform: Transform = eqx.field(static=True)
    prior: Prior | None = eqx.field(static=True)
    name: str | None = eqx.field(static=True)

    def __init__(
        self,
        *,
        unconstrained: ParameterValue | float,
        transform: Transform | None = None,
        prior: Prior | None = None,
        name: str | None = None,
    ) -> None:
        self.unconstrained = jnp.asarray(unconstrained)
        self.transform = IdentityTransform() if transform is None else transform
        self.prior = prior
        self.name = name

    @property
    def value(self) -> ParameterValue:
        return self.transform.forward(self.unconstrained)

    def log_prior(self) -> Scalar:
        """Return the joint log density on ``value``, or zero without a prior."""
        if self.prior is None:
            return jnp.zeros(())
        return self.prior.log_prob(self.value)



class PositiveParameter(Parameter):
    """A positive scalar or array optimized through a softplus transform.

    Args:
        value: Positive initial scalar or array, for example a lengthscale vector
            of shape ``(d,)``. Values must be finite.
        prior: Optional joint prior on the complete positive value. Defaults to
            ``None``. The optimization transform does not add a density Jacobian.
        name: Optional diagnostic name. Defaults to ``None``.

    Construction supports JIT, batching, and differentiation of valid values.
    The transform, prior, and name remain static metadata.
    """

    def __init__(
        self,
        value: ParameterValue | float,
        *,
        prior: Prior | None = None,
        name: str | None = None,
    ) -> None:
        value = jnp.asarray(value)
        value = error_if(
            value,
            jnp.any(~jnp.isfinite(value) | (value <= 0)),
            "positive parameters must be initialized with finite values above zero",
        )
        transform = SoftplusTransform()
        super().__init__(
            unconstrained=transform.inverse(value),
            transform=transform,
            prior=prior,
            name=name,
        )



class CholeskyParameter(Parameter):
    """A trainable lower-triangular matrix with positive diagonal.

    Args:
        value: Complete initial matrix of shape ``(d, d)``. It must be finite,
            lower triangular, and have a positive diagonal.
        prior: Optional joint prior on the complete constrained matrix. MAP
            requires a prior; maximum likelihood does not. Defaults to ``None``.
        name: Optional diagnostic name. Defaults to ``None``.

    Only the ``d * (d + 1) / 2`` free entries are optimized, in row-major order.
    Diagonal entries use softplus; strict-lower entries are unrestricted. Priors
    are evaluated on ``value`` without an optimization-coordinate Jacobian.
    Construction supports JIT, batching, and differentiation of valid values;
    the matrix shape, prior, and name remain static.
    """

    def __init__(
        self,
        value: ParameterValue,
        *,
        prior: Prior | None = None,
        name: str | None = None,
    ) -> None:
        value = validate_cholesky_factor(value)
        transform = CholeskyTransform(value.shape[0])
        super().__init__(
            unconstrained=transform.inverse(value),
            transform=transform,
            prior=prior,
            name=name,
        )



def log_prior(tree: Any, *, require_all: bool = False) -> Scalar:
    """Sum priors explicitly attached to all parameters in a PyTree."""
    parameters = jax.tree_util.tree_leaves(
        tree, is_leaf=lambda item: isinstance(item, Parameter)
    )
    parameters = [item for item in parameters if isinstance(item, Parameter)]
    if not parameters:
        return jnp.zeros(())
    if require_all:
        missing = [
            parameter.name or "<unnamed>"
            for parameter in parameters
            if parameter.prior is None
        ]
        if missing:
            raise ValueError(
                "MAP requires explicit priors for every Parameter; missing: "
                + ", ".join(missing)
            )
    return sum((parameter.log_prior() for parameter in parameters), jnp.zeros(()))



__all__ = ["Parameter", "PositiveParameter", "CholeskyParameter", "log_prior"]
