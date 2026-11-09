"""Maps between optimization coordinates and constrained parameter values."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

ParameterValue = Float[Array, "*parameter_shape"]


class Transform(Protocol):
    """Map between optimization coordinates and constrained parameter values."""

    @abstractmethod
    def forward(self, unconstrained: ParameterValue) -> ParameterValue: ...

    @abstractmethod
    def inverse(self, constrained: ParameterValue) -> ParameterValue: ...



@dataclass(frozen=True)
class IdentityTransform(Transform):
    def forward(self, unconstrained: ParameterValue) -> ParameterValue:
        return unconstrained

    def inverse(self, constrained: ParameterValue) -> ParameterValue:
        return constrained



@dataclass(frozen=True)
class ExpTransform(Transform):
    def forward(self, unconstrained: ParameterValue) -> ParameterValue:
        return jnp.exp(unconstrained)

    def inverse(self, constrained: ParameterValue) -> ParameterValue:
        return jnp.log(constrained)



@dataclass(frozen=True)
class SoftplusTransform(Transform):
    def forward(self, unconstrained: ParameterValue) -> ParameterValue:
        return jax.nn.softplus(unconstrained)

    def inverse(self, constrained: ParameterValue) -> ParameterValue:
        return constrained + jnp.log(-jnp.expm1(-constrained))



@dataclass(frozen=True)
class CholeskyTransform(Transform):
    """Pack a lower-triangular factor using unconstrained diagonal coordinates.

    Args:
        dimension: Positive matrix dimension ``d``, fixed under JAX transformations.

    ``forward`` maps a vector of shape ``(d * (d + 1) // 2,)`` to a matrix of
    shape ``(d, d)``. Entries follow row-major lower-triangle order. Diagonal
    coordinates use softplus, and strict-lower entries remain unrestricted.
    ``inverse`` expects a finite lower-triangular matrix with positive diagonal.
    Both operations support JIT, vmap, and differentiation of valid inputs.
    """

    dimension: int

    def forward(self, unconstrained: ParameterValue) -> ParameterValue:
        rows, columns = jnp.tril_indices(self.dimension)
        # Transform only the diagonal: unrestricted entries remain untouched.
        diagonal = jnp.arange(self.dimension)
        diagonal_positions = (diagonal + 1) * (diagonal + 2) // 2 - 1
        entries = unconstrained.at[diagonal_positions].set(
            jax.nn.softplus(unconstrained[diagonal_positions])
        )
        return (
            jnp.zeros((self.dimension, self.dimension)).at[rows, columns].set(entries)
        )

    def inverse(self, constrained: ParameterValue) -> ParameterValue:
        diagonal = jnp.diag(constrained)
        raw_diagonal = SoftplusTransform().inverse(diagonal)
        indices = jnp.arange(self.dimension)
        return constrained.at[indices, indices].set(raw_diagonal)[
            jnp.tril_indices(self.dimension)
        ]



__all__ = [
    "Transform",
    "IdentityTransform",
    "ExpTransform",
    "SoftplusTransform",
    "CholeskyTransform",
]
