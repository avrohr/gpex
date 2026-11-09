"""GP covariance descriptions consumed by linear-algebra backends."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from .linalg.base import (
    AppendedMatrix,
    CrossMatrix,
    LinearSystemUpdate,
    RightHandSide,
    SquareMatrix,
    SymmetricLinearOperator,
    Vector,
    symmetrize,
)
from .moments.assembly import (
    ScalarKernel,
    apply_covariance,
    assemble_covariance,
    assemble_covariance_diagonal,
)
from .moments.measurements import PreparedFunctionals
from .validation.checks import validate_jitter


class CovarianceSystem(eqx.Module, SymmetricLinearOperator):
    """A stabilized GP covariance exposed as generic linear-algebra operations.

    ``variance`` contains measurement-error variances in canonical scalar-output order.
    """

    measurements: PreparedFunctionals
    kernel: ScalarKernel
    variance: Vector
    jitter: float = eqx.field(static=True)

    def __post_init__(self) -> None:
        validate_jitter(self.jitter)

    @property
    def capacity(self) -> int:
        return self.measurements.size

    def diagonal_shift(self) -> Vector:
        """Return the positive diagonal observation shift."""
        return jnp.where(self.measurements.active, self.variance + self.jitter, 1)

    def matmul(
        self, right_hand_side: RightHandSide, *, tile_size: int = 128
    ) -> RightHandSide:
        """Apply covariance, noise, and jitter using bounded pairwise tiles.

        Inputs have shape ``(capacity, *columns)`` in canonical order. Inactive
        slots act as independent identity rows, unlike a prepared solve which
        ignores inactive inputs. Supports JIT, vmap, and differentiation.
        """
        right_hand_side = jnp.asarray(right_hand_side)
        covariance_product = apply_covariance(
            self.measurements, self.kernel, right_hand_side, tile_size=tile_size
        )
        shift = self.diagonal_shift().reshape(
            (-1,) + (1,) * (right_hand_side.ndim - 1)
        )
        return covariance_product + shift * right_hand_side

    def diagonal(self) -> Vector:
        """Return the stabilized diagonal, including unit inactive entries."""
        return (
            assemble_covariance_diagonal(self.measurements, self.kernel)
            + self.diagonal_shift()
        )

    def materialize(self, *, tile_size: int | None) -> SquareMatrix:
        covariance = assemble_covariance(
            self.measurements,
            self.measurements,
            self.kernel,
            tile_size=tile_size,
        )
        matrix = covariance + jnp.diag(self.diagonal_shift())
        return symmetrize(matrix)


class CovarianceSystemUpdate(eqx.Module, LinearSystemUpdate):
    """Activation of reserved scalar slots, with compact dense update blocks."""

    system: CovarianceSystem
    previous: PreparedFunctionals
    appended: PreparedFunctionals
    indices: jax.Array

    @property
    def previous_capacity(self) -> int:
        return self.previous.size

    def trailing_blocks(
        self, *, tile_size: int | None
    ) -> tuple[CrossMatrix, AppendedMatrix]:
        cross = assemble_covariance(
            self.previous,
            self.appended,
            self.system.kernel,
            tile_size=tile_size,
        )
        appended = assemble_covariance(
            self.appended,
            self.appended,
            self.system.kernel,
            tile_size=tile_size,
        )
        shift = jnp.where(
            self.appended.active,
            self.system.variance[self.indices] + self.system.jitter,
            1,
        )
        appended = appended + jnp.diag(shift)
        return cross, symmetrize(appended)
