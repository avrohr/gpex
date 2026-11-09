"""Dense linear systems using cholops with reserved-slot updates."""

from __future__ import annotations

from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from cholops import PaddedCholesky

from ..validation.checks import validate_jitter
from .base import (
    Activity,
    AppendedMatrix,
    ColumnValues,
    Count,
    CovarianceProjection,
    CrossMatrix,
    GramMatrix,
    LinearSystem,
    LinearSystemUpdate,
    Matrix,
    RightHandSide,
    Scalar,
    SolverConfiguration,
    SquareMatrix,
    SymmetricLinearOperator,
    Vector,
)

__all__ = ["CholeskyProjection", "CholeskySystem", "DenseBackend"]

# Failed Cholesky factorizations may return NaNs for callers to detect. Raising
# explicitly currently adds substantial tracing overhead. Revisit these calls
# when a cheaper way to report failure is available.


class DenseBackend(eqx.Module, SolverConfiguration):
    """Stateless backend preparing dense Cholesky systems.

    Jitter is owned by the input system. Preparation returns a reusable
    :class:`CholeskySystem` containing the factorization and activity state.
    """

    def prepare(
        self,
        system: SymmetricLinearOperator,
        *,
        materialization_tile_size: int | None = None,
        active: Activity | None = None,
    ) -> CholeskySystem:
        """Prepare a factorization from a backend-neutral operator.

        If supplied, ``active`` identifies used rows. Inactive rows must be an
        independent identity system. They become reusable factor storage.
        Numerical consumers ignore inactive input entries. If no activity
        mask is supplied, every row is active.
        """
        matrix = system.materialize(tile_size=materialization_tile_size)
        if active is None:
            active = jnp.ones(system.capacity, dtype=bool)
        factor = PaddedCholesky.from_matrix(matrix, active=active)
        return CholeskySystem(factor, materialization_tile_size)


class CholeskySystem(eqx.Module, LinearSystem):
    """A dense system backed by a cholops fixed-capacity factor.

    cholops owns storage, ordering, inactive-slot masking, and numerical
    updates. Failed factorizations retain NaNs. Numerical operations support
    JIT, vmap, and first-order differentiation.
    """

    factor: PaddedCholesky
    tile_size: int | None = eqx.field(static=True)

    @classmethod
    def from_matrix(cls, matrix: SquareMatrix, jitter: float) -> CholeskySystem:
        """Factorize the symmetric part of a square matrix with diagonal jitter."""
        validate_jitter(jitter)
        matrix = jnp.asarray(matrix)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("matrix must be square")
        stabilized = matrix + jitter * jnp.eye(matrix.shape[0])
        factor = PaddedCholesky.from_matrix(
            stabilized, active=jnp.ones(matrix.shape[0], dtype=bool)
        )
        return cls(factor, None)

    @property
    def size(self) -> Count:
        """Number of active scalar entries, as a dynamic JAX integer."""
        return jnp.sum(self.factor.active, axis=-1)

    @property
    def capacity(self) -> int:
        """Reserved scalar dimension, including inactive slots."""
        return self.factor.active.shape[-1]

    def solve(
        self,
        right_hand_side: RightHandSide,
    ) -> RightHandSide:
        """Solve on active rows for vector or array RHS, including empty batches.

        Inactive input entries are ignored and inactive solution entries are zero.
        """
        right_hand_side = jnp.asarray(right_hand_side)
        if right_hand_side.ndim == 0 or right_hand_side.shape[0] != self.capacity:
            raise ValueError("right-hand side must start with the system capacity")
        shape = right_hand_side.shape
        columns = right_hand_side.reshape((self.capacity, -1))
        return self.factor.solve(columns).reshape(shape)

    def quadratic_form(self, right_hand_side: Vector) -> Scalar:
        """Return the active inverse quadratic form with one triangular solve.

        The vector has shape ``(capacity,)`` in canonical order. Supports JIT,
        vectorization, and differentiation through the factor and vector.
        Inactive entries are ignored. Failed factorizations propagate non-finite
        values on the active system.
        """
        right_hand_side = jnp.asarray(right_hand_side)
        if right_hand_side.shape != (self.capacity,):
            raise ValueError(
                "right-hand side must be a vector with the system capacity"
            )
        return self.factor.quadratic(right_hand_side)

    def logdet(self) -> Scalar:
        """Return the exact log determinant of the stabilized matrix."""
        return self.factor.logdet()

    def covariance_reduction(self, cross: Matrix) -> GramMatrix:
        """Compute the covariance reduction, ignoring inactive cross rows."""
        return self.project(cross).gram()

    def variance_reduction(self, cross: Matrix) -> ColumnValues:
        """Reduce active cross rows without constructing a query-sized matrix."""
        return self.project(cross).diagonal()

    def append(self, update: LinearSystemUpdate) -> CholeskySystem:
        """Activate reserved slots with a Schur-complement factor update."""
        cross, appended_covariance = self._trailing_blocks(update)
        with jax.default_matmul_precision("highest"):
            factor = self.factor.activate(update.indices, cross, appended_covariance)
        return eqx.tree_at(lambda s: s.factor, self, factor)

    def append_solve(
        self,
        update: LinearSystemUpdate,
        solution: Vector,
        right_hand_side: Vector,
    ) -> tuple[CholeskySystem, Vector]:
        """Activate reserved slots and reuse the existing coefficients.

        Inputs and outputs use canonical ordering. Old active RHS entries
        must be unchanged. Inactive inputs are ignored, and returned inactive
        coefficients are zero.
        """
        cross, appended_covariance = self._trailing_blocks(update)
        with jax.default_matmul_precision("highest"):
            factor, solution = self.factor.activate_solve(
                update.indices,
                cross,
                appended_covariance,
                solution=solution,
                rhs=right_hand_side,
            )
        return eqx.tree_at(lambda s: s.factor, self, factor), solution

    def project(self, cross: Matrix) -> CholeskyProjection:
        """Cache a triangular projection, ignoring inactive cross rows."""
        return CholeskyProjection(self.factor.whiten(cross))

    def _trailing_blocks(
        self,
        update: LinearSystemUpdate,
    ) -> tuple[CrossMatrix, AppendedMatrix]:
        if (
            update.previous_capacity != self.capacity
            or update.system.capacity != self.capacity
        ):
            raise ValueError("system update must preserve capacity")
        return update.trailing_blocks(tile_size=self.tile_size)


class CholeskyProjection(eqx.Module, CovarianceProjection):
    """Factors for a symmetric bilinear covariance reduction.

    Dense systems store ``factor = L⁻¹ C``. Cross reductions require
    projections from the same system.
    All factors are dynamic leaves, including across an outer JIT boundary.
    """

    factor: Matrix

    def gram(self) -> GramMatrix:
        """Return the full symmetric covariance reduction."""
        return jnp.matmul(
            self.factor.T, self.factor, precision=jax.lax.Precision.HIGHEST
        )

    def diagonal(self) -> ColumnValues:
        """Return the reduction diagonal without a query-sized matrix."""
        return jnp.einsum(
            "ij,ij->j", self.factor, self.factor, precision=jax.lax.Precision.HIGHEST
        )

    def cross_gram(self, other: Self) -> Matrix:
        """Return the cross reduction for two targets of the same system."""
        return jnp.matmul(
            self.factor.T, other.factor, precision=jax.lax.Precision.HIGHEST
        )
