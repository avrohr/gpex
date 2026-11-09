"""Numerical operator, solver, and covariance-reduction interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

from jaxtyping import Array, Bool, Float, Int

Activity = Bool[Array, "capacity"]
Indices = Int[Array, "indices"]
Count = Int[Array, ""]

SquareMatrix = Float[Array, "capacity capacity"]
RightHandSide = Float[Array, "capacity *columns"]
Scalar = Float[Array, ""]
Vector = Float[Array, "capacity"]
Matrix = Float[Array, "capacity columns"]
GramMatrix = Float[Array, "columns columns"]
ColumnValues = Float[Array, "columns"]
CrossMatrix = Float[Array, "capacity appended_size"]
AppendedMatrix = Float[Array, "appended_size appended_size"]


__all__ = [
    "AppendedMatrix",
    "ColumnValues",
    "CrossMatrix",
    "CovarianceProjection",
    "GramMatrix",
    "LinearSystem",
    "LinearSystemUpdate",
    "Matrix",
    "RightHandSide",
    "Scalar",
    "SolverConfiguration",
    "SquareMatrix",
    "SymmetricLinearOperator",
    "Vector",
    "symmetrize",
]


class SolverConfiguration(ABC):
    """Abstract backend preparing a numerical system with optional tiling."""

    @abstractmethod
    def prepare(
        self,
        system: SymmetricLinearOperator,
        *,
        materialization_tile_size: int | None = None,
        active: Activity | None = None,
    ) -> LinearSystem: ...


class LinearSystem(ABC):
    """Abstract prepared system shared by inference and exact fitting.

    Inputs and outputs use canonical observation order. Numerical state must
    remain differentiable and cross JIT boundaries as dynamic PyTree leaves.
    Invalid fitting trials retain non-finite quadratic forms and log
    determinants so optimizers can reject them without changing the system.
    """

    @property
    @abstractmethod
    def size(self) -> Count:
        """Number of active scalar entries, as a dynamic JAX integer."""
        ...

    @property
    @abstractmethod
    def capacity(self) -> int:
        """Static scalar dimension of input and output arrays, including padding."""
        ...

    @abstractmethod
    def solve(
        self,
        right_hand_side: RightHandSide,
    ) -> RightHandSide: ...

    @abstractmethod
    def quadratic_form(self, right_hand_side: Vector) -> Scalar:
        """Return ``rhs.T @ A⁻¹ @ rhs`` without forming the inverse."""
        ...

    @abstractmethod
    def logdet(self) -> Scalar:
        """Return the log determinant of the prepared system."""
        ...

    @abstractmethod
    def covariance_reduction(self, cross: Matrix) -> GramMatrix:
        """Compute ``cross.T @ A⁻¹ @ cross``."""
        ...

    @abstractmethod
    def variance_reduction(self, cross: Matrix) -> ColumnValues:
        """Return only the diagonal of ``cross.T @ A⁻¹ @ cross``."""
        ...

    @abstractmethod
    def project(self, cross: Matrix) -> CovarianceProjection:
        """Prepare reusable factors for consistent joint covariance reductions."""
        ...

    @abstractmethod
    def append(self, update: LinearSystemUpdate) -> LinearSystem: ...

    @abstractmethod
    def append_solve(
        self,
        update: LinearSystemUpdate,
        solution: Vector,
        right_hand_side: Vector,
    ) -> tuple[LinearSystem, Vector]: ...


class SymmetricLinearOperator(ABC):
    """Symmetric numerical input supporting matrix-free and dense operations.

    Backends consume this interface without knowing about GP measurements.
    They choose when to materialize the matrix and how to prepare its state.
    """

    @property
    @abstractmethod
    def capacity(self) -> int:
        """Materialized matrix dimension, including any reserved slots."""
        ...

    @abstractmethod
    def matmul(
        self, right_hand_side: RightHandSide, *, tile_size: int = 128
    ) -> RightHandSide:
        """Apply the system to canonical rows without materializing the matrix.

        The input has shape ``(capacity, *columns)``. Reserved inactive slots
        are independent identity rows, so their input values pass through.
        ``tile_size`` bounds pairwise work, subject to indivisible operator
        outputs. Numerical inputs support JIT, vmap, and differentiation.
        """
        ...

    @abstractmethod
    def diagonal(self) -> Vector:
        """Return canonical diagonal entries without a full matrix."""
        ...

    @abstractmethod
    def materialize(self, *, tile_size: int | None) -> SquareMatrix:
        """Construct the same system explicitly for dense methods."""
        ...


class LinearSystemUpdate(ABC):
    """Activate unused slots without changing the system capacity.

    ``indices`` are distinct canonical slots that were inactive. Cross rows use
    canonical ordering and are zero at previously inactive slots. Existing
    active entries of the system and right-hand side must remain unchanged.
    """

    indices: Indices
    system: SymmetricLinearOperator

    @property
    @abstractmethod
    def previous_capacity(self) -> int:
        """Materialized dimension before the update, including reserved slots."""
        ...

    @abstractmethod
    def trailing_blocks(
        self, *, tile_size: int | None
    ) -> tuple[CrossMatrix, AppendedMatrix]: ...


class CovarianceProjection(ABC):
    """Reusable reductions for targets of one prepared system.

    Implementations own their numerical representation and expose it as dynamic
    PyTree leaves. Cross reductions require projections from the same system.
    """

    @abstractmethod
    def gram(self) -> GramMatrix:
        """Return ``C.T @ A⁻¹ @ C`` for the projected cross-covariance C."""
        ...

    @abstractmethod
    def diagonal(self) -> ColumnValues:
        """Return the reduction diagonal without a query-sized matrix."""
        ...

    @abstractmethod
    def cross_gram(self, other: Self) -> Matrix:
        """Return ``C.T @ A⁻¹ @ D`` for two targets of the same system."""
        ...


def symmetrize(matrix: SquareMatrix) -> SquareMatrix:
    """Remove asymmetry introduced by floating-point covariance arithmetic."""
    return 0.5 * (matrix + matrix.T)
