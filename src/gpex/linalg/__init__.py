"""Linear-system backends used by Gaussian-process inference."""

from .base import (
    CovarianceProjection,
    LinearSystem,
    LinearSystemUpdate,
    SolverConfiguration,
    SymmetricLinearOperator,
)
from .dense import CholeskyProjection, CholeskySystem, DenseBackend

__all__ = [
    "CholeskyProjection",
    "CholeskySystem",
    "CovarianceProjection",
    "DenseBackend",
    "LinearSystem",
    "LinearSystemUpdate",
    "SolverConfiguration",
    "SymmetricLinearOperator",
]
