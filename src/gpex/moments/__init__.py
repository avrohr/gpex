"""Gaussian mean and covariance assembly for linear functionals."""

from .assembly import (
    apply_covariance,
    assemble_covariance,
    assemble_covariance_diagonal,
    assemble_mean,
)

__all__ = [
    "apply_covariance",
    "assemble_mean",
    "assemble_covariance",
    "assemble_covariance_diagonal",
]
