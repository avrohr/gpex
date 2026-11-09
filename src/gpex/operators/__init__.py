"""Linear operators for observations and posterior queries."""

from .base import LinearCombination, LinearOperator
from .differences import Increment
from .local import (
    DirectionalDerivative,
    Gradient,
    Hessian,
    Identity,
    Laplacian,
    PartialDerivative,
)
from .quadrature import (
    BoxAverage,
    CumulativeIntegral1D,
    GaussianSmoothing,
    HaarWavelet1D,
    QuadratureOperator,
    TensorQuadratureOperator,
)

__all__ = [
    "BoxAverage",
    "CumulativeIntegral1D",
    "DirectionalDerivative",
    "GaussianSmoothing",
    "Gradient",
    "HaarWavelet1D",
    "Hessian",
    "Identity",
    "Increment",
    "Laplacian",
    "LinearCombination",
    "LinearOperator",
    "PartialDerivative",
    "QuadratureOperator",
    "TensorQuadratureOperator",
]
