"""Distributions on constrained parameter values."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from math import isfinite
from typing import Protocol

import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array, Float
from numpy.typing import ArrayLike

from ..validation.cholesky import is_cholesky_factor, validate_factor_shape

ParameterValue = Float[Array, "*parameter_shape"]
Scalar = Float[Array, ""]


class Prior(Protocol):
    """A scalar log density evaluated jointly on a parameter's complete value."""

    @abstractmethod
    def log_prob(self, value: ParameterValue) -> Scalar: ...


@dataclass(frozen=True)
class NormalPrior(Prior):
    mean: float = 0.0
    standard_deviation: float = 1.0

    def __post_init__(self) -> None:
        if self.standard_deviation <= 0:
            raise ValueError("standard deviation must be positive")

    def log_prob(self, value: ParameterValue) -> Scalar:
        standardized = (value - self.mean) / self.standard_deviation
        normalization = jnp.log(self.standard_deviation) + 0.5 * jnp.log(2 * jnp.pi)
        return jnp.sum(-0.5 * standardized**2 - normalization)


@dataclass(frozen=True)
class LogNormalPrior(Prior):
    """Independent log-normal densities with shared scalar hyperparameters.

    Args:
        log_mean: Mean of each log-value. Defaults to 0.
        log_standard_deviation: Positive standard deviation of each log-value.
            Defaults to 1. Vector inputs use the same distribution per entry.
    """

    log_mean: float = 0.0
    log_standard_deviation: float = 1.0

    def __post_init__(self) -> None:
        if self.log_standard_deviation <= 0:
            raise ValueError("log standard deviation must be positive")

    def log_prob(self, value: ParameterValue) -> Scalar:
        log_value = jnp.log(value)
        standardized = (log_value - self.log_mean) / self.log_standard_deviation
        normalization = (
            jnp.log(value)
            + jnp.log(self.log_standard_deviation)
            + 0.5 * jnp.log(2 * jnp.pi)
        )
        return jnp.sum(-0.5 * standardized**2 - normalization)


@dataclass(frozen=True, init=False)
class MultivariateLogNormalPrior(Prior):
    """A joint log-normal prior on a positive vector of shape ``(d,)``.

    Args:
        log_mean: Finite mean vector of ``log(value)``, with shape ``(d,)``.
        log_covariance: Finite, symmetric positive-definite covariance of
            ``log(value)``, with shape ``(d, d)``. A diagonal matrix gives
            independent entries; off-diagonal entries allow correlations.

    Construct the prior outside JIT. Inputs may be arrays or Python sequences;
    they are stored as immutable tuples because priors are static metadata on
    a ``Parameter``. The log density includes the log-normal Jacobian and is
    evaluated on positive values, independently of their optimization transform.
    """

    log_mean: tuple[float, ...]
    log_covariance: tuple[tuple[float, ...], ...]
    _scale_tril: tuple[tuple[float, ...], ...] = field(repr=False, compare=False)

    def __init__(
        self,
        log_mean: ArrayLike,
        log_covariance: ArrayLike,
    ) -> None:
        mean = np.asarray(log_mean)
        covariance = np.asarray(log_covariance)
        # Conversion to Python floats below would discard NumPy imaginary parts.
        if np.iscomplexobj(mean) or np.iscomplexobj(covariance):
            raise ValueError("log mean and covariance must be real")
        if mean.ndim != 1 or mean.size == 0 or not np.all(np.isfinite(mean)):
            raise ValueError("log mean must be a nonempty finite vector")
        if covariance.shape != (mean.size, mean.size):
            raise ValueError("log covariance must have shape (d, d) matching log mean")
        if not np.all(np.isfinite(covariance)) or not np.allclose(
            covariance, covariance.T, rtol=1e-7, atol=1e-12
        ):
            raise ValueError("log covariance must be finite and symmetric")
        covariance = 0.5 * (covariance + covariance.T)
        try:
            factor = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as error:
            raise ValueError("log covariance must be positive definite") from error
        object.__setattr__(self, "log_mean", tuple(float(x) for x in mean))
        object.__setattr__(
            self,
            "log_covariance",
            tuple(tuple(float(x) for x in row) for row in covariance),
        )
        object.__setattr__(
            self, "_scale_tril", tuple(tuple(float(x) for x in row) for row in factor)
        )

    def log_prob(self, value: ParameterValue) -> Scalar:
        """Return the scalar joint log density, or ``-inf`` outside support.

        Args:
            value: Positive finite vector with shape ``(d,)``.

        Returns:
            Scalar log density with respect to the positive vector. Entries that are
                nonpositive or nonfinite give a density of ``-inf``.
        """
        value = jnp.asarray(value)
        if value.shape != (len(self.log_mean),):
            raise ValueError("prior value must have shape (d,) matching log mean")
        if jnp.issubdtype(value.dtype, jnp.complexfloating):
            raise ValueError("prior value must be real")
        valid = (value > 0) & jnp.isfinite(value)
        log_value = jnp.log(jnp.where(valid, value, 1.0))
        mean = jnp.asarray(self.log_mean)
        factor = jnp.asarray(self._scale_tril)
        standardized = jsp.linalg.solve_triangular(factor, log_value - mean, lower=True)
        log_density = (
            -0.5 * jnp.sum(standardized**2)
            - jnp.sum(jnp.log(jnp.diag(factor)))
            - 0.5 * value.size * jnp.log(2 * jnp.pi)
            - jnp.sum(log_value)
        )
        return jnp.where(jnp.all(valid), log_density, -jnp.inf)


@dataclass(frozen=True)
class GammaPrior(Prior):
    concentration: float
    rate: float

    def __post_init__(self) -> None:
        if self.concentration <= 0 or self.rate <= 0:
            raise ValueError("concentration and rate must be positive")

    def log_prob(self, value: ParameterValue) -> Scalar:
        elementwise = (
            self.concentration * jnp.log(self.rate)
            - jsp.special.gammaln(self.concentration)
            + (self.concentration - 1) * jnp.log(value)
            - self.rate * value
        )
        return jnp.sum(elementwise)


@dataclass(frozen=True)
class IndependentCholeskyPrior(Prior):
    """Independent log-normal diagonal and normal strict-lower entries.

    Args:
        diagonal_prior: Log-normal distribution shared by the positive diagonal
            entries. Defaults to ``LogNormalPrior()``.
        off_diagonal_prior: Normal distribution shared by the unrestricted
            strict-lower entries. Defaults to ``NormalPrior()``.

    The normalized joint density is with respect to the free entries of ``L``.
    """

    diagonal_prior: LogNormalPrior = LogNormalPrior()
    off_diagonal_prior: NormalPrior = NormalPrior()

    def log_prob(self, value: ParameterValue) -> Scalar:
        """Return the joint log density, or ``-inf`` outside Cholesky support."""
        value = validate_factor_shape(value)
        valid = is_cholesky_factor(value)
        matrix = jnp.where(valid, value, jnp.eye(value.shape[0]))
        density = self.diagonal_prior.log_prob(jnp.diag(matrix))
        density += self.off_diagonal_prior.log_prob(
            matrix[jnp.tril_indices(value.shape[0], k=-1)]
        )
        return jnp.where(valid, density, -jnp.inf)


@dataclass(frozen=True)
class ScaledLKJCholeskyPrior(Prior):
    """Positive row scales and an LKJ correlation factor as a prior on full ``L``.

    Args:
        concentration: Positive LKJ concentration. Defaults to 1 (uniform over
            correlation matrices). Larger values favor weaker correlations.
        scale_prior: Joint prior on the positive row-norm vector of shape ``(d,)``.
            Defaults to independent ``LogNormalPrior()`` densities. Custom priors
            must be supported on positive vectors.

    Decompose ``L = diag(scales) @ C``, where each row of ``C`` has unit norm.
    The normalized density includes the Jacobian from ``(scales, C)`` to the
    free entries of ``L``. For generalized kernels this describes the precision
    matrix ``L @ L.T``, not a covariance or lengthscale matrix.
    """

    concentration: float = 1.0
    scale_prior: Prior = LogNormalPrior()

    def __post_init__(self) -> None:
        if not isfinite(self.concentration) or self.concentration <= 0:
            raise ValueError("LKJ concentration must be finite and positive")

    def log_prob(self, value: ParameterValue) -> Scalar:
        """Return the joint log density, or ``-inf`` outside Cholesky support."""
        value = validate_factor_shape(value)
        valid = is_cholesky_factor(value)
        dimension = value.shape[0]
        matrix = jnp.where(valid, value, jnp.eye(dimension))
        # Scale before squaring to avoid overflow for finite large factors.
        row_max = jnp.max(jnp.abs(matrix), axis=1)
        scales = row_max * jnp.sqrt(jnp.sum((matrix / row_max[:, None]) ** 2, axis=1))
        log_scales = jnp.log(scales)
        log_diagonal = jnp.log(jnp.diag(matrix)) - log_scales

        # Row i (zero-based) has i free correlation coordinates in a unit ball.
        # Integrating its density gives pi**(i/2) * Gamma(a) / Gamma(a+i/2).
        rows = jnp.arange(1.0, dimension)
        alpha = self.concentration + (dimension - 1 - rows) / 2
        log_normalizer = jnp.sum(
            rows / 2 * jnp.log(jnp.pi)
            + jsp.special.gammaln(alpha)
            - jsp.special.gammaln(alpha + rows / 2)
        )
        lkj_density = jnp.sum((2 * alpha - 2) * log_diagonal[1:]) - log_normalizer
        # |d L_i / d (scale_i, C_i)| = scale_i**i / C_ii.
        log_jacobian = jnp.sum(rows * log_scales[1:] - log_diagonal[1:])
        density = self.scale_prior.log_prob(scales) + lkj_density - log_jacobian
        return jnp.where(valid, density, -jnp.inf)


__all__ = [
    "Prior",
    "NormalPrior",
    "LogNormalPrior",
    "MultivariateLogNormalPrior",
    "GammaPrior",
    "IndependentCholeskyPrior",
    "ScaledLKJCholeskyPrior",
]
