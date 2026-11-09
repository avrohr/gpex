"""Linear operator interface and weighted combinations of operators."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

Point = Float[Array, "input_dimension"]
FunctionalValue = Float[Array, "*output_shape"]
PointFunction = Callable[[Point], FunctionalValue]
OperatorCoefficients = Float[Array, "operators"]


class LinearOperator(eqx.Module):
    """Base class for linear measurements and query operators.

    Subclasses declare the result produced at one input location with
    [`output_shape`][gpex.operators.LinearOperator.output_shape] and
    implement the linear action with [`apply`][gpex.operators.LinearOperator.apply].
    ``apply`` must remain linear in ``function`` and use JAX-compatible
    operations.
    """

    @property
    def derivative_order(self) -> int:
        """Required derivative order of the latent function.

        The default, zero, is for operators using only function values. Custom
        differential operators must override this so covariance compilation can
        check the kernel's smoothness before applying automatic differentiation.
        """
        return 0

    @abstractmethod
    def output_shape(self, input_dimension: int) -> tuple[int, ...]:
        """Return the operator output shape at one input location."""

    @abstractmethod
    def apply(self, function: PointFunction, point: Point) -> FunctionalValue:
        """Apply the operator to ``function`` at ``point``."""


class LinearCombination(LinearOperator):
    """A weighted sum of operators with a common output shape."""

    operators: tuple[LinearOperator, ...]
    coefficients: OperatorCoefficients

    def __init__(
        self,
        operators: tuple[LinearOperator, ...] | list[LinearOperator],
        coefficients: OperatorCoefficients,
    ) -> None:
        operators = tuple(operators)
        coefficients = jnp.asarray(coefficients)
        if not operators:
            raise ValueError("at least one operator is required")
        if coefficients.shape != (len(operators),):
            raise ValueError("coefficients must contain one value per operator")
        self.operators = operators
        self.coefficients = coefficients

    @property
    def derivative_order(self) -> int:
        return max(operator.derivative_order for operator in self.operators)

    def output_shape(self, input_dimension: int) -> tuple[int, ...]:
        shapes = [operator.output_shape(input_dimension) for operator in self.operators]
        if any(shape != shapes[0] for shape in shapes[1:]):
            raise ValueError("combined operators must have the same output shape")
        return shapes[0]

    def apply(self, function: PointFunction, point: Point) -> FunctionalValue:
        values = jnp.stack(
            [operator.apply(function, point) for operator in self.operators]
        )
        coefficient_shape = (len(self.operators),) + (1,) * (values.ndim - 1)
        return jnp.sum(values * self.coefficients.reshape(coefficient_shape), axis=0)


__all__ = ["LinearOperator", "LinearCombination"]
