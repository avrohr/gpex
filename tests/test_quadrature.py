"""Quadrature actions against independent finite polynomial moments."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpex import FunctionalBlock
from gpex.moments import assemble_covariance, assemble_mean
from gpex.operators import (
    BoxAverage,
    CumulativeIntegral1D,
    GaussianSmoothing,
    Gradient,
    HaarWavelet1D,
    Hessian,
    Identity,
    QuadratureOperator,
)


def _tolerance():
    return 3e-11 if jax.config.read("jax_enable_x64") else 2e-5


class _LocationDependentRule(QuadratureOperator):
    shift: jax.Array

    def quadrature(self, point):
        points = jnp.stack((point - self.shift, point + self.shift))
        weights = point[0] * jnp.array([1.0, -2.0])
        return points, weights


@pytest.mark.parametrize("compiled", [False, True])
def test_custom_rule_differentiates_points_and_signed_unnormalized_weights(compiled):
    def evaluate(x, shift):
        operator = _LocationDependentRule(shift)

        def matrix(point):
            t = point[0]
            return jnp.array([[1.0, t], [t, t**2]])

        block = FunctionalBlock(x[None, None], operator, "custom")
        mean = assemble_mean((block,), lambda point: point[0] ** 2)[0]
        covariance = assemble_covariance(
            (block,), (block,), lambda a, b: a[0] ** 2 * b[0] ** 2
        )[0, 0]
        return mean, covariance, operator.apply(matrix, x[None])

    def reference(x, shift):
        constant = -x
        linear = -(x**2) - 3 * x * shift
        quadratic = -(x**3) - 6 * x**2 * shift - x * shift**2
        return (
            quadratic,
            quadratic**2,
            jnp.array([[constant, linear], [linear, quadratic]]),
        )

    operation = jax.jit(evaluate) if compiled else evaluate
    x, shift = jnp.asarray(0.3), jnp.asarray(0.2)
    actual = jax.vmap(operation, in_axes=(0, None))(jnp.array([-0.6, 0.3, 0.9]), shift)
    expected = jax.vmap(reference, in_axes=(0, None))(
        jnp.array([-0.6, 0.3, 0.9]), shift
    )
    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_allclose(got, want, rtol=_tolerance(), atol=_tolerance())

    actual_derivatives = jax.jacfwd(operation, argnums=(0, 1))(x, shift)
    expected_derivatives = jax.jacfwd(reference, argnums=(0, 1))(x, shift)
    for got, want in zip(
        jax.tree.leaves(actual_derivatives),
        jax.tree.leaves(expected_derivatives),
        strict=True,
    ):
        np.testing.assert_allclose(got, want, rtol=_tolerance(), atol=_tolerance())

    reverse = jax.grad(lambda x, shift: evaluate(x, shift)[1], argnums=(0, 1))
    if compiled:
        reverse = jax.jit(reverse)
    expected_reverse = jax.grad(lambda x, shift: reference(x, shift)[1], argnums=(0, 1))
    np.testing.assert_allclose(
        reverse(x, shift),
        expected_reverse(x, shift),
        rtol=_tolerance(),
        atol=_tolerance(),
    )


def _monomials(x):
    return jnp.stack((jnp.ones_like(x), x, x**2, x**3), axis=-1)


@pytest.mark.parametrize("order", [2, 5])
def test_mixed_means_covariances_and_gradients_match_polynomial_moments(order):
    coefficients = jnp.array([0.7, -0.3, 0.5, 0.2])
    variances = jnp.array([0.8, 1.2, 0.4, 0.1])

    def actual(parameters):
        radius, sigma, base, scale, shift = parameters
        points = jnp.array([[-0.4], [0.2], [0.9]]) + shift
        operators = (
            Identity(),
            Gradient(),
            Hessian(),
            BoxAverage(radius, quadrature_order=order),
            GaussianSmoothing(sigma, quadrature_order=order),
            CumulativeIntegral1D(base, quadrature_order=order),
            HaarWavelet1D(scale, quadrature_order=order),
        )
        blocks = tuple(
            FunctionalBlock(points, operator, str(index))
            for index, operator in enumerate(operators)
        )

        def mean(point):
            return jnp.sum(coefficients * _monomials(point[0]))

        def kernel(left, right):
            return jnp.sum(variances * _monomials(left[0]) * _monomials(right[0]))

        return assemble_mean(blocks, mean), assemble_covariance(blocks, blocks, kernel)

    @jax.jit
    def reference(parameters):
        radius, sigma, base, scale, shift = parameters
        x = jnp.array([-0.4, 0.2, 0.9]) + shift
        one, zero = jnp.ones_like(x), jnp.zeros_like(x)
        # Finite midpoint rules have a smaller second moment than the integral.
        box_variance = radius**2 * (1 - 1 / order**2) / 3
        box = jnp.stack(
            (one, x, x**2 + box_variance, x**3 + 3 * x * box_variance), axis=-1
        )
        gaussian = jnp.stack(
            (one, x, x**2 + sigma**2, x**3 + 3 * x * sigma**2), axis=-1
        )
        powers = jnp.arange(1, 5)
        integral = (x[:, None] ** powers - base**powers) / powers
        haar = jnp.stack(
            (
                zero,
                -one * scale**1.5 / 4,
                -x * scale**1.5 / 2,
                -3 * x**2 * scale**1.5 / 4 - scale**3.5 * (2 - 1 / order**2) / 64,
            ),
            axis=-1,
        )
        gradient = jnp.stack((zero, one, 2 * x, 3 * x**2), axis=-1)
        hessian = jnp.stack((zero, zero, 2 * one, 6 * x), axis=-1)
        features = jnp.concatenate(
            (_monomials(x), gradient, hessian, box, gaussian, integral, haar)
        )
        return features @ coefficients, (features * variances) @ features.T

    parameters = jnp.array([0.35, 0.2, -0.15, 0.6, 0.1])
    # Eager quadrature and its derivatives are checked on the focused custom rule.
    evaluate = jax.jit(actual)
    for got, want in zip(evaluate(parameters), reference(parameters), strict=True):
        np.testing.assert_allclose(got, want, rtol=_tolerance(), atol=_tolerance())

    def loss(parameters, operation):
        mean, covariance = operation(parameters)
        return jnp.sum(mean**2) + jnp.sum(covariance**2)

    gradient = jax.jit(jax.grad(lambda p: loss(p, actual)))
    np.testing.assert_allclose(
        gradient(parameters),
        jax.jit(jax.grad(lambda p: loss(p, reference)))(parameters),
        rtol=_tolerance(),
        atol=_tolerance(),
    )


@pytest.mark.parametrize("operator_type", [BoxAverage, GaussianSmoothing])
def test_tensor_rules_preserve_array_outputs_and_coordinate_moments(operator_type):
    def actual(parameters):
        point, widths = parameters[:2], parameters[2:]

        def polynomial(p):
            return jnp.array([[1.0, p[0]], [p[1], p[0] * p[1]], [p[0] ** 2, p[1] ** 2]])

        return operator_type(widths, quadrature_order=3).apply(polynomial, point)

    def reference(parameters):
        x, y, a, b = parameters
        factor = (1 - 1 / 9) / 3 if operator_type is BoxAverage else 1.0
        return jnp.array(
            [[1.0, x], [y, x * y], [x**2 + factor * a**2, y**2 + factor * b**2]]
        )

    parameters = jnp.array([0.3, -0.4, 0.2, 0.7])
    np.testing.assert_allclose(
        jax.jit(actual)(parameters),
        reference(parameters),
        rtol=_tolerance(),
        atol=_tolerance(),
    )
    np.testing.assert_allclose(
        jax.jit(jax.jacfwd(actual))(parameters),
        jax.jacfwd(reference)(parameters),
        rtol=_tolerance(),
        atol=_tolerance(),
    )
