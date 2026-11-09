"""Factored contractions against full quadrature and polynomial moments."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import jaxtyped

from gpex import FunctionalBlock
from gpex.kernels import (
    GeneralizedRBFKernel,
    KernelFactorization,
    Matern52Kernel,
    RBFKernel,
    SeparableKernel,
)
from gpex.moments import assemble_covariance, assemble_mean
from gpex.operators import (
    BoxAverage,
    GaussianSmoothing,
    Gradient,
    Hessian,
    Identity,
    TensorQuadratureOperator,
)
from gpex.parameters import PositiveParameter


def _assert_close(actual, expected):
    tolerance = 2e-11 if jax.config.read("jax_enable_x64") else 2e-5
    for got, want in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_allclose(got, want, rtol=tolerance, atol=tolerance)


class _AffineFactor(eqx.Module):
    shift: jax.Array

    def __call__(self, left, right):
        return 1 + (left + self.shift) * (right + self.shift)


class _PolynomialKernel(SeparableKernel):
    amplitude: jax.Array
    shifts: jax.Array

    def factorization(self, input_dimension):
        if self.shifts.shape != (input_dimension,):
            raise ValueError("shifts must match the input dimension")
        return KernelFactorization(
            self.amplitude, tuple(_AffineFactor(shift) for shift in self.shifts)
        )


class _MovingRule(TensorQuadratureOperator):
    """Unequal coordinate orders, signed weights, and cross-coordinate state."""

    widths: jax.Array

    def output_shape(self, input_dimension):
        if input_dimension != 2:
            raise ValueError("this rule requires two coordinates")
        return ()

    def coordinate_quadrature(self, point):
        x, y = point
        return (
            (
                x + self.widths[0] * jnp.array([-1.0, 1.0]),
                jnp.array([1 + y, -0.2 * x]),
            ),
            (
                y + self.widths[1] * jnp.array([-1.0, 0.0, 2.0]),
                jnp.array([x, 0.5, -0.1 * y]),
            ),
        )


def _features(point, shifts):
    x, y = point + shifts
    return jnp.array([1.0, x, y, x * y])


def _moving_features(point, widths, shifts):
    x, y = point
    mass_x = 1 + y - 0.2 * x
    mass_y = x + 0.5 - 0.1 * y
    first_x = (1 + y) * (x - widths[0] + shifts[0]) - 0.2 * x * (
        x + widths[0] + shifts[0]
    )
    first_y = (
        x * (y - widths[1] + shifts[1])
        + 0.5 * (y + shifts[1])
        - 0.1 * y * (y + 2 * widths[1] + shifts[1])
    )
    return jnp.array(
        [mass_x * mass_y, first_x * mass_y, mass_x * first_y, first_x * first_y]
    )


@pytest.mark.parametrize("compiled", [False, True])
def test_nonstationary_kernel_and_moving_rule_match_independent_moments(compiled):
    points = jnp.array([[-0.4, 0.2], [0.6, -0.1]])
    coefficients = jnp.array([0.2, -0.3, 0.4, 0.5])

    def actual(parameters):
        amplitude, widths, shifts, locations = parameters
        kernel = _PolynomialKernel(amplitude, shifts)
        blocks = (FunctionalBlock(locations, _MovingRule(widths), "integrals"),)
        # Keep the eager derivative check focused on moving nodes and weights;
        # the compiled case also exercises mixed operator contractions.
        if compiled:
            blocks += (
                FunctionalBlock(locations, Identity(), "values"),
                FunctionalBlock(locations, Gradient(), "gradients"),
            )
        return (
            assemble_mean(
                blocks, lambda point: _features(point, shifts) @ coefficients
            ),
            assemble_covariance(blocks, blocks, kernel),
        )

    @jax.jit
    def reference(parameters):
        amplitude, widths, shifts, locations = parameters
        features = jax.vmap(_moving_features, in_axes=(0, None, None))(
            locations, widths, shifts
        )
        if compiled:
            features = jnp.concatenate(
                (
                    features,
                    jax.vmap(_features, in_axes=(0, None))(locations, shifts),
                    jax.vmap(jax.jacfwd(_features), in_axes=(0, None))(
                        locations, shifts
                    )
                    .transpose(0, 2, 1)
                    .reshape(-1, 4),
                )
            )
        return features @ coefficients, amplitude * (features @ features.T)

    parameters = (
        jnp.asarray(1.2),
        jnp.array([0.3, 0.2]),
        jnp.array([0.1, -0.2]),
        points,
    )
    operation = jax.jit(actual) if compiled else actual
    _assert_close(operation(parameters), reference(parameters))
    _assert_close(jax.jacfwd(operation)(parameters), jax.jacfwd(reference)(parameters))

    # Reverse mode must retain dependence on nodes, weights, locations and kernel.
    def objective(p):
        return sum(jnp.sum(value**2) for value in actual(p))

    def expected_objective(p):
        return sum(jnp.sum(value**2) for value in reference(p))

    gradient = jax.jit(jax.grad(objective)) if compiled else jax.grad(objective)
    _assert_close(gradient(parameters), jax.grad(expected_objective)(parameters))


@pytest.mark.parametrize("lengthscale", [0.8, jnp.array([0.8, 1.3])])
def test_rbf_factorization_is_dynamic_and_matches_point_evaluation(lengthscale):
    kernel = RBFKernel(variance=1.4, lengthscale=lengthscale)
    x, y = jnp.array([0.2, -0.4]), jnp.array([-0.1, 0.5])
    representation = eqx.filter_jit(lambda k: k.factorization(2))(kernel)
    value = representation.coefficient
    for factor, left, right in zip(representation.factors, x, y, strict=True):
        value = value * factor(left, right)
    _assert_close(value, kernel(x, y))


@pytest.mark.parametrize("kind", ["rbf", "sum", "mixed_sum", "product", "generalized"])
def test_mixed_builtin_covariances_and_gradients_match_full_grid(kind):
    points = jnp.array([[-0.3, 0.2], [0.4, -0.1]])

    def covariance(parameters, *, factored):
        variance, length, radius, sigma, shift = parameters
        first = RBFKernel(variance=variance, lengthscale=length * jnp.array([1.0, 1.4]))
        second = RBFKernel(variance=0.3, lengthscale=1.2)
        if kind == "sum":
            kernel = first + second
        elif kind == "mixed_sum":
            kernel = first + Matern52Kernel(variance=0.3, lengthscale=1.2)
        elif kind == "product":
            kernel = first * second
        elif kind == "generalized":
            kernel = GeneralizedRBFKernel(
                variance=variance, L=jnp.array([[length, 0.0], [0.2, 1.1]])
            )
        else:
            kernel = first
        operators = (
            BoxAverage(radius, quadrature_order=3),
            GaussianSmoothing(sigma, quadrature_order=2),
            Gradient(),
        )
        # The base kernel covers all block pairings. Composite kernels need the
        # tensor/derivative paths, without repeating identity and same-rule pairs.
        if kind == "rbf":
            operators += (BoxAverage(1.3 * radius, quadrature_order=3), Identity())
        blocks = tuple(
            FunctionalBlock(points + shift, operator, str(i))
            for i, operator in enumerate(operators)
        )
        # A plain callable deliberately requests the general nested-apply path.
        evaluate = kernel if factored else lambda x, y: kernel(x, y)
        return assemble_covariance(blocks, blocks, evaluate)

    actual = jax.jit(lambda p: covariance(p, factored=True))
    reference = jax.jit(lambda p: covariance(p, factored=False))
    parameters = jnp.array([1.2, 0.7, 0.2, 0.15, 0.1])
    _assert_close(actual(parameters), reference(parameters))
    _assert_close(
        jax.jit(jax.grad(lambda p: jnp.sum(actual(p) ** 2)))(parameters),
        jax.jit(jax.grad(lambda p: jnp.sum(reference(p) ** 2)))(parameters),
    )


def test_mixed_hessian_preserves_component_axes():
    points = jnp.array([[-0.3, 0.2], [0.4, -0.1]])
    kernel = RBFKernel(lengthscale=jnp.array([0.7, 1.2]))
    blocks = (
        FunctionalBlock(points, BoxAverage(0.2, quadrature_order=3), "box"),
        FunctionalBlock(points, Hessian(), "hessian"),
    )
    actual = jax.jit(lambda: assemble_covariance(blocks, blocks, kernel))()
    expected = jax.jit(
        lambda: assemble_covariance(blocks, blocks, lambda x, y: kernel(x, y))
    )()
    _assert_close(actual, expected)


@pytest.mark.parametrize("compiled", [False, True])
def test_dispatch_does_not_bind_unrelated_output_shapes(compiled):
    @jaxtyped(typechecker=None)
    def covariance(points):
        blocks = (
            FunctionalBlock(points, Identity(), "value"),
            FunctionalBlock(points, Gradient(), "gradient"),
            FunctionalBlock(points, BoxAverage(0.2, quadrature_order=2), "box"),
            FunctionalBlock(
                points, GaussianSmoothing(0.1, quadrature_order=2), "smooth"
            ),
        )
        kernel = RBFKernel() + RBFKernel(lengthscale=0.5)
        return assemble_covariance(blocks, blocks, kernel)

    points = jnp.array([[0.2, -0.1]])
    operation = jax.jit(covariance) if compiled else covariance
    result = operation(points)
    assert result.shape == (5, 5)
    _assert_close(result, result.T)


def test_trainable_kernel_leaves_flow_through_factored_covariance():
    blocks = (FunctionalBlock(jnp.array([[0.2, -0.1]]), BoxAverage(0.3), "box"),)
    kernel = RBFKernel(
        variance=PositiveParameter(1.2),
        lengthscale=PositiveParameter(jnp.array([0.7, 1.3])),
    )

    def objective(model):
        return assemble_covariance(blocks, blocks, model)[0, 0]

    actual = eqx.filter_jit(eqx.filter_value_and_grad(objective))(kernel)
    expected = eqx.filter_jit(
        eqx.filter_value_and_grad(lambda k: objective(lambda x, y: k(x, y)))
    )(kernel)
    _assert_close(actual, expected)


def test_high_dimensional_covariance_and_backward_never_expand_cartesian_grid(
    monkeypatch,
):
    def reject_grid(self, point):
        raise AssertionError("the factored covariance must not expand the grid")

    monkeypatch.setattr(TensorQuadratureOperator, "quadrature", reject_grid)
    dimension, order = 8, 8
    points = jnp.linspace(-0.2, 0.3, 2 * dimension).reshape(2, dimension)

    def evaluate(length):
        blocks = (
            FunctionalBlock(points, BoxAverage(0.2, quadrature_order=order), "box"),
        )
        return jnp.sum(
            assemble_covariance(blocks, blocks, RBFKernel(lengthscale=length))
        )

    def reference(length):
        offsets = 0.2 * (2 * (jnp.arange(order) + 0.5) / order - 1)
        differences = (
            points[:, None, :, None, None]
            - points[None, :, :, None, None]
            + offsets[None, None, None, :, None]
            - offsets[None, None, None, None, :]
        )
        factors = jnp.mean(jnp.exp(-0.5 * (differences / length) ** 2), axis=(-2, -1))
        return jnp.sum(jnp.prod(factors, axis=-1))

    _assert_close(
        jax.jit(jax.value_and_grad(evaluate))(jnp.asarray(0.7)),
        jax.jit(jax.value_and_grad(reference))(jnp.asarray(0.7)),
    )
