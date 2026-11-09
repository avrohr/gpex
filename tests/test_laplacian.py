import jax
import jax.numpy as jnp
import pytest

from gpex import FunctionalBlock
from gpex.blocks import Locations
from gpex.kernels import GeneralizedMatern52Kernel, GeneralizedRBFKernel, RBFKernel
from gpex.moments import assemble_covariance
from gpex.operators import Gradient, Hessian, Identity, Laplacian


@pytest.mark.parametrize("dimension", [1, 3])
@pytest.mark.parametrize("compiled", [False, True])
def test_laplacian_preserves_tensor_outputs(dimension, compiled):
    points = jnp.linspace(-0.4, 0.7, 2 * dimension).reshape(2, dimension)
    weights = jnp.linspace(0.6, 1.1, dimension)

    def function(point):
        phase = weights @ point
        return jnp.array(
            [[jnp.sum(weights * point**4), jnp.sin(phase)], [jnp.exp(phase), 2.0]]
        )

    def expected(point):
        phase = weights @ point
        squared_norm = jnp.sum(weights**2)
        return jnp.array(
            [
                [12 * jnp.sum(weights * point**2), -squared_norm * jnp.sin(phase)],
                [squared_norm * jnp.exp(phase), 0.0],
            ]
        )

    evaluate = jax.vmap(lambda point: Laplacian().apply(function, point))
    if compiled:
        evaluate = jax.jit(evaluate)
    actual = evaluate(points)
    tolerance = 1e-12 if jax.config.read("jax_enable_x64") else 5e-7
    assert actual.shape == (2, 2, 2)

    assert jnp.allclose(
        actual, jax.vmap(expected)(points), rtol=tolerance, atol=tolerance
    )


@pytest.mark.parametrize("dimension", [1, 2, 5])
@pytest.mark.parametrize("compiled", [False, True])
def test_laplacian_rbf_covariances_match_analytic_reference(dimension, compiled):
    points = jnp.linspace(-0.6, 0.8, 3 * dimension).reshape(3, dimension)
    lengthscale = jnp.linspace(0.7, 1.2, dimension)
    variance = 1.3
    model = RBFKernel(variance=variance, lengthscale=lengthscale)

    def covariance(coordinates):
        locations = Locations(coordinates)
        left = (
            FunctionalBlock(locations, Identity(), "values"),
            FunctionalBlock(locations, Laplacian(), "laplacians"),
        )
        right = (FunctionalBlock(locations, Laplacian(), "laplacians"),)
        return assemble_covariance(left, right, model)

    evaluate = jax.jit(covariance) if compiled else covariance
    actual = evaluate(points)
    displacement = points[:, None, :] - points[None, :, :]
    precision = lengthscale**-2
    kernel = variance * jnp.exp(-0.5 * jnp.sum(displacement**2 * precision, axis=-1))
    trace_factor = jnp.sum(displacement**2 * precision**2, axis=-1) - jnp.sum(precision)
    laplacian = kernel * trace_factor
    bilaplacian = kernel * (
        trace_factor**2
        - 4 * jnp.sum(displacement**2 * precision**3, axis=-1)
        + 2 * jnp.sum(precision**2)
    )
    expected = jnp.concatenate((laplacian, bilaplacian))

    tolerance = 1e-11 if jax.config.read("jax_enable_x64") else 5e-6

    assert jnp.allclose(actual, expected, rtol=tolerance, atol=tolerance)


class _HessianTraceLaplacian(Laplacian):
    def apply(self, function, point):
        return jnp.trace(Hessian().apply(function, point), axis1=-2, axis2=-1)


@pytest.mark.parametrize(
    ("kernel_type", "coefficient"),
    [(GeneralizedRBFKernel, 1.0), (GeneralizedMatern52Kernel, 25.0 / 3.0)],
)
def test_laplacian_with_full_metric_and_coincident_points(kernel_type, coefficient):
    factor = jnp.array([[1.2, 0.0], [0.4, 0.7]])
    kernel = kernel_type(variance=1.7, L=factor)
    points = jnp.array([[0.0, 0.0], [1e-7, 0.0], [0.3, -0.1]])

    def covariance(coordinates, operator):
        blocks = (FunctionalBlock(Locations(coordinates), operator, "laplacians"),)
        return assemble_covariance(blocks, blocks, kernel)

    actual = jax.jit(covariance)(points, Laplacian())
    reference = jax.jit(covariance)(points, _HessianTraceLaplacian())
    metric = factor @ factor.T
    expected_variance = (
        1.7 * coefficient * (jnp.trace(metric) ** 2 + 2 * jnp.sum(metric**2))
    )
    tolerance = 1e-11 if jax.config.read("jax_enable_x64") else 2e-6
    assert jnp.allclose(actual, reference, rtol=tolerance, atol=tolerance)
    assert jnp.allclose(
        jnp.diag(actual), expected_variance, rtol=tolerance, atol=tolerance
    )


@pytest.mark.parametrize("operation", ["dense", "tiled"])
def test_mixed_laplacian_products_and_gradients_match_hessian_trace(operation):
    points = jnp.array([[-0.6, 0.3], [0.1, -0.4], [0.8, 0.5]])
    parameters = jnp.log(jnp.array([1.3, 0.7, 1.2]))
    rhs = jnp.linspace(-0.7, 0.9, 6 * 2).reshape(6, 2)

    def objective(log_parameters, coordinates, matrix, *, reference):
        locations = Locations(coordinates)
        laplacian = _HessianTraceLaplacian() if reference else Laplacian()
        left = (
            FunctionalBlock(locations, Gradient(), "gradients"),
            FunctionalBlock(locations, Hessian(), "hessians"),
            FunctionalBlock(locations, laplacian, "laplacians"),
        )
        right = (
            FunctionalBlock(locations, Identity(), "values"),
            FunctionalBlock(locations, laplacian, "laplacians"),
        )

        def kernel(x, y):
            displacement = (x - y) / jnp.exp(log_parameters[1:])
            return jnp.exp(log_parameters[0] - 0.5 * jnp.sum(displacement**2))

        tile_size = 4 if operation == "tiled" and not reference else None
        covariance = assemble_covariance(left, right, kernel, tile_size=tile_size)
        product = covariance @ matrix
        return jnp.mean(jnp.sin(product)), product

    def evaluate(reference):
        return jax.jit(
            jax.value_and_grad(
                lambda *args: objective(*args, reference=reference),
                argnums=(0, 1, 2),
                has_aux=True,
            )
        )(parameters, points, rhs)

    actual, expected = evaluate(False), evaluate(True)
    tolerance = 2e-10 if jax.config.read("jax_enable_x64") else 2e-5
    for result, reference in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        assert jnp.allclose(result, reference, rtol=tolerance, atol=tolerance)
