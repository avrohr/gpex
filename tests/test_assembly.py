import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpex import FunctionalBlock, FunctionalQuery
from gpex.blocks import Locations
from gpex.kernels import RBFKernel
from gpex.moments import assemble_covariance, assemble_mean
from gpex.operators import (
    BoxAverage,
    CumulativeIntegral1D,
    DirectionalDerivative,
    GaussianSmoothing,
    Gradient,
    HaarWavelet1D,
    Hessian,
    Identity,
    Increment,
    LinearCombination,
    PartialDerivative,
)


@pytest.mark.parametrize("compiled", [False, True])
def test_covariance_entry_points_reject_dimensions_before_evaluating_kernel(compiled):
    left = FunctionalBlock(Locations(jnp.zeros((2, 1))), Identity(), "left")
    right = FunctionalBlock(Locations(jnp.zeros((2, 2))), Identity(), "right")

    def kernel(x, y):
        pytest.fail("incompatible blocks reached kernel evaluation")

    operations = (
        lambda: assemble_covariance((left,), (right,), kernel),
        lambda: assemble_covariance((left,), (right,), kernel, tile_size=1),
    )
    for operation in operations:
        evaluate = jax.jit(operation) if compiled else operation
        with pytest.raises(ValueError, match="same input dimension"):
            evaluate()


def test_tiled_covariance_matches_dense_mixed_blocks() -> None:
    left_locations = Locations(
        jnp.array(
            [
                [-0.8, 0.2],
                [-0.3, -0.4],
                [0.1, 0.7],
                [0.6, -0.2],
                [0.9, 0.5],
            ]
        )
    )
    right_locations = Locations(jnp.array([[-0.6, -0.1], [0.2, 0.4], [0.8, -0.5]]))
    left = (
        FunctionalBlock(left_locations, Identity(), "left_values"),
        FunctionalBlock(left_locations, Gradient(), "left_gradients"),
    )
    right = (
        FunctionalBlock(right_locations, Gradient(), "right_gradients"),
        FunctionalBlock(right_locations, Identity(), "right_values"),
    )
    kernel = RBFKernel(variance=1.3, lengthscale=jnp.array([0.7, 1.1]))

    dense = assemble_covariance(left, right, kernel, tile_size=None)
    tiled = assemble_covariance(left, right, kernel, tile_size=3)

    assert tiled.shape == dense.shape
    assert jnp.allclose(tiled, dense, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("size", [2048, 4096, 8192])
def test_mixed_covariance_cuda_fusion(size, float_info) -> None:
    if jax.devices()[0].platform != "gpu":
        pytest.skip("regression for CUDA covariance fusion")

    # These shapes exposed illegal addresses on H100 with NVIDIA 615.71.09
    # and JAX/jaxlib 0.10.2. Small matrices do not exercise the failing fusions.
    points = jnp.asarray(np.linspace(-4, 4, size)[:, None])
    indices = np.array_split(np.arange(size), 3)
    blocks = tuple(
        FunctionalBlock(Locations(points[index]), operator, name)
        for index, operator, name in zip(
            indices,
            (Identity(), Gradient(), BoxAverage(0.15)),
            ("values", "slopes", "averages"),
            strict=True,
        )
    )
    kernel = RBFKernel(variance=1.0, lengthscale=0.8)
    covariance = np.asarray(jax.jit(assemble_covariance)(blocks, blocks, kernel))
    assert np.isfinite(covariance).all()

    # Check representative entries against small, independently evaluated
    # block pairs, which do not share the large assembly's fusion graph.
    selected = [index[[0, len(index) // 2, -1]] for index in indices]
    small_blocks = tuple(
        FunctionalBlock(Locations(points[index]), block.operator, block.name)
        for index, block in zip(selected, blocks, strict=True)
    )
    reference = np.block(
        [
            [
                np.asarray(assemble_covariance((left,), (right,), kernel))
                for right in small_blocks
            ]
            for left in small_blocks
        ]
    )
    flat_indices = np.concatenate(selected)
    np.testing.assert_allclose(
        covariance[np.ix_(flat_indices, flat_indices)],
        reference,
        rtol=64 * float_info.eps,
        atol=64 * float_info.eps,
    )


@pytest.mark.parametrize("mixed_blocks", [False, True])
def test_tiled_covariance_is_jittable_and_differentiable(mixed_blocks) -> None:
    points = jnp.array(
        [
            [-0.8, 0.2],
            [-0.3, -0.4],
            [0.1, 0.7],
            [0.6, -0.2],
            [0.9, 0.5],
        ]
    )

    def objective(
        lengthscale: jax.Array,
        locations: jax.Array,
        tile_size: int | None,
    ) -> jax.Array:
        block = FunctionalBlock(Locations(locations), Gradient(), "gradients")
        blocks = (block,)
        if mixed_blocks:
            blocks = (
                FunctionalBlock(Locations(locations), Identity(), "values"),
                block,
            )

        def kernel(left: jax.Array, right: jax.Array) -> jax.Array:
            displacement = (left - right) / lengthscale
            return jnp.exp(-0.5 * jnp.sum(displacement**2))

        covariance = assemble_covariance(
            blocks,
            blocks,
            kernel,
            tile_size=tile_size,
        )
        return jnp.sum(covariance**2)

    dense = jax.value_and_grad(objective, argnums=(0, 1))(
        jnp.asarray(0.8),
        points,
        None,
    )
    tiled = jax.jit(
        jax.value_and_grad(
            lambda lengthscale, locations: objective(lengthscale, locations, 3),
            argnums=(0, 1),
        )
    )(jnp.asarray(0.8), points)

    assert jnp.allclose(tiled[0], dense[0], rtol=3e-5, atol=3e-5)
    assert jnp.allclose(tiled[1][0], dense[1][0], rtol=3e-5, atol=3e-5)
    assert jnp.allclose(tiled[1][1], dense[1][1], rtol=3e-5, atol=3e-5)

    batched = jax.jit(
        jax.vmap(
            jax.value_and_grad(lambda lengthscale: objective(lengthscale, points, None))
        )
    )(jnp.array([0.8, 1.1]))
    assert jnp.allclose(batched[0][0], dense[0], rtol=3e-5, atol=3e-5)
    assert jnp.allclose(batched[1][0], dense[1][0], rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize("tile_size", [0, -1, 1.5, True])
def test_covariance_tile_size_must_be_positive_integer(tile_size: object) -> None:
    block = FunctionalBlock(Locations(jnp.zeros((2, 1))), Identity(), "values")
    with pytest.raises(ValueError, match="tile_size"):
        assemble_covariance(
            (block,),
            (block,),
            RBFKernel(),
            tile_size=tile_size,  # type: ignore[arg-type]
        )


def test_rbf_value_gradient_covariances_match_analytic_derivatives() -> None:
    left_points = jnp.array([[0.2, -0.1], [0.7, 0.4]])
    right_points = jnp.array([[-0.3, 0.5]])
    left = FunctionalBlock(Locations(left_points), Gradient(), "left")
    right = FunctionalBlock(Locations(right_points), Identity(), "right")
    kernel = RBFKernel(variance=1.7, lengthscale=jnp.array([0.8, 1.2]))

    actual = assemble_covariance((left,), (right,), kernel).reshape(2, 2)
    displacement = left_points - right_points[0]
    values = jnp.array([kernel(point, right_points[0]) for point in left_points])
    expected = -displacement / kernel.lengthscales.lengthscale**2 * values[:, None]

    assert jnp.allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_mixed_functional_covariance_is_symmetric_and_psd() -> None:
    locations = Locations(jnp.array([[-0.7], [0.1], [0.8]]))
    query = FunctionalQuery(
        [
            FunctionalBlock(locations, Identity(), "values"),
            FunctionalBlock(locations, Gradient(), "gradients"),
            FunctionalBlock(locations, Hessian(), "hessians"),
        ]
    )
    covariance = assemble_covariance(
        query.blocks, query.blocks, RBFKernel(lengthscale=0.9)
    )

    assert jnp.allclose(covariance, covariance.T, rtol=2e-5, atol=2e-5)
    assert jnp.linalg.eigvalsh(covariance).min() > -2e-4


def test_directional_derivative_and_linear_combination() -> None:
    point = Locations(jnp.array([[0.4, -0.2]]))
    direction = jnp.array([2.0, -0.5])

    def mean(x: jnp.ndarray) -> jnp.ndarray:
        return x[0] ** 2 + 3 * x[1]

    directional = FunctionalBlock(
        point, DirectionalDerivative(direction), "directional"
    )
    combination = FunctionalBlock(
        point,
        LinearCombination(
            [PartialDerivative(0), PartialDerivative(1)],
            jnp.array([2.0, -0.5]),
        ),
        "combination",
    )

    directional_value = assemble_mean((directional,), mean)
    combination_value = assemble_mean((combination,), mean)
    expected = 2.0 * (2 * 0.4) - 0.5 * 3.0

    assert jnp.allclose(directional_value, expected)
    assert jnp.allclose(combination_value, expected)


def test_increment_matches_translated_difference() -> None:
    locations = Locations(jnp.array([[0.4, -0.2], [-0.1, 0.6]]))
    displacement = jnp.array([0.25, -0.4])
    block = FunctionalBlock(locations, Increment(displacement), "increments")

    def quadratic_mean(x: jnp.ndarray) -> jnp.ndarray:
        return 1.3 + x[0] ** 2 + 2.0 * x[1] ** 2

    actual = assemble_mean((block,), quadratic_mean)
    expected = jnp.array(
        [
            quadratic_mean(point + displacement) - quadratic_mean(point)
            for point in locations.points
        ]
    )

    assert jnp.allclose(actual, expected)


def test_cumulative_integral_matches_polynomial_antiderivative() -> None:
    base = 0.3
    locations = Locations(jnp.array([[-0.8], [base], [1.1]]))
    block = FunctionalBlock(
        locations,
        CumulativeIntegral1D(base, quadrature_order=3),
        "integrals",
    )

    def polynomial(x: jnp.ndarray) -> jnp.ndarray:
        return 1.0 + 2.0 * x[0] + 3.0 * x[0] ** 2

    def antiderivative(x: jnp.ndarray) -> jnp.ndarray:
        return x + x**2 + x**3

    actual = assemble_mean((block,), polynomial)
    expected = antiderivative(locations.points[:, 0]) - antiderivative(
        jnp.asarray(base)
    )

    assert jnp.allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_increment_covariance_matches_kernel_differences() -> None:
    points = jnp.array([[-0.7], [0.1]])
    displacement = jnp.array([0.2])
    blocks = (
        FunctionalBlock(points, Identity(), "values"),
        FunctionalBlock(points, Increment(displacement), "increments"),
    )
    actual = assemble_covariance(blocks, blocks, RBFKernel(lengthscale=0.8))

    def rbf(left, right):
        return np.exp(-0.5 * ((left - right.T) / 0.8) ** 2)

    x = np.asarray(points)
    shifted = np.asarray(points + displacement)
    base = rbf(x, x)
    left = rbf(shifted, x)
    right = rbf(x, shifted)
    both = rbf(shifted, shifted)
    expected = np.block(
        [[base, right - base], [left - base, both - left - right + base]]
    )
    tolerance = 1e-12 if jax.config.read("jax_enable_x64") else 2e-6
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


def test_increment_and_cumulative_integral_validate_arguments() -> None:
    with pytest.raises(ValueError, match="non-empty one-dimensional"):
        Increment(jnp.asarray(0.2))
    with pytest.raises(ValueError, match="displacement shape"):
        FunctionalBlock(
            Locations(jnp.zeros((2, 2))),
            Increment(jnp.ones(3)),
            "increments",
        )
    with pytest.raises(ValueError, match="base must be scalar"):
        CumulativeIntegral1D(jnp.ones(2))
    with pytest.raises(ValueError):
        CumulativeIntegral1D(0.0, quadrature_order=0)
    with pytest.raises(ValueError, match="one-dimensional"):
        FunctionalBlock(
            Locations(jnp.zeros((2, 2))),
            CumulativeIntegral1D(0.0),
            "integrals",
        )


def test_box_average_validates_radius_and_quadrature() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        BoxAverage(0.0)
    with pytest.raises(ValueError, match="quadrature order"):
        BoxAverage(0.5, quadrature_order=0)
    with pytest.raises(ValueError, match="radius shape"):
        FunctionalBlock(
            Locations(jnp.zeros((2, 2))),
            BoxAverage(jnp.ones(3)),
            "averages",
        )


def test_gaussian_smoothing_validates_scale_and_quadrature() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        GaussianSmoothing(0.0)
    with pytest.raises(ValueError):
        GaussianSmoothing(0.5, quadrature_order=0)
    with pytest.raises(ValueError, match="scale shape"):
        FunctionalBlock(
            Locations(jnp.zeros((2, 2))),
            GaussianSmoothing(jnp.ones(3)),
            "smoothed",
        )


def test_haar_wavelet_validates_scale_quadrature_and_dimension() -> None:
    with pytest.raises(ValueError, match="scale must be scalar"):
        HaarWavelet1D(jnp.ones(2))
    with pytest.raises(ValueError, match="strictly positive"):
        HaarWavelet1D(0.0)
    with pytest.raises(ValueError, match="quadrature order"):
        HaarWavelet1D(0.5, quadrature_order=0)
    with pytest.raises(ValueError, match="one-dimensional"):
        FunctionalBlock(
            Locations(jnp.zeros((2, 2))),
            HaarWavelet1D(0.5),
            "wavelet",
        )
