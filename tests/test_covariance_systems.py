"""Matrix-free covariance agrees with the dense system and bounds storage."""

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpex.blocks import FunctionalBlock, Locations, ObservationDesign
from gpex.blocks import ObservationDesignBlock as DesignBlock
from gpex.covariance_systems import CovarianceSystem
from gpex.kernels import Matern12Kernel, RBFKernel
from gpex.moments.measurements import prepare_functionals
from gpex.operators import BoxAverage, Gradient, Identity


def _mixed_system(points, scale, radius, variance, count):
    # Repeated identities straddle a vector-output group, so canonical order
    # differs from group order. Padding and partial tiles are both exercised.
    design = ObservationDesign(
        [
            DesignBlock(
                FunctionalBlock(Locations(points[:3]), Identity(), "values"),
                variance=variance,
                max_size=4,
                count=count,
            ),
            DesignBlock(
                FunctionalBlock(Locations(points[3:5]), Gradient(), "gradients"),
                variance=2 * variance,
                max_size=3,
            ),
            DesignBlock(
                FunctionalBlock(Locations(points[5:7]), Identity(), "more_values"),
                variance=3 * variance,
                max_size=2,
            ),
            DesignBlock(
                FunctionalBlock(
                    Locations(points[7:]),
                    BoxAverage(radius, quadrature_order=2),
                    "average",
                ),
                variance=4 * variance,
                max_size=1,
            ),
        ]
    )
    return CovarianceSystem(
        prepare_functionals(design.blocks),
        RBFKernel(lengthscale=scale),
        design.variance_flat,
        0.02,
    )


def _inputs():
    points = jnp.sin(jnp.arange(16.0).reshape(8, 2))
    return points, jnp.array(0.8), jnp.array(0.15), jnp.array(0.1)


@pytest.mark.parametrize("compiled", [False, True])
def test_operator_views_match_dense_with_grouping_padding_and_rhs_layouts(compiled):
    system = _mixed_system(*_inputs(), jnp.array(2))
    matrix = system.materialize(tile_size=None)
    tolerance = 3e-5 if matrix.dtype == jnp.float32 else 2e-12

    def evaluate(system, rhs, tile_size):
        return system.matmul(rhs, tile_size=tile_size), system.diagonal()

    operation = eqx.filter_jit(evaluate) if compiled else evaluate
    # A tile smaller than one gradient output must still include that output.
    # The other cases cover a partial last tile and the single-tile path.
    for shape, tile in (((13,), 1), ((13, 2, 2), 3), ((13, 0), 32)):
        rhs = jnp.cos(jnp.arange(np.prod(shape), dtype=float)).reshape(shape)
        product, diagonal = operation(system, rhs, tile)
        expected = (matrix @ rhs.reshape(13, -1)).reshape(shape)
        np.testing.assert_allclose(product, expected, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(
            diagonal, jnp.diag(matrix), rtol=tolerance, atol=tolerance
        )
        np.testing.assert_array_equal(
            product[~system.measurements.active], rhs[~system.measurements.active]
        )


def test_operator_values_and_derivatives_match_dense_under_jit_and_vmap():
    rhs = jnp.cos(jnp.arange(26.0).reshape(13, 2))

    def values(points, scale, radius, variance, rhs, count, *, dense):
        system = _mixed_system(points, scale, radius, variance, count)
        if dense:
            matrix = system.materialize(tile_size=None)
            return matrix @ rhs, jnp.diag(matrix)
        return system.matmul(rhs, tile_size=3), system.diagonal()

    def loss(*args, dense):
        product, diagonal = values(*args, dense=dense)
        return jnp.sum(jnp.sin(product)) + jnp.sum(diagonal**2)

    args = (*_inputs(), rhs, jnp.array(2))
    tangent = tuple(jnp.full_like(value, 0.03) for value in args[:-1])
    tolerance = 2e-4 if rhs.dtype == jnp.float32 else 2e-10
    results = []
    for dense in (False, True):

        def operation(*x, dense=dense):
            return loss(*x, dense=dense)

        reverse = jax.jit(jax.value_and_grad(operation, argnums=(0, 1, 2, 3, 4)))(*args)
        forward = jax.jit(
            lambda *x, dense=dense: jax.jvp(
                partial(values, count=args[-1], dense=dense), x, tangent
            )
        )(*args[:-1])
        # Counts change inside vmap, including an entirely inactive first block.
        batched = jax.jit(
            jax.vmap(lambda count, dense=dense: values(*args[:-1], count, dense=dense))
        )(jnp.array([0, 3]))
        results.append((reverse, forward, batched))
    for actual, expected in zip(
        jax.tree.leaves(results[0]), jax.tree.leaves(results[1]), strict=True
    ):
        np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


def test_operator_rejects_invalid_shapes_tiles_and_kernel_smoothness():
    system = _mixed_system(*_inputs(), jnp.array(2))
    for rhs in (jnp.array(1.0), jnp.ones(12)):
        with pytest.raises(ValueError, match="right-hand side"):
            system.matmul(rhs)
    for tile_size in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="tile_size"):
            system.matmul(jnp.ones(13), tile_size=tile_size)  # type: ignore[arg-type]
    rough = eqx.tree_at(lambda s: s.kernel, system, Matern12Kernel())
    for operation in (rough.diagonal, lambda: rough.matmul(jnp.ones(13))):
        with pytest.raises(ValueError, match="derivative order"):
            operation()


def test_inactive_nonfinite_rhs_does_not_contaminate_active_operator_rows():
    system = _mixed_system(*_inputs(), jnp.array(0))
    active = system.measurements.active
    rhs = jnp.where(active, 1.0, jnp.nan)
    result = jax.jit(lambda s, b: s.matmul(b, tile_size=3))(system, rhs)
    reference = system.materialize(tile_size=None) @ jnp.where(active, rhs, 0)
    np.testing.assert_allclose(result[active], reference[active], rtol=3e-5, atol=3e-5)
    assert jnp.isnan(result[~active]).all()
    bad = jnp.where(active, jnp.nan, 0.0)
    assert jnp.isnan(system.matmul(bad, tile_size=3)[active]).any()


@pytest.mark.parametrize("differentiate", [False, True])
def test_tiled_operator_storage_grows_linearly_in_forward_and_reverse(differentiate):
    if jax.default_backend() != "cpu":
        pytest.skip("CPU compiler memory regression check")

    def multiply(points, rhs, scale):
        block = FunctionalBlock(Locations(points), Identity(), "values")
        system = CovarianceSystem(
            prepare_functionals((block,)),
            RBFKernel(lengthscale=scale),
            jnp.full(points.shape[0], 0.1),
            1e-6,
        )
        return system.matmul(rhs, tile_size=16)

    operation = multiply
    if differentiate:
        operation = jax.grad(
            lambda x, b, s: jnp.sum(multiply(x, b, s) ** 2), argnums=(0, 1, 2)
        )
    storage = []
    # Fourfold size growth distinguishes linear workspace from a saved tape
    # for every tile pair. Tiny systems are dominated by fixed loop buffers.
    for size in (256, 1024):
        points = jnp.linspace(-2, 2, size)[:, None]
        executable = (
            jax.jit(operation)
            .lower(points, jnp.cos(points[:, 0]), jnp.array(0.8))
            .compile()
        )
        stats = executable.memory_analysis()
        assert stats is not None
        used = stats.temp_size_in_bytes + stats.output_size_in_bytes
        assert used < size * size * points.dtype.itemsize
        storage.append(used)
    assert storage[1] < 6 * storage[0]
