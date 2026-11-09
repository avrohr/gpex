"""Grouping preserves public order, parameter ownership and numerical operations."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from gpex import (
    FunctionalBlock,
    FunctionalDataset,
    FunctionalQuery,
    GaussianProcess,
    ObservationBlock,
    ObservationDesign,
    ObservationDesignBlock,
)
from gpex.kernels import RBFKernel
from gpex.moments import (
    assemble_covariance,
    assemble_covariance_diagonal,
    assemble_mean,
)
from gpex.moments.measurements import prepare_functionals
from gpex.operators import BoxAverage, Gradient, Identity, LinearOperator
from gpex.parameters import PositiveParameter


def _blocks(parameters):
    points = jnp.array([[-0.8, 0.2], [-0.2, -0.3], [0.1, 0.4], [0.7, -0.2]])
    points = points.at[0, 0].add(parameters[0])
    return (
        FunctionalBlock(points[:2], Identity(), "a"),
        FunctionalBlock(points[:2], BoxAverage(parameters[1], quadrature_order=2), "b"),
        FunctionalBlock(points[1:2], Gradient(), "c"),
        FunctionalBlock(points[2:], Identity(), "d"),
        FunctionalBlock(points[2:], BoxAverage(parameters[2], quadrature_order=2), "e"),
        FunctionalBlock(points[3:], Gradient(), "f"),
    )


def _reference(left, right, kernel):
    """Independent block assembly with one operator shared over each point axis."""
    rows = []
    for a in left:
        row = []
        for b in right:

            def pair(x, y, a=a, b=b):
                return b.operator.apply(
                    lambda v: a.operator.apply(lambda u: kernel(u, v), x), y
                )

            values = jax.vmap(
                lambda x, b=b, pair=pair: jax.vmap(lambda y: pair(x, y))(
                    b.locations.points
                )
            )(a.locations.points)
            left_rank = len(a.shape) - 1
            permutation = (
                (0,)
                + tuple(range(2, 2 + left_rank))
                + (1,)
                + tuple(range(2 + left_rank, values.ndim))
            )
            row.append(values.transpose(permutation).reshape((a.size, b.size)))
        rows.append(jnp.concatenate(row, axis=1))
    return jnp.concatenate(rows)


def _tolerance():
    return 3e-11 if jax.config.read("jax_enable_x64") else 3e-4


def test_grouped_covariance_diagonal_mean_and_gradients():
    def actual(p, tile_size):
        left = prepare_functionals(_blocks(p))
        right = prepare_functionals(_blocks(p)[::-1])
        kernel = RBFKernel(lengthscale=0.8)
        return assemble_covariance(left, right, kernel, tile_size=tile_size)

    @jax.jit
    def reference(p):
        return _reference(_blocks(p), _blocks(p)[::-1], RBFKernel(lengthscale=0.8))

    p = jnp.array([0.07, 0.13, 0.27])
    expected_covariance = reference(p)
    expected = jax.jit(jax.grad(lambda p: jnp.sum(reference(p) ** 2)))(p)
    evaluate = jax.jit(actual, static_argnums=1)
    gradient = jax.jit(
        jax.grad(lambda p, tile: jnp.sum(actual(p, tile) ** 2)), static_argnums=1
    )
    for tile_size in (None, 3):
        np.testing.assert_allclose(
            evaluate(p, tile_size),
            expected_covariance,
            rtol=_tolerance(),
            atol=_tolerance(),
        )
        np.testing.assert_allclose(
            gradient(p, tile_size), expected, rtol=_tolerance(), atol=_tolerance()
        )

    left = prepare_functionals(_blocks(p))
    kernel = RBFKernel(lengthscale=0.8)
    covariance = jax.jit(lambda p: _reference(_blocks(p), _blocks(p), kernel))(p)
    np.testing.assert_allclose(
        assemble_covariance_diagonal(left, kernel),
        jnp.diag(covariance),
        rtol=_tolerance(),
        atol=_tolerance(),
    )

    def mean(x):
        return jnp.sin(x[0]) + x[1] ** 2

    expected_mean = jnp.concatenate(
        [
            jax.vmap(lambda x, b=b: b.operator.apply(mean, x))(
                b.locations.points
            ).reshape(-1)
            for b in _blocks(p)
        ]
    )
    np.testing.assert_allclose(
        jax.jit(lambda m: assemble_mean(m, mean))(left),
        expected_mean,
        rtol=_tolerance(),
        atol=_tolerance(),
    )


def test_incremental_groups_preserve_interleaved_order_under_jit():
    blocks = _blocks(jnp.array([0.07, 0.13, 0.27]))
    designs = [
        ObservationDesignBlock(b, variance=0.1, max_size=b.locations.count)
        for b in blocks
    ]
    design = ObservationDesign(
        [
            b if i == 0 else b.replace(b.functional.locations.points[:0])
            for i, b in enumerate(designs)
        ]
    )
    conditioning = GaussianProcess(RBFKernel()).prepare_conditioning(design)
    append = eqx.filter_jit(lambda g, x, name: g.append(x, block=name))
    for block in blocks[1:]:
        conditioning = append(conditioning, block.locations, block.name)
    packed = conditioning.measurements
    # Every public block is present from the start. Activate them in interleaved
    # group order while preserving the canonical covariance layout.
    expected = _reference(blocks, blocks, RBFKernel())
    actual = jax.jit(lambda m: assemble_covariance(m, m, RBFKernel(), tile_size=3))(
        packed
    )
    np.testing.assert_allclose(actual, expected, rtol=_tolerance(), atol=_tolerance())
    assert all(isinstance(leaf, jax.Array) for leaf in jax.tree.leaves(packed))


def test_grouped_posterior_append_and_independent_variance_gradients():
    gp = GaussianProcess(RBFKernel(lengthscale=0.6))
    points = jnp.linspace(-1, 1, 8)[:, None]
    query = FunctionalQuery(
        [FunctionalBlock(jnp.array([[-0.17], [0.23]]), Identity(), "q")]
    )

    def calculate(variances, append):
        blocks = [
            ObservationBlock.from_arrays(
                points[i : i + 2],
                jnp.sin(points[i : i + 2, 0]),
                variance=PositiveParameter(variances[i // 2]),
                name=str(i),
                max_size=(points[i : i + 2]).shape[0],
            )
            for i in (0, 2, 4, 6)
        ]
        if append:
            empty = [
                b.replace(b.functional.locations.points[:0], b.y[:0])
                for b in blocks[2:]
            ]
            posterior = gp.condition(FunctionalDataset([*blocks[:2], *empty]))
            for b in blocks[2:]:
                posterior = posterior.append(
                    b.functional.locations, b.y, block=b.functional.name
                )
        else:
            posterior = gp.condition(FunctionalDataset(blocks))
        prediction = posterior.predict_marginals(query)
        return jnp.sum(prediction.mean.flat) + jnp.sum(prediction.variance.flat)

    variances = jnp.array([0.03, 0.05, 0.02, 0.07])
    actual = jax.jit(jax.value_and_grad(lambda n: calculate(n, True)))(variances)
    expected = jax.jit(jax.value_and_grad(lambda n: calculate(n, False)))(variances)
    tolerance = 5e-4 if not jax.config.read("jax_enable_x64") else 3e-11
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_allclose(a, b, rtol=tolerance, atol=tolerance)


def test_many_blocks_share_one_numerical_compilation():
    points = jnp.linspace(-1, 1, 1000)[:, None]
    together = prepare_functionals((FunctionalBlock(points, Identity(), "all"),))
    separate = prepare_functionals(
        tuple(
            FunctionalBlock(points[i : i + 1], Identity(), str(i)) for i in range(1000)
        )
    )
    traces = []

    @jax.jit
    def evaluate(measurements):
        traces.append(None)
        return assemble_mean(measurements, lambda x: x[0] ** 2)

    np.testing.assert_array_equal(evaluate(together), evaluate(separate))
    assert len(traces) == 1


def test_grouped_operator_parameters_support_vmap():
    def evaluate(parameters):
        measurements = prepare_functionals(_blocks(parameters))
        return assemble_covariance(measurements, measurements, RBFKernel(), tile_size=3)

    p = jnp.array([[0.07, 0.13, 0.27], [0.03, 0.2, 0.1]])
    actual = jax.jit(jax.vmap(evaluate))(p)
    np.testing.assert_allclose(
        actual[1],
        _reference(_blocks(p[1]), _blocks(p[1]), RBFKernel()),
        rtol=_tolerance(),
        atol=_tolerance(),
    )


def test_custom_operator_static_leaves_survive_preparation_and_jit():
    class ScaledValue(LinearOperator):
        scale: float

        def output_shape(self, input_dimension):
            return ()

        def apply(self, function, point):
            return self.scale * function(point)

    points = jnp.linspace(-1, 1, 6)[:, None]
    blocks = (
        FunctionalBlock(points[:3], ScaledValue(2.0), "a"),
        FunctionalBlock(points[3:5], ScaledValue(3.0), "b"),
        FunctionalBlock(points[5:], ScaledValue(2.0), "c"),
    )
    measurements = prepare_functionals(blocks)
    kernel = RBFKernel()

    @jax.jit
    def evaluate(m):
        return (
            assemble_covariance(m, m, kernel, tile_size=3),
            assemble_covariance_diagonal(m, kernel),
            assemble_mean(m, lambda x: x[0]),
        )

    covariance, diagonal, mean = evaluate(measurements)
    scale = jnp.array([2.0, 2.0, 2.0, 3.0, 3.0, 2.0])
    reference = jax.vmap(lambda x: jax.vmap(lambda y: kernel(x, y))(points))(points)
    reference = reference * scale[:, None] * scale[None, :]
    for actual, expected in (
        (covariance, reference),
        (diagonal, scale**2),
        (mean, points[:, 0] * scale),
    ):
        np.testing.assert_allclose(
            actual, expected, rtol=_tolerance(), atol=_tolerance()
        )
