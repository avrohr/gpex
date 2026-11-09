"""Joint prediction and marginal variance prediction."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpex import (
    CovarianceResult,
    FunctionalBlock,
    FunctionalDataset,
    FunctionalQuery,
    GaussianProcess,
    ObservationBlock,
    ObservationDesignBlock,
    StructuredValues,
)
from gpex.blocks import Locations
from gpex.kernels import RBFKernel
from gpex.moments.measurements import prepare_functionals
from gpex.operators import GaussianSmoothing, Gradient, Hessian, Identity


def _problem(tile_size=None, *, mixed_operators=False):
    locations = Locations(jnp.array([[-0.7, 0.2], [0.0, -0.5], [0.8, 0.6]]))
    dataset = FunctionalDataset(
        [
            ObservationBlock(
                ObservationDesignBlock(
                    FunctionalBlock(locations, Identity(), "values"),
                    variance=0.05,
                    max_size=locations.count,
                ),
                y=jnp.array([0.1, 0.3, -0.2]),
            ),
            ObservationBlock(
                ObservationDesignBlock(
                    FunctionalBlock(
                        Locations(jnp.array([[0.2, 0.1]])), Gradient(), "slopes"
                    ),
                    variance=0.1,
                    max_size=1,
                ),
                y=jnp.array([[0.4, -0.1]]),
            ),
        ]
    )
    if not mixed_operators:
        dataset = FunctionalDataset([dataset.blocks[0]])
    target_locations = Locations(jnp.array([[-0.3, 0.4], [0.6, -0.2]]))
    targets = [
        FunctionalBlock(target_locations, Identity(), "values"),
        FunctionalBlock(target_locations, Gradient(), "gradient"),
    ]
    if mixed_operators:
        targets.extend(
            [
                FunctionalBlock(target_locations, Hessian(), "hessian"),
                FunctionalBlock(
                    target_locations,
                    GaussianSmoothing(0.1, quadrature_order=2),
                    "average",
                ),
            ]
        )
    query = FunctionalQuery(targets)
    gp = GaussianProcess(RBFKernel(lengthscale=0.8), covariance_tile_size=tile_size)
    return gp, dataset, query


@pytest.mark.parametrize("compiled", [False, True])
def test_covariance_result_constructor_derives_variance_and_validates_layout(compiled):
    _, _, query = _problem(mixed_operators=True)
    index_map = query.index_map

    def construct(matrix):
        return CovarianceResult(matrix, index_map)

    if compiled:
        construct = eqx.filter_jit(construct)
    diagonal = jnp.arange(1.0, index_map.size + 1)
    result = construct(jnp.diag(diagonal))
    np.testing.assert_array_equal(result.variance.flat, diagonal)
    np.testing.assert_array_equal(
        result.variance["gradient"], index_map.unpack(diagonal)["gradient"]
    )
    assert result.block("gradient", "hessian").shape == (2, 2, 2, 2, 2)
    for shape in ((index_map.size,), (index_map.size, 1), (1, 1)):
        with pytest.raises(ValueError, match="covariance has shape"):
            construct(jnp.ones(shape))


@pytest.mark.parametrize("compiled", [False, True])
def test_structured_values_validate_the_flat_layout(compiled):
    _, _, query = _problem(mixed_operators=True)
    index_map = query.index_map

    def construct(values):
        return StructuredValues(values, index_map)

    if compiled:
        construct = jax.jit(construct)
    for shape in ((), (index_map.size - 1,), (2, index_map.size + 1)):
        with pytest.raises(ValueError, match="last axis must have size"):
            construct(jnp.ones(shape))
    values = jnp.arange(2.0 * index_map.size).reshape(2, index_map.size)
    result = construct(values)
    np.testing.assert_array_equal(
        result["hessian"], index_map.unpack(values)["hessian"]
    )
    with pytest.raises(KeyError):
        result["unknown"]


def test_result_views_follow_canonical_arrays_after_jit_vmap_and_updates():
    _, _, query = _problem(mixed_operators=True)
    index_map = query.index_map
    size = index_map.size
    means = jnp.arange(2.0 * size).reshape(2, size)
    factor = jnp.arange(1.0, size * size + 1).reshape(size, size) / size
    matrix = factor @ factor.T + jnp.eye(size)
    covariances = jnp.stack((matrix, 2 * matrix))

    @jax.jit
    @jax.vmap
    def construct(mean, covariance):
        return StructuredValues(mean, index_map), CovarianceResult(
            covariance, index_map
        )

    values, covariance = construct(means, covariances)
    # Canonical updates must be visible through every derived view.
    values = eqx.tree_at(lambda result: result.flat, values, means + 1)
    covariance = eqx.tree_at(lambda result: result.full, covariance, covariances * 3)
    expected_means = index_map.unpack(means + 1)
    expected_variances = index_map.unpack(
        jnp.diagonal(covariances * 3, axis1=-2, axis2=-1)
    )
    for block in index_map.blocks:
        np.testing.assert_array_equal(values[block.name], expected_means[block.name])
        np.testing.assert_array_equal(values.by_name[block.name], values[block.name])
        np.testing.assert_array_equal(
            covariance.variance[block.name], expected_variances[block.name]
        )
    left, right = index_map.block("gradient"), index_map.block("hessian")
    np.testing.assert_array_equal(
        covariance.block("gradient", "hessian"),
        (covariances * 3)[:, left.slice, right.slice].reshape(
            (2, *left.shape, *right.shape)
        ),
    )

    def sum_views(mean, matrix):
        values, covariance = construct(mean, matrix)
        return values["gradient"].sum() + covariance.variance["gradient"].sum()

    mean_gradient, covariance_gradient = jax.jit(jax.grad(sum_views, argnums=(0, 1)))(
        means, covariances
    )
    expected_mean_gradient = jnp.zeros_like(means).at[:, left.slice].set(1)
    expected_covariance_gradient = jnp.stack(
        [jnp.diag(row) for row in expected_mean_gradient]
    )
    np.testing.assert_array_equal(mean_gradient, expected_mean_gradient)
    np.testing.assert_array_equal(covariance_gradient, expected_covariance_gradient)


def test_variance_matches_full_covariance_for_mixed_operators():
    gp, dataset, query = _problem(tile_size=3, mixed_operators=True)
    posterior = gp.condition(dataset, jitter=2e-4)
    prior = eqx.filter_jit(lambda model, q: model.prior_variance(q))(gp, query)
    actual = eqx.filter_jit(lambda p, q: p.predict_variance(q))(posterior, query)
    expected = posterior.predict_covariance(query).variance
    prior_reference = gp.prior_covariance(query).variance
    np.testing.assert_allclose(prior.flat, prior_reference.flat, atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(actual.flat, expected.flat, atol=2e-5, rtol=2e-5)
    assert actual["gradient"].shape == (2, 2)
    assert actual["hessian"].shape == (2, 2, 2)
    assert actual.index_map == query.index_map


def test_variance_kernel_gradients_match_full_prediction():
    gp, dataset, query = _problem()

    def prediction(lengthscale, diagonal):
        model = eqx.tree_at(
            lambda p: p.kernel.lengthscales.lengthscale_parameter, gp, lengthscale
        )
        posterior = model.condition(dataset)
        if diagonal:
            return jnp.sum(posterior.predict_variance(query).flat)
        return jnp.trace(posterior.predict_covariance(query).full)

    actual = eqx.filter_jit(jax.value_and_grad(lambda x: prediction(x, True)))(
        jnp.array(0.8)
    )
    expected = eqx.filter_jit(jax.value_and_grad(lambda x: prediction(x, False)))(
        jnp.array(0.8)
    )
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_allclose(a, b, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("kind", ["joint", "marginals"])
def test_joint_prediction_matches_separate_predictions(kind):
    gp, dataset, query = _problem(tile_size=3)
    posterior = gp.condition(dataset, jitter=2e-4)

    def operation(p, q):
        return p.predict(q) if kind == "joint" else p.predict_marginals(q)

    actual = eqx.filter_jit(operation)(posterior, query)
    expected_mean = posterior.predict_mean(query)
    expected_covariance = posterior.predict_covariance(query)
    np.testing.assert_allclose(
        actual.mean.flat, expected_mean.flat, atol=2e-5, rtol=2e-5
    )
    actual_uncertainty = actual.covariance if kind == "joint" else actual.variance
    np.testing.assert_allclose(
        actual_uncertainty.full if kind == "joint" else actual_uncertainty.flat,
        expected_covariance.full
        if kind == "joint"
        else expected_covariance.variance.flat,
        atol=2e-5,
        rtol=2e-5,
    )
    assert actual.mean.index_map == actual_uncertainty.index_map == query.index_map


def test_marginals_do_not_build_full_query_covariance(monkeypatch):
    import gpex.gp as gp_module

    gp, dataset, query = _problem()
    posterior = gp.condition(dataset)
    assemble_covariance = gp_module.assemble_covariance
    q = query.index_map.size

    def bounded_covariance(left, right, kernel, *, tile_size):
        sizes = (prepare_functionals(left).size, prepare_functionals(right).size)
        assert sizes != (q, q), "marginals must not construct full target covariance"
        return assemble_covariance(left, right, kernel, tile_size=tile_size)

    monkeypatch.setattr(gp_module, "assemble_covariance", bounded_covariance)
    eqx.filter_jit(lambda p, q: p.predict_marginals(q))(posterior, query)


@pytest.mark.parametrize("kind", ["joint", "marginals"])
def test_joint_prediction_kernel_and_query_gradients_match_separate_predictions(kind):
    gp, dataset, query = _problem(tile_size=3)
    points = query.blocks[0].locations.points

    def objective(lengthscale, points, *, joint):
        model = eqx.tree_at(
            lambda model: model.kernel.lengthscales.lengthscale_parameter,
            gp,
            lengthscale,
        )
        target = FunctionalQuery(
            [
                FunctionalBlock(Locations(points), block.operator, block.name)
                for block in query.blocks
            ]
        )
        posterior = model.condition(dataset)
        if joint:
            if kind == "marginals":
                result = posterior.predict_marginals(target)
                return jnp.sum(result.mean.flat) + jnp.sum(result.variance.flat)
            result = posterior.predict(target)
            return jnp.sum(result.mean.flat) + jnp.trace(result.covariance.full)
        return jnp.sum(posterior.predict_mean(target).flat) + jnp.trace(
            posterior.predict_covariance(target).full
        )

    def evaluate(joint):
        return jax.jit(
            jax.value_and_grad(
                lambda scale, x: objective(scale, x, joint=joint), argnums=(0, 1)
            )
        )(jnp.asarray(0.8), points)

    for actual, expected in zip(
        jax.tree.leaves(evaluate(True)), jax.tree.leaves(evaluate(False)), strict=True
    ):
        np.testing.assert_allclose(actual, expected, atol=3e-4, rtol=3e-4)


@pytest.mark.parametrize("method", ["predict_variance", "predict_marginals"])
def test_variance_prediction_avoids_quadratic_query_storage(method):
    if jax.default_backend() != "cpu":
        pytest.skip("CPU compiler memory regression check")
    n, q = 16, 512
    gp = GaussianProcess(RBFKernel())
    dataset = FunctionalDataset(
        [
            ObservationBlock(
                ObservationDesignBlock(
                    FunctionalBlock(
                        Locations(jnp.linspace(-1, 1, n)[:, None]), Identity(), "data"
                    ),
                    variance=0.1,
                    max_size=(jnp.linspace(-1, 1, n)[:, None]).shape[0],
                ),
                y=jnp.zeros(n),
            )
        ]
    )
    query = FunctionalQuery(
        [
            FunctionalBlock(
                Locations(jnp.linspace(-1, 1, q)[:, None]), Identity(), "target"
            )
        ]
    )
    posterior = gp.condition(dataset)
    executable = (
        eqx.filter_jit(lambda p, t: getattr(p, method)(t))
        .lower(posterior, query)
        .compile()
    )
    stats = executable.compiled.memory_analysis()
    dense_bytes = q * q * query.blocks[0].locations.points.dtype.itemsize
    assert stats.temp_size_in_bytes + stats.output_size_in_bytes < dense_bytes


def test_complete_marginals_cross_outer_jit_and_vmap():
    gp, dataset, query = _problem()

    @jax.jit
    @jax.vmap
    def predict(offset):
        shifted = eqx.tree_at(
            lambda d: d.blocks[0].y, dataset, dataset.blocks[0].y + offset
        )
        return gp.condition(shifted).predict_marginals(query)

    results = predict(jnp.array([0.0, 0.2]))
    assert results.mean["gradient"].shape == (2, 2, 2)
    assert results.variance["gradient"].shape == (2, 2, 2)
    reference = gp.condition(dataset).predict(query)
    np.testing.assert_allclose(results.mean.flat[0], reference.mean.flat, atol=2e-5)
    np.testing.assert_allclose(
        results.variance.flat[0], reference.covariance.variance.flat, atol=2e-5
    )
    np.testing.assert_allclose(
        results.variance.flat[1], results.variance.flat[0], atol=2e-5
    )
