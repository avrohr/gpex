"""Prepared design targets agree with complete conditioning and its gradients."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpex import (
    FunctionalBlock,
    FunctionalQuery,
    GaussianProcess,
    ObservationDesign,
    ObservationDesignBlock,
    PreparedQuery,
)
from gpex.blocks import Locations
from gpex.kernels import RBFKernel
from gpex.operators import DirectionalDerivative, Gradient, Identity


def _problem(tile_size=None, *, mixed_operators=False):
    design = ObservationDesign(
        [
            ObservationDesignBlock(
                FunctionalBlock(
                    Locations(jnp.array([[-0.8, 0.1], [-0.2, -0.6], [0.5, 0.4]])),
                    Identity(),
                    "values",
                ),
                variance=0.06,
                max_size=3,
            ),
            ObservationDesignBlock(
                FunctionalBlock(
                    Locations(jnp.array([[0.0, 0.2]])), Gradient(), "slopes"
                ),
                variance=0.1,
                max_size=1,
            ),
        ]
    )
    query = FunctionalQuery(
        [
            FunctionalBlock(
                Locations(jnp.array([[0.1, -0.1]])), Gradient(), "gradient"
            ),
            FunctionalBlock(
                Locations(jnp.array([[-0.4, 0.3], [0.6, -0.2]])), Identity(), "values"
            ),
        ]
    )
    if not mixed_operators:
        design = ObservationDesign([design.blocks[0]])
        query = FunctionalQuery([query.blocks[1]])
    gp = GaussianProcess(RBFKernel(lengthscale=0.8), covariance_tile_size=tile_size)
    return gp, design, query


def _candidate(points, *, mixed_operators=False):
    return ObservationDesignBlock(
        FunctionalBlock(
            Locations(points), Gradient() if mixed_operators else Identity(), "future"
        ),
        variance=0.05,
        max_size=points.shape[0],
    )


def test_prepared_query_matches_append_and_rebuild():
    gp, design, query = _problem(tile_size=3, mixed_operators=True)
    conditioning = gp.prepare_conditioning(design, jitter=2e-3)
    prepared = eqx.filter_jit(PreparedQuery)(conditioning, query)
    np.testing.assert_allclose(
        prepared.covariance.full,
        conditioning.predict_covariance(query).full,
        atol=2e-5,
        rtol=2e-5,
    )

    @eqx.filter_jit
    def predict(target, points):
        block = _candidate(points, mixed_operators=True)
        return target.predict_covariance_after(block), target.predict_variance_after(
            block
        )

    @eqx.filter_jit
    def references(points):
        block = _candidate(points, mixed_operators=True)
        reserved = gp.prepare_conditioning(
            ObservationDesign(
                [*design.blocks, block.replace(block.functional.locations.points[:0])]
            ),
            jitter=conditioning.jitter,
        )
        appended = reserved.append(
            block.functional.locations, block="future"
        ).predict_covariance(query)
        complete = ObservationDesign([*design.blocks, block])
        rebuilt = gp.prepare_conditioning(complete, jitter=conditioning.jitter)
        return appended, rebuilt.predict_covariance(query)

    for points in (jnp.array([[0.2, -0.1]]), jnp.array([[-0.4, 0.5]])):
        covariance, variance = predict(prepared, points)
        for expected in references(points):
            np.testing.assert_allclose(
                covariance.full, expected.full, atol=3e-5, rtol=3e-5
            )
            np.testing.assert_allclose(
                variance.flat, expected.variance.flat, atol=3e-5, rtol=3e-5
            )
        assert variance["gradient"].shape == (1, 2)
        assert covariance.index_map == query.index_map
    assert prepared.conditioning.index_map.size == conditioning.index_map.size
    with pytest.raises(AttributeError):
        prepared.conditioning = conditioning


def test_prepared_query_coordinate_gradients_and_vmap_match_append():
    gp, design, query = _problem()
    conditioning = gp.prepare_conditioning(design)
    prepared = PreparedQuery(conditioning, query)
    points = jnp.array([[0.2, -0.1]])
    empty = _candidate(points).replace(points[:0])
    reserved = gp.prepare_conditioning(ObservationDesign([*design.blocks, empty]))

    def cached(x, target):
        return jnp.sum(target.predict_variance_after(_candidate(x)).flat)

    def appended(x):
        return jnp.trace(
            reserved.append(Locations(x), block="future").predict_covariance(query).full
        )

    value, gradient = eqx.filter_jit(jax.value_and_grad(cached))(points, prepared)
    expected_value, expected_gradient = eqx.filter_jit(jax.value_and_grad(appended))(
        points
    )
    np.testing.assert_allclose(value, expected_value, atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(gradient, expected_gradient, atol=2e-4, rtol=2e-4)
    proposals = jnp.stack((points, points + 0.1))
    actual = eqx.filter_jit(jax.vmap(cached, in_axes=(0, None)))(proposals, prepared)
    expected = jnp.stack([appended(x) for x in proposals])
    np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-5)


def test_prepared_query_operator_and_variance_gradients_match_rebuild():
    gp, design, query = _problem()
    conditioning = gp.prepare_conditioning(design)
    prepared = PreparedQuery(conditioning, query)
    locations = Locations(jnp.array([[0.2, -0.1], [-0.3, 0.4]]))
    template = ObservationDesignBlock(
        FunctionalBlock(locations, DirectionalDerivative(jnp.ones(2)), "future"),
        variance=0.06,
        max_size=locations.count,
    )

    def objective(parameters, cached):
        block = eqx.tree_at(
            lambda b: (b.functional.operator.direction, b.variance),
            template,
            (parameters[:2], jnp.exp(parameters[2])),
        )
        if cached:
            return jnp.sum(prepared.predict_variance_after(block).flat)
        return jnp.trace(
            gp.prepare_conditioning(ObservationDesign([*design.blocks, block]))
            .predict_covariance(query)
            .full
        )

    parameters = jnp.array([0.5, -0.8, -3.0])
    actual = eqx.filter_jit(jax.value_and_grad(lambda p: objective(p, True)))(
        parameters
    )
    expected = eqx.filter_jit(jax.value_and_grad(lambda p: objective(p, False)))(
        parameters
    )
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_allclose(a, b, atol=2e-4, rtol=2e-4)


def test_preparation_preserves_kernel_and_target_gradients():
    gp, design, query = _problem()
    future = _candidate(jnp.array([[0.2, -0.1]]))

    def objective(parameters, cached):
        model = eqx.tree_at(
            lambda p: p.kernel.lengthscales.lengthscale_parameter,
            gp,
            jnp.exp(parameters[0]),
        )
        shifted = eqx.tree_at(
            lambda q: q.blocks[0].locations.points,
            query,
            query.blocks[0].locations.points + parameters[1],
        )
        conditioning = model.prepare_conditioning(design)
        if cached:
            return jnp.sum(
                PreparedQuery(conditioning, shifted).predict_variance_after(future).flat
            )
        reserved = model.prepare_conditioning(
            ObservationDesign(
                [*design.blocks, future.replace(future.functional.locations.points[:0])]
            )
        )
        return jnp.trace(
            reserved.append(future.functional.locations, block="future")
            .predict_covariance(shifted)
            .full
        )

    parameters = jnp.array([-0.2, 0.1])
    actual = eqx.filter_jit(jax.value_and_grad(lambda p: objective(p, True)))(
        parameters
    )
    expected = eqx.filter_jit(jax.value_and_grad(lambda p: objective(p, False)))(
        parameters
    )
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_allclose(a, b, atol=2e-4, rtol=2e-4)


def test_prepared_query_rejects_incompatible_dimensions():
    gp, design, query = _problem()
    conditioning = gp.prepare_conditioning(design)
    prepared = PreparedQuery(conditioning, query)
    wrong = ObservationDesignBlock(
        FunctionalBlock(Locations(jnp.ones((1, 1))), Identity(), "wrong"),
        variance=0.1,
        max_size=1,
    )
    with pytest.raises(ValueError, match="same input dimension"):
        prepared.predict_covariance_after(wrong)
    with pytest.raises(ValueError, match="same input dimension"):
        PreparedQuery(conditioning, FunctionalQuery([wrong.functional]))


@pytest.mark.parametrize("compiled", [False, True])
def test_prepared_query_returns_nans_for_singular_measurement_covariance(compiled):
    gp, design, query = _problem()
    prepared = PreparedQuery(gp.prepare_conditioning(design, jitter=0.0), query)
    candidate = ObservationDesignBlock(
        # An exactly zero functional makes a deterministic zero pivot. A duplicate
        # nonzero row can acquire a positive pivot through Cholesky roundoff.
        FunctionalBlock(
            Locations(jnp.zeros((1, 2))), DirectionalDerivative(jnp.zeros(2)), "zero"
        ),
        variance=0.0,
        max_size=1,
    )

    def operation(p, b):
        return p.predict_covariance_after(b)

    if compiled:
        operation = eqx.filter_jit(operation)
    result = operation(prepared, candidate)
    assert jnp.any(jnp.isnan(result.full))


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("regularization", ["noise", "jitter"])
def test_prepared_query_regularizes_duplicate_measurements(compiled, regularization):
    gp, design, query = _problem()
    jitter = 0.01 if regularization == "jitter" else 0.0
    candidate = ObservationDesignBlock(
        FunctionalBlock(Locations(jnp.zeros((2, 2))), Identity(), "duplicate"),
        variance=0.01 if regularization == "noise" else 0.0,
        max_size=2,
    )

    def predict(gp, design, query, candidate):
        conditioning = gp.prepare_conditioning(design, jitter=jitter)
        return (
            PreparedQuery(conditioning, query).predict_covariance_after(candidate).full
        )

    operation = eqx.filter_jit(predict) if compiled else predict
    actual = operation(gp, design, query, candidate)
    expected = gp.prepare_conditioning(
        ObservationDesign([*design.blocks, candidate]), jitter=jitter
    )
    np.testing.assert_allclose(
        actual, expected.predict_covariance(query).full, atol=2e-6, rtol=2e-5
    )
    assert jnp.all(jnp.isfinite(actual))
    assert jnp.linalg.eigvalsh(actual).min() >= -2e-6
