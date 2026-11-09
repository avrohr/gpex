"""Fixed storage updates preserve the active GP and compiled executable."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpex import (
    FeaturePosteriorPath,
    FunctionalBlock,
    FunctionalDataset,
    FunctionalQuery,
    GaussianProcess,
    ObservationBlock,
    PreparedQuery,
)
from gpex.blocks import Locations
from gpex.covariance_systems import CovarianceSystem
from gpex.kernels import RBFKernel
from gpex.means import LinearMean
from gpex.operators import BoxAverage, DirectionalDerivative, Gradient, Identity


def _tolerance():
    # Small systems with noise >= .1. The float32 budget covers repeated Schur
    # subtractions.
    return 3e-9 if jax.config.x64_enabled else 8e-5


def _empty(name, operator, capacity, variance):
    return ObservationBlock.from_arrays(
        jnp.empty((0, 2)),
        jnp.empty((0, *operator.output_shape(2))),
        name=name,
        operator=operator,
        variance=variance,
        max_size=capacity,
        padding_point=jnp.zeros(2),
    )


@eqx.filter_jit
def _covariance(left, right, kernel):
    """Independent direct operator application, without covariance compilation."""
    rows = []
    for a in left:
        row = []
        for b in right:

            def pair(x, y, a=a, b=b):
                return a.operator.apply(
                    lambda u: b.operator.apply(lambda v: kernel(u, v), y), x
                )

            tensor = jax.vmap(
                lambda x, b=b: jax.vmap(lambda y: pair(x, y))(b.locations.points)
            )(a.locations.points)
            row.append(jnp.moveaxis(tensor, 1, len(a.shape)).reshape(a.size, b.size))
        rows.append(jnp.concatenate(row, axis=1))
    return jnp.concatenate(rows)


def _independent_prediction(gp, blocks, query):
    active = [b for b in blocks if int(b.count)]
    functionals = [
        FunctionalBlock(
            b.functional.locations.points[: int(b.count)],
            b.functional.operator,
            b.functional.name,
        )
        for b in active
    ]
    variance = jnp.concatenate(
        [
            jnp.full(f.size, b.design.variance)
            for b, f in zip(active, functionals, strict=True)
        ]
    )
    y = jnp.concatenate([b.y[: int(b.count)].reshape(-1) for b in active])

    def means(fs):
        return jnp.concatenate(
            [
                jax.vmap(lambda x, f=f: f.operator.apply(gp.mean, x))(
                    f.locations.points
                ).reshape(-1)
                for f in fs
            ]
        )

    matrix = np.asarray(
        _covariance(functionals, functionals, gp.kernel) + jnp.diag(variance + 1e-6),
        dtype=np.float64,
    )
    cross = np.asarray(
        _covariance(functionals, query.blocks, gp.kernel), dtype=np.float64
    )
    prior = np.asarray(
        _covariance(query.blocks, query.blocks, gp.kernel), dtype=np.float64
    )
    alpha = np.linalg.solve(
        matrix, np.asarray(y - means(functionals), dtype=np.float64)
    )
    return (
        np.asarray(means(query.blocks)) + cross.T @ alpha,
        prior - cross.T @ np.linalg.solve(matrix, cross),
    )


@pytest.mark.parametrize("compiled", [False, True])
def test_interleaved_append_matches_independent_reference(compiled):
    blocks = [
        _empty("a", Identity(), 3, 0.13),
        _empty("gradient", Gradient(), 2, 0.17),
        _empty("b", Identity(), 2, 0.11),
        _empty("average", BoxAverage(0.2, quadrature_order=2), 2, 0.15),
    ]
    gp = GaussianProcess(
        RBFKernel(lengthscale=0.8),
        LinearMean(jnp.array([0.2, -0.3]), intercept=0.4),
    )
    query = FunctionalQuery(
        [
            FunctionalBlock(jnp.array([[0.1, 0.3], [-0.4, 0.2]]), Identity(), "q"),
            FunctionalBlock(jnp.array([[0.3, -0.2]]), Gradient(), "dq"),
        ]
    )
    posterior = gp.condition(FunctionalDataset(blocks))
    original = posterior
    shape = jax.tree.map(lambda x: x.shape, eqx.filter(posterior, eqx.is_array))

    def append(p, x, y, name):
        return p.append(Locations(x), y, block=name)

    operation = eqx.filter_jit(append) if compiled else append
    # Keep the path under test eager/compiled independently. Reference rebuilds
    # and repeated predictions need only one compiled execution path.
    predict = eqx.filter_jit(lambda p: p.predict(query))
    variance = eqx.filter_jit(lambda p: p.predict_variance(query).flat)
    rebuild = eqx.filter_jit(gp.condition)
    solve = eqx.filter_jit(lambda p, rhs: p.conditioning.solve(rhs))
    # Interleave operator groups, use a two-row batch, fill an odd scalar reserve.
    sequence = [
        (2, [[0.6, -0.1]], [0.3]),
        (1, [[0.4, 0.5]], [[0.2, -0.1]]),
        (0, [[-0.7, 0.1], [-0.2, -0.5]], [0.5, -0.3]),
        (3, [[0.1, -0.3]], [0.4]),
        (0, [[0.5, 0.2]], [0.1]),
        (1, [[-0.1, 0.2]], [[0.3, 0.2]]),
        (2, [[0.7, 0.5]], [0.7]),
        (3, [[-0.4, -0.3]], [-0.1]),
    ]
    for index, x, y in sequence:
        x, y = jnp.asarray(x), jnp.asarray(y)
        previous = posterior
        before = predict(previous)
        posterior = operation(posterior, x, y, blocks[index].functional.name)
        blocks[index] = blocks[index].append(Locations(x), y)
        assert (
            jax.tree.map(lambda x: x.shape, eqx.filter(posterior, eqx.is_array))
            == shape
        )
        mean, covariance = _independent_prediction(gp, blocks, query)
        actual = predict(posterior)
        np.testing.assert_allclose(
            actual.mean.flat, mean, atol=_tolerance(), rtol=_tolerance()
        )
        np.testing.assert_allclose(
            actual.covariance.full, covariance, atol=_tolerance(), rtol=_tolerance()
        )
        np.testing.assert_allclose(
            variance(posterior),
            np.diag(covariance),
            atol=_tolerance(),
            rtol=_tolerance(),
        )
        rebuilt = rebuild(FunctionalDataset(blocks))
        np.testing.assert_allclose(
            posterior.alpha,
            rebuilt.alpha,
            atol=10 * _tolerance(),
            rtol=10 * _tolerance(),
        )
        # General RHSs, not just the observed residual, exercise both permutations.
        rhs = jnp.arange(posterior.observation_size) / 10
        np.testing.assert_allclose(
            solve(posterior, rhs),
            solve(rebuilt, rhs),
            atol=10 * _tolerance(),
            rtol=10 * _tolerance(),
        )
        np.testing.assert_array_equal(predict(previous).mean.flat, before.mean.flat)
    np.testing.assert_allclose(
        original.predict(query).covariance.full,
        gp.prior_covariance(query).full,
        atol=_tolerance(),
        rtol=_tolerance(),
    )
    assert (
        int(posterior.conditioning.measurements.active.sum())
        == posterior.observation_size
    )


@pytest.mark.parametrize("multiple_blocks", [False, True])
def test_append_reuses_one_executable_and_returned_state(multiple_blocks):
    block = ObservationBlock.from_arrays(
        jnp.array([[0.0]]), jnp.array([0.2]), name="v", variance=0.1, max_size=6
    )
    extra = (
        [
            ObservationBlock.from_arrays(
                jnp.array([[0.8]]),
                jnp.array([0.5]),
                name="other",
                variance=0.1,
                max_size=2,
            )
        ]
        if multiple_blocks
        else []
    )
    posterior = eqx.filter_jit(
        lambda data: GaussianProcess(RBFKernel()).condition(data)
    )(FunctionalDataset([block, *extra]))
    query = FunctionalQuery([FunctionalBlock(jnp.array([[0.3]]), Identity(), "q")])
    traces = []

    @eqx.filter_jit
    def step(p, x, y, q):
        traces.append(None)
        p = p.append(Locations(x), y, block="v")
        return p, p.predict(q)

    for i in range(1, 6):
        posterior, prediction = step(
            posterior, jnp.array([[i / 5]]), jnp.array([i / 10]), query
        )
        assert jnp.all(jnp.isfinite(prediction.mean.flat))
    assert len(traces) == 1
    with pytest.raises(Exception, match="exceed"):
        step(posterior, jnp.array([[2.0]]), jnp.array([0.4]), query)


@pytest.mark.parametrize("compiled", [False, True])
def test_append_rejects_invalid_shapes_and_unknown_names(compiled):
    posterior = GaussianProcess(RBFKernel()).condition(
        FunctionalDataset([_empty("g", Gradient(), 2, 0.1)])
    )

    def append(x, y, name):
        return posterior.append(Locations(x), y, block=name)

    operation = eqx.filter_jit(append) if compiled else append
    with pytest.raises(KeyError, match="unknown"):
        operation(jnp.ones((1, 2)), jnp.ones((1, 2)), "unknown")
    with pytest.raises(ValueError, match="values.*shape"):
        operation(jnp.ones((1, 2)), jnp.ones((1,)), "g")
    with pytest.raises(ValueError, match="dimension"):
        operation(jnp.ones((1, 3)), jnp.ones((1, 2)), "g")
    with pytest.raises(ValueError, match="exceed"):
        operation(jnp.ones((3, 2)), jnp.ones((3, 2)), "g")


@pytest.mark.parametrize("multiple_blocks", [False, True])
def test_append_gradients_and_vmap_match_fresh_conditioning(multiple_blocks):
    points = jnp.array([[-0.5], [0.2]])
    values = jnp.array([0.1, 0.4])
    query = FunctionalQuery(
        [FunctionalBlock(jnp.array([[0.1], [0.7]]), Identity(), "q")]
    )

    def objective(theta, append):
        x = theta[:2, None]
        y = theta[2:4]
        gp = GaussianProcess(
            RBFKernel(lengthscale=jnp.exp(theta[4])),
            LinearMean(jnp.array([theta[5]])),
        )
        block = ObservationBlock.from_arrays(
            points, values, variance=0.13, name="v", max_size=7
        )
        extra = (
            [
                ObservationBlock.from_arrays(
                    jnp.array([[-0.8]]),
                    jnp.array([0.3]),
                    variance=0.13,
                    name="other",
                    max_size=2,
                )
            ]
            if multiple_blocks
            else []
        )
        if append:
            p = gp.condition(FunctionalDataset([block, *extra])).append(
                Locations(x), y, block="v"
            )
        else:
            p = gp.condition(FunctionalDataset([block.append(Locations(x), y), *extra]))
        result = p.predict(query)
        return jnp.sum(result.mean.flat) + jnp.sum(result.covariance.full)

    theta = jnp.array([0.5, 1.1, 0.6, -0.2, -0.3, 0.2])
    a = eqx.filter_jit(jax.value_and_grad(lambda t: objective(t, True)))(theta)
    b = eqx.filter_jit(jax.value_and_grad(lambda t: objective(t, False)))(theta)
    for actual, reference in zip(a, b, strict=True):
        np.testing.assert_allclose(
            actual, reference, atol=20 * _tolerance(), rtol=20 * _tolerance()
        )
    batch = jnp.stack((theta, theta + 0.03))
    np.testing.assert_allclose(
        eqx.filter_jit(jax.vmap(lambda t: objective(t, True)))(batch),
        eqx.filter_jit(jax.vmap(lambda t: objective(t, False)))(batch),
        atol=10 * _tolerance(),
        rtol=10 * _tolerance(),
    )


@pytest.mark.parametrize("multiple_blocks", [False, True])
def test_repeated_conditioning_append_jacobian_matches_reconditioning(multiple_blocks):
    gp = GaussianProcess(RBFKernel(lengthscale=0.8))
    initial = ObservationBlock.from_arrays(
        jnp.array([[-0.4, 0.2]]),
        jnp.zeros((1, 2)),
        operator=Gradient(),
        name="gradient",
        variance=0.17,
        max_size=3,
    )
    extra = (
        [
            ObservationBlock.from_arrays(
                jnp.array([[0.6, -0.3]]),
                jnp.zeros(1),
                name="value",
                variance=0.13,
                max_size=2,
            )
        ]
        if multiple_blocks
        else []
    )
    query = FunctionalQuery(
        [FunctionalBlock(jnp.array([[0.1, -0.2]]), Gradient(), "query")]
    )

    conditioning = gp.prepare_conditioning(FunctionalDataset([initial, *extra]))

    def covariances(points, conditioning, append):
        block = initial
        results = []
        # Each location adds two scalar rows. The second update fills capacity,
        # and the multiblock case must retain its mapping across both updates.
        for i in range(2):
            locations = Locations(points[i : i + 1])
            block = block.append(locations, jnp.zeros((1, 2)))
            conditioning = (
                conditioning.append(locations, block="gradient")
                if append
                else gp.prepare_conditioning(FunctionalDataset([block, *extra]))
            )
            results.append(conditioning.predict_covariance(query).full)
        return jnp.stack(results)

    @eqx.filter_jit
    def evaluate(points, conditioning, append):
        return covariances(points, conditioning, append), jax.jacrev(
            lambda x: covariances(x, conditioning, append)
        )(points)

    points = jnp.array([[0.3, 0.5], [-0.2, -0.6]])
    actual = evaluate(points, conditioning, True)
    expected = evaluate(points, conditioning, False)
    # Noise .17 keeps this six/eight-scalar problem well conditioned. Float32
    # allows accumulated Schur and reverse-mode rounding; x64 remains tight.
    tolerance = 2e-9 if jax.config.x64_enabled else 3e-5
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_allclose(a, b, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("multiple_blocks", [False, True])
def test_append_does_not_factor_or_materialize_reserved_covariance(
    monkeypatch, multiple_blocks
):
    block = ObservationBlock.from_arrays(
        jnp.array([[0.0]]), jnp.array([0.1]), name="v", variance=0.1, max_size=11
    )
    extra = (
        [
            ObservationBlock.from_arrays(
                jnp.array([[0.9]]),
                jnp.array([0.4]),
                name="other",
                variance=0.1,
                max_size=2,
            )
        ]
        if multiple_blocks
        else []
    )
    p = GaussianProcess(RBFKernel()).condition(FunctionalDataset([block, *extra]))
    original = jnp.linalg.cholesky

    def bounded(matrix, *args, **kwargs):
        # Only the two newly active scalars need factorization.
        assert matrix.shape[-1] <= 2, "append refactorized the reserve"
        return original(matrix, *args, **kwargs)

    def reject(*args, **kwargs):
        raise AssertionError("append materialized the full covariance")

    monkeypatch.setattr(jnp.linalg, "cholesky", bounded)
    monkeypatch.setattr(CovarianceSystem, "materialize", reject)
    updated = eqx.filter_jit(
        lambda p: p.append(
            Locations(jnp.array([[0.3], [0.7]])), jnp.array([0.2, 0.4]), block="v"
        )
    )(p)
    assert jnp.all(jnp.isfinite(updated.alpha))


def test_append_singular_schur_failure_is_visible_under_vmap():
    def operation(direction):
        block = _empty("d", DirectionalDerivative(direction), 2, 0.0)
        p = GaussianProcess(RBFKernel()).condition(
            FunctionalDataset([block]), jitter=0.0
        )
        return p.append(
            Locations(jnp.array([[0.2, 0.3]])), jnp.array([1.0]), block="d"
        ).alpha

    results = eqx.filter_jit(jax.vmap(operation))(jnp.array([[1.0, 0.0], [0.0, 0.0]]))
    assert jnp.all(jnp.isfinite(results[0]))
    assert jnp.any(~jnp.isfinite(results[1]))


def test_design_and_callable_paths_after_append():
    block = _empty("v", Identity(), 4, 0.1).append(
        Locations(jnp.array([[0.1, 0.3]])), jnp.array([0.2])
    )
    gp = GaussianProcess(RBFKernel())
    old = gp.condition(FunctionalDataset([block]))
    x, y = jnp.array([[-0.4, 0.5]]), jnp.array([0.6])
    updated = old.append(Locations(x), y, block="v")
    rebuilt = gp.condition(FunctionalDataset([block.append(Locations(x), y)]))
    query = FunctionalQuery([FunctionalBlock(jnp.array([[0.2, 0.4]]), Gradient(), "q")])
    candidate = _empty("future", Identity(), 1, 0.2).design.append(
        Locations(jnp.array([[0.7, 0.1]]))
    )
    a = (
        PreparedQuery(updated.conditioning, query)
        .predict_covariance_after(candidate)
        .full
    )
    b = (
        PreparedQuery(rebuilt.conditioning, query)
        .predict_covariance_after(candidate)
        .full
    )
    np.testing.assert_allclose(a, b, atol=10 * _tolerance(), rtol=10 * _tolerance())
    paths = [
        FeaturePosteriorPath(p, jax.random.key(42), feature_count=16)
        for p in (updated, rebuilt)
    ]
    for point in (jnp.array([0.3, 0.1]), jnp.array([-0.1, 0.5])):
        for operator in (Identity(), Gradient()):
            a, b = [operator.apply(path, point) for path in paths]
            np.testing.assert_allclose(
                a, b, atol=10 * _tolerance(), rtol=10 * _tolerance()
            )
