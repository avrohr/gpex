"""Reserved storage is equivalent to the active statistical problem."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from gpex import (
    FeaturePosteriorPath,
    FunctionalBlock,
    FunctionalDataset,
    FunctionalQuery,
    GaussianProcess,
    ObservationBlock,
    ObservationDesign,
    PreparedQuery,
)
from gpex.blocks import Locations
from gpex.covariance_systems import CovarianceSystem
from gpex.fitting import fit, negative_log_marginal_likelihood, negative_log_posterior
from gpex.fitting.subsets import _select_random_subset
from gpex.kernels import RBFKernel
from gpex.kernels.base import Kernel
from gpex.means import LinearMean
from gpex.moments import assemble_covariance, assemble_mean
from gpex.moments.measurements import prepare_functionals
from gpex.operators import BoxAverage, Gradient, Identity
from gpex.parameters import PositiveParameter, log_prior
from gpex.parameters.priors import GammaPrior


def _tol():
    # The small reference systems have observation variances >= 0.12. This covers
    # float32 factorization rounding for this well-conditioned problem.
    return 2e-9 if jax.config.x64_enabled else 4e-5


def _specification():
    return [
        (
            jnp.array([[-0.7, 0.2], [0.4, -0.3]]),
            Identity(),
            jnp.array([0.3, 0.7]),
            0.12,
            4,
        ),
        (jnp.array([[0.2, 0.6]]), Gradient(), jnp.array([[0.5, -0.1]]), 0.2, 2),
        (jnp.array([[0.6, 0.1]]), Identity(), jnp.array([0.8]), 0.15, 3),
        (
            jnp.array([[-0.3, -0.2]]),
            BoxAverage(0.1, quadrature_order=2),
            jnp.array([0.4]),
            0.2,
            2,
        ),
    ]


def _dataset(specification):
    return FunctionalDataset(
        [
            ObservationBlock.from_arrays(
                x, y, operator=op, variance=variance, name=str(i), max_size=capacity
            )
            for i, (x, op, y, variance, capacity) in enumerate(specification)
        ]
    )


def _reference_covariance(left, right, kernel):
    """Apply the public operators directly, without measurement preparation."""
    rows = []
    for a in left:
        row = []
        for b in right:

            def pair(x, y, a=a, b=b):
                return a.operator.apply(
                    lambda u: b.operator.apply(lambda v: kernel(u, v), y), x
                )

            values = jax.vmap(
                lambda x, b=b: jax.vmap(lambda y: pair(x, y))(b.locations.points)
            )(a.locations.points)
            values = jnp.moveaxis(values, 1, len(a.shape))
            row.append(values.reshape(a.size, b.size))
        rows.append(jnp.concatenate(row, axis=1))
    return jnp.concatenate(rows)


def test_reserved_inference_matches_independent_active_problem():
    spec = _specification()
    data = _dataset(spec)
    compact = [
        FunctionalBlock(x, op, str(i)) for i, (x, op, _, _, _) in enumerate(spec)
    ]
    query = FunctionalQuery(
        [
            FunctionalBlock(jnp.array([[-0.4, 0.5], [0.8, 0.3]]), Identity(), "values"),
            FunctionalBlock(jnp.array([[0.3, 0.1]]), Gradient(), "gradient"),
        ]
    )
    gp = GaussianProcess(
        RBFKernel(lengthscale=0.8),
        LinearMean(jnp.array([0.3, -0.2]), 0.4),
    )

    @jax.jit
    def reference():
        matrix = _reference_covariance(compact, compact, gp.kernel)
        variance = jnp.concatenate(
            [jnp.full(y.size, variance) for _, _, y, variance, _ in spec]
        )
        matrix += jnp.diag(variance + 1e-6)
        residual = jnp.concatenate(
            [
                (y - jax.vmap(lambda p, op=op: op.apply(gp.mean, p))(x)).reshape(-1)
                for x, op, y, _, _ in spec
            ]
        )
        cross = _reference_covariance(compact, query.blocks, gp.kernel)
        mean = jnp.concatenate(
            [
                jax.vmap(lambda x, b=b: b.operator.apply(gp.mean, x))(
                    b.locations.points
                ).reshape(-1)
                for b in query.blocks
            ]
        ) + cross.T @ jnp.linalg.solve(matrix, residual)
        covariance = _reference_covariance(
            query.blocks, query.blocks, gp.kernel
        ) - cross.T @ jnp.linalg.solve(matrix, cross)
        nll = (
            residual @ jnp.linalg.solve(matrix, residual)
            + jnp.linalg.slogdet(matrix)[1]
            + residual.size * jnp.log(2 * jnp.pi)
        ) / 2
        return mean, covariance, nll

    mean, covariance, nll = reference()
    # The complete posterior and all public result types cross an outer JIT.
    posterior = eqx.filter_jit(gp.condition)(data)
    full = jax.jit(lambda p: p.predict(query))(posterior)
    marginal = jax.jit(lambda p: p.predict_marginals(query))(posterior)
    for value in (
        full.mean.flat,
        marginal.mean.flat,
        posterior.predict_mean(query).flat,
    ):
        np.testing.assert_allclose(value, mean, atol=_tol(), rtol=_tol())
    np.testing.assert_allclose(
        full.covariance.full, covariance, atol=_tol(), rtol=_tol()
    )
    for value in (marginal.variance.flat, posterior.predict_variance(query).flat):
        np.testing.assert_allclose(
            value, jnp.diag(covariance), atol=_tol(), rtol=_tol()
        )
    np.testing.assert_allclose(
        negative_log_marginal_likelihood(gp, data), nll, atol=_tol(), rtol=_tol()
    )


def test_system_views_mask_canonical_slots_and_keep_unit_padding():
    data = _dataset(_specification())
    measurements = prepare_functionals(data.design.blocks)
    system = CovarianceSystem(measurements, RBFKernel(), data.variance_flat, 1e-6)
    covariance = assemble_covariance(measurements, measurements, system.kernel)
    matrix = jax.jit(lambda s: s.materialize(tile_size=3))(system)
    inactive = ~measurements.active
    np.testing.assert_array_equal(covariance[inactive], 0)
    np.testing.assert_array_equal(covariance[:, inactive], 0)
    np.testing.assert_array_equal(
        matrix[inactive][:, inactive], jnp.eye(int(inactive.sum()))
    )
    np.testing.assert_array_equal(
        assemble_mean(measurements, LinearMean(jnp.ones(2), 5.0))[inactive], 0
    )


def test_one_conditioning_executable_accepts_replaced_appended_and_empty_data():
    gp = GaussianProcess(
        RBFKernel(),
    )
    block = ObservationBlock.from_arrays(
        jnp.array([[1.0]]), jnp.array([0.3]), variance=0.1, name="data", max_size=4
    )
    condition = eqx.filter_jit(eqx.debug.assert_max_traces(gp.condition, max_traces=1))

    blocks = [
        block,
        block.append(Locations(jnp.array([[1.3], [1.7]])), jnp.array([0.4, 0.9])),
        block.replace(jnp.array([[2.0], [2.5]]), jnp.array([0.8, 0.7])),
        block.replace(jnp.empty((0, 1)), jnp.empty(0)),
        ObservationBlock.from_arrays(
            jnp.array([[4.0]]), jnp.array([0.9]), variance=0.2, name="data", max_size=4
        ),
    ]
    query = FunctionalQuery(
        [FunctionalBlock(jnp.array([[1.2], [2.2]]), Identity(), "q")]
    )
    for block in blocks:
        data = FunctionalDataset([block])
        posterior = condition(data)
        np.testing.assert_allclose(
            gp.condition(data).alpha, posterior.alpha, atol=_tol()
        )
        assert posterior.alpha.shape == (4,)
        if int(block.count) == 0:
            prediction = posterior.predict(query)
            np.testing.assert_allclose(
                prediction.covariance.full, gp.prior_covariance(query).full, atol=_tol()
            )
            np.testing.assert_array_equal(posterior.alpha, 0)
            assert negative_log_marginal_likelihood(gp, data) == 0
    assert eqx.debug.get_num_traces(condition) == 1


@pytest.mark.parametrize("compiled", [False, True])
def test_capacity_overflow_is_visible_and_storage_is_immutable(compiled):
    block = ObservationBlock.from_arrays(
        jnp.array([[1.0]]), jnp.array([0.3]), variance=0.1, name="data", max_size=2
    )

    def operation(b):
        return b.append(Locations(jnp.array([[2.0]])), jnp.array([0.5]))

    if compiled:
        operation = jax.jit(operation)
    filled = operation(block)
    assert int(block.count) == 1
    assert int(filled.count) == 2
    np.testing.assert_array_equal(block.y, jnp.array([0.3, 0]))
    with pytest.raises(Exception, match="exceed max_size"):
        jax.block_until_ready(operation(filled))
    with pytest.raises(ValueError, match="exceed max_size"):
        block.replace(jnp.ones((3, 1)), jnp.ones(3))


def test_default_capacity_and_empty_block_construction():
    block = ObservationBlock.from_arrays(
        jnp.ones((2, 2)), jnp.ones((2, 2)), operator=Gradient(), variance=0.2, name="g"
    )
    assert block.capacity == 1024
    assert block.functional.size == 2048
    assert int(block.design.active_size) == 4
    large = ObservationBlock.from_arrays(
        jnp.ones((1025, 1)), jnp.ones(1025), variance=0.1, name="large"
    )
    assert large.capacity == 1025
    empty = ObservationBlock.from_arrays(
        jnp.empty((0, 2)),
        jnp.empty((0, 2)),
        operator=Gradient(),
        variance=0.2,
        name="g",
        padding_point=jnp.array([1.0, 2.0]),
    )
    assert empty.capacity == 1024
    assert int(empty.count) == 0
    np.testing.assert_array_equal(
        empty.functional.locations.points,
        jnp.broadcast_to(jnp.array([1.0, 2.0]), (1024, 2)),
    )
    with pytest.raises(ValueError, match="padding_point"):
        ObservationBlock.from_arrays(
            jnp.empty((0, 1)), jnp.empty(0), variance=0.1, name="empty"
        )


def test_gradients_and_nonzero_mean_exclude_padding():
    # Six independent numerical inputs exercise data, locations, mean and kernel
    # derivatives. The reference constructs the active RBF system directly.
    points = jnp.array([0.4, 0.9, 1.5])
    values = jnp.array([0.3, 0.8, 0.6])

    def actual(p):
        x = points.at[0].add(p[0])[:, None]
        y = values.at[1].add(p[1])
        gp = GaussianProcess(
            RBFKernel(lengthscale=p[2], variance=p[3]),
            LinearMean(jnp.array([p[4]]), 0.2),
        )
        data = FunctionalDataset(
            [ObservationBlock.from_arrays(x, y, variance=p[5], name="v", max_size=7)]
        )
        return negative_log_marginal_likelihood(gp, data)

    def reference(p):
        x = points.at[0].add(p[0])
        y = values.at[1].add(p[1]) - (p[4] * x + 0.2)
        matrix = p[3] * jnp.exp(-0.5 * ((x[:, None] - x[None, :]) / p[2]) ** 2)
        matrix += jnp.eye(3) * (p[5] + 1e-6)
        return (
            y @ jnp.linalg.solve(matrix, y)
            + jnp.linalg.slogdet(matrix)[1]
            + 3 * jnp.log(2 * jnp.pi)
        ) / 2

    p = jnp.array([0.03, 0.1, 0.7, 1.2, 0.3, 0.15])
    actual_value, actual_grad = jax.jit(jax.value_and_grad(actual))(p)
    expected_value, expected_grad = jax.jit(jax.value_and_grad(reference))(p)
    np.testing.assert_allclose(actual_value, expected_value, atol=_tol(), rtol=_tol())
    np.testing.assert_allclose(actual_grad, expected_grad, atol=_tol(), rtol=_tol())


def test_prepared_design_ignores_empty_slots_in_existing_and_candidate_data():
    gp = GaussianProcess(
        RBFKernel(),
    )
    original = ObservationBlock.from_arrays(
        jnp.array([[-0.8], [0.2]]),
        jnp.array([0.5, 0.3]),
        variance=0.1,
        name="old",
        max_size=4,
    )
    query = FunctionalQuery(
        [FunctionalBlock(jnp.array([[-0.4], [0.6]]), Gradient(), "q")]
    )
    conditioning = gp.prepare_conditioning(FunctionalDataset([original]))
    prepared = jax.jit(lambda g: PreparedQuery(g, query))(conditioning)

    def proposal(x, capacity):
        return ObservationBlock.from_arrays(
            x[:, None], jnp.zeros(x.shape), variance=0.15, name="new", max_size=capacity
        ).design

    def actual(x):
        return prepared.predict_covariance_after(proposal(x, 3)).full

    def reference(x):
        compact = ObservationBlock.from_arrays(
            original.functional.locations.points[:2],
            original.y[:2],
            variance=0.1,
            name="old",
            max_size=2,
        )
        return (
            gp.prepare_conditioning(ObservationDesign([compact.design, proposal(x, 1)]))
            .predict_covariance(query)
            .full
        )

    x = jnp.array([0.7])
    np.testing.assert_allclose(
        jax.jit(actual)(x), jax.jit(reference)(x), atol=_tol(), rtol=_tol()
    )
    np.testing.assert_allclose(
        jax.jit(jax.grad(lambda x: jnp.trace(actual(x))))(x),
        jax.jit(jax.grad(lambda x: jnp.trace(reference(x))))(x),
        atol=_tol(),
        rtol=_tol(),
    )
    np.testing.assert_allclose(
        prepared.predict_variance_after(proposal(x, 3)).flat,
        jnp.diag(actual(x)),
        atol=_tol(),
        rtol=_tol(),
    )
    empty = ObservationBlock.from_arrays(
        jnp.empty((0, 1)),
        jnp.empty(0),
        variance=0.1,
        name="empty",
        max_size=3,
        padding_point=jnp.array([0.5]),
    )
    np.testing.assert_allclose(
        prepared.predict_covariance_after(empty.design).full,
        prepared.covariance.full,
        atol=_tol(),
        rtol=_tol(),
    )


def test_posterior_path_matches_active_pathwise_correction():
    block = ObservationBlock.from_arrays(
        jnp.array([[0.2], [0.8]]),
        jnp.array([0.7, 0.4]),
        variance=0.1,
        name="v",
        max_size=5,
    )
    gp = GaussianProcess(RBFKernel(lengthscale=0.6), LinearMean(jnp.array([0.2]), 0.3))
    posterior = eqx.filter_jit(gp.condition)(FunctionalDataset([block]))
    key = jax.random.key(8)
    path = jax.jit(lambda p: FeaturePosteriorPath(p, key, feature_count=16))(posterior)
    feature_key, weight_key, noise_key = jax.random.split(key, 3)
    features = gp.kernel.random_features(feature_key, 1, feature_count=16)
    weights = jax.random.normal(weight_key, (16,))

    def prior(point):
        return jnp.dot(weights, features(point), precision=jax.lax.Precision.HIGHEST)

    x = block.functional.locations.points[:2]
    noise_draw = jnp.sqrt(0.1 + 1e-6) * jax.random.normal(noise_key, (5,))[:2]
    matrix = jax.vmap(lambda u: jax.vmap(lambda v: gp.kernel(u, v))(x))(x) + jnp.eye(
        2
    ) * (0.1 + 1e-6)
    correction = jnp.linalg.solve(
        matrix, block.y[:2] - jax.vmap(gp.mean)(x) - jax.vmap(prior)(x) - noise_draw
    )

    def reference(point):
        cross = jax.vmap(lambda u: gp.kernel(point, u))(x)
        return gp.mean(point) + prior(point) + cross @ correction

    point = jnp.array([0.5])
    np.testing.assert_allclose(path(point), reference(point), atol=_tol(), rtol=_tol())
    np.testing.assert_allclose(
        Gradient().apply(path, point),
        jax.grad(reference)(point),
        atol=_tol(),
        rtol=_tol(),
    )
    np.testing.assert_array_equal(path.correction_weights[2:], 0)


def test_conditioning_supports_model_gradients_and_vmap():
    gp = GaussianProcess(RBFKernel())
    block = ObservationBlock.from_arrays(
        jnp.array([[0.2]]), jnp.array([0.4]), variance=0.1, name="v", max_size=3
    )
    data = FunctionalDataset([block])
    condition = eqx.filter_jit(GaussianProcess.condition)
    derivative = eqx.filter_grad(lambda model: jnp.sum(condition(model, data).alpha))(
        gp
    )
    # For one observation, alpha divides y by the kernel variance, observation
    # variance, and jitter added together.
    np.testing.assert_allclose(
        derivative.kernel.variance_parameter, -0.4 / (1.1 + 1e-6) ** 2, atol=_tol()
    )
    datasets = jax.vmap(
        lambda y: FunctionalDataset([block.replace(jnp.array([[0.2]]), y)])
    )(jnp.array([[0.1], [0.6]]))
    posteriors = jax.jit(jax.vmap(gp.condition))(datasets)
    np.testing.assert_allclose(
        posteriors.alpha[:, 0], jnp.array([0.1, 0.6]) / (1.1 + 1e-6), atol=_tol()
    )
    unbound = condition(gp, data)
    np.testing.assert_allclose(unbound.alpha, gp.condition(data).alpha, atol=_tol())


def test_padding_uses_valid_locations_and_has_zero_data_derivatives():
    class PositiveDomainKernel(Kernel):
        def __call__(self, x, y):
            return jnp.exp(-jnp.sum((jnp.log(x) - jnp.log(y)) ** 2))

    gp = GaussianProcess(PositiveDomainKernel())

    def value(x, capacity):
        block = ObservationBlock.from_arrays(
            x, jnp.array([0.3, 0.6]), variance=0.1, name="v", max_size=capacity
        )
        return negative_log_marginal_likelihood(gp, FunctionalDataset([block]))

    x = jnp.array([[1.0], [2.0]])
    padded = jax.jit(jax.value_and_grad(lambda x: value(x, 5)))(x)
    compact = jax.jit(jax.value_and_grad(lambda x: value(x, 2)))(x)
    for actual, expected in zip(padded, compact, strict=True):
        np.testing.assert_allclose(actual, expected, atol=_tol(), rtol=_tol())
    block = ObservationBlock.from_arrays(
        x, jnp.array([0.3, 0.6]), variance=0.1, name="v", max_size=5
    )

    def loss(points, values):
        updated = eqx.tree_at(
            lambda b: (b.design.functional.locations.points, b.y),
            block,
            (points, values),
        )
        return negative_log_marginal_likelihood(gp, FunctionalDataset([updated]))

    gradients = jax.jit(jax.grad(loss, argnums=(0, 1)))(
        block.functional.locations.points, block.y
    )
    for gradient in gradients:
        np.testing.assert_array_equal(gradient[2:], 0)


def test_map_and_fitting_keep_one_prior_per_parameter_and_the_active_objective():
    prior = GammaPrior(2.0, 3.0)
    gp = GaussianProcess(
        RBFKernel(variance=PositiveParameter(1.2, prior=prior), lengthscale=0.6)
    )
    x, y = jnp.array([[-0.5], [0.2], [0.7]]), jnp.array([0.3, 0.8, 0.4])

    def model(capacity):
        block = ObservationBlock.from_arrays(
            x,
            y,
            variance=PositiveParameter(0.15, prior=prior),
            name="v",
            max_size=capacity,
        )
        return gp, FunctionalDataset([block])

    def objective(model):
        return negative_log_posterior(*model)

    compact, padded = model(3), model(6)
    expected_prior = prior.log_prob(jnp.asarray(1.2)) + prior.log_prob(
        jnp.asarray(0.15)
    )
    np.testing.assert_allclose(
        log_prior(padded), expected_prior, atol=_tol(), rtol=_tol()
    )
    np.testing.assert_allclose(
        objective(padded),
        negative_log_marginal_likelihood(*compact) - expected_prior,
        atol=_tol(),
        rtol=_tol(),
    )
    fitted = [
        fit(m, objective, optax.sgd(0.01), step_count=3) for m in (compact, padded)
    ]
    np.testing.assert_allclose(
        fitted[0].losses, fitted[1].losses, atol=_tol(), rtol=_tol()
    )
    for result in fitted:
        # Validate the reported value against the objective, rather than only
        # requiring improvement from three deliberately short optimizer steps.
        np.testing.assert_allclose(
            result.losses[-1], objective(result.model), atol=_tol(), rtol=_tol()
        )


def test_subsets_sample_only_active_locations_and_reject_insufficient_data():
    x = jnp.array([[0.1], [0.5], [0.9]])
    y = jnp.array([2.0, 4.0, 6.0])
    block = ObservationBlock.from_arrays(x, y, variance=0.1, name="v", max_size=8)
    model = GaussianProcess(RBFKernel()), FunctionalDataset([block])
    sample = jax.jit(lambda key: _select_random_subset(model, (2,), key))
    for key in jax.random.split(jax.random.key(17), 4):
        _, data = sample(key)
        selected = data.blocks[0]
        assert selected.capacity == 2
        assert len(np.unique(selected.y)) == 2
        assert np.isin(selected.y, y).all()
        np.testing.assert_allclose(
            selected.y,
            5 * selected.functional.locations.points[:, 0] + 1.5,
            atol=_tol(),
        )
    with pytest.raises(Exception, match="active location count"):
        jax.block_until_ready(
            jax.jit(lambda m: _select_random_subset(m, (4,), jax.random.key(0)))(model)
        )
