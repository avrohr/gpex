from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.core import Tracer

from gpex import (
    ConditioningState,
    FunctionalBlock,
    FunctionalDataset,
    FunctionalQuery,
    GaussianProcess,
    ObservationBlock,
    ObservationDesign,
    ObservationDesignBlock,
)
from gpex.blocks import Locations
from gpex.fitting import negative_log_marginal_likelihood, negative_log_posterior
from gpex.kernels import GeneralizedRBFKernel, RBFKernel
from gpex.linalg import CholeskySystem, DenseBackend
from gpex.means import LinearMean
from gpex.operators import Gradient, Identity
from gpex.parameters import PositiveParameter
from gpex.parameters.priors import GammaPrior, LogNormalPrior


def _empty(block):
    return block.replace(block.functional.locations.points[:0], block.y[:0])


def test_likelihood_uses_backend_statistics_and_preserves_their_gradients():
    # Deliberately weight the statistics differently so bypassing either backend
    # operation changes the objective and its gradient. The activity-dependent
    # weight also catches a missing mask when preparing a padded fitting system.
    class WeightedStatistics(CholeskySystem):
        def __init__(self, prepared: CholeskySystem):
            super().__init__(prepared.factor, prepared.tile_size)

        def quadratic_form(self, right_hand_side):
            return (1 + self.size) * super().quadratic_form(right_hand_side)

        def logdet(self):
            return 3 * super().logdet()

    class WeightedBackend(DenseBackend):
        def prepare(self, system, *, materialization_tile_size=None, **kwargs):
            prepared = super().prepare(
                system, materialization_tile_size=materialization_tile_size, **kwargs
            )
            return WeightedStatistics(prepared)

    points = jnp.array([-0.8, 0.1, 0.9])
    targets = jnp.array([-0.2, 0.3, 0.8])
    observation_variance = 0.07
    dataset = FunctionalDataset(
        [
            ObservationBlock.from_arrays(
                points[:, None],
                targets,
                variance=observation_variance,
                name="values",
                max_size=5,
            )
        ]
    )

    def objective(log_lengthscale):
        gp = GaussianProcess(
            RBFKernel(lengthscale=jnp.exp(log_lengthscale)),
            solver=WeightedBackend(),
            covariance_tile_size=2,
        )
        return negative_log_marginal_likelihood(gp, dataset)

    def reference(log_lengthscale):
        distance = (points[:, None] - points[None, :]) / jnp.exp(log_lengthscale)
        matrix = jnp.exp(-0.5 * distance**2) + (observation_variance + 1e-6) * jnp.eye(
            points.size
        )
        quadratic = targets @ jnp.linalg.solve(matrix, targets)
        logdet = jnp.linalg.slogdet(matrix)[1]
        return 0.5 * (
            (1 + points.size) * quadratic
            + 3 * logdet
            + points.size * jnp.log(2 * jnp.pi)
        )

    values = jnp.array([-0.4, 0.2])
    actual = jax.jit(jax.vmap(jax.value_and_grad(objective)))(values)
    expected = jax.jit(jax.vmap(jax.value_and_grad(reference)))(values)
    tolerance = 3e-5 if values.dtype == jnp.float32 else 1e-11
    for result, target in zip(actual, expected, strict=True):
        np.testing.assert_allclose(result, target, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("compile_function", [jax.jit, eqx.filter_jit])
def test_complete_prediction_results_can_be_consumed_outside_jit(compile_function):
    gp = GaussianProcess(RBFKernel())
    dataset, query = _dataset(jnp.array([-0.2, 0.3, 0.8])), _query()

    def predict(model, data, target):
        posterior = model.condition(data)
        return (
            posterior.predict(target),
            model.prior_mean(target),
            model.prior_covariance(target),
            posterior.sample(jax.random.key(1), target, sample_count=2),
            model.sample_prior(jax.random.key(2), target, sample_count=2),
        )

    expected = predict(gp, dataset, query)
    compiled = compile_function(predict)
    for actual in (compiled(gp, dataset, query), compiled(gp, dataset, query)):
        # All numerical leaves must be concrete arrays, including named views.
        for result, reference in zip(
            jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
        ):
            assert isinstance(result, jax.Array)
            assert not isinstance(result, Tracer)
            assert jnp.allclose(result + 1, reference + 1, atol=2e-5)
        prediction = actual[0]
        assert prediction.mean.index_map == query.index_map
        assert jnp.allclose(prediction.mean["values"], expected[0].mean["values"])
        assert jnp.allclose(
            prediction.covariance.variance["gradients"],
            expected[0].covariance.variance["gradients"],
        )
        assert jnp.allclose(
            prediction.covariance.block("values", "gradients"),
            expected[0].covariance.block("values", "gradients"),
        )
        assert actual[3]["gradients"].shape == (2, 2, 1)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize(
    "kernel",
    [
        RBFKernel(lengthscale=jnp.ones(1)),
        GeneralizedRBFKernel(L=jnp.eye(1)),
    ],
)
def test_conditioning_rejects_kernel_observation_dimension_mismatch(compiled, kernel):
    dataset = FunctionalDataset(
        [
            ObservationBlock(
                ObservationDesignBlock(
                    FunctionalBlock(
                        Locations(jnp.array([[0.0, 1.0], [1.0, 0.0]])),
                        Identity(),
                        "values",
                    ),
                    variance=0.1,
                    max_size=2,
                ),
                y=jnp.array([0.0, 1.0]),
            )
        ]
    )
    gp = GaussianProcess(kernel)
    condition = eqx.filter_jit(gp.condition) if compiled else gp.condition
    with pytest.raises(ValueError, match="dimension"):
        condition(dataset)


@pytest.mark.parametrize("method", ["predict_mean", "predict_covariance", "predict"])
@pytest.mark.parametrize("compiled", [False, True])
def test_prediction_rejects_query_training_dimension_mismatch(method, compiled):
    gp = GaussianProcess(RBFKernel())
    dataset = FunctionalDataset(
        [
            ObservationBlock(
                ObservationDesignBlock(
                    FunctionalBlock(
                        Locations(jnp.array([[0.0, 1.0], [1.0, 0.0]])),
                        Identity(),
                        "values",
                    ),
                    variance=0.1,
                    max_size=2,
                ),
                y=jnp.array([0.0, 1.0]),
            )
        ]
    )
    posterior = gp.condition(dataset)
    predict = getattr(posterior, method)
    if compiled:
        predict = eqx.filter_jit(predict)
    with pytest.raises(ValueError, match="same input dimension"):
        predict(_query())


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("conditioning_only", [False, True])
def test_append_rejects_unknown_block_names(compiled, conditioning_only):
    gp = GaussianProcess(RBFKernel())
    dataset = _dataset(jnp.array([-0.2, 0.3, 0.8]))
    posterior = gp.condition(dataset)
    block = dataset.blocks[0]

    def append():
        if conditioning_only:
            return posterior.conditioning.append(
                block.functional.locations, block="missing"
            )
        return posterior.append(block.functional.locations, block.y, block="missing")

    operation = eqx.filter_jit(append) if compiled else append
    with pytest.raises(KeyError, match="missing"):
        operation()


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("conditioning_only", [False, True])
def test_duplicate_noiseless_observations_return_nans_on_factorization_failure(
    compiled, conditioning_only
):
    gp = GaussianProcess(RBFKernel(variance=100.0))
    dataset = FunctionalDataset(
        [
            ObservationBlock(
                ObservationDesignBlock(
                    FunctionalBlock(
                        Locations(jnp.zeros((2, 1))),
                        Identity(),
                        "values",
                    ),
                    variance=0.0,
                    max_size=2,
                ),
                y=jnp.ones(2),
            )
        ]
    )
    query = FunctionalQuery(
        [FunctionalBlock(Locations(jnp.zeros((1, 1))), Identity(), "query")]
    )

    def condition(data):
        if conditioning_only:
            return (
                gp.prepare_conditioning(data, jitter=0.0).predict_variance(query).flat
            )
        return gp.condition(data, jitter=0.0).predict_mean(query).flat

    if compiled:
        condition = eqx.filter_jit(condition)
    result = condition(dataset)
    assert jnp.any(jnp.isnan(result))
    recovered = gp.condition(dataset, jitter=1e-3)
    assert jnp.all(jnp.isfinite(recovered.alpha))


def test_invalid_likelihood_trials_remain_nonfinite_under_jit_and_vmap():
    dataset = FunctionalDataset(
        [
            ObservationBlock(
                ObservationDesignBlock(
                    FunctionalBlock(
                        Locations(jnp.zeros((2, 1))),
                        Identity(),
                        "values",
                    ),
                    variance=0.0,
                    max_size=2,
                ),
                y=jnp.ones(2),
            )
        ]
    )
    variance = PositiveParameter(1.0)
    gp = GaussianProcess(RBFKernel(variance=variance))

    def objective(raw):
        model = eqx.tree_at(
            lambda model: model.kernel.variance_parameter.unconstrained, gp, raw
        )
        return negative_log_marginal_likelihood(model, dataset)

    raw = jnp.stack(
        [
            variance.unconstrained,
            # At this scale the default jitter rounds away even in float64.
            PositiveParameter(1e12).unconstrained,
        ]
    )
    losses, gradients = jax.jit(jax.vmap(jax.value_and_grad(objective)))(raw)
    assert jnp.isfinite(losses[0])
    assert jnp.isfinite(gradients[0])
    assert not jnp.isfinite(losses[1])
    assert not jnp.isfinite(gradients[1])


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("operation", ["condition", "likelihood", "sample"])
def test_public_inference_rejects_invalid_jitter(compiled, operation):
    gp = GaussianProcess(RBFKernel())
    dataset = _dataset(jnp.array([0.1, 0.2, 0.3]))
    query = _query()

    def run(jitter):
        if operation == "condition":
            return gp.condition(dataset, jitter=jitter)
        if operation == "likelihood":
            return negative_log_marginal_likelihood(gp, dataset, jitter=jitter)
        return gp.sample_prior(jax.random.key(0), query, jitter=jitter)

    evaluate = eqx.filter_jit(run) if compiled else run
    for jitter in (-0.5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="jitter must be finite and nonnegative"):
            evaluate(jitter)


def _dataset(y: jax.Array) -> FunctionalDataset:
    locations = Locations(jnp.array([[-0.8], [0.0], [0.9]]))
    return FunctionalDataset(
        [
            ObservationBlock(
                ObservationDesignBlock(
                    FunctionalBlock(locations, Identity(), "values"),
                    variance=0.04,
                    max_size=locations.count,
                ),
                y=y,
            )
        ]
    )


def _query() -> FunctionalQuery:
    locations = Locations(jnp.array([[-0.4], [0.5]]))
    return FunctionalQuery(
        [
            FunctionalBlock(locations, Identity(), "values"),
            FunctionalBlock(locations, Gradient(), "gradients"),
        ]
    )


def test_conditioning_matches_direct_gaussian_formula() -> None:
    gp = GaussianProcess(RBFKernel(variance=1.3, lengthscale=0.7))
    dataset = _dataset(jnp.array([-0.2, 0.3, 0.8]))
    query = _query()
    posterior = gp.condition(dataset)
    prediction = posterior.predict(query)

    observation_points = dataset.blocks[0].functional.locations.points[:, 0]
    query_points = query.blocks[0].locations.points[:, 0]
    kernel = gp.kernel
    gram = jnp.array(
        [
            [kernel(jnp.array([x]), jnp.array([xp])) for xp in observation_points]
            for x in observation_points
        ]
    ) + (0.04 + 1e-6) * jnp.eye(3)
    cross = jnp.array(
        [
            [kernel(jnp.array([x]), jnp.array([xp])) for xp in observation_points]
            for x in query_points
        ]
    )
    prior = jnp.array(
        [
            [kernel(jnp.array([x]), jnp.array([xp])) for xp in query_points]
            for x in query_points
        ]
    )
    expected_mean = cross @ jnp.linalg.solve(gram, dataset.values_flat)
    expected_covariance = prior - cross @ jnp.linalg.solve(gram, cross.T)

    assert jnp.allclose(prediction.mean["values"], expected_mean, atol=2e-5)
    assert jnp.allclose(
        prediction.covariance.full[:2, :2], expected_covariance, atol=2e-5
    )


def test_tiled_gp_matches_dense_gp() -> None:
    kernel = RBFKernel(variance=1.3, lengthscale=0.7)
    dataset = _dataset(jnp.array([-0.2, 0.3, 0.8]))
    query = _query()

    dense = GaussianProcess(kernel).condition(dataset).predict(query)
    tiled = (
        GaussianProcess(kernel, covariance_tile_size=1)
        .condition(dataset)
        .predict(query)
    )

    assert jnp.allclose(tiled.mean.flat, dense.mean.flat, rtol=2e-5, atol=2e-5)
    assert jnp.allclose(
        tiled.covariance.full,
        dense.covariance.full,
        rtol=2e-5,
        atol=2e-5,
    )


@pytest.mark.parametrize("tile_size", [0, -1, True, 1.5])
def test_gp_covariance_tile_size_must_be_positive_integer(
    tile_size: Any,
) -> None:
    with pytest.raises(ValueError, match="covariance_tile_size"):
        GaussianProcess(RBFKernel(), covariance_tile_size=tile_size)


def test_posterior_covariance_is_independent_of_values() -> None:
    gp = GaussianProcess(RBFKernel())
    query = _query()
    first = gp.prepare_conditioning(_dataset(jnp.array([0.0, 1.0, 2.0])))
    second = gp.prepare_conditioning(_dataset(jnp.array([9.0, -4.0, 3.0])))

    assert jnp.allclose(
        first.predict_covariance(query).full,
        second.predict_covariance(query).full,
    )


def test_conditioning_append_solve_matches_complete_rebuild() -> None:
    first = ObservationDesignBlock(
        FunctionalBlock(Locations(jnp.array([[-0.7], [0.0]])), Identity(), "values"),
        variance=0.03,
        max_size=2,
    )
    second = ObservationDesignBlock(
        FunctionalBlock(Locations(jnp.array([[0.2], [0.9]])), Gradient(), "gradients"),
        variance=0.05,
        max_size=2,
    )
    gp = GaussianProcess(RBFKernel(lengthscale=0.6))
    conditioning = gp.prepare_conditioning(
        ObservationDesign(
            [first, second.replace(second.functional.locations.points[:0])]
        )
    )
    right_hand_side = jnp.array([-0.2, 0.4, 0.7, -0.1])
    solution = conditioning.solve(right_hand_side.at[first.functional.size :].set(0))
    updated, actual = eqx.filter_jit(
        lambda g, b, x, rhs: g.append_solve(
            b.functional.locations, x, rhs, block=b.functional.name
        )
    )(conditioning, second, solution, right_hand_side)
    rebuilt = gp.prepare_conditioning(ObservationDesign([first, second]))

    assert updated.index_map == rebuilt.index_map
    assert jnp.allclose(actual, rebuilt.solve(right_hand_side), atol=2e-5, rtol=2e-5)
    assert jnp.allclose(
        updated.predict_covariance(_query()).full,
        rebuilt.predict_covariance(_query()).full,
        atol=2e-5,
        rtol=2e-5,
    )


def test_streaming_append_matches_complete_rebuild() -> None:
    gp = GaussianProcess(
        RBFKernel(variance=1.2, lengthscale=0.6),
        LinearMean(jnp.array([0.4]), intercept=0.7),
    )
    first = ObservationBlock(
        ObservationDesignBlock(
            FunctionalBlock(Locations(jnp.array([[-0.8], [0.0]])), Identity(), "first"),
            variance=0.03,
            max_size=2,
        ),
        y=jnp.array([-0.4, 0.2]),
    )
    second = ObservationBlock(
        ObservationDesignBlock(
            FunctionalBlock(Locations(jnp.array([[0.3], [0.9]])), Gradient(), "second"),
            variance=0.05,
            max_size=2,
        ),
        y=jnp.array([[0.7], [-0.1]]),
    )
    updated = gp.condition(FunctionalDataset([first, _empty(second)])).append(
        second.functional.locations, second.y, block=second.functional.name
    )
    rebuilt = gp.condition(FunctionalDataset([first, second]))
    query = _query()

    assert jnp.allclose(
        updated.predict_mean(query).flat,
        rebuilt.predict_mean(query).flat,
        rtol=3e-5,
        atol=3e-5,
    )
    assert jnp.allclose(
        updated.predict_covariance(query).full,
        rebuilt.predict_covariance(query).full,
        rtol=3e-5,
        atol=3e-5,
    )
    assert updated.observation_size == first.functional.size + second.functional.size


def test_dense_streaming_append_does_not_resolve_complete_system(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gp = GaussianProcess(RBFKernel())
    first = ObservationBlock(
        ObservationDesignBlock(
            FunctionalBlock(
                Locations(jnp.array([[-0.5], [0.0]])), Identity(), "values"
            ),
            variance=0.03,
            max_size=2,
        ),
        y=jnp.array([-0.2, 0.1]),
    )
    second = ObservationBlock(
        ObservationDesignBlock(
            FunctionalBlock(Locations(jnp.array([[0.6]])), Identity(), "new_values"),
            variance=0.03,
            max_size=1,
        ),
        y=jnp.array([0.4]),
    )
    posterior = gp.condition(FunctionalDataset([first, _empty(second)]))

    def reject_complete_solve(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("complete posterior system was solved again")

    monkeypatch.setattr(ConditioningState, "solve", reject_complete_solve)

    updated = posterior.append(
        second.functional.locations, second.y, block=second.functional.name
    )

    assert updated.observation_size == 3
    assert jnp.all(jnp.isfinite(updated.alpha))


def test_dense_streaming_append_is_jittable_and_differentiable() -> None:
    first = ObservationBlock(
        ObservationDesignBlock(
            FunctionalBlock(Locations(jnp.array([[-0.8], [0.0]])), Identity(), "first"),
            variance=0.03,
            max_size=2,
        ),
        y=jnp.array([-0.4, 0.2]),
    )
    second = ObservationBlock(
        ObservationDesignBlock(
            FunctionalBlock(Locations(jnp.array([[0.3], [0.9]])), Gradient(), "second"),
            variance=0.05,
            max_size=2,
        ),
        y=jnp.array([[0.7], [-0.1]]),
    )
    query = _query()

    def streaming_objective(gp: GaussianProcess) -> jax.Array:
        posterior = gp.condition(FunctionalDataset([first, _empty(second)])).append(
            second.functional.locations, second.y, block=second.functional.name
        )
        return jnp.sum(posterior.predict_mean(query).flat)

    def rebuild_objective(gp: GaussianProcess) -> jax.Array:
        posterior = gp.condition(FunctionalDataset([first, second]))
        return jnp.sum(posterior.predict_mean(query).flat)

    gp = GaussianProcess(
        RBFKernel(
            variance=PositiveParameter(1.0, name="kernel_variance"),
            lengthscale=PositiveParameter(jnp.exp(-0.2), name="kernel_lengthscale"),
        )
    )
    streaming_value, streaming_gradient = eqx.filter_jit(
        eqx.filter_value_and_grad(streaming_objective)
    )(gp)
    rebuild_value, rebuild_gradient = eqx.filter_jit(
        eqx.filter_value_and_grad(rebuild_objective)
    )(gp)

    assert jnp.allclose(streaming_value, rebuild_value, rtol=3e-5, atol=3e-5)
    assert jnp.allclose(
        streaming_gradient.kernel.lengthscales.lengthscale_parameter.unconstrained,
        rebuild_gradient.kernel.lengthscales.lengthscale_parameter.unconstrained,
        rtol=5e-5,
        atol=5e-5,
    )


@pytest.mark.parametrize("compiled", [False, True])
def test_likelihood_and_map_objective_are_differentiable(compiled) -> None:
    dataset = _dataset(jnp.array([-0.2, 0.3, 0.8]))

    def objective(log_lengthscale: jax.Array) -> jax.Array:
        kernel = RBFKernel(
            variance=PositiveParameter(1.0, name="kernel_variance"),
            lengthscale=PositiveParameter(
                jnp.exp(log_lengthscale), name="kernel_lengthscale"
            ),
        )
        return negative_log_marginal_likelihood(GaussianProcess(kernel), dataset)

    value_and_grad = jax.value_and_grad(objective)
    if compiled:
        value_and_grad = jax.jit(value_and_grad)
    log_lengthscale = jnp.array(-0.2)
    value, gradient = value_and_grad(log_lengthscale)
    # Independent Gaussian formula and analytic derivative with respect to log ell.
    x = np.array([-0.8, 0.0, 0.9])
    squared_distance = (x[:, None] - x[None, :]) ** 2
    ell_squared = np.exp(2 * float(log_lengthscale))
    kernel_matrix = np.exp(-0.5 * squared_distance / ell_squared)
    covariance = kernel_matrix + (0.04 + 1e-6) * np.eye(3)
    y = np.asarray(dataset.values_flat, dtype=np.float64)
    alpha = np.linalg.solve(covariance, y)
    expected_value = 0.5 * (
        y @ alpha + np.linalg.slogdet(covariance)[1] + 3 * np.log(2 * np.pi)
    )
    expected_gradient = 0.5 * np.sum(
        (np.linalg.inv(covariance) - np.outer(alpha, alpha))
        * kernel_matrix
        * squared_distance
        / ell_squared
    )
    tolerance = 5e-6 if value.dtype == jnp.float32 else 1e-12
    np.testing.assert_allclose(value, expected_value, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(
        gradient, expected_gradient, rtol=tolerance, atol=tolerance
    )

    variance = PositiveParameter(
        1.0,
        prior=GammaPrior(concentration=2.0, rate=1.5),
        name="variance",
    )
    lengthscale = PositiveParameter(
        0.8,
        prior=LogNormalPrior(log_mean=0.0, log_standard_deviation=0.7),
        name="lengthscale",
    )
    observation_variance = PositiveParameter(
        0.04,
        prior=GammaPrior(concentration=2.0, rate=20.0),
        name="observation_variance",
    )
    block = dataset.blocks[0]
    dataset_with_priors = FunctionalDataset(
        [
            ObservationBlock(
                ObservationDesignBlock(
                    block.functional,
                    variance=observation_variance,
                    max_size=(block.functional).locations.count,
                ),
                y=block.y,
            )
        ]
    )
    gp = GaussianProcess(RBFKernel(variance=variance, lengthscale=lengthscale))
    nll = negative_log_marginal_likelihood(gp, dataset_with_priors)
    map_loss = negative_log_posterior(gp, dataset_with_priors)
    expected_log_prior = (
        variance.log_prior()
        + lengthscale.log_prior()
        + observation_variance.log_prior()
    )
    assert jnp.allclose(map_loss, nll - expected_log_prior)


@pytest.mark.parametrize("conditioned", [False, True], ids=["prior", "posterior"])
def test_samples_match_joint_gaussian_moments(conditioned) -> None:
    gp = GaussianProcess(RBFKernel(variance=1.5))
    points = jnp.array([[-0.2], [0.6]])
    locations = Locations(points)
    query = FunctionalQuery([FunctionalBlock(locations, Identity(), "values")])

    # Independent RBF moments, with one noisy reading for the posterior case.
    x = np.asarray(points[:, 0])
    mean = np.zeros(2)
    covariance = 1.5 * np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2)
    sample = gp.sample_prior
    if conditioned:
        dataset = FunctionalDataset([
            ObservationBlock.from_arrays(
                points=jnp.array([[0.1]]),
                y=jnp.array([1.2]),
                variance=0.2,
                name="observed",
                max_size=1,
            )
        ])
        sample = gp.condition(dataset, jitter=0.0).sample
        cross_covariance = 1.5 * np.exp(-0.5 * (x - 0.1) ** 2)
        mean = cross_covariance * 1.2 / (1.5 + 0.2)
        covariance -= np.outer(cross_covariance, cross_covariance) / (1.5 + 0.2)

    sample_count = 20_000
    samples = sample(jax.random.key(12), query, sample_count=sample_count, jitter=0.0)

    assert samples.flat.shape == (20_000, 2)
    assert samples["values"].shape == (20_000, 2)
    draws = np.asarray(samples.flat)
    # Five Gaussian sampling standard errors allow Monte Carlo variation in both
    # precisions, while rejecting lost correlations or added measurement noise.
    variances = np.diag(covariance)
    mean_error = np.sqrt(variances / sample_count)
    covariance_error = np.sqrt(
        (np.outer(variances, variances) + covariance**2) / (sample_count - 1)
    )
    np.testing.assert_array_less(np.abs(draws.mean(axis=0) - mean) / mean_error, 5)
    np.testing.assert_array_less(
        np.abs(np.cov(draws, rowvar=False) - covariance) / covariance_error, 5
    )
