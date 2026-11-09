from math import log, pi, sqrt
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

from gpex import (
    FunctionalBlock,
    FunctionalDataset,
    GaussianProcess,
    ObservationBlock,
    ObservationDesignBlock,
)
from gpex.blocks import Locations
from gpex.fitting import (
    fit,
    fit_lbfgs,
    fit_multistart,
    fit_multistart_random_subsets,
    fit_random_subsets,
    negative_log_marginal_likelihood,
)
from gpex.kernels import GeneralizedRBFKernel, RBFKernel
from gpex.operators import Gradient, Identity
from gpex.parameters import (
    CholeskyParameter,
    Parameter,
    PositiveParameter,
    log_prior,
)
from gpex.parameters.priors import GammaPrior


def _quadratic(parameter):
    return 0.5 * parameter.value**2


@pytest.mark.parametrize("start_count", [None, 1, 3])
def test_converged_lbfgs_only_evaluates_initial_objective(start_count):
    evaluations = []

    def objective(parameter):
        jax.debug.callback(evaluations.append, parameter.value)
        return _quadratic(parameter)

    def fit_one(model):
        return fit_lbfgs(model, objective, max_steps=8)

    if start_count is None:
        result = fit_one(Parameter(unconstrained=0.0))
        assert result.step_count == 0
        assert result.evaluation_count == 1
        assert result.losses[0] == 0
        assert jnp.all(jnp.isnan(result.losses[1:]))
    else:
        result = fit_multistart([Parameter(unconstrained=0.0)] * start_count, fit_one)
        assert jnp.all(result.step_counts == 0)
        assert jnp.all(result.evaluation_counts == 1)
        assert jnp.all(result.losses[:, 0] == 0)
        assert jnp.all(jnp.isnan(result.losses[:, 1:]))
    jax.block_until_ready(result.losses)
    jax.effects_barrier()
    assert len(evaluations) == (1 if start_count is None else start_count)
    assert jnp.all(result.converged)


@pytest.mark.parametrize("multistart", [False, True])
def test_lbfgs_convergence_stops_objective_work_with_unused_steps(multistart):
    evaluations = []

    def objective(parameter):
        jax.debug.callback(evaluations.append, parameter.value)
        return _quadratic(parameter)

    counts = []
    for max_steps in (8, 16):
        evaluations.clear()
        reported_evaluations = None
        if multistart:
            result = fit_multistart(
                [Parameter(unconstrained=x) for x in (0.0, 1.0, 4.0)],
                lambda model, limit=max_steps: fit_lbfgs(
                    model, objective, max_steps=limit
                ),
            )
            assert result.step_counts[0] == 0
            assert jnp.all((result.step_counts[1:] > 0) & (result.step_counts[1:] < 8))
            # Staggered convergence exercises masking of already-finished starts.
            assert result.step_counts[1] != result.step_counts[2]
        else:
            result = fit_lbfgs(
                Parameter(unconstrained=4.0), objective, max_steps=max_steps
            )
            assert 0 < result.step_count < 8
            reported_evaluations = result.evaluation_count
        jax.block_until_ready(result.losses)
        jax.effects_barrier()
        counts.append(len(evaluations))
        assert jnp.all(result.converged)
        if reported_evaluations is not None:
            assert reported_evaluations == len(evaluations)
    assert counts[0] == counts[1]


def test_fixed_step_fit_uses_full_budget_even_at_optimum():
    evaluations = []

    def objective(parameter):
        jax.debug.callback(evaluations.append, parameter.value)
        return _quadratic(parameter)

    result = fit(Parameter(unconstrained=0.0), objective, optax.adam(0.1), step_count=8)
    jax.block_until_ready(result)
    jax.effects_barrier()
    assert len(evaluations) == 9
    assert result.step_count == 8
    assert result.evaluation_count == 9
    assert jnp.all(result.losses == 0)
    assert not result.converged


def test_fixed_step_fit_returns_final_iterate_even_after_overshoot():
    result = fit(Parameter(unconstrained=1.0), _quadratic, optax.sgd(3.0), step_count=2)
    assert result.model.value == 4.0
    assert result.loss == result.losses[-1] == 8.0


@pytest.mark.parametrize("exponent", [None, -0.65, 0.65], ids=["zero", "tiny", "huge"])
def test_finite_gradients_have_stable_norms_and_allow_sgd_updates(exponent, float_info):
    # The entries and update stay representable, but unscaled squares do not.
    limit = float(float_info.max)
    gradient_scale = 0.0 if exponent is None else limit**exponent
    scale = jnp.asarray(gradient_scale)
    model = (
        Parameter(unconstrained=jnp.asarray(0.0)),
        Parameter(unconstrained=jnp.zeros(2)),
    )

    def objective(candidate):
        scalar, vector = candidate
        return scale * (3.0 * scalar.value - 4.0 * vector.value[1])

    learning_rate = 1.0 if gradient_scale == 0.0 else 1.0 / gradient_scale
    result = fit(model, objective, optax.sgd(learning_rate), step_count=1)

    expected_scalar = 0.0 if gradient_scale == 0.0 else -3.0
    expected_vector = 0.0 if gradient_scale == 0.0 else 4.0
    assert result.model[0].value == pytest.approx(expected_scalar)
    assert result.model[1].value.tolist() == pytest.approx([0.0, expected_vector])
    assert result.loss == pytest.approx(-25.0 * gradient_scale, rel=1e-6, abs=0.0)

    assert result.gradient_norm == pytest.approx(
        5.0 * gradient_scale, rel=1e-6, abs=0.0
    )


def test_finite_gradients_are_accepted_when_norm_overflows(float_info):
    model = Parameter(unconstrained=jnp.zeros(2))
    scale = 0.85 * float(float_info.max)

    def objective(parameter):
        return scale * jnp.sum(parameter.value)

    # Keep the iterate fixed to isolate validation from optimizer arithmetic.
    result = fit(model, objective, optax.sgd(0.0), step_count=1)

    assert result.loss == 0.0
    assert jnp.array_equal(result.model.value, model.value)
    assert jnp.isposinf(result.gradient_norm)


def test_optax_fit_is_reverse_mode_differentiable():
    def fitted_value(initial):
        return fit(
            Parameter(unconstrained=initial), _quadratic, optax.sgd(0.1), step_count=8
        ).model.value

    assert fitted_value(1.0) == pytest.approx(0.9**8)
    assert jax.grad(fitted_value)(1.0) == pytest.approx(0.9**8)


@pytest.mark.parametrize("lbfgs", [False, True])
def test_complete_fit_result_crosses_outer_jit_and_keeps_raw_arrays_fixed(lbfgs):
    def objective(model):
        return (model[0].value - model[1]) ** 2

    @eqx.filter_jit
    def run(model):
        if lbfgs:
            return fit_lbfgs(model, objective, max_steps=8)
        return fit(model, objective, optax.adam(0.1), step_count=8)

    result = run((Parameter(unconstrained=4.0), jnp.asarray(2.0)))
    assert result.model[1] == 2.0
    assert result.loss == pytest.approx(float(objective(result.model)))


@pytest.mark.parametrize("lbfgs", [False, True])
@pytest.mark.parametrize("multistart", [False, True])
def test_nonfinite_initial_gradient_raises(lbfgs, multistart):
    def objective(parameter):
        return jnp.sqrt(parameter.value)

    def fit_one(model):
        if lbfgs:
            return fit_lbfgs(model, objective, max_steps=2)
        return fit(model, objective, optax.sgd(0.01), step_count=2)

    with pytest.raises(eqx.EquinoxRuntimeError, match="non-finite"):
        if multistart:
            fit_multistart(
                [Parameter(unconstrained=0.0), Parameter(unconstrained=4.0)], fit_one
            )
        else:
            fit_one(Parameter(unconstrained=0.0))


def test_nonfinite_update_raises_instead_of_returning_previous_model():
    with pytest.raises(eqx.EquinoxRuntimeError, match="non-finite"):
        fit(
            Parameter(unconstrained=1.0),
            lambda p: jnp.sqrt(p.value),
            optax.sgd(10.0),
            step_count=2,
        )


@pytest.mark.parametrize("vectorized", [False, True])
@pytest.mark.parametrize(
    "output", ["parameter", "loss", "losses", "gradient_norm", "optimizer_state"]
)
def test_optax_fit_remembers_invalid_intermediate_gradient(output, vectorized):
    # Unit updates ignore the gradient: -1 -> 0 -> 1 has a finite final state,
    # but sqrt(x**2) has a non-finite autodiff gradient at the intermediate zero.
    def update(updates, state, params=None):
        del params
        return jax.tree.map(jnp.ones_like, updates), state + 1

    optimizer = optax.GradientTransformation(lambda params: jnp.asarray(0), update)

    def one(initial):
        result = fit(
            Parameter(unconstrained=initial),
            lambda p: jnp.sqrt(p.value**2),
            optimizer,
            step_count=2,
        )
        return result.model.value if output == "parameter" else getattr(result, output)

    run = eqx.filter_jit(jax.vmap(one) if vectorized else one)
    valid = run(jnp.array([2.0, 3.0]) if vectorized else jnp.asarray(2.0))
    assert jnp.all(jnp.isfinite(valid))
    with pytest.raises(eqx.EquinoxRuntimeError, match="non-finite"):
        jax.block_until_ready(
            run(jnp.array([-1.0, 2.0]) if vectorized else jnp.asarray(-1.0))
        )


def test_lbfgs_iteration_limit_does_not_claim_convergence():
    result = fit_lbfgs(Parameter(unconstrained=4.0), _quadratic, max_steps=1)
    assert result.step_count == 1
    assert not result.converged
    assert result.loss == result.losses[-1]


def test_lbfgs_reports_failed_line_search():
    with pytest.raises(eqx.EquinoxRuntimeError, match="line search failed"):
        fit_lbfgs(
            Parameter(unconstrained=1.0),
            lambda p: -p.value,
            max_linesearch_steps=1,
            max_steps=2,
        )


def test_lbfgs_can_reject_nonfinite_trials_and_accept_a_valid_step():
    trials = []

    def objective(parameter):
        jax.debug.callback(trials.append, parameter.value)
        return -jnp.log(parameter.value) + 100.0 * parameter.value

    result = fit_lbfgs(Parameter(unconstrained=1.0), objective, gradient_tolerance=1e-3)
    jax.block_until_ready(result)
    jax.effects_barrier()
    assert min(trials) <= 0.0  # Outside the logarithm's domain.
    assert result.model.value == pytest.approx(0.01, abs=1e-6)
    assert result.converged


@pytest.mark.parametrize("lbfgs", [False, True])
def test_nonfinite_check_survives_jit_when_only_loss_is_returned(lbfgs):
    def objective(parameter):
        return 0.0 * parameter.value + jnp.nan

    @eqx.filter_jit
    def run(model):
        if lbfgs:
            return fit_lbfgs(model, objective).loss
        return fit(model, objective, optax.sgd(0.1), step_count=1).loss

    with pytest.raises(eqx.EquinoxRuntimeError, match="non-finite"):
        run(Parameter(unconstrained=1.0))


@pytest.mark.parametrize("lbfgs", [False, True])
@pytest.mark.parametrize("start_count", [1, 3])
def test_multistart_matches_independent_fits(lbfgs, start_count):
    starts = tuple(Parameter(unconstrained=x) for x in (4.0, 0.0, -2.0)[:start_count])

    def fit_one(model):
        if lbfgs:
            return fit_lbfgs(model, _quadratic, max_steps=8)
        return fit(
            model,
            _quadratic,
            optax.chain(optax.clip_by_global_norm(2.0), optax.adam(0.1)),
            step_count=8,
        )

    result = fit_multistart(starts, fit_one)
    separate = [fit_one(start) for start in starts]
    assert result.losses.shape == (start_count, 9)
    assert jnp.allclose(
        result.losses, jnp.stack([r.losses for r in separate]), equal_nan=True
    )
    assert jnp.allclose(result.final_losses, jnp.stack([r.loss for r in separate]))
    assert result.loss == jnp.min(result.final_losses)
    for index, single in enumerate(separate):
        assert jnp.allclose(result.models[index].value, single.model.value)
        assert result.step_counts[index] == single.step_count
        assert result.evaluation_counts[index] == single.evaluation_count
        for batched, unbatched in zip(
            jax.tree_util.tree_leaves(result.optimizer_state),
            jax.tree_util.tree_leaves(single.optimizer_state),
            strict=True,
        ):
            assert jnp.allclose(batched[index], unbatched, equal_nan=True)


@pytest.mark.parametrize("invalid", [0, -1, True, 1.5])
def test_fit_validates_iteration_budget(invalid):
    with pytest.raises((TypeError, ValueError), match="step_count"):
        fit(
            Parameter(unconstrained=1.0), _quadratic, optax.sgd(0.1), step_count=invalid
        )
    with pytest.raises((TypeError, ValueError), match="max_steps"):
        fit_lbfgs(Parameter(unconstrained=1.0), _quadratic, max_steps=invalid)


@pytest.mark.parametrize("invalid", [-1.0, float("nan"), float("inf")])
def test_lbfgs_validates_tolerance(invalid):
    with pytest.raises(ValueError, match="gradient_tolerance"):
        fit_lbfgs(Parameter(unconstrained=1.0), _quadratic, gradient_tolerance=invalid)


def test_fit_requires_explicit_parameters():
    with pytest.raises(ValueError, match="explicit Parameter"):
        fit(jnp.asarray(1.0), lambda x: x**2, optax.sgd(0.1), step_count=2)


def test_multistart_validates_starts():
    def fit_one(model):
        return fit(model, lambda m: m[0].value ** 2, optax.sgd(0.1), step_count=2)

    with pytest.raises(ValueError, match="at least one start"):
        fit_multistart([], fit_one)
    with pytest.raises(ValueError, match="fixed leaves"):
        fit_multistart(
            [(Parameter(unconstrained=1.0), 1.0), (Parameter(unconstrained=2.0), 2.0)],
            fit_one,
        )
    with pytest.raises(ValueError, match="parameter shapes"):
        fit_multistart(
            [
                (Parameter(unconstrained=jnp.zeros(1)), 1.0),
                (Parameter(unconstrained=jnp.zeros(2)), 1.0),
            ],
            fit_one,
        )


def _data() -> FunctionalDataset:
    points = jnp.linspace(-1.0, 1.0, 8)[:, None]
    return FunctionalDataset(
        [
            ObservationBlock(
                ObservationDesignBlock(
                    FunctionalBlock(Locations(points), Identity(), "values"),
                    variance=0.02,
                    max_size=points.shape[0],
                ),
                y=jnp.sin(2.0 * points[:, 0]),
            )
        ]
    )


def test_fit_can_optimize_explicit_kernel_and_variance_parameters_together() -> None:
    original = _data()
    block = original.blocks[0]
    trainable_data = FunctionalDataset(
        [
            ObservationBlock(
                ObservationDesignBlock(
                    block.functional,
                    variance=PositiveParameter(
                        0.2, prior=GammaPrior(2.0, 10.0), name="observation_variance"
                    ),
                    max_size=(block.functional).locations.count,
                ),
                y=block.y,
            )
        ]
    )
    gp = GaussianProcess(
        RBFKernel(
            variance=PositiveParameter(0.4, name="variance"),
            lengthscale=PositiveParameter(1.8, name="lengthscale"),
        )
    )
    initial = (gp, trainable_data)

    def objective(model: tuple[GaussianProcess, FunctionalDataset]) -> jnp.ndarray:
        model_gp, model_data = model
        return negative_log_marginal_likelihood(model_gp, model_data)

    result = fit(initial, objective, optax.adam(0.02), step_count=5)
    fitted_gp, fitted_data = result.model

    assert result.losses[-1] < result.losses[0]
    assert jnp.array_equal(fitted_data.values_flat, trainable_data.values_flat)
    assert not jnp.allclose(
        fitted_gp.kernel.lengthscales.lengthscale,
        gp.kernel.lengthscales.lengthscale,
    )
    assert not jnp.allclose(fitted_data.variance_flat, trainable_data.variance_flat)


def _random_subset_start(variance: float = 1.0):
    return (
        GaussianProcess(
            RBFKernel(variance=PositiveParameter(variance, name="variance"))
        ),
        _data(),
    )


def test_random_subsets_are_fresh_and_reproducible() -> None:
    model = _random_subset_start()

    def subset_objective(candidate):
        gp, selected = candidate
        return (
            100.0 * selected.values_flat.size
            + jnp.mean(selected.values_flat)
            + 0.0 * gp.kernel.variance
        )

    def reference_objective(candidate):
        gp, data = candidate
        return jnp.mean(data.values_flat) + 0.0 * gp.kernel.variance

    def run(key):
        return fit_random_subsets(
            model,
            subset_objective,
            subset_size=3,
            reference_objective=reference_objective,
            key=key,
            optimizer=optax.adam(0.01),
            step_count=8,
        )

    first = run(jax.random.key(4))
    repeated = run(jax.random.key(4))
    changed = run(jax.random.key(5))

    assert jnp.array_equal(first.subset_losses, repeated.subset_losses)
    assert not jnp.array_equal(first.subset_losses, changed.subset_losses)
    assert jnp.unique(first.subset_losses).size > 1
    assert jnp.all((first.subset_losses > 299.0) & (first.subset_losses < 301.0))
    assert first.evaluation_count == 9


def test_random_subset_supports_exact_gp_objective_and_fixed_variance() -> None:
    model = _random_subset_start()

    def objective(candidate):
        gp, data = candidate
        return negative_log_marginal_likelihood(gp, data)

    result = fit_random_subsets(
        model,
        objective,
        subset_size=4,
        reference_objective=objective,
        key=jax.random.key(8),
        optimizer=optax.adam(0.01),
        step_count=3,
    )

    assert result.subset_losses.shape == (3,)
    assert jnp.isfinite(result.reference_loss)


def test_random_subset_size_can_be_set_per_observation_block() -> None:
    first = _data().blocks[0]
    second_points = jnp.linspace(-0.8, 0.8, 5)[:, None]
    data = FunctionalDataset(
        [
            first,
            ObservationBlock(
                ObservationDesignBlock(
                    FunctionalBlock(Locations(second_points), Identity(), "secondary"),
                    variance=0.03,
                    max_size=second_points.shape[0],
                ),
                y=jnp.cos(second_points[:, 0]),
            ),
        ]
    )
    model = (_random_subset_start()[0], data)

    def subset_objective(candidate):
        gp, selected = candidate
        first_size, second_size = (
            block.functional.locations.count for block in selected.blocks
        )
        return first_size + 10.0 * second_size + 0.0 * gp.kernel.variance

    result = fit_random_subsets(
        model,
        subset_objective,
        subset_size={"values": 3, "secondary": 2},
        reference_objective=lambda candidate: candidate[0].kernel.variance,
        key=jax.random.key(9),
        optimizer=optax.adam(0.01),
        step_count=2,
    )

    assert jnp.array_equal(result.subset_losses, jnp.array([23.0, 23.0]))


def test_multistart_random_subsets_split_keys_and_use_reference_loss() -> None:
    starts = (_random_subset_start(0.5), _random_subset_start(0.5))

    def subset_objective(candidate):
        gp, selected = candidate
        return (gp.kernel.variance - jnp.mean(selected.values_flat)) ** 2

    def reference_objective(candidate):
        gp, data = candidate
        return (gp.kernel.variance - jnp.mean(data.values_flat)) ** 2

    result = fit_multistart_random_subsets(
        starts,
        subset_objective,
        subset_size=3,
        reference_objective=reference_objective,
        key=jax.random.key(3),
        optimizer=optax.adam(0.01),
        step_count=12,
    )

    assert not jnp.array_equal(result.subset_losses[0], result.subset_losses[1])
    assert result.loss == jnp.min(result.reference_losses)
    assert eqx.tree_equal(result.model, result.models[result.best_index])
    for index, key in enumerate(jax.random.split(jax.random.key(3), 2)):
        single = fit_random_subsets(
            starts[index],
            subset_objective,
            optax.adam(0.01),
            step_count=12,
            subset_size=3,
            reference_objective=reference_objective,
            key=key,
        )
        assert jnp.allclose(result.subset_losses[index], single.subset_losses)
        assert jnp.allclose(result.reference_losses[index], single.reference_loss)


def test_random_subsets_keep_locations_values_and_variance_aligned():
    x = jnp.arange(8.0)
    dataset = FunctionalDataset(
        [
            ObservationBlock(
                ObservationDesignBlock(
                    FunctionalBlock(Locations(x[:, None]), Identity(), "values"),
                    variance=0.1,
                    max_size=(x[:, None]).shape[0],
                ),
                y=2.0 * x,
            )
        ]
    )

    def objective(candidate):
        gp, selected = candidate
        block = selected.blocks[0]
        locations = block.functional.locations.points[:, 0]
        return (
            jnp.sum((selected.values_flat - 2.0 * locations) ** 2)
            + jnp.sum((selected.variance_flat - 0.1) ** 2)
            + 0.0 * gp.kernel.variance
        )

    result = fit_random_subsets(
        (_random_subset_start()[0], dataset),
        objective,
        optax.sgd(0.01),
        step_count=5,
        subset_size=3,
        reference_objective=objective,
        key=jax.random.key(3),
    )
    assert jnp.all(result.subset_losses == 0.0)
    assert result.reference_loss == 0.0
    assert jnp.array_equal(result.model[1].values_flat, dataset.values_flat)


def test_random_subsets_preserve_shared_variance_gradients_and_prior():
    x = jnp.arange(4.0)
    component_offsets = jnp.array([1.0, 2.0])
    offsets = x[:, None] + component_offsets
    prior = GammaPrior(2.0, 3.0)
    parameter = PositiveParameter(0.3, prior=prior, name="observation_variance")
    dataset = FunctionalDataset(
        [
            ObservationBlock(
                ObservationDesignBlock(
                    FunctionalBlock(
                        Locations(jnp.stack([x, x], axis=1)), Gradient(), "gradients"
                    ),
                    parameter,
                    max_size=(jnp.stack([x, x], axis=1)).shape[0],
                ),
                y=2.0 * offsets,
            )
        ]
    )

    def objective(candidate):
        _, selected = candidate
        block = selected.blocks[0]
        expected_offsets = block.functional.locations.points[:, :1] + component_offsets
        alignment_error = jnp.sum((block.y - 2.0 * expected_offsets) ** 2)
        return (
            alignment_error
            + jnp.sum(block.design.variances)
            - log_prior(selected.design, require_all=True)
        )

    # Two locations with two outputs: the loss contains four copies of the
    # shared variance and one prior on the original parameter.
    def expected_loss(raw):
        value = jax.nn.softplus(raw)
        return 4.0 * value - prior.log_prob(value)

    initial_loss, gradient = jax.value_and_grad(expected_loss)(parameter.unconstrained)
    expected_raw = parameter.unconstrained - 0.01 * gradient

    @eqx.filter_jit
    def run(model):
        return fit_random_subsets(
            model,
            objective,
            optax.sgd(0.01),
            step_count=1,
            subset_size=2,
            reference_objective=objective,
            key=jax.random.key(3),
        )

    result = run((GaussianProcess(RBFKernel()), dataset))
    fitted_variance = result.model[1].blocks[0].design.variance
    expected_variance = jax.nn.softplus(expected_raw)
    assert isinstance(fitted_variance, PositiveParameter)
    assert jnp.allclose(fitted_variance.unconstrained, expected_raw)
    assert jnp.allclose(result.subset_losses[0], initial_loss)
    assert jnp.allclose(
        result.reference_loss,
        8.0 * expected_variance - prior.log_prob(expected_variance),
    )
    assert jnp.allclose(
        result.model[1].variance_flat.reshape(4, 2), jnp.full((4, 2), expected_variance)
    )


@pytest.mark.parametrize("invalid_reference", [False, True])
def test_random_subsets_raise_on_invalid_objectives(invalid_reference):
    def valid(candidate):
        return candidate[0].kernel.variance

    def invalid(candidate):
        return jnp.sqrt(0.0 * candidate[0].kernel.variance)

    with pytest.raises(eqx.EquinoxRuntimeError, match="non-finite"):
        fit_random_subsets(
            _random_subset_start(),
            valid if invalid_reference else invalid,
            optax.adam(0.01),
            step_count=2,
            subset_size=3,
            reference_objective=invalid if invalid_reference else valid,
            key=jax.random.key(3),
        )


@pytest.mark.parametrize("output", ["subset_losses", "reference_loss"])
def test_random_subsets_reject_nonfinite_updates_when_only_losses_are_used(output):
    def constant_objective(model):
        return jnp.asarray(0.0)

    @eqx.filter_jit
    def run(model):
        result = fit_random_subsets(
            model,
            constant_objective,
            optax.sgd(float("inf")),
            step_count=1,
            subset_size=3,
            reference_objective=constant_objective,
            key=jax.random.key(3),
        )
        return getattr(result, output)

    with pytest.raises(eqx.EquinoxRuntimeError, match="non-finite"):
        jax.block_until_ready(run(_random_subset_start()))


@pytest.mark.parametrize("vectorized", [False, True])
def test_random_subset_final_reference_check_survives_partial_results(vectorized):
    def reference(candidate):
        return jnp.sqrt(0.0 * candidate[0].kernel.variance)

    @eqx.filter_jit
    def run(model):
        def one(key):
            result = fit_random_subsets(
                model,
                lambda candidate: candidate[0].kernel.variance,
                optax.sgd(0.01),
                step_count=1,
                subset_size=3,
                reference_objective=reference,
                key=key,
            )
            return result.subset_losses

        key = jax.random.key(3)
        return jax.vmap(one)(jax.random.split(key, 2)) if vectorized else one(key)

    with pytest.raises(eqx.EquinoxRuntimeError, match="non-finite"):
        jax.block_until_ready(run(_random_subset_start()))


def test_random_subsets_validate_model_and_size() -> None:
    model = _random_subset_start()

    def objective(candidate):
        gp, data = candidate
        return gp.kernel.variance + 0.0 * jnp.mean(data.values_flat)

    def fit_with_size(size):
        return fit_random_subsets(
            model,
            objective,
            optax.adam(0.01),
            step_count=2,
            subset_size=size,
            reference_objective=objective,
            key=jax.random.key(1),
        )

    # Python considers True == 1; validate before the JIT cache can equate them.
    fit_with_size(1)
    with pytest.raises(TypeError, match="integers"):
        fit_with_size(True)

    with pytest.raises(ValueError, match="between 1 and 8"):
        fit_random_subsets(
            model,
            objective,
            subset_size=9,
            reference_objective=objective,
            key=jax.random.key(1),
            optimizer=optax.adam(0.01),
            step_count=2,
        )

    with pytest.raises(ValueError, match="every observation block"):
        fit_random_subsets(
            model,
            objective,
            subset_size={"missing": 2},
            reference_objective=objective,
            key=jax.random.key(1),
            optimizer=optax.adam(0.01),
            step_count=2,
        )

    with pytest.raises(TypeError, match=r"\(gp, dataset\)"):
        fit_random_subsets(
            cast(Any, Parameter(unconstrained=1.0)),
            cast(Any, objective),
            subset_size=1,
            reference_objective=cast(Any, objective),
            key=jax.random.key(1),
            optimizer=optax.adam(0.01),
            step_count=2,
        )


def test_fit_updates_full_mahalanobis_metric_parameters() -> None:
    points = jnp.array([[-1.0, -0.5], [-0.6, 0.8], [0.0, 0.1], [0.5, -0.7], [0.9, 0.6]])
    values = jnp.sin(points[:, 0]) + 0.4 * points[:, 1]
    data = FunctionalDataset(
        [
            ObservationBlock(
                ObservationDesignBlock(
                    FunctionalBlock(Locations(points), Identity(), "values"),
                    variance=0.04,
                    max_size=points.shape[0],
                ),
                y=values,
            )
        ]
    )
    L = CholeskyParameter(jnp.array([[0.8, 0.0], [0.2, 1.3]]), name="L")
    initial = GaussianProcess(GeneralizedRBFKernel(L=L))

    result = fit(
        initial,
        lambda gp: negative_log_marginal_likelihood(gp, data),
        optax.adam(0.02),
        step_count=4,
    )

    fitted_L = result.model.kernel.lengthscales.L
    assert result.losses[-1] < result.losses[0]
    assert not jnp.allclose(jnp.diag(fitted_L), jnp.diag(L.value))
    assert not jnp.allclose(fitted_L[1, 0], L.value[1, 0])
    assert jnp.all(jnp.diag(fitted_L) > 0)
    assert jnp.all(jnp.triu(fitted_L, 1) == 0)


def test_lbfgs_converges_and_reports_diagnostics() -> None:
    initial = Parameter(unconstrained=jnp.array([7.0, -3.0]), name="point")

    result = fit_lbfgs(
        initial,
        lambda parameter: jnp.sum((parameter.value - 2.0) ** 2),
        max_steps=20,
    )

    assert jnp.allclose(result.model.value, 2.0, atol=1e-5)
    assert result.loss < 1e-8
    assert result.gradient_norm < 1e-4
    assert result.step_count < 20
    assert result.evaluation_count >= result.step_count + 1
    assert result.converged


def test_multistart_lbfgs_recovers_known_gp_optimum() -> None:
    # This covariance stays well-conditioned near its optimum.
    # For y = [1, 2], the sum/difference eigenvectors give optimal covariance
    # eigenvalues 4.5 and 0.5: diagonal 2.5, off-diagonal 2.0.
    observation_variance = 0.1
    data = FunctionalDataset(
        [
            ObservationBlock(
                ObservationDesignBlock(
                    FunctionalBlock(
                        Locations(jnp.array([[0.0], [1.0]])), Identity(), "values"
                    ),
                    variance=observation_variance,
                    max_size=2,
                ),
                y=jnp.array([1.0, 2.0]),
            )
        ]
    )
    expected_variance = 2.5 - observation_variance
    expected_lengthscale = 1.0 / sqrt(2.0 * log(expected_variance / 2.0))
    expected_loss = 1.0 + log(2.0 * pi) + 0.5 * log(4.5 * 0.5)
    starts = tuple(
        GaussianProcess(
            RBFKernel(
                variance=PositiveParameter(variance, name="variance"),
                lengthscale=PositiveParameter(lengthscale, name="lengthscale"),
            )
        )
        for variance, lengthscale in ((0.5, 0.5), (1.0, 1.5), (3.0, 0.7))
    )

    def objective(gp: GaussianProcess) -> jax.Array:
        return negative_log_marginal_likelihood(gp, data, jitter=0.0)

    initial_losses = jnp.stack([objective(start) for start in starts])
    gradient_tolerance = 1e-5
    result = fit_multistart(
        starts,
        lambda model: fit_lbfgs(
            model, objective, max_steps=30, gradient_tolerance=gradient_tolerance
        ),
    )

    assert jnp.all(result.converged)
    assert jnp.all(result.gradient_norms <= gradient_tolerance)
    assert jnp.all(result.final_losses < initial_losses)
    assert result.loss == jnp.min(result.final_losses)
    assert result.final_losses.tolist() == pytest.approx(
        [expected_loss] * len(starts), abs=1e-6
    )
    for model in result.models:
        assert float(model.kernel.variance) == pytest.approx(
            expected_variance, rel=1e-4
        )
        assert float(model.kernel.lengthscales.lengthscale) == pytest.approx(
            expected_lengthscale, rel=1e-4
        )
