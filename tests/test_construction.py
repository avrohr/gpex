"""Dynamic model construction and failures that must survive transformations."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from gpex.operators import (
    BoxAverage,
    DirectionalDerivative,
    GaussianSmoothing,
    Gradient,
    HaarWavelet1D,
    Identity,
    LinearCombination,
)
from gpex.parameters import CholeskyParameter, PositiveParameter


@pytest.mark.parametrize(
    ("parameter_type", "value"),
    [
        (PositiveParameter, jnp.array([0.4, 1.2])),
        (CholeskyParameter, jnp.array([[1.2, 0.0], [-0.3, 0.7]])),
    ],
)
def test_constrained_parameter_construction_supports_transformations(
    parameter_type, value
):
    parameter = eqx.filter_jit(parameter_type)(value)
    assert jnp.allclose(parameter.value, value)

    def construct(candidate):
        return jnp.sum(parameter_type(candidate).value ** 2)

    def update(candidate):
        updated = eqx.tree_at(
            lambda p: p.unconstrained,
            parameter,
            parameter.transform.inverse(candidate),
        )
        return jnp.sum(updated.value**2)

    actual = jax.jit(jax.value_and_grad(construct))(value)
    reference = jax.jit(jax.value_and_grad(update))(value)
    for result, expected in zip(actual, reference, strict=True):
        assert jnp.allclose(result, expected)
    batch = jnp.stack([value, 2.0 * value])
    assert jnp.allclose(
        jax.jit(jax.vmap(lambda x: parameter_type(x).value))(batch), batch
    )


@pytest.mark.parametrize("invalid", [-0.2, 0.0, jnp.inf, jnp.nan])
@pytest.mark.parametrize("execution", ["eager", "jit", "vmap"])
def test_positive_parameter_rejects_invalid_dynamic_values(invalid, execution):
    def evaluate(value):
        return PositiveParameter(value).value

    value = jnp.asarray(invalid)
    if execution == "jit":
        evaluate = jax.jit(evaluate)
    elif execution == "vmap":
        evaluate = jax.jit(jax.vmap(evaluate))
        value = jnp.array([1.0, invalid])
    with pytest.raises(Exception, match="finite values above zero"):
        jax.block_until_ready(evaluate(value))


@pytest.mark.parametrize(
    "invalid",
    [
        jnp.array([[1.0, 0.2], [0.0, 1.0]]),
        jnp.array([[1.0, 0.0], [0.0, -1.0]]),
        jnp.array([[1.0, 0.0], [jnp.inf, 1.0]]),
    ],
)
@pytest.mark.parametrize("execution", ["eager", "jit", "vmap"])
def test_cholesky_construction_does_not_discard_invalid_entries(invalid, execution):
    def evaluate(value):
        return CholeskyParameter(value).value

    value = invalid
    if execution == "jit":
        evaluate = jax.jit(evaluate)
    elif execution == "vmap":
        evaluate = jax.jit(jax.vmap(evaluate))
        value = jnp.stack([jnp.eye(2), invalid])
    with pytest.raises(Exception, match="finite and lower triangular"):
        jax.block_until_ready(evaluate(value))


@pytest.mark.parametrize(
    ("operator_type", "field"),
    [(BoxAverage, "radius"), (GaussianSmoothing, "scale"), (HaarWavelet1D, "scale")],
)
def test_nonlocal_operator_construction_supports_transformations(operator_type, field):
    scale = jnp.asarray(0.7)
    point = jnp.array([0.3])
    operator = eqx.filter_jit(operator_type)(scale)
    assert jnp.array_equal(getattr(operator, field), scale)

    def function(x):
        return x[0] + x[0] ** 2

    def construct(value):
        return operator_type(value).apply(function, point)

    def update(value):
        return eqx.tree_at(lambda op: getattr(op, field), operator, value).apply(
            function, point
        )

    actual = jax.jit(jax.value_and_grad(construct))(scale)
    reference = jax.jit(jax.value_and_grad(update))(scale)
    for result, expected in zip(actual, reference, strict=True):
        assert jnp.allclose(result, expected)
    values = jnp.array([0.3, 0.7])
    assert jnp.allclose(
        jax.jit(jax.vmap(construct))(values),
        jnp.stack([construct(value) for value in values]),
    )


@pytest.mark.parametrize(
    "operator_type", [BoxAverage, GaussianSmoothing, HaarWavelet1D]
)
@pytest.mark.parametrize("invalid", [-0.2, jnp.inf, jnp.nan])
@pytest.mark.parametrize("execution", ["eager", "jit", "vmap"])
def test_nonlocal_operators_reject_invalid_dynamic_values(
    operator_type, invalid, execution
):
    def evaluate(value):
        return operator_type(value).apply(lambda x: x[0] ** 2, jnp.array([0.3]))

    value = jnp.asarray(invalid)
    if execution == "jit":
        evaluate = jax.jit(evaluate)
    elif execution == "vmap":
        evaluate = jax.jit(jax.vmap(evaluate))
        value = jnp.array([0.5, invalid])
    with pytest.raises(Exception, match="finite and strictly positive"):
        jax.block_until_ready(evaluate(value))


def test_operator_shape_errors_reach_adequate_natural_exceptions():
    point = jnp.ones(2)

    def function(x):
        return jnp.sum(x**2)

    with pytest.raises((TypeError, ValueError)):
        DirectionalDerivative(jnp.ones(1)).apply(function, point)
    with pytest.raises(ValueError):
        LinearCombination((Identity(), Gradient()), jnp.ones(2)).apply(function, point)
