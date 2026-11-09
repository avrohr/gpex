import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from gpex import (
    FunctionalBlock,
    FunctionalDataset,
    FunctionalQuery,
    ObservationBlock,
    ObservationDesignBlock,
)
from gpex.blocks import IndexMap, Locations
from gpex.operators import DirectionalDerivative, Gradient, Identity
from gpex.parameters import Parameter, PositiveParameter, log_prior
from gpex.parameters.priors import GammaPrior


@pytest.mark.parametrize("compiled", [False, True])
def test_fixed_variance_requires_one_finite_nonnegative_scalar(compiled):
    functional = FunctionalBlock(Locations(jnp.zeros((2, 2))), Gradient(), "gradient")

    def construct(value):
        return ObservationDesignBlock(
            functional, value, max_size=functional.locations.count
        )

    operation = eqx.filter_jit(construct) if compiled else construct
    error = eqx.EquinoxRuntimeError if compiled else ValueError
    for value in (-0.5, float("nan"), float("inf"), -float("inf")):
        with pytest.raises(error, match="finite and nonnegative"):
            jax.block_until_ready(operation(jnp.asarray(value)).variances)
    for value in (0.0, 0.5):
        assert jnp.array_equal(operation(value).variances, jnp.full((2, 2), value))
    for value in (jnp.ones(1), jnp.ones(2), jnp.ones((2, 1))):
        with pytest.raises(ValueError, match="must be scalar"):
            operation(value)


def test_trainable_variance_uses_the_positive_parameter_contract():
    functional = FunctionalBlock(jnp.zeros((2, 2)), Gradient(), "gradient")
    with pytest.raises(TypeError):
        ObservationDesignBlock(
            functional,
            Parameter(unconstrained=0.2),  # type: ignore[arg-type]
            max_size=functional.locations.count,
        )
    with pytest.raises(ValueError, match="must be scalar"):
        ObservationDesignBlock(
            functional,
            PositiveParameter(jnp.ones(2)),
            max_size=functional.locations.count,
        )

    variance = PositiveParameter(0.2)
    design = ObservationDesignBlock(
        functional, variance, max_size=functional.locations.count
    )

    @jax.jit
    @jax.vmap
    def variances(raw):
        updated = eqx.tree_at(lambda b: b.variance.unconstrained, design, raw)
        return updated.variances

    raw = jnp.array([-8.0, 1.0, 10.0])
    expected = jnp.broadcast_to(jax.nn.softplus(raw)[:, None, None], (3, 2, 2))
    assert jnp.allclose(variances(raw), expected)


@pytest.mark.parametrize("compiled", [False, True])
def test_block_append_retains_operator_variance_and_prior(compiled):
    prior = GammaPrior(2.0, 3.0)
    variance = PositiveParameter(0.2, prior=prior)
    functional = FunctionalBlock(
        Locations(jnp.array([[0.0, 1.0]])), Gradient(), "gradient"
    )
    original = ObservationBlock(
        ObservationDesignBlock(functional, variance, max_size=3),
        jnp.array([[1.0, 2.0]]),
    )
    locations = Locations(jnp.array([[2.0, 3.0], [4.0, 5.0]]))
    outcomes = jnp.array([[3.0, 4.0], [5.0, 6.0]])

    def append(block, x, y):
        return block.append(x, y)

    operation = eqx.filter_jit(append) if compiled else append
    updated = operation(original, locations, outcomes)
    assert updated.functional.name == "gradient"
    assert updated.functional.shape == (3, 2)
    assert jnp.array_equal(updated.functional.locations.points[1:], locations.points)
    assert jnp.array_equal(updated.y, jnp.arange(1.0, 7.0).reshape(3, 2))
    assert jnp.allclose(updated.design.variances, jnp.full((3, 2), 0.2))
    assert jnp.allclose(log_prior(updated), prior.log_prob(variance.value))
    assert original.functional.shape == (3, 2)
    assert int(original.count) == 1
    assert jnp.array_equal(original.y, jnp.array([[1.0, 2.0], [0.0, 0.0], [0.0, 0.0]]))

    def loss(raw):
        block = eqx.tree_at(lambda b: b.design.variance.unconstrained, original, raw)
        enlarged = operation(block, locations, outcomes)
        return jnp.sum(enlarged.design.variances) - log_prior(enlarged)

    def expected(raw):
        value = jax.nn.softplus(raw)
        return 6 * value - prior.log_prob(value)

    assert jnp.allclose(
        jax.grad(loss)(variance.unconstrained),
        jax.grad(expected)(variance.unconstrained),
    )
    with pytest.raises(ValueError, match="same input dimension"):
        operation(original, Locations(jnp.zeros((1, 3))), jnp.zeros((1, 2)))
    with pytest.raises(ValueError, match="appended values.*expected"):
        operation(original, locations, jnp.ones(2))


def test_index_map_round_trip_preserves_block_shapes() -> None:
    locations = Locations(jnp.array([[0.0, 1.0], [2.0, 3.0]]))
    query = FunctionalQuery(
        [
            FunctionalBlock(locations, Identity(), "values"),
            FunctionalBlock(locations, Gradient(), "gradients"),
        ]
    )

    structured = {
        "values": jnp.array([1.0, 2.0]),
        "gradients": jnp.array([[3.0, 4.0], [5.0, 6.0]]),
    }
    flat = query.index_map.pack(structured)
    unpacked = query.index_map.unpack(flat)

    assert flat.shape == (6,)
    assert jnp.array_equal(unpacked["values"], structured["values"])
    assert jnp.array_equal(unpacked["gradients"], structured["gradients"])


def test_index_map_establishes_unique_names_at_construction() -> None:
    block = FunctionalBlock(Locations(jnp.zeros((1, 1))), Identity(), "values")
    valid = IndexMap.from_blocks([block])
    assert valid.block("values").shape == (1,)
    with pytest.raises(KeyError, match="missing"):
        valid.block("missing")
    with pytest.raises(ValueError, match="names must be unique"):
        IndexMap.from_blocks([block, block])
    with pytest.raises(ValueError, match="names must be unique"):
        IndexMap(valid.blocks * 2, size=2)


@pytest.mark.parametrize("compiled", [False, True])
def test_observation_output_shape_is_validated(compiled) -> None:
    functional = FunctionalBlock(
        Locations(jnp.array([[0.0], [1.0]])), Identity(), "values"
    )
    make_design = (
        eqx.filter_jit(ObservationDesignBlock) if compiled else ObservationDesignBlock
    )
    make_observation = (
        eqx.filter_jit(ObservationBlock) if compiled else ObservationBlock
    )

    design = make_design(functional, 0.1, max_size=2)
    with pytest.raises(ValueError, match="values.*expected"):
        make_observation(design, jnp.ones((2, 1)))


@pytest.mark.parametrize("compiled", [False, True])
def test_composed_block_views_follow_tree_updates(compiled) -> None:
    functional = FunctionalBlock(
        Locations(jnp.array([[0.0, 1.0], [2.0, 3.0]])),
        DirectionalDerivative(jnp.array([1.0, 0.0])),
        "directional",
    )
    design = ObservationDesignBlock(
        functional, PositiveParameter(0.1), max_size=functional.locations.count
    )
    observed = ObservationBlock(design, y=jnp.array([0.2, 0.4]))

    def update(block, points, direction, variance, outcomes):
        return eqx.tree_at(
            lambda b: (
                b.design.functional.locations.points,
                b.design.functional.operator.direction,
                b.design.variance,
                b.y,
            ),
            block,
            (points, direction, variance, outcomes),
        )

    operation = eqx.filter_jit(update) if compiled else update
    points = functional.locations.points + 1
    direction = jnp.array([0.0, 1.0])
    outcomes = jnp.array([0.6, 0.8])
    updated = operation(
        observed,
        points,
        direction,
        PositiveParameter(0.3),
        outcomes,
    )
    assert jnp.array_equal(updated.functional.locations.points, points)
    assert jnp.array_equal(updated.functional.operator.direction, direction)
    assert jnp.allclose(updated.design.variances, jnp.full((2,), 0.3))
    assert jnp.array_equal(updated.y, outcomes)
    assert updated.functional.name == "directional"
    assert jnp.allclose(observed.design.variances, jnp.full((2,), 0.1))
    dataset = FunctionalDataset([updated])
    assert jnp.array_equal(dataset.values_flat, outcomes)
    assert jnp.array_equal(dataset.variance_flat, updated.design.variances)
    assert eqx.tree_equal(dataset.query.blocks[0], updated.functional)
    assert eqx.tree_equal(dataset.design.blocks[0], updated.design)


def test_block_views_preserve_tensor_shapes_and_broadcast_variance_under_jit() -> None:
    locations = Locations(jnp.array([[0.0, 1.0], [2.0, 3.0]]))
    observed = ObservationBlock(
        ObservationDesignBlock(
            FunctionalBlock(locations, Gradient(), "gradient"),
            variance=0.1,
            max_size=locations.count,
        ),
        y=jnp.ones((2, 2)),
    )
    functional = observed.functional
    assert functional.shape == (2, 2)
    assert functional.size == 4

    dataset = FunctionalDataset([observed])
    query, variance = eqx.filter_jit(lambda d: (d.query, d.variance_flat))(dataset)
    assert query.index_map == dataset.index_map
    assert query.blocks[0].shape == (2, 2)
    assert jnp.allclose(variance, jnp.full((4,), 0.1))
    assert jnp.array_equal(variance, dataset.design.variance_flat)
    with pytest.raises(ValueError, match="block names cannot be empty"):
        FunctionalBlock(locations, Gradient(), "")


def test_collection_names_are_unique() -> None:
    locations = Locations(jnp.array([[0.0]]))
    block = FunctionalBlock(locations, Identity(), "same")

    with pytest.raises(ValueError, match="unique"):
        FunctionalQuery([block, block])


def test_datasets_keep_distinct_blocks_and_reject_duplicate_names() -> None:
    first = ObservationBlock(
        ObservationDesignBlock(
            FunctionalBlock(Locations(jnp.array([[0.0]])), Identity(), "values"),
            variance=0.1,
            max_size=1,
        ),
        y=jnp.array([1.0]),
    )
    second = ObservationBlock(
        ObservationDesignBlock(
            FunctionalBlock(Locations(jnp.array([[2.0]])), Identity(), "later_values"),
            variance=0.2,
            max_size=1,
        ),
        y=jnp.array([3.0]),
    )

    combined = FunctionalDataset([first]) + FunctionalDataset([second])

    assert len(combined.blocks) == 2
    assert jnp.array_equal(combined.values_flat, jnp.array([1.0, 3.0]))
    assert jnp.array_equal(combined.variance_flat, jnp.array([0.1, 0.2]))
    with pytest.raises(ValueError, match="names must be unique"):
        FunctionalDataset([first, first])
    with pytest.raises(ValueError, match="names must be unique"):
        _ = FunctionalDataset([first]) + FunctionalDataset([first])


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("operator", [None, Gradient()])
def test_array_construction_matches_explicit_blocks(compiled, operator):
    points = jnp.array([[0.0, 1.0], [2.0, 3.0]])
    values = jnp.ones((2,) if operator is None else (2, 2))

    def construct(x, y, variance):
        return FunctionalDataset(
            [
                ObservationBlock.from_arrays(
                    x,
                    y,
                    variance=variance,
                    name="measurements",
                    operator=operator,
                    max_size=x.shape[0],
                )
            ]
        )

    operation = eqx.filter_jit(construct) if compiled else construct
    actual = operation(points, values, jnp.array(0.2))
    expected = FunctionalDataset(
        [
            ObservationBlock(
                ObservationDesignBlock(
                    FunctionalBlock(
                        Locations(points),
                        Identity() if operator is None else operator,
                        "measurements",
                    ),
                    0.2,
                    max_size=points.shape[0],
                ),
                values,
            )
        ]
    )
    assert eqx.tree_equal(actual, expected)
    assert (
        actual.index_map.unpack(actual.values_flat)["measurements"].shape
        == values.shape
    )
    with pytest.raises(ValueError):
        operation(points, jnp.ones((2, 1, 1)), 0.2)


def test_collection_metadata_survives_numeric_updates_and_batched_construction():
    @jax.jit
    @jax.vmap
    def construct(points, values):
        data = FunctionalDataset(
            [
                ObservationBlock.from_arrays(
                    points,
                    values,
                    variance=0.1,
                    name="values",
                    max_size=points.shape[0],
                )
            ]
        )
        return data, data.query, data.design

    points = jnp.arange(12.0).reshape(2, 3, 2)
    values = jnp.arange(6.0).reshape(2, 3)
    data, query, design = construct(points, values)
    assert data.index_map == query.index_map == design.index_map
    assert data.index_map.block("values").shape == (3,)
    assert jnp.array_equal(data.blocks[0].y, values)
    assert jnp.array_equal(query.functional_blocks[0].locations.points, points)

    single = FunctionalDataset(
        [
            ObservationBlock.from_arrays(
                points[0],
                values[0],
                variance=0.1,
                name="values",
                max_size=4,
            )
        ]
    )
    updated = FunctionalDataset([single.blocks[0].replace(points[0], values[0] + 10)])
    assert jnp.array_equal(
        updated.values_flat, jnp.concatenate((values[0] + 10, jnp.zeros(1)))
    )
    assert updated.index_map == single.index_map
    extended = FunctionalDataset(
        [single.blocks[0].append(Locations(jnp.ones((1, 2))), jnp.array([7.0]))]
    )
    assert extended.index_map.block("values").shape == (4,)
    assert single.index_map.block("values").shape == (4,)
    assert extended.index_map == single.index_map
    assert int(single.active_size) == 3
    assert int(extended.active_size) == 4
