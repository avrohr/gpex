import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from gpex import (
    FunctionalBlock,
    FunctionalDataset,
    GaussianProcess,
    ObservationBlock,
    ObservationDesignBlock,
)
from gpex.blocks import Locations
from gpex.fitting import negative_log_marginal_likelihood, negative_log_posterior
from gpex.kernels import RBFKernel
from gpex.operators import Identity
from gpex.parameters import (
    ExpTransform,
    IdentityTransform,
    Parameter,
    PositiveParameter,
)
from gpex.parameters.priors import GammaPrior, LogNormalPrior, NormalPrior


@pytest.mark.parametrize("transform", [None, IdentityTransform(), ExpTransform()])
@pytest.mark.parametrize("compiled", [False, True])
def test_parameter_constructor_stores_raw_coordinates(transform, compiled):
    prior = NormalPrior()

    def construct(raw):
        return Parameter(
            unconstrained=raw, transform=transform, prior=prior, name="raw"
        )

    raw = jnp.array([-2.0, 1.0])
    expected_value = raw if transform is None else transform.forward(raw)
    parameter = eqx.filter_jit(construct)(raw) if compiled else construct(raw)

    assert jnp.array_equal(parameter.unconstrained, raw)
    assert jnp.allclose(parameter.value, expected_value)
    assert parameter.name == "raw"
    assert jnp.allclose(parameter.log_prior(), prior.log_prob(expected_value))


def test_raw_parameter_constructor_preserves_transform_gradients():
    prior = NormalPrior()

    def objective(raw):
        parameter = Parameter(unconstrained=raw, transform=ExpTransform(), prior=prior)
        return jnp.sum(parameter.value) - parameter.log_prior()

    raw = jnp.array([-2.0, 1.0])
    value, gradient = jax.jit(jax.value_and_grad(objective))(raw)
    expected_value = jnp.sum(jnp.exp(raw)) - prior.log_prob(jnp.exp(raw))
    expected_gradient = jnp.exp(raw) + jnp.exp(raw) ** 2
    assert jnp.allclose(value, expected_value)
    assert jnp.allclose(gradient, expected_gradient)


def test_priors_are_explicit_and_do_not_change_the_likelihood() -> None:
    locations = Locations(jnp.array([[0.0], [1.0]]))
    dataset = FunctionalDataset(
        [
            ObservationBlock(
                ObservationDesignBlock(
                    FunctionalBlock(locations, Identity(), "y"),
                    variance=0.05,
                    max_size=locations.count,
                ),
                y=jnp.array([0.2, -0.1]),
            )
        ]
    )
    fixed_gp = GaussianProcess(RBFKernel(variance=1.2, lengthscale=0.7))
    parameterized_gp = GaussianProcess(
        RBFKernel(
            variance=PositiveParameter(
                1.2, prior=GammaPrior(2.0, 1.0), name="variance"
            ),
            lengthscale=PositiveParameter(
                0.7,
                prior=LogNormalPrior(0.0, 0.5),
                name="lengthscale",
            ),
        )
    )

    assert jnp.allclose(
        negative_log_marginal_likelihood(fixed_gp, dataset),
        negative_log_marginal_likelihood(parameterized_gp, dataset),
    )


def test_map_rejects_trainable_parameters_without_explicit_priors() -> None:
    locations = Locations(jnp.array([[0.0]]))
    dataset = FunctionalDataset(
        [
            ObservationBlock(
                ObservationDesignBlock(
                    FunctionalBlock(locations, Identity(), "y"),
                    variance=0.05,
                    max_size=locations.count,
                ),
                y=jnp.array([0.2]),
            )
        ]
    )
    gp = GaussianProcess(RBFKernel(variance=PositiveParameter(1.0, name="variance")))

    with pytest.raises(ValueError, match="variance"):
        negative_log_posterior(gp, dataset)
