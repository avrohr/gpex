import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
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
from gpex.parameters import PositiveParameter, log_prior
from gpex.parameters.priors import LogNormalPrior, MultivariateLogNormalPrior


def test_diagonal_prior_matches_distinct_scalar_priors() -> None:
    mean = jnp.array([-0.3, 0.5])
    standard_deviation = jnp.array([0.4, 0.8])
    value = jnp.array([0.7, 1.4])
    prior = MultivariateLogNormalPrior(mean, jnp.diag(standard_deviation**2))
    expected = sum(
        LogNormalPrior(float(m), float(s)).log_prob(x)
        for m, s, x in zip(mean, standard_deviation, value, strict=True)
    )
    assert jnp.allclose(prior.log_prob(value), expected)


@pytest.mark.parametrize("correlation", [-0.5, 0.6])
def test_joint_prior_density_and_compiled_gradient(correlation: float) -> None:
    mean = np.array([-0.3, 0.2])
    covariance = np.array([[0.25, correlation * 0.3], [correlation * 0.3, 0.36]])
    value = np.array([0.7, 1.4])
    prior = MultivariateLogNormalPrior(mean, covariance)
    difference = np.log(value) - mean
    precision_difference = np.linalg.solve(covariance, difference)
    expected = (
        -0.5 * difference @ precision_difference
        - 0.5 * np.linalg.slogdet(covariance)[1]
        - np.log(2 * np.pi)
        - np.log(value).sum()
    )
    expected_gradient = -(precision_difference + 1) / value
    actual, gradient = jax.jit(jax.value_and_grad(prior.log_prob))(jnp.asarray(value))
    assert jnp.allclose(actual, expected, atol=1e-6)
    assert jnp.allclose(gradient, expected_gradient, atol=1e-6)


def test_joint_prior_supports_compiled_batched_map() -> None:
    prior = MultivariateLogNormalPrior([0.0, 0.1], [[0.25, 0.1], [0.1, 0.36]])
    lengthscale = PositiveParameter(jnp.array([0.7, 1.2]), prior=prior)
    gp = GaussianProcess(RBFKernel(lengthscale=lengthscale))
    dataset = FunctionalDataset(
        [
            ObservationBlock(
                ObservationDesignBlock(
                    FunctionalBlock(
                        Locations(jnp.array([[0.0, 0.0], [0.5, 0.3], [1.0, 0.8]])),
                        Identity(),
                        "values",
                    ),
                    variance=0.05,
                    max_size=3,
                ),
                y=jnp.array([0.0, 0.3, 0.7]),
            )
        ]
    )
    value, gradient = eqx.filter_jit(
        eqx.filter_value_and_grad(lambda model: negative_log_posterior(model, dataset))
    )(gp)
    expected = negative_log_marginal_likelihood(gp, dataset) - log_prior(gp)
    assert jnp.allclose(value, expected)
    assert jnp.all(
        jnp.isfinite(gradient.kernel.lengthscales.lengthscale_parameter.unconstrained)
    )
    batch = eqx.filter_vmap(
        lambda scale: eqx.tree_at(
            lambda model: model.kernel.lengthscales.lengthscale_parameter.unconstrained,
            gp,
            lengthscale.unconstrained + scale,
        )
    )(jnp.array([0.0, 0.1]))
    losses = eqx.filter_jit(
        eqx.filter_vmap(lambda model: negative_log_posterior(model, dataset))
    )(batch)
    assert losses.shape == (2,)
    assert jnp.all(jnp.isfinite(losses))


@pytest.mark.parametrize(
    "mean,covariance,message",
    [
        ([], [], "nonempty finite vector"),
        (0.0, [[1.0]], "nonempty finite vector"),
        ([np.nan], [[1.0]], "nonempty finite vector"),
        ([0.0, 0.0], [[1.0]], "shape"),
        ([0.0, 0.0], [[1.0, 0.5], [0.0, 1.0]], "symmetric"),
        ([0.0], [[np.inf]], "finite and symmetric"),
        ([0.0, 0.0], [[1.0, 1.0], [1.0, 1.0]], "positive definite"),
        ([0.0, 0.0], [[1.0, 2.0], [2.0, 1.0]], "positive definite"),
        ([1j], [[1.0]], "real"),
    ],
)
def test_invalid_prior_parameters(mean, covariance, message) -> None:
    with pytest.raises(ValueError, match=message):
        MultivariateLogNormalPrior(mean, covariance)


@pytest.mark.parametrize("value", [0.0, -1.0, np.inf, np.nan])
def test_joint_prior_outside_support(value: float) -> None:
    prior = MultivariateLogNormalPrior([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])
    assert jnp.isneginf(jax.jit(prior.log_prob)(jnp.array([1.0, value])))


@pytest.mark.parametrize("value", [1.0, [1.0], [[1.0, 2.0]]])
def test_joint_prior_requires_one_complete_vector(value) -> None:
    prior = MultivariateLogNormalPrior([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="shape"):
        prior.log_prob(jnp.asarray(value))


def test_one_dimensional_joint_prior_matches_scalar_prior() -> None:
    prior = MultivariateLogNormalPrior([0.2], [[0.25]])
    assert jnp.allclose(
        prior.log_prob(jnp.array([0.7])),
        LogNormalPrior(0.2, 0.5).log_prob(jnp.array(0.7)),
    )
