"""Matrix constraints, density measures, and integration with ordinary fitting."""

from math import log, pi

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from scipy.special import betaln
from scipy.stats import lognorm, norm

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
    negative_log_marginal_likelihood,
    negative_log_posterior,
)
from gpex.kernels import (
    GeneralizedMatern32Kernel,
    GeneralizedRBFKernel,
)
from gpex.operators import Identity
from gpex.parameters import (
    CholeskyParameter,
    Parameter,
    PositiveParameter,
    log_prior,
)
from gpex.parameters.priors import (
    IndependentCholeskyPrior,
    LogNormalPrior,
    MultivariateLogNormalPrior,
    NormalPrior,
    ScaledLKJCholeskyPrior,
)

PRIORS = [IndependentCholeskyPrior(), ScaledLKJCholeskyPrior(concentration=2.0)]


@pytest.mark.parametrize("dimension", [1, 2, 4])
def test_cholesky_parameter_roundtrip_and_packed_updates(dimension):
    initial = jnp.tril(jnp.arange(1.0, dimension**2 + 1).reshape(dimension, dimension))
    parameter = CholeskyParameter(initial)
    assert parameter.unconstrained.shape == (dimension * (dimension + 1) // 2,)
    assert jnp.allclose(eqx.filter_jit(lambda p: p.value)(parameter), initial)
    raw_parameter = Parameter(
        unconstrained=parameter.unconstrained, transform=parameter.transform
    )
    assert jnp.array_equal(raw_parameter.unconstrained, parameter.unconstrained)
    assert jnp.allclose(raw_parameter.value, initial)
    restored = CholeskyParameter(raw_parameter.value)
    assert jnp.allclose(restored.unconstrained, parameter.unconstrained)
    updated = eqx.tree_at(
        lambda p: p.unconstrained, parameter, parameter.unconstrained - 3.0
    )
    assert jnp.all(jnp.diag(updated.value) > 0)
    assert jnp.all(jnp.triu(updated.value, k=1) == 0)
    rows, columns = jnp.tril_indices(dimension, k=-1)
    assert jnp.allclose(updated.value[rows, columns], initial[rows, columns] - 3.0)
    jacobian = jax.jacfwd(
        lambda raw: eqx.tree_at(lambda p: p.unconstrained, parameter, raw).value[
            jnp.tril_indices(dimension)
        ]
    )(parameter.unconstrained)
    assert jnp.linalg.matrix_rank(jacobian) == parameter.unconstrained.size


@pytest.mark.parametrize(
    "value",
    [
        jnp.ones(2),
        jnp.empty((0, 0)),
        jnp.ones((2, 3)),
        jnp.eye(2, dtype=complex),
        jnp.array([[1.0, 0.1], [0.0, 1.0]]),
        jnp.array([[1.0, 0.0], [0.2, 0.0]]),
        jnp.array([[-1.0, 0.0], [0.2, 1.0]]),
        jnp.array([[1.0, 0.0], [jnp.inf, 1.0]]),
        jnp.array([[1.0, 0.0], [0.0, jnp.nan]]),
    ],
)
def test_invalid_factors_are_rejected_at_construction(value):
    with pytest.raises(ValueError, match="L must"):
        CholeskyParameter(value)
    with pytest.raises(ValueError, match="L must"):
        GeneralizedRBFKernel(L=value)


def test_generalized_kernel_rejects_unconstrained_parameter():
    with pytest.raises(TypeError, match="CholeskyParameter"):
        GeneralizedRBFKernel(L=Parameter(unconstrained=jnp.eye(2)))  # type: ignore[arg-type]


def test_independent_prior_matches_scalar_distributions():
    L = jnp.array([[1.4, 0.0, 0.0], [-0.7, 0.6, 0.0], [0.2, 1.1, 2.0]])
    prior = IndependentCholeskyPrior(LogNormalPrior(0.2, 0.7), NormalPrior(-0.1, 0.4))
    expected = lognorm.logpdf(np.diag(L), s=0.7, scale=np.exp(0.2)).sum()
    expected += norm.logpdf(np.asarray(L)[np.tril_indices(3, -1)], -0.1, 0.4).sum()
    assert jnp.allclose(jax.jit(prior.log_prob)(L), expected)


@pytest.mark.parametrize("concentration", [0.4, 1.0, 3.5])
def test_scaled_lkj_two_dimensions_matches_beta_density_and_explicit_jacobian(
    concentration,
):
    scales = jnp.array([1.4, 0.7])
    rho = 0.6
    L = jnp.array(
        [[scales[0], 0.0], [scales[1] * rho, scales[1] * np.sqrt(1 - rho**2)]]
    )
    prior = ScaledLKJCholeskyPrior(concentration, LogNormalPrior(0.2, 0.5))
    # In 2D, rho = 2 * Beta(eta, eta) - 1; this density is w.r.t. rho.
    log_rho_density = (concentration - 1) * log(1 - rho**2) - betaln(0.5, concentration)
    scale_density = lognorm.logpdf(np.asarray(scales), s=0.5, scale=np.exp(0.2)).sum()
    expected = (
        scale_density + log_rho_density - log(float(scales[1]) / np.sqrt(1 - rho**2))
    )
    assert jnp.allclose(jax.jit(prior.log_prob)(L), expected, atol=2e-6)


@pytest.mark.parametrize(
    "dimension,correlation_density",
    [(1, 1.0), (2, 0.5), (3, 2 / pi**2), (4, 27 / (32 * pi**2))],
)
def test_scaled_lkj_normalization_at_identity(dimension, correlation_density):
    prior = ScaledLKJCholeskyPrior()
    expected = norm.logpdf(np.zeros(dimension)).sum() + log(correlation_density)
    assert jnp.allclose(prior.log_prob(jnp.eye(dimension)), expected, atol=3e-6)


def test_scaled_lkj_full_coordinate_change_matches_autodiff_jacobian():
    # The three strict-lower correlation entries and three scales form coordinates.
    coordinates = jnp.array([1.2, 0.8, 1.5, 0.3, -0.2, 0.4])

    def assemble(values):
        s = values[:3]
        a, b, c = values[3:]
        correlation_factor = jnp.array(
            [
                [1.0, 0.0, 0.0],
                [a, jnp.sqrt(1 - a * a), 0.0],
                [b, c, jnp.sqrt(1 - b * b - c * c)],
            ]
        )
        return (s[:, None] * correlation_factor)[jnp.tril_indices(3)]

    matrix = jnp.zeros((3, 3)).at[jnp.tril_indices(3)].set(assemble(coordinates))
    log_jacobian = jnp.linalg.slogdet(jax.jacfwd(assemble)(coordinates))[1]
    # eta=1: normalized 3D correlation-factor density is (2/pi²) * C_22.
    log_correlation_density = log(2 / pi**2) + 0.5 * jnp.log(1 - coordinates[3] ** 2)
    scale_prior = MultivariateLogNormalPrior(jnp.zeros(3), jnp.eye(3) * 0.4)
    expected = (
        scale_prior.log_prob(coordinates[:3]) + log_correlation_density - log_jacobian
    )
    actual = ScaledLKJCholeskyPrior(scale_prior=scale_prior).log_prob(matrix)
    assert jnp.allclose(actual, expected, atol=3e-6)


@pytest.mark.parametrize("prior", PRIORS)
def test_cholesky_prior_support_and_gradient(prior):
    for value in [
        jnp.array([[1.0, 0.1], [0.0, 1.0]]),
        jnp.array([[0.0, 0.0], [0.0, 1.0]]),
        jnp.array([[1.0, 0.0], [0.0, -1.0]]),
        jnp.array([[1.0, 0.0], [jnp.inf, 1.0]]),
        jnp.array([[1.0, 0.0], [0.0, jnp.nan]]),
    ]:
        assert jax.jit(prior.log_prob)(value) == -jnp.inf
    initial = jnp.array([[1.3, 0.0], [-0.4, 0.7]])
    gradient = jax.jit(jax.grad(prior.log_prob))(initial)
    direction = jnp.array([[0.2, 0.0], [0.5, -0.3]])
    step = 1e-5
    # Evaluate the difference in float64, including the prior's array leaves.
    # Subtracting nearly equal float32 log densities is a poor gradient oracle.
    with jax.enable_x64(True):
        reference_prior, point, tangent = jax.tree.map(
            lambda x: x.astype(jnp.float64) if eqx.is_inexact_array(x) else x,
            (prior, initial, direction),
        )
        reference = (
            reference_prior.log_prob(point + step * tangent)
            - reference_prior.log_prob(point - step * tangent)
        ) / (2 * step)
    assert jnp.all(jnp.isfinite(gradient))
    np.testing.assert_allclose(
        jnp.sum(gradient * direction), reference, rtol=1e-5, atol=5e-4
    )
    parameter = CholeskyParameter(initial, prior=prior)
    assert jnp.allclose(log_prior(parameter), prior.log_prob(initial))


@pytest.mark.parametrize("concentration", [0.0, -1.0, np.inf, np.nan])
def test_scaled_lkj_rejects_invalid_concentration(concentration):
    with pytest.raises(ValueError, match="concentration"):
        ScaledLKJCholeskyPrior(concentration)


def _dataset():
    points = jnp.array([[-0.9, 0.1], [-0.4, -0.7], [0.0, 0.5], [0.5, -0.3], [0.9, 0.7]])
    values = jnp.sin(points[:, 0]) + 0.3 * points[:, 1]
    return FunctionalDataset(
        [
            ObservationBlock(
                ObservationDesignBlock(
                    FunctionalBlock(Locations(points), Identity(), "y"),
                    variance=0.03,
                    max_size=points.shape[0],
                ),
                y=values,
            )
        ]
    )


@pytest.mark.parametrize("prior,optimizer", [(PRIORS[0], "adam"), (PRIORS[1], "lbfgs")])
def test_map_fits_matrix_without_special_optimizer_handling(prior, optimizer):
    L = CholeskyParameter(jnp.array([[1.2, 0.0], [0.4, 0.8]]), prior=prior)
    gp = GaussianProcess(GeneralizedMatern32Kernel(L=L))
    dataset = _dataset()

    def objective(model):
        return negative_log_posterior(model, dataset)

    if optimizer == "adam":
        result = fit(gp, objective, optax.adam(0.03), step_count=5)
    else:
        result = fit_lbfgs(gp, objective, max_steps=5)
    assert objective(result.model) < objective(gp)
    assert jnp.all(jnp.diag(result.model.kernel.lengthscales.L) > 0)
    assert jnp.all(jnp.triu(result.model.kernel.lengthscales.L, k=1) == 0)
    assert not jnp.allclose(
        result.model.kernel.lengthscales.L, gp.kernel.lengthscales.L
    )


def test_fixed_matrix_stays_fixed_and_matrix_prior_is_required_only_for_map():
    L = jnp.array([[1.2, 0.0], [-0.4, 0.8]])
    dataset = _dataset()
    fixed = GaussianProcess(GeneralizedRBFKernel(L=L))
    trainable = GaussianProcess(
        GeneralizedRBFKernel(L=CholeskyParameter(L, name="precision"))
    )
    assert jnp.allclose(
        negative_log_marginal_likelihood(fixed, dataset),
        negative_log_marginal_likelihood(trainable, dataset),
    )
    with pytest.raises(ValueError, match="precision"):
        negative_log_posterior(trainable, dataset)
    model = GaussianProcess(GeneralizedRBFKernel(L=L, variance=PositiveParameter(1.3)))
    result = fit(
        model,
        lambda gp: negative_log_marginal_likelihood(gp, dataset),
        optax.adam(0.02),
        step_count=3,
    )
    assert jnp.array_equal(result.model.kernel.lengthscales.L, L)
    assert not jnp.allclose(result.model.kernel.variance, model.kernel.variance)
