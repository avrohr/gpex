import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from gpex import (
    FeaturePosteriorPath,
    FunctionalBlock,
    FunctionalDataset,
    GaussianProcess,
    ObservationBlock,
    ObservationDesignBlock,
)
from gpex.blocks import Locations
from gpex.kernels import (
    FeatureKernel,
    GeneralizedMatern12Kernel,
    GeneralizedMatern32Kernel,
    GeneralizedMatern52Kernel,
    GeneralizedRBFKernel,
    Matern12Kernel,
    Matern32Kernel,
    Matern52Kernel,
    RBFKernel,
)
from gpex.operators import Gradient, Identity


def _posterior(*, jitter: float = 1e-6, kernel=None):
    points = jnp.array([[-0.8], [0.1], [0.9]])
    values = jnp.sin(points[:, 0])
    dataset = FunctionalDataset(
        [
            ObservationBlock(
                ObservationDesignBlock(
                    FunctionalBlock(Locations(points), Identity(), "values"),
                    variance=0.02,
                    max_size=points.shape[0],
                ),
                y=values,
            )
        ]
    )
    kernel = RBFKernel(lengthscale=0.7) if kernel is None else kernel
    return GaussianProcess(kernel).condition(dataset, jitter=jitter)


def test_posterior_path_returns_one_consistent_differentiable_path() -> None:
    posterior = _posterior()
    path = FeaturePosteriorPath(posterior, jax.random.key(4), feature_count=256)
    constructed = eqx.filter_jit(FeaturePosteriorPath)(
        posterior, jax.random.key(4), feature_count=256
    )
    points = jnp.linspace(-1.0, 1.0, 11)[:, None]

    first = jax.vmap(path)(points)
    second = jax.jit(jax.vmap(path))(points)
    derivative = jax.grad(path)(jnp.array([0.2]))
    operator_derivative = Gradient().apply(path, jnp.array([0.2]))

    assert first.shape == (11,)
    assert jnp.allclose(first, second, rtol=1e-5, atol=1e-6)
    assert jnp.allclose(first, jax.vmap(constructed)(points), rtol=1e-5, atol=1e-6)
    assert derivative.shape == (1,)
    assert jnp.allclose(operator_derivative, derivative)


@pytest.mark.parametrize(
    "kernel_type", [RBFKernel, Matern12Kernel, Matern32Kernel, Matern52Kernel]
)
def test_noiseless_posterior_path_interpolates_observations(kernel_type) -> None:
    points = jnp.array([[-0.7], [0.2], [0.8]])
    values = jnp.array([-0.4, 0.3, 0.9])
    dataset = FunctionalDataset(
        [
            ObservationBlock(
                ObservationDesignBlock(
                    FunctionalBlock(Locations(points), Identity(), "values"),
                    variance=0.0,
                    max_size=points.shape[0],
                ),
                y=values,
            )
        ]
    )
    posterior = GaussianProcess(kernel_type(lengthscale=0.6)).condition(
        dataset, jitter=0.0
    )
    path = FeaturePosteriorPath(posterior, jax.random.key(8), feature_count=64)

    assert jnp.allclose(jax.vmap(path)(points), values, atol=2e-5)


def test_posterior_path_validates_approximation_and_point_shape() -> None:
    posterior = _posterior()

    with pytest.raises(ValueError, match="feature_count"):
        FeaturePosteriorPath(posterior, jax.random.key(0), feature_count=0)

    path = FeaturePosteriorPath(posterior, jax.random.key(0), feature_count=16)
    with pytest.raises(ValueError, match="path points"):
        path(jnp.array([[0.1]]))


@pytest.mark.parametrize("compiled", [False, True])
def test_posterior_path_rejects_kernels_without_feature_capability(compiled) -> None:
    posterior = _posterior(kernel=RBFKernel() + RBFKernel())
    construct = (
        eqx.filter_jit(FeaturePosteriorPath) if compiled else FeaturePosteriorPath
    )
    with pytest.raises(TypeError, match="FeatureKernel.*SumKernel"):
        construct(posterior, jax.random.key(0), feature_count=16)


class _AffineKernel(FeatureKernel):
    def __call__(self, left, right):
        return 1.0 + jnp.dot(left, right)

    def random_features(self, key, input_dimension, *, feature_count=1024):
        assert input_dimension == 1 and feature_count == 2
        return _AffineFeatures()


class _AffineFeatures(eqx.Module):
    def __call__(self, point):
        return jnp.concatenate((jnp.ones(1), point))


def test_posterior_path_uses_custom_kernel_features_without_kernel_specific_fields():
    posterior = _posterior(kernel=_AffineKernel())
    key = jax.random.key(17)
    path = eqx.filter_jit(FeaturePosteriorPath)(posterior, key, feature_count=2)
    point = jnp.array([0.4])
    value = eqx.filter_jit(path)(point)
    assert jnp.isfinite(value)
    assert jnp.allclose(
        value, FeaturePosteriorPath(posterior, key, feature_count=2)(point)
    )
    assert jnp.all(jnp.isfinite(jax.grad(path)(point)))


@pytest.mark.parametrize(
    "kernel_type",
    [
        GeneralizedRBFKernel,
        GeneralizedMatern12Kernel,
        GeneralizedMatern32Kernel,
        GeneralizedMatern52Kernel,
    ],
)
def test_posterior_path_moments_with_full_metric(kernel_type):
    from gpex import FunctionalBlock, FunctionalQuery

    posterior = _posterior(kernel=kernel_type(variance=1.3, L=jnp.array([[1.4]])))
    points = jnp.array([[-0.4], [0.4]])
    query = FunctionalQuery([FunctionalBlock(Locations(points), Identity(), "query")])
    prediction = posterior.predict(query)

    def sample(key):
        path = FeaturePosteriorPath(posterior, key, feature_count=256)
        return jax.vmap(path)(points)

    draws = jax.jit(jax.vmap(sample))(jax.random.split(jax.random.key(82), 2048))
    assert jnp.allclose(jnp.mean(draws, axis=0), prediction.mean.flat, atol=0.04)
    assert jnp.allclose(
        jnp.cov(draws.T), prediction.covariance.full, atol=0.04, rtol=0.1
    )
