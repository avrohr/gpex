from typing import Any, cast

import jax
import jax.numpy as jnp
import pytest

from gpex import FunctionalBlock, FunctionalQuery
from gpex.blocks import Locations
from gpex.kernels import (
    FeatureKernel,
    GeneralizedRBFKernel,
    Kernel,
    Lengthscales,
    RadialFeatureKernel,
    RadialKernel,
    RBFKernel,
)
from gpex.means import ConstantMean, LinearMean
from gpex.moments import assemble_covariance, assemble_mean
from gpex.operators import Gradient, Identity
from gpex.parameters import (
    CholeskyParameter,
    Parameter,
    PositiveParameter,
    log_prior,
)
from gpex.parameters.priors import IndependentCholeskyPrior, LogNormalPrior, NormalPrior


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("left_shape,right_shape", [((1,), (2,)), ((2, 1), (2, 1))])
def test_rbf_rejects_mismatched_or_nonvector_points(
    compiled: bool, left_shape: tuple[int, ...], right_shape: tuple[int, ...]
) -> None:
    kernel = RBFKernel()
    evaluate = jax.jit(lambda x, y: kernel(x, y)) if compiled else kernel
    with pytest.raises(ValueError, match="same one-dimensional shape"):
        evaluate(jnp.ones(left_shape), jnp.ones(right_shape))


@pytest.mark.parametrize("lengthscale", [jnp.ones(1), jnp.ones(3)])
def test_rbf_rejects_wrong_ard_dimension(lengthscale: jax.Array) -> None:
    kernel = RBFKernel(lengthscale=lengthscale)
    with pytest.raises(ValueError, match="ARD lengthscale shape"):
        jax.jit(lambda x, y: kernel(x, y))(jnp.zeros(2), jnp.ones(2))


def test_rbf_rejects_empty_lengthscale() -> None:
    with pytest.raises(ValueError, match="nonempty vector"):
        RBFKernel(lengthscale=jnp.empty(0))


def test_kernel_sum_and_product_compose_through_moment_assembly() -> None:
    first = RBFKernel(variance=1.3, lengthscale=0.7)
    second = RBFKernel(variance=0.4, lengthscale=1.8)
    summed = first + second
    product = first * second
    left = jnp.array([0.2, -0.4])
    right = jnp.array([-0.3, 0.7])

    assert jnp.allclose(summed(left, right), first(left, right) + second(left, right))
    assert jnp.allclose(product(left, right), first(left, right) * second(left, right))

    locations = Locations(jnp.array([[-0.5, 0.1], [0.2, -0.3], [0.7, 0.6]]))
    query = FunctionalQuery(
        [
            FunctionalBlock(locations, Identity(), "values"),
            FunctionalBlock(locations, Gradient(), "gradients"),
        ]
    )
    summed_covariance = assemble_covariance(query.blocks, query.blocks, summed)
    expected_sum = assemble_covariance(
        query.blocks, query.blocks, first
    ) + assemble_covariance(query.blocks, query.blocks, second)
    product_covariance = assemble_covariance(query.blocks, query.blocks, product)

    assert jnp.allclose(summed_covariance, expected_sum, rtol=3e-5, atol=3e-5)
    assert jnp.linalg.eigvalsh(product_covariance).min() > -3e-4


def test_mean_sum_and_product_remain_differentiable() -> None:
    affine = LinearMean(jnp.array([1.2, -0.7]), intercept=0.3)
    mean = (affine + ConstantMean(1.5)) * ConstantMean(2.0)
    point = Locations(jnp.array([[0.4, -0.2]]))
    query = (
        FunctionalBlock(point, Identity(), "value"),
        FunctionalBlock(point, Gradient(), "gradient"),
    )
    compiled = assemble_mean(query, mean)

    assert jnp.allclose(compiled[0], 2.0 * (affine(point.points[0]) + 1.5))
    assert jnp.allclose(compiled[1:], 2.0 * affine.weights)


def test_generalized_kernel_uses_precision_factor_and_joint_prior() -> None:
    L = CholeskyParameter(
        jnp.array([[1.0, 0.0], [0.3, 2.0]]),
        prior=IndependentCholeskyPrior(
            diagonal_prior=LogNormalPrior(log_standard_deviation=0.6),
            off_diagonal_prior=NormalPrior(standard_deviation=0.5),
        ),
        name="L",
    )
    variance = PositiveParameter(1.4, prior=LogNormalPrior())
    kernel = GeneralizedRBFKernel(variance=variance, L=L)
    left, right = jnp.array([0.2, -0.4]), jnp.array([-0.3, 0.7])
    displacement = left - right
    expected_distance = (
        displacement @ (kernel.lengthscales.L @ kernel.lengthscales.L.T) @ displacement
    )
    assert jnp.allclose(
        kernel(left, right), kernel.variance * jnp.exp(-0.5 * expected_distance)
    )
    assert jnp.allclose(log_prior(kernel), L.log_prior() + variance.log_prior())
    assert jnp.all(jnp.isfinite(jax.grad(lambda point: kernel(point, right))(left)))


def test_custom_covariance_does_not_require_random_features() -> None:
    class LinearKernel(Kernel):
        def __call__(self, left, right):
            return left @ right

    kernel = LinearKernel()
    assert kernel(jnp.ones(2), jnp.ones(2)) == 2.0
    assert not hasattr(kernel, "random_features")


def test_lengthscale_kernel_construction_transforms_with_model_values() -> None:
    left, right = jnp.array([0.2, -0.4]), jnp.array([-0.3, 0.7])

    @jax.jit
    def construct(variance, lengthscale):
        return RBFKernel(variance=variance, lengthscale=lengthscale)

    def covariance(values):
        kernel = construct(values[0], values[1:])
        return kernel(left, right)

    def reference(values):
        return values[0] * jnp.exp(-0.5 * jnp.sum(((left - right) / values[1:]) ** 2))

    values = jnp.array([[1.3, 0.7, 1.2], [0.8, 0.4, 1.1]])
    actual = jax.jit(jax.vmap(jax.value_and_grad(covariance)))(values)
    expected = jax.vmap(jax.value_and_grad(reference))(values)
    assert jnp.allclose(actual[0], expected[0])
    assert jnp.allclose(actual[1], expected[1])


def test_precision_kernel_construction_transforms_with_model_values() -> None:
    left, right = jnp.array([0.2, -0.4]), jnp.array([-0.3, 0.7])

    @jax.jit
    def construct(L):
        return GeneralizedRBFKernel(L=L)

    def covariance(values):
        L = jnp.array([[values[0], 0.0], [values[1], values[2]]])
        return construct(L)(left, right)

    def reference(values):
        L = jnp.array([[values[0], 0.0], [values[1], values[2]]])
        return jnp.exp(-0.5 * jnp.sum((L.T @ (left - right)) ** 2))

    values = jnp.array([[1.3, 0.7, 1.2], [0.8, 0.4, 1.1]])
    actual = jax.jit(jax.vmap(jax.value_and_grad(covariance)))(values)
    expected = jax.vmap(jax.value_and_grad(reference))(values)
    assert jnp.allclose(actual[0], expected[0])
    assert jnp.allclose(actual[1], expected[1])


@pytest.mark.parametrize("field", ["variance", "lengthscale"])
def test_invalid_plain_kernel_arguments_fail_under_jit(field) -> None:
    @jax.jit
    def covariance(value):
        kernel = RBFKernel(**{field: value})
        return kernel(jnp.zeros(1), jnp.ones(1))

    with pytest.raises(Exception, match=f"{field} must be finite and positive"):
        covariance(jnp.asarray(-1.0)).block_until_ready()


@pytest.mark.parametrize("field", ["variance", "lengthscale"])
def test_trainable_radial_arguments_require_positive_parameters(field):
    with pytest.raises(TypeError):
        RBFKernel(**{field: cast(Any, Parameter(unconstrained=jnp.asarray(1.0)))})


def test_radial_covariance_does_not_require_a_spectral_sampler():
    class CovarianceOnlyRadial(RadialKernel):
        __init__ = RadialKernel.__init__
        _correlation = staticmethod(lambda r2: jnp.exp(-r2 / 2))

    kernel = CovarianceOnlyRadial(Lengthscales(0.7), variance=1.3)
    left, right = jnp.array([0.2]), jnp.array([0.6])
    assert jnp.allclose(
        kernel(left, right), RBFKernel(lengthscale=0.7, variance=1.3)(left, right)
    )
    assert not isinstance(kernel, FeatureKernel)
    assert not hasattr(kernel, "random_features")


def test_feature_capabilities_require_their_construction_methods():
    class MissingFeatures(FeatureKernel):
        def __call__(self, left, right):
            return left @ right

    class MissingSpectrum(RadialFeatureKernel):
        _correlation = staticmethod(lambda r2: jnp.exp(-r2 / 2))

    with pytest.raises(TypeError, match="abstract.*random_features"):
        MissingFeatures()  # type: ignore[abstract]
    with pytest.raises(TypeError, match="abstract.*_sample_frequencies"):
        MissingSpectrum(Lengthscales(1.0), variance=1.0)  # type: ignore[abstract]
