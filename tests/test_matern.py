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
from gpex.fitting import negative_log_marginal_likelihood
from gpex.kernels import (
    GeneralizedMatern12Kernel,
    GeneralizedMatern32Kernel,
    GeneralizedMatern52Kernel,
    GeneralizedRBFKernel,
    Matern12Kernel,
    Matern32Kernel,
    Matern52Kernel,
    RBFKernel,
)
from gpex.moments import assemble_covariance, assemble_covariance_diagonal
from gpex.operators import Gradient, Hessian, Identity, Laplacian, LinearCombination
from gpex.parameters import CholeskyParameter, PositiveParameter

KERNELS = [RBFKernel, Matern12Kernel, Matern32Kernel, Matern52Kernel]
GENERALIZED = dict(
    zip(
        KERNELS,
        [
            GeneralizedRBFKernel,
            GeneralizedMatern12Kernel,
            GeneralizedMatern32Kernel,
            GeneralizedMatern52Kernel,
        ],
        strict=True,
    )
)


@pytest.mark.parametrize("kernel_type", KERNELS)
def test_nonfinite_distances_are_not_treated_as_coincident_points(kernel_type):
    kernel = kernel_type()
    assert jnp.isnan(kernel(jnp.array([jnp.nan]), jnp.zeros(1)))


@pytest.mark.parametrize("kernel_type", KERNELS)
def test_scalar_ard_and_full_metric_are_the_same_kernel(kernel_type):
    left, right = jnp.array([0.2, -0.3]), jnp.array([-0.4, 0.7])
    isotropic = kernel_type(variance=1.7, lengthscale=0.8)
    ard = kernel_type(variance=1.7, lengthscale=jnp.array([0.8, 0.8]))
    full = GENERALIZED[kernel_type](variance=1.7, L=jnp.diag(jnp.array([1.25, 1.25])))
    for kernel in [ard, full]:
        assert jnp.allclose(kernel(left, right), isotropic(left, right))


@pytest.mark.parametrize(
    "kernel_type,nu",
    [(Matern12Kernel, 0.5), (Matern32Kernel, 1.5), (Matern52Kernel, 2.5)],
)
def test_matern_values_against_closed_form_with_rotated_metric(kernel_type, nu):
    L = jnp.array([[1.4, 0.0], [0.5, 0.7]])
    kernel = GENERALIZED[kernel_type](variance=1.6, L=L)
    left, right = jnp.array([0.2, -0.3]), jnp.array([-0.4, 0.7])
    delta = np.asarray(left - right)
    radius = np.sqrt(2 * nu * (delta @ np.asarray(L @ L.T) @ delta))
    polynomial = {0.5: 1, 1.5: 1 + radius, 2.5: 1 + radius + radius**2 / 3}[nu]
    assert jnp.allclose(
        eqx.filter_jit(kernel)(left, right), 1.6 * polynomial * np.exp(-radius)
    )
    assert kernel(left, left) == kernel.variance


@pytest.mark.parametrize(
    "kernel_type,coefficient", [(Matern32Kernel, 3.0), (Matern52Kernel, 5.0 / 3.0)]
)
def test_gradient_covariance_at_and_near_coincident_points(kernel_type, coefficient):
    L = jnp.array([[1.2, 0.0], [0.4, 0.7]])
    kernel = GENERALIZED[kernel_type](variance=1.7, L=L)
    origin = jnp.zeros(2)
    expected = 1.7 * coefficient * (L @ L.T)
    covariance = jax.jit(jax.jacfwd(jax.jacrev(kernel, 0), 1))
    for distance in [0.0, 1e-7, 1e-5]:
        actual = covariance(origin, jnp.array([distance, 0.0]))
        assert jnp.allclose(actual, expected, rtol=1e-4, atol=1e-5)


def test_matern52_hessian_covariance_at_and_near_coincident_points():
    L = jnp.array([[1.2, 0.0], [0.4, 0.7]])
    kernel = GeneralizedMatern52Kernel(variance=1.7, L=L)
    matrix = L @ L.T
    expected = (1.7 * 25.0 / 3.0) * (
        jnp.einsum("ij,kl->ijkl", matrix, matrix)
        + jnp.einsum("ik,jl->ijkl", matrix, matrix)
        + jnp.einsum("il,jk->ijkl", matrix, matrix)
    )
    covariance = jax.jit(
        jax.jacfwd(jax.jacfwd(jax.jacfwd(jax.jacfwd(kernel, 0), 0), 1), 1)
    )
    for distance in [0.0, 1e-7, 1e-5]:
        actual = covariance(jnp.zeros(2), jnp.array([distance, 0.0]))
        assert jnp.allclose(actual, expected, rtol=3e-4, atol=1e-4)


@pytest.mark.parametrize(
    "kernel_type,operator",
    [
        (Matern12Kernel, Identity()),
        (Matern32Kernel, Gradient()),
        (Matern52Kernel, Hessian()),
    ],
)
@pytest.mark.parametrize("full_metric", [False, True])
def test_matern_functional_covariance_and_hyperparameter_gradients(
    kernel_type, operator, full_metric
):
    locations = Locations(jnp.array([[-0.8], [0.0], [0.7]]))
    blocks = (
        FunctionalBlock(locations, Identity(), "values"),
        FunctionalBlock(locations, operator, "functionals"),
    )
    if full_metric:
        kernel = GENERALIZED[kernel_type](L=CholeskyParameter(jnp.array([[1.3]])))
    else:
        kernel = kernel_type(lengthscale=PositiveParameter(0.8))
    kernel = eqx.tree_at(lambda k: k.variance_parameter, kernel, PositiveParameter(1.4))

    def objective(model, measurements=blocks):
        covariance = assemble_covariance(measurements, measurements, model)
        # A factorization exercises diagonal, cross, and repeated-location blocks.
        factor = jnp.linalg.cholesky(covariance + 0.02 * jnp.eye(covariance.shape[0]))
        return jnp.sum(jnp.log(jnp.diag(factor)))

    value, gradient = eqx.filter_jit(eqx.filter_value_and_grad(objective))(kernel)
    assert jnp.isfinite(value)
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(gradient))
    covariance = jax.jit(lambda: assemble_covariance(blocks, blocks, kernel))()
    assert jnp.allclose(covariance, covariance.T, rtol=1e-5, atol=1e-5)
    assert jnp.linalg.eigvalsh(covariance).min() > -1e-4

    # Check trainable-parameter gradients against a directional finite difference.
    step = 1e-5
    with jax.enable_x64(True):
        reference_kernel, reference_blocks = jax.tree.map(
            lambda x: x.astype(jnp.float64) if eqx.is_inexact_array(x) else x,
            (kernel, blocks),
        )
        perturbation = jax.tree.map(jnp.ones_like, reference_kernel)
        plus = eqx.apply_updates(
            reference_kernel, jax.tree.map(lambda x: step * x, perturbation)
        )
        minus = eqx.apply_updates(
            reference_kernel, jax.tree.map(lambda x: -step * x, perturbation)
        )
        finite_difference = (
            objective(plus, reference_blocks) - objective(minus, reference_blocks)
        ) / (2 * step)
    actual = sum(jnp.sum(leaf) for leaf in jax.tree.leaves(gradient))
    np.testing.assert_allclose(actual, finite_difference, rtol=3e-3, atol=3e-3)


@pytest.mark.parametrize(
    "kernel_type,operator",
    [
        (Matern12Kernel, Gradient()),
        (Matern32Kernel, Hessian()),
        (Matern32Kernel, LinearCombination([Identity(), Laplacian()], jnp.ones(2))),
    ],
)
@pytest.mark.parametrize("operation", ["full", "diagonal"])
def test_invalid_derivative_observations_fail_at_compilation(
    kernel_type, operator, operation
):
    blocks = (FunctionalBlock(Locations(jnp.zeros((1, 1))), operator, "measurement"),)
    kernel = kernel_type()

    def compile():
        if operation == "full":
            return assemble_covariance(blocks, blocks, kernel)
        return assemble_covariance_diagonal(blocks, kernel)

    with pytest.raises(ValueError, match="supports derivative order at most"):
        jax.jit(compile)()


@pytest.mark.parametrize(
    "kernel", [RBFKernel() + Matern12Kernel(), RBFKernel() * Matern32Kernel()]
)
def test_composed_kernel_retains_smoothness_limit(kernel):
    blocks = (FunctionalBlock(Locations(jnp.zeros((1, 1))), Hessian(), "curvature"),)
    with pytest.raises(ValueError, match="supports derivative order at most"):
        assemble_covariance(blocks, blocks, kernel)


@pytest.mark.parametrize("kernel_type", KERNELS)
@pytest.mark.parametrize("scaling", ["isotropic", "ard", "full"])
def test_random_features_recover_kernel_in_multiple_dimensions(kernel_type, scaling):
    kwargs = {
        "isotropic": {"lengthscale": 0.7},
        "ard": {"lengthscale": jnp.array([0.7, 1.4])},
        "full": {"L": jnp.array([[1.4, 0.0], [0.6, 0.7]])},
    }[scaling]
    constructor = GENERALIZED[kernel_type] if scaling == "full" else kernel_type
    kernel = constructor(variance=1.3, **kwargs)
    points = jnp.array([[0.0, 0.0], [0.3, -0.4], [-0.5, 0.2], [0.8, 0.9]])
    features = eqx.filter_jit(
        lambda k: k.random_features(jax.random.key(21), 2, feature_count=32768)
    )(kernel)
    matrix = jax.jit(jax.vmap(features))(points)
    expected = jax.vmap(jax.vmap(kernel, in_axes=(None, 0)), in_axes=(0, None))(
        points, points
    )
    assert jnp.allclose(matrix @ matrix.T, expected, atol=0.035)


@pytest.mark.parametrize("kernel_type", KERNELS)
def test_feature_construction_remains_differentiable_in_metric_parameters(kernel_type):
    factor = CholeskyParameter(jnp.array([[1.2, 0.0], [0.3, 0.8]]))
    kernel = GENERALIZED[kernel_type](
        variance=PositiveParameter(1.3),
        L=factor,
    )
    left, right = jnp.array([0.2, -0.4]), jnp.array([-0.3, 0.6])

    def objective(model):
        features = model.random_features(jax.random.key(12), 2, feature_count=64)
        return jnp.dot(features(left), features(right))

    gradient = eqx.filter_jit(eqx.filter_grad(objective))(kernel)
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(gradient))
    # Balance subtraction error against the central difference's truncation error.
    step = 1e-5 if jax.config.read("jax_enable_x64") else 2e-3
    raw = factor.unconstrained

    def perturbed(offset):
        model = eqx.tree_at(
            lambda k: k.lengthscales.L_parameter.unconstrained, kernel, raw + offset
        )
        return objective(model)

    finite_difference = (perturbed(step) - perturbed(-step)) / (2 * step)
    actual = jnp.sum(gradient.lengthscales.L_parameter.unconstrained)
    assert jnp.allclose(actual, finite_difference, rtol=2e-3, atol=2e-3)


def test_random_features_validate_new_dimensions_and_feature_count():
    kernel = RBFKernel(lengthscale=jnp.ones(2))
    for count in [False, 0, 1.5]:
        with pytest.raises(ValueError, match="feature_count"):
            kernel.random_features(jax.random.key(0), 2, feature_count=count)
    with pytest.raises(ValueError, match="ARD lengthscale shape"):
        kernel.random_features(jax.random.key(0), 1)
    full = GeneralizedRBFKernel(L=jnp.eye(2))
    with pytest.raises(ValueError, match="L dimension"):
        full.random_features(jax.random.key(0), 1)


@pytest.mark.parametrize(
    "kernel_type", [Matern12Kernel, Matern32Kernel, Matern52Kernel]
)
def test_matern_conditioning_and_likelihood_work_with_duplicate_points(kernel_type):
    locations = Locations(jnp.array([[-0.4], [0.2], [0.2], [0.8]]))
    dataset = FunctionalDataset(
        [
            ObservationBlock(
                ObservationDesignBlock(
                    FunctionalBlock(locations, Identity(), "values"),
                    variance=0.05,
                    max_size=locations.count,
                ),
                y=jnp.array([0.1, 0.2, 0.3, 0.5]),
            )
        ]
    )
    kernel = kernel_type(lengthscale=PositiveParameter(0.7))
    gp = GaussianProcess(kernel)
    value, gradient = eqx.filter_jit(
        eqx.filter_value_and_grad(negative_log_marginal_likelihood)
    )(gp, dataset)
    assert jnp.isfinite(value)
    assert jnp.isfinite(
        gradient.kernel.lengthscales.lengthscale_parameter.unconstrained
    )
    prediction = gp.condition(dataset).predict(dataset.query)
    x = locations.points[:, 0]
    matrix = jax.vmap(lambda a: jax.vmap(lambda b: kernel(a, b))(locations.points))(
        locations.points
    )
    observed = matrix + (0.05 + 1e-6) * jnp.eye(x.size)
    expected = matrix @ jnp.linalg.solve(observed, dataset.values_flat)
    assert jnp.allclose(prediction.mean.flat, expected, rtol=1e-4, atol=1e-5)
