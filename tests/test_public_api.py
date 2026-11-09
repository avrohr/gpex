"""Documented workflow imports and discoverable specialist namespaces."""

import pytest

import gpex


@pytest.mark.parametrize(
    "name",
    [
        "GaussianProcess",
        "ConditioningState",
        "ConditionedPosterior",
        "FunctionalBlock",
        "FunctionalDataset",
        "FunctionalQuery",
        "ObservationBlock",
        "ObservationDesign",
        "ObservationDesignBlock",
        "StructuredValues",
        "CovarianceResult",
        "Prediction",
        "MarginalPrediction",
        "PreparedQuery",
        "FeaturePosteriorPath",
    ],
)
def test_workflow_types_are_available_at_root(name):
    assert name in gpex.__all__
    assert isinstance(getattr(gpex, name), type)


@pytest.mark.parametrize(
    "namespace,name",
    [
        ("blocks", "Locations"),
        ("blocks", "IndexMap"),
        ("kernels", "RBFKernel"),
        ("kernels", "RadialKernel"),
        ("kernels", "FeatureKernel"),
        ("kernels", "RadialFeatureKernel"),
        ("kernels", "Lengthscales"),
        ("kernels", "GeneralizedLengthscales"),
        ("kernels", "SeparableKernel"),
        ("kernels", "KernelFactorization"),
        ("means", "ConstantMean"),
        ("moments", "apply_covariance"),
        ("moments", "assemble_mean"),
        ("moments", "assemble_covariance"),
        ("moments", "assemble_covariance_diagonal"),
        ("operators", "Gradient"),
        ("operators", "QuadratureOperator"),
        ("operators", "TensorQuadratureOperator"),
        ("parameters", "Parameter"),
        ("parameters", "CholeskyTransform"),
        ("priors", "GammaPrior"),
        ("linalg", "DenseBackend"),
        ("linalg", "CholeskySystem"),
        ("linalg", "CholeskyProjection"),
        ("linalg", "CovarianceProjection"),
        ("fitting", "negative_log_marginal_likelihood"),
        ("fitting", "negative_log_posterior"),
        ("fitting", "fit"),
        ("fitting", "FitResult"),
    ],
)
def test_specialist_types_and_operations_are_available_in_their_namespaces(
    namespace, name
):
    assert namespace in gpex.__all__
    module = getattr(gpex, namespace)
    assert name in module.__all__
    assert callable(getattr(module, name))
