"""Gaussian-process inference through named linear measurements."""

from . import (
    blocks,
    fitting,
    kernels,
    linalg,
    means,
    moments,
    operators,
    parameters,
)
from .blocks import (
    FunctionalBlock,
    FunctionalDataset,
    FunctionalQuery,
    ObservationBlock,
    ObservationDesign,
    ObservationDesignBlock,
    StructuredValues,
)
from .feature_paths import FeaturePosteriorPath
from .gp import ConditionedPosterior, ConditioningState, GaussianProcess
from .parameters import priors
from .prepared_query import PreparedQuery
from .results import CovarianceResult, MarginalPrediction, Prediction

__all__ = [
    "ConditioningState",
    "ConditionedPosterior",
    "CovarianceResult",
    "FunctionalBlock",
    "FunctionalDataset",
    "FunctionalQuery",
    "GaussianProcess",
    "MarginalPrediction",
    "ObservationBlock",
    "ObservationDesign",
    "ObservationDesignBlock",
    "FeaturePosteriorPath",
    "Prediction",
    "PreparedQuery",
    "StructuredValues",
    "blocks",
    "kernels",
    "means",
    "moments",
    "operators",
    "parameters",
    "priors",
    "linalg",
    "fitting",
]
