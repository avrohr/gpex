"""Explicit Optax fitting, L-BFGS, and independent multistart orchestration."""

from .lbfgs import fit_lbfgs
from .multistart import fit_multistart, fit_multistart_random_subsets
from .objectives import negative_log_marginal_likelihood, negative_log_posterior
from .optax import fit
from .results import (
    FitResult,
    MultiStartFitResult,
    MultiStartRandomSubsetFitResult,
    RandomSubsetFitResult,
)
from .subsets import fit_random_subsets

__all__ = [
    "FitResult",
    "MultiStartFitResult",
    "MultiStartRandomSubsetFitResult",
    "RandomSubsetFitResult",
    "negative_log_marginal_likelihood",
    "negative_log_posterior",
    "fit",
    "fit_lbfgs",
    "fit_multistart",
    "fit_multistart_random_subsets",
    "fit_random_subsets",
]
