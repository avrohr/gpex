"""Immutable fitting results; single-run results are JAX PyTrees."""

from dataclasses import dataclass
from typing import Any

import equinox as eqx
from jaxtyping import Array, Bool, Float, Int

from ._common import Scalar

LossHistory = Float[Array, " steps"]
MultiStartLossHistory = Float[Array, "starts steps"]
StartScalars = Float[Array, " starts"]
StartIntegers = Int[Array, " starts"]
StartBooleans = Bool[Array, " starts"]
ScalarInteger = Int[Array, ""]
ScalarBoolean = Bool[Array, ""]


class FitResult[Model](eqx.Module):
    """Final model and its diagnostics.

    ``losses`` includes the initial objective. L-BFGS fills unused entries with
    NaN; use the first ``step_count + 1`` entries when plotting. ``converged``
    reports L-BFGS's gradient criterion and is always false for fixed-step fits.
    ``evaluation_count`` includes line-search trials. ``optimizer_state`` belongs
    to the final iterate. Reaching the iteration limit returns a finite model;
    encountering a non-finite accepted iterate raises instead.
    """

    model: Model
    losses: LossHistory
    optimizer_state: Any
    loss: Scalar
    gradient_norm: Scalar
    step_count: ScalarInteger
    evaluation_count: ScalarInteger
    converged: ScalarBoolean


@dataclass(frozen=True)
class MultiStartFitResult[Model]:
    """Independent fits in input order, selected by their final objective.

    ``final_losses`` holds one value per start; ``losses`` holds their histories.
    ``model`` and ``loss`` select the start at ``best_index``. Optimizer state
    and all diagnostic arrays have a leading start dimension.
    """

    models: tuple[Model, ...]
    losses: MultiStartLossHistory
    optimizer_state: Any
    final_losses: StartScalars
    gradient_norms: StartScalars
    step_counts: StartIntegers
    evaluation_counts: StartIntegers
    converged: StartBooleans
    best_index: int

    @property
    def model(self) -> Model:
        return self.models[self.best_index]

    @property
    def loss(self) -> Scalar:
        return self.final_losses[self.best_index]


class RandomSubsetFitResult[Model](eqx.Module):
    """Final subset iterate, assessed once on the full reference objective.

    ``subset_losses`` records the objective before each update. Values from
    different subsets are not directly comparable. ``reference_loss`` and
    ``reference_gradient_norm`` describe the returned model on the full data.
    """

    model: Model
    subset_losses: LossHistory
    optimizer_state: Any
    reference_loss: Scalar
    reference_gradient_norm: Scalar
    step_count: ScalarInteger
    evaluation_count: ScalarInteger

    @property
    def loss(self) -> Scalar:
        return self.reference_loss


@dataclass(frozen=True)
class MultiStartRandomSubsetFitResult[Model]:
    """Independent subset fits, selected by their final reference objective."""

    models: tuple[Model, ...]
    subset_losses: MultiStartLossHistory
    optimizer_state: Any
    reference_losses: StartScalars
    reference_gradient_norms: StartScalars
    step_counts: StartIntegers
    evaluation_counts: StartIntegers
    best_index: int

    @property
    def model(self) -> Model:
        return self.models[self.best_index]

    @property
    def loss(self) -> Scalar:
        return self.reference_losses[self.best_index]
