"""Reusable fixed targets for differentiable experimental design."""

from __future__ import annotations

import equinox as eqx

from .blocks import (
    FunctionalQuery,
    ObservationDesignBlock,
    StructuredValues,
)
from .covariance_systems import CovarianceSystem
from .gp import ConditioningState
from .linalg.base import CovarianceProjection, Matrix, symmetrize
from .linalg.dense import CholeskySystem
from .moments.assembly import assemble_covariance
from .moments.measurements import PreparedFunctionals, prepare_functionals
from .results import CovarianceResult


class PreparedQuery(eqx.Module):
    """A fixed target and conditioning state cached for comparing measurements.

    Construct with ``PreparedQuery(conditioning, query)``. Each ``*_after`` call
    evaluates one proposal against the original conditioning state;
    calls do not accumulate measurements. Prepare a new query after changing
    the GP, observation design, or target. The cache is an immutable PyTree.

    Candidate noise and the conditioning's jitter enter the small conditional
    measurement system. Target covariance remains latent.
    Storage is ``O(N*q + q*q)`` for ``N`` observations and ``q`` target outputs.

    Args:
        conditioning: Original conditioning state.
        query: Fixed target locations and operators.

    Attributes:
        conditioning: Original conditioning state.
        covariance: Target covariance before a proposed measurement.
        projection: Cached backend factors for observation-query reductions.
    """

    conditioning: ConditioningState
    target: PreparedFunctionals
    covariance: CovarianceResult
    projection: CovarianceProjection

    def __init__(self, conditioning: ConditioningState, query: FunctionalQuery) -> None:
        target = prepare_functionals(query.blocks)
        prior = assemble_covariance(
            target,
            target,
            conditioning.gp.kernel,
            tile_size=conditioning.gp.covariance_tile_size,
        )
        cross = assemble_covariance(
            conditioning.measurements,
            target,
            conditioning.gp.kernel,
            tile_size=conditioning.gp.covariance_tile_size,
        )
        projection = conditioning.linear_system.project(cross)
        self.conditioning = conditioning
        self.target = target
        self.covariance = CovarianceResult(
            symmetrize(prior - projection.gram()), query.index_map
        )
        self.projection = projection

    def predict_covariance_after(
        self,
        block: ObservationDesignBlock,
    ) -> CovarianceResult:
        """Return target covariance after assimilating a proposed measurement.

        Args:
            block: Proposed measurement design, including its shared noise
                variance. For observed data, pass ``observation.design``.

        Returns:
            Full latent target covariance, equivalent to appending the block
                and predicting at the target, up to numerical solver accuracy.
                Candidate coordinates, operator parameters, and noise can be
                differentiated through this operation.
        """
        system, cross = self._conditional_measurement(block)
        covariance = self.covariance.full - system.covariance_reduction(cross)
        return CovarianceResult(symmetrize(covariance), self.covariance.index_map)

    def predict_variance_after(
        self,
        block: ObservationDesignBlock,
    ) -> StructuredValues:
        """Return target variances after a proposal without a full update matrix.

        Args:
            block: Proposed measurement design, including its shared noise
                variance. For observed data, pass ``observation.design``.

        Returns:
            Differentiable latent variances by target block name. Initial query
                preparation still stores the current full target covariance.
        """
        system, cross = self._conditional_measurement(block)
        variance = self.covariance.variance.flat - system.variance_reduction(cross)
        return StructuredValues(variance, self.covariance.index_map)

    def _conditional_measurement(
        self,
        block: ObservationDesignBlock,
    ) -> tuple[CholeskySystem, Matrix]:
        conditioning = self.conditioning
        candidate = prepare_functionals((block,))
        cross = assemble_covariance(
            conditioning.measurements,
            candidate,
            conditioning.gp.kernel,
            tile_size=conditioning.gp.covariance_tile_size,
        )
        measurement = CovarianceSystem(
            candidate,
            conditioning.gp.kernel,
            block.variances.reshape(-1),
            conditioning.jitter,
        ).materialize(
            tile_size=conditioning.gp.covariance_tile_size,
        )
        projection = conditioning.linear_system.project(cross)
        conditional = measurement - projection.gram()
        # Only the proposed measurements are factorized.
        system = CholeskySystem.from_matrix(conditional, jitter=0.0)
        target_cross = assemble_covariance(
            candidate,
            self.target,
            conditioning.gp.kernel,
            tile_size=conditioning.gp.covariance_tile_size,
        )
        reduction = projection.cross_gram(self.projection)
        return system, target_cross - reduction


__all__ = ["PreparedQuery"]
