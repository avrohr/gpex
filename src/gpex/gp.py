"""Exact Gaussian-process inference."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from cholops import cholesky
from jaxtyping import Array, Float, PRNGKeyArray

from .blocks import (
    FunctionalDataset,
    FunctionalQuery,
    FunctionalValues,
    IndexMap,
    Locations,
    ObservationDesign,
    StructuredValues,
)
from .covariance_systems import CovarianceSystem, CovarianceSystemUpdate
from .kernels.base import Kernel
from .linalg.base import LinearSystem, SolverConfiguration, symmetrize
from .linalg.dense import DenseBackend
from .means import Mean, ZeroMean
from .moments.assembly import (
    ScalarFunction,
    assemble_covariance,
    assemble_covariance_diagonal,
    assemble_mean,
)
from .moments.measurements import (
    PreparedFunctionals,
    measurement_block_layout,
    prepare_functionals,
)
from .results import (
    CovarianceMatrix,
    CovarianceResult,
    MarginalPrediction,
    Prediction,
)
from .validation.checks import validate_jitter

ObservationVector = Float[Array, "observation_size"]
FunctionalVector = Float[Array, "functional_size"]
SampleMatrix = Float[Array, "samples functional_size"]
QueryCross = Float[Array, "observation_size query_size"]


class GaussianProcess[KernelType: Kernel](eqx.Module):
    """A scalar latent GP whose observations and queries may use linear operators.

    Args:
        kernel: Scalar covariance kernel evaluated on points of shape ``(d,)``.
        mean: Scalar mean function. ``None`` (the default) selects ``ZeroMean``.
        covariance_tile_size: Positive tile size for materialized covariance
            construction. Defaults to ``None`` for untiled construction.
        solver: Numerical backend, such as ``DenseBackend()``.
            ``None`` (the default) selects dense Cholesky inference.
    """

    kernel: KernelType
    mean: Mean | ScalarFunction
    covariance_tile_size: int | None = eqx.field(static=True)
    solver: SolverConfiguration

    def __init__(
        self,
        kernel: KernelType,
        mean: Mean | ScalarFunction | None = None,
        *,
        covariance_tile_size: int | None = None,
        solver: SolverConfiguration | None = None,
    ) -> None:
        if covariance_tile_size is not None and (
            isinstance(covariance_tile_size, bool)
            or not isinstance(covariance_tile_size, int)
            or covariance_tile_size < 1
        ):
            raise ValueError("covariance_tile_size must be a positive integer or None")
        self.kernel = kernel
        self.mean = ZeroMean() if mean is None else mean
        self.covariance_tile_size = covariance_tile_size
        self.solver = DenseBackend() if solver is None else solver

    def prior_mean(self, query: FunctionalQuery) -> StructuredValues:
        """Return prior means for a query, with named arrays in each block's shape.

        Args:
            query: Locations and linear operators to evaluate.

        Returns:
            Structured means with ``flat`` shape ``(q,)`` for ``q`` scalar outputs.
        """
        flat = assemble_mean(query.blocks, self.mean)
        return StructuredValues(flat, query.index_map)

    def prior_covariance(self, query: FunctionalQuery) -> CovarianceResult:
        """Return prior covariance for a query, without observation noise.

        Args:
            query: Locations and linear operators to evaluate.

        Returns:
            Full covariance of shape ``(q, q)``, named blocks, and variances.
        """
        covariance = assemble_covariance(
            query.blocks,
            query.blocks,
            self.kernel,
            tile_size=self.covariance_tile_size,
        )
        covariance = symmetrize(covariance)
        return CovarianceResult(covariance, query.index_map)

    def prior_variance(self, query: FunctionalQuery) -> StructuredValues:
        """Return latent marginal variances without forming a full covariance.

        Args:
            query: Locations and linear operators to evaluate.

        Returns:
            Variances accessible by block name, in each block's output shape.
        """
        variance = assemble_covariance_diagonal(query.blocks, self.kernel)
        return StructuredValues(variance, query.index_map)

    def sample_prior(
        self,
        key: PRNGKeyArray,
        query: FunctionalQuery,
        *,
        sample_count: int = 1,
        jitter: float = 1e-6,
    ) -> StructuredValues:
        """Draw joint samples at a fixed query from the latent prior.

        Args:
            key: JAX random key; supply a fresh key for independent draws.
            query: Locations and linear operators to sample jointly.
            sample_count: Number of draws. Defaults to 1.
            jitter: Diagonal stabilization for sampling. Defaults to ``1e-6``.

        Returns:
            Samples with ``flat`` shape ``(sample_count, q)`` and named arrays of shape
                ``(sample_count, *block.shape)``.
        """
        mean = self.prior_mean(query)
        covariance = self.prior_covariance(query)
        samples = _sample_gaussian(
            key, mean.flat, covariance.full, sample_count, jitter
        )
        return StructuredValues(samples, query.index_map)

    def prepare_conditioning(
        self,
        design: ObservationDesign | FunctionalDataset,
        *,
        jitter: float = 1e-6,
    ) -> ConditioningState[KernelType]:
        """Prepare conditional covariance without using observation outcomes.

        Args:
            design: Measurement locations, operators, and noise. A dataset is
                also accepted; its observed values are ignored.
            jitter: Diagonal stabilization of the observation covariance.
                Defaults to ``1e-6``.

        Returns:
            Reusable conditioning state for solves, covariance prediction, and
            design updates.
        """
        if isinstance(design, FunctionalDataset):
            design = design.design
        measurements = prepare_functionals(design.blocks)
        system = CovarianceSystem(
            measurements,
            self.kernel,
            design.variance_flat,
            jitter,
        )
        linear_system = self.solver.prepare(
            system,
            materialization_tile_size=self.covariance_tile_size,
            active=measurements.active,
        )
        return ConditioningState(
            self,
            system,
            linear_system,
            design.index_map,
            measurement_block_layout(design.blocks),
        )

    def condition(
        self, dataset: FunctionalDataset, *, jitter: float = 1e-6
    ) -> ConditionedPosterior[KernelType]:
        """Condition on a dataset with ``gp.condition(dataset, jitter=1e-6)``.

        The callable captures this immutable GP and can be compiled directly:
        ``condition = eqx.filter_jit(gp.condition)``. Reuse it across datasets
        with the same reserved block structure and dtypes. Values, locations,
        noise and active counts may change without compiling again. To change
        or differentiate model parameters, pass the GP into a compiled function.

        Returns a posterior for the current contents on every call. This does
        not reuse a previous factorization. Dense work scales with the reserved
        scalar capacity, including unused slots.

        Args:
            dataset: Named observation blocks with reserved locations, observed
                operator outputs, active counts and scalar noise variances.
            jitter: Diagonal stabilization of the observation covariance.
                Defaults to ``1e-6``. Unused slots retain a unit diagonal.

        Returns:
            A posterior supporting prediction and sampling. Failed dense
                factorization propagates non-finite numerical results.
        """
        conditioning = self.prepare_conditioning(dataset, jitter=jitter)
        observation_mean = assemble_mean(conditioning.measurements, self.mean)
        values = dataset.values_flat
        alpha = conditioning.solve(values - observation_mean)
        return ConditionedPosterior(conditioning, values, alpha)


class ConditioningState[KernelType: Kernel](eqx.Module):
    """Outcome-independent covariance state created by ``prepare_conditioning``.

    Use ``GaussianProcess.prepare_conditioning`` to construct this state from a
    design or dataset. Its solves act on flattened scalar observation outputs.
    """

    gp: GaussianProcess[KernelType]
    measurements: PreparedFunctionals
    variance_flat: ObservationVector
    index_map: IndexMap = eqx.field(static=True)
    block_layout: tuple[tuple[int, int], ...] = eqx.field(static=True)
    linear_system: LinearSystem
    jitter: float = eqx.field(static=True)

    def __init__(
        self,
        gp: GaussianProcess[KernelType],
        system: CovarianceSystem,
        linear_system: LinearSystem,
        index_map: IndexMap,
        block_layout: tuple[tuple[int, int], ...],
    ) -> None:
        self.gp = gp
        self.measurements = system.measurements
        self.variance_flat = system.variance
        self.index_map = index_map
        self.block_layout = block_layout
        self.linear_system = linear_system
        self.jitter = system.jitter

    def solve(
        self,
        right_hand_side: FunctionalVector,
    ) -> FunctionalVector:
        """Solve the stabilized observation covariance system.

        Args:
            right_hand_side: Array with shape ``(N, *rhs_shape)``, where ``N`` is
                the number of scalar observation outputs.

        Returns:
            Solution with the same shape as ``right_hand_side``.
        """
        return self.linear_system.solve(right_hand_side)

    def predict_covariance(self, query: FunctionalQuery) -> CovarianceResult:
        """Return conditional latent covariance at a query, without outcome values.

        Args:
            query: Locations and operators with the training input dimension.

        Returns:
            Full covariance of shape ``(q, q)``, named blocks, and variances.
                Observation noise is not added to the query covariance.
        """
        prior = assemble_covariance(
            query.blocks,
            query.blocks,
            self.gp.kernel,
            tile_size=self.gp.covariance_tile_size,
        )
        cross = _assemble_query_cross(self, query)
        covariance = symmetrize(prior - self.linear_system.covariance_reduction(cross))
        return CovarianceResult(covariance, query.index_map)

    def predict_variance(self, query: FunctionalQuery) -> StructuredValues:
        """Return conditional latent variances without a full query covariance.

        Args:
            query: Locations and operators with the training input dimension.

        Returns:
            Variances accessible by block name, without observation noise.
                Storage still includes the observation-query cross-covariance.
        """
        prior = assemble_covariance_diagonal(query.blocks, self.gp.kernel)
        cross = _assemble_query_cross(self, query)
        variance = prior - self.linear_system.variance_reduction(cross)
        return StructuredValues(variance, query.index_map)

    def append(
        self, locations: Locations, *, block: str
    ) -> ConditioningState[KernelType]:
        """Activate locations in a declared block, retaining its operator/noise.

        Existing measurements and numerical settings stay fixed. Shapes and
        index maps retain their reserved sizes. Unknown names and capacity
        overflow raise, including overflow under JIT.
        """
        update = self._prepare_update(locations, block=block)
        linear_system = self.linear_system.append(update)
        return ConditioningState(
            self.gp, update.system, linear_system, self.index_map, self.block_layout
        )

    def append_solve(
        self,
        locations: Locations,
        solution: ObservationVector,
        right_hand_side: ObservationVector,
        *,
        block: str,
    ) -> tuple[ConditioningState[KernelType], ObservationVector]:
        """Activate locations and update a canonical-order vector solution.

        ``solution`` and the new RHS both have the reserved scalar size. The
        RHS must retain all existing active entries and be zero in unused slots.
        Dense updates reuse the factor and update the existing coefficients.
        """
        update = self._prepare_update(locations, block=block)
        linear_system, updated_solution = self.linear_system.append_solve(
            update, solution, right_hand_side
        )
        conditioning = ConditioningState(
            self.gp, update.system, linear_system, self.index_map, self.block_layout
        )
        return conditioning, updated_solution

    def _prepare_update(
        self, locations: Locations, *, block: str
    ) -> CovarianceSystemUpdate:
        index = self.index_map.block(block)
        group_index, location_start = self.block_layout[
            self.index_map.blocks.index(index)
        ]
        measurements, appended, indices = self.measurements.append(
            locations,
            group_index=group_index,
            location_start=location_start,
            capacity=index.shape[0],
            scalar_start=index.start,
        )
        system = CovarianceSystem(
            measurements, self.gp.kernel, self.variance_flat, self.jitter
        )
        return CovarianceSystemUpdate(system, self.measurements, appended, indices)


class ConditionedPosterior[KernelType: Kernel](eqx.Module):
    """An immutable posterior state supporting streaming updates."""

    conditioning: ConditioningState[KernelType]
    y_flat: ObservationVector
    alpha: ObservationVector

    @property
    def observation_size(self) -> int:
        """Number of flattened scalar observation functionals in the state."""
        return self.conditioning.index_map.size

    def predict_mean(self, query: FunctionalQuery) -> StructuredValues:
        """Return posterior means in each query block's output shape.

        Args:
            query: Locations and operators with the training input dimension.

        Returns:
            Structured means with ``flat`` shape ``(q,)`` and named block arrays.
        """
        prior_mean = assemble_mean(query.blocks, self.conditioning.gp.mean)
        cross = _assemble_query_cross(self.conditioning, query)
        flat = prior_mean + jnp.matmul(
            cross.T, self.alpha, precision=jax.lax.Precision.HIGHEST
        )
        return StructuredValues(flat, query.index_map)

    def predict_covariance(self, query: FunctionalQuery) -> CovarianceResult:
        """Delegate to
        [`predict_covariance`][gpex.gp.ConditioningState.predict_covariance].
        """
        return self.conditioning.predict_covariance(query)

    def predict_variance(self, query: FunctionalQuery) -> StructuredValues:
        """Delegate to
        [`predict_variance`][gpex.gp.ConditioningState.predict_variance].
        """
        return self.conditioning.predict_variance(query)

    def predict_marginals(self, query: FunctionalQuery) -> MarginalPrediction:
        """Return latent means and marginal variances, sharing cross-covariance work.

        The result contains named arrays in each query block's output shape and
        can cross an outer ``jit`` or ``vmap`` boundary. Storage includes the
        observation/query cross-covariance, but no full query covariance.
        Observation noise is excluded.
        """
        conditioning = self.conditioning
        cross = _assemble_query_cross(conditioning, query)
        mean = assemble_mean(query.blocks, conditioning.gp.mean) + jnp.matmul(
            cross.T, self.alpha, precision=jax.lax.Precision.HIGHEST
        )
        variance = assemble_covariance_diagonal(query.blocks, conditioning.gp.kernel)
        variance = variance - conditioning.linear_system.variance_reduction(cross)
        return MarginalPrediction(
            StructuredValues(mean, query.index_map),
            StructuredValues(variance, query.index_map),
        )

    def predict(self, query: FunctionalQuery) -> Prediction:
        """Return the joint latent mean and covariance at a query.

        Args:
            query: Named blocks of locations and linear operators to predict.

        Returns:
            A ``Prediction`` with structured means and a full ``(q, q)`` covariance for
                ``q`` flattened query outputs. The complete result is a PyTree and can
                be returned from JIT-compiled functions.
        """
        conditioning = self.conditioning
        gp = conditioning.gp
        cross = _assemble_query_cross(conditioning, query)
        prior_mean = assemble_mean(query.blocks, gp.mean)
        mean = StructuredValues(
            prior_mean
            + jnp.matmul(cross.T, self.alpha, precision=jax.lax.Precision.HIGHEST),
            query.index_map,
        )
        prior_covariance = assemble_covariance(
            query.blocks,
            query.blocks,
            gp.kernel,
            tile_size=gp.covariance_tile_size,
        )
        covariance = symmetrize(
            prior_covariance - conditioning.linear_system.covariance_reduction(cross)
        )
        return Prediction(mean, CovarianceResult(covariance, query.index_map))

    def sample(
        self,
        key: PRNGKeyArray,
        query: FunctionalQuery,
        *,
        sample_count: int = 1,
        jitter: float = 1e-6,
    ) -> StructuredValues:
        """Draw joint latent posterior samples at a fixed query.

        Args:
            key: JAX random key; supply a fresh key for independent draws.
            query: Locations and linear operators to sample jointly.
            sample_count: Number of draws. Defaults to 1.
            jitter: Diagonal stabilization for sampling. Defaults to ``1e-6``.

        Returns:
            Samples with ``flat`` shape ``(sample_count, q)`` and named arrays of shape
                ``(sample_count, *block.shape)``. Observation noise is excluded. Use
                ``FeaturePosteriorPath(posterior, key)`` for a callable approximate
                sample path.
        """
        prediction = self.predict(query)
        samples = _sample_gaussian(
            key, prediction.mean.flat, prediction.covariance.full, sample_count, jitter
        )
        return StructuredValues(samples, query.index_map)

    def append(
        self,
        locations: Locations,
        values: FunctionalValues,
        *,
        block: str,
    ) -> ConditionedPosterior[KernelType]:
        """Add observations to a declared block without enlarging storage.

        ``locations`` has shape ``(m, d)``. ``values`` has shape
        ``(m, *operator_output_shape)`` for the selected block. Its operator,
        noise variance, model parameters and previous observations remain fixed.
        Returns a new posterior; overflow raises instead of resizing.

        Repeated fixed-shape batches into the same block reuse an outer JIT.
        Dense systems update their factor and coefficients through the Schur
        complement.
        """
        index = self.conditioning.index_map.block(block)
        values = jnp.asarray(values)
        expected = (locations.count, *index.shape[1:])
        if values.shape != expected:
            raise ValueError(
                f"appended values have shape {values.shape}; expected {expected}"
            )
        update = self.conditioning._prepare_update(locations, block=block)
        y_flat = self.y_flat.at[update.indices].set(values.reshape(-1))
        right_hand_side = y_flat - assemble_mean(
            update.system.measurements, self.conditioning.gp.mean
        )
        linear_system, alpha = self.conditioning.linear_system.append_solve(
            update, self.alpha, right_hand_side
        )
        conditioning = ConditioningState(
            self.conditioning.gp,
            update.system,
            linear_system,
            self.conditioning.index_map,
            self.conditioning.block_layout,
        )
        return ConditionedPosterior(conditioning, y_flat, alpha)


def _assemble_query_cross(
    conditioning: ConditioningState, query: FunctionalQuery
) -> QueryCross:
    """Assemble the observation/query cross-covariance in canonical orientation."""
    return assemble_covariance(
        conditioning.measurements,
        query.blocks,
        conditioning.gp.kernel,
        tile_size=conditioning.gp.covariance_tile_size,
    )


def _sample_gaussian(
    key: PRNGKeyArray,
    mean: FunctionalVector,
    covariance: CovarianceMatrix,
    sample_count: int,
    jitter: float,
) -> SampleMatrix:
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    validate_jitter(jitter)
    stabilized = covariance + jitter * jnp.eye(covariance.shape[0])
    lower = cholesky(stabilized)
    standard_normal = jax.random.normal(key, (sample_count, mean.shape[0]))
    return mean + jnp.matmul(
        standard_normal, lower.T, precision=jax.lax.Precision.HIGHEST
    )


__all__ = ["GaussianProcess", "ConditioningState", "ConditionedPosterior"]
