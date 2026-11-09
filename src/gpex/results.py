"""Structured latent predictions and covariance results."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from .blocks import IndexMap, StructuredValues

CovarianceMatrix = Float[Array, "functional_size functional_size"]
CovarianceBlock = Float[Array, "..."]


class CovarianceResult(eqx.Module):
    """Covariance of the latent query outputs, without observation noise.

    Args:
        full: Dense covariance with shape ``(q, q)`` for ``q`` scalar outputs.
        index_map: Static map between named blocks and flattened output indices.

    Attributes:
        full: Dense latent covariance.
        variance: Diagonal entries, accessible by block name in each block's shape.
        index_map: Mapping used for named covariance blocks and variances.
    """

    full: CovarianceMatrix
    index_map: IndexMap = eqx.field(static=True)

    def __init__(self, full: CovarianceMatrix, index_map: IndexMap) -> None:
        full = jnp.asarray(full)
        expected_shape = (index_map.size, index_map.size)
        if full.shape != expected_shape:
            raise ValueError(
                f"covariance has shape {full.shape}; expected {expected_shape}"
            )
        self.full = full
        self.index_map = index_map

    @property
    def variance(self) -> StructuredValues:
        """Marginal variances derived from the full covariance."""
        return StructuredValues(
            jnp.diagonal(self.full, axis1=-2, axis2=-1), self.index_map
        )

    def block(self, left_name: str, right_name: str) -> CovarianceBlock:
        """Return a named cross-covariance with shape ``(*left.shape, *right.shape)``.

        Args:
            left_name: Row block name from the query.
            right_name: Column block name from the query.

        Returns:
            The covariance between the two blocks, preserving their output axes.
                Leading batch axes introduced by ``vmap`` are preserved too.
        """
        left = self.index_map.block(left_name)
        right = self.index_map.block(right_name)
        leading_shape = self.full.shape[:-2]
        return self.full[..., left.slice, right.slice].reshape(
            (*leading_shape, *left.shape, *right.shape)
        )


class Prediction(eqx.Module):
    """Joint latent predictions returned by ``ConditionedPosterior.predict``.

    Attributes:
        mean: Means accessible as ``mean[block_name]``, in each block's shape.
        covariance: Full covariance, named cross-blocks, and marginal variances.
    """

    mean: StructuredValues
    covariance: CovarianceResult


class MarginalPrediction(eqx.Module):
    """Latent means and variances without a full query covariance.

    Both fields contain flat arrays of shape ``(q,)`` and named views matching
    each query block. Leading batch axes from ``vmap`` are preserved.
    """

    mean: StructuredValues
    variance: StructuredValues


__all__ = ["CovarianceResult", "Prediction", "MarginalPrediction"]
