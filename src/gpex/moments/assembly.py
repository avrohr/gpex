"""Assembly of prepared functionals into Gaussian mean and covariance arrays."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..blocks import FunctionalBlock, ObservationDesignBlock
from ..kernels.base import Kernel
from .measurements import MeasurementGroup, PreparedFunctionals, prepare_functionals
from .operator_covariance import operator_covariance

Point = Float[Array, "input_dimension"]
Scalar = Float[Array, ""]
FlatMean = Float[Array, "functional_size"]
FlatCovariance = Float[Array, "left_size right_size"]
FlatVector = Float[Array, "functional_size"]
FunctionalColumns = Float[Array, "functional_size *columns"]
GroupColumns = Float[Array, "group_size columns"]
ScalarFunction = Callable[[Point], Scalar]
ScalarKernel = Callable[[Point, Point], Scalar]
Functionals = Sequence[FunctionalBlock | ObservationDesignBlock] | PreparedFunctionals


@dataclass(frozen=True)
class _TileLayout:
    """Tiles of one numerical group, independent of public block boundaries."""

    group: MeasurementGroup
    point_count: int
    points_per_tile: int

    @classmethod
    def from_group(cls, group: MeasurementGroup, tile_size: int | None) -> _TileLayout:
        point_count = group.points.shape[0]
        points_per_tile = point_count
        if tile_size is not None:
            points_per_tile = min(
                point_count, max(1, tile_size // prod(group.output_shape))
            )
        padding = (-point_count) % points_per_tile
        if padding:
            group = jax.tree.map(
                lambda array: jnp.pad(
                    array, ((0, padding), *((0, 0),) * (array.ndim - 1)), mode="edge"
                ),
                group,
            )
        return cls(group, point_count, points_per_tile)

    @property
    def component_count(self) -> int:
        return prod(self.group.output_shape)

    @property
    def size(self) -> int:
        return self.point_count * self.component_count

    @property
    def padded_size(self) -> int:
        return self.group.size

    @property
    def tile_count(self) -> int:
        return self.group.points.shape[0] // self.points_per_tile

    @property
    def tile_width(self) -> int:
        return self.points_per_tile * self.component_count

    def offset(self, index: int | Array) -> int | Array:
        return index * self.tile_width

    def tile(self, index: int | Array) -> MeasurementGroup:
        return jax.tree.map(
            lambda array: jax.lax.dynamic_slice_in_dim(
                array, index * self.points_per_tile, self.points_per_tile
            ),
            self.group,
        )


def assemble_mean(blocks: Functionals, mean: ScalarFunction) -> FlatMean:
    """Apply each functional to the mean and return canonical scalar ordering.

    Accept public blocks or prepared groups. Inactive reserved entries are zero.
    Supports JIT, vmap, and differentiation of numerical inputs.
    """
    measurements = prepare_functionals(blocks)

    def scalar_mean(point: Point) -> Scalar:
        value = jnp.asarray(mean(point))
        if value.ndim != 0:
            raise ValueError("mean functions must return a scalar")
        return value

    parts = [
        eqx.filter_vmap(lambda operator, point: operator.apply(scalar_mean, point))(
            group.operators, group.points
        ).reshape(-1)
        for group in measurements.groups
    ]
    return measurements.mask(measurements.to_canonical(jnp.concatenate(parts)))


def assemble_covariance(
    left_blocks: Functionals,
    right_blocks: Functionals,
    kernel: ScalarKernel,
    *,
    tile_size: int | None = None,
) -> FlatCovariance:
    """Materialize covariance, optionally bounding pairwise working tiles.

    Tiles traverse numerical operator groups rather than public blocks. One
    operator output is never split. The final dense matrix is still allocated.
    With ``None``, each complete group pair is vectorized.
    """
    if tile_size is not None:
        _validate_tile_size(tile_size)
    left, right = prepare_functionals(left_blocks), prepare_functionals(right_blocks)
    if left.input_dimension != right.input_dimension:
        raise ValueError("covariance blocks must use the same input dimension")
    _validate_kernel_compatibility((left, right), kernel)
    rows = []
    for group in left.groups:
        row = jnp.concatenate(
            [
                _assemble_group_covariance(group, other, kernel, tile_size=tile_size)
                for other in right.groups
            ],
            axis=1,
        )
        if len(left.groups) > 1 or len(right.groups) > 1:
            # TODO: Remove this temporary compiler/driver workaround once the
            # mixed CUDA regression passes without it on the fixed toolchain.
            # JAX/jaxlib 0.10.2 on H100 with NVIDIA 615.71.09 can raise an illegal
            # address when fusing values/slopes/averages covariance assembly.
            # Separate completed rows from the final join. Barriers on each
            # individual pair instead expose another failure at smaller sizes.
            row = jax.lax.optimization_barrier(row)
        rows.append(row)
    covariance = left.to_canonical(jnp.concatenate(rows, axis=0))
    return left.mask(right.mask(right.to_canonical(covariance.T)).T)


def assemble_covariance_diagonal(
    blocks: Functionals, kernel: ScalarKernel
) -> FlatVector:
    """Assemble the prior diagonal without allocating a full covariance."""
    measurements = prepare_functionals(blocks)
    _validate_kernel_compatibility((measurements,), kernel)
    parts = []
    for group in measurements.groups:
        components = prod(group.output_shape)

        def point_diagonal(operator, point, components=components):
            covariance = operator_covariance(
                kernel, operator, operator, x=point, y=point
            )
            return jnp.diag(covariance.reshape((components, components)))

        parts.append(
            eqx.filter_vmap(point_diagonal)(group.operators, group.points).reshape(-1)
        )
    return measurements.mask(measurements.to_canonical(jnp.concatenate(parts)))


def apply_covariance(
    blocks: Functionals,
    kernel: ScalarKernel,
    right_hand_side: FunctionalColumns,
    *,
    tile_size: int = 128,
) -> FunctionalColumns:
    """Apply latent covariance to canonical rows without constructing its matrix.

    Accept vectors or arrays with leading dimension equal to the scalar
    functional size. Inactive inputs are ignored and inactive outputs are zero.
    One operator output is indivisible and can exceed ``tile_size``. Pairwise
    tiles are recomputed in reverse mode, bounding working storage in both
    directions. Supports JIT, vmap, and differentiation of numerical inputs.
    General kernels still require quadratic pairwise work per application.
    """
    _validate_tile_size(tile_size)
    measurements = prepare_functionals(blocks)
    _validate_kernel_compatibility((measurements,), kernel)
    right_hand_side = jnp.asarray(right_hand_side)
    if right_hand_side.ndim == 0 or right_hand_side.shape[0] != measurements.size:
        raise ValueError("right-hand side must start with the functional size")
    shape = right_hand_side.shape
    columns = measurements.to_grouped(measurements.mask(right_hand_side)).reshape(
        (measurements.size, prod(shape[1:]))
    )
    layouts = tuple(
        _TileLayout.from_group(group, tile_size) for group in measurements.groups
    )
    inputs = []
    start = 0
    for layout in layouts:
        inputs.append(
            jnp.pad(
                columns[start : start + layout.size],
                ((0, layout.padded_size - layout.size), (0, 0)),
            )
        )
        start += layout.size
    parts = []
    for left in layouts:
        product = jnp.zeros((left.size, columns.shape[1]))
        for right, values in zip(layouts, inputs, strict=True):
            product = product + _apply_group_covariance(left, right, kernel, values)
        parts.append(product)
    result = measurements.to_canonical(jnp.concatenate(parts))
    return measurements.mask(result).reshape(shape)


def _apply_group_covariance(
    left: _TileLayout,
    right: _TileLayout,
    kernel: ScalarKernel,
    values: GroupColumns,
) -> GroupColumns:
    # Checkpoint rows as well as tiles: otherwise reverse mode can retain a
    # separate inner-loop tape for every row, restoring quadratic storage.
    @jax.checkpoint
    def row(left_index):
        @jax.checkpoint
        def accumulate(right_index, total):
            covariance = _assemble_point_covariance(
                left.tile(left_index), right.tile(right_index), kernel
            )
            rhs = jax.lax.dynamic_slice_in_dim(
                values, right.offset(right_index), right.tile_width
            )
            return total + jnp.matmul(
                covariance, rhs, precision=jax.lax.Precision.HIGHEST
            )

        return jax.lax.fori_loop(
            0,
            right.tile_count,
            accumulate,
            jnp.zeros((left.tile_width, values.shape[1])),
        )

    rows = jax.lax.map(row, jnp.arange(left.tile_count))
    return rows.reshape((left.padded_size, values.shape[1]))[: left.size]


def _assemble_group_covariance(
    left: MeasurementGroup,
    right: MeasurementGroup,
    kernel: ScalarKernel,
    *,
    tile_size: int | None,
) -> FlatCovariance:
    left_layout = _TileLayout.from_group(left, tile_size)
    right_layout = _TileLayout.from_group(right, tile_size)
    if left_layout.tile_count == 1 and right_layout.tile_count == 1:
        return _assemble_point_covariance(
            left_layout.group,
            right_layout.group,
            kernel,
        )

    covariance = jnp.zeros((left_layout.padded_size, right_layout.padded_size))

    def fill_tile(index: int, output: FlatCovariance) -> FlatCovariance:
        left_index = index // right_layout.tile_count
        right_index = index % right_layout.tile_count
        tile = _assemble_point_covariance(
            left_layout.tile(left_index),
            right_layout.tile(right_index),
            kernel,
        )
        return jax.lax.dynamic_update_slice(
            output,
            tile,
            (left_layout.offset(left_index), right_layout.offset(right_index)),
        )

    covariance = jax.lax.fori_loop(
        0, left_layout.tile_count * right_layout.tile_count, fill_tile, covariance
    )
    return covariance[: left_layout.size, : right_layout.size]


def _assemble_point_covariance(
    left: MeasurementGroup,
    right: MeasurementGroup,
    kernel: ScalarKernel,
) -> FlatCovariance:
    tensor = eqx.filter_vmap(
        lambda left_operator, left_point: eqx.filter_vmap(
            lambda right_operator, right_point: operator_covariance(
                kernel, left_operator, right_operator, x=left_point, y=right_point
            )
        )(right.operators, right.points)
    )(left.operators, left.points)
    left_rank, right_rank = len(left.output_shape), len(right.output_shape)
    permutation = (
        (0,)
        + tuple(range(2, 2 + left_rank))
        + (1,)
        + tuple(range(2 + left_rank, 2 + left_rank + right_rank))
    )
    return tensor.transpose(permutation).reshape((left.size, right.size))


def _validate_kernel_compatibility(
    measurements: Sequence[PreparedFunctionals], kernel: ScalarKernel
) -> None:
    """Check that measurement operators respect the kernel's derivative limit."""
    if isinstance(kernel, Kernel):
        for item in measurements:
            for group in item.groups:
                if group.operators.derivative_order > kernel.max_derivative_order:
                    raise ValueError(
                        f"{type(kernel).__name__} supports derivative order at most "
                        f"{kernel.max_derivative_order:g}; operator requires "
                        f"order {group.operators.derivative_order}"
                    )


def _validate_tile_size(tile_size: int) -> None:
    if isinstance(tile_size, bool) or not isinstance(tile_size, int) or tile_size < 1:
        raise ValueError("tile_size must be a positive integer")


__all__ = [
    "apply_covariance",
    "assemble_mean",
    "assemble_covariance",
    "assemble_covariance_diagonal",
]
