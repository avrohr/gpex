"""Build the publication overview figure for gpex."""

# Configure JAX precision before importing gpex.
# ruff: noqa: E402

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

jax.config.update("jax_enable_x64", True)

from gpex import (
    FunctionalBlock,
    FunctionalDataset,
    FunctionalQuery,
    GaussianProcess,
    ObservationBlock,
    ObservationDesignBlock,
)
from gpex.blocks import Locations
from gpex.kernels import RBFKernel
from gpex.operators import (
    BoxAverage,
    CumulativeIntegral1D,
    GaussianSmoothing,
    Identity,
    Increment,
    Laplacian,
    PartialDerivative,
)

COLORS = {
    "navy": "#082B59",
    "blue": "#2F80ED",
    "cyan": "#22A6B3",
    "orange": "#F2994A",
    "purple": "#8B5CF6",
    "ink": "#172033",
    "muted": "#66758A",
    "border": "#D8E2EF",
    "grid": "#E8EEF6",
    "paper": "#F7F9FC",
    "white": "#FFFFFF",
}
DRAW_COLORS = ("#8CB9E8", "#B5CDEA")


@dataclass(frozen=True)
class ObservationChannel:
    name: str
    label: str
    symbol: str
    operator: Any
    locations: jax.Array
    variance: float
    observations: jax.Array
    color: str


@dataclass(frozen=True)
class QueryView:
    name: str
    label: str
    symbol: str
    operator: Any
    color: str
    truth: np.ndarray
    mean: np.ndarray
    standard_deviation: np.ndarray
    draws: np.ndarray


def truth_function(point: jax.Array) -> jax.Array:
    """Multiscale function used to generate the synthetic measurements."""
    x = point[0]
    return (
        0.55
        - 0.08 * x
        + 0.65 * jnp.sin(0.85 * x)
        + 0.32 * jnp.cos(2.4 * x + 0.35)
        + 0.12 * jnp.sin(6.5 * x)
    )


def _series(values: jax.Array) -> np.ndarray:
    values = jnp.asarray(values)
    flattened = values.reshape(values.shape[0], -1)
    if flattened.shape[1] != 1:
        raise ValueError("the poster expects one scalar per locations point")
    return np.asarray(flattened[:, 0])


def _draw_series(values: jax.Array) -> np.ndarray:
    values = jnp.asarray(values)
    flattened = values.reshape(values.shape[0], values.shape[1], -1)
    if flattened.shape[2] != 1:
        raise ValueError("the poster expects scalar-output query blocks")
    return np.asarray(flattened[:, :, 0])


def _operator_values(operator: Any, points: jax.Array) -> jax.Array:
    return jax.vmap(lambda point: operator.apply(truth_function, point))(points)


def _plot_axis(axis: Axes, *, show_x: bool = False) -> None:
    axis.set_facecolor(COLORS["white"])
    axis.grid(axis="y", color=COLORS["grid"], linewidth=0.65)
    axis.tick_params(
        axis="both",
        colors=COLORS["muted"],
        labelsize=7.5,
        length=0,
    )
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color(COLORS["border"])
    axis.spines["bottom"].set_linewidth(0.8)
    axis.set_xlim(-4.0, 4.0)
    axis.set_yticks([])
    axis.set_xticks([-4.0, 0.0, 4.0] if show_x else [])


def _draw_operator_glyph(axis: Axes, operator: Any, color: str) -> None:
    """Draw how an operator acts around one locations point."""
    coordinate = np.linspace(-1.05, 1.05, 160)
    curve = 0.10 + 0.34 * np.sin(2.2 * coordinate) + 0.08 * np.cos(4.1 * coordinate)
    center_value = 0.18
    axis.set(xlim=(-1.1, 1.1), ylim=(-0.75, 0.82))
    axis.axis("off")

    if isinstance(operator, Identity):
        axis.plot(coordinate, curve, color=COLORS["muted"], linewidth=1.15)
        axis.vlines(0.0, -0.57, center_value, color=color, linewidth=1.2)
        axis.scatter(
            [0.0],
            [center_value],
            s=31,
            color=color,
            edgecolor=COLORS["white"],
            linewidth=0.8,
            zorder=4,
        )
        symbol = r"$I$"
    elif isinstance(operator, PartialDerivative):
        axis.plot(coordinate, curve, color=COLORS["muted"], linewidth=1.15)
        tangent_x = np.linspace(-0.52, 0.52, 30)
        tangent = center_value + 0.42 * tangent_x
        axis.plot(tangent_x, tangent, color=color, linewidth=2.0)
        axis.annotate(
            "",
            xy=(0.49, tangent[-2]),
            xytext=(0.24, center_value + 0.42 * 0.24),
            arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.5},
        )
        axis.scatter([0.0], [center_value], s=17, color=color, zorder=4)
        symbol = r"$D$"
    elif isinstance(operator, Increment):
        left = -0.48
        right = left + float(operator.displacement[0])
        left_value = float(np.interp(left, coordinate, curve))
        right_value = float(np.interp(right, coordinate, curve))
        axis.plot(coordinate, curve, color=COLORS["muted"], linewidth=1.15)
        axis.vlines(
            [left, right],
            -0.57,
            [left_value, right_value],
            color=color,
            linewidth=0.9,
            linestyles="dotted",
        )
        axis.scatter(
            [left, right],
            [left_value, right_value],
            s=25,
            color=color,
            edgecolor=COLORS["white"],
            linewidth=0.7,
            zorder=4,
        )
        axis.annotate(
            "",
            xy=(right, -0.50),
            xytext=(left, -0.50),
            arrowprops={"arrowstyle": "<->", "color": color, "lw": 1.35},
        )
        axis.text(
            (left + right) / 2,
            -0.44,
            r"$h$",
            color=color,
            fontsize=7.5,
            ha="center",
            va="bottom",
        )
        axis.text(
            left - 0.03,
            left_value + 0.10,
            r"$-$",
            color=color,
            fontsize=10,
            fontweight="bold",
            ha="right",
        )
        axis.text(
            right + 0.03,
            right_value + 0.10,
            r"$+$",
            color=color,
            fontsize=10,
            fontweight="bold",
            ha="left",
        )
        symbol = r"$\delta_h$"
    elif isinstance(operator, Laplacian):
        axis.plot(coordinate, curve, color=COLORS["muted"], linewidth=1.15)
        local_x = np.linspace(-0.52, 0.52, 40)
        local_quadratic = center_value - 0.67 * local_x**2
        axis.plot(local_x, local_quadratic, color=color, linewidth=2.1)
        axis.scatter([0.0], [center_value], s=17, color=color, zorder=4)
        axis.annotate(
            "",
            xy=(0.0, center_value - 0.34),
            xytext=(0.0, center_value - 0.08),
            arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.4},
        )
        symbol = r"$\Delta$"
    elif isinstance(operator, BoxAverage):
        axis.axvspan(-0.66, 0.66, color=color, alpha=0.13, linewidth=0)
        axis.plot(coordinate, curve, color=COLORS["muted"], linewidth=1.15)
        inside = (coordinate >= -0.66) & (coordinate <= 0.66)
        average = float(np.mean(curve[inside]))
        axis.hlines(average, -0.66, 0.66, color=color, linewidth=2.0)
        axis.vlines([-0.66, 0.66], -0.58, -0.45, color=color, linewidth=1.0)
        axis.hlines(-0.515, -0.66, 0.66, color=color, linewidth=1.0)
        symbol = r"$A_r$"
    elif isinstance(operator, GaussianSmoothing):
        weights = np.exp(-0.5 * (coordinate / 0.38) ** 2)
        axis.fill_between(
            coordinate,
            -0.58,
            -0.58 + 0.46 * weights,
            color=color,
            alpha=0.22,
            linewidth=0,
        )
        axis.plot(coordinate, curve, color=COLORS["muted"], linewidth=1.15)
        smoothed = float(np.sum(weights * curve) / np.sum(weights))
        axis.scatter(
            [0.0],
            [smoothed],
            s=26,
            color=color,
            edgecolor=COLORS["white"],
            linewidth=0.7,
            zorder=4,
        )
        symbol = r"$G_\sigma$"
    elif isinstance(operator, CumulativeIntegral1D):
        base = -0.78
        endpoint = 0.62
        interval = (coordinate >= base) & (coordinate <= endpoint)
        axis.plot(coordinate, curve, color=COLORS["muted"], linewidth=1.15)
        axis.fill_between(
            coordinate[interval],
            0.0,
            curve[interval],
            color=color,
            alpha=0.23,
            linewidth=0,
        )
        axis.vlines(
            [base, endpoint],
            -0.57,
            [
                float(np.interp(base, coordinate, curve)),
                float(np.interp(endpoint, coordinate, curve)),
            ],
            color=color,
            linewidth=1.0,
            linestyles="dotted",
        )
        axis.annotate(
            "",
            xy=(endpoint, -0.50),
            xytext=(base, -0.50),
            arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.35},
        )
        axis.text(base, -0.63, r"$x_0$", color=color, fontsize=7, ha="center")
        axis.text(endpoint, -0.63, r"$x$", color=color, fontsize=7, ha="center")
        symbol = r"$J_{x_0}$"
    else:
        raise TypeError(f"no poster glyph for {type(operator).__name__}")

    axis.text(
        -0.98,
        0.66,
        symbol,
        color=color,
        fontsize=9.5,
        fontweight="bold",
        va="top",
    )


def _make_observations(seed: int) -> tuple[FunctionalDataset, list[ObservationChannel]]:
    specifications = (
        (
            "values",
            r"Point values   $I_xf=f(x)$",
            r"$I$",
            Identity(),
            jnp.array([-3.35, -1.15, 1.10, 3.15]),
            0.020,
            COLORS["orange"],
        ),
        (
            "slopes",
            r"Derivative   $D_xf=f'(x)$",
            r"$D$",
            PartialDerivative(0),
            jnp.array([-2.55, 0.05, 2.45]),
            0.045,
            COLORS["cyan"],
        ),
        (
            "averages",
            r"Box average   $A_{r,x}f$",
            r"$A_r$",
            BoxAverage(0.45, quadrature_order=8),
            jnp.array([-1.85, 1.65]),
            0.018,
            COLORS["purple"],
        ),
        (
            "increments",
            r"Increment   $\delta_hf=f(x+h)-f(x)$",
            r"$\delta_h$",
            Increment(jnp.array([0.65])),
            jnp.array([-2.90, -0.45, 2.35]),
            0.025,
            COLORS["blue"],
        ),
    )
    keys = jax.random.split(jax.random.key(seed), len(specifications))
    blocks = []
    channels = []
    for key, (name, label, symbol, operator, locations, variance, color) in zip(
        keys,
        specifications,
        strict=True,
    ):
        points = locations[:, None]
        clean = _operator_values(operator, points)
        observed = clean + jnp.sqrt(variance) * jax.random.normal(key, clean.shape)
        blocks.append(
            ObservationBlock(
                ObservationDesignBlock(
                    FunctionalBlock(Locations(points), operator, name),
                    variance=variance,
                ),
                y=observed,
            )
        )
        channels.append(
            ObservationChannel(
                name,
                label,
                symbol,
                operator,
                locations,
                variance,
                observed,
                color,
            )
        )
    return FunctionalDataset(blocks), channels


def _make_queries(
    posterior: Any,
    locations: Locations,
    *,
    seed: int,
) -> tuple[Any, list[QueryView]]:
    specifications = (
        (
            "smoothed_query",
            r"Gaussian smoothing   $G_{0.45}f$",
            r"$G_\sigma f$",
            GaussianSmoothing(0.45, quadrature_order=8),
            COLORS["blue"],
        ),
        (
            "integral_query",
            r"Cumulative integral   $J_{-3.5}f$",
            r"$J_{x_0}f$",
            CumulativeIntegral1D(-3.5, quadrature_order=10),
            COLORS["orange"],
        ),
        (
            "laplacian_query",
            r"Laplacian   $\Delta f=f''$",
            r"$\Delta f$",
            Laplacian(),
            COLORS["purple"],
        ),
    )
    blocks = [FunctionalBlock(locations, Identity(), "function")]
    blocks.extend(
        FunctionalBlock(locations, operator, name)
        for name, _, _, operator, _ in specifications
    )
    query = FunctionalQuery(blocks)
    prediction = posterior.predict(query)
    samples = posterior.sample(
        jax.random.key(seed + 1),
        query,
        sample_count=2,
        jitter=1e-8,
    )
    views = []
    for name, label, symbol, operator, color in specifications:
        views.append(
            QueryView(
                name,
                label,
                symbol,
                operator,
                color,
                _series(_operator_values(operator, locations.points)),
                _series(prediction.mean[name]),
                np.sqrt(np.maximum(_series(prediction.covariance.variance[name]), 0.0)),
                _draw_series(samples[name]),
            )
        )
    return (prediction, samples), views


def _draw_observation_section(
    figure: Figure,
    x: np.ndarray,
    locations: Locations,
    channels: list[ObservationChannel],
) -> None:
    figure.text(
        0.040,
        0.795,
        r"1  Observation blocks   $y_b=L_bf+\varepsilon_b$",
        color=COLORS["navy"],
        fontsize=11.5,
        fontweight="bold",
    )
    figure.text(
        0.040,
        0.765,
        "One operator and noise model per block.",
        color=COLORS["muted"],
        fontsize=8.4,
    )
    figure.text(
        0.043,
        0.731,
        "Operator action",
        color=COLORS["muted"],
        fontsize=7.5,
        fontweight="bold",
    )
    figure.text(
        0.137,
        0.731,
        r"Simulated $L_bf$ and measurements",
        color=COLORS["muted"],
        fontsize=7.5,
        fontweight="bold",
    )
    positions = (0.625, 0.505, 0.385, 0.265)
    for index, (channel, bottom) in enumerate(zip(channels, positions, strict=True)):
        glyph = figure.add_axes((0.040, bottom, 0.077, 0.092))
        _draw_operator_glyph(glyph, channel.operator, channel.color)
        axis = figure.add_axes((0.137, bottom, 0.145, 0.088))
        _plot_axis(axis, show_x=index == len(positions) - 1)
        truth = _series(_operator_values(channel.operator, locations.points))
        axis.plot(
            x,
            truth,
            color=COLORS["ink"],
            linewidth=1.0,
            linestyle="--",
            alpha=0.78,
        )
        axis.errorbar(
            np.asarray(channel.locations),
            _series(channel.observations),
            yerr=1.96 * np.sqrt(channel.variance),
            fmt="o",
            color=channel.color,
            ecolor=channel.color,
            markersize=4.0,
            elinewidth=0.8,
            capsize=1.8,
            markeredgecolor=COLORS["white"],
            markeredgewidth=0.6,
            zorder=5,
        )
        axis.text(
            0.0,
            1.02,
            channel.label,
            transform=axis.transAxes,
            color=COLORS["ink"],
            fontsize=8.6,
            fontweight="bold",
            va="bottom",
        )


def _draw_posterior_section(
    figure: Figure,
    x: np.ndarray,
    truth: np.ndarray,
    mean: np.ndarray,
    standard_deviation: np.ndarray,
    draws: np.ndarray,
    channels: list[ObservationChannel],
) -> None:
    observation_count = sum(len(channel.locations) for channel in channels)
    figure.text(
        0.510,
        0.795,
        "2  Posterior over the latent function",
        ha="center",
        color=COLORS["navy"],
        fontsize=12,
        fontweight="bold",
    )
    figure.text(
        0.510,
        0.765,
        f"{observation_count} measurements share one covariance system.",
        ha="center",
        color=COLORS["muted"],
        fontsize=8.7,
    )
    axis = figure.add_axes((0.335, 0.390, 0.350, 0.320))
    _plot_axis(axis, show_x=True)
    axis.fill_between(
        x,
        mean - 1.96 * standard_deviation,
        mean + 1.96 * standard_deviation,
        color=COLORS["blue"],
        alpha=0.16,
        linewidth=0,
        label="95% pointwise interval",
    )
    for index, (draw, color) in enumerate(zip(draws, DRAW_COLORS, strict=True)):
        axis.plot(
            x,
            draw,
            color=color,
            linewidth=0.9,
            alpha=0.82,
            label="joint posterior draws" if index == 0 else None,
        )
    axis.plot(
        x,
        truth,
        color=COLORS["ink"],
        linestyle="--",
        linewidth=1.25,
        label="generating function",
    )
    axis.plot(
        x,
        mean,
        color=COLORS["navy"],
        linewidth=2.35,
        label="posterior mean",
    )
    axis.set_ylabel(r"latent function $f(x)$", color=COLORS["ink"])
    axis.set_yticks([-1.0, 0.0, 1.0])
    axis.legend(
        loc="upper center",
        frameon=False,
        ncols=2,
        fontsize=7.3,
        handlelength=1.8,
        columnspacing=1.2,
    )

    strip = figure.add_axes((0.335, 0.275, 0.350, 0.075))
    strip.set(xlim=(-4.0, 4.0), ylim=(-0.6, 3.6))
    strip.set_xticks([-4.0, 0.0, 4.0])
    strip.set_xlabel(
        "observation locations by block",
        fontsize=8,
        color=COLORS["muted"],
    )
    rows = (3.0, 2.0, 1.0, 0.0)
    strip.set_yticks(rows)
    strip.set_yticklabels(
        [channel.symbol for channel in channels],
        fontsize=8.8,
    )
    for label, channel in zip(strip.get_yticklabels(), channels, strict=True):
        label.set_color(channel.color)
    strip.tick_params(axis="both", length=0, labelsize=7.5, pad=2)
    strip.spines[:].set_visible(False)
    for row, channel in zip(rows, channels, strict=True):
        strip.hlines(row, -4.0, 4.0, color=COLORS["grid"], linewidth=0.7)
        strip.scatter(
            np.asarray(channel.locations),
            np.full_like(np.asarray(channel.locations), row),
            s=26,
            color=channel.color,
            edgecolor=COLORS["white"],
            linewidth=0.7,
            zorder=3,
        )


def _draw_query_section(figure: Figure, x: np.ndarray, views: list[QueryView]) -> None:
    figure.text(
        0.735,
        0.795,
        r"3  Query blocks   $Qf\mid y$",
        color=COLORS["navy"],
        fontsize=11.5,
        fontweight="bold",
    )
    figure.text(
        0.735,
        0.765,
        "Every query uses the same posterior.",
        color=COLORS["muted"],
        fontsize=8.4,
    )
    positions = (0.600, 0.430, 0.260)
    for index, (view, bottom) in enumerate(zip(views, positions, strict=True)):
        glyph = figure.add_axes((0.735, bottom + 0.014, 0.060, 0.096))
        _draw_operator_glyph(glyph, view.operator, view.color)
        axis = figure.add_axes((0.810, bottom, 0.150, 0.118))
        _plot_axis(axis, show_x=index == len(positions) - 1)
        axis.fill_between(
            x,
            view.mean - 1.96 * view.standard_deviation,
            view.mean + 1.96 * view.standard_deviation,
            color=view.color,
            alpha=0.16,
            linewidth=0,
        )
        for draw, color in zip(view.draws, DRAW_COLORS, strict=True):
            axis.plot(x, draw, color=color, linewidth=0.8, alpha=0.78)
        axis.plot(
            x,
            view.truth,
            color=COLORS["ink"],
            linewidth=1.0,
            linestyle="--",
        )
        axis.plot(x, view.mean, color=view.color, linewidth=2.0)
        axis.text(
            0.0,
            1.03,
            view.label,
            transform=axis.transAxes,
            color=COLORS["navy"],
            fontsize=10,
            fontweight="bold",
            va="bottom",
        )


def _normalized_correlation(matrix: np.ndarray) -> np.ndarray:
    scale = np.sqrt(np.maximum(np.diag(matrix), 1e-14))
    return matrix / np.outer(scale, scale)


def _draw_workflow(
    figure: Figure,
    covariance: np.ndarray,
    cholesky: np.ndarray,
    channels: list[ObservationChannel],
    views: list[QueryView],
) -> None:
    figure.add_artist(
        Line2D(
            [0.04, 0.96],
            [0.220, 0.220],
            transform=figure.transFigure,
            color=COLORS["border"],
            linewidth=1.0,
        )
    )
    figure.text(
        0.040,
        0.188,
        "How gpex constructs the posterior",
        color=COLORS["navy"],
        fontsize=11.5,
        fontweight="bold",
    )

    steps = (
        (0.040, "1. Define blocks", "Support · operator · values · noise"),
        (0.275, "2. Compile covariance", r"Mixed covariance $K$"),
        (0.515, "3. Factor once", r"$K=CC^\mathsf{T}$"),
        (0.755, "4. Predict new views", r"Reuse $C$ for each query $Q$"),
    )
    for x_position, title, description in steps:
        figure.text(
            x_position,
            0.156,
            title,
            color=COLORS["navy"],
            fontsize=8.8,
            fontweight="bold",
        )
        figure.text(
            x_position,
            0.135,
            description,
            color=COLORS["muted"],
            fontsize=7.2,
        )
    for x_position in (0.247, 0.487, 0.727):
        figure.text(
            x_position,
            0.060,
            "→",
            ha="center",
            va="center",
            color=COLORS["border"],
            fontsize=22,
            fontweight="bold",
        )

    block_axis = figure.add_axes((0.045, 0.018, 0.180, 0.078))
    block_axis.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    block_axis.axis("off")
    node_positions = ((0.12, 0.75), (0.12, 0.25), (0.47, 0.75), (0.47, 0.25))
    target = (0.84, 0.50)
    for channel, position in zip(channels, node_positions, strict=True):
        block_axis.plot(
            [position[0], target[0]],
            [position[1], target[1]],
            color=COLORS["border"],
            linewidth=1.2,
            zorder=0,
        )
        block_axis.scatter(
            [position[0]],
            [position[1]],
            s=330,
            color=channel.color,
            edgecolor=COLORS["white"],
            linewidth=1.2,
            zorder=2,
        )
        block_axis.text(
            *position,
            channel.symbol,
            ha="center",
            va="center",
            color=COLORS["white"],
            fontsize=8.0,
            fontweight="bold",
            zorder=3,
        )
    block_axis.scatter(
        [target[0]],
        [target[1]],
        marker="s",
        s=210,
        color=COLORS["navy"],
        zorder=3,
    )
    block_axis.text(
        target[0],
        0.16,
        "dataset",
        ha="center",
        color=COLORS["muted"],
        fontsize=7.2,
    )

    covariance_axis = figure.add_axes((0.343, 0.018, 0.078, 0.078))
    correlation = np.abs(_normalized_correlation(covariance))
    covariance_axis.imshow(correlation, cmap="Blues", vmin=0.0, vmax=1.0)
    covariance_axis.set_xticks([])
    covariance_axis.set_yticks([])
    boundaries = np.cumsum([len(channel.locations) for channel in channels])[:-1]
    for boundary in boundaries:
        covariance_axis.axhline(boundary - 0.5, color=COLORS["white"], linewidth=1.1)
        covariance_axis.axvline(boundary - 0.5, color=COLORS["white"], linewidth=1.1)
    for spine in covariance_axis.spines.values():
        spine.set_color(COLORS["border"])

    factor_axis = figure.add_axes((0.578, 0.018, 0.078, 0.078))
    magnitude = np.abs(cholesky)
    magnitude /= max(float(np.max(magnitude)), 1e-14)
    factor_axis.imshow(magnitude, cmap="Blues", vmin=0.0, vmax=1.0)
    factor_axis.set_xticks([])
    factor_axis.set_yticks([])
    for spine in factor_axis.spines.values():
        spine.set_color(COLORS["border"])

    query_axis = figure.add_axes((0.770, 0.016, 0.185, 0.080))
    query_axis.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    query_axis.axis("off")
    query_axis.scatter(
        [0.14],
        [0.50],
        s=700,
        color=COLORS["navy"],
        edgecolor="#7DDCE3",
        linewidth=1.4,
        zorder=3,
    )
    query_axis.text(
        0.14,
        0.50,
        r"$f\mid y$",
        ha="center",
        va="center",
        color=COLORS["white"],
        fontsize=8.5,
        fontweight="bold",
        zorder=4,
    )
    spark_x = np.linspace(0.48, 0.98, 70)
    for view, y_position, phase in zip(
        views,
        (0.78, 0.50, 0.22),
        (0.0, 0.7, 1.4),
        strict=True,
    ):
        query_axis.annotate(
            "",
            xy=(0.44, y_position),
            xytext=(0.25, 0.50),
            arrowprops={"arrowstyle": "-|>", "color": COLORS["border"], "lw": 1.0},
        )
        spark = y_position + 0.055 * np.sin(13.0 * spark_x + phase)
        query_axis.fill_between(
            spark_x,
            spark - 0.045,
            spark + 0.045,
            color=view.color,
            alpha=0.14,
            linewidth=0,
        )
        query_axis.plot(spark_x, spark, color=view.color, linewidth=1.6)
        query_axis.text(
            0.46,
            y_position + 0.075,
            view.symbol,
            color=view.color,
            fontsize=7.3,
            fontweight="bold",
        )


def build_intro_poster(
    gp: GaussianProcess,
    *,
    seed: int = 910,
) -> tuple[Figure, dict[str, float]]:
    """Build the gpex overview figure from public inference APIs."""
    x = jnp.linspace(-4.0, 4.0, 241)
    locations = Locations(x[:, None])
    dataset, channels = _make_observations(seed)
    posterior = gp.condition(dataset, jitter=1e-9)
    (prediction, samples), views = _make_queries(posterior, locations, seed=seed)

    function_truth = np.asarray(jax.vmap(truth_function)(locations.points))
    function_mean = _series(prediction.mean["function"])
    function_sd = np.sqrt(
        np.maximum(_series(prediction.covariance.variance["function"]), 0.0)
    )
    function_draws = _draw_series(samples["function"])

    figure = plt.figure(figsize=(16.0, 9.0), facecolor=COLORS["paper"])
    icon_axis = figure.add_axes((0.037, 0.880, 0.056, 0.096))
    icon_axis.imshow(
        plt.imread(Path(__file__).parents[1] / "docs" / "assets" / "gpex_icon_64.png")
    )
    icon_axis.axis("off")
    figure.text(
        0.105,
        0.951,
        "gpex",
        ha="left",
        va="center",
        color=COLORS["navy"],
        fontsize=18,
        fontweight="bold",
    )
    figure.text(
        0.105,
        0.909,
        "Gaussian processes for experimental design in JAX",
        ha="left",
        color=COLORS["ink"],
        fontsize=20,
        fontweight="bold",
    )
    figure.text(
        0.105,
        0.870,
        "Condition on heterogeneous linear measurements and predict new linear views.",
        ha="left",
        color=COLORS["muted"],
        fontsize=10.5,
    )
    figure.add_artist(
        Line2D(
            [0.04, 0.96],
            [0.842, 0.842],
            transform=figure.transFigure,
            color=COLORS["border"],
            linewidth=1.0,
        )
    )
    for x_position in (0.305, 0.715):
        figure.add_artist(
            Line2D(
                [x_position, x_position],
                [0.245, 0.815],
                transform=figure.transFigure,
                color=COLORS["border"],
                linewidth=0.9,
            )
        )

    x_plot = np.asarray(x)
    _draw_observation_section(figure, x_plot, locations, channels)
    _draw_posterior_section(
        figure,
        x_plot,
        function_truth,
        function_mean,
        function_sd,
        function_draws,
        channels,
    )
    _draw_query_section(figure, x_plot, views)
    lower = np.asarray(posterior.conditioning.linear_system.lower)
    _draw_workflow(figure, lower @ lower.T, lower, channels, views)

    metrics = {
        "latent_rmse": float(np.sqrt(np.mean((function_mean - function_truth) ** 2))),
        "latent_mean_standard_deviation": float(np.mean(function_sd)),
    }
    return figure, metrics


def main() -> None:
    """Render the publication assets."""
    gp = GaussianProcess(RBFKernel(variance=0.90, lengthscale=0.5))
    figure, metrics = build_intro_poster(gp)
    output = Path(__file__).parents[1] / "docs" / "assets"
    figure.savefig(
        output / "gpex_intro_poster.svg",
        facecolor=figure.get_facecolor(),
    )
    print(
        "latent posterior: "
        f"RMSE={metrics['latent_rmse']:.3f}, "
        f"mean SD={metrics['latent_mean_standard_deviation']:.3f}"
    )


if __name__ == "__main__":
    main()
