"""Render the small PNG used by the overview-figure generator."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "docs" / "assets"
BASE_SIZE = 256
WORK_SIZE = 2048
SCALE = WORK_SIZE / BASE_SIZE


def scaled(point: tuple[float, float]) -> tuple[int, int]:
    """Scale a point from the SVG coordinate system to the render canvas."""
    return round(point[0] * SCALE), round(point[1] * SCALE)


def cubic_bezier(
    start: tuple[float, float],
    control_1: tuple[float, float],
    control_2: tuple[float, float],
    end: tuple[float, float],
    *,
    samples: int = 80,
) -> list[tuple[int, int]]:
    """Sample one cubic Bezier segment in render coordinates."""
    points = []
    for value in np.linspace(0.0, 1.0, samples, endpoint=False):
        inverse = 1.0 - value
        x = (
            inverse**3 * start[0]
            + 3.0 * inverse**2 * value * control_1[0]
            + 3.0 * inverse * value**2 * control_2[0]
            + value**3 * end[0]
        )
        y = (
            inverse**3 * start[1]
            + 3.0 * inverse**2 * value * control_1[1]
            + 3.0 * inverse * value**2 * control_2[1]
            + value**3 * end[1]
        )
        points.append(scaled((x, y)))
    return points


def curve_points() -> list[tuple[int, int]]:
    """Return the GP curve shared by the uncertainty band and mean line."""
    segments = (
        ((51, 156), (75, 105), (96, 101), (117, 131)),
        ((117, 131), (139, 163), (158, 180), (180, 148)),
        ((180, 148), (194, 128), (202, 107), (210, 91)),
    )
    points = [point for segment in segments for point in cubic_bezier(*segment)]
    points.append(scaled(segments[-1][-1]))
    return points


def rounded_line(
    layer: Image.Image,
    points: list[tuple[int, int]],
    *,
    fill: tuple[int, int, int, int],
    width: int,
) -> None:
    """Draw an antialiased polyline with round joins and end caps."""
    draw = ImageDraw.Draw(layer)
    draw.line(points, fill=fill, width=width, joint="curve")
    radius = width // 2
    for x, y in points:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)


def render() -> Image.Image:
    """Render the icon on a transparent canvas."""
    y, x = np.mgrid[0:WORK_SIZE, 0:WORK_SIZE]
    diagonal = (0.45 * x + 0.55 * y) / (WORK_SIZE - 1)
    start = np.array([23.0, 63.0, 115.0])
    end = np.array([8.0, 29.0, 61.0])
    rgb = start[None, None, :] * (1.0 - diagonal[..., None])
    rgb += end[None, None, :] * diagonal[..., None]

    glow_x, glow_y = scaled((180, 70))
    radius = 165 * SCALE
    glow = np.exp(-2.4 * ((x - glow_x) ** 2 + (y - glow_y) ** 2) / radius**2)
    glow_color = np.array([62.0, 202.0, 211.0])
    rgb = rgb * (1.0 - 0.18 * glow[..., None])
    rgb += glow_color[None, None, :] * 0.18 * glow[..., None]
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    mask = Image.new("L", (WORK_SIZE, WORK_SIZE), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle(
        (*scaled((8, 8)), *scaled((248, 248))),
        radius=round(52 * SCALE),
        fill=255,
    )
    background = Image.fromarray(rgb, mode="RGB").convert("RGBA")
    background.putalpha(mask)
    canvas = Image.new("RGBA", (WORK_SIZE, WORK_SIZE), (0, 0, 0, 0))
    canvas.alpha_composite(background)

    curve = curve_points()
    uncertainty = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    rounded_line(
        uncertainty,
        curve,
        fill=(114, 216, 228, 43),
        width=round(42 * SCALE),
    )
    canvas.alpha_composite(uncertainty)

    mean = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    rounded_line(
        mean,
        curve,
        fill=(247, 251, 255, 255),
        width=round(11 * SCALE),
    )
    canvas.alpha_composite(mean)

    foreground = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    bracket_color = (72, 202, 212, 255)
    bracket_width = round(13 * SCALE)
    left_bracket = [
        scaled(point)
        for point in ((68, 48), (43, 48), (34, 58), (34, 198), (43, 208), (68, 208))
    ]
    right_bracket = [
        scaled(point)
        for point in (
            (188, 48),
            (213, 48),
            (222, 58),
            (222, 198),
            (213, 208),
            (188, 208),
        )
    ]
    rounded_line(foreground, left_bracket, fill=bracket_color, width=bracket_width)
    rounded_line(foreground, right_bracket, fill=bracket_color, width=bracket_width)

    draw = ImageDraw.Draw(foreground)
    for center, color in (
        ((61, 139), (241, 163, 101, 255)),
        ((117, 131), (72, 202, 212, 255)),
        ((180, 148), (228, 106, 145, 255)),
    ):
        cx, cy = scaled(center)
        outer = round(10 * SCALE)
        inner = round(7.4 * SCALE)
        draw.ellipse(
            (cx - outer, cy - outer, cx + outer, cy + outer),
            fill=(16, 45, 87, 255),
        )
        draw.ellipse((cx - inner, cy - inner, cx + inner, cy + inner), fill=color)
    canvas.alpha_composite(foreground)

    border = ImageDraw.Draw(canvas)
    border.rounded_rectangle(
        (*scaled((8, 8)), *scaled((248, 248))),
        radius=round(52 * SCALE),
        outline=(125, 219, 226, 72),
        width=round(2 * SCALE),
    )
    return canvas


def main() -> None:
    """Write the small PNG used inside the overview figure."""
    icon = render()
    icon.resize((64, 64), Image.Resampling.LANCZOS).save(
        ASSET_DIR / "gpex_icon_64.png",
        optimize=True,
    )


if __name__ == "__main__":
    main()
