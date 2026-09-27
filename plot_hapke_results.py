"""Create a dependency-light comparison plot from Hapke result CSV files."""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUTPUT_DIRECTORY = ROOT / "outputs" / "hapke"
SERPENTINE_CSV = OUTPUT_DIRECTORY / "serpentine_hapke_test.csv"
MIXTURE_CSV = OUTPUT_DIRECTORY / "mixture_hapke_test.csv"
OUTPUT = OUTPUT_DIRECTORY / "hapke_test_comparison.png"


def read_result(path: Path, xmax: float = 10.0) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            wavelength = float(row["wavelength_microns"])
            if wavelength <= xmax:
                points.append((wavelength, float(row["reflectance"])))
    return points


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/arialbd.ttf") if bold else Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/meiryob.ttc") if bold else Path("C:/Windows/Fonts/meiryo.ttc"),
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def main() -> None:
    scale = 2
    width, height = 1400 * scale, 850 * scale
    left, right, top, bottom = 145 * scale, 60 * scale, 125 * scale, 125 * scale
    plot_left, plot_right = left, width - right
    plot_top, plot_bottom = top, height - bottom

    series = [
        ("Serpentine (single component)", read_result(SERPENTINE_CSV), "#1565C0"),
        ("70% serpentine + 30% organics", read_result(MIXTURE_CSV), "#D84315"),
    ]
    ymax_data = max(y for _, points, _ in series for _, y in points)
    ymax = max(0.4, (int(ymax_data / 0.05) + 2) * 0.05)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = font(31 * scale, bold=True)
    subtitle_font = font(18 * scale)
    label_font = font(21 * scale)
    tick_font = font(17 * scale)
    legend_font = font(18 * scale)

    def px(x: float) -> float:
        return plot_left + (x / 10.0) * (plot_right - plot_left)

    def py(y: float) -> float:
        return plot_bottom - (y / ymax) * (plot_bottom - plot_top)

    for x in range(0, 11):
        xx = px(float(x))
        draw.line((xx, plot_top, xx, plot_bottom), fill="#E5E7EB", width=scale)
        text = str(x)
        box = draw.textbbox((0, 0), text, font=tick_font)
        draw.text((xx - (box[2] - box[0]) / 2, plot_bottom + 14 * scale), text, fill="#333333", font=tick_font)

    y_ticks = int(round(ymax / 0.05))
    for index in range(y_ticks + 1):
        value = index * 0.05
        yy = py(value)
        draw.line((plot_left, yy, plot_right, yy), fill="#E5E7EB", width=scale)
        text = f"{value:.2f}"
        box = draw.textbbox((0, 0), text, font=tick_font)
        draw.text((plot_left - 16 * scale - (box[2] - box[0]), yy - (box[3] - box[1]) / 2), text, fill="#333333", font=tick_font)

    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#111827", width=2 * scale)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#111827", width=2 * scale)

    for _, points, color in series:
        coordinates = [(px(x), py(y)) for x, y in points]
        if len(coordinates) >= 2:
            draw.line(coordinates, fill=color, width=3 * scale, joint="curve")

    title = "Hapke Model: Preliminary Python-Port Test"
    title_box = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((width - title_box[2]) / 2, 28 * scale), title, fill="#111827", font=title_font)
    subtitle = "Provisional density and particle-size values; not a scientifically validated result"
    subtitle_box = draw.textbbox((0, 0), subtitle, font=subtitle_font)
    draw.text(((width - subtitle_box[2]) / 2, 75 * scale), subtitle, fill="#9A3412", font=subtitle_font)

    xlabel = "Wavelength (microns)"
    xlabel_box = draw.textbbox((0, 0), xlabel, font=label_font)
    draw.text(((plot_left + plot_right - xlabel_box[2]) / 2, height - 66 * scale), xlabel, fill="#111827", font=label_font)

    ylabel = "Bidirectional reflectance"
    temp = Image.new("RGBA", (550 * scale, 70 * scale), (255, 255, 255, 0))
    temp_draw = ImageDraw.Draw(temp)
    temp_draw.text((0, 0), ylabel, fill="#111827", font=label_font)
    rotated = temp.rotate(90, expand=True)
    image.paste(rotated, (22 * scale, int((height - rotated.height) / 2)), rotated)
    draw = ImageDraw.Draw(image)

    legend_x, legend_y = plot_right - 450 * scale, plot_top + 25 * scale
    draw.rounded_rectangle(
        (legend_x, legend_y, plot_right - 20 * scale, legend_y + 92 * scale),
        radius=10 * scale,
        fill="white",
        outline="#CBD5E1",
        width=scale,
    )
    for row, (name, _, color) in enumerate(series):
        yy = legend_y + (27 + row * 37) * scale
        draw.line((legend_x + 20 * scale, yy, legend_x + 72 * scale, yy), fill=color, width=4 * scale)
        draw.text((legend_x + 88 * scale, yy - 12 * scale), name, fill="#111827", font=legend_font)

    image.resize((width // scale, height // scale), Image.Resampling.LANCZOS).save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
