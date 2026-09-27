"""Sensitivity of current Hapke/Shkuratov spectra to non-mineral parameters.

Run from shkuratov_python:
    python validation/experimental_comparison/compare_parameter_sensitivity.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROJECT = ROOT.parent
OUT = HERE / "hapke_shkuratov_comparison"
sys.path[:0] = [str(ROOT), str(PROJECT)]

from hapke_model import bd_ref, single_scattering_albedo  # noqa: E402
from shkuratov_model import (  # noqa: E402
    Component, reflectance,
)
from validation.experimental_comparison.compare_usgs_spectra import (  # noqa: E402
    SAMPLES, band, read_usgs, comparison_constants, comparison_mask,
)

PATHS = (30., 100., 300.)
POROSITIES = (0., .3, .7)
INCIDENCES = (0., 30., 60.)


def diagnostics(w, observed, model, material):
    observed_norm = observed / np.interp(1., w, observed)
    model_norm = model / np.interp(1., w, model)
    result = {
        "normalized_shape_rmse": float(np.sqrt(np.mean((model_norm - observed_norm) ** 2))),
        "shape_pearson": float(np.corrcoef(model, observed)[0, 1]),
        "value_at_1um": float(np.interp(1., w, model)),
    }
    if material in ("calcite", "dolomite"):
        feature = band(w, model, 2.20, 2.48)
        result["carbonate_band_center_um"] = feature["center_um"]
        result["carbonate_band_depth"] = feature["depth"]
    else:
        result["carbonate_band_center_um"] = ""
        result["carbonate_band_depth"] = ""
    return result


def render_band_depth(rows, observed_bands):
    image = Image.new("RGB", (1050, 570), "white")
    draw = ImageDraw.Draw(image)
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    font = lambda size: ImageFont.truetype(str(font_path), size) if font_path.exists() else ImageFont.load_default()
    styles = {"Hapke": (200, 70, 30), "Shkuratov": (25, 125, 75)}
    for j, material in enumerate(("calcite", "dolomite")):
        x0, x1 = 75 + j * 510, 475 + j * 510
        y0, y1 = 70, 460
        draw.rectangle((x0, y0, x1, y1), outline=(40, 40, 40), width=2)
        draw.text((x0, 33), material.title(), fill=(35, 35, 35), font=font(22))
        draw.text((x0 + 100, y1 + 38), "S or D (um)", fill=(35, 35, 35), font=font(17))
        yp = lambda depth: y1 - depth / .32 * (y1 - y0)
        xp = lambda size: x0 + (np.log(size) - np.log(30)) / (np.log(300) - np.log(30)) * (x1 - x0)
        for depth in (0., .1, .2, .3):
            y = int(yp(depth))
            draw.line((x0, y, x1, y), fill=(227, 230, 234))
            draw.text((x0 - 52, y - 8), f"{depth:.1f}", fill=(35, 35, 35), font=font(16))
        obs = observed_bands[material]["depth"]
        y = int(yp(obs))
        draw.line((x0, y, x1, y), fill=(30, 75, 145), width=3)
        for model, selected in (("Hapke", {"incidence_deg": 0.}),
                                ("Shkuratov", {"porosity": .3})):
            values = [r for r in rows if r["material"] == material and
                      r["model"] == model and all(r[k] == v for k, v in selected.items())]
            points = [(int(xp(r["path_or_diameter_um"])), int(yp(r["carbonate_band_depth"]))) for r in values]
            draw.line(points, fill=styles[model], width=3)
            for x, y in points:
                draw.ellipse((x-4, y-4, x+4, y+4), fill=styles[model])
        for size in PATHS:
            x = int(xp(size))
            draw.text((x - 14, y1 + 8), str(int(size)), fill=(35, 35, 35), font=font(16))
    for j, (name, color) in enumerate((("USGS", (30, 75, 145)),
                                       ("Hapke i=0", styles["Hapke"]),
                                       ("Shkuratov q=0.3", styles["Shkuratov"]))):
        x = 75 + j * 275
        draw.line((x, 540, x + 45, 540), fill=color, width=4)
        draw.text((x + 55, 527), name, fill=(35, 35, 35), font=font(17))
    image.save(OUT / "carbonate_band_depth_sensitivity.png")


def main():
    OUT.mkdir(exist_ok=True)
    rows, observed_bands = [], {}
    for material, (spectrum_file, constants_file, kind) in SAMPLES.items():
        constants = comparison_constants(constants_file, kind)
        observed_data = read_usgs(HERE / spectrum_file)
        observed_data = observed_data[comparison_mask(observed_data[:, 0], constants)]
        w, observed = observed_data[:, 0], observed_data[:, 1]
        n, k = constants.sample(w)
        if material in ("calcite", "dolomite"):
            observed_bands[material] = band(w, observed, 2.20, 2.48)
        for path_um in PATHS:
            ssa = single_scattering_albedo(w, n, k, path_um, quadrature_order=64)
            if not np.all(np.isfinite(ssa)) or np.any((ssa < 0) | (ssa > 1)):
                raise ValueError(f"Invalid Hapke SSA for {material}, D={path_um}")
            for incidence in INCIDENCES:
                # Emergence=0 makes phase angle equal incidence for this
                # isotropic, no-opposition, zero-slope diagnostic scenario.
                angle = np.deg2rad(incidence)
                curve = np.pi * bd_ref(angle, 0., angle, 0., 0., .05, ssa, 0.)
                rows.append({"material": material, "model": "Hapke",
                             "path_or_diameter_um": path_um,
                             "incidence_deg": incidence, "porosity": "",
                             **diagnostics(w, observed, curve, material)})
            for porosity in POROSITIES:
                curve = reflectance(w, [Component(constants, 1., path_um)],
                                    porosity=porosity, order=64)
                rows.append({"material": material, "model": "Shkuratov",
                             "path_or_diameter_um": path_um,
                             "incidence_deg": "", "porosity": porosity,
                             **diagnostics(w, observed, curve, material)})
    columns = list(rows[0])
    with (OUT / "parameter_sensitivity.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "observed_carbonate_bands.json").write_text(
        json.dumps(observed_bands, indent=2), encoding="utf-8")
    render_band_depth(rows, observed_bands)
    print(f"Wrote {len(rows)} configurations")


if __name__ == "__main__":
    main()
