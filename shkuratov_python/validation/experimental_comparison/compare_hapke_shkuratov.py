"""Compare the current Hapke and Shkuratov forward implementations.

Run from shkuratov_python:
    python validation/experimental_comparison/compare_hapke_shkuratov.py

The outputs have different photometric definitions. This is a forward-model
comparison, not a controlled accuracy test against laboratory reflectance.
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
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(ROOT))

from hapke_model import bd_ref, single_scattering_albedo  # noqa: E402
from shkuratov_model import (  # noqa: E402
    Component, reflectance,
)
from validation.experimental_comparison.compare_usgs_spectra import (  # noqa: E402
    SAMPLES, band, read_usgs, comparison_constants, comparison_mask,
)


def font(size):
    path = Path("C:/Windows/Fonts/arial.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def draw_figure(curves):
    image = Image.new("RGB", (1400, 930), "white")
    draw = ImageDraw.Draw(image)
    colors = {"USGS": (25, 70, 140), "Hapke": (200, 75, 25),
              "Shkuratov": (30, 125, 75)}
    for i, (name, (w, series)) in enumerate(curves.items()):
        column, row = i % 2, i // 2
        x0, y0 = 90 + 690 * column, 70 + 430 * row
        x1, y1 = x0 + 570, y0 + 315
        extrema = np.concatenate(list(series.values()))
        lo, hi = max(0., float(extrema.min()) - .08), float(extrema.max()) + .08
        xp = lambda x: x0 + (x - .5) / 2 * (x1 - x0)
        yp = lambda y: y1 - (y - lo) / (hi - lo) * (y1 - y0)
        for tick in (.5, 1., 1.5, 2., 2.5):
            x = int(xp(tick))
            draw.line((x, y0, x, y1), fill=(227, 230, 234))
            draw.text((x - 13, y1 + 6), str(tick), font=font(16), fill=(40, 40, 40))
        for tick in np.linspace(lo, hi, 5):
            y = int(yp(tick))
            draw.line((x0, y, x1, y), fill=(227, 230, 234))
            draw.text((x0 - 62, y - 8), f"{tick:.2f}", font=font(15), fill=(40, 40, 40))
        draw.rectangle((x0, y0, x1, y1), outline=(40, 40, 40), width=2)
        label = 'Serpentine (Clark HS318 n,k)' if name == 'serpentine_proxy' else name.title()
        draw.text((x0, y0 - 32), label, font=font(21), fill=(35, 35, 35))
        draw.text((x0 + 218, y1 + 32), "Wavelength (um)", font=font(16), fill=(35, 35, 35))
        for label, y in series.items():
            points = [(int(xp(x)), int(yp(value))) for x, value in zip(w, y)]
            draw.line(points, fill=colors[label], width=3)
    draw.text((90, 4), "Spectral shape: each curve / value at 1 um", font=font(24), fill=(35, 35, 35))
    for j, label in enumerate(("USGS", "Hapke", "Shkuratov")):
        x = 90 + j * 240
        draw.line((x, 905, x + 45, 905), fill=colors[label], width=4)
        draw.text((x + 55, 894), label, font=font(18), fill=(35, 35, 35))
    image.save(OUT / "normalized_shape.png")


def main():
    OUT.mkdir(exist_ok=True)
    records, curves = [], {}
    for name, (spectrum_file, constants_file, kind) in SAMPLES.items():
        constants = comparison_constants(constants_file, kind)
        data = read_usgs(HERE / spectrum_file)
        data = data[comparison_mask(data[:, 0], constants)]
        w, observed = data[:, 0], data[:, 1]
        n, k = constants.sample(w)
        # Single-component Hapke: isotropic phase function, no opposition or
        # macroscopic roughness, i=e=0 deg. Pi * bidirectional reflectance is I/F.
        ssa = single_scattering_albedo(w, n, k, 30., quadrature_order=64)
        if not np.all(np.isfinite(ssa)) or np.any((ssa < 0) | (ssa > 1)):
            raise ValueError(f"Hapke single-scattering albedo outside [0, 1]: {name}")
        hapke_if = np.pi * bd_ref(0., 0., 0., 0., 0., .05, ssa, 0.)
        shkuratov = reflectance(w, [Component(constants, 1., 30.)],
                                porosity=.3, order=64)
        if not np.all(np.isfinite(hapke_if)):
            raise ValueError(f"Non-finite Hapke radiance factor: {name}")
        normalized = {"USGS": observed / np.interp(1., w, observed),
                      "Hapke": hapke_if / np.interp(1., w, hapke_if),
                      "Shkuratov": shkuratov / np.interp(1., w, shkuratov)}
        curves[name] = (w, normalized)
        result = {
            "material": name, "n_k_input": constants_file,
            "wavelength_range_um": [float(w[0]), float(w[-1])],
            "points": len(w), "effective_path_or_diameter_um": 30.,
            "hapke": {"incidence_deg": 0., "emergence_deg": 0.,
                      "phase_function_b": 0., "phase_function_c": 0.,
                      "opposition_amplitude_s0": 0., "mean_slope_deg": 0.,
                      "output": "radiance factor I/F = pi times bidirectional reflectance"},
            "shkuratov": {"porosity": .3, "output": "model_reflectance; photometric equivalence with Hapke I/F not established"},
            "pearson_hapke_shkuratov": float(np.corrcoef(hapke_if, shkuratov)[0, 1]),
            "pearson_usgs_hapke": float(np.corrcoef(observed, hapke_if)[0, 1]),
            "pearson_usgs_shkuratov": float(np.corrcoef(observed, shkuratov)[0, 1]),
            "at_1um": {"usgs": float(np.interp(1., w, observed)),
                       "hapke_I_over_F": float(np.interp(1., w, hapke_if)),
                       "shkuratov_model_reflectance": float(np.interp(1., w, shkuratov))},
        }
        if name in ("calcite", "dolomite"):
            result["carbonate_band_2p3um"] = {
                "definition": "linear continuum through 2.20 and 2.48 um",
                "usgs": band(w, observed, 2.20, 2.48),
                "hapke": band(w, hapke_if, 2.20, 2.48),
                "shkuratov": band(w, shkuratov, 2.20, 2.48),
            }
        records.append(result)
        with (OUT / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["wavelength_um", "USGS_measured_reflectance",
                             "Hapke_radiance_factor_I_over_F_i0_e0",
                             "Shkuratov_model_reflectance", "Hapke_single_scattering_albedo",
                             "USGS_normalized_at_1um", "Hapke_normalized_at_1um",
                             "Shkuratov_normalized_at_1um"])
            writer.writerows(zip(w, observed, hapke_if, shkuratov, ssa,
                                 normalized["USGS"], normalized["Hapke"],
                                 normalized["Shkuratov"]))
    (OUT / "metrics.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    draw_figure(curves)
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
