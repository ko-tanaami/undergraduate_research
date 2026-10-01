"""Compare fixed Shkuratov predictions with independent USGS mineral samples.

Run from the shkuratov_python directory:
    python validation/experimental_comparison/compare_usgs_spectra.py

The comparison is diagnostic. The optical constants, sample texture and
reflectance measurement geometry do not form a controlled validation set.
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
sys.path.insert(0, str(ROOT))
from validation.local_data import opcon_directory
from shkuratov_model import (  # noqa: E402
    Component, load_constants, load_opcon, load_optool, load_refractiveindex_yml, reflectance,
)

SAMPLES = {
    "calcite": ("calcite_gds304.1485.asc", "roush2021_calcite.csv", "pds"),
    "dolomite": ("dolomite_hs102.2464.asc", "roush2021_dolomite.csv", "pds"),
    "magnetite": ("magnetite_hs195.4243.asc", "magnetite_Querry1985.yml", "yml"),
    "serpentine_proxy": ("serpentine_hs8.6201.asc", "Sz_nk", "opcon"),
}


def comparison_constants(filename, kind):
    """Prefer the local Clark table for the current serpentine comparison."""
    if kind == 'opcon':
        return load_opcon(opcon_directory(), 'Sz')
    loader = {'pds': lambda p: load_constants(p, columns=(0, 2, 3)),
              'yml': load_refractiveindex_yml, 'optool': load_optool}[kind]
    return loader(ROOT / 'data/optical_constants_full' / filename)


def comparison_mask(wavelength, constants):
    valid = (wavelength >= .5) & (wavelength <= 2.5)
    valid &= (wavelength >= constants.wavelength[0]) & (wavelength <= constants.wavelength[-1])
    for left, right in constants.gaps:
        valid &= ~((wavelength > left) & (wavelength < right))
    return valid


def read_usgs(path: Path):
    """Read the three 15-character fixed-width fields, omitting deleted rows."""
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[16:]:
        if len(line) < 45:
            continue
        try:
            rows.append(tuple(float(line[i:i + 15]) for i in (0, 15, 30)))
        except ValueError:
            continue
    data = np.asarray(rows)
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError(f"No spectral data in {path}")
    if np.any(np.diff(data[:, 0]) <= 0):
        raise ValueError(f"Non-increasing wavelength in {path}")
    return data


def band(wavelength, values, left, right):
    """Linear continuum through endpoints; minimum in the interval."""
    mask = (wavelength >= left) & (wavelength <= right)
    if mask.sum() < 5:
        raise ValueError("Too few points in absorption band")
    w, y = wavelength[mask], values[mask]
    baseline = np.interp(w, [left, right], np.interp([left, right], wavelength, values))
    normalized = y / baseline
    idx = int(np.argmin(normalized))
    return {"center_um": float(w[idx]), "depth": float(1 - normalized[idx])}


def render_plot(curves, target):
    """Draw a self-contained diagnostic PNG without optional plotting packages."""
    width, height = 1400, 920
    im = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 20)
        small = ImageFont.truetype('DejaVuSans.ttf', 16)
    except OSError:
        font = ImageFont.load_default(size=20)
        small = ImageFont.load_default(size=16)
    dark = (35, 45, 55)
    for i, (name, (wave, obs, mod)) in enumerate(curves.items()):
        col, row = i % 2, i // 2
        x0, y0 = 75 + col * 690, 75 + row * 430
        x1, y1 = x0 + 570, y0 + 315
        y_min = max(0, min(float(obs.min()), float(mod.min())) - .06)
        y_max = min(1, max(float(obs.max()), float(mod.max())) + .06)
        xp = lambda w: x0 + (w - .5) / 2 * (x1 - x0)
        yp = lambda r: y1 - (r - y_min) / (y_max - y_min) * (y1 - y0)
        for tick in (.5, 1., 1.5, 2., 2.5):
            x = int(xp(tick))
            draw.line((x, y0, x, y1), fill=(225, 230, 235))
            draw.text((x - 13, y1 + 8), str(tick), font=small, fill=dark)
        for tick in np.linspace(y_min, y_max, 5):
            y = int(yp(tick))
            draw.line((x0, y, x1, y), fill=(225, 230, 235))
            draw.text((x0 - 58, y - 9), f"{tick:.2f}", font=small, fill=dark)
        draw.rectangle((x0, y0, x1, y1), outline=dark, width=2)
        for values, color in ((obs, (20, 80, 150)), (mod, (190, 55, 35))):
            points = [(int(xp(w)), int(yp(v))) for w, v in zip(wave, values)]
            draw.line(points, fill=color, width=3, joint="curve")
        label = 'Serpentine (Clark HS318 n,k)' if name == 'serpentine_proxy' else name.title()
        draw.text((x0, y0 - 32), label, font=font, fill=dark)
        draw.text((x0 - 57, y0 - 31), "R", font=small, fill=dark)
        draw.text((x0 + 210, y1 + 35), "Wavelength (um)", font=small, fill=dark)
    draw.line((75, 900, 120, 900), fill=(20, 80, 150), width=4)
    draw.text((130, 890), "USGS measured", font=small, fill=dark)
    draw.line((330, 900, 375, 900), fill=(190, 55, 35), width=4)
    draw.text((385, 890), "Model: S=30 um, q=0.3", font=small, fill=dark)
    im.save(target)


def main():
    records, curves = [], {}
    for name, (spectrum_file, constants_file, kind) in SAMPLES.items():
        observed = read_usgs(HERE / spectrum_file)
        constants = comparison_constants(constants_file, kind)
        wavelength, measured, sigma = observed.T
        valid = comparison_mask(wavelength, constants)
        wavelength, measured, sigma = wavelength[valid], measured[valid], sigma[valid]
        modeled = reflectance(wavelength, [Component(constants, 1., 30.)], porosity=.3, order=64)
        residual = modeled - measured
        record = {
            "material": name,
            "sample": spectrum_file,
            "optical_constants": constants_file,
            "points": len(wavelength),
            "wavelength_range_um": [float(wavelength[0]), float(wavelength[-1])],
            "fixed_effective_path_um": 30., "fixed_porosity": .3,
            "mae_reflectance": float(np.mean(np.abs(residual))),
            "rmse_reflectance": float(np.sqrt(np.mean(residual ** 2))),
            "mean_signed_error": float(np.mean(residual)),
            "pearson_shape_correlation": float(np.corrcoef(measured, modeled)[0, 1]),
            "uncertainty_note": "Zero in the USGS standard-deviation column means unmeasured, not zero uncertainty" if not np.any(sigma) else "Reported USGS per-channel standard deviations are present but omit model/sample systematic uncertainty",
        }
        if name in ("calcite", "dolomite"):
            record["carbonate_band_2p3um"] = {
                "definition": "linear continuum through 2.20 and 2.48 um, minimum of R/continuum",
                "measured": band(wavelength, measured, 2.20, 2.48),
                "modeled": band(wavelength, modeled, 2.20, 2.48),
            }
            # Preselected context values from sample descriptions, not optimized fits.
            sensitivity = []
            for path_um in ((75., 150.) if name == "calcite" else (150., 285.)):
                alternate = reflectance(wavelength, [Component(constants, 1., path_um)],
                                        porosity=.3, order=64)
                sensitivity.append({
                    "effective_path_um": path_um,
                    "rmse_reflectance": float(np.sqrt(np.mean((alternate - measured) ** 2))),
                    "band": band(wavelength, alternate, 2.20, 2.48),
                })
            record["path_length_sensitivity"] = sensitivity
        records.append(record)
        curves[name] = (wavelength, measured, modeled)
        with (HERE / f"{name}_comparison.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["wavelength_um", "usgs_reflectance", "model_reflectance", "usgs_reported_stddev"])
            writer.writerows(zip(wavelength, measured, modeled, sigma))
    (HERE / "comparison_metrics.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    render_plot(curves, HERE / "comparison_curves.png")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
