"""Python port of Driss Takir's ``hapke1.2.pro`` (IDL, version 1.2).

The Hapke/Roush equations and function boundaries are kept close to the IDL
source.  Four defects that prevent a meaningful multi-component calculation
are corrected:

1. ``E1(e)`` is evaluated as ``E1(e, theta)``.
2. The phase angle remains in radians when passed to trigonometric functions.
3. Component metadata is retained instead of being reallocated per component.
4. Mixture numerator and denominator terms are summed instead of overwritten.

Optical-constant files must contain three whitespace-separated columns:
wavelength in microns, real refractive index n, and imaginary index k.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


PI = np.pi
SCATTERING_COEFFICIENT = 1.0e-17


@dataclass(frozen=True)
class Component:
    path: Path
    mass_fraction: float
    density: float
    diameter_microns: float


def E1(x: np.ndarray | float, theta: float) -> np.ndarray | float:
    """Roughness sub-function; Hapke (1993), eq. 12.45b."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return np.exp((-2.0 / PI) / np.tan(theta) / np.tan(x))


def E2(x: np.ndarray | float, theta: float) -> np.ndarray | float:
    """Roughness sub-function; literal port of IDL's eq. 12.45c line."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return np.exp((-1.0 / PI) / np.tan(theta) ** 2 / np.tan(x)) ** 2


def fs(pse: float) -> float:
    """Roughness sub-function f(pse); Hapke (1993), eq. 12.51."""
    return float(np.exp(-2.0 * np.tan(pse / 2.0)))


def chi(theta: float) -> float:
    """Roughness sub-function chi(theta); Hapke (1993), eq. 12.45a."""
    return float(1.0 / np.sqrt(1.0 + PI * np.tan(theta) ** 2))


def Sr(i: float, e: float, pse: float, theta: float) -> float:
    """Macroscopic roughness correction from the original IDL routine."""
    if np.isclose(theta, 0.0):
        return 1.0

    ch = chi(theta)
    sin_half_sq = np.sin(pse / 2.0) ** 2
    mu0e0 = ch * (
        np.cos(i) + np.sin(i) * np.tan(theta) * E2(i, theta) / (2.0 - E1(i, theta))
    )
    mue0 = ch * (
        np.cos(e) + np.sin(e) * np.tan(theta) * E2(e, theta) / (2.0 - E1(e, theta))
    )

    if i <= e:
        mu0e = ch * (
            np.cos(i)
            + np.sin(i)
            * np.tan(theta)
            * (np.cos(pse) * E2(e, theta) + sin_half_sq * E2(i, theta))
            / (2.0 - E1(e, theta) - (pse / PI) * E1(i, theta))
        )
        # The IDL source calls E1(e) here; theta is the missing required argument.
        mue = ch * (
            np.cos(e)
            + np.sin(e)
            * np.tan(theta)
            * (E2(e, theta) - sin_half_sq * E2(i, theta))
            / (2.0 - E1(e, theta) - (pse / PI) * E1(i, theta))
        )
    else:
        mu0e = ch * (
            np.cos(i)
            + np.sin(i)
            * np.tan(theta)
            * (E2(i, theta) - sin_half_sq * E2(e, theta))
            / (2.0 - E1(i, theta) - (pse / PI) * E1(e, theta))
        )
        mue = ch * (
            np.cos(e)
            + np.sin(e)
            * np.tan(theta)
            * (np.cos(pse) * E2(e, theta) + sin_half_sq * E2(e, theta))
            / (2.0 - E1(i, theta) - (pse / PI) * E1(e, theta))
        )

    return float(
        (mue / mue0)
        * (mu0e / mu0e0)
        * ch
        / (1.0 - fs(pse) + fs(pse) * ch * (mu0e / mu0e0))
    )


def phase_function(b: float, c: float, g: float) -> float:
    """Phase function P(g); Roush (1984)."""
    return float((1.0 + b * np.cos(g)) + c * (1.5 * np.cos(g) ** 2 - 0.5))


def Bg(
    h: float, s0: float, w: np.ndarray, b: float, c: float, g: float
) -> np.ndarray:
    """Opposition-effect function; Hapke (1993), eqs. 8.86 and 8.90."""
    if h <= 0.0:
        raise ValueError("h must be positive")
    with np.errstate(divide="ignore", invalid="ignore"):
        b0 = s0 / (w * phase_function(b, c, g))
        return b0 / (1.0 + np.tan(g / 2.0) / h)


def Hf(x: float, w: np.ndarray) -> np.ndarray:
    """Chandrasekhar H-function approximation; Hapke (1993), eq. 8.55."""
    return (1.0 + 2.0 * x) / (1.0 + 2.0 * np.sqrt(1.0 - w) * x)


def ext_ref(x: np.ndarray, n: np.ndarray, k: np.ndarray) -> np.ndarray:
    """External-reflection integrand from Roush (1994)."""
    sin_half = np.sin(x / 2.0)
    cos_half = np.cos(x / 2.0)
    a = n**2 - k**2 - sin_half**2
    bb = 4.0 * n**2 * k**2
    root = np.sqrt(a**2 + bb)
    u = np.sqrt(0.5 * (a + root))
    v = np.sqrt(0.5 * (-a + root))

    r_perp = ((cos_half - u) ** 2 + v**2) / ((cos_half + u) ** 2 + v**2)
    ee = ((n**2 - k**2) * cos_half - u) ** 2
    ff = (2.0 * n * k * cos_half - v) ** 2
    gg = ((n**2 - k**2) * cos_half + u) ** 2
    hh = (2.0 * n * k * cos_half + v) ** 2
    r_para = (ee + ff) / (gg + hh)
    return (r_perp + r_para) * np.sin(x) * np.cos(x)


def integrated_external_reflectance(
    n: np.ndarray, k: np.ndarray, quadrature_order: int = 128
) -> np.ndarray:
    """IDL QROMB(ext_ref, 0, pi/2) replacement using Gauss-Legendre quadrature."""
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    x = ((nodes + 1.0) * PI / 4.0)[:, None]
    scaled_weights = (weights * PI / 4.0)[:, None]
    values = ext_ref(x, n[None, :], k[None, :])
    return np.sum(scaled_weights * values, axis=0)


def bd_ref(
    ic: float,
    em: float,
    g: float,
    b: float,
    c: float,
    h: float,
    w: np.ndarray,
    s0: float,
) -> np.ndarray:
    """Bidirectional reflectance; Hapke (1981)."""
    u = np.cos(em)
    u0 = np.cos(ic)
    return (w / (4.0 * PI)) * (u0 / (u0 + u)) * (
        (1.0 + Bg(h, s0, w, b, c, g)) * phase_function(b, c, g)
        + Hf(u, w) * Hf(u0, w)
        - 1.0
    )


def load_optical_constants(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(path, dtype=float)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(f"{path}: expected at least three numeric columns")
    data = data[:, :3]
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{path}: non-finite optical constants found")
    order = np.argsort(data[:, 0])
    wavelength, n, k = data[order].T
    # The supplied serpentine table contains one duplicated wavelength.  IDL
    # accepts it, but interpolation requires a unique grid; average only exact
    # duplicates while leaving all other samples unchanged.
    unique_wavelength, inverse, counts = np.unique(
        wavelength, return_inverse=True, return_counts=True
    )
    if unique_wavelength.size != wavelength.size:
        n_sum = np.zeros_like(unique_wavelength)
        k_sum = np.zeros_like(unique_wavelength)
        np.add.at(n_sum, inverse, n)
        np.add.at(k_sum, inverse, k)
        wavelength = unique_wavelength
        n = n_sum / counts
        k = k_sum / counts
    return wavelength, n, k


def single_scattering_albedo(
    wavelength: np.ndarray,
    n: np.ndarray,
    k: np.ndarray,
    diameter_microns: float,
    quadrature_order: int = 128,
) -> np.ndarray:
    if diameter_microns <= 0.0:
        raise ValueError("particle diameter must be positive")

    se = integrated_external_reflectance(n, k, quadrature_order)
    modulus_squared = n**2 + k**2
    si = integrated_external_reflectance(
        n / modulus_squared, k / modulus_squared, quadrature_order
    )
    alpha = 4.0 * PI * k / wavelength
    root = np.sqrt(alpha * (alpha + SCATTERING_COEFFICIENT))
    r1 = (1.0 - np.sqrt(alpha / (alpha + SCATTERING_COEFFICIENT))) / (
        1.0 + np.sqrt(alpha / (alpha + SCATTERING_COEFFICIENT))
    )
    numerator = (1.0 - se) * (1.0 - si) * (
        r1 + np.exp(-2.0 * diameter_microns * root)
    )
    denominator = 1.0 - r1 * si + (
        r1 - si * np.exp(-(2.0 / 3.0) * diameter_microns * root)
    )
    return se + numerator / denominator


def hapke(
    incidence_deg: float,
    emergence_deg: float,
    azimuth_deg: float,
    b: float,
    c: float,
    h: float,
    s0: float,
    mean_slope_deg: float,
    components: Sequence[Component],
    quadrature_order: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate wavelength, bidirectional reflectance, and mixture albedo."""
    if not components:
        raise ValueError("at least one component is required")
    for component in components:
        if component.mass_fraction < 0.0:
            raise ValueError("mass fractions cannot be negative")
        if component.density <= 0.0:
            raise ValueError("solid densities must be positive")

    loaded = [(component, *load_optical_constants(component.path)) for component in components]
    lower = max(wavelength[0] for _, wavelength, _, _ in loaded)
    upper = min(wavelength[-1] for _, wavelength, _, _ in loaded)
    if lower >= upper:
        raise ValueError("component wavelength ranges do not overlap")

    reference_wavelength = loaded[0][1]
    wavelength = reference_wavelength[
        (reference_wavelength >= lower) & (reference_wavelength <= upper)
    ]
    if wavelength.size < 2:
        raise ValueError("too few points in the shared wavelength range")

    weighted_qs = np.zeros_like(wavelength)
    total_weight = 0.0
    for component, source_wavelength, source_n, source_k in loaded:
        n = np.interp(wavelength, source_wavelength, source_n)
        k = np.interp(wavelength, source_wavelength, source_k)
        qs = single_scattering_albedo(
            wavelength, n, k, component.diameter_microns, quadrature_order
        )
        weight = component.mass_fraction / (
            component.density * component.diameter_microns
        )
        weighted_qs += weight * qs
        total_weight += weight

    if total_weight <= 0.0:
        raise ValueError("the mixture has zero total optical weight")
    w = weighted_qs / total_weight

    ic = np.deg2rad(incidence_deg)
    em = np.deg2rad(emergence_deg)
    pse = np.deg2rad(azimuth_deg)
    theta = np.deg2rad(mean_slope_deg)
    cosine_g = np.cos(ic) * np.cos(em) + np.sin(ic) * np.sin(em) * np.cos(pse)
    g = float(np.arccos(np.clip(cosine_g, -1.0, 1.0)))
    reflectance = bd_ref(ic, em, g, b, c, h, w, s0) * Sr(ic, em, pse, theta)
    return wavelength, reflectance, w


def parse_component(value: str) -> Component:
    try:
        path_text, mass, density, diameter = value.rsplit(",", 3)
        return Component(Path(path_text), float(mass), float(density), float(diameter))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "component must be FILE,MASS_FRACTION,DENSITY,DIAMETER_MICRONS"
        ) from error


def write_csv(
    path: Path, wavelength: np.ndarray, reflectance: np.ndarray, albedo: np.ndarray
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("wavelength_microns", "reflectance", "single_scattering_albedo"))
        writer.writerows(zip(wavelength, reflectance, albedo))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incidence", type=float, required=True, help="incidence angle (degrees)")
    parser.add_argument("--emergence", type=float, required=True, help="emergence angle (degrees)")
    parser.add_argument("--azimuth", type=float, required=True, help="relative azimuth (degrees)")
    parser.add_argument("--b", type=float, required=True, help="phase-function coefficient b")
    parser.add_argument("--c", type=float, required=True, help="phase-function coefficient c")
    parser.add_argument("--h", type=float, required=True, help="opposition-effect width h")
    parser.add_argument("--s0", type=float, required=True, help="opposition-effect amplitude s0")
    parser.add_argument("--mean-slope", type=float, required=True, help="mean slope angle (degrees)")
    parser.add_argument(
        "--component",
        action="append",
        type=parse_component,
        required=True,
        metavar="FILE,MASS,DENSITY,DIAMETER",
        help="repeat once for each mixture component",
    )
    parser.add_argument("--quadrature-order", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("hapke_reflectance.csv"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    wavelength, reflectance, albedo = hapke(
        args.incidence,
        args.emergence,
        args.azimuth,
        args.b,
        args.c,
        args.h,
        args.s0,
        args.mean_slope,
        args.component,
        args.quadrature_order,
    )
    write_csv(args.output, wavelength, reflectance, albedo)
    print(f"Wrote {wavelength.size} rows to {args.output}")


if __name__ == "__main__":
    main()
