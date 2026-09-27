# Hapke Pythonコード：日本語注釈付き説明資料
# 対象：プロジェクト直下の hapke_model.py ／ 作成日：2026-09-26
#
# 【このファイルの使い方】
# 元の全コードを残し、# で始まる解説だけを追加した読解用コピー。
# 実行可能だが、研究計算の基準は元ファイルとする。修正版の提案ではない。
# 各解説の「原コード L...」は作成時の元ファイルの行番号。
# 推奨読順：Component → hapke → single_scattering_albedo → bd_ref → Sr。
# 物理の流れを先に読み、最後にload_optical_constantsとmainで入出力を確認する。
#
# 【何を計算するか】
# 入力：波長λごとの屈折率n・消衰係数k、成分の質量比・密度・粒径、測定幾何。
# λ,n,k → 各成分の単一散乱アルベドqs → 混合アルベドw
#       → 位相関数・衝効果・多重散乱 → 巨視的粗さ補正 → 双方向反射率r(λ)。
# ここでwは粒子による散乱と消散の比を表す量で、表面反射率rとは異なる。
# この実装はhapke1.2.proからの移植であり、Hapke理論の全バージョンを実装しない。
# 元の式に対する説明と、その式の物理的妥当性の証明は区別する。
#
# 【変数と単位】
# λ=wavelength：µm。n：屈折率の実部、k：虚部の大きさ。両方無次元。
# D=diameter_microns：µm。density：全成分で同じ単位の固体密度を与える。
# mass_fraction：相対質量。合計1の検査はなく、比だけが結果に効く。
# weight=m/(ρD)：散乱への重み。球形・同じ幾何係数なら質量から断面積比への換算。
# ic=i：入射角、em=e：出射角、pse=ψ：相対方位角、g：位相角。
# CLIは度、関数内部の三角関数はラジアン。ψとgは別物。
# theta：巨視的平均傾斜角。粒径でも入射角でもない。
# b,c：P(g)の係数。h：衝効果の幅を決める無次元係数。s0：衝効果の強度の入力。
# このb,cは二項Henyey–Greenstein関数のパラメータとは同一視しない。
# quadrature_order：数値積分の点数。既定128。物理パラメータではない。
#
# 【主要式：このコードが実際に評価する式】
# α=4πk/λ、s=SCATTERING_COEFFICIENT=1e-17、β=√[α(α+s)]。
# 内部散乱係数sの単位は、λに合わせるならµm⁻¹。非常に小さい固定値。
# r1=(1-√[α/(α+s)])/(1+√[α/(α+s)])。
# qs=Se + (1-Se)(1-Si)(r1+exp(-2Dβ))
#           / {1-r1 Si + r1-Si exp[-(2/3)Dβ]}。
# w=Σ[mj/(ρj Dj) qsj] / Σ[mj/(ρj Dj)]。
# cos(g)=cos(i)cos(e)+sin(i)sin(e)cos(ψ)。
# P(g)=1+b cos(g)+c[1.5 cos²(g)-0.5]。
# H(x,w)=(1+2x)/(1+2x√(1-w))。
# r=w/(4π) × μ0/(μ0+μ) × {(1+B)P+H(μ)H(μ0)-1} × Sr。
# μ0=cos(i)、μ=cos(e)。Srは別関数で計算する粗さ補正。
# CSVのreflectanceはこのrで、πrや正規化スペクトルではない。
# πrはこの測光規約でI/Fに対応する。観測値との比較は規約を確認して行う。
#
# 【同梱データで実行する例：作業ディレクトリはプロジェクト直下】
# 次のコマンドは1行にして実行する。値は操作説明用で、フィット結果ではない。
# python shkuratov_python/説明資料/コード解説/hapke_注釈付き.py --incidence 0 --emergence 0 --azimuth 0
#   --b 0 --c 0 --h 0.1 --s0 0 --mean-slope 0
#   --component shkuratov_python/data/serpentine_optical_constants/Gser_nk_original.txt,1,2.5,30
#   --output shkuratov_python/説明資料/コード解説/hapke_実行例.csv
# 出力：wavelength_microns,reflectance,single_scattering_albedoの3列。
# componentのパスは実行時の作業ディレクトリ基準。密度2.5は説明用の仮定。
# 2成分なら--componentを2回指定する。反射率の平均ではなくqsの重み付き平均を取る。
#
# 【読むときに区別すべき点】
# ・load_optical_constantsは波長重複を平均するが、n>0,k≥0,λ>0の検査はない。
# ・w∈[0,1]、角度範囲、P(g)>0等の検査はなく、異常入力でNaN等が出る可能性がある。
# ・np.errstateは警告を抑えるだけで、数学的な特異点を修復しない。
# ・E2はexp[-1/(π tan²θ tan x)]を二乗している。expの引数内のtan²xではない。
#   原IDLの行を維持した実装であり、標準粗さ式と同一だと断定しない。
# ・Srのi>e分岐のmueにもE2(e)が2回現れる。対称性を仮定して読み替えない。
# ・qsの指数と分母も移植式のまま。標準Hapke式に独立照合済みとはしない。
# ・θ=0ならSr=1。まずこの条件で幾何と散乱計算を理解すると追いやすい。
# ・このファイルにはフィット処理、空隙率パラメータ、装置分解能の畳み込みはない。
#
# 【理解確認】
# 単成分ではw=qs。全mass_fractionを同じ倍率にしてもwは変わらない。
# 2成分の質量比が同じでもρやDが違えば散乱への寄与は同じにならない。
# k,Dが指数に入り吸収に影響する一方、nは界面反射にも影響する。
# 粒径変更の効果を常に単調と断定せず、式と対象範囲で確認する。
# 資料はコードの読解用であり、この実装全体の独立した物理検証ではない。
# 【注釈版の動作確認】
# 元コードとの構文木一致を確認。2成分混合、粗さなし・i>e・i≤eの3条件で
# 出力配列が元コードと完全一致。同梱蛇紋石表のCLI例も986点の有限値を出力。
# これらは注釈による計算変更がないことの確認であり、物理モデルの検証ではない。
# 元ファイル SHA256: 424bba1d76764ba78fae14cc89a7356c9c231569c5aa50df054c30d7024b6831
#

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


# --------------------------------------------------------------
# 【Component】原コード L32–L36
# 入力成分をまとめるデータクラス。frozen=Trueなので生成後に属性を書き換えない。
# pathは光学定数表、mass_fractionは質量比、densityは固体密度、diameter_micronsは粒径。
@dataclass(frozen=True)
class Component:
    path: Path
    mass_fraction: float
    density: float
    diameter_microns: float


# --------------------------------------------------------------
# 【E1】原コード L39–L42
# 粗さ補正の補助関数。x,thetaはラジアン。tanが分母に入るため0付近の極限に注意。
def E1(x: np.ndarray | float, theta: float) -> np.ndarray | float:
    """Roughness sub-function; Hapke (1993), eq. 12.45b."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return np.exp((-2.0 / PI) / np.tan(theta) / np.tan(x))


# --------------------------------------------------------------
# 【E2】原コード L45–L48
# 移植元の書き方を保持した粗さ補助関数。末尾の**2はexpの計算結果の二乗。
# 数式の一般形に置き換えず、この関数が評価する式として読む。
def E2(x: np.ndarray | float, theta: float) -> np.ndarray | float:
    """Roughness sub-function; literal port of IDL's eq. 12.45c line."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return np.exp((-1.0 / PI) / np.tan(theta) ** 2 / np.tan(x)) ** 2


# --------------------------------------------------------------
# 【fs】原コード L51–L53
# 相対方位角ψから粗さ補正の重みf(ψ)を作る。位相角gを渡す関数ではない。
def fs(pse: float) -> float:
    """Roughness sub-function f(pse); Hapke (1993), eq. 12.51."""
    return float(np.exp(-2.0 * np.tan(pse / 2.0)))


# --------------------------------------------------------------
# 【chi】原コード L56–L58
# 平均傾斜角thetaから粗さの補助係数を作る。theta=0なら1。
def chi(theta: float) -> float:
    """Roughness sub-function chi(theta); Hapke (1993), eq. 12.45a."""
    return float(1.0 / np.sqrt(1.0 + PI * np.tan(theta) ** 2))


# --------------------------------------------------------------
# 【Sr】原コード L61–L112
# 入射角・出射角・相対方位角・平均傾斜角から、波長に依存しない粗さ係数を返す。
# mu0e0,mue0は基準の有効方向余弦、mu0e,mueは方位を含む有効方向余弦。
# i≤eとi>eで式を切り替える。最終的にbd_refの出力へ乗算する。
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


# --------------------------------------------------------------
# 【phase_function】原コード L115–L117
# 単粒子の方向依存を表すP(g)。b=c=0ならP=1（等方）。
# 入力によっては負にもなり得るが、この実装は正値を検査しない。
def phase_function(b: float, c: float, g: float) -> float:
    """Phase function P(g); Roush (1984)."""
    return float((1.0 + b * np.cos(g)) + c * (1.5 * np.cos(g) ** 2 - 0.5))


# --------------------------------------------------------------
# 【Bg】原コード L120–L128
# 衝効果B(g)を波長ごとに返す。h>0が必要。
# B0=s0/[w P(g)]であり、s0そのものがB0ではない。s0=0なら通常B=0。
# w=0やP=0ではゼロ除算が残るので、s0=0だけで安全になるとは限らない。
def Bg(
    h: float, s0: float, w: np.ndarray, b: float, c: float, g: float
) -> np.ndarray:
    """Opposition-effect function; Hapke (1993), eqs. 8.86 and 8.90."""
    if h <= 0.0:
        raise ValueError("h must be positive")
    with np.errstate(divide="ignore", invalid="ignore"):
        b0 = s0 / (w * phase_function(b, c, g))
        return b0 / (1.0 + np.tan(g / 2.0) / h)


# --------------------------------------------------------------
# 【Hf】原コード L131–L133
# 多重散乱に使うH関数の近似。xはμまたはμ0、wは波長配列。
# √(1-w)を含み、実数として計算するにはw≤1が必要。
def Hf(x: float, w: np.ndarray) -> np.ndarray:
    """Chandrasekhar H-function approximation; Hapke (1993), eq. 8.55."""
    return (1.0 + 2.0 * x) / (1.0 + 2.0 * np.sqrt(1.0 - w) * x)


# --------------------------------------------------------------
# 【ext_ref】原コード L136–L152
# 外部反射の積分被積分関数。n,kの配列と積分角xを受ける。
# u,vは複素屈折に関係する実量、r_perp/r_paraは2偏光の反射寄与。
# コードはsin(x/2),cos(x/2)を使う。xをそのまま入射角と読み替えない。
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


# --------------------------------------------------------------
# 【integrated_external_reflectance】原コード L155–L163
# 区間[0,π/2]をGauss–Legendre求積で積分し、各波長に1個の値を返す。
# IDLのQROMBの代替。点数の変更は積分近似を変える。
# [:,None]で積分点方向と波長方向を分け、配列全体を一括計算する。
def integrated_external_reflectance(
    n: np.ndarray, k: np.ndarray, quadrature_order: int = 128
) -> np.ndarray:
    """IDL QROMB(ext_ref, 0, pi/2) replacement using Gauss-Legendre quadrature."""
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    x = ((nodes + 1.0) * PI / 4.0)[:, None]
    scaled_weights = (weights * PI / 4.0)[:, None]
    values = ext_ref(x, n[None, :], k[None, :])
    return np.sum(scaled_weights * values, axis=0)


# --------------------------------------------------------------
# 【bd_ref】原コード L166–L183
# 粗さ補正前の双方向反射率。u=μ,u0=μ0。
# (1+B)Pは単一散乱項、H(u)H(u0)-1は多重散乱項。
# 粗さはここでは掛けず、hapke側でSrを掛ける。
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


# --------------------------------------------------------------
# 【load_optical_constants】原コード L186–L209
# 空白区切り数値表の先頭3列をλ,n,kとして読む。CSVヘッダーの自動処理はない。
# 有限値を検査し波長昇順へ並べる。同じλが複数あるとn,kを算術平均する。
# 元のデータを保存し直す処理ではなく、メモリ上で整理する。
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


# --------------------------------------------------------------
# 【single_scattering_albedo】原コード L212–L238
# 1成分のqsを計算する。Seは外側からの界面反射、Siは内側反射の移植近似。
# Siの計算にはn/(n²+k²),k/(n²+k²)を再び積分関数へ渡す。
# αは吸収係数、rootはβ、r1は内部散乱の補助量。D>0だけをここで検査する。
def single_scattering_albedo(
    wavelength: np.ndarray,
    n: np.ndarray,
    k: np.ndarray,
    diameter_microns: float,
    quadrature_order: int = 128,
) -> np.ndarray:
    if diameter_microns <= 0.0:
        raise ValueError("particle diameter must be positive")

    # 読解：Se：外側の反射。以後すべて波長配列として計算。
    se = integrated_external_reflectance(n, k, quadrature_order)
    modulus_squared = n**2 + k**2
    si = integrated_external_reflectance(
        n / modulus_squared, k / modulus_squared, quadrature_order
    )
    # 読解：λがµmなのでαはµm⁻¹。D×rootは無次元の指数になる。
    alpha = 4.0 * PI * k / wavelength
    root = np.sqrt(alpha * (alpha + SCATTERING_COEFFICIENT))
    r1 = (1.0 - np.sqrt(alpha / (alpha + SCATTERING_COEFFICIENT))) / (
        1.0 + np.sqrt(alpha / (alpha + SCATTERING_COEFFICIENT))
    )
    # 読解：粒子内部へ入って、吸収・内部散乱を経て外へ出る項。式は移植版のまま。
    numerator = (1.0 - se) * (1.0 - si) * (
        r1 + np.exp(-2.0 * diameter_microns * root)
    )
    # 読解：この括弧の構造を保って読む。指数2Dと(2/3)Dはコードで異なる。
    denominator = 1.0 - r1 * si + (
        r1 - si * np.exp(-(2.0 / 3.0) * diameter_microns * root)
    )
    return se + numerator / denominator


# --------------------------------------------------------------
# 【hapke】原コード L241–L300
# 全体の前進計算。戻り値は(共通波長,双方向反射率,混合w)。
# 全成分の波長範囲の共通部分を取り、第1成分の波長点を計算格子として使う。
# 成分順を替えると格子が変わり得る。他成分のn,kはそこへ線形補間する。
# qsを質量・密度・粒径からの光学重みで混合し、幾何と粗さを適用する。
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
        # 読解：質量分率そのものを平均の重みにはしない。ρDで割った値を使う。
        weight = component.mass_fraction / (
            component.density * component.diameter_microns
        )
        weighted_qs += weight * qs
        total_weight += weight

    if total_weight <= 0.0:
        raise ValueError("the mixture has zero total optical weight")
    # 読解：規格化するので質量比の総和が1でなくても相対比が同じなら同じw。
    w = weighted_qs / total_weight

    ic = np.deg2rad(incidence_deg)
    em = np.deg2rad(emergence_deg)
    pse = np.deg2rad(azimuth_deg)
    theta = np.deg2rad(mean_slope_deg)
    # 読解：相対方位角ψから位相角gを計算。clipは丸め誤差によるarccosの範囲逸脱を防ぐ。
    cosine_g = np.cos(ic) * np.cos(em) + np.sin(ic) * np.sin(em) * np.cos(pse)
    g = float(np.arccos(np.clip(cosine_g, -1.0, 1.0)))
    # 読解：本体の双方向反射率と粗さ係数を掛け、CSVへ渡す。ここにπ倍はない。
    reflectance = bd_ref(ic, em, g, b, c, h, w, s0) * Sr(ic, em, pse, theta)
    return wavelength, reflectance, w


# --------------------------------------------------------------
# 【parse_component】原コード L303–L310
# CLIのFILE,MASS,DENSITY,DIAMETERをComponentへ変換。
# rsplit(...,3)で右から3区切りを使う。密度と粒径の値の妥当性は計算側が検査する。
def parse_component(value: str) -> Component:
    try:
        path_text, mass, density, diameter = value.rsplit(",", 3)
        return Component(Path(path_text), float(mass), float(density), float(diameter))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "component must be FILE,MASS_FRACTION,DENSITY,DIAMETER_MICRONS"
        ) from error


# --------------------------------------------------------------
# 【write_csv】原コード L313–L319
# 波長、反射率r、混合wを同じ行に保存する。π倍も正規化も行わない。
def write_csv(
    path: Path, wavelength: np.ndarray, reflectance: np.ndarray, albedo: np.ndarray
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("wavelength_microns", "reflectance", "single_scattering_albedo"))
        writer.writerows(zip(wavelength, reflectance, albedo))


# --------------------------------------------------------------
# 【build_parser】原コード L322–L342
# 必須の測定幾何と散乱パラメータを定義する。--componentは繰り返し指定可能。
# 既定の求積点数は128、出力名はhapke_reflectance.csv。
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


# --------------------------------------------------------------
# 【main】原コード L345–L360
# コマンドラインを解析→hapkeを実行→CSV保存。
# importしただけでは実行されず、ファイルを直接起動したときに呼ばれる。
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
