"""
Loaders for published reference data used to validate cochleasim's model
output against independent literature. Kept separate from
cochleasim.analysis: nothing in this module computes anything from a
CochlearParams -- it only reads citable numbers from
cochleasim/data/reference/*.csv and exposes them as plain arrays/dataclasses.
"""

import csv
import os
from dataclasses import dataclass

import numpy as np

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "reference")


@dataclass
class ReferenceCurve:
    """A published reference curve, ready to compare against model output."""
    x: np.ndarray
    y: np.ndarray
    label: str  # short name for legends, e.g. "Greenwood (1990), human"
    source: str  # full citation


@dataclass
class ReferencePoints:
    """
    A small set of published (x, y) reference points -- typically too few
    to interpolate meaningfully, used for direct point-by-point comparison
    instead (e.g. the two gain figures Neely & Kim give in their text).
    """
    x: np.ndarray
    y: np.ndarray
    label: str
    source: str


def greenwood_frequency(x_from_apex, A, a, k):
    """
    Greenwood-form frequency-place function, F = A*(10**(a*x) - k).

    x_from_apex : fractional distance along the cochlea measured FROM
        THE APEX (0 = apex, 1 = base) -- Greenwood's own convention, the
        opposite of solve_pressure's x (measured from the stapes/base).
        Use x_from_apex_fraction() to convert model positions first.
    A, a, k : species-specific coefficients.
    """
    x_from_apex = np.asarray(x_from_apex, dtype=float)
    return A * (10.0 ** (a * x_from_apex) - k)


def x_from_apex_fraction(x_cm, L_cm):
    """Convert model position (x_cm, 0=stapes) to Greenwood's convention (0=apex, 1=base)."""
    x_cm = np.asarray(x_cm, dtype=float)
    return 1.0 - x_cm / L_cm


def load_greenwood_curve(species, n_points=500):
    """
    Build a ReferenceCurve of Greenwood-predicted CF vs. model-convention
    position (x_cm as a fraction of L, base-referenced), for one species.

    species : row key in data/reference/greenwood_coefficients.csv (e.g.
        "human", "cat").

    Returns
    -------
    ReferenceCurve
        x : fractional position from the base (0=stapes, 1=apex).
        y : CF (Hz) predicted by Greenwood's formula at that position.
    """
    path = os.path.join(_DATA_DIR, "greenwood_coefficients.csv")
    with open(path, newline="") as f:
        rows = {r["species"]: r for r in csv.DictReader(f)}

    if species not in rows:
        raise ValueError(f"No Greenwood coefficients for species='{species}' in {path}")

    r = rows[species]
    A, a, k = float(r["A"]), float(r["a"]), float(r["k"])

    x_base_frac = np.linspace(0.0, 1.0, n_points)
    x_apex_frac = 1.0 - x_base_frac
    freq_hz = greenwood_frequency(x_apex_frac, A, a, k)

    return ReferenceCurve(
        x=x_base_frac,
        y=freq_hz,
        label=f"Greenwood, {species}",
        source=r["source"],
    )


def erb_hz(f0_hz):
    """
    Equivalent rectangular bandwidth (Hz) of the human auditory filter
    at center frequency f0_hz (Moore & Glasberg 1983, eq. 8). Valid for
    100 < f0_hz < 10000; not enforced here, only documented.
    """
    f0_hz = np.asarray(f0_hz, dtype=float)
    return 6.23e-6 * f0_hz ** 2 + 93.39e-3 * f0_hz + 28.52


def load_erb_q_curve(cf_range_hz=(150.0, 16000.0), n_points=200):
    """
    Q_ERB(CF) = CF / ERB(CF), the human psychoacoustic tuning-sharpness
    reference (Moore & Glasberg 1983), over a range of CFs. Analytic --
    no digitization involved.

    Returns
    -------
    ReferenceCurve
        x : CF (Hz)
        y : Q_ERB (dimensionless)
    """
    cf_hz = np.logspace(np.log10(cf_range_hz[0]), np.log10(cf_range_hz[1]), n_points)
    q_erb = cf_hz / erb_hz(cf_hz)

    return ReferenceCurve(
        x=cf_hz,
        y=q_erb,
        label="Q_ERB, Moore & Glasberg (1983)",
        source="Moore, B.C.J., Glasberg, B.R. (1983). \"Suggested formulae for calculating "
               "auditory-filter bandwidths and excitation patterns.\" J. Acoust. Soc. Am. 74(3), 750-753.",
    )


def load_gain_reference(species):
    """
    Load published active/passive gain figures (dB) at specific frequencies.

    species : value in the 'species' column of data/reference/neely_kim_1986_gain.csv.

    Returns
    -------
    ReferencePoints
        x : frequency (Hz)
        y : gain (dB)
    """
    path = os.path.join(_DATA_DIR, "neely_kim_1986_gain.csv")
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["species"] == species]

    if not rows:
        raise ValueError(f"No gain reference points for species='{species}' in {path}")

    return ReferencePoints(
        x=np.array([float(r["freq_hz"]) for r in rows]),
        y=np.array([float(r["gain_db"]) for r in rows]),
        label=f"Neely & Kim (1986), {species}",
        source=rows[0]["source"],
    )


def load_phase_accumulation_reference(species):
    """
    Load the published reference for total phase accumulation (cycles)
    between two f/CF points.

    species : value in the 'species' column of
        data/reference/tubelli_phase_accumulation.csv.

    Returns
    -------
    ReferencePoints
        x : (f_over_cf_low, f_over_cf_high)
        y : delta_cycles (single value, repeated to match x's length)
    """
    path = os.path.join(_DATA_DIR, "tubelli_phase_accumulation.csv")
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["species"] == species]

    if not rows:
        raise ValueError(f"No phase accumulation reference for species='{species}' in {path}")

    r = rows[0]
    return ReferencePoints(
        x=np.array([float(r["f_over_cf_low"]), float(r["f_over_cf_high"])]),
        y=np.array([float(r["delta_cycles"]), float(r["delta_cycles"])]),
        label=f"Tubelli et al., {species}",
        source=r["source"],
    )
