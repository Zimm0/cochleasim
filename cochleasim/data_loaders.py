"""
Loaders for tabulated reference data bundled with the package.

Each function loads a built-in reference dataset by default. Pass a
custom filepath to load your own CSV instead.
"""

import csv
from importlib.resources import files
from pathlib import Path
import numpy as np

from cochleasim.models.spatial_reconstruction import LookupTable


def load_frequency_response_table(filepath: str = None) -> LookupTable:
    """
    Load a frequency-response look-up table from a CSV with columns
    f_over_BF, amplitude_Pa, phase_cycles.

    Parameters
    ----------
    filepath : str, optional
        Path to a custom CSV file. If not provided, the built-in
        reference table bundled with the package is used.

    Returns
    -------
    LookupTable
    """
    if filepath is not None:
        source = open(filepath, newline="")
    else:
        source = (files("cochleasim.data") / "frequency_response_table.csv").open(
            "r", newline=""
        )

    f_over_BF, amplitude, phase_cycles = [], [], []
    with source as f:
        for row in csv.DictReader(f):
            f_over_BF.append(float(row["f_over_BF"]))
            amplitude.append(float(row["amplitude_Pa"]))
            phase_cycles.append(float(row["phase_cycles"]))

    return LookupTable(
        f_over_BF=np.array(f_over_BF),
        amplitude=np.array(amplitude),
        phase_cycles=np.array(phase_cycles),
    )


def load_gerbil_cf_map(filepath: str = None):
    """
    Load a tonotopic map from a CSV with columns distance_mm, CF_kHz.

    Parameters
    ----------
    filepath : str, optional
        Path to a custom CSV file. If not provided, the built-in
        gerbil tonotopic map bundled with the package is used.

    Returns
    -------
    distance_cm : ndarray
        Distance from the cochlear base, in cm.
    CF_hz : ndarray
        Characteristic frequency at each distance, in Hz.
    """
    if filepath is not None:
        source = open(filepath, newline="")
    else:
        source = (files("cochleasim.data") / "gerbil_tonotopic_map.csv").open(
            "r", newline=""
        )

    distance_mm, CF_kHz = [], []
    with source as f:
        for row in csv.DictReader(f):
            distance_mm.append(float(row["distance_mm"]))
            CF_kHz.append(float(row["CF_kHz"]))

    distance_cm = np.array(distance_mm) / 10.0
    CF_hz = np.array(CF_kHz) * 1000.0
    return distance_cm, CF_hz
