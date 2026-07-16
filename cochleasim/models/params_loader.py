"""
CSV-driven parameter table loader for the Neely & Kim (1986) active
cochlear model.

Lets any parameter table (a species, a published revision, a custom
fit) be swapped in without touching model code: write a CSV with one
row per parameter, using whichever unit system and x-unit the source
table was published in, and load_params_csv() builds a CochlearParams
object with everything already converted to cgs.

CSV columns
-----------
parameter    : name, e.g. "k1", "c3", "H", "L", "gamma"
kind         : "exp" | "linear" | "const"
a, b, c      : coefficients (see formulas below)
offset       : additive shift inside the exponent (0 if unused)
unit_system  : "cgs" | "SI" -- unit system of a/b/c as published
x_unit       : "cm" | "m"   -- unit of x the formula expects (spatial
               parameters only; ignored for scalars)
units        : free-text, for documentation only
source       : free-text citation, for documentation only

Formulas
--------
kind == "exp"     value(x) = a * exp(b * (x + offset)) + c
kind == "linear"  value(x) = a + b * x
kind == "const"   value(x) = a                (x-independent)

Every row must supply all of a, b, c, offset, unit_system, x_unit,
even when unused by that row's kind (use 0 / cgs / cm as filler).
"""

import csv
import os
import numpy as np
from cochleasim.models.cochlear_model import CochlearParams


# SI -> cgs conversion factor for each physical quantity type
_SI_TO_CGS = {
    "stiffness": 0.1,       # N/m^3 -> dyn/cm^3
    "damping": 0.1,         # N.s/m^3 -> dyn.s/cm^3
    "mass_area": 0.1,       # kg/m^2 -> g/cm^2
    "length": 100.0,        # m -> cm
    "area": 1.0e4,          # m^2 -> cm^2
    "density": 0.001,       # kg/m^3 -> g/cm^3
    "dimensionless": 1.0,
}

# which physical quantity each parameter is, for unit conversion
_PARAM_QUANTITY = {
    "k1": "stiffness", "k2": "stiffness", "k3": "stiffness", "k4": "stiffness", "k_m": "stiffness",
    "c1": "damping", "c2": "damping", "c3": "damping", "c4": "damping", "c_m": "damping", "c_h": "damping",
    "m1": "mass_area", "m2": "mass_area", "m_m": "mass_area",
    "L": "length", "H": "length", "W": "length",
    "A_s": "area", "A_m": "area",
    "rho": "density",
    "gamma": "dimensionless", "g": "dimensionless", "b": "dimensionless",
    "G_m": "dimensionless", "N": "dimensionless",
}

# H is spatial-capable (it can taper) even though most tables give it as a constant
SPATIAL_PARAMS = {"k1", "c1", "m1", "k2", "c2", "m2", "k3", "c3", "k4", "c4", "H"}
SCALAR_PARAMS = {"gamma", "g", "b", "L", "W", "k_m", "c_m", "m_m", "A_s", "A_m", "G_m", "rho", "c_h", "N"}
ALL_PARAMS = SPATIAL_PARAMS | SCALAR_PARAMS


def _make_spatial_function(kind, a, b, c, offset, x_unit, quantity, unit_system):
    """Build f(x_cm) -> value_in_cgs from one CSV row. x_cm is always
    the model's internal position variable, in cm."""
    conv = _SI_TO_CGS[quantity] if unit_system == "SI" else 1.0
    x_div = 100.0 if x_unit == "m" else 1.0  # cm -> formula's native x_unit

    def f(x_cm):
        x_native = np.asarray(x_cm, dtype=float) / x_div
        if kind == "exp":
            raw = a * np.exp(b * (x_native + offset)) + c
        elif kind == "linear":
            raw = a + b * x_native
        elif kind == "const":
            raw = a * np.ones_like(x_native)
        else:
            raise ValueError(f"Unknown kind '{kind}'")
        return raw * conv
    return f


def _load_scalar_value(kind, a, quantity, unit_system):
    if kind != "const":
        raise ValueError(f"Scalar parameters must have kind='const' (got '{kind}')")
    conv = _SI_TO_CGS[quantity] if unit_system == "SI" else 1.0
    return a * conv


def load_params_csv(path):
    """Load a CochlearParams object from a CSV parameter table.

    Returns
    -------
    CochlearParams
        Spatial parameters (k1..c4, m1, m2, H) as callables x_cm -> value_cgs.
        Scalar parameters as plain floats. Everything already in cgs.
    """
    rows = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows[row["parameter"].strip()] = row

    missing = ALL_PARAMS - set(rows.keys())
    if missing:
        raise ValueError(f"{path}: missing required parameters: {sorted(missing)}")

    def _row_floats(r):
        return dict(a=float(r["a"]), b=float(r["b"]), c=float(r["c"]), offset=float(r["offset"]))

    spatial = {}
    for name in SPATIAL_PARAMS:
        r = rows[name]
        spatial[name] = _make_spatial_function(
            kind=r["kind"], x_unit=r["x_unit"],
            quantity=_PARAM_QUANTITY[name], unit_system=r["unit_system"],
            **_row_floats(r),
        )

    scalars = {}
    for name in SCALAR_PARAMS:
        r = rows[name]
        scalars[name] = _load_scalar_value(
            kind=r["kind"], a=float(r["a"]),
            quantity=_PARAM_QUANTITY[name], unit_system=r["unit_system"],
        )

    return CochlearParams(
        k1=spatial["k1"], c1=spatial["c1"], m1=spatial["m1"],
        k2=spatial["k2"], c2=spatial["c2"], m2=spatial["m2"],
        k3=spatial["k3"], c3=spatial["c3"],
        k4=spatial["k4"], c4=spatial["c4"],
        H=spatial["H"],
        gamma=scalars["gamma"], g=scalars["g"], b=scalars["b"],
        L=scalars["L"], W=scalars["W"],
        k_m=scalars["k_m"], c_m=scalars["c_m"], m_m=scalars["m_m"],
        A_s=scalars["A_s"], A_m=scalars["A_m"], G_m=scalars["G_m"],
        rho=scalars["rho"], c_h=scalars["c_h"],
        N=int(scalars["N"]),
    )


# Standard tables, pre-loaded once so notebooks/scripts don't repeat load_params_csv()
# calls. Paths are resolved relative to the package, not the working directory.
_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../cochleasim
_DATA_DIR = os.path.join(_PACKAGE_DIR, "data")

CAT_PARAMS = load_params_csv(os.path.join(_DATA_DIR, "cat_table_params.csv"))
UPDATED_CAT_PARAMS = load_params_csv(os.path.join(_DATA_DIR, "updated_cat_table_params.csv"))
HUMAN_PARAMS = load_params_csv(os.path.join(_DATA_DIR, "human_table_params.csv"))

