"""
Control-theory analysis of the cochlear model's active feedback loop, as
formalized in the memoria (apartado 4.1.3, ecuaciones 17-20):

    Zp(x, s) = Zp_pasiva(x, s) * (1 + L(x, s))
    L(x, s)  = gamma * L1(x, s),   L1(x, s) = -Z4(x,s) * H(x,s) / Zp_pasiva(x,s)

Because gamma enters L(s) as a pure multiplicative scalar (it does not
appear inside L1), the classic gain margin of L1(s) computed at gamma=1
is *exactly* the critical gamma at which the closed loop becomes
unstable: the system is marginally stable when gamma * |L1(j*w_pc)| = 1,
i.e. gamma_crit = 1 / |L1(j*w_pc)| = GM(L1). This also means a root
locus of L1(s) (varying its overall gain from 0 upward) directly shows
how the zeros of Zp -- the poles of the mechanical response xi_b(s) --
migrate as gamma increases, crossing into the right half-plane exactly
at gamma_crit.

All functions here take a single position x_cm and return python-control
TransferFunction objects or scalars; nothing here knows about frequency
sweeps over the whole cochlea (that's ResponseGrid, in response.py) or
about a specific parameter table's species.
"""

import control as ct

from cochleasim.models.cochlear_model import CochlearParams


def _impedance_tf(s, k, c, m=0.0):
    """Z(s) = m*s + c + k/s, as a control.TransferFunction (mass term optional)."""
    return m * s + c + k / s


def open_loop_gain_tf(params: CochlearParams, x_cm: float) -> ct.TransferFunction:
    """
    L1(x, s): the loop gain of the active feedback loop with gamma
    factored out (i.e. L(x, s) = gamma * L1(x, s)).

    Parameters
    ----------
    params : CochlearParams
    x_cm : float
        Position along the cochlea (cm) at which the spatially-varying
        coefficients (k1..k4, c1..c4, m1, m2) are evaluated.

    Returns
    -------
    control.TransferFunction
    """
    s = ct.tf("s")

    Z1 = _impedance_tf(s, params.k1(x_cm), params.c1(x_cm), params.m1(x_cm))
    Z2 = _impedance_tf(s, params.k2(x_cm), params.c2(x_cm), params.m2(x_cm))
    Z3 = _impedance_tf(s, params.k3(x_cm), params.c3(x_cm))
    Z4 = _impedance_tf(s, params.k4(x_cm), params.c4(x_cm))

    H = params.g * Z2 / (Z2 + Z3)
    Zp_pasiva = (params.g / params.b) * (Z1 + Z2 * Z3 / (Z2 + Z3))

    return -Z4 * H / Zp_pasiva


def critical_gamma(params: CochlearParams, x_cm: float):
    """
    gamma_crit at a given position: the gain margin of L1(x, s).

    Returns
    -------
    gamma_crit : float
        NaN if the loop has no phase crossover (gain margin undefined).
    margins : dict
        Raw output of control.margin(): gm, pm, wg, wp (gain margin,
        phase margin, gain crossover freq, phase crossover freq).
    """
    L1 = open_loop_gain_tf(params, x_cm)
    gm, pm, wg, wp = ct.margin(L1)
    return gm, {"gm": gm, "pm": pm, "wg": wg, "wp": wp}


def gamma_crit_profile(params: CochlearParams, freqs_hz, N=1000):
    """
    critical gamma at the characteristic place of each frequency in freqs_hz.

    Parameters
    ----------
    params : CochlearParams
    freqs_hz : array_like
        Frequencies whose characteristic place is used as the test
        position (same pattern as q10db_at_cf / plot_tuning_curves).
    N : int
        Passed to characteristic_place.

    Returns
    -------
    x_cm : ndarray
    gamma_crit : ndarray
        NaN where control.margin() could not find a gain margin.
    """
    import numpy as np

    from cochleasim.analysis.response import characteristic_place

    freqs_hz = np.asarray(freqs_hz, dtype=float)
    x_cm = np.empty_like(freqs_hz)
    gamma_crit = np.empty_like(freqs_hz)

    for i, f in enumerate(freqs_hz):
        x0 = characteristic_place(params, f, N=N)
        x_cm[i] = x0
        try:
            gm, _ = critical_gamma(params, x0)
            gamma_crit[i] = gm if gm is not None else np.nan
        except Exception:
            gamma_crit[i] = np.nan

    return x_cm, gamma_crit
