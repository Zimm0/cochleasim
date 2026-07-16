"""
Active mechanical model of the cochlea (Neely & Kim, 1986).

Implements the linear, frequency-domain, one-dimensional box model of
cochlear macromechanics coupled to a two-degree-of-freedom (BM + TM)
micromechanical model of the organ of Corti, with an active pressure
source representing outer hair cell (OHC) electromotility ("negative
damping" cochlear amplifier).

This module is agnostic to any specific species or parameter table:
CochlearParams bundles the ten spatially-varying micromechanical
functions (k1..k4, c1..c4, m1, m2), the scala height H (also
spatially-varying, to allow a tapered cross-section), and the scalar
macromechanical/middle-ear parameters, as a single explicit object.
Load a parameter table from CSV with params_loader.load_params_csv();
see cochleasim/data/*.csv for example tables (cat, human).

References
----------
Neely, S.T., Kim, D.O. (1986). "A model for active elements in
cochlear biomechanics." J. Acoust. Soc. Am. 79(5), 1472-1480.
Ku, E.M. (2010). "Modelling the Human Cochlea." PhD thesis, University
of Southampton, Institute of Sound and Vibration Research.
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable
from scipy.linalg import solve_banded


P_REF_DYN_CM2 = 2.0e-4
N_DEFAULT = 1000


@dataclass
class CochlearParams:
    # spatially-varying micromechanical functions: x_cm -> value_cgs
    k1: Callable
    c1: Callable
    m1: Callable
    k2: Callable
    c2: Callable
    m2: Callable    
    k3: Callable
    c3: Callable
    k4: Callable
    c4: Callable
    H: Callable  # scala height (cm); constant or tapered depending on the table

    # scalar parameters (cgs)
    gamma: float
    g: float
    b: float
    L: float
    W: float
    k_m: float
    c_m: float
    m_m: float
    A_s: float
    A_m: float
    G_m: float
    rho: float
    c_h: float
    N: int  # table's suggested grid size (informational; solve_pressure's own N argument governs the actual solve)


def impedance_Z1(x, omega, params: CochlearParams):
    """Mechanical impedance of the organ of Corti (OC), Z1 = k1/(i*omega) + c1 + i*omega*m1."""
    return params.k1(x) / (1j * omega) + params.c1(x) + 1j * omega * params.m1(x)


def impedance_Z2(x, omega, params: CochlearParams):
    """Mechanical impedance of the tectorial membrane (TM), Z2 = k2/(i*omega) + c2 + i*omega*m2."""
    return params.k2(x) / (1j * omega) + params.c2(x) + 1j * omega * params.m2(x)


def impedance_Z3(x, omega, params: CochlearParams):
    """Coupling impedance between OC and TM, Z3 = k3/(i*omega) + c3."""
    return params.k3(x) / (1j * omega) + params.c3(x)


def impedance_Z4(x, omega, params: CochlearParams):
    """Impedance shaping the active pressure source's frequency-dependent phase, Z4 = k4/(i*omega) + c4."""
    return params.k4(x) / (1j * omega) + params.c4(x)


def partition_impedance(x, omega, params: CochlearParams):
    """
    Driving-point impedance of the cochlear partition, Zp (eq. 12):

        Zp = (g/b) * [Z1 + Z2*(Z3 - gamma*Z4)/(Z2+Z3)]
    """
    Z1 = impedance_Z1(x, omega, params)
    Z2 = impedance_Z2(x, omega, params)
    Z3 = impedance_Z3(x, omega, params)
    Z4 = impedance_Z4(x, omega, params)
    return (params.g / params.b) * (Z1 + Z2 * (Z3 - params.gamma * Z4) / (Z2 + Z3))


def _middle_ear_impedance(omega, params: CochlearParams):
    """Zm = k_m/(i*omega) + c_m + i*omega*m_m."""
    return params.k_m / (1j * omega) + params.c_m + 1j * omega * params.m_m


def solve_pressure(omega, Pe, params, N=251):
    """
    Resuelve el sistema tridiagonal derivado para Pd(x) a una frecuencia omega.
 
    Parameters
    ----------
    omega : float
        Frecuencia angular (rad/s).
    Pe : complex
        Presión de entrada en el tímpano.
    params : object
        Debe exponer: L, H, W, rho, As, Am, Gm, ch, k_m, c_m, m_m, Zp(x, omega)
    N : int
        Número de nodos de la malla.
 
    Returns
    -------
    x : ndarray (N,)
    Pd : ndarray (N,) complex
    """
    L, H, W, rho = params.L, params.H, params.W, params.rho
    As, Am, Gm, ch = params.A_s, params.A_m, params.G_m, params.c_h
 
    Delta = L / (N - 1)
    x = np.arange(N) * Delta
 
    Zm = _middle_ear_impedance(omega, params)
    H = params.H(0.0)  # H se trata como constante
 
    # --- semilla del oído medio (base), método de un solo lado ---
    # a0 = -2i*omega*rho*Delta*As / (Zm*W*H)
    a0 = -2j * omega * rho * Delta * As / (Zm * W * H)
    # p0 = (Am/(Gm*As)) * a0 * Pe   -> As se cancela
    p0 = -2j * omega * rho * Delta * Am * Pe / (Zm * Gm * W * H)
 
    a = np.zeros(N + 1, dtype=complex)  # a[0]=a0 ; a[1..N] <-> a1..aN
    p = np.zeros(N + 1, dtype=complex)
    a[0] = a0
    p[0] = p0
 
    # fila 1: (1+a0) Pd1 - Pd2 = p0  ->  a1 = 1/(1+a0)
    a[1] = 1.0 / (1.0 + a[0])
    p[1] = a[1] * p[0]
 
    # filas interiores n=2..N-1
    for n in range(2, N):
        zn = (H / Delta) * partition_impedance(x[n - 1], omega, params)  # x[n-1] = x_n (x 0-indexado)
        a[n] = 1.0 / (2.0 - a[n - 1] + 2j * omega * rho * Delta / zn)
        p[n] = a[n] * p[n - 1]
 
    # cierre en el apice: cruce de (dagger) con la fila N -> primer valor numerico
    # PdN = p_{N-1} / (1 - a_{N-1} - 2i*omega*rho*Delta/ch)
    Pd = np.zeros(N, dtype=complex)
    if ch is not None and ch != 0:
        denom = 1.0 - a[N - 1] - 2j * omega * rho * Delta / ch
        Pd[N - 1] = p[N - 1] / denom
    else:
        # limite ch -> 0 (liberacion de presion): el termino 2i*omega*rho*Delta/ch
        # diverge, y Pd(x_N) -> 0 directamente -- no se fuerza la division por cero.
        Pd[N - 1] = 0.0
 
    # sustitucion regresiva: Pd(x_n) = p_n + a_n * Pd(x_{n+1}), n=N-1..1
    for n in range(N - 1, 0, -1):
        Pd[n - 1] = p[n] + a[n] * Pd[n]
 
    return x, Pd


def displacement_from_pressure(x, Pd, omega, params: CochlearParams):
    """
    Compute basilar membrane displacement xi_b(x) and hair-bundle shear
    displacement xi_c(x) from the solved pressure Pd(x) (eqs. 13-14).

    Returns
    -------
    xi_b, xi_c : ndarray of complex
    """
    Zp = partition_impedance(x, omega, params)
    xi_b = Pd / (1j * omega * params.b * Zp)

    Z2 = impedance_Z2(x, omega, params)
    Z3 = impedance_Z3(x, omega, params)
    xi_c = (params.g * Z2 / (Z2 + Z3)) * xi_b

    return xi_b, xi_c
