"""
Species-agnostic computation of the model's frequency/space response.

Everything in this module operates on a CochlearParams object and plain
numeric arrays. It has no notion of species, of a specific published
parameter table, or of any external reference data -- it only computes
quantities that solve_pressure and displacement_from_pressure can, in
principle, always produce. Comparison against literature (Greenwood,
Liberman, Kiang & Moxon, Neely & Kim's own gain figures, ...) lives in
cochleasim.validation, one layer above this one.
"""

import dataclasses
from dataclasses import dataclass

import numpy as np

from cochleasim.models.cochlear_model import solve_pressure, displacement_from_pressure


@dataclass
class ResponseGrid:
    """
    Basilar-membrane response xi_b(x, f), and the underlying fluid
    pressure Pd(x, f), over a grid of positions and stimulus frequencies,
    for one parameter set.

    Row i of xi_b/Pd is the full spatial response to freqs_hz[i]; column
    j is the tuning curve at x_cm[j]. Pe is the eardrum input pressure
    this grid was computed with (needed to express Pd in dB re: Pe, as
    in Neely & Kim's own Fig. 4).
    """
    x_cm: np.ndarray
    freqs_hz: np.ndarray
    xi_b: np.ndarray
    Pd: np.ndarray
    Pe: complex = 1.0


def compute_response_grid(params, freqs_hz, Pe=1.0, N=1000) -> ResponseGrid:
    """Solve the model at each frequency in freqs_hz, over the full cochlea."""
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    xi_b = np.empty((freqs_hz.size, N), dtype=complex)
    Pd_grid = np.empty((freqs_hz.size, N), dtype=complex)
    x_cm = None

    for i, f in enumerate(freqs_hz):
        omega = 2.0 * np.pi * f
        x, Pd = solve_pressure(omega, Pe, params, N=N)
        xb, _ = displacement_from_pressure(x, Pd, omega, params)
        xi_b[i, :] = xb
        Pd_grid[i, :] = Pd
        x_cm = x

    return ResponseGrid(x_cm=x_cm, freqs_hz=freqs_hz, xi_b=xi_b, Pd=Pd_grid, Pe=Pe)


def characteristic_place(params, freq_hz, Pe=1.0, N=1000):
    """
    Position of maximum |xi_b| for a single stimulus frequency.

    Same definition tonotopic_map uses, one frequency at a time: the
    "characteristic place" of a frequency is where the response peaks
    spatially. Useful to locate the position whose tuning curve should
    be measured for a target CF, without a full tonotopic_map sweep.

    Note this is a different maximization than tonotopic_map's CF(x):
    max over x at fixed f here, vs. max over f at fixed x there. In a
    non-separable model the two need not agree exactly -- the position
    found here may have its own best frequency slightly off from
    freq_hz. q10db_at_cf() reports the tuning curve's own measured CF
    for that reason.

    Returns
    -------
    float
        Position (cm) of maximum response.
    """
    omega = 2.0 * np.pi * freq_hz
    x, Pd = solve_pressure(omega, Pe, params, N=N)
    xb, _ = displacement_from_pressure(x, Pd, omega, params)
    return float(x[np.argmax(np.abs(xb))])


def tonotopic_map(grid: ResponseGrid, min_x_frac=0.01):
    """
    Characteristic frequency at each position, from a ResponseGrid.

    CF(x) is the stimulus frequency that maximizes |xi_b(x, .)| at that
    position. Two situations are reported as NaN rather than a
    misleading number:

    1. Positions whose maximum falls on the first or last row of the
       grid -- the true peak likely lies outside freqs_hz.
    2. Positions within min_x_frac of the stapes (x=0): a confirmed
       boundary-condition artifact where CF sticks at a spurious low
       plateau right at the base before jumping to the physically
       expected high-frequency values just past it.

    min_x_frac default (0.01) is based on direct inspection of where
    the artifact ends for HUMAN_PARAMS; widen it if it persists for
    other parameter sets.

    Returns
    -------
    x_cm : ndarray
    cf_hz : ndarray
        NaN at positions affected by either artifact above.
    """
    idx = np.argmax(np.abs(grid.xi_b), axis=0)
    cf_hz = grid.freqs_hz[idx].copy()

    on_edge = (idx == 0) | (idx == len(grid.freqs_hz) - 1)
    near_base = (grid.x_cm / grid.x_cm.max()) < min_x_frac
    cf_hz[on_edge | near_base] = np.nan

    return grid.x_cm, cf_hz


def tuning_curve_at_position(params, x_cm, freqs_hz, Pe=1.0, N=1000):
    """
    Frequency response xi_b(f) at a single fixed cochlear position.

    Internally solves the full cochlea at each frequency (solve_pressure
    has no cheaper single-position mode) and picks out the node nearest
    to x_cm.

    Returns
    -------
    freqs_hz : ndarray
    xi_b : ndarray of complex
        Response at the grid node closest to x_cm.
    """
    grid = compute_response_grid(params, freqs_hz, Pe=Pe, N=N)
    j = int(np.argmin(np.abs(grid.x_cm - x_cm)))
    return grid.freqs_hz, grid.xi_b[:, j]


def _interp_crossing(freqs_hz, magnitude, i_peak, target, side):
    """
    Frequency at which `magnitude` crosses `target`, searching outward
    from i_peak toward lower (side='below') or higher (side='above')
    frequencies, via linear interpolation between the two bracketing
    samples. Returns np.nan if no crossing is found within freqs_hz.
    """
    seg = range(i_peak, -1, -1) if side == "below" else range(i_peak, len(freqs_hz))

    prev = i_peak
    for i in seg:
        if magnitude[i] <= target:
            f0, f1 = freqs_hz[prev], freqs_hz[i]
            m0, m1 = magnitude[prev], magnitude[i]
            if m1 == m0:
                return f1
            frac = (target - m0) / (m1 - m0)
            return f0 + frac * (f1 - f0)
        prev = i
    return np.nan


def q10db(freqs_hz, magnitude):
    """
    CF, 10 dB bandwidth and Q10dB of a single-peaked tuning curve.

    Pure signal processing on the two arrays supplied: does not know
    where they came from. Assumes `magnitude` (linear units, not dB)
    has a single dominant peak and decreases monotonically on both
    sides down to the -10 dB level; curves with secondary lobes above
    -10 dB will give an unreliable bandwidth.

    Returns
    -------
    cf_hz : float
    bandwidth_hz, q10db : float
        Both np.nan if the -10 dB level is not reached on both sides
        within the supplied frequency range.
    """
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    magnitude = np.asarray(magnitude, dtype=float)

    i_peak = int(np.argmax(magnitude))
    cf_hz = freqs_hz[i_peak]
    target = magnitude[i_peak] * 10.0 ** (-10.0 / 20.0)

    f_lo = _interp_crossing(freqs_hz, magnitude, i_peak, target, side="below")
    f_hi = _interp_crossing(freqs_hz, magnitude, i_peak, target, side="above")

    if np.isnan(f_lo) or np.isnan(f_hi):
        return cf_hz, np.nan, np.nan

    bandwidth_hz = f_hi - f_lo
    return cf_hz, bandwidth_hz, cf_hz / bandwidth_hz


def phase_accumulation(params, cf_hz, f_over_cf_range=(0.02, 0.9), n_freqs=150, Pe=1.0, N=1000):
    """
    Total accumulated phase (cycles) between two f/CF points, at the
    position whose characteristic place is cf_hz.

    Matches the metric reported by Tubelli et al. for a human FE model
    validated against real cochlear motion measurements: phase change
    from f/BF=0.02 to f/BF=0.9 (long-wavelength region, below the
    dispersive short-wavelength region near/above BF). f_over_cf_range
    default matches their definition.

    Returns
    -------
    delta_cycles : float
        Signed; negative = phase lag accumulating with increasing
        frequency, as in the reference.
    """
    x0 = characteristic_place(params, cf_hz, Pe=Pe, N=N)
    freqs = np.linspace(f_over_cf_range[0] * cf_hz, f_over_cf_range[1] * cf_hz, n_freqs)
    _, xi_b = tuning_curve_at_position(params, x0, freqs, Pe=Pe, N=N)
    _, phase_cycles = phase_vs_ratio(freqs, xi_b, cf_hz)
    return phase_cycles[-1] - phase_cycles[0]


def find_converged_N(N_values, values, rel_tol=0.01, reference="last"):
    """
    Smallest N (from N_values, must be sorted ascending) beyond which
    every subsequent value stays within rel_tol of the reference value
    -- a statistical criterion for "N is large enough" rather than
    eyeballing a convergence plot.

    Uses a monotonic threshold crossing: a candidate N only qualifies if
    ALL larger N in the sweep also stay within tolerance, not just that
    one point happening to land close by chance (a mesh sweep can dip
    below tolerance briefly before drifting back out, especially right
    after a mesh-resolution artifact).

    reference : "last" or float
        "last" (default) uses values[-1], the finest mesh tested.

    Returns
    -------
    N_converged : int or None
        None if no N in the sweep satisfies the criterion.
    rel_error : ndarray
        Relative error (fraction, not %) of every point vs. the
        reference, for plotting/inspection.
    """
    N_values = np.asarray(N_values)
    values = np.asarray(values, dtype=float)
    ref = values[-1] if reference == "last" else float(reference)

    rel_error = np.abs(values - ref) / np.abs(ref)

    for i in range(len(N_values)):
        if np.all(rel_error[i:] < rel_tol):
            return int(N_values[i]), rel_error

    return None, rel_error


def q10db_at_cf(params, cf_hz, span_octaves=1.5, n_freqs=120, Pe=1.0, N=1000,
                 max_cf_deviation_octaves=0.5):
    """
    Tuning sharpness (CF, bandwidth, Q10dB) at a target characteristic
    frequency, for one parameter set.

    Chains three steps: locate the position whose characteristic place
    is cf_hz (characteristic_place), sweep a fine local frequency range
    around cf_hz at that fixed position (tuning_curve_at_position), and
    measure the -10 dB bandwidth of the resulting curve (q10db).

    At low frequencies, characteristic_place can lock onto a spatial
    response peak that is not the genuine tonotopic place for cf_hz --
    e.g. a global cavity resonance rather than a local traveling-wave
    peak (this model shows one around 60-170 Hz). When that happens,
    the resulting tuning curve's own measured CF ends up far from the
    requested cf_hz, with a spuriously narrow bandwidth. This is
    detected by checking the deviation (max_cf_deviation_octaves) and
    reported as NaN rather than returned as a misleadingly sharp Q10dB.

    span_octaves : float
        Local sweep covers [cf_hz / 2**span_octaves, cf_hz * 2**span_octaves].
        Widen if q10db() returns NaN (the -10 dB level wasn't reached).

    Returns
    -------
    measured_cf_hz : float
        The tuning curve's own peak, which may differ slightly from the
        requested cf_hz (see characteristic_place's docstring).
    bandwidth_hz, q10db : float
        (nan, nan, nan) if measured_cf_hz deviates from cf_hz by more
        than max_cf_deviation_octaves.
    """
    x0 = characteristic_place(params, cf_hz, Pe=Pe, N=N)
    freqs = np.logspace(
        np.log10(cf_hz / 2.0 ** span_octaves),
        np.log10(cf_hz * 2.0 ** span_octaves),
        n_freqs,
    )
    _, xi_b = tuning_curve_at_position(params, x0, freqs, Pe=Pe, N=N)
    measured_cf, bandwidth_hz, q = q10db(freqs, np.abs(xi_b))

    if abs(np.log2(measured_cf / cf_hz)) > max_cf_deviation_octaves:
        return np.nan, np.nan, np.nan
    return measured_cf, bandwidth_hz, q


def phase_vs_ratio(freqs_hz, complex_response, cf_hz):
    """Unwrapped phase (cycles) vs. normalized frequency f/CF."""
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    complex_response = np.asarray(complex_response, dtype=complex)

    phase_cycles = np.unwrap(np.angle(complex_response)) / (2.0 * np.pi)
    return freqs_hz / cf_hz, phase_cycles


@dataclass
class GainProfile:
    """
    Spatial gain profile of the active model relative to its own passive
    (gamma=0) counterpart, at one stimulus frequency.

    active_mag/passive_mag: |xi_b(x)| with the model's own gamma, and
    with gamma forced to 0, same frequency and grid otherwise.
    """
    x_cm: np.ndarray
    freq_hz: float
    active_mag: np.ndarray
    passive_mag: np.ndarray

    @property
    def gain_db(self) -> np.ndarray:
        """
        20*log10(active/passive) at every position. Boundary nodes
        where passive_mag is exactly 0 (e.g. the apex under a
        pressure-release condition, c_h=0) are NaN rather than raising
        a division warning.
        """
        valid = self.passive_mag > 0
        gain = np.full_like(self.active_mag, np.nan)
        gain[valid] = 20.0 * np.log10(self.active_mag[valid] / self.passive_mag[valid])
        return gain

    @property
    def gain_db_at_cf(self) -> float:
        """Gain at the active model's own characteristic place."""
        i_cf = int(np.argmax(self.active_mag))
        return self.gain_db[i_cf]


def gain_active_vs_passive(params, freq_hz, Pe=1.0, N=1000) -> GainProfile:
    """Compare the active model against itself with gamma forced to 0."""
    omega = 2.0 * np.pi * freq_hz

    x, Pd_active = solve_pressure(omega, Pe, params, N=N)
    xb_active, _ = displacement_from_pressure(x, Pd_active, omega, params)

    passive_params = dataclasses.replace(params, gamma=0.0)
    _, Pd_passive = solve_pressure(omega, Pe, passive_params, N=N)
    xb_passive, _ = displacement_from_pressure(x, Pd_passive, omega, passive_params)

    return GainProfile(
        x_cm=x,
        freq_hz=freq_hz,
        active_mag=np.abs(xb_active),
        passive_mag=np.abs(xb_passive),
    )
