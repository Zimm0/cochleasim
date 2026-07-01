"""
Spatial response reconstruction via frequency-response scaling symmetry.

Implements a frequency-to-space reconstruction method based on scaling
symmetry: given a frequency response measured (or otherwise obtained) at
a single cochlear position -- amplitude and phase as a function of f/BF
-- and a tonotopic map CF(x) for the species, the spatial response to a
fixed stimulus frequency f0 is reconstructed by assuming the frequency-
response shape is the same at every position, only shifted according to
CF(x).

Method
------
At each position x:
    1. Compute the normalized frequency f0 / CF(x).
    2. Look up the response amplitude and phase at that normalized
       frequency in the frequency-response table (linear interpolation
       over f/BF).
    3. Positions where f0/CF(x) falls outside the table's covered range
       cannot be estimated and are reported as NaN.

This module is agnostic to the source of the look-up table: it may come
from real measurements or from a simulated
frequency response. The table is supplied by the caller.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class LookupTable:
    """
    Frequency response measured (or computed) at a single cochlear
    position, to be used as a look-up table for spatial reconstruction.

    Attributes
    ----------
    f_over_BF : ndarray
        Stimulus frequency normalized by the position's best frequency
        (f / BF), strictly increasing.
    amplitude : ndarray
        Response amplitude at each f_over_BF (same units as desired output,
        e.g. pressure in Pa or BM displacement in nm).
    phase_cycles : ndarray
        Response phase at each f_over_BF, in cycles, relative to the
        cochlear input (e.g. stapes motion).
    """
    f_over_BF: np.ndarray
    amplitude: np.ndarray
    phase_cycles: np.ndarray
    BF_hz: float = None

    def __post_init__(self):
        self.f_over_BF = np.asarray(self.f_over_BF, dtype=float)
        self.amplitude = np.asarray(self.amplitude, dtype=float)
        self.phase_cycles = np.asarray(self.phase_cycles, dtype=float)

        if not (self.f_over_BF.shape == self.amplitude.shape == self.phase_cycles.shape):
            raise ValueError(
                "f_over_BF, amplitude and phase_cycles must have the same shape; "
                f"got {self.f_over_BF.shape}, {self.amplitude.shape}, {self.phase_cycles.shape}"
            )
        if self.f_over_BF.size < 2:
            raise ValueError("LookupTable needs at least 2 points to interpolate.")
        if np.any(np.diff(self.f_over_BF) <= 0):
            raise ValueError("f_over_BF must be strictly increasing.")


@dataclass
class SpatialResponse:
    """
    Reconstructed spatial response to a fixed stimulus frequency, obtained
    via scaling-symmetry interpolation of a LookupTable.

    Attributes
    ----------
    frequency_hz : float
        Stimulus frequency (f0) this response was reconstructed for.
    positions_cm : ndarray
        Positions from the cochlear base (cm), as supplied by the caller.
    amplitude : ndarray
        Reconstructed amplitude at each position. NaN where f0/CF(x) falls
        outside the look-up table's range.
    phase_cycles : ndarray
        Reconstructed phase (cycles) at each position. NaN where out of range.
    """
    frequency_hz: float
    positions_cm: np.ndarray
    amplitude: np.ndarray
    phase_cycles: np.ndarray


def reconstruct_spatial_response(
    lookup_table: LookupTable,
    positions_cm: np.ndarray,
    cf_map: np.ndarray,
    frequency_hz: float,
) -> SpatialResponse:
    """
    Reconstruct the spatial response to a fixed stimulus frequency from a
    frequency-response look-up table and a tonotopic map, by assuming the
    frequency response has the same shape at every cochlear position and
    is only shifted according to the local characteristic frequency
    (scaling symmetry).

    Parameters
    ----------
    lookup_table : LookupTable
        Frequency response (amplitude & phase vs f/BF) at one position.
    positions_cm : ndarray
        Positions along the cochlea (cm) at which to reconstruct the
        response.
    cf_map : ndarray
        Characteristic frequency (Hz) at each entry of positions_cm,
        i.e. cf_map[i] = CF(positions_cm[i]). Same shape as positions_cm.
    frequency_hz : float
        Stimulus frequency f0 (Hz) to reconstruct the spatial response for.

    Returns
    -------
    SpatialResponse
        Amplitude and phase at each position. Positions where
        f0 / CF(x) is outside [lookup_table.f_over_BF.min(),
        lookup_table.f_over_BF.max()] are set to NaN.
    """
    positions_cm = np.asarray(positions_cm, dtype=float)
    cf_map = np.asarray(cf_map, dtype=float)
    if positions_cm.shape != cf_map.shape:
        raise ValueError(
            f"positions_cm and cf_map must have the same shape; "
            f"got {positions_cm.shape} and {cf_map.shape}"
        )

    f_ratio = frequency_hz / cf_map

    lo = lookup_table.f_over_BF[0]
    hi = lookup_table.f_over_BF[-1]
    in_range = (f_ratio >= lo) & (f_ratio <= hi)

    amplitude = np.full_like(f_ratio, np.nan)
    phase_cycles = np.full_like(f_ratio, np.nan)

    amplitude[in_range] = np.interp(
        f_ratio[in_range], lookup_table.f_over_BF, lookup_table.amplitude
    )
    phase_cycles[in_range] = np.interp(
        f_ratio[in_range], lookup_table.f_over_BF, lookup_table.phase_cycles
    )

    return SpatialResponse(
        frequency_hz=frequency_hz,
        positions_cm=positions_cm,
        amplitude=amplitude,
        phase_cycles=phase_cycles,
    )

def compute_spatial_responses(
    lookup_table: LookupTable,
    cf_distance_cm: np.ndarray,
    cf_hz: np.ndarray,
    frequencies_hz: list,
    n_points: int = 500,
) -> dict:
    """
    Reconstruct the spatial response for a set of stimulus frequencies,
    from a frequency-response look-up table and a tabulated tonotopic map.

    Builds a fine position grid spanning the tonotopic map's range and
    interpolates CF onto it, so the caller does not need to handle grid
    construction or interpolation directly.

    Parameters
    ----------
    lookup_table : LookupTable
        Frequency response (amplitude & phase vs f/BF) at one position.
    cf_distance_cm : ndarray
        Positions (cm) at which the tonotopic map is tabulated.
    cf_hz : ndarray
        Characteristic frequency (Hz) at each entry of cf_distance_cm.
    frequencies_hz : list of float
        Stimulus frequencies to reconstruct the spatial response for.
    n_points : int
        Number of points in the position grid used for reconstruction.

    Returns
    -------
    dict
        Maps each requested frequency (Hz) to its SpatialResponse.
    """
    cf_distance_cm = np.asarray(cf_distance_cm, dtype=float)
    cf_hz = np.asarray(cf_hz, dtype=float)

    positions_cm = np.linspace(cf_distance_cm.min(), cf_distance_cm.max(), n_points)
    cf_map = np.interp(positions_cm, cf_distance_cm, cf_hz)

    return {
        f0: reconstruct_spatial_response(lookup_table, positions_cm, cf_map, f0)
        for f0 in frequencies_hz
    }


def plot_spatial_responses(
    responses: dict,
    ax_amplitude=None,
    ax_phase=None,
):
    """
    Plot reconstructed spatial responses (amplitude and phase vs position)
    for a set of stimulus frequencies.

    Parameters
    ----------
    responses : dict
        Maps stimulus frequency (Hz) to SpatialResponse, as returned by
        compute_spatial_responses.
    ax_amplitude, ax_phase : matplotlib.axes.Axes, optional
        Axes to draw on. If not provided, a new figure with two stacked
        subplots is created.

    Returns
    -------
    fig, ax_amplitude, ax_phase
        The figure and axes used, for further customization by the caller.
    """
    import matplotlib.pyplot as plt

    if ax_amplitude is None or ax_phase is None:
        fig, (ax_amplitude, ax_phase) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    else:
        fig = ax_amplitude.figure

    all_amplitudes = []
    all_phases = []
    all_positions = []

    for f0, sr in responses.items():
        label = f"{f0/1000:g} kHz" if f0 >= 1000 else f"{f0:g} Hz"
        ax_amplitude.semilogy(sr.positions_cm * 10, sr.amplitude, label=label)
        ax_phase.plot(sr.positions_cm * 10, sr.phase_cycles, label=label)

        valid = ~np.isnan(sr.amplitude)
        if np.any(valid):
            all_amplitudes.append(sr.amplitude[valid])
            all_phases.append(sr.phase_cycles[valid])
        all_positions.append(sr.positions_cm)

    if all_positions:
        positions_concat = np.concatenate(all_positions)
        x_range = positions_concat.max() - positions_concat.min()
        x_margin = 0.08 * x_range
        ax_amplitude.set_xlim(
            (positions_concat.min() - x_margin) * 10,
            (positions_concat.max() + x_margin) * 10,
        )

    if all_amplitudes:
        amp_concat = np.concatenate(all_amplitudes)
        phase_concat = np.concatenate(all_phases)
        ax_amplitude.set_ylim(amp_concat.min() * 0.8, amp_concat.max() * 1.2)
        phase_margin = 0.1 * (phase_concat.max() - phase_concat.min() + 1e-9)
        ax_phase.set_ylim(phase_concat.min() - phase_margin, phase_concat.max() + phase_margin)

    ax_amplitude.set_ylabel("amplitude (Pa)")
    ax_amplitude.set_title("Reconstructed spatial response")
    ax_amplitude.legend()
    ax_amplitude.grid(True, which="both", alpha=0.3)

    ax_phase.set_xlabel("distance from base (mm)")
    ax_phase.set_ylabel("phase (cycles)")
    ax_phase.legend()
    ax_phase.grid(True, alpha=0.3)

    return fig, ax_amplitude, ax_phase


def plot_frequency_response(
    lookup_table: LookupTable,
    ax_amplitude=None,
    ax_phase=None,
):
    """
    Plot the frequency response stored in a LookupTable (amplitude and
    phase as a function of frequency).

    If the LookupTable has a BF_hz value, the x-axis shows frequency in
    Hz. Otherwise it shows the normalized frequency f/BF.

    Parameters
    ----------
    lookup_table : LookupTable
        Frequency response look-up table to plot.
    ax_amplitude, ax_phase : matplotlib.axes.Axes, optional
        Axes to draw on. If not provided, a new figure is created.

    Returns
    -------
    fig, ax_amplitude, ax_phase
    """
    import matplotlib.pyplot as plt

    if ax_amplitude is None or ax_phase is None:
        fig, (ax_amplitude, ax_phase) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    else:
        fig = ax_amplitude.figure

    if lookup_table.BF_hz is not None:
        x = lookup_table.f_over_BF * lookup_table.BF_hz
        xlabel = "frequency (Hz)"
    else:
        x = lookup_table.f_over_BF
        xlabel = "frequency / BF"

    ax_amplitude.semilogy(x, lookup_table.amplitude)
    ax_amplitude.set_ylabel("amplitude (Pa)")
    ax_amplitude.set_title("Frequency response")
    ax_amplitude.grid(True, which="both", alpha=0.3)

    ax_phase.plot(x, lookup_table.phase_cycles)
    ax_phase.set_ylabel("phase (cycles)")
    ax_phase.set_xlabel(xlabel)
    ax_phase.grid(True, alpha=0.3)

    return fig, ax_amplitude, ax_phase


def plot_tonotopic_map(
    cf_distance_cm: np.ndarray,
    cf_hz: np.ndarray,
    ax=None,
):
    """
    Plot a tonotopic map: characteristic frequency as a function of
    distance from the cochlear base.

    Parameters
    ----------
    cf_distance_cm : ndarray
        Distance from the cochlear base (cm).
    cf_hz : ndarray
        Characteristic frequency (Hz) at each position.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If not provided, a new figure is created.

    Returns
    -------
    fig, ax
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure

    ax.semilogy(cf_distance_cm * 10, cf_hz / 1000)
    ax.set_xlabel("distance from base (mm)")
    ax.set_ylabel("CF (kHz)")
    ax.set_title("Tonotopic map")
    ax.grid(True, which="both", alpha=0.3)

    return fig, ax