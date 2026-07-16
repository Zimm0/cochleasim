"""
GammaTone filter (Holdsworth et al. 1988) for auditory-filterbank
frequency decomposition.

Implements a single GammaTone channel, designed so its equivalent
rectangular bandwidth (ERB) matches the human auditory filter at its
center frequency (Moore & Glasberg 1983), and filtered via the
impulse-invariant cascade-of-recursive-filters digital implementation
described in Holdsworth et al. (1988), Section 5.

References
----------
Holdsworth, J., Patterson, R., Nimmo-Smith, I., Rice, P. (1988).
    "Implementing a GammaTone Filter Bank." Annex C of the SVOS Final
    Report (Part A: The Auditory Filter Bank).
Moore, B. C. J., & Glasberg, B. R. (1983). "Suggested formulae for
    calculating auditory-filter bandwidths and excitation patterns."
    J. Acoust. Soc. Am., 74(3), 750-753.
"""

from dataclasses import dataclass
from math import factorial, pi

import numpy as np
from scipy.signal import lfilter

def _c_n(n: int) -> float:
    """
    3 dB bandwidth constant c_n, such that the 3 dB bandwidth of an
    order-n GammaTone filter is c_n * b Hz:

        c_n = 2 * sqrt(2^(1/n) - 1)

    Derived directly from the narrow-band magnitude response (Holdsworth
    et al. 1988, eq. 3): |GT(f)|^2 = [1 + ((f-f0)/b)^2]^-n. Solving
    [1 + (df/b)^2]^-n = 1/2 for df gives the half-power (-3 dB) offset
    from f0; the full bandwidth is 2*df. Matches Holdsworth's Table 1
    to the table's quoted 3-decimal precision (e.g. c_4 = 0.870).
    """
    if n < 1:
        raise ValueError(f"order n must be >= 1; got {n}")
    return 2.0 * (2.0 ** (1.0 / n) - 1.0) ** 0.5


def _a_n(n: int) -> float:
    """
    ERB-to-b proportionality constant a_n (Holdsworth et al. 1988, eq. 6):

        a_n = pi * (2n-2)! * 2^-(2n-2) / [(n-1)!]^2

    such that ERB[GT] = a_n * b for large f0/b. Cross-checked against
    Holdsworth's Table 1 (e.g. a_4 = 0.982).
    """
    if n < 1:
        raise ValueError(f"order n must be >= 1; got {n}")
    return pi * factorial(2 * n - 2) * 2.0 ** (-(2 * n - 2)) / factorial(n - 1) ** 2


def erb_hz(f0_hz):
    """
    Equivalent rectangular bandwidth of the human auditory filter at
    center frequency f0_hz (Moore & Glasberg 1983, eq. 8). Valid for
    100 < f0_hz < 10000; not enforced here, only documented.

    Parameters
    ----------
    f0_hz : float or ndarray

    Returns
    -------
    float or ndarray
        ERB in Hz.
    """
    f0_hz = np.asarray(f0_hz, dtype=float)
    return 6.23e-6 * f0_hz**2 + 93.39e-3 * f0_hz + 28.52


def gammatone_bandwidth_hz(f0_hz, order: int = 4):
    """
    One-sided pole bandwidth b (Hz) such that an order-`order` GammaTone
    filter centered at f0_hz has equivalent rectangular bandwidth equal
    to the human auditory ERB(f0_hz) (Holdsworth et al. 1988, Section 3).

    Parameters
    ----------
    f0_hz : float or ndarray
    order : int

    Returns
    -------
    float or ndarray
        b in Hz.
    """
    return erb_hz(f0_hz) / _a_n(order)


def gammatone_3db_bandwidth_hz(f0_hz, order: int = 4):
    """
    3 dB bandwidth (Hz) of a GammaTone filter designed to match
    ERB(f0_hz) at the given order (Holdsworth et al. 1988, eq. 7).

    Parameters
    ----------
    f0_hz : float or ndarray
    order : int

    Returns
    -------
    float or ndarray
        3 dB bandwidth in Hz.
    """
    b = gammatone_bandwidth_hz(f0_hz, order)
    return _c_n(order) * b


@dataclass
class GammatoneFilter:
    """
    Specification of a single GammaTone auditory filter channel.

    By default, `bandwidth_hz` is derived from `center_frequency_hz` and
    `order` so the filter's equivalent rectangular bandwidth matches the
    human auditory ERB (Moore & Glasberg 1983) at that center frequency.
    A custom `bandwidth_hz` may be supplied instead (e.g. once a
    HearingProfile modulates bandwidth to model outer hair cell damage).

    Attributes
    ----------
    center_frequency_hz : float
        f0, the filter's center frequency (Hz).
    order : int
        Filter order n. Holdsworth et al. (1988) use n=4 as a standard
        choice for auditory modeling.
    bandwidth_hz : float, optional
        b, the one-sided pole bandwidth (Hz). If not supplied, computed
        from ERB(center_frequency_hz) and order.
    phase_rad : float
        Carrier phase phi (radians), as in Holdsworth et al. (1988) eq. 1.
    """
    center_frequency_hz: float
    order: int = 4
    bandwidth_hz: float = None
    phase_rad: float = 0.0

    def __post_init__(self):
        if self.center_frequency_hz <= 0:
            raise ValueError(
                f"center_frequency_hz must be positive; got {self.center_frequency_hz}"
            )
        if self.order < 1:
            raise ValueError(f"order must be >= 1; got {self.order}")
        if self.bandwidth_hz is None:
            self.bandwidth_hz = gammatone_bandwidth_hz(self.center_frequency_hz, self.order)
        if self.bandwidth_hz <= 0:
            raise ValueError(f"bandwidth_hz must be positive; got {self.bandwidth_hz}")

    @property
    def bandwidth_3db_hz(self) -> float:
        """3 dB bandwidth (Hz) implied by this filter's b and order."""
        return _c_n(self.order) * self.bandwidth_hz


def _recursive_lowpass_stage(x: np.ndarray, alpha: float) -> np.ndarray:
    """
    One stage of Holdsworth et al. (1988) eq. 10, the impulse-invariant
    first-order recursive low-pass filter:

        w[k] = w[k-1] + alpha * (x[k-1] - w[k-1])
             = (1-alpha) * w[k-1] + alpha * x[k-1]

    implemented as a standard IIR difference equation via scipy.signal.lfilter
    (equivalent to, but much faster than, an explicit Python loop). Supports
    complex input, as needed for the frequency-shifted baseband signal.
    """
    b = [0.0, alpha]
    a = [1.0, -(1.0 - alpha)]
    return lfilter(b, a, x)


def apply_gammatone_filter(
    signal: np.ndarray,
    gt_filter: GammatoneFilter,
    sample_rate_hz: float,
) -> np.ndarray:
    """
    Filter a real-valued signal through a single GammaTone channel using
    the digital implementation of Holdsworth et al. (1988), Section 5:
    frequency-shift to baseband (eq. 9), cascade of `order` first-order
    recursive low-pass filters (eq. 10), frequency-shift back and take
    the real part (eq. 11).

    Parameters
    ----------
    signal : ndarray, shape (n_samples,)
        Real-valued input waveform.
    gt_filter : GammatoneFilter
        Filter specification (center frequency, order, bandwidth).
    sample_rate_hz : float
        Sampling rate of `signal` (Hz).

    Returns
    -------
    ndarray, shape (n_samples,)
        Real-valued filtered signal.

    Notes
    -----
    This does not implement Holdsworth's data-doubling (over-sampling)
    safeguard for center frequencies close to the Nyquist frequency
    (f0 > (4*dt)^-1); callers working near the Nyquist limit should
    up-sample before filtering.
    """
    signal = np.asarray(signal, dtype=float)
    dt = 1.0 / sample_rate_hz
    n_samples = signal.shape[0]
    k = np.arange(n_samples)

    f0 = gt_filter.center_frequency_hz
    b = gt_filter.bandwidth_hz
    n = gt_filter.order

    # eq. 9: shift down to baseband
    z = np.exp(-1j * 2 * np.pi * f0 * k * dt) * signal

    # eq. 10, cascaded `n` times
    alpha = 1.0 - np.exp(-2 * np.pi * b * dt)
    w = z
    for _ in range(n):
        w = _recursive_lowpass_stage(w, alpha)

    # eq. 11: shift back up, take real part
    y = np.real(np.exp(1j * 2 * np.pi * f0 * k * dt) * w)
    return y


def gammatone_impulse_response(
    t,
    f0_hz: float,
    bandwidth_hz: float,
    order: int = 4,
    phase_rad: float = 0.0,
    amplitude: float = 1.0,
):
    """
    Analytic GammaTone impulse response (Holdsworth et al. 1988, eq. 1):

        gt(t) = A * t^(n-1) * exp(-2*pi*b*t) * cos(2*pi*f0*t + phi),  t >= 0

    Used as the closed-form reference against which the digital cascade
    implementation (`apply_gammatone_filter`) is validated.

    Parameters
    ----------
    t : float or ndarray
        Time (s), may include negative values (response is 0 there).
    f0_hz : float
        Center frequency (Hz).
    bandwidth_hz : float
        b, one-sided pole bandwidth (Hz).
    order : int
        Filter order n.
    phase_rad : float
        Carrier phase phi (radians).
    amplitude : float
        Scale factor A.

    Returns
    -------
    float or ndarray
    """
    t = np.asarray(t, dtype=float)
    envelope = np.where(
        t >= 0, t ** (order - 1) * np.exp(-2 * np.pi * bandwidth_hz * t), 0.0
    )
    return amplitude * envelope * np.cos(2 * np.pi * f0_hz * t + phase_rad)


def gammatone_frequency_response(f_hz, f0_hz: float, bandwidth_hz: float, order: int = 4, exact: bool = False):
    """
    Analytic GammaTone frequency response (Holdsworth et al. 1988).

    By default returns the narrow-band approximation (eq. 3):

        GT(f) ~= [1 + j*(f-f0)/b]^-n     (0 < f < inf)

    which is the one actually realized by the cascade-of-recursive-filters
    digital implementation, and is a good approximation whenever f0/b is
    large -- always true in the auditory range Holdsworth targets
    (4 < f0/b < 8).

    If `exact=True`, returns the full two-sided expression (eq. 2):

        GT(f) = [1 + j*(f-f0)/b]^-n + [1 + j*(f+f0)/b]^-n

    Parameters
    ----------
    f_hz : float or ndarray
    f0_hz : float
    bandwidth_hz : float
    order : int
    exact : bool

    Returns
    -------
    complex or ndarray of complex
        Complex frequency response (not yet converted to dB).
    """
    f = np.asarray(f_hz, dtype=float)
    term1 = (1 + 1j * (f - f0_hz) / bandwidth_hz) ** (-order)
    if not exact:
        return term1
    term2 = (1 + 1j * (f + f0_hz) / bandwidth_hz) ** (-order)
    return term1 + term2