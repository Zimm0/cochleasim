"""
Diagnostic plots of the model's own behavior: tuning-curve shape, phase
accumulation, spatial envelopes and the Pd(x, f) surface. Nothing here
compares against external reference data (see cochleasim.validation for
that) -- these are qualitative figures.

Every function accepts xlim/ylim (zlim for the 3D surface) so the caller
can zoom to the informative part of a curve without editing the function
itself -- far skirts routinely fall many tens/hundreds of dB below the
peak and, left unclipped, flatten the part that actually matters.
"""

import numpy as np
import matplotlib.pyplot as plt

from cochleasim.analysis.response import characteristic_place, tuning_curve_at_position, phase_vs_ratio


def plot_gamma_crit(x_cm, gamma_crit, L_cm=None, ax=None, xlabel="x/L (0=base)", ylabel="critical gamma"):
    """Critical gamma vs. position, from gamma_crit_profile(). L_cm plots x/L instead of raw x_cm."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    x = np.asarray(x_cm) / L_cm if L_cm is not None else np.asarray(x_cm)
    ax.plot(x, gamma_crit, "o-")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, label="nominal gamma = 1")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig, ax


def plot_convergence(N_values, values, ax=None, ylabel="value", reference="last"):
    """
    Mesh-convergence plot: a scalar metric vs. number of spatial nodes N.

    reference : "last" or float
        Converged value used for the reference line. "last" (default)
        uses values[-1], the finest mesh tested.
    """
    N_values = np.asarray(N_values)
    values = np.asarray(values, dtype=float)
    ref = values[-1] if reference == "last" else float(reference)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    ax.semilogx(N_values, values, "o-")
    ax.axhline(ref, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("N (spatial nodes)")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.3)
    return fig, ax


def plot_tuning_curves(params, cf_targets_hz, span_octaves=1.5, n_freqs=150,
                        Pe=1.0, N=1000, ax=None, normalize=True,
                        xlim=None, ylim=(-60.0, 5.0)):
    """
    Overlay tuning curves (|xi_b| vs. stimulus frequency) at several
    target CFs; each curve is one fixed cochlear position, swept over
    frequency. Purely illustrative -- no reference curve or error metric
    (see cochleasim.validation.reference.load_erb_q_curve for that).

    normalize : bool
        True (default): each curve in dB relative to its own peak
        (shape only). False: dB re: Pe (comparable absolute level).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    for cf in cf_targets_hz:
        x0 = characteristic_place(params, cf, Pe=Pe, N=N)
        freqs = np.logspace(
            np.log10(cf / 2.0 ** span_octaves),
            np.log10(cf * 2.0 ** span_octaves),
            n_freqs,
        )
        _, xi_b = tuning_curve_at_position(params, x0, freqs, Pe=Pe, N=N)
        mag = np.abs(xi_b)

        ref = mag.max() if normalize else abs(Pe)
        mag_db = 20.0 * np.log10(mag / ref)

        label = f"{cf / 1000:g} kHz" if cf >= 1000 else f"{cf:g} Hz"
        ax.semilogx(freqs, mag_db, label=label)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_xlabel("stimulus frequency (Hz)")
    ax.set_ylabel("level relative to peak (dB)" if normalize else "|xi_b| re: Pe (dB)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    return fig, ax


def plot_phase_vs_ratio(params, cf_targets_hz, span_octaves=1.5, n_freqs=150,
                         Pe=1.0, N=1000, ax=None, xlim=None, ylim=None):
    """Overlay unwrapped phase (cycles) vs. normalized frequency f/CF, at several target CFs."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    for cf in cf_targets_hz:
        x0 = characteristic_place(params, cf, Pe=Pe, N=N)
        freqs = np.logspace(
            np.log10(cf / 2.0 ** span_octaves),
            np.log10(cf * 2.0 ** span_octaves),
            n_freqs,
        )
        _, xi_b = tuning_curve_at_position(params, x0, freqs, Pe=Pe, N=N)
        f_over_cf, phase_cycles = phase_vs_ratio(freqs, xi_b, cf)

        label = f"{cf / 1000:g} kHz" if cf >= 1000 else f"{cf:g} Hz"
        ax.plot(f_over_cf, phase_cycles, label=label)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.axvline(1.0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("f / CF")
    ax.set_ylabel("accumulated phase (cycles)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig, ax


def plot_spatial_response(grid, freqs_to_show=None, quantity="xi_b", ax=None,
                           xlim=None, ylim=(-60.0, 10.0)):
    """
    Spatial envelope |xi_b(x)| or |Pd(x)| at a handful of fixed stimulus
    frequencies, sliced from an already-computed ResponseGrid (cf. Neely
    & Kim's Fig. 5/6).

    xi_b is in dB re: 1 nm; at grid.Pe == P_REF_DYN_CM2 (0 dB SPL) this
    matches the paper's own convention. Values far from a curve's peak
    are legitimately very negative -- expected physics, not a bug; ylim
    keeps the readable part readable.

    freqs_to_show : array_like, optional
        Frequencies (Hz) to show, snapped to the nearest grid row.
        Defaults to 5 log-spaced frequencies spanning the grid.
    quantity : "xi_b" (dB re: 1 nm) or "Pd" (dB re: Pe).
    """
    if quantity == "xi_b":
        data = grid.xi_b
    elif quantity == "Pd":
        data = grid.Pd
    else:
        raise ValueError('quantity must be "xi_b" or "Pd"')

    if freqs_to_show is None:
        idx = np.unique(np.linspace(0, len(grid.freqs_hz) - 1, 5).astype(int))
        label_freqs = grid.freqs_hz[idx]
    else:
        idx = [int(np.argmin(np.abs(grid.freqs_hz - f))) for f in freqs_to_show]
        label_freqs = list(freqs_to_show)  # label with the requested value, not the snapped one

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    x_frac = grid.x_cm / grid.x_cm.max()

    for i, f_label in zip(idx, label_freqs):
        mag = np.abs(data[i, :])
        if quantity == "xi_b":
            mag_nm = mag * 1.0e7  # cm -> nm
            valid = mag_nm > 0
            y = np.full_like(mag_nm, np.nan)
            y[valid] = 20.0 * np.log10(mag_nm[valid])
        else:
            ratio = mag / abs(grid.Pe)
            valid = ratio > 0
            y = np.full_like(ratio, np.nan)
            y[valid] = 20.0 * np.log10(ratio[valid])

        label = f"{f_label / 1000:g} kHz" if f_label >= 1000 else f"{f_label:g} Hz"
        ax.plot(x_frac, y, label=label)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_xlabel("x/L (0=base)")
    ax.set_ylabel("|xi_b| (dB re: 1 nm)" if quantity == "xi_b" else "|Pd| / |Pe| (dB)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig, ax


def _nice_freq_ticks(fmin, fmax):
    candidates = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    return [f for f in candidates if fmin <= f <= fmax]


def plot_pd_surface(grid, ax=None, view=(25, -60),
                     max_freq_points=150, max_position_points=150,
                     xlim=None, ylim=None, zlim=(-80.0, None)):
    """
    3D surface of |Pd(x, f)| / |Pe|, in dB, over the (frequency, position)
    grid from compute_response_grid. Shows the tonotopic ridge and, at
    low frequencies, the separate horizontal band from the global cavity
    resonance discussed alongside the control-theory / stability analysis.

    max_freq_points, max_position_points : int
        The grid is strided down to at most this many points per axis
        before plotting -- interactive backends get slow/unresponsive at
        full density (N=1000 spatial nodes is far denser than needed to
        see the surface's shape). Only affects the plot, not grid itself.
    zlim : tuple(float or None, float or None)
        (low, high) dB clip applied before plotting. Default floor -80 dB
        keeps a few very quiet points from compressing the color/height
        scale; high=None leaves the natural peak. (None, None) disables.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers the '3d' projection

    n_freq, n_pos = grid.Pd.shape
    freq_stride = max(1, n_freq // max_freq_points)
    pos_stride = max(1, n_pos // max_position_points)

    freqs_hz = grid.freqs_hz[::freq_stride]
    x_cm = grid.x_cm[::pos_stride]
    Pd = grid.Pd[::freq_stride, ::pos_stride]

    log_f = np.log10(freqs_hz)
    x_frac = x_cm / grid.x_cm.max()  # 0 = base, 1 = apex
    F, X = np.meshgrid(log_f, x_frac, indexing="ij")

    zlo, zhi = zlim if zlim is not None else (None, None)

    mag = np.abs(Pd) / abs(grid.Pe)
    mag_db = np.full_like(mag, zlo if zlo is not None else -np.inf)
    valid = mag > 0
    mag_db[valid] = 20.0 * np.log10(mag[valid])
    if zlo is not None:
        mag_db = np.clip(mag_db, zlo, None)
    if zhi is not None:
        mag_db = np.clip(mag_db, None, zhi)

    if ax is None:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    ax.plot_surface(F, X, mag_db, cmap="viridis", linewidth=0, antialiased=True)

    ticks = _nice_freq_ticks(freqs_hz.min(), freqs_hz.max())
    ax.set_xticks(np.log10(ticks))
    ax.set_xticklabels([f"{f / 1000:g}k" if f >= 1000 else f"{f:g}" for f in ticks])

    if xlim is not None:
        ax.set_xlim(np.log10(xlim[0]), np.log10(xlim[1]))
    if ylim is not None:
        ax.set_ylim(*ylim)
    if zlo is not None or zhi is not None:
        ax.set_zlim(zlo if zlo is not None else mag_db.min(),
                    zhi if zhi is not None else mag_db.max())

    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("position (fraction of L, 0=base)")
    ax.set_zlabel("|Pd| / |Pe| (dB)")
    ax.view_init(elev=view[0], azim=view[1])

    return fig, ax