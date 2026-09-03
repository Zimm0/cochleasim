"""
Generic comparison of model output against a published reference curve
or a small set of reference points. Nothing here knows what "model" or
"reference" mean physically -- it only interpolates and computes error
between two (x, y) series. Species- and study-specific choices (which
reference to load, which model params to run) are made by the caller,
typically in a notebook, using cochleasim.validation.reference and
cochleasim.analysis.

    from cochleasim.analysis.response import compute_response_grid, tonotopic_map
    from cochleasim.validation.reference import load_greenwood_curve
    from cochleasim.validation.compare import compare_curves

    grid = compute_response_grid(HUMAN_PARAMS, freqs_hz)
    x, cf = tonotopic_map(grid)
    result = compare_curves({"Human": (x / HUMAN_PARAMS.L, cf)}, load_greenwood_curve("human"))
    result.plot()
"""

from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class ComparisonResult:
    """
    model : dict[str, tuple[ndarray, ndarray]]
        Label -> (x, y) for each model curve being compared.
    reference : ReferenceCurve or ReferencePoints
    metric : str
        "percent" -> 100*(model-ref)/ref; "absolute" -> model-ref.
    interpolated, error : dict
        Populated by compare_curves: each model curve's y resampled onto
        reference.x, and its per-point error against reference.y.
    """
    model: dict
    reference: object
    metric: str
    interpolated: dict = field(default_factory=dict)
    error: dict = field(default_factory=dict)

    def summary_table(self):
        """
        Per-point comparison as a list of plain dicts (one per reference
        x-value, per model curve) -- pass directly to pandas.DataFrame(...)
        if pandas is available, or use as-is.

        Returns
        -------
        list of dict, each with keys: model, x, model_y, reference_y, error
        """
        rows = []
        ref_x, ref_y = np.asarray(self.reference.x), np.asarray(self.reference.y)
        for label in self.model:
            for xi, my, ry, e in zip(ref_x, self.interpolated[label], ref_y, self.error[label]):
                rows.append({"model": label, "x": xi, "model_y": my, "reference_y": ry, "error": e})
        return rows

    def goodness_of_fit(self, log=False):
        """
        R^2, RMSE and Pearson correlation between each model curve
        (interpolated onto the reference's x) and the reference itself.

        log : bool
            If True, compute on log10 of both series (meaningful for
            roughly exponential/multiplicative quantities, e.g. CF or
            Q10dB -- a fixed % error is a fixed absolute log-error,
            unlike in linear space). Points where either series is <= 0
            are dropped when log=True.

        Returns
        -------
        dict[str, dict]
            label -> {"r2": ..., "rmse": ..., "r": ..., "n": ...}
        """
        ref_y = np.asarray(self.reference.y, dtype=float)
        out = {}
        for label, y_interp in self.interpolated.items():
            y_interp = np.asarray(y_interp, dtype=float)
            valid = ~(np.isnan(y_interp) | np.isnan(ref_y))

            a, b = y_interp[valid], ref_y[valid]
            if log:
                pos = (a > 0) & (b > 0)
                a, b = np.log10(a[pos]), np.log10(b[pos])

            if len(a) < 2:
                out[label] = {"r2": np.nan, "rmse": np.nan, "r": np.nan, "n": len(a)}
                continue

            residuals = a - b
            rmse = float(np.sqrt(np.mean(residuals ** 2)))
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((b - b.mean()) ** 2)
            r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
            r = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else np.nan

            out[label] = {"r2": r2, "rmse": rmse, "r": r, "n": len(a)}
        return out

    def plot(self, ax=None, xlabel="x", ylabel="y", y_scale=1.0):
        """
        y_scale : float
            Multiplies plotted y-values only (e.g. 0.001 to display Hz
            as kHz). Display-time only -- summary_table(), goodness_of_fit()
            and self.error stay in the original units.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 5))
        else:
            fig = ax.figure

        for label, (x, y) in self.model.items():
            ax.plot(x, np.asarray(y) * y_scale, label=label)

        ref = self.reference
        ref_style = "o" if len(np.atleast_1d(ref.x)) <= 10 else "--"
        ax.plot(ref.x, np.asarray(ref.y) * y_scale, ref_style, label=ref.label, color="black")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        return fig, ax


def compare_curves(model, reference, metric="percent"):
    """
    Compare one or more model curves against a single reference, by
    interpolating each model curve onto the reference's x-values.

    model : dict[str, tuple[array_like, array_like]] or tuple[array_like, array_like]
        {label: (x, y)} for multiple curves, or a single (x, y) tuple
        (treated as one curve labeled "model").
    reference : ReferenceCurve or ReferencePoints
    metric : "percent" or "absolute" (see ComparisonResult).

    Returns
    -------
    ComparisonResult
    """
    if isinstance(model, tuple):
        model = {"model": model}

    ref_x, ref_y = np.asarray(reference.x, dtype=float), np.asarray(reference.y, dtype=float)

    interpolated, error = {}, {}
    for label, (x, y) in model.items():
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        order = np.argsort(x)
        y_interp = np.interp(ref_x, x[order], y[order], left=np.nan, right=np.nan)
        interpolated[label] = y_interp

        if metric == "percent":
            error[label] = 100.0 * (y_interp - ref_y) / ref_y
        elif metric == "absolute":
            error[label] = y_interp - ref_y
        else:
            raise ValueError(f"Unknown metric '{metric}'")

    return ComparisonResult(model=model, reference=reference, metric=metric,
                             interpolated=interpolated, error=error)
