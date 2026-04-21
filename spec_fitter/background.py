"""
background.py — Pure functions for background subtraction.

To add a new background model, add a function here and import it in app.py.
"""

import numpy as np
from numpy.polynomial.chebyshev import Chebyshev


def _zscale(data: np.ndarray, contrast: float = 0.25):
    """Return (vmin, vmax) using a simple zscale algorithm."""
    valid = data[np.isfinite(data)].ravel()
    if valid.size == 0:
        return 0.0, 1.0
    valid = np.sort(valid)
    n = valid.size
    idx = np.linspace(0, n - 1, min(n, 600)).astype(int)
    sample = valid[idx]
    m = len(sample)
    xfit = np.arange(m, dtype=float)
    coeffs = np.polyfit(xfit, sample, 1)
    slope = coeffs[0] / max(contrast, 1e-10)
    mid = np.polyval(coeffs, m / 2.0)
    vmin = mid - slope * (m / 2.0)
    vmax = mid + slope * (m / 2.0)
    return (
        float(np.clip(vmin, valid[0], valid[-1])),
        float(np.clip(vmax, valid[0], valid[-1])),
    )


def _fit_with_outliers(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    degree: int,
    sigma_upper: float,
    sigma_lower: float,
    max_iters: int,
):
    """
    Fit a Chebyshev polynomial with iterative sigma clipping.

    Returns
    -------
    cheb : Chebyshev
    outlier_mask : np.ndarray of bool, shape = x_fit.shape
        True where a point was sigma-clipped (outlier).
    """
    domain = [float(x_fit[0]), float(x_fit[-1])]
    good = np.ones(len(y_fit), dtype=bool)
    cheb = Chebyshev.fit(x_fit, y_fit, degree, domain=domain)

    for _ in range(max(0, int(max_iters))):
        if good.sum() <= degree + 1:
            break
        cheb = Chebyshev.fit(x_fit[good], y_fit[good], degree, domain=domain)
        resid = y_fit - cheb(x_fit)
        std = float(np.std(resid[good]))
        if std == 0.0:
            break
        good = (resid <= sigma_upper * std) & (resid >= -sigma_lower * std)

    return cheb, ~good  # outlier_mask: True = clipped
