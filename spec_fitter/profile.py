"""
profile.py — Pure functions for Gauss-Hermite spatial profile fitting.

To add a new spatial profile model, add a function here and import it in app.py.
"""

import math

import numpy as np


def _hermite_poly_e(n, z):
    """Probabilists' Hermite polynomial He_n(z) via recurrence.

    He_0 = 1,  He_1 = z,  He_{k+1} = z·He_k - k·He_{k-1}
    """
    if n == 0:
        return np.ones_like(z, dtype=float)
    he_prev = np.ones_like(z, dtype=float)
    he_curr = z.copy().astype(float)
    for k in range(1, n):
        he_next = z * he_curr - k * he_prev
        he_prev = he_curr
        he_curr = he_next
    return he_curr


def _gauss_hermite_profile(x, amplitude, mu, sigma, *h_coeffs):
    """N-th order Gauss-Hermite spatial profile.

    h_coeffs[0] = h3, h_coeffs[1] = h4, …, h_coeffs[k] = h_{k+3}.
    Each term uses the probabilists' Hermite polynomial He_n normalized
    by sqrt(n!) so that all h_n coefficients remain of order unity.
    Pure Gaussian when h_coeffs is empty.
    """
    z = (x - mu) / max(abs(sigma), 1e-10)
    correction = np.ones_like(z, dtype=float)
    for k, hk in enumerate(h_coeffs):
        n = k + 3
        norm = float(np.sqrt(float(math.factorial(n))))
        correction = correction + hk * _hermite_poly_e(n, z) / norm
    return amplitude * np.exp(-0.5 * z ** 2) * correction
