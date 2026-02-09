# -*- coding: utf-8 -*-
"""
Least-squares estimation (LSE) smoothing of frequency traces.

Converts noisy frequency-vs-time scatter from comb filter measurements
into smooth quasi-horizontal lines, implementing the "LSE, Frequency vs
Time Analysis" shown in Hurst's Figure AI-6.

Reference: J.M. Hurst, The Profit Magic of Stock Transaction Timing,
           Appendix A, Figure AI-6
"""

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter


def smooth_frequency_trace(times, freqs, center_freq=None, method='median_savgol',
                            median_window=15, savgol_window=31, savgol_order=2):
    """
    Smooth a frequency-vs-time trace using robust filtering.

    First applies a median filter to remove outliers, then Savitzky-Golay
    for smooth trend estimation.

    Parameters
    ----------
    times : array
        Sample indices at which frequencies were measured
    freqs : array
        Measured frequencies in rad/yr
    center_freq : float, optional
        Expected center frequency for outlier removal. If None, uses
        the median of freqs.
    method : str
        'median_savgol' (default), 'median_only', or 'savgol_only'
    median_window : int
        Window size for median filter (must be odd)
    savgol_window : int
        Window size for Savitzky-Golay filter (must be odd)
    savgol_order : int
        Polynomial order for Savitzky-Golay filter

    Returns
    -------
    dict with:
        times : array - input times
        freqs_raw : array - input frequencies (after outlier removal)
        freqs_smoothed : array - smoothed frequencies
        n_outliers : int - number of outlier points removed
    """
    if len(times) < 5:
        return {
            'times': times,
            'freqs_raw': freqs,
            'freqs_smoothed': freqs.copy() if len(freqs) > 0 else np.array([]),
            'n_outliers': 0
        }

    # Outlier removal: keep points within 30% of center frequency
    if center_freq is None:
        center_freq = np.median(freqs)
    tolerance = center_freq * 0.3
    valid = (freqs > center_freq - tolerance) & (freqs < center_freq + tolerance)
    n_outliers = np.sum(~valid)

    t_clean = times[valid]
    f_clean = freqs[valid]

    if len(f_clean) < 5:
        return {
            'times': t_clean,
            'freqs_raw': f_clean,
            'freqs_smoothed': f_clean.copy(),
            'n_outliers': n_outliers
        }

    # Sort by time
    sort_idx = np.argsort(t_clean)
    t_sorted = t_clean[sort_idx]
    f_sorted = f_clean[sort_idx]

    # Apply smoothing
    if method == 'median_savgol' or method == 'median_only':
        # Median filter for robust outlier handling
        mw = min(median_window, len(f_sorted))
        if mw % 2 == 0:
            mw += 1
        f_smoothed = median_filter(f_sorted, size=mw)

        if method == 'median_savgol' and len(f_smoothed) >= savgol_window:
            sw = min(savgol_window, len(f_smoothed))
            if sw % 2 == 0:
                sw += 1
            so = min(savgol_order, sw - 1)
            f_smoothed = savgol_filter(f_smoothed, sw, so)
    elif method == 'savgol_only':
        sw = min(savgol_window, len(f_sorted))
        if sw % 2 == 0:
            sw += 1
        so = min(savgol_order, sw - 1)
        f_smoothed = savgol_filter(f_sorted, sw, so)
    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        'times': t_sorted,
        'freqs_raw': f_sorted,
        'freqs_smoothed': f_smoothed,
        'n_outliers': n_outliers
    }


def fit_frequency_line(times, freqs, center_freq=None):
    """
    Fit a linear model to a frequency trace: f(t) = f0 + drift * t.

    This estimates whether a frequency line is truly stationary or
    drifting slowly over time.

    Parameters
    ----------
    times : array
        Sample indices
    freqs : array
        Measured frequencies in rad/yr
    center_freq : float, optional
        Expected center for outlier removal

    Returns
    -------
    dict with:
        mean_freq : float - time-averaged frequency (rad/yr)
        drift_rate : float - frequency drift (rad/yr per sample)
        drift_rate_annual : float - drift per year (rad/yr/yr)
        intercept : float - f(0) from linear fit
        r_squared : float - R^2 of linear fit
        std_dev : float - standard deviation of residuals
        period_weeks : float - period corresponding to mean_freq
        n_points : int - number of points used
    """
    if len(times) < 3 or len(freqs) < 3:
        mean_f = np.mean(freqs) if len(freqs) > 0 else 0.0
        return {
            'mean_freq': mean_f,
            'drift_rate': 0.0,
            'drift_rate_annual': 0.0,
            'intercept': mean_f,
            'r_squared': 0.0,
            'std_dev': 0.0,
            'period_weeks': 2 * np.pi / mean_f * 52 if mean_f > 0 else np.inf,
            'n_points': len(freqs)
        }

    # Outlier removal
    if center_freq is None:
        center_freq = np.median(freqs)
    tolerance = center_freq * 0.3
    valid = (freqs > center_freq - tolerance) & (freqs < center_freq + tolerance)
    t_clean = times[valid]
    f_clean = freqs[valid]

    if len(t_clean) < 3:
        mean_f = np.mean(freqs)
        return {
            'mean_freq': mean_f,
            'drift_rate': 0.0,
            'drift_rate_annual': 0.0,
            'intercept': mean_f,
            'r_squared': 0.0,
            'std_dev': np.std(freqs),
            'period_weeks': 2 * np.pi / mean_f * 52 if mean_f > 0 else np.inf,
            'n_points': len(f_clean)
        }

    # Linear fit: f = intercept + drift * t
    coeffs = np.polyfit(t_clean, f_clean, 1)
    drift_rate = coeffs[0]
    intercept = coeffs[1]

    # Fit quality
    f_fitted = np.polyval(coeffs, t_clean)
    residuals = f_clean - f_fitted
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((f_clean - np.mean(f_clean)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    mean_freq = np.mean(f_clean)

    return {
        'mean_freq': mean_freq,
        'drift_rate': drift_rate,
        'drift_rate_annual': drift_rate * 52,  # samples/yr = 52
        'intercept': intercept,
        'r_squared': r_squared,
        'std_dev': np.std(residuals),
        'period_weeks': 2 * np.pi / mean_freq * 52 if mean_freq > 0 else np.inf,
        'n_points': len(f_clean)
    }
