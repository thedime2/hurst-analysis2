# -*- coding: utf-8 -*-
"""
Envelope Fitting for Fourier-Lanczos Spectra

This module implements power-law envelope fitting a(ω) = k/ω for spectral analysis,
supporting the reproduction of Hurst's Appendix A Figure AI-1 envelope analysis.

The envelope model represents the peak-to-peak amplitude decay observed in
market spectra, interpreted by Hurst as evidence of discrete line spectra with
characteristic frequency structure.

Reference: J.M. Hurst, The Profit Magic of Stock Transaction Timing, Appendix A
"""

import numpy as np
from scipy import stats


def fit_power_law_envelope(frequencies, amplitudes, fixed_slope=-1.0):
    """
    Fit a power-law envelope model to spectral data.

    The model is: a(ω) = k / ω^α
    In log-log space: log(a) = log(k) - α × log(ω)

    For Hurst's analysis, α = 1 (fixed), so the model simplifies to a(ω) = k/ω.

    Parameters
    ----------
    frequencies : array_like
        Frequency values in radians per year (ω)
    amplitudes : array_like
        Corresponding amplitude values
    fixed_slope : float, optional
        Power-law exponent α (default: -1.0 for k/ω model)
        If None, fits α as a free parameter

    Returns
    -------
    fit_params : dict
        Dictionary containing:
        - 'k': Fitted envelope parameter
        - 'alpha': Power-law exponent (equals fixed_slope if provided)
        - 'r_squared': R² coefficient of determination
        - 'rmse': Root mean squared error
        - 'log_k': log(k) for reference
        - 'fit_method': 'fixed_slope' or 'free_slope'

    Notes
    -----
    - Excludes frequencies <= 0 and infinite amplitudes
    - Fits in log-log space using linear regression
    - For fixed slope, solves: log_k = mean(log(a) + α × log(ω))
    - R² is computed in log-log space

    Example
    -------
    >>> fit = fit_power_law_envelope(peak_freq, peak_amp)
    >>> print(f"Envelope: a(ω) = {fit['k']:.4f} / ω")
    >>> print(f"R² = {fit['r_squared']:.4f}")
    """
    frequencies = np.asarray(frequencies)
    amplitudes = np.asarray(amplitudes)

    # Filter out invalid values (freq <= 0, inf, nan)
    valid_mask = (frequencies > 0) & np.isfinite(frequencies) & \
                 (amplitudes > 0) & np.isfinite(amplitudes)
    freq_valid = frequencies[valid_mask]
    amp_valid = amplitudes[valid_mask]

    if len(freq_valid) < 2:
        raise ValueError("Not enough valid data points for envelope fitting")

    # Convert to log-log space
    log_freq = np.log(freq_valid)
    log_amp = np.log(amp_valid)

    if fixed_slope is not None:
        # Fixed-slope fit: log(a) = log(k) - alpha × log(ω)
        # Solve for log(k) = mean(log(a) + alpha × log(ω))
        alpha = fixed_slope
        log_k = np.mean(log_amp - alpha * log_freq)
        k = np.exp(log_k)

        # Compute fitted values for R² calculation
        log_amp_fitted = log_k + alpha * log_freq

        fit_method = 'fixed_slope'

    else:
        # Free-slope fit: use linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_freq, log_amp)

        alpha = slope  # Note: slope will be negative for decay
        log_k = intercept
        k = np.exp(log_k)
        log_amp_fitted = intercept + slope * log_freq

        fit_method = 'free_slope'

    # Compute R² in log-log space
    ss_res = np.sum((log_amp - log_amp_fitted) ** 2)
    ss_tot = np.sum((log_amp - np.mean(log_amp)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Compute RMSE in original space
    amp_fitted = envelope_model(freq_valid, k, alpha)
    rmse = np.sqrt(np.mean((amp_valid - amp_fitted) ** 2))

    return {
        'k': k,
        'alpha': alpha,
        'r_squared': r_squared,
        'rmse': rmse,
        'log_k': log_k,
        'fit_method': fit_method
    }


def fit_upper_envelope(peak_frequencies, peak_amplitudes, fixed_slope=-1.0):
    """
    Fit the upper envelope to spectral peak points.

    This represents the peak-to-peak amplitude envelope, analogous to
    Hurst's upper line in Figure AI-1 (ktop = 0.1875 in hardcoded test).

    Parameters
    ----------
    peak_frequencies : array_like
        Frequencies at detected peaks (rad/year)
    peak_amplitudes : array_like
        Amplitude values at peaks
    fixed_slope : float, optional
        Power-law exponent (default: -1.0)

    Returns
    -------
    fit_params : dict
        Fitted envelope parameters (see fit_power_law_envelope)

    Example
    -------
    >>> peak_idx, peak_freq, peak_amp = find_spectral_peaks(amp, omega_yr)
    >>> upper_fit = fit_upper_envelope(peak_freq, peak_amp)
    >>> print(f"Upper envelope k = {upper_fit['k']:.4f}")
    """
    return fit_power_law_envelope(peak_frequencies, peak_amplitudes, fixed_slope)


def fit_lower_envelope(trough_frequencies, trough_amplitudes, fixed_slope=-1.0):
    """
    Fit the lower envelope to spectral trough points.

    This represents the trough-to-trough amplitude envelope, analogous to
    Hurst's lower line in Figure AI-1 (kbot = 0.0575 in hardcoded test).

    Parameters
    ----------
    trough_frequencies : array_like
        Frequencies at detected troughs (rad/year)
    trough_amplitudes : array_like
        Amplitude values at troughs
    fixed_slope : float, optional
        Power-law exponent (default: -1.0)

    Returns
    -------
    fit_params : dict
        Fitted envelope parameters (see fit_power_law_envelope)

    Example
    -------
    >>> trough_idx, trough_freq, trough_amp = find_spectral_troughs(amp, omega_yr)
    >>> lower_fit = fit_lower_envelope(trough_freq, trough_amp)
    >>> print(f"Lower envelope k = {lower_fit['k']:.4f}")
    """
    return fit_power_law_envelope(trough_frequencies, trough_amplitudes, fixed_slope)


def envelope_model(frequencies, k, alpha=-1.0):
    """
    Evaluate the power-law envelope model at given frequencies.

    Model: a(ω) = k / ω^α

    Parameters
    ----------
    frequencies : array_like
        Frequency values in radians per year (ω)
    k : float
        Envelope parameter
    alpha : float, optional
        Power-law exponent (default: -1.0 for k/ω model)

    Returns
    -------
    envelope_values : ndarray
        Envelope amplitude a(ω) = k / ω^α

    Notes
    -----
    - For frequencies <= 0, returns np.inf
    - Used for visualization and error calculation

    Example
    -------
    >>> omega_yr = np.linspace(0.5, 20, 100)
    >>> upper_line = envelope_model(omega_yr, k_upper)
    >>> plt.plot(omega_yr, upper_line, 'r--', label='Upper envelope')
    """
    frequencies = np.asarray(frequencies, dtype=float)

    # Handle edge case: freq <= 0 gives inf
    envelope = np.where(
        frequencies > 0,
        k / (frequencies ** (-alpha)),  # k * freq^(-alpha)
        np.inf
    )

    return envelope


def compute_fit_quality(amplitudes, fitted_amplitudes):
    """
    Compute quality metrics for envelope fit.

    Parameters
    ----------
    amplitudes : array_like
        Observed amplitude values
    fitted_amplitudes : array_like
        Fitted amplitude values from envelope model

    Returns
    -------
    metrics : dict
        Dictionary containing:
        - 'r_squared': R² coefficient of determination
        - 'rmse': Root mean squared error
        - 'mae': Mean absolute error
        - 'max_error': Maximum absolute error

    Example
    -------
    >>> fitted = envelope_model(frequencies, k_fit)
    >>> metrics = compute_fit_quality(amplitudes, fitted)
    >>> print(f"RMSE = {metrics['rmse']:.4f}")
    """
    amplitudes = np.asarray(amplitudes)
    fitted_amplitudes = np.asarray(fitted_amplitudes)

    # Filter out non-finite values
    valid_mask = np.isfinite(amplitudes) & np.isfinite(fitted_amplitudes)
    amp_valid = amplitudes[valid_mask]
    fit_valid = fitted_amplitudes[valid_mask]

    if len(amp_valid) == 0:
        return {
            'r_squared': np.nan,
            'rmse': np.nan,
            'mae': np.nan,
            'max_error': np.nan
        }

    # R² coefficient of determination
    ss_res = np.sum((amp_valid - fit_valid) ** 2)
    ss_tot = np.sum((amp_valid - np.mean(amp_valid)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Error metrics
    errors = amp_valid - fit_valid
    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(np.abs(errors))
    max_error = np.max(np.abs(errors))

    return {
        'r_squared': r_squared,
        'rmse': rmse,
        'mae': mae,
        'max_error': max_error
    }


def fit_dual_envelope(frequencies, amplitudes, peak_indices, trough_indices):
    """
    Fit both upper (peak) and lower (trough) envelopes simultaneously.

    This is a convenience function that fits both envelopes and returns
    the parameters along with a measure of envelope width.

    Parameters
    ----------
    frequencies : array_like
        Full frequency array (rad/year)
    amplitudes : array_like
        Full amplitude array
    peak_indices : array_like
        Indices of detected peaks
    trough_indices : array_like
        Indices of detected troughs

    Returns
    -------
    dual_fit : dict
        Dictionary containing:
        - 'upper': Upper envelope fit parameters
        - 'lower': Lower envelope fit parameters
        - 'envelope_ratio': k_upper / k_lower
        - 'mean_width': Mean difference between upper and lower envelopes

    Example
    -------
    >>> peak_idx, _, _ = find_spectral_peaks(amp, omega_yr)
    >>> trough_idx, _, _ = find_spectral_troughs(amp, omega_yr)
    >>> dual = fit_dual_envelope(omega_yr, amp, peak_idx, trough_idx)
    >>> print(f"Envelope ratio: {dual['envelope_ratio']:.4f}")
    """
    frequencies = np.asarray(frequencies)
    amplitudes = np.asarray(amplitudes)
    peak_indices = np.asarray(peak_indices)
    trough_indices = np.asarray(trough_indices)

    # Fit upper envelope
    peak_freq = frequencies[peak_indices]
    peak_amp = amplitudes[peak_indices]
    upper_fit = fit_upper_envelope(peak_freq, peak_amp)

    # Fit lower envelope
    trough_freq = frequencies[trough_indices]
    trough_amp = amplitudes[trough_indices]
    lower_fit = fit_lower_envelope(trough_freq, trough_amp)

    # Compute envelope ratio (measure of envelope spread)
    envelope_ratio = upper_fit['k'] / lower_fit['k'] if lower_fit['k'] > 0 else np.inf

    # Compute mean width: difference between upper and lower envelopes
    # Evaluate at a common set of frequencies
    common_freq = np.linspace(
        max(peak_freq.min(), trough_freq.min()),
        min(peak_freq.max(), trough_freq.max()),
        50
    )
    upper_vals = envelope_model(common_freq, upper_fit['k'])
    lower_vals = envelope_model(common_freq, lower_fit['k'])
    mean_width = np.mean(upper_vals - lower_vals)

    return {
        'upper': upper_fit,
        'lower': lower_fit,
        'envelope_ratio': envelope_ratio,
        'mean_width': mean_width
    }
