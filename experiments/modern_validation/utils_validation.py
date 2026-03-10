# -*- coding: utf-8 -*-
"""
Utility functions for modern validation experiments (Phase 6).

Provides data loading, fs computation, similarity scoring, and
era-comparison plotting for extending Hurst's analysis to modern data.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pandas as pd
from src.spectral.lanczos import lanczos_spectrum
from src.spectral.peak_detection import find_spectral_peaks, find_spectral_troughs
from src.spectral.envelopes import fit_power_law_envelope


# =============================================================================
# DATA LOADING
# =============================================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

DATA_FILES = {
    'djia_weekly':  os.path.join(BASE_DIR, 'data/raw/^dji_w.csv'),
    'djia_daily':   os.path.join(BASE_DIR, 'data/raw/^dji_d.csv'),
    'spx_weekly':   os.path.join(BASE_DIR, 'data/raw/^spx_w.csv'),
    'spx_daily':    os.path.join(BASE_DIR, 'data/raw/^spx_d.csv'),
}


def load_data(symbol='djia', freq='weekly', start=None, end=None):
    """
    Load price data with computed sampling rate.

    Parameters
    ----------
    symbol : str
        'djia' or 'spx'
    freq : str
        'weekly' or 'daily'
    start, end : str or None
        Date range filter (e.g., '1921-04-29', '1965-05-21')

    Returns
    -------
    dict with:
        'close': ndarray of close prices
        'dates': DatetimeIndex
        'fs': float - samples per year (computed empirically)
        'n_samples': int
        'years': float - duration in years
        'label': str - human-readable description
    """
    key = f'{symbol}_{freq}'
    path = DATA_FILES[key]

    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    if start is not None:
        df = df[df.Date >= start]
    if end is not None:
        df = df[df.Date <= end]
    df = df.reset_index(drop=True)

    close = df['Close'].values.astype(np.float64)
    dates = df['Date']

    # Compute empirical fs
    duration_days = (dates.iloc[-1] - dates.iloc[0]).days
    duration_years = duration_days / 365.25
    fs = len(close) / duration_years

    label = f"{symbol.upper()} {freq} {dates.iloc[0].strftime('%Y')}-{dates.iloc[-1].strftime('%Y')}"

    return {
        'close': close,
        'dates': dates,
        'fs': fs,
        'n_samples': len(close),
        'years': duration_years,
        'label': label,
    }


# =============================================================================
# SPECTRAL ANALYSIS PIPELINE
# =============================================================================

def run_spectrum(close, fs, omega_max=22.0, peak_min_distance=2,
                 peak_prominence_frac=0.03):
    """
    Run Fourier-Lanczos spectrum and peak detection.

    Parameters
    ----------
    close : ndarray
        Price series
    fs : float
        Samples per year
    omega_max : float
        Max frequency to analyze (rad/yr). Set higher for daily data.
    peak_min_distance : int
        Minimum distance between peaks in spectral bins.
        Use 2 for fine (harmonic-level) detection, 6 for coarse (lobe-level).
    peak_prominence_frac : float
        Minimum prominence as fraction of amplitude range.

    Returns
    -------
    dict with:
        'omega': ndarray - frequencies in rad/yr
        'amp': ndarray - amplitudes
        'peak_freqs': ndarray - detected peak frequencies (rad/yr)
        'peak_amps': ndarray - amplitudes at peaks
        'trough_freqs': ndarray - detected trough frequencies
        'envelope_upper_k': float - upper envelope k parameter
        'envelope_upper_r2': float - upper envelope R²
        'envelope_lower_k': float
        'envelope_lower_r2': float
        'n_peaks': int
    """
    w, wRad, cosprt, sinprt, amp, phRad, phGrad = lanczos_spectrum(close, 1, fs)
    omega_yr = w * fs  # convert to rad/yr

    # Restrict to analysis range
    mask = (omega_yr > 0.3) & (omega_yr <= omega_max)
    omega_sub = omega_yr[mask]
    amp_sub = amp[mask]

    # Peak detection
    amp_range = amp_sub.max() - amp_sub.min()
    prominence = peak_prominence_frac * amp_range

    peak_indices, peak_freqs, peak_amps = find_spectral_peaks(
        amp_sub, omega_sub,
        min_distance=peak_min_distance,
        prominence=prominence,
        freq_range=(0.3, omega_max)
    )

    # Trough detection for envelope
    trough_indices, trough_freqs, trough_amps = find_spectral_troughs(
        amp_sub, omega_sub,
        min_distance=peak_min_distance,
        prominence=prominence * 0.5,
        freq_range=(0.3, omega_max)
    )

    # Envelope fitting
    try:
        upper_fit = fit_power_law_envelope(peak_freqs, peak_amps)
        upper_k = upper_fit['k']
        upper_r2 = upper_fit['r_squared']
    except Exception:
        upper_k, upper_r2 = np.nan, np.nan

    try:
        lower_fit = fit_power_law_envelope(trough_freqs, trough_amps)
        lower_k = lower_fit['k']
        lower_r2 = lower_fit['r_squared']
    except Exception:
        lower_k, lower_r2 = np.nan, np.nan

    return {
        'omega': omega_yr,
        'amp': amp,
        'omega_sub': omega_sub,
        'amp_sub': amp_sub,
        'peak_freqs': peak_freqs,
        'peak_amps': peak_amps,
        'trough_freqs': trough_freqs,
        'trough_amps': trough_amps,
        'envelope_upper_k': upper_k,
        'envelope_upper_r2': upper_r2,
        'envelope_lower_k': lower_k,
        'envelope_lower_r2': lower_r2,
        'n_peaks': len(peak_freqs),
    }


# =============================================================================
# LINE SPACING ANALYSIS
# =============================================================================

HURST_SPACING = 0.3676  # rad/yr


def map_to_harmonic(freq, spacing=HURST_SPACING):
    """Map a frequency to nearest harmonic number N."""
    N = max(1, round(freq / spacing))
    omega_N = N * spacing
    error = freq - omega_N
    return int(N), omega_N, error


def measure_line_spacing(peak_freqs, omega_max=12.5):
    """
    Compute line spacing from detected peaks using two methods:

    1. raw_spacing: np.diff of sorted peaks (naive, overestimates if gaps exist)
    2. harmonic_spacing: linear regression of omega vs N (robust to missing harmonics)

    The harmonic method maps each peak to N = round(freq/0.3676), then fits
    omega = spacing * N via least-squares. This gives the best-fit fundamental
    spacing even when many harmonics are missing.

    Only uses peaks below omega_max to match Hurst's analysis range.
    """
    valid = peak_freqs[peak_freqs <= omega_max]
    if len(valid) < 3:
        return {'mean_spacing': np.nan, 'std_spacing': np.nan,
                'harmonic_spacing': np.nan, 'harmonic_spacing_std': np.nan,
                'n_lines': len(valid), 'spacings': np.array([])}

    spacings = np.diff(np.sort(valid))

    # Harmonic-aware spacing: map peaks to N, fit omega = spacing * N
    N_vals = np.array([max(1, round(f / HURST_SPACING)) for f in valid])
    # Remove duplicates (keep the one closest to N*HURST_SPACING)
    unique_N = np.unique(N_vals)
    best_freqs = []
    best_Ns = []
    for n in unique_N:
        mask = N_vals == n
        candidates = valid[mask]
        errors = np.abs(candidates - n * HURST_SPACING)
        best_freqs.append(candidates[np.argmin(errors)])
        best_Ns.append(n)
    best_Ns = np.array(best_Ns, dtype=float)
    best_freqs = np.array(best_freqs)

    # Least-squares fit: omega = spacing * N (no intercept)
    if len(best_Ns) >= 3:
        harmonic_spacing = float(np.sum(best_Ns * best_freqs) / np.sum(best_Ns ** 2))
        residuals = best_freqs - harmonic_spacing * best_Ns
        harmonic_std = float(np.std(residuals))
    else:
        harmonic_spacing = np.nan
        harmonic_std = np.nan

    return {
        'mean_spacing': float(np.mean(spacings)),
        'median_spacing': float(np.median(spacings)),
        'std_spacing': float(np.std(spacings)),
        'harmonic_spacing': harmonic_spacing,
        'harmonic_spacing_std': harmonic_std,
        'n_lines': len(valid),
        'spacings': spacings,
    }


def compute_harmonic_fit(peak_freqs, spacing=HURST_SPACING, max_N=34):
    """
    Map peaks to harmonic numbers and compute linear fit quality.

    Returns
    -------
    dict with:
        'N_values': list of assigned harmonic numbers
        'omega_values': list of actual frequencies
        'omega_predicted': list of N * spacing
        'residuals': list of (actual - predicted)
        'rms_error': float
        'r_squared': float - how well ω_n = spacing * N fits
        'coverage': float - fraction of N=1..max_N that have a match
    """
    N_values = []
    omega_values = []
    assigned = set()

    for f in np.sort(peak_freqs):
        N, omega_N, err = map_to_harmonic(f, spacing)
        if N <= max_N and N not in assigned and abs(err) < spacing * 0.4:
            N_values.append(N)
            omega_values.append(f)
            assigned.add(N)

    N_arr = np.array(N_values)
    omega_arr = np.array(omega_values)
    omega_pred = N_arr * spacing
    residuals = omega_arr - omega_pred

    if len(N_arr) < 2:
        return {'N_values': N_values, 'omega_values': omega_values,
                'omega_predicted': omega_pred.tolist(), 'residuals': [],
                'rms_error': np.nan, 'r_squared': np.nan,
                'coverage': len(assigned) / max_N}

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((omega_arr - np.mean(omega_arr)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        'N_values': N_values,
        'omega_values': omega_values,
        'omega_predicted': omega_pred.tolist(),
        'residuals': residuals.tolist(),
        'rms_error': float(np.sqrt(np.mean(residuals ** 2))),
        'r_squared': float(r2),
        'coverage': len(assigned) / max_N,
    }


# =============================================================================
# HURST SIMILARITY SCORE
# =============================================================================

def hurst_similarity_score(spectrum_result, spacing_result, harmonic_result,
                           reference_spacing=HURST_SPACING):
    """
    Compute 0-1 score measuring how 'Hurst-like' a spectrum is.

    Components (weighted):
      40% - Line spacing match (|spacing - 0.3676| < 0.05 → 1.0)
      20% - Envelope shape (R² of upper 1/ω fit)
      20% - Harmonic coverage (fraction of N=1..34 detected)
      20% - Harmonic fit quality (R² of ω_n = 0.3676*N)

    Returns
    -------
    dict with:
        'total_score': float (0-1)
        'spacing_score': float
        'envelope_score': float
        'coverage_score': float
        'fit_score': float
    """
    # Spacing score: use harmonic-aware spacing (robust to missing harmonics)
    # Falls back to mean_spacing if harmonic_spacing unavailable
    harm_sp = spacing_result.get('harmonic_spacing', np.nan)
    mean_sp = spacing_result.get('mean_spacing', np.nan)
    sp = harm_sp if not np.isnan(harm_sp) else mean_sp
    if np.isnan(sp):
        spacing_score = 0.0
    else:
        delta = abs(sp - reference_spacing)
        spacing_score = max(0.0, 1.0 - delta / 0.15)

    # Envelope score: R² of upper envelope
    upper_r2 = spectrum_result.get('envelope_upper_r2', 0.0)
    envelope_score = max(0.0, upper_r2) if not np.isnan(upper_r2) else 0.0

    # Coverage score: fraction of harmonics found
    coverage = harmonic_result.get('coverage', 0.0)
    coverage_score = coverage

    # Fit score: R² of harmonic fit
    fit_r2 = harmonic_result.get('r_squared', 0.0)
    fit_score = max(0.0, fit_r2) if not np.isnan(fit_r2) else 0.0

    total = 0.40 * spacing_score + 0.20 * envelope_score + \
            0.20 * coverage_score + 0.20 * fit_score

    return {
        'total_score': float(total),
        'spacing_score': float(spacing_score),
        'envelope_score': float(envelope_score),
        'coverage_score': float(coverage_score),
        'fit_score': float(fit_score),
    }
