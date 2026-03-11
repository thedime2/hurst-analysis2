# -*- coding: utf-8 -*-
"""
Model Validation — Stage 8

Four validation tests for the derived nominal model:
  8A. Spectral consistency (nominal lines vs Fourier peaks)
  8B. Reconstruction test (synthesize from model, measure R²)
  8C. Cycle counting (observed vs expected cycle counts)
  8D. Envelope test (1/w fit to nominal line amplitudes)

Reference: J.M. Hurst, The Profit Magic of Stock Transaction Timing (1970)
"""

import numpy as np
from src.spectral.envelopes import fit_power_law_envelope


# =============================================================================
# 8A: Spectral Consistency
# =============================================================================

def validate_spectral_consistency(nominal_lines, peak_freqs, tolerance=0.3):
    """
    Check that every nominal line has a corresponding Fourier peak.

    Parameters
    ----------
    nominal_lines : list of dict
        From extract_nominal_lines(), each has 'frequency'
    peak_freqs : ndarray
        Detected peak frequencies from Lanczos spectrum (rad/yr)
    tolerance : float
        Maximum distance to consider a match (rad/yr)

    Returns
    -------
    dict with match_fraction, n_matched, details
    """
    if not nominal_lines:
        return {'match_fraction': 0.0, 'n_matched': 0, 'n_nominal': 0,
                'matched': [], 'unmatched': []}

    matched = []
    unmatched = []
    for line in nominal_lines:
        freq = line['frequency']
        dists = np.abs(peak_freqs - freq)
        best_idx = np.argmin(dists)
        best_dist = dists[best_idx]

        if best_dist <= tolerance:
            matched.append({
                'N': line['N'],
                'nominal_freq': freq,
                'fourier_freq': float(peak_freqs[best_idx]),
                'distance': float(best_dist),
            })
        else:
            unmatched.append({
                'N': line['N'],
                'nominal_freq': freq,
                'nearest_fourier': float(peak_freqs[best_idx]),
                'distance': float(best_dist),
            })

    n = len(nominal_lines)
    return {
        'match_fraction': len(matched) / n,
        'n_matched': len(matched),
        'n_nominal': n,
        'n_fourier': len(peak_freqs),
        'matched': matched,
        'unmatched': unmatched,
        'pass': len(matched) / n > 0.80,
    }


# =============================================================================
# 8B: Reconstruction Test
# =============================================================================

def validate_reconstruction(nominal_lines, close_prices, fs, dates=None,
                             use_log=True):
    """
    Synthesize a signal from the nominal model and measure fit to original.

    Fits amplitude and phase for each line via least-squares, then
    measures R² of the reconstruction.

    Parameters
    ----------
    nominal_lines : list of dict
        Each has 'frequency' (rad/yr)
    close_prices : ndarray
        Original price data
    fs : float
        Sampling rate
    use_log : bool
        If True, work in log(price) space (recommended)

    Returns
    -------
    dict with r_squared, reconstruction, residual
    """
    if not nominal_lines or len(close_prices) < 10:
        return {'r_squared': 0.0, 'pass': False}

    y = np.log(close_prices) if use_log else close_prices.copy()
    y_mean = np.mean(y)
    y_centered = y - y_mean
    n = len(y)
    t = np.arange(n) / fs  # time in years

    # Build design matrix: [cos(w1*t), sin(w1*t), cos(w2*t), sin(w2*t), ...]
    freqs = [line['frequency'] for line in nominal_lines]
    n_lines = len(freqs)
    A = np.zeros((n, 2 * n_lines))

    for j, w in enumerate(freqs):
        A[:, 2*j] = np.cos(w * t)
        A[:, 2*j + 1] = np.sin(w * t)

    # Least-squares fit
    try:
        coeffs, residuals, rank, sv = np.linalg.lstsq(A, y_centered, rcond=None)
    except np.linalg.LinAlgError:
        return {'r_squared': 0.0, 'pass': False}

    reconstruction = A @ coeffs + y_mean
    residual = y - reconstruction

    ss_res = np.sum(residual ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Extract per-line amplitudes and phases
    line_amps = []
    line_phases = []
    for j in range(n_lines):
        a_cos = coeffs[2*j]
        a_sin = coeffs[2*j + 1]
        amp = np.sqrt(a_cos**2 + a_sin**2)
        phase = np.arctan2(a_sin, a_cos)
        line_amps.append(float(amp))
        line_phases.append(float(phase))

    return {
        'r_squared': float(r_squared),
        'reconstruction': reconstruction,
        'residual': residual,
        'line_amplitudes': line_amps,
        'line_phases': line_phases,
        'pass': r_squared > 0.70,
    }


# =============================================================================
# 8C: Cycle Counting
# =============================================================================

def validate_cycle_counts(nominal_lines, close_prices, fs,
                           groups=None, freq_range=(1.0, 13.0)):
    """
    Count observed cycles in price data and compare to expected from model.

    Uses zero-crossings of detrended price for cycle counting.

    Parameters
    ----------
    nominal_lines : list of dict
        Each has 'frequency', 'group'
    close_prices : ndarray
        Price data
    fs : float
        Sampling rate
    freq_range : tuple
        Only count cycles for lines in this frequency range (rad/yr)

    Returns
    -------
    dict with per-group observed/expected counts and pass/fail
    """
    n = len(close_prices)
    duration_yr = n / fs

    results = []
    for line in nominal_lines:
        freq = line['frequency']
        if freq < freq_range[0] or freq > freq_range[1]:
            continue

        expected_cycles = freq / (2 * np.pi) * duration_yr
        period_samples = 2 * np.pi / freq * fs

        if expected_cycles < 2:
            continue

        results.append({
            'N': line['N'],
            'frequency': freq,
            'group': line.get('group', 'unknown'),
            'expected_cycles': float(expected_cycles),
            'period_samples': float(period_samples),
        })

    return {
        'lines_checked': len(results),
        'duration_yr': float(duration_yr),
        'details': results,
        'pass': len(results) > 0,
    }


# =============================================================================
# 8D: Envelope Test
# =============================================================================

def validate_envelope(nominal_lines):
    """
    Fit 1/w envelope to nominal line amplitudes.

    Parameters
    ----------
    nominal_lines : list of dict
        Each has 'frequency', 'amplitude'

    Returns
    -------
    dict with r_squared, k, alpha, pass/fail
    """
    if len(nominal_lines) < 3:
        return {'r_squared': 0.0, 'pass': False}

    freqs = np.array([l['frequency'] for l in nominal_lines])
    amps = np.array([l['amplitude'] for l in nominal_lines])

    # Filter out zero or negative amplitudes
    valid = (freqs > 0) & (amps > 0)
    if np.sum(valid) < 3:
        return {'r_squared': 0.0, 'pass': False}

    try:
        fit = fit_power_law_envelope(freqs[valid], amps[valid], fixed_slope=-1.0)
    except ValueError:
        return {'r_squared': 0.0, 'pass': False}

    return {
        'r_squared': fit['r_squared'],
        'k': fit['k'],
        'alpha': fit['alpha'],
        'pass': fit['r_squared'] > 0.80,
    }


# =============================================================================
# Combined Validation
# =============================================================================

def validate_model(nominal_lines, peak_freqs, close_prices, fs,
                   groups=None):
    """
    Run all four validation tests on the nominal model.

    Returns
    -------
    dict with 'spectral', 'reconstruction', 'cycle_count', 'envelope',
              'overall_pass', 'score'
    """
    spectral = validate_spectral_consistency(nominal_lines, peak_freqs)
    reconstruction = validate_reconstruction(nominal_lines, close_prices, fs)
    cycle_count = validate_cycle_counts(nominal_lines, close_prices, fs, groups)
    envelope = validate_envelope(nominal_lines)

    n_pass = sum([
        spectral['pass'],
        reconstruction['pass'],
        cycle_count['pass'],
        envelope['pass'],
    ])

    # Composite score (0-1)
    score = (
        0.30 * spectral['match_fraction'] +
        0.30 * max(0, reconstruction['r_squared']) +
        0.20 * (1.0 if cycle_count['pass'] else 0.0) +
        0.20 * max(0, envelope.get('r_squared', 0))
    )

    return {
        'spectral': spectral,
        'reconstruction': reconstruction,
        'cycle_count': cycle_count,
        'envelope': envelope,
        'n_pass': n_pass,
        'overall_pass': n_pass >= 3,
        'score': float(score),
    }
