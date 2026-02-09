# -*- coding: utf-8 -*-
"""
Nominal Model derivation from line spectrum analysis.

Extracts discrete line frequencies from smoothed frequency-vs-time data,
computes line spacings, and builds the period hierarchy table that is the
central result of Hurst's spectral analysis.

Reference: J.M. Hurst, The Profit Magic of Stock Transaction Timing,
           Appendix A, Figure AI-6
"""

import numpy as np


def identify_line_frequencies(line_fits, min_r_squared=0.0, min_points=5):
    """
    Extract the set of nominal line frequencies from fitted frequency traces.

    Parameters
    ----------
    line_fits : list of dict
        Each dict is the output of fit_frequency_line() for one filter
        or one smoothed trace. Must have keys: 'mean_freq', 'r_squared',
        'std_dev', 'n_points', 'drift_rate_annual'.
    min_r_squared : float
        Minimum R^2 to include a line (default 0.0, no filtering)
    min_points : int
        Minimum number of measurement points to include

    Returns
    -------
    dict with:
        frequencies : array - nominal line frequencies (rad/yr), sorted
        periods_weeks : array - corresponding periods in weeks
        periods_years : array - corresponding periods in years
        stabilities : array - std_dev of each line (rad/yr)
        drift_rates : array - annual drift rate for each line
        n_lines : int - number of identified lines
    """
    valid_fits = [
        f for f in line_fits
        if f['n_points'] >= min_points and f['r_squared'] >= min_r_squared
    ]

    if not valid_fits:
        return {
            'frequencies': np.array([]),
            'periods_weeks': np.array([]),
            'periods_years': np.array([]),
            'stabilities': np.array([]),
            'drift_rates': np.array([]),
            'n_lines': 0
        }

    freqs = np.array([f['mean_freq'] for f in valid_fits])
    stabilities = np.array([f['std_dev'] for f in valid_fits])
    drift_rates = np.array([f['drift_rate_annual'] for f in valid_fits])

    # Sort by frequency
    sort_idx = np.argsort(freqs)
    freqs = freqs[sort_idx]
    stabilities = stabilities[sort_idx]
    drift_rates = drift_rates[sort_idx]

    periods_weeks = 2 * np.pi / freqs * 52
    periods_years = 2 * np.pi / freqs

    return {
        'frequencies': freqs,
        'periods_weeks': periods_weeks,
        'periods_years': periods_years,
        'stabilities': stabilities,
        'drift_rates': drift_rates,
        'n_lines': len(freqs)
    }


def compute_line_spacings(line_frequencies):
    """
    Compute spacings between adjacent line frequencies.

    Parameters
    ----------
    line_frequencies : array
        Sorted nominal frequencies in rad/yr

    Returns
    -------
    dict with:
        spacings : array - delta_omega between adjacent lines
        mean_spacing : float - mean spacing
        std_spacing : float - standard deviation of spacings
        median_spacing : float - median spacing
        min_spacing : float - minimum spacing
        max_spacing : float - maximum spacing
        regularity : float - coefficient of variation (std/mean),
            lower means more regular spacing
    """
    if len(line_frequencies) < 2:
        return {
            'spacings': np.array([]),
            'mean_spacing': 0.0,
            'std_spacing': 0.0,
            'median_spacing': 0.0,
            'min_spacing': 0.0,
            'max_spacing': 0.0,
            'regularity': np.inf
        }

    spacings = np.diff(line_frequencies)

    return {
        'spacings': spacings,
        'mean_spacing': np.mean(spacings),
        'std_spacing': np.std(spacings),
        'median_spacing': np.median(spacings),
        'min_spacing': np.min(spacings),
        'max_spacing': np.max(spacings),
        'regularity': np.std(spacings) / np.mean(spacings) if np.mean(spacings) > 0 else np.inf
    }


def build_nominal_model(line_frequencies, fs=52):
    """
    Build the Nominal Model period hierarchy table.

    Parameters
    ----------
    line_frequencies : array
        Sorted nominal frequencies in rad/yr
    fs : float
        Sampling rate (samples/year)

    Returns
    -------
    list of dict, one per line:
        'line_number': int (1-indexed)
        'frequency': float (rad/yr)
        'period_weeks': float
        'period_years': float
        'period_months': float
        'spacing_from_prev': float or None (rad/yr)
    """
    model = []
    for i, freq in enumerate(line_frequencies):
        period_years = 2 * np.pi / freq
        period_weeks = period_years * fs
        period_months = period_years * 12

        spacing = float(freq - line_frequencies[i - 1]) if i > 0 else None

        model.append({
            'line_number': i + 1,
            'frequency': float(freq),
            'period_weeks': float(period_weeks),
            'period_years': float(period_years),
            'period_months': float(period_months),
            'spacing_from_prev': spacing
        })

    return model


def validate_against_fourier(nominal_frequencies, fourier_peak_frequencies,
                              tolerance=0.3):
    """
    Cross-validate nominal model frequencies against Fourier spectrum peaks.

    Parameters
    ----------
    nominal_frequencies : array
        Line frequencies from comb filter analysis (rad/yr)
    fourier_peak_frequencies : array
        Peak frequencies from Phase 1 Fourier-Lanczos spectrum (rad/yr)
    tolerance : float
        Maximum distance (rad/yr) to consider a match

    Returns
    -------
    dict with:
        n_nominal : int - number of nominal lines
        n_fourier : int - number of Fourier peaks
        n_matched : int - number of matches found
        match_fraction : float - fraction of nominal lines with a
            Fourier match
        matches : list of dict - matched pairs with distances
        unmatched_nominal : array - nominal freqs with no Fourier match
        unmatched_fourier : array - Fourier peaks with no nominal match
    """
    matches = []
    matched_nominal = set()
    matched_fourier = set()

    for i, nf in enumerate(nominal_frequencies):
        distances = np.abs(fourier_peak_frequencies - nf)
        best_idx = np.argmin(distances)
        best_dist = distances[best_idx]

        if best_dist <= tolerance:
            matches.append({
                'nominal_freq': float(nf),
                'fourier_freq': float(fourier_peak_frequencies[best_idx]),
                'distance': float(best_dist),
                'nominal_idx': i,
                'fourier_idx': int(best_idx)
            })
            matched_nominal.add(i)
            matched_fourier.add(int(best_idx))

    unmatched_nominal = nominal_frequencies[
        [i for i in range(len(nominal_frequencies)) if i not in matched_nominal]
    ]
    unmatched_fourier = fourier_peak_frequencies[
        [i for i in range(len(fourier_peak_frequencies)) if i not in matched_fourier]
    ]

    n_nominal = len(nominal_frequencies)
    return {
        'n_nominal': n_nominal,
        'n_fourier': len(fourier_peak_frequencies),
        'n_matched': len(matches),
        'match_fraction': len(matches) / n_nominal if n_nominal > 0 else 0.0,
        'matches': matches,
        'unmatched_nominal': unmatched_nominal,
        'unmatched_fourier': unmatched_fourier
    }
