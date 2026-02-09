# -*- coding: utf-8 -*-
"""
Frequency measurement functions for comb filter outputs.

Implements Hurst's discrete measurement method: measure period between
successive peaks, troughs, or zero crossings of filtered signals to
estimate instantaneous frequency.

Reference: J.M. Hurst, The Profit Magic of Stock Transaction Timing,
           Appendix A, Figures AI-4 through AI-6
"""

import numpy as np
from scipy.signal import find_peaks


def measure_freq_at_peaks(signal_real, phase_unwrapped=None, fs=52):
    """
    Measure frequency at peaks of the real filtered signal.

    Computes period as time between successive peaks. If unwrapped phase
    is provided, also computes average phase derivative between peaks
    (more accurate).

    Parameters
    ----------
    signal_real : array
        Real part of filtered signal
    phase_unwrapped : array, optional
        Unwrapped phase from analytic signal
    fs : float
        Sampling rate (samples/year)

    Returns
    -------
    dict with:
        times : array - sample indices at midpoints between peaks
        freqs_period : array - frequency from peak-to-peak period (rad/yr)
        freqs_phase : array or None - frequency from phase derivative (rad/yr)
        peak_indices : array - indices of detected peaks
    """
    peaks, _ = find_peaks(signal_real)
    if len(peaks) < 2:
        return {'times': np.array([]), 'freqs_period': np.array([]),
                'freqs_phase': None, 'peak_indices': peaks}

    # Period-based: time between successive peaks
    dt_samples = np.diff(peaks)
    dt_years = dt_samples / fs
    freqs_period = 2 * np.pi / dt_years  # rad/year
    times = (peaks[:-1] + peaks[1:]) / 2.0  # midpoint between peaks

    # Phase-based: average dphi/dt between successive peaks
    freqs_phase = None
    if phase_unwrapped is not None:
        dphi = np.diff(phase_unwrapped[peaks])
        freqs_phase = dphi / dt_years  # rad/year (phase is already in radians)

    return {'times': times, 'freqs_period': freqs_period,
            'freqs_phase': freqs_phase, 'peak_indices': peaks}


def measure_freq_at_troughs(signal_real, phase_unwrapped=None, fs=52):
    """
    Measure frequency at troughs of the real filtered signal.
    Same method as peaks but on inverted signal.

    Parameters
    ----------
    signal_real : array
        Real part of filtered signal
    phase_unwrapped : array, optional
        Unwrapped phase from analytic signal
    fs : float
        Sampling rate (samples/year)

    Returns
    -------
    dict with:
        times : array - sample indices at midpoints between troughs
        freqs_period : array - frequency from trough-to-trough period (rad/yr)
        freqs_phase : array or None - frequency from phase derivative (rad/yr)
        trough_indices : array - indices of detected troughs
    """
    troughs, _ = find_peaks(-signal_real)
    if len(troughs) < 2:
        return {'times': np.array([]), 'freqs_period': np.array([]),
                'freqs_phase': None, 'trough_indices': troughs}

    dt_samples = np.diff(troughs)
    dt_years = dt_samples / fs
    freqs_period = 2 * np.pi / dt_years
    times = (troughs[:-1] + troughs[1:]) / 2.0

    freqs_phase = None
    if phase_unwrapped is not None:
        dphi = np.diff(phase_unwrapped[troughs])
        freqs_phase = dphi / dt_years

    return {'times': times, 'freqs_period': freqs_period,
            'freqs_phase': freqs_phase, 'trough_indices': troughs}


def measure_freq_at_zero_crossings(signal_real, fs=52):
    """
    Measure frequency at zero crossings of the real filtered signal.

    Two successive zero crossings define a half-period.
    Frequency = 2pi / (2 * half_period).

    Parameters
    ----------
    signal_real : array
        Real part of filtered signal
    fs : float
        Sampling rate (samples/year)

    Returns
    -------
    dict with:
        times : array - sample indices at midpoints between crossings
        freqs : array - frequency in rad/year
        crossing_indices : array - indices of zero crossings
    """
    # Detect sign changes
    signs = np.sign(signal_real)
    sign_changes = np.diff(signs)
    crossing_indices = np.where(sign_changes != 0)[0]

    if len(crossing_indices) < 2:
        return {'times': np.array([]), 'freqs': np.array([]),
                'crossing_indices': crossing_indices}

    # Each pair of successive crossings = half period
    dt_samples = np.diff(crossing_indices)
    dt_years = dt_samples / fs
    half_period_years = dt_years
    freqs = np.pi / half_period_years  # 2pi / (2 * half_period) = pi / half_period
    times = (crossing_indices[:-1] + crossing_indices[1:]) / 2.0

    return {'times': times, 'freqs': freqs,
            'crossing_indices': crossing_indices}
