# -*- coding: utf-8 -*-
"""
Peak and Trough Detection for Spectral Analysis

This module provides functions for detecting peaks and troughs in Fourier-Lanczos
spectra, supporting the reproduction of Hurst's Appendix A analysis.

Reference: J.M. Hurst, The Profit Magic of Stock Transaction Timing, Appendix A
"""

import numpy as np
from scipy.signal import find_peaks


def find_spectral_peaks(amplitude, frequencies, min_distance=3, prominence=None,
                        freq_range=(0.5, 20.0)):
    """
    Detect local maxima (peaks) in a spectral amplitude array.

    Parameters
    ----------
    amplitude : array_like
        Array of spectral amplitudes (from lanczos_spectrum)
    frequencies : array_like
        Array of frequencies in radians per year (omega_yr)
    min_distance : int, optional
        Minimum number of samples separating peaks (default: 3)
        Helps avoid detecting noise as multiple peaks
    prominence : float, optional
        Minimum prominence of peaks (default: None, auto-computed)
        If None, uses 5% of amplitude range as threshold
    freq_range : tuple, optional
        (min_freq, max_freq) range in rad/year to consider (default: 0.5 to 20.0)
        Filters out DC component and very high frequencies

    Returns
    -------
    peak_indices : ndarray
        Indices of detected peaks in the input arrays
    peak_frequencies : ndarray
        Frequencies (rad/year) at peak locations
    peak_amplitudes : ndarray
        Amplitude values at peak locations

    Notes
    -----
    - Handles wRad[0] = inf by skipping the DC component
    - Uses scipy.signal.find_peaks() as the detection engine
    - Default min_distance=3 works well for Lanczos spectra with fine structure

    Example
    -------
    >>> w, wRad, _, _, amp, _, _ = lanczos_spectrum(data, 1, 52)
    >>> omega_yr = w * 52
    >>> peak_idx, peak_freq, peak_amp = find_spectral_peaks(amp, omega_yr)
    """
    amplitude = np.asarray(amplitude)
    frequencies = np.asarray(frequencies)

    # Auto-compute prominence if not provided
    if prominence is None:
        # Use 5% of the amplitude range as minimum prominence
        # This filters out small noise fluctuations
        prominence = 0.05 * (np.max(amplitude) - np.min(amplitude))

    # Detect peaks using scipy
    peak_indices, properties = find_peaks(
        amplitude,
        distance=min_distance,
        prominence=prominence
    )

    # Extract peak frequencies and amplitudes
    peak_frequencies = frequencies[peak_indices]
    peak_amplitudes = amplitude[peak_indices]

    # Filter by frequency range
    if freq_range is not None:
        min_freq, max_freq = freq_range
        valid_mask = (peak_frequencies >= min_freq) & (peak_frequencies <= max_freq)
        peak_indices = peak_indices[valid_mask]
        peak_frequencies = peak_frequencies[valid_mask]
        peak_amplitudes = peak_amplitudes[valid_mask]

    return peak_indices, peak_frequencies, peak_amplitudes


def find_spectral_troughs(amplitude, frequencies, min_distance=3, prominence=None,
                          freq_range=(0.5, 20.0)):
    """
    Detect local minima (troughs) in a spectral amplitude array.

    This function inverts the amplitude and finds peaks in the inverted signal,
    which correspond to troughs in the original signal.

    Parameters
    ----------
    amplitude : array_like
        Array of spectral amplitudes (from lanczos_spectrum)
    frequencies : array_like
        Array of frequencies in radians per year (omega_yr)
    min_distance : int, optional
        Minimum number of samples separating troughs (default: 3)
    prominence : float, optional
        Minimum prominence of troughs (default: None, auto-computed)
        If None, uses 5% of amplitude range as threshold
    freq_range : tuple, optional
        (min_freq, max_freq) range in rad/year to consider (default: 0.5 to 20.0)

    Returns
    -------
    trough_indices : ndarray
        Indices of detected troughs in the input arrays
    trough_frequencies : ndarray
        Frequencies (rad/year) at trough locations
    trough_amplitudes : ndarray
        Amplitude values at trough locations

    Notes
    -----
    - Troughs are detected by inverting the signal and finding peaks
    - Important for identifying spectral gaps and "meaningless frequencies"
    - Used for fitting lower envelope in Hurst's analysis

    Example
    -------
    >>> w, wRad, _, _, amp, _, _ = lanczos_spectrum(data, 1, 52)
    >>> omega_yr = w * 52
    >>> trough_idx, trough_freq, trough_amp = find_spectral_troughs(amp, omega_yr)
    """
    amplitude = np.asarray(amplitude)
    frequencies = np.asarray(frequencies)

    # Invert the amplitude to find troughs as peaks in inverted signal
    inverted_amplitude = -amplitude

    # Auto-compute prominence if not provided
    if prominence is None:
        prominence = 0.05 * (np.max(amplitude) - np.min(amplitude))

    # Detect peaks in inverted signal (= troughs in original)
    trough_indices, properties = find_peaks(
        inverted_amplitude,
        distance=min_distance,
        prominence=prominence
    )

    # Extract trough frequencies and amplitudes (from original signal)
    trough_frequencies = frequencies[trough_indices]
    trough_amplitudes = amplitude[trough_indices]  # Use original, not inverted

    # Filter by frequency range
    if freq_range is not None:
        min_freq, max_freq = freq_range
        valid_mask = (trough_frequencies >= min_freq) & (trough_frequencies <= max_freq)
        trough_indices = trough_indices[valid_mask]
        trough_frequencies = trough_frequencies[valid_mask]
        trough_amplitudes = trough_amplitudes[valid_mask]

    return trough_indices, trough_frequencies, trough_amplitudes


def filter_peaks_by_frequency_range(peak_indices, peak_frequencies, peak_amplitudes,
                                    freq_range=(0.5, 20.0)):
    """
    Filter detected peaks to a specific frequency range.

    This is useful for focusing on the Hurst-relevant frequency range
    (typically 0.5 to 20 rad/year) and avoiding DC component or noise.

    Parameters
    ----------
    peak_indices : array_like
        Indices of peaks
    peak_frequencies : array_like
        Frequencies at peak locations (rad/year)
    peak_amplitudes : array_like
        Amplitudes at peak locations
    freq_range : tuple
        (min_freq, max_freq) in rad/year

    Returns
    -------
    filtered_indices : ndarray
        Peak indices within the frequency range
    filtered_frequencies : ndarray
        Peak frequencies within the range
    filtered_amplitudes : ndarray
        Peak amplitudes within the range

    Example
    -------
    >>> # Focus on low-frequency structure only
    >>> low_freq_peaks = filter_peaks_by_frequency_range(
    ...     peak_idx, peak_freq, peak_amp, freq_range=(0.5, 5.0)
    ... )
    """
    peak_indices = np.asarray(peak_indices)
    peak_frequencies = np.asarray(peak_frequencies)
    peak_amplitudes = np.asarray(peak_amplitudes)

    min_freq, max_freq = freq_range
    valid_mask = (peak_frequencies >= min_freq) & (peak_frequencies <= max_freq)

    return (
        peak_indices[valid_mask],
        peak_frequencies[valid_mask],
        peak_amplitudes[valid_mask]
    )


def detect_fine_structure_spacing(peak_frequencies, max_spacing=1.0):
    """
    Detect regular spacing in peak frequencies (fine structure).

    Hurst identified fine frequency structure with spacing ~0.3676 rad/year.
    This function computes spacings between consecutive peaks and identifies
    clusters of similar spacing.

    Parameters
    ----------
    peak_frequencies : array_like
        Array of peak frequencies (rad/year), should be sorted
    max_spacing : float, optional
        Maximum spacing to consider as "fine structure" (default: 1.0 rad/year)

    Returns
    -------
    spacings : ndarray
        Array of spacings between consecutive peaks
    mean_spacing : float
        Mean spacing of peaks within max_spacing threshold
    std_spacing : float
        Standard deviation of spacing

    Notes
    -----
    - Hurst mentions fine structure spacing of 0.3676 rad/year (Appendix A)
    - This corresponds to T ≈ 17.1 years per the relation T = 2π/ω
    - Used to validate the Nominal Model derivation

    Example
    -------
    >>> spacings, mean_sp, std_sp = detect_fine_structure_spacing(peak_freq)
    >>> print(f"Mean spacing: {mean_sp:.4f} rad/year")
    """
    peak_frequencies = np.asarray(peak_frequencies)

    # Ensure peaks are sorted by frequency
    peak_frequencies = np.sort(peak_frequencies)

    # Compute spacing between consecutive peaks
    spacings = np.diff(peak_frequencies)

    # Filter to fine structure (small spacings)
    fine_spacings = spacings[spacings <= max_spacing]

    if len(fine_spacings) > 0:
        mean_spacing = np.mean(fine_spacings)
        std_spacing = np.std(fine_spacings)
    else:
        mean_spacing = np.nan
        std_spacing = np.nan

    return spacings, mean_spacing, std_spacing
