# -*- coding: utf-8 -*-
"""
CMW Scalogram -- Dense Time-Frequency Representation

Computes a scalogram by sweeping a Complex Morlet Wavelet (CMW)
across a range of center frequencies, using the existing apply_cmw()
infrastructure. Each row of the output matrix is the envelope
(|analytic signal|) at one center frequency.

Design choices:
  - Constant-Q (FWHM = f0 / Q) gives good time-frequency tradeoff:
    narrow at low freq (good frequency resolution), wide at high freq
    (good time resolution). Q=5 is the default.
  - Log-spaced center frequencies match the log-frequency axis natural
    for cycle analysis (equal spacing in octaves).

All frequencies are in radians per year (rad/yr).

Reference: Cohen (2019), NeuroImage 199:81-86 (CMW FWHM design)
"""

import numpy as np
from .cmw import apply_cmw


def compute_scalogram(signal, freq_range, n_scales, fs,
                      fwhm_mode='constant_q', q_factor=5.0,
                      freq_spacing='log', analytic=True):
    """
    Compute a CMW scalogram (time-frequency envelope matrix).

    Calls apply_cmw() for each of n_scales center frequencies and
    stacks the envelope amplitudes into a 2D matrix.

    Parameters
    ----------
    signal : ndarray, shape (N,)
        Real-valued input signal (e.g., DJIA close prices).
    freq_range : tuple (f_lo, f_hi)
        Frequency range in rad/yr (e.g., (0.5, 40.0)).
    n_scales : int
        Number of center frequencies (e.g., 150).
    fs : float
        Sampling rate in samples/year (52 for weekly).
    fwhm_mode : str
        'constant_q' -- FWHM = f0 / Q (default, proportional to freq)
        'constant_bw' -- FWHM = q_factor (fixed bandwidth for all scales)
    q_factor : float
        For 'constant_q': this is Q (default 5.0, so FWHM = f0/5).
        For 'constant_bw': this is the fixed FWHM in rad/yr.
    freq_spacing : str
        'log' for logarithmic (default), 'linear' for linear spacing.
    analytic : bool
        If True (default), use analytic CMW for envelope extraction.

    Returns
    -------
    result : dict
        'matrix'        : ndarray (n_scales, N) -- envelope amplitudes
        'frequencies'   : ndarray (n_scales,) -- center frequencies (rad/yr)
        'periods_weeks' : ndarray (n_scales,) -- periods in weeks
        'fwhm_per_scale': ndarray (n_scales,) -- FWHM at each scale (rad/yr)
        'phase_matrix'  : ndarray (n_scales, N) -- unwrapped phase (if analytic)
        'fs'            : float
        'fwhm_mode'     : str
        'q_factor'      : float
    """
    signal = np.asarray(signal, dtype=np.float64)
    N = len(signal)
    f_lo, f_hi = freq_range

    # Generate center frequencies
    if freq_spacing == 'log':
        frequencies = np.geomspace(f_lo, f_hi, n_scales)
    else:
        frequencies = np.linspace(f_lo, f_hi, n_scales)

    # Compute FWHM per scale
    if fwhm_mode == 'constant_q':
        fwhm_per_scale = frequencies / q_factor
    elif fwhm_mode == 'constant_bw':
        fwhm_per_scale = np.full(n_scales, q_factor)
    else:
        raise ValueError(f"fwhm_mode must be 'constant_q' or 'constant_bw', "
                         f"got '{fwhm_mode}'")

    # Nyquist check
    nyq = np.pi * fs  # rad/yr
    if f_hi > nyq:
        raise ValueError(
            f"freq_range upper bound {f_hi:.1f} rad/yr exceeds "
            f"Nyquist {nyq:.1f} rad/yr (fs={fs})")

    # Build scalogram matrix
    matrix = np.zeros((n_scales, N))
    phase_matrix = np.zeros((n_scales, N)) if analytic else None

    for i, (f0, fwhm) in enumerate(zip(frequencies, fwhm_per_scale)):
        result = apply_cmw(signal, f0, fwhm, fs, analytic=analytic)
        matrix[i, :] = result['envelope'] if analytic else np.abs(result['signal'])
        if analytic and result['phase'] is not None:
            phase_matrix[i, :] = result['phase']

    # Periods in weeks
    periods_weeks = 2.0 * np.pi / frequencies * 52.0

    return {
        'matrix': matrix,
        'frequencies': frequencies,
        'periods_weeks': periods_weeks,
        'fwhm_per_scale': fwhm_per_scale,
        'phase_matrix': phase_matrix,
        'fs': fs,
        'fwhm_mode': fwhm_mode,
        'q_factor': q_factor,
    }
