# -*- coding: utf-8 -*-
"""
Complex Morlet Wavelet (CMW) — Frequency-Domain FWHM Design

Implements CMW filters designed in the frequency domain using the FWHM
(Full Width at Half Maximum) parameterization from:

    Cohen (2019), "A better way to define and describe Morlet wavelets
    for time-frequency analysis", NeuroImage 199:81-86

Key design principle: The CMW is a Gaussian in the frequency domain,
characterized by center frequency f0 and spectral FWHM. The FWHM
targets are matched to Ormsby filter skirt midpoints so the two filter
types have comparable spectral selectivity.

All frequencies are in radians per year (rad/yr) unless noted otherwise.
"""

import numpy as np


# ============================================================================
# Constants
# ============================================================================

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # ≈ 0.4247


# ============================================================================
# Ormsby → CMW parameter conversion
# ============================================================================

def ormsby_spec_to_cmw_params(spec):
    """
    Convert an Ormsby filter specification to matched CMW design parameters.

    For a bandpass Ormsby with edges [w1, w2, w3, w4]:
        f0   = (w2 + w3) / 2          — center frequency
        FWHM = (w3+w4)/2 - (w1+w2)/2  — half-gain width matches skirt midpoints

    For a lowpass Ormsby with [f_pass, f_stop]:
        f0   = 0                       — centered at DC
        FWHM = f_pass + f_stop         — symmetric about DC, 0.5 gain at skirt midpoint

    Parameters
    ----------
    spec : dict
        Ormsby filter specification with keys:
        - 'type': 'lp' or 'bp'
        - For 'bp': 'f1', 'f2', 'f3', 'f4' (rad/year)
        - For 'lp': 'f_pass', 'f_stop' (rad/year)

    Returns
    -------
    params : dict
        {'f0': float, 'fwhm': float, 'sigma_f': float, 'label': str}
        All frequencies in rad/year.
    """
    if spec['type'] == 'lp':
        f_pass = spec['f_pass']
        f_stop = spec['f_stop']
        f0 = 0.0
        fwhm = f_pass + f_stop  # 2 × skirt midpoint
        label = spec.get('label', 'CMW-LP')
    else:
        w1, w2, w3, w4 = spec['f1'], spec['f2'], spec['f3'], spec['f4']
        f0 = (w2 + w3) / 2.0
        lower_half = (w1 + w2) / 2.0
        upper_half = (w3 + w4) / 2.0
        fwhm = upper_half - lower_half
        label = spec.get('label', f'CMW fc={f0:.2f}')

    sigma_f = fwhm * FWHM_TO_SIGMA

    return {
        'f0': f0,
        'fwhm': fwhm,
        'sigma_f': sigma_f,
        'label': label,
        'ormsby_spec': spec,
    }


# ============================================================================
# Frequency-domain Gaussian construction
# ============================================================================

def cmw_freq_domain(f0, fwhm, fs, nfft, analytic=True):
    """
    Create a CMW filter as a Gaussian in the frequency domain.

    The Gaussian peaks at 1.0 (unity gain) at the center frequency f0.
    For analytic mode, only positive frequencies are populated and scaled
    by 2 to preserve amplitude (standard analytic signal convention).

    Parameters
    ----------
    f0 : float
        Center frequency in rad/year.
    fwhm : float
        Full width at half maximum in rad/year.
    fs : float
        Sampling rate (samples per year, e.g., 52 for weekly).
    nfft : int
        FFT length.
    analytic : bool
        If True, one-sided (positive freq only) → complex output.
        If False, two-sided (±f0) → real output.

    Returns
    -------
    result : dict
        'H': ndarray shape (nfft,) — frequency-domain filter
        'freqs_rad': ndarray — frequency axis in rad/year
        'f0', 'fwhm', 'sigma_f': design parameters
    """
    sigma_f = fwhm * FWHM_TO_SIGMA

    # Full frequency axis in rad/year
    # fftfreq returns cycles/sample; multiply by fs gives cycles/year; ×2π → rad/year
    freqs_rad = np.fft.fftfreq(nfft, d=1.0 / fs) * (2.0 * np.pi)

    if analytic:
        # One-sided Gaussian: only positive frequencies
        H = np.zeros(nfft, dtype=np.float64)
        pos_mask = freqs_rad >= 0
        H[pos_mask] = np.exp(-0.5 * ((freqs_rad[pos_mask] - f0) / sigma_f) ** 2)
        # Scale ×2 for single-sided (analytic signal convention), DC not doubled
        H[pos_mask] *= 2.0
        H[0] /= 2.0  # DC bin
    else:
        # Two-sided Gaussian: symmetric at ±f0
        G_pos = np.exp(-0.5 * ((freqs_rad - f0) / sigma_f) ** 2)
        G_neg = np.exp(-0.5 * ((freqs_rad + f0) / sigma_f) ** 2)
        H = G_pos + G_neg
        # For f0=0 (lowpass), G_pos and G_neg overlap perfectly → divide by 2
        if f0 == 0:
            H /= 2.0

    return {
        'H': H,
        'freqs_rad': freqs_rad,
        'f0': f0,
        'fwhm': fwhm,
        'sigma_f': sigma_f,
    }


# ============================================================================
# Apply CMW to signal
# ============================================================================

def apply_cmw(signal, f0, fwhm, fs, analytic=True):
    """
    Apply a single CMW to a signal via FFT multiplication.

    Parameters
    ----------
    signal : ndarray, shape (L,)
        Real-valued input signal.
    f0 : float
        Center frequency in rad/year.
    fwhm : float
        Spectral FWHM in rad/year.
    fs : float
        Sampling rate (samples per year).
    analytic : bool
        If True, returns complex analytic signal with envelope/phase.

    Returns
    -------
    result : dict
        Same structure as apply_ormsby_filter():
        'signal': ndarray — filtered output (complex if analytic, real if not)
        'envelope': ndarray or None — |z(t)| if analytic
        'phase': ndarray or None — unwrapped phase if analytic
        'phasew': ndarray or None — wrapped phase if analytic
        'frequency': ndarray or None — instantaneous freq in cycles/year
    """
    signal = np.asarray(signal, dtype=np.float64)
    L = len(signal)

    # Choose nfft as next power of 2 for efficiency
    nfft = int(2 ** np.ceil(np.log2(L)))

    # Build frequency-domain filter
    cmw_result = cmw_freq_domain(f0, fwhm, fs, nfft, analytic=analytic)
    H = cmw_result['H']

    # Apply via FFT multiplication
    signal_fft = np.fft.fft(signal, n=nfft)
    filtered_fft = H * signal_fft
    filtered_full = np.fft.ifft(filtered_fft)

    # Trim to original signal length
    y = filtered_full[:L]

    # Build output dict matching apply_ormsby_filter() structure
    out = {}

    if analytic:
        # Complex output → extract envelope, phase, frequency
        out['signal'] = y
        out['envelope'] = np.abs(y)
        phi = np.angle(y)
        phi_unwrapped = np.unwrap(phi)
        out['phase'] = phi_unwrapped
        out['phasew'] = phi
        # Instantaneous frequency: dφ/dt / (2π), in cycles/year
        f_inst = np.gradient(phi_unwrapped, 1.0 / fs) / (2.0 * np.pi)
        out['frequency'] = f_inst
    else:
        # Real output
        out['signal'] = y.real
        out['envelope'] = None
        out['phase'] = None
        out['phasew'] = None
        out['frequency'] = None

    return out


# ============================================================================
# Apply bank of CMWs
# ============================================================================

def apply_cmw_bank(signal, cmw_params_list, fs, analytic=True):
    """
    Apply a bank of CMWs to a signal.

    Parameters
    ----------
    signal : ndarray
        Real-valued input signal.
    cmw_params_list : list of dict
        Each dict has 'f0', 'fwhm' (rad/year), and optionally 'label'.
        Typically produced by ormsby_spec_to_cmw_params().
    fs : float
        Sampling rate.
    analytic : bool
        If True, extract envelopes and phase.

    Returns
    -------
    results : dict
        'filter_outputs': list of dicts (same as apply_cmw() output + 'spec', 'index')
        'filter_specs': list of CMW param dicts
        'signal': original input signal
    """
    results = {
        'filter_outputs': [],
        'filter_specs': cmw_params_list,
        'signal': signal,
    }

    for i, params in enumerate(cmw_params_list):
        # For lowpass (f0=0), always use non-analytic (real output)
        use_analytic = analytic and (params['f0'] != 0)

        output = apply_cmw(signal, params['f0'], params['fwhm'], fs,
                           analytic=use_analytic)
        output['spec'] = params
        output['index'] = i
        results['filter_outputs'].append(output)

    return results
