# -*- coding: utf-8 -*-
"""
Extended CMW Comb Bank Design — Stage 6

Designs CMW filter banks for both standard (weekly) and extended (daily)
frequency ranges. Includes narrowband CMW design for resolving individual
spectral lines.

Key insight: CMW with FWHM ~ 0.1-0.2 rad/yr can potentially resolve
individual harmonics spaced at 0.3676 rad/yr, eliminating the need
for Ormsby comb filters.

Reference: J.M. Hurst, The Profit Magic of Stock Transaction Timing (1970)
"""

import numpy as np
from src.time_frequency.cmw import apply_cmw, apply_cmw_bank


# =============================================================================
# Standard CMW Comb Bank (matches Ormsby 23-filter design)
# =============================================================================

def design_standard_cmw_bank(n_filters=23, w1_start=7.2, w_step=0.2,
                              passband_width=0.2, skirt_width=0.3):
    """
    Design a standard CMW comb bank matching Hurst's Ormsby comb bank.

    This replicates the 23-filter bank from Appendix A but using CMW
    instead of Ormsby filters.

    Returns
    -------
    cmw_params : list of dict
        Each dict has 'f0', 'fwhm', 'label'
    """
    cmw_params = []
    for i in range(n_filters):
        w1 = w1_start + i * w_step
        w2 = w1 + skirt_width
        w3 = w2 + passband_width
        w4 = w3 + skirt_width

        f0 = (w2 + w3) / 2.0
        fwhm = (w3 + w4) / 2.0 - (w1 + w2) / 2.0

        cmw_params.append({
            'f0': f0,
            'fwhm': fwhm,
            'label': f'CMW-{i+1} fc={f0:.2f}',
        })

    return cmw_params


# =============================================================================
# Extended CMW Comb Bank for Daily Data
# =============================================================================

def design_extended_cmw_bank(fs=252, omega_min=0.5, omega_max=80.0):
    """
    Design an extended CMW filter bank covering the full daily spectrum.

    Uses regional bandwidth strategy:
    - Low-freq  (0.5-3 rad/yr):  ~10 filters, BW~0.3 each
    - Mid-freq  (3-13 rad/yr):   ~30 filters, BW~0.35 each
    - High-freq (13-40 rad/yr):  ~40 filters, BW~0.7 each
    - Very-high (40-80 rad/yr):  ~20 filters, BW~2.0 each

    Parameters
    ----------
    fs : float
        Sampling rate (samples/year)
    omega_min : float
        Minimum center frequency (rad/yr)
    omega_max : float
        Maximum center frequency (rad/yr)

    Returns
    -------
    cmw_params : list of dict
        Each dict has 'f0', 'fwhm', 'label', 'region'
    """
    nyquist = np.pi * fs
    omega_max = min(omega_max, nyquist * 0.9)

    regions = [
        {'name': 'low',       'start': 0.5,  'end': 3.0,  'bw': 0.30, 'step': 0.25},
        {'name': 'mid',       'start': 3.0,  'end': 13.0, 'bw': 0.35, 'step': 0.35},
        {'name': 'high',      'start': 13.0, 'end': 40.0, 'bw': 0.70, 'step': 0.70},
        {'name': 'very_high', 'start': 40.0, 'end': omega_max, 'bw': 2.0, 'step': 2.0},
    ]

    cmw_params = []
    idx = 0
    for region in regions:
        f0 = region['start'] + region['step'] / 2
        while f0 < region['end']:
            cmw_params.append({
                'f0': f0,
                'fwhm': region['bw'],
                'label': f'Ext-{idx+1} fc={f0:.2f} ({region["name"]})',
                'region': region['name'],
            })
            f0 += region['step']
            idx += 1

    return cmw_params


# =============================================================================
# Narrowband CMW Bank — resolves individual harmonics
# =============================================================================

def design_narrowband_cmw_bank(w0, max_N=None, fs=52,
                                fwhm_factor=0.5, omega_min=0.5):
    """
    Design a narrowband CMW bank that targets individual harmonics.

    Each CMW is centered at N*w0 with FWHM = w0 * fwhm_factor.
    With fwhm_factor=0.5 and w0=0.3676, FWHM≈0.18 rad/yr — narrow
    enough to resolve lines spaced at 0.3676 rad/yr.

    This is the KEY new capability: instead of a comb bank that sees
    2-3 lines per filter, we use one CMW per harmonic.

    Parameters
    ----------
    w0 : float
        Fundamental spacing (rad/yr)
    max_N : int or None
        Maximum harmonic number. Auto from Nyquist if None.
    fs : float
        Sampling rate
    fwhm_factor : float
        FWHM as fraction of w0. Default 0.5 gives FWHM ≈ w0/2,
        which provides good isolation between adjacent harmonics.
        Use 0.3 for very narrow (more spectral leakage but better isolation).
        Use 0.8 for wider (smoother envelope but some harmonic blending).
    omega_min : float
        Minimum center frequency (skip very low N where cycles are too long)

    Returns
    -------
    cmw_params : list of dict
        One per harmonic, with 'f0', 'fwhm', 'N', 'period_yr', 'label'
    """
    nyquist = np.pi * fs
    if max_N is None:
        max_N = int(nyquist * 0.9 / w0)

    fwhm_base = w0 * fwhm_factor

    cmw_params = []
    for N in range(1, max_N + 1):
        f0 = N * w0
        if f0 < omega_min:
            continue
        if f0 > nyquist * 0.9:
            break

        # For low frequencies (long periods), use slightly wider FWHM
        # to avoid excessively long time-domain wavelets
        period_yr = 2 * np.pi / f0
        min_data_years = 3 * period_yr  # Need at least 3 cycles
        fwhm = max(fwhm_base, f0 * 0.1)  # Floor at 10% of center freq

        cmw_params.append({
            'f0': f0,
            'fwhm': fwhm,
            'N': N,
            'period_yr': period_yr,
            'period_wk': period_yr * 52,
            'label': f'N={N} fc={f0:.2f} T={period_yr:.2f}yr',
        })

    return cmw_params


# =============================================================================
# Run CMW Comb Bank Analysis
# =============================================================================

def run_cmw_comb_bank(signal, fs, cmw_params, analytic=True,
                       spacing=1, startidx=0, interp='none'):
    """
    Apply a CMW filter bank to a signal and extract frequency traces.

    Parameters
    ----------
    signal : ndarray
        Price data (raw close or log close)
    fs : float
        Sampling rate
    cmw_params : list of dict
        CMW parameters (from design_* functions)
    analytic : bool
        Extract envelope and phase
    spacing, startidx, interp : int, int, str
        Decimation parameters

    Returns
    -------
    results : dict
        'filter_outputs': list of dicts with signal, envelope, frequency, etc.
        'filter_specs': the CMW params
        'median_freqs': median instantaneous frequency per filter (rad/yr)
        'mean_envelopes': mean envelope amplitude per filter
    """
    bank_result = apply_cmw_bank(signal, cmw_params, fs,
                                  analytic=analytic,
                                  spacing=spacing, startidx=startidx,
                                  interp=interp)

    # Extract summary statistics
    median_freqs = []
    mean_envelopes = []
    freq_stabilities = []

    for i, output in enumerate(bank_result['filter_outputs']):
        if output['frequency'] is not None:
            # Convert cycles/yr to rad/yr
            freq_rad = output['frequency'] * 2 * np.pi
            # Use middle 80% to avoid edge effects
            n = len(freq_rad)
            trim = int(n * 0.1)
            if trim > 0:
                freq_trimmed = freq_rad[trim:-trim]
            else:
                freq_trimmed = freq_rad

            med_f = float(np.median(freq_trimmed))
            std_f = float(np.std(freq_trimmed))
            median_freqs.append(med_f)
            freq_stabilities.append(std_f / med_f if med_f > 0 else 1.0)
        else:
            median_freqs.append(cmw_params[i]['f0'])
            freq_stabilities.append(1.0)

        if output['envelope'] is not None:
            mean_envelopes.append(float(np.mean(output['envelope'])))
        else:
            mean_envelopes.append(0.0)

    bank_result['median_freqs'] = np.array(median_freqs)
    bank_result['mean_envelopes'] = np.array(mean_envelopes)
    bank_result['freq_stabilities'] = np.array(freq_stabilities)

    return bank_result


# =============================================================================
# Line extraction from narrowband CMW results
# =============================================================================

def extract_lines_from_narrowband(bank_result, w0, min_envelope_ratio=0.05,
                                   max_freq_cv=0.3):
    """
    Extract confirmed spectral lines from narrowband CMW analysis.

    A harmonic N is "confirmed" if:
    1. Its envelope amplitude is above the expected 1/w envelope floor
    2. Its frequency is stable (CV < max_freq_cv)
    3. Its median frequency is within 20% of N*w0

    The amplitude threshold accounts for the 1/w decay: higher harmonics
    naturally have smaller envelopes, so we compare each line against
    the expected k/w envelope rather than against the global max.

    Parameters
    ----------
    bank_result : dict
        Output from run_cmw_comb_bank with narrowband params
    w0 : float
        Fundamental spacing
    min_envelope_ratio : float
        Minimum envelope as fraction of expected 1/w value at that frequency
    max_freq_cv : float
        Maximum coefficient of variation for frequency stability

    Returns
    -------
    confirmed_lines : list of dict
        Each has N, frequency, amplitude, stability, confidence
    """
    specs = bank_result['filter_specs']
    med_freqs = bank_result['median_freqs']
    mean_envs = bank_result['mean_envelopes']
    freq_cvs = bank_result['freq_stabilities']

    if len(mean_envs) == 0:
        return []

    # Estimate expected envelope: fit k/w to the mean envelopes
    # Use robust estimate: median of (amp * freq) across filters
    env_arr = np.array(mean_envs)
    freq_arr = np.array([s.get('f0', 1.0) for s in specs])
    valid = (env_arr > 0) & (freq_arr > 0)
    if np.any(valid):
        aw_products = env_arr[valid] * freq_arr[valid]
        k_est = np.median(aw_products)
    else:
        k_est = np.max(env_arr)

    confirmed = []
    for i, spec in enumerate(specs):
        if 'N' not in spec:
            continue

        N = spec['N']
        expected_f = N * w0
        measured_f = med_freqs[i]
        env_amp = mean_envs[i]
        cv = freq_cvs[i]

        # Expected amplitude from 1/w envelope
        expected_amp = k_est / expected_f if expected_f > 0 else 0

        # Check criteria
        env_ok = env_amp > min_envelope_ratio * expected_amp
        freq_ok = cv < max_freq_cv
        match_ok = abs(measured_f - expected_f) / expected_f < 0.20

        if env_ok and freq_ok and match_ok:
            # Confidence based on frequency stability and relative amplitude
            amp_ratio = env_amp / expected_amp if expected_amp > 0 else 0
            if cv < 0.05 and amp_ratio > 0.5:
                conf = 'high'
            elif cv < 0.15:
                conf = 'medium'
            else:
                conf = 'low'

            confirmed.append({
                'N': N,
                'frequency': float(expected_f),
                'measured_freq': float(measured_f),
                'period_yr': float(2 * np.pi / expected_f),
                'period_wk': float(2 * np.pi / expected_f * 52),
                'amplitude': float(env_amp),
                'freq_cv': float(cv),
                'confidence': conf,
            })

    return confirmed
