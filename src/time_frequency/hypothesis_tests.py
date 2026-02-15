# -*- coding: utf-8 -*-
"""
Beating vs Drift Hypothesis Tests

Four quantitative tests to distinguish whether comb filter envelope
wobble arises from multi-line beating (interference between closely-spaced
spectral components) or single-line frequency drift over time.

Tests:
  1. Drift rate distribution -- are ridge frequencies stationary?
  2. Envelope wobble spectrum -- does envelope FFT show beat frequency peaks?
  3. FM-AM coupling -- does frequency variation correlate with amplitude?
  4. Synthetic beating -- does a two-component model reproduce real envelopes?

All frequencies in radians per year (rad/yr).

Reference: Hurst Appendix A; prd/supplementary_parametric_methods.md
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import ttest_1samp, pearsonr


def test_drift_rate_distribution(ridges):
    """
    Test 1: If lines are stationary, drift rates cluster near zero.

    Computes drift_rate for all ridges and tests H0: mean drift = 0
    using a two-sided t-test.

    Parameters
    ----------
    ridges : list of dict
        From detect_ridges(). Each has 'drift_rate' (rad/yr per year).

    Returns
    -------
    result : dict
        'drift_rates': ndarray
        'mean_drift': float
        'std_drift': float
        'median_drift': float
        't_statistic': float
        'p_value': float (two-sided, H0: mean=0)
        'conclusion': str ('stationary' or 'drifting')
    """
    drifts = np.array([r['drift_rate'] for r in ridges])

    if len(drifts) < 2:
        return {
            'drift_rates': drifts,
            'mean_drift': float(np.mean(drifts)) if len(drifts) else 0.0,
            'std_drift': 0.0,
            'median_drift': float(np.median(drifts)) if len(drifts) else 0.0,
            't_statistic': 0.0,
            'p_value': 1.0,
            'conclusion': 'insufficient data',
        }

    t_stat, p_val = ttest_1samp(drifts, 0.0)

    return {
        'drift_rates': drifts,
        'mean_drift': float(np.mean(drifts)),
        'std_drift': float(np.std(drifts)),
        'median_drift': float(np.median(drifts)),
        't_statistic': float(t_stat),
        'p_value': float(p_val),
        'conclusion': 'stationary' if p_val > 0.05 else 'drifting',
    }


def test_envelope_wobble_spectrum(filter_outputs, fs=52,
                                   nominal_line_spacing=0.3719):
    """
    Test 2: FFT of each filter envelope to detect beat frequencies.

    If beating dominates, the envelope should show a spectral peak near
    the nominal line spacing (~0.37 rad/yr, period ~17 years).

    Parameters
    ----------
    filter_outputs : list of dict
        Each dict has 'envelope' (ndarray) and optionally 'spec' with
        'f_center'.
    fs : float
        Sampling rate (samples/year).
    nominal_line_spacing : float
        Expected beating frequency (rad/yr). Default 0.3719.

    Returns
    -------
    result : dict
        'per_filter': list of dict, each with:
            'filter_index': int
            'center_freq': float or None
            'envelope_spectrum_freqs': ndarray (rad/yr)
            'envelope_spectrum_amps': ndarray
            'peak_frequency': float (dominant envelope modulation freq)
            'peak_period_years': float
            'has_beat_peak': bool (peak near nominal_line_spacing)
        'fraction_with_beat_peak': float
        'conclusion': str
    """
    per_filter = []

    for i, output in enumerate(filter_outputs):
        env = output.get('envelope')
        if env is None:
            continue

        # Remove mean (DC) from envelope
        env = np.asarray(env, dtype=np.float64)
        env_detrend = env - np.mean(env)

        # Zero-pad for better frequency resolution
        nfft = max(len(env_detrend) * 4, 8192)
        env_fft = np.abs(np.fft.rfft(env_detrend, n=nfft))
        freqs_hz = np.fft.rfftfreq(nfft, d=1.0 / fs)
        freqs_rad = freqs_hz * 2.0 * np.pi  # convert to rad/yr

        # Find dominant peak (exclude DC: start from index 1)
        if len(env_fft) > 1:
            peak_idx = np.argmax(env_fft[1:]) + 1
            peak_freq = freqs_rad[peak_idx]
            peak_period = 2.0 * np.pi / peak_freq if peak_freq > 0 else np.inf
        else:
            peak_freq = 0.0
            peak_period = np.inf

        # Check if there's a peak near the nominal line spacing
        search_lo = nominal_line_spacing * 0.5
        search_hi = nominal_line_spacing * 2.0
        mask = (freqs_rad >= search_lo) & (freqs_rad <= search_hi)
        has_beat_peak = False
        if np.any(mask):
            region = env_fft[mask]
            overall_rms = np.sqrt(np.mean(env_fft[1:] ** 2))
            if region.max() > 2.0 * overall_rms:
                has_beat_peak = True

        center_freq = None
        spec = output.get('spec')
        if spec is not None:
            center_freq = spec.get('f_center', None)

        per_filter.append({
            'filter_index': i,
            'center_freq': center_freq,
            'envelope_spectrum_freqs': freqs_rad,
            'envelope_spectrum_amps': env_fft,
            'peak_frequency': float(peak_freq),
            'peak_period_years': float(peak_period),
            'has_beat_peak': has_beat_peak,
        })

    n_with_beat = sum(1 for p in per_filter if p['has_beat_peak'])
    fraction = n_with_beat / len(per_filter) if per_filter else 0.0

    return {
        'per_filter': per_filter,
        'fraction_with_beat_peak': fraction,
        'conclusion': 'beating likely' if fraction > 0.5 else 'no clear beating',
    }


def test_fm_am_coupling(filter_outputs, fs=52):
    """
    Test 3: Correlate instantaneous frequency with envelope amplitude.

    For beating, the apparent frequency oscillates between component
    frequencies, and the envelope peaks when components are in phase.
    Strong correlation between |f - f_mean| and envelope indicates
    FM-AM coupling characteristic of beating.

    Parameters
    ----------
    filter_outputs : list of dict
        Each has 'envelope' and 'frequency' keys.
    fs : float
        Sampling rate.

    Returns
    -------
    result : dict
        'per_filter': list of dict with 'correlation', 'p_value'
        'mean_abs_correlation': float
        'fraction_significant': float (p < 0.05)
        'conclusion': str
    """
    per_filter = []

    for i, output in enumerate(filter_outputs):
        env = output.get('envelope')
        freq = output.get('frequency')

        if env is None or freq is None:
            continue

        env = np.asarray(env, dtype=np.float64)
        freq = np.asarray(freq, dtype=np.float64)

        # Remove NaN
        valid = ~(np.isnan(env) | np.isnan(freq))
        if valid.sum() < 10:
            continue

        env_v = env[valid]
        freq_v = freq[valid]

        # Frequency deviation from mean
        freq_dev = np.abs(freq_v - np.mean(freq_v))

        try:
            corr, p_val = pearsonr(env_v, freq_dev)
        except Exception:
            corr, p_val = 0.0, 1.0

        per_filter.append({
            'filter_index': i,
            'correlation': float(corr),
            'p_value': float(p_val),
        })

    if not per_filter:
        return {
            'per_filter': [],
            'mean_abs_correlation': 0.0,
            'fraction_significant': 0.0,
            'conclusion': 'no data',
        }

    abs_corrs = [abs(p['correlation']) for p in per_filter]
    sig_count = sum(1 for p in per_filter if p['p_value'] < 0.05)
    fraction_sig = sig_count / len(per_filter)

    return {
        'per_filter': per_filter,
        'mean_abs_correlation': float(np.mean(abs_corrs)),
        'fraction_significant': fraction_sig,
        'conclusion': 'FM-AM coupling detected' if fraction_sig > 0.5
                      else 'no significant FM-AM coupling',
    }


def test_synthetic_beating(f1, f2, duration_samples, fs=52,
                            filter_func=None, a1=1.0, a2=1.0):
    """
    Test 4: Create two sinusoids at adjacent frequencies and compare
    the resulting envelope to expected beat pattern.

    If the real comb filter envelopes match this synthetic pattern,
    beating is the dominant mechanism.

    Parameters
    ----------
    f1, f2 : float
        Frequencies of two sinusoids (rad/yr).
    duration_samples : int
        Signal length in samples.
    fs : float
        Sampling rate (samples/year).
    filter_func : callable or None
        If provided, called as filter_func(signal) to apply a filter.
        If None, no filtering is applied.
    a1, a2 : float
        Amplitudes of the two sinusoids.

    Returns
    -------
    result : dict
        'synthetic_signal': ndarray -- raw two-tone signal
        'filtered_signal': ndarray or None -- after filtering
        'envelope': ndarray -- envelope of (filtered or raw) signal
        'beat_freq_expected': float -- |f1 - f2| (rad/yr)
        'beat_period_expected': float -- 2*pi / |f1 - f2| (years)
        'beat_freq_measured': float -- from envelope FFT
        'beat_period_measured': float
        'period_match': bool -- within 20%
    """
    t = np.arange(duration_samples) / fs  # time in years

    # Two-tone signal
    signal = a1 * np.sin(f1 * t) + a2 * np.sin(f2 * t)

    # Apply filter if provided
    if filter_func is not None:
        filtered = filter_func(signal)
    else:
        filtered = None

    # Compute envelope via Hilbert transform
    from scipy.signal import hilbert
    target = filtered if filtered is not None else signal
    analytic = hilbert(target)
    envelope = np.abs(analytic)

    # Expected beat
    beat_freq_expected = abs(f1 - f2)
    beat_period_expected = 2.0 * np.pi / beat_freq_expected if beat_freq_expected > 0 else np.inf

    # Measure beat from envelope FFT
    env_detrend = envelope - np.mean(envelope)
    nfft = max(len(env_detrend) * 4, 8192)
    env_fft = np.abs(np.fft.rfft(env_detrend, n=nfft))
    freqs_hz = np.fft.rfftfreq(nfft, d=1.0 / fs)
    freqs_rad = freqs_hz * 2.0 * np.pi

    # Find dominant peak (skip DC)
    if len(env_fft) > 1:
        peak_idx = np.argmax(env_fft[1:]) + 1
        beat_freq_measured = float(freqs_rad[peak_idx])
        beat_period_measured = 2.0 * np.pi / beat_freq_measured if beat_freq_measured > 0 else np.inf
    else:
        beat_freq_measured = 0.0
        beat_period_measured = np.inf

    # Check if periods match within 20%
    if beat_period_expected > 0 and beat_period_measured > 0:
        ratio = beat_period_measured / beat_period_expected
        period_match = 0.8 <= ratio <= 1.2
    else:
        period_match = False

    return {
        'synthetic_signal': signal,
        'filtered_signal': filtered,
        'envelope': envelope,
        'beat_freq_expected': float(beat_freq_expected),
        'beat_period_expected': float(beat_period_expected),
        'beat_freq_measured': beat_freq_measured,
        'beat_period_measured': beat_period_measured,
        'period_match': period_match,
    }
