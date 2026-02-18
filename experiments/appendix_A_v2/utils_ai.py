# -*- coding: utf-8 -*-
"""
Shared utilities for Appendix A Figures AI-2 through AI-7 reproduction.

Provides:
- Data loading (weekly and daily DJIA)
- Comb filter bank design (Hurst's 23-filter spec)
- Ormsby kernel creation
- CMW frequency-domain response computation
- Display window helpers
- MPM 1-mode frequency estimator
- Prony sliding-window LSE

Reference: J.M. Hurst, The Profit Magic of Stock Transaction Timing,
           Appendix A, Figures AI-2 through AI-7
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pandas as pd
from scipy.linalg import hankel

from src.filters import (
    design_hurst_comb_bank,
    create_filter_kernels,
    apply_filter_bank,
)
from src.time_frequency.cmw import ormsby_spec_to_cmw_params


# ============================================================================
# CONSTANTS
# ============================================================================

# Data paths (relative to project root)
_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
CSV_WEEKLY = os.path.join(_BASE, 'data/raw/^dji_w.csv')
CSV_DAILY  = os.path.join(_BASE, 'data/raw/^dji_d.csv')

# Display window: the period Hurst used for his comb filter example figures
DATE_DISPLAY_START  = '1934-12-07'   # corrected from Hurst's editorial error
DATE_DISPLAY_END    = '1940-01-26'   # ~267 weeks

# Legacy aliases kept for back-compat
DATE_ANALYSIS_START = '1921-04-29'
DATE_ANALYSIS_END   = '1965-05-21'

# Hurst's comb filter bank specification (Appendix A, p.192)
N_FILTERS       = 23
W1_START        = 7.2    # rad/yr - lower skirt edge of first filter
W_STEP          = 0.2    # rad/yr - step between successive filters
PASSBAND_WIDTH  = 0.2    # rad/yr - flat passband width
SKIRT_WIDTH     = 0.3    # rad/yr - transition band width
NW_WEEKLY       = 3501   # Filter taps for weekly data

FS_WEEKLY = 52           # samples/year for weekly data


# ============================================================================
# DATA LOADING
# ============================================================================

def load_weekly_data(date_start=None, date_end=None):
    """
    Load DJIA weekly data.

    Loads ALL rows in the CSV by default (date_start=None, date_end=None),
    so the full record is available for edge-effect-free filtering.
    Pass explicit dates only when you need a specific sub-range.

    Returns
    -------
    close : ndarray
    dates_dt : DatetimeIndex
    """
    df = pd.read_csv(CSV_WEEKLY)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    if date_start is not None:
        df = df[df.Date >= pd.to_datetime(date_start)]
    if date_end is not None:
        df = df[df.Date <= pd.to_datetime(date_end)]
    df = df.reset_index(drop=True)
    return df.Close.values, pd.to_datetime(df.Date.values)


def load_daily_data(date_start=None, date_end=None):
    """
    Load DJIA daily data.

    Loads ALL rows in the CSV by default. Computes effective fs from the
    actual date span of the loaded data.

    Returns
    -------
    close : ndarray
    dates_dt : DatetimeIndex
    fs_daily : float  (trading days per year)
    """
    df = pd.read_csv(CSV_DAILY)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    if date_start is not None:
        df = df[df.Date >= pd.to_datetime(date_start)]
    if date_end is not None:
        df = df[df.Date <= pd.to_datetime(date_end)]
    df = df.reset_index(drop=True)
    dates_dt = pd.to_datetime(df.Date.values)
    close = df.Close.values
    total_years = (dates_dt[-1] - dates_dt[0]).days / 365.25
    fs_daily = len(close) / total_years
    return close, dates_dt, fs_daily


# ============================================================================
# FILTER BANK DESIGN
# ============================================================================

def design_comb_bank(fs=FS_WEEKLY, nw=NW_WEEKLY):
    """
    Design Hurst's 23-filter uniform-step comb bank.

    Parameters
    ----------
    fs : float
        Sampling rate (samples/year)
    nw : int
        Filter length in samples

    Returns
    -------
    specs : list of dict
        Filter specifications with f1, f2, f3, f4, f_center, label etc.
    """
    return design_hurst_comb_bank(
        n_filters=N_FILTERS,
        w1_start=W1_START,
        w_step=W_STEP,
        passband_width=PASSBAND_WIDTH,
        skirt_width=SKIRT_WIDTH,
        nw=nw,
        fs=fs
    )


def daily_nw(fs_daily):
    """
    Compute appropriate filter length for daily data.
    Same time span as the weekly filter (3501 weeks).
    """
    nw = int(NW_WEEKLY * fs_daily / FS_WEEKLY)
    if nw % 2 == 0:
        nw += 1
    return nw


def make_ormsby_kernels(specs, fs=FS_WEEKLY):
    """
    Create complex analytic Ormsby kernels for all filter specs.

    Returns
    -------
    filters : list of dict
        Each with 'kernel' (complex ndarray), 'spec', 'nw'
    """
    return create_filter_kernels(specs, fs=fs, filter_type='modulate', analytic=True)


def apply_comb_bank(signal, filters, fs=FS_WEEKLY, mode='reflect'):
    """
    Apply all comb filters to signal. Returns list of per-filter dicts.

    Each dict has: signal (complex), envelope, phase, frequency, spec, index.
    """
    results = apply_filter_bank(signal, filters, fs=fs, mode=mode)
    return results['filter_outputs']


# ============================================================================
# CMW FREQUENCY-DOMAIN RESPONSE
# ============================================================================

def cmw_frequency_response(specs, fs, nfft=65536):
    """
    Compute CMW frequency-domain responses (Gaussian envelopes) matched
    to the Ormsby filter specifications.

    Parameters
    ----------
    specs : list of dict
        Ormsby filter specs from design_comb_bank()
    fs : float
        Sampling rate (samples/year)
    nfft : int
        Number of frequency points

    Returns
    -------
    freqs_radyr : ndarray
        Frequency axis in rad/yr (positive only)
    H_cmw : ndarray, shape (n_filters, len(freqs_radyr))
        Normalized CMW responses (peak = 1.0 each)
    """
    # Positive frequency axis: 0 to Nyquist in rad/yr
    nyq_radyr = np.pi * fs
    freqs_radyr = np.linspace(0, nyq_radyr, nfft // 2 + 1)

    H_cmw = np.zeros((len(specs), len(freqs_radyr)))

    for i, spec in enumerate(specs):
        params = ormsby_spec_to_cmw_params(spec)
        f0 = params['f0']
        sigma_f = params['sigma_f']
        # Gaussian in frequency domain: one-sided (positive f)
        H = np.exp(-0.5 * ((freqs_radyr - f0) / sigma_f) ** 2)
        # Normalize to peak = 1.0
        peak = np.max(H)
        if peak > 0:
            H /= peak
        H_cmw[i] = H

    return freqs_radyr, H_cmw


def ormsby_frequency_response(filters, fs, nfft=65536):
    """
    Compute actual FFT frequency responses for Ormsby filter kernels.

    Parameters
    ----------
    filters : list of dict
        From make_ormsby_kernels()
    fs : float
        Sampling rate (samples/year)
    nfft : int
        FFT size

    Returns
    -------
    freqs_radyr : ndarray
        Frequency axis in rad/yr
    H_ormsby : ndarray, shape (n_filters, len(freqs_radyr))
        Normalized responses (peak = 1.0 each)
    """
    # For complex analytic kernels, use full FFT and take positive freqs
    n_filters = len(filters)
    first_kernel = filters[0]['kernel']
    is_complex = np.iscomplexobj(first_kernel)

    if is_complex:
        full_freqs = np.fft.fftfreq(nfft, d=1.0 / fs) * 2 * np.pi  # rad/yr
        pos_mask = full_freqs >= 0
        freqs_radyr = full_freqs[pos_mask]
        H_ormsby = np.zeros((n_filters, np.sum(pos_mask)))
        for i, f in enumerate(filters):
            H = np.abs(np.fft.fft(f['kernel'], n=nfft)[pos_mask])
            peak = np.max(H)
            H_ormsby[i] = H / peak if peak > 0 else H
    else:
        freqs_radyr = np.fft.rfftfreq(nfft, d=1.0 / fs) * 2 * np.pi
        H_ormsby = np.zeros((n_filters, len(freqs_radyr)))
        for i, f in enumerate(filters):
            H = np.abs(np.fft.rfft(f['kernel'], n=nfft))
            peak = np.max(H)
            H_ormsby[i] = H / peak if peak > 0 else H

    return freqs_radyr, H_ormsby


def idealized_response(spec, freqs_radyr):
    """
    Compute idealized trapezoidal response for one filter spec.

    Returns
    -------
    H : ndarray, same length as freqs_radyr
    """
    f1, f2, f3, f4 = spec['f1'], spec['f2'], spec['f3'], spec['f4']
    H = np.zeros_like(freqs_radyr, dtype=float)
    # Rising skirt
    mask_rise = (freqs_radyr >= f1) & (freqs_radyr < f2)
    H[mask_rise] = (freqs_radyr[mask_rise] - f1) / (f2 - f1)
    # Passband
    mask_pass = (freqs_radyr >= f2) & (freqs_radyr <= f3)
    H[mask_pass] = 1.0
    # Falling skirt
    mask_fall = (freqs_radyr > f3) & (freqs_radyr <= f4)
    H[mask_fall] = (f4 - freqs_radyr[mask_fall]) / (f4 - f3)
    return H


# ============================================================================
# DISPLAY WINDOW HELPER
# ============================================================================

def get_window(dates_dt, date_start=DATE_DISPLAY_START, date_end=DATE_DISPLAY_END):
    """
    Return (s_idx, e_idx) for the given date window.
    If dates not found, returns (0, len(dates_dt)).
    """
    mask = (dates_dt >= pd.to_datetime(date_start)) & \
           (dates_dt <= pd.to_datetime(date_end))
    if not mask.any():
        return 0, len(dates_dt)
    idx = np.where(mask)[0]
    return idx[0], idx[-1] + 1


# ============================================================================
# MPM: 1-MODE MATRIX PENCIL METHOD
# ============================================================================

def mpm_1mode(z, L_pencil=None):
    """
    1-mode Matrix Pencil Method for frequency estimation.

    Fits a single complex exponential z[n] ~ A * exp(j * omega * n)
    to the input sequence and returns the angular frequency.

    Parameters
    ----------
    z : array-like (complex or real)
        Input sequence (preferably complex analytic signal)
    L_pencil : int, optional
        Pencil parameter L (default: len(z) // 3)

    Returns
    -------
    omega : float
        Angular frequency in rad/sample (range: -pi to pi)
        Returns NaN if estimation fails.

    Notes
    -----
    See: Hua & Sarkar (1990), "Matrix Pencil Method for Estimating Parameters
    of Exponentially Damped/Undamped Sinusoids in Noise", IEEE Trans. ASSP.
    """
    z = np.asarray(z, dtype=complex)
    N = len(z)
    if N < 4:
        return np.nan

    if L_pencil is None:
        L_pencil = max(2, N // 3)
    L_pencil = min(L_pencil, N - 2)

    try:
        # Form shifted Hankel matrices Y0 and Y1
        # Y0: rows indexed by [0..N-L-1], cols by [0..L]   (shape: N-L x L+1)
        # But for standard MPM: form Y of shape (N-L) x (L+1)
        # Y0 = Y[:, :-1],  Y1 = Y[:, 1:]
        row = z[:N - L_pencil]
        col = z[N - L_pencil - 1:]
        Y = hankel(row, col)           # shape: (N-L) x (L+1)
        Y0 = Y[:, :-1]                 # shape: (N-L) x L
        Y1 = Y[:, 1:]                  # shape: (N-L) x L

        # SVD of Y0, keep only 1 mode
        U, s, Vh = np.linalg.svd(Y0, full_matrices=False)
        U1  = U[:, :1]
        s1  = s[:1]
        Vh1 = Vh[:1, :]

        if s1[0] < 1e-12:
            return np.nan

        # Compute Z = pinv(Y0_1mode) @ Y1
        # pinv of rank-1 approx: Y0_pinv = Vh1.H @ diag(1/s1) @ U1.H
        Y0_pinv = Vh1.conj().T @ np.diag(1.0 / s1) @ U1.conj().T
        Z_matrix = Y0_pinv @ Y1

        # Single eigenvalue
        pole = np.linalg.eigvals(Z_matrix)
        if len(pole) == 0:
            return np.nan

        # Take the dominant pole (largest magnitude)
        pole_main = pole[np.argmax(np.abs(pole))]
        omega = np.angle(pole_main)   # rad/sample
        return omega

    except (np.linalg.LinAlgError, ValueError):
        return np.nan


def mpm_to_radyr(omega_rad_sample, fs):
    """Convert MPM output (rad/sample) to rad/yr."""
    return omega_rad_sample * fs


# ============================================================================
# FREQUENCY MEASUREMENT: ZERO-CROSSING HALF-PERIOD
# ============================================================================

def measure_phase_halfperiod(analytic_signal, fs):
    """
    Measure instantaneous frequency from the **wrapped phase** of an analytic
    signal, using the times when the phase passes through 0 (peaks) and ±π
    (troughs) of the real part.

    For y(t) = A(t)*exp(jφ(t)):
      - real(y) peaks  when φ = 0   (mod 2π)  → wrapped phase crosses 0 upward
      - real(y) troughs when φ = ±π (mod 2π)  → wrapped phase jumps near ±π

    Half-period between consecutive peak↔trough events → frequency.
    Sub-sample interpolation is performed at each event for accuracy.

    Parameters
    ----------
    analytic_signal : ndarray (complex)
    fs : float  (samples/year)

    Returns
    -------
    times : ndarray  (sample positions, fractional)
    freqs : ndarray  (rad/yr)
    event_types : list of str  ('peak' or 'trough') – one per time point
    """
    phi = np.angle(analytic_signal)   # wrapped phase ∈ (-π, π]
    N = len(phi)

    peaks   = []   # fractional sample positions
    troughs = []

    for i in range(N - 1):
        p0, p1 = phi[i], phi[i + 1]
        diff = p1 - p0   # can be large if wrapping occurs

        # ---- Peak: phase crosses 0 upward (p0 < 0, p1 >= 0, no large jump) ----
        if p0 < 0 and p1 >= 0 and abs(diff) < np.pi:
            frac = abs(p0) / (abs(p0) + p1) if (abs(p0) + p1) > 0 else 0.5
            peaks.append(i + frac)

        # ---- Trough: phase jumps near ±π  (large negative jump means ↑→±π) ----
        # Wrapped phase at a trough: goes from ~+π to ~-π  → diff ≈ -2π
        elif diff < -np.pi:
            # Interpolate crossing of +π
            # p0 is close to +π, p1 is close to -π after wrap
            # Effective p1_unwrapped = p1 + 2π
            p1_uw = p1 + 2 * np.pi
            frac = (np.pi - p0) / (p1_uw - p0) if (p1_uw - p0) > 0 else 0.5
            frac = max(0.0, min(1.0, frac))
            troughs.append(i + frac)

    peaks   = np.array(peaks)
    troughs = np.array(troughs)

    # Interleave peaks and troughs chronologically
    events = ([(t, 'peak')   for t in peaks] +
              [(t, 'trough') for t in troughs])
    events.sort(key=lambda x: x[0])

    if len(events) < 2:
        return np.array([]), np.array([]), []

    times, freqs, etypes = [], [], []
    for k in range(len(events) - 1):
        t1, e1 = events[k]
        t2, e2 = events[k + 1]
        dt = t2 - t1
        if dt < 0.5:
            continue
        dt_yr = dt / fs
        f = np.pi / dt_yr        # half-period → angular frequency
        times.append((t1 + t2) / 2.0)
        freqs.append(f)
        etypes.append(f'{e1[0]}-{e2[0]}')   # 'p-t' or 't-p'

    return np.array(times), np.array(freqs), etypes


def measure_zerocross_halfperiod(signal_real, fs):
    """
    Measure frequency at each half-cycle of a filtered signal using
    zero-crossing intervals.

    Finds all zero crossings (sign changes), measures time between
    consecutive crossings (= half-period), converts to frequency.

    Parameters
    ----------
    signal_real : ndarray
        Real part of filtered signal
    fs : float
        Sampling rate (samples/year)

    Returns
    -------
    times : ndarray
        Sample index at midpoint of each half-cycle
    freqs : ndarray
        Instantaneous frequency in rad/yr for each half-cycle
    """
    s = np.asarray(signal_real, dtype=float)
    # Detect sign changes
    signs = np.sign(s)
    # Ignore zero-value samples (rare)
    nz = signs != 0
    idx = np.where(nz)[0]
    if len(idx) < 2:
        return np.array([]), np.array([])

    # Linear interpolation for sub-sample zero crossing
    crossings = []
    for i in range(len(idx) - 1):
        a, b = idx[i], idx[i + 1]
        if signs[a] != signs[b]:  # sign change between consecutive non-zero samples
            # Linear interpolation: crossing at a + |s[a]| / (|s[a]| + |s[b]|)
            frac = abs(s[a]) / (abs(s[a]) + abs(s[b]))
            crossings.append(a + frac)

    crossings = np.array(crossings)
    if len(crossings) < 2:
        return np.array([]), np.array([])

    # Half-period between consecutive crossings
    dt_samples = np.diff(crossings)
    dt_years   = dt_samples / fs
    freqs      = np.pi / dt_years      # rad/yr (half-period)
    times      = (crossings[:-1] + crossings[1:]) / 2.0  # midpoint

    return times, freqs


# ============================================================================
# PRONY SLIDING-WINDOW LSE
# ============================================================================

def prony_sliding_lse(analytic_signal, fs, f_center, n_periods=1.5, step_frac=0.5):
    """
    Sliding-window 1-mode Prony (MPM) frequency analysis.

    For each window position, estimates the instantaneous frequency using
    MPM on the complex analytic signal.

    Parameters
    ----------
    analytic_signal : ndarray (complex)
        Complex analytic filter output
    fs : float
        Sampling rate (samples/year)
    f_center : float
        Filter center frequency in rad/yr (used to set window length)
    n_periods : float
        Window length in signal periods (default 1.5)
    step_frac : float
        Step size as fraction of window length (default 0.5 = 50% overlap)

    Returns
    -------
    t_centers : ndarray
        Window center positions in samples
    f_estimates : ndarray
        Frequency estimates in rad/yr
    t_starts : ndarray
        Window start positions in samples
    t_ends : ndarray
        Window end positions in samples
    """
    N = len(analytic_signal)

    # Window half-length in samples
    period_samples = 2 * np.pi / f_center * fs   # one period in samples
    W_half = max(10, int(round(period_samples * n_periods / 2)))
    W = 2 * W_half
    step = max(1, int(round(W * step_frac)))

    t_centers   = []
    f_estimates = []
    t_starts    = []
    t_ends      = []

    t = W_half
    while t + W_half <= N:
        seg = analytic_signal[t - W_half : t + W_half]
        # Detrend (remove mean)
        seg = seg - np.mean(seg)
        omega_samp = mpm_1mode(seg)
        if not np.isnan(omega_samp):
            freq_radyr = abs(omega_samp) * fs
            # Sanity check: must be within 2x of filter center
            if 0.3 * f_center < freq_radyr < 3.0 * f_center:
                t_centers.append(t)
                f_estimates.append(freq_radyr)
                t_starts.append(t - W_half)
                t_ends.append(t + W_half)
        t += step

    return (np.array(t_centers), np.array(f_estimates),
            np.array(t_starts), np.array(t_ends))


# ============================================================================
# PRINT SUMMARY
# ============================================================================

def print_comb_bank_summary(specs):
    """Print filter bank summary to console."""
    print(f"Comb filter bank: {len(specs)} filters")
    print(f"  Centers: {specs[0]['f_center']:.1f} to {specs[-1]['f_center']:.1f} rad/yr")
    print(f"  Passband: {PASSBAND_WIDTH} rad/yr, Skirt: {SKIRT_WIDTH} rad/yr")
    print(f"  Filter 1:  "
          f"f1={specs[0]['f1']:.1f} f2={specs[0]['f2']:.1f} "
          f"f3={specs[0]['f3']:.1f} f4={specs[0]['f4']:.1f} rad/yr")
    print(f"  Filter 23: "
          f"f1={specs[-1]['f1']:.1f} f2={specs[-1]['f2']:.1f} "
          f"f3={specs[-1]['f3']:.1f} f4={specs[-1]['f4']:.1f} rad/yr")
