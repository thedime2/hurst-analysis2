# -*- coding: utf-8 -*-
"""
Automated Nominal Model Derivation Pipeline — Stages 0-7

Derives the Nominal Model from raw price data in a single function call.
Implements the 10-stage pipeline from prd/nominal_model_pipeline.md.

Reference: J.M. Hurst, The Profit Magic of Stock Transaction Timing (1970),
           Appendix A
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from src.spectral.lanczos import lanczos_spectrum
from src.spectral.peak_detection import (
    find_spectral_peaks, find_spectral_troughs, detect_fine_structure_spacing
)
from src.spectral.envelopes import (
    fit_power_law_envelope, fit_upper_envelope, fit_lower_envelope
)


# =============================================================================
# Result container
# =============================================================================

@dataclass
class NominalModelResult:
    """Container for the complete pipeline output."""
    # Stage 0
    close: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    dates: pd.Series = field(default_factory=pd.Series, repr=False)
    fs: float = 0.0
    n_samples: int = 0
    years: float = 0.0
    label: str = ''

    # Stage 1
    omega_yr: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    amp: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)

    # Stage 2
    peak_freqs: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    peak_amps: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    trough_freqs: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    trough_amps: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)

    # Stage 3
    envelope_upper: dict = field(default_factory=dict)
    envelope_lower: dict = field(default_factory=dict)
    envelope_is_harmonic: bool = False
    aw_products_cv: float = 0.0

    # Stage 4
    w0: float = 0.0
    w0_confidence: str = ''
    w0_methods: dict = field(default_factory=dict)

    # Stage 5
    group_boundaries: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    groups: list = field(default_factory=list)

    # Stage 7
    nominal_lines: list = field(default_factory=list)
    nominal_df: Optional[pd.DataFrame] = None

    # Diagnostics
    diagnostics: dict = field(default_factory=dict)

    def summary(self):
        """Print a human-readable summary."""
        print(f"=== Nominal Model: {self.label} ===")
        print(f"Data: {self.n_samples} samples, {self.years:.1f} years, fs={self.fs:.1f}")
        print(f"Peaks: {len(self.peak_freqs)}, Troughs: {len(self.trough_freqs)}")
        print(f"Envelope: upper R²={self.envelope_upper.get('r_squared', 0):.3f}, "
              f"harmonic={self.envelope_is_harmonic}")
        print(f"Fundamental: w0={self.w0:.4f} rad/yr "
              f"(T={2*np.pi/self.w0:.1f} yr), confidence={self.w0_confidence}")
        print(f"Groups: {len(self.groups)}")
        print(f"Nominal lines: {len(self.nominal_lines)}")
        if self.nominal_df is not None:
            print(f"\n{self.nominal_df.to_string(index=False)}")


# =============================================================================
# Stage 0: Data Loading
# =============================================================================

def load_data(symbol='djia', freq='weekly', start=None, end=None,
              base_dir=None):
    """
    Load price data and compute sampling rate.

    Parameters
    ----------
    symbol : str
        'djia' or 'spx'
    freq : str
        'weekly' or 'daily'
    start, end : str or None
        Date range (e.g., '1921-04-29')
    base_dir : str or None
        Project root directory. Auto-detected if None.

    Returns
    -------
    dict with 'close', 'dates', 'fs', 'n_samples', 'years', 'label'
    """
    import os
    if base_dir is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

    freq_code = 'w' if freq == 'weekly' else 'd'
    symbol_map = {'djia': '^dji', 'spx': '^spx'}
    sym = symbol_map.get(symbol, symbol)
    path = os.path.join(base_dir, f'data/raw/{sym}_{freq_code}.csv')

    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    if start is not None:
        df = df[df.Date >= start]
    if end is not None:
        df = df[df.Date <= end]
    df = df.dropna(subset=['Close']).reset_index(drop=True)

    close = df['Close'].values.astype(np.float64)
    dates = df['Date']

    duration_days = (dates.iloc[-1] - dates.iloc[0]).days
    duration_years = duration_days / 365.25
    fs = len(close) / duration_years

    label = (f"{symbol.upper()} {freq} "
             f"{dates.iloc[0].strftime('%Y')}-{dates.iloc[-1].strftime('%Y')}")

    return {
        'close': close, 'dates': dates, 'fs': fs,
        'n_samples': len(close), 'years': duration_years, 'label': label,
    }


# =============================================================================
# Stage 1: Fourier-Lanczos Spectrum
# =============================================================================

def compute_spectrum(close, fs):
    """Compute Fourier-Lanczos spectrum, return omega_yr and amplitude."""
    w, wRad, cosprt, sinprt, amp, phRad, phGrad = lanczos_spectrum(close, 1, fs)
    omega_yr = w * fs
    return omega_yr, amp


# =============================================================================
# Stage 2: Peak and Trough Detection
# =============================================================================

def detect_features(omega_yr, amp, fs, prominence_frac=0.01, min_distance=2):
    """
    Detect peaks and troughs in spectrum.

    Uses 1% prominence threshold (PRD spec) and min_distance=2
    for fine harmonic-level detection.
    """
    # Frequency range depends on data type
    max_freq = 40.0 if fs > 100 else 13.0
    freq_range = (0.3, max_freq)

    # Restrict to analysis range for prominence calculation
    mask = (omega_yr >= 0.3) & (omega_yr <= max_freq)
    amp_range = amp[mask].max() - amp[mask].min()
    prominence = prominence_frac * amp_range

    peak_idx, peak_freqs, peak_amps = find_spectral_peaks(
        amp, omega_yr, min_distance=min_distance,
        prominence=prominence, freq_range=freq_range
    )

    trough_idx, trough_freqs, trough_amps = find_spectral_troughs(
        amp, omega_yr, min_distance=min_distance,
        prominence=prominence * 0.5, freq_range=freq_range
    )

    # Adaptive: if too few peaks, relax prominence
    if len(peak_freqs) < 5:
        prominence *= 0.5
        peak_idx, peak_freqs, peak_amps = find_spectral_peaks(
            amp, omega_yr, min_distance=min_distance,
            prominence=prominence, freq_range=freq_range
        )

    return peak_freqs, peak_amps, trough_freqs, trough_amps


# =============================================================================
# Stage 3: Envelope Fitting and Validation
# =============================================================================

def fit_and_validate_envelope(peak_freqs, peak_amps, trough_freqs, trough_amps):
    """
    Fit upper/lower envelopes and test for harmonic structure.

    Returns
    -------
    dict with 'upper', 'lower', 'is_harmonic', 'aw_cv'
    """
    # Upper envelope (peaks)
    upper_fixed = fit_upper_envelope(peak_freqs, peak_amps, fixed_slope=-1.0)
    upper_free = fit_upper_envelope(peak_freqs, peak_amps, fixed_slope=None)

    # Lower envelope (troughs)
    try:
        lower_fixed = fit_lower_envelope(trough_freqs, trough_amps, fixed_slope=-1.0)
    except ValueError:
        lower_fixed = {'k': 0, 'alpha': -1, 'r_squared': 0, 'fit_method': 'failed'}

    # Test 1: R² > 0.85 with fixed alpha=-1 suggests harmonic structure
    r2_fixed = upper_fixed['r_squared']

    # Test 2: A*w products — if CV < 20%, confirms equal rate of change
    aw_products = peak_amps * peak_freqs
    aw_cv = np.std(aw_products) / np.mean(aw_products) if np.mean(aw_products) > 0 else 1.0

    is_harmonic = (r2_fixed > 0.85) or (aw_cv < 0.20)

    return {
        'upper': upper_fixed,
        'upper_free': upper_free,
        'lower': lower_fixed,
        'is_harmonic': is_harmonic,
        'aw_cv': float(aw_cv),
        'r2_fixed': float(r2_fixed),
    }


# =============================================================================
# Stage 4: Fundamental Frequency Estimation (3-method consensus)
# =============================================================================

def _estimate_w0_fine_structure(peak_freqs, freq_band=(7.0, 13.0)):
    """Method 4A: Fine structure spacing from peaks in narrow band.

    Handles sub-harmonic degeneracy: if measured spacing is ~2×w0 because
    odd harmonics are missing, divides by 2 to get the fundamental.
    """
    mask = (peak_freqs >= freq_band[0]) & (peak_freqs <= freq_band[1])
    band_peaks = peak_freqs[mask]
    if len(band_peaks) < 5:
        return np.nan, 0.0

    spacings, mean_sp, std_sp = detect_fine_structure_spacing(band_peaks, max_spacing=1.5)
    if np.isnan(mean_sp) or len(spacings) < 3:
        return np.nan, 0.0

    # Sub-harmonic correction: if spacing > 0.50, it's likely 2*w0 or 3*w0
    w0_est = mean_sp
    for divisor in [2, 3]:
        candidate = mean_sp / divisor
        if 0.30 <= candidate <= 0.45:
            w0_est = candidate
            break

    cv = std_sp / mean_sp if mean_sp > 0 else 1.0
    confidence = max(0, 1.0 - cv)
    return float(w0_est), float(confidence)


def _estimate_w0_trough_mapping(trough_freqs, search_range=(0.30, 0.45)):
    """Method 4B: Map troughs to half-integer harmonics, grid search w0."""
    if len(trough_freqs) < 3:
        return np.nan, 0.0

    w0_grid = np.linspace(search_range[0], search_range[1], 300)
    best_residual = np.inf
    best_w0 = np.nan

    for w0 in w0_grid:
        # Troughs should fall at half-integers: N = w/w0 should be near n+0.5
        N_vals = trough_freqs / w0
        # Distance to nearest half-integer
        half_int_dist = np.abs(N_vals - np.round(N_vals * 2) / 2)
        residual = np.mean(half_int_dist ** 2)
        if residual < best_residual:
            best_residual = residual
            best_w0 = w0

    confidence = max(0, 1.0 - best_residual * 10)
    return float(best_w0), float(confidence)


def _estimate_w0_peak_mapping(peak_freqs, peak_amps, search_range=(0.30, 0.45)):
    """Method 4C: Map peaks to integer harmonics via least-squares fit.

    For each candidate w0, assign N=round(freq/w0), then fit omega=w0*N
    via weighted least-squares. The w0 with lowest residual wins.
    """
    if len(peak_freqs) < 5:
        return np.nan, 0.0

    w0_grid = np.linspace(search_range[0], search_range[1], 500)
    best_residual = np.inf
    best_w0 = np.nan

    # Normalize weights by amplitude
    weights = peak_amps / np.sum(peak_amps)

    for w0_candidate in w0_grid:
        N_vals = np.array([max(1, round(f / w0_candidate)) for f in peak_freqs])
        # Least-squares fit: w0_fit = sum(w*N*freq) / sum(w*N^2)
        wN = weights * N_vals
        w0_fit = np.sum(wN * peak_freqs) / np.sum(wN * N_vals)
        residuals = peak_freqs - w0_fit * N_vals
        residual = np.sum(weights * residuals ** 2)
        if residual < best_residual:
            best_residual = residual
            best_w0 = w0_fit

    # Constrain to search range
    best_w0 = np.clip(best_w0, search_range[0], search_range[1])
    confidence = max(0, 1.0 - best_residual * 50)
    return float(best_w0), float(confidence)


def estimate_fundamental(peak_freqs, peak_amps, trough_freqs):
    """
    Estimate fundamental frequency w0 using 3 methods with consensus.

    Returns
    -------
    w0 : float
        Best estimate of fundamental spacing (rad/yr)
    confidence : str
        'high', 'medium', or 'low'
    methods : dict
        Results from each method
    """
    w0_fine, conf_fine = _estimate_w0_fine_structure(peak_freqs)
    w0_trough, conf_trough = _estimate_w0_trough_mapping(trough_freqs)
    w0_peak, conf_peak = _estimate_w0_peak_mapping(peak_freqs, peak_amps)

    methods = {
        'fine_structure': {'w0': w0_fine, 'confidence': conf_fine},
        'trough_mapping': {'w0': w0_trough, 'confidence': conf_trough},
        'peak_mapping': {'w0': w0_peak, 'confidence': conf_peak},
    }

    # Collect valid estimates
    estimates = []
    confs = []
    for name, m in methods.items():
        if not np.isnan(m['w0']) and 0.25 < m['w0'] < 0.50:
            estimates.append(m['w0'])
            confs.append(m['confidence'])

    if len(estimates) == 0:
        # Fallback: use Hurst's known value
        return 0.3676, 'fallback', methods

    estimates = np.array(estimates)
    confs = np.array(confs)

    # Check consensus: all within 10% of each other?
    spread = (np.max(estimates) - np.min(estimates)) / np.mean(estimates)

    if spread < 0.10 and len(estimates) >= 2:
        # Good consensus — use weighted mean
        w0 = float(np.average(estimates, weights=confs))
        confidence = 'high'
    elif len(estimates) >= 2:
        # Partial agreement — use the one with best confidence
        best_idx = np.argmax(confs)
        w0 = float(estimates[best_idx])
        confidence = 'medium'
    else:
        w0 = float(estimates[0])
        confidence = 'low'

    return w0, confidence, methods


# =============================================================================
# Stage 5: Group Boundary Detection (Trough Dividers)
# =============================================================================

# Hurst's nominal group definitions by harmonic index range
HURST_GROUPS = [
    {'name': 'Trend',     'N_range': (1, 2),    'period_label': '18yr + 9yr'},
    {'name': '54-month',  'N_range': (3, 5),    'period_label': '4.5yr'},
    {'name': '18-month',  'N_range': (6, 10),   'period_label': '18mo'},
    {'name': '40-week',   'N_range': (11, 18),  'period_label': '40wk'},
    {'name': '20-week',   'N_range': (19, 34),  'period_label': '20wk'},
    {'name': '10-week',   'N_range': (35, 68),  'period_label': '10wk'},
    {'name': '5-week',    'N_range': (69, 136), 'period_label': '5wk'},
]


def define_groups(trough_freqs, trough_amps, w0, lower_envelope_k=None):
    """
    Define spectral groups from trough dividers.

    Deep troughs (below lower envelope) mark group boundaries.
    Each group spans a range of harmonic indices N.
    """
    if len(trough_freqs) < 2:
        return np.array([]), HURST_GROUPS

    # Map troughs to harmonic indices
    N_troughs = trough_freqs / w0

    # Identify "deep" troughs: amplitude below median trough amplitude
    median_amp = np.median(trough_amps)
    deep_mask = trough_amps <= median_amp
    deep_trough_freqs = trough_freqs[deep_mask]
    deep_N = N_troughs[deep_mask]

    if len(deep_trough_freqs) < 2:
        deep_trough_freqs = trough_freqs
        deep_N = N_troughs

    # Build group definitions from observed trough positions
    boundaries = np.sort(deep_trough_freqs)
    groups = []

    # Map each deep trough to a group boundary
    for i, trough_f in enumerate(boundaries):
        N = trough_f / w0
        # Find which Hurst group boundary this matches
        for hg in HURST_GROUPS:
            n_lo, n_hi = hg['N_range']
            if n_lo <= round(N) <= n_hi + 2:
                groups.append({
                    'name': hg['name'],
                    'N_range': hg['N_range'],
                    'period_label': hg['period_label'],
                    'boundary_freq': float(trough_f),
                    'boundary_N': float(N),
                })
                break

    # If no groups matched, use Hurst's defaults
    if not groups:
        groups = HURST_GROUPS

    return boundaries, groups


# =============================================================================
# Stage 7: Line Extraction (from peaks + w0)
# =============================================================================

def extract_nominal_lines(peak_freqs, peak_amps, w0, max_N=None, fs=52):
    """
    Map detected peaks to integer harmonics and build the nominal model.

    Parameters
    ----------
    peak_freqs : array
        Detected peak frequencies (rad/yr)
    peak_amps : array
        Amplitudes at peaks
    w0 : float
        Fundamental spacing (rad/yr)
    max_N : int or None
        Maximum harmonic number. Auto-computed from fs if None.
    fs : float
        Sampling rate

    Returns
    -------
    lines : list of dict
        Each dict has N, frequency, period_yr, period_wk, amplitude, confidence
    df : pd.DataFrame
        Tabular representation
    """
    if max_N is None:
        nyquist = np.pi * fs
        max_N = int(nyquist / w0)

    # Map each peak to nearest integer harmonic
    assigned = {}  # N -> (freq, amp, error)
    for freq, amp_val in zip(peak_freqs, peak_amps):
        N = max(1, round(freq / w0))
        if N > max_N:
            continue
        error = abs(freq - N * w0)
        # Only assign if within 40% of w0 spacing
        if error > w0 * 0.4:
            continue
        # Keep the strongest peak for each N
        if N not in assigned or amp_val > assigned[N][1]:
            assigned[N] = (freq, amp_val, error)

    # Build line list
    lines = []
    for N in sorted(assigned.keys()):
        freq, amp_val, error = assigned[N]
        period_yr = 2 * np.pi / (N * w0)
        period_wk = period_yr * 52

        # Confidence based on how close to expected and amplitude
        rel_error = error / w0
        if rel_error < 0.1:
            conf = 'high'
        elif rel_error < 0.25:
            conf = 'medium'
        else:
            conf = 'low'

        # Find which group this harmonic belongs to
        group = 'unknown'
        for hg in HURST_GROUPS:
            n_lo, n_hi = hg['N_range']
            if n_lo <= N <= n_hi:
                group = hg['name']
                break

        lines.append({
            'N': N,
            'frequency': float(N * w0),  # Use nominal frequency, not measured
            'measured_freq': float(freq),
            'period_yr': float(period_yr),
            'period_wk': float(period_wk),
            'period_months': float(period_yr * 12),
            'amplitude': float(amp_val),
            'error': float(error),
            'confidence': conf,
            'group': group,
        })

    # Build DataFrame
    if lines:
        df = pd.DataFrame(lines)
        df = df[['N', 'frequency', 'period_yr', 'period_wk', 'period_months',
                 'amplitude', 'group', 'confidence']]
    else:
        df = pd.DataFrame()

    return lines, df


# =============================================================================
# Main Pipeline Function
# =============================================================================

def derive_nominal_model(symbol='djia', freq='weekly',
                         start='1921-04-29', end='1965-05-21',
                         prominence_frac=0.01, min_distance=2,
                         w0_override=None, max_N=None,
                         base_dir=None, verbose=True):
    """
    Full automated pipeline from raw data to nominal model.

    Parameters
    ----------
    symbol : str
        'djia' or 'spx'
    freq : str
        'weekly' or 'daily'
    start, end : str
        Date range
    prominence_frac : float
        Peak detection prominence as fraction of amplitude range (default 1%)
    min_distance : int
        Minimum distance between spectral peaks (default 2 for fine detection)
    w0_override : float or None
        If provided, skip w0 estimation and use this value
    max_N : int or None
        Maximum harmonic number (auto if None)
    base_dir : str or None
        Project root
    verbose : bool
        Print progress

    Returns
    -------
    NominalModelResult
        Complete pipeline output
    """
    result = NominalModelResult()

    # --- Stage 0: Load Data ---
    if verbose:
        print("Stage 0: Loading data...")
    data = load_data(symbol, freq, start, end, base_dir)
    result.close = data['close']
    result.dates = data['dates']
    result.fs = data['fs']
    result.n_samples = data['n_samples']
    result.years = data['years']
    result.label = data['label']

    # --- Stage 1: Spectrum ---
    if verbose:
        print(f"Stage 1: Computing Lanczos spectrum ({result.n_samples} samples, "
              f"fs={result.fs:.1f})...")
    result.omega_yr, result.amp = compute_spectrum(result.close, result.fs)

    # --- Stage 2: Peaks and Troughs ---
    if verbose:
        print("Stage 2: Detecting peaks and troughs...")
    result.peak_freqs, result.peak_amps, result.trough_freqs, result.trough_amps = \
        detect_features(result.omega_yr, result.amp, result.fs,
                        prominence_frac=prominence_frac,
                        min_distance=min_distance)
    if verbose:
        print(f"  Found {len(result.peak_freqs)} peaks, "
              f"{len(result.trough_freqs)} troughs")

    # --- Stage 3: Envelope ---
    if verbose:
        print("Stage 3: Fitting envelopes...")
    env = fit_and_validate_envelope(
        result.peak_freqs, result.peak_amps,
        result.trough_freqs, result.trough_amps
    )
    result.envelope_upper = env['upper']
    result.envelope_lower = env['lower']
    result.envelope_is_harmonic = env['is_harmonic']
    result.aw_products_cv = env['aw_cv']
    if verbose:
        print(f"  Upper R²={env['r2_fixed']:.3f}, A*w CV={env['aw_cv']:.1%}, "
              f"harmonic={env['is_harmonic']}")

    # --- Stage 4: Fundamental w0 ---
    if w0_override is not None:
        result.w0 = w0_override
        result.w0_confidence = 'override'
        result.w0_methods = {}
        if verbose:
            print(f"Stage 4: Using override w0={w0_override:.4f} rad/yr")
    else:
        if verbose:
            print("Stage 4: Estimating fundamental w0...")
        result.w0, result.w0_confidence, result.w0_methods = estimate_fundamental(
            result.peak_freqs, result.peak_amps, result.trough_freqs
        )
        if verbose:
            print(f"  w0={result.w0:.4f} rad/yr "
                  f"(T={2*np.pi/result.w0:.1f} yr), "
                  f"confidence={result.w0_confidence}")
            for name, m in result.w0_methods.items():
                w0v = m['w0']
                w0str = f"{w0v:.4f}" if not np.isnan(w0v) else "N/A"
                print(f"    {name}: w0={w0str}, conf={m['confidence']:.2f}")

    # --- Stage 5: Groups ---
    if verbose:
        print("Stage 5: Defining groups from trough dividers...")
    result.group_boundaries, result.groups = define_groups(
        result.trough_freqs, result.trough_amps, result.w0,
        lower_envelope_k=result.envelope_lower.get('k')
    )
    if verbose:
        print(f"  {len(result.group_boundaries)} boundaries, "
              f"{len(result.groups)} groups")

    # --- Stage 7: Line Extraction ---
    if verbose:
        print("Stage 7: Extracting nominal lines...")
    result.nominal_lines, result.nominal_df = extract_nominal_lines(
        result.peak_freqs, result.peak_amps, result.w0,
        max_N=max_N, fs=result.fs
    )
    if verbose:
        print(f"  {len(result.nominal_lines)} lines extracted")

    if verbose:
        print("\nPipeline complete.")
        result.summary()

    return result
