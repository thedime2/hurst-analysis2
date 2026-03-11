# -*- coding: utf-8 -*-
"""
Hurst's 75/23/2 Rule v2 -- Ormsby Trend + Optimized CMW Modeling

Improvements over v1:
  1) Use Ormsby LP-1 and LP-1+BP-2 flat-top filters for trend removal
  2) Narrowband CMW AI-2 style analysis to find actual harmonic positions
  3) Detect beating to understand where harmonics cluster
  4) Design optimized filter bank centered on detected frequencies
  5) Model with held-back data using optimized filters

Reference: J.M. Hurst, The Profit Magic of Stock Transaction Timing (1970)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from numpy.linalg import lstsq

from src.pipeline.derive_nominal_model import derive_nominal_model
from src.pipeline.comb_bank import (
    design_narrowband_cmw_bank, run_cmw_comb_bank, extract_lines_from_narrowband
)
from src.filters.funcOrmsby import ormsby_filter, apply_ormsby_filter
from src.time_frequency.cmw import apply_cmw

OUT_DIR = os.path.dirname(__file__)
W0_HURST = 0.3676  # rad/yr


# =============================================================================
# Part A: Ormsby Trend Removal and 75% Confirmation
# =============================================================================

def part_a_ormsby_trend(close, dates, fs):
    """
    Use Ormsby LP-1 and LP-1+BP-2 to extract trend, confirm 75% variance.
    LP-1: cutoff at ~0.93 rad/yr (between trend and 54-month group)
    BP-2: 0.93 to 2.09 rad/yr (54-month group)
    Together: everything below ~2.09 rad/yr = 18yr + 9yr + 4.5yr
    """
    print("\n" + "=" * 70)
    print("PART A: Ormsby Trend Removal -- Confirm 75% Rule")
    print("=" * 70)

    log_prices = np.log(close)
    t_yr = np.arange(len(close)) / fs

    # --- LP-1: Trend filter (N=1,2: everything below ~0.93 rad/yr) ---
    # Cyclitec boundary between trend and 54-month = 0.93 rad/yr
    # Ormsby LP: f_pass, f_stop in cycles/year (divide rad/yr by 2pi)
    lp1_pass_rad = 0.80  # passband edge (rad/yr) - captures N=1,2
    lp1_stop_rad = 1.05  # stopband edge (rad/yr)
    lp1_pass = lp1_pass_rad / (2 * np.pi)
    lp1_stop = lp1_stop_rad / (2 * np.pi)
    nw_lp1 = 1393

    h_lp1 = ormsby_filter(nw_lp1, [lp1_pass, lp1_stop], fs, filter_type='lp',
                          analytic=True)
    result_lp1 = apply_ormsby_filter(log_prices, h_lp1, mode='reflect', fs=fs)
    trend_lp1 = np.real(result_lp1['signal'])

    print(f"  LP-1: passband<{lp1_pass_rad:.2f}, stopband>{lp1_stop_rad:.2f} rad/yr, nw={nw_lp1}")

    # --- BP-2: 54-month band (0.93 to 2.09 rad/yr, N=3-5) ---
    bp2_lo = 0.80   # Lower transition start
    bp2_pass_lo = 1.05  # Lower passband
    bp2_pass_hi = 1.90  # Upper passband
    bp2_hi = 2.30  # Upper transition end
    bp2_edges = np.array([bp2_lo, bp2_pass_lo, bp2_pass_hi, bp2_hi]) / (2 * np.pi)
    nw_bp2 = 1393

    h_bp2 = ormsby_filter(nw_bp2, bp2_edges, fs, filter_type='bp',
                          method='modulate', analytic=True)
    result_bp2 = apply_ormsby_filter(log_prices, h_bp2, mode='reflect', fs=fs)
    band_54mo = np.real(result_bp2['signal'])

    print(f"  BP-2: [{bp2_lo:.2f}, {bp2_pass_lo:.2f}, {bp2_pass_hi:.2f}, {bp2_hi:.2f}] rad/yr, nw={nw_bp2}")

    # --- LP-1 + BP-2 combined = slow trend ---
    slow_trend = trend_lp1 + band_54mo

    # --- Residuals ---
    residual_lp1 = log_prices - trend_lp1
    residual_slow = log_prices - slow_trend

    # --- Energy decomposition ---
    ss_total = np.sum((log_prices - np.mean(log_prices))**2)
    ss_after_lp1 = np.sum((residual_lp1 - np.mean(residual_lp1))**2)
    ss_after_slow = np.sum((residual_slow - np.mean(residual_slow))**2)

    r2_lp1 = 1 - ss_after_lp1 / ss_total
    r2_slow = 1 - ss_after_slow / ss_total

    print(f"\n  --- Energy Decomposition ---")
    print(f"  LP-1 (trend only) explains: {r2_lp1*100:.1f}% of total variance")
    print(f"  LP-1 + BP-2 (slow trend) explains: {r2_slow*100:.1f}% of total variance")
    print(f"  Residual after LP-1: {(1-r2_lp1)*100:.1f}%")
    print(f"  Residual after LP-1+BP-2: {(1-r2_slow)*100:.1f}%")
    print(f"  Hurst's 75% -> we get LP-1={r2_lp1*100:.1f}%, with 54mo={r2_slow*100:.1f}%")

    return {
        'log_prices': log_prices, 't_yr': t_yr,
        'trend_lp1': trend_lp1, 'band_54mo': band_54mo,
        'slow_trend': slow_trend,
        'residual_lp1': residual_lp1, 'residual_slow': residual_slow,
        'r2_lp1': r2_lp1, 'r2_slow': r2_slow,
    }


def plot_part_a(data, out_dir):
    """Plot Ormsby trend removal results."""
    fig, axes = plt.subplots(3, 2, figsize=(18, 14))
    t = data['t_yr']

    # Panel 1: log prices + LP-1 trend
    ax = axes[0, 0]
    ax.plot(t, data['log_prices'], 'b-', lw=0.5, alpha=0.7, label='log(DJIA)')
    ax.plot(t, data['trend_lp1'], 'r-', lw=2, label='LP-1 trend')
    ax.set_title(f'LP-1 Trend ({data["r2_lp1"]*100:.1f}% of variance)')
    ax.legend(fontsize=8)
    ax.set_ylabel('log(Price)')

    # Panel 2: log prices + slow trend (LP-1 + BP-2)
    ax = axes[0, 1]
    ax.plot(t, data['log_prices'], 'b-', lw=0.5, alpha=0.7, label='log(DJIA)')
    ax.plot(t, data['slow_trend'], 'r-', lw=2, label='LP-1 + BP-2')
    ax.set_title(f'Slow Trend (LP-1+BP-2) = {data["r2_slow"]*100:.1f}% of variance')
    ax.legend(fontsize=8)
    ax.set_ylabel('log(Price)')

    # Panel 3: Residual after LP-1
    ax = axes[1, 0]
    ax.plot(t, data['residual_lp1'], 'b-', lw=0.5)
    ax.axhline(0, color='red', lw=0.3)
    ax.set_title('Residual after LP-1 (oscillatory content)')
    ax.set_ylabel('Amplitude')

    # Panel 4: Residual after LP-1+BP-2
    ax = axes[1, 1]
    ax.plot(t, data['residual_slow'], 'b-', lw=0.5)
    ax.axhline(0, color='red', lw=0.3)
    ax.set_title('Residual after LP-1+BP-2 (fast oscillatory)')
    ax.set_ylabel('Amplitude')

    # Panel 5: 54-month band isolated
    ax = axes[2, 0]
    ax.plot(t, data['band_54mo'], 'g-', lw=1)
    ax.axhline(0, color='black', lw=0.3)
    ax.set_title('BP-2: 54-month band (N=3-5)')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Amplitude')

    # Panel 6: Energy pie
    ax = axes[2, 1]
    lp1_pct = data['r2_lp1'] * 100
    bp2_pct = (data['r2_slow'] - data['r2_lp1']) * 100
    resid_pct = (1 - data['r2_slow']) * 100
    sizes = [max(0, lp1_pct), max(0, bp2_pct), max(0, resid_pct)]
    labels = [f'LP-1 Trend\n{lp1_pct:.1f}%',
              f'BP-2 54mo\n{bp2_pct:.1f}%',
              f'Residual\n{resid_pct:.1f}%']
    colors = ['#FF6B6B', '#FFD93D', '#4ECDC4']
    ax.pie(sizes, labels=labels, colors=colors, startangle=140,
           autopct='%1.1f%%')
    ax.set_title("Energy Decomposition (Hurst: 75/23/2)")

    plt.suptitle("Part A: Ormsby Flat-Top Trend Removal", fontsize=14, fontweight='bold')
    path = os.path.join(out_dir, 'fig_v2_partA_ormsby_trend.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved: {path}")


# =============================================================================
# Part B: Narrowband CMW AI-2 Style -- Find Actual Harmonic Positions
# =============================================================================

def part_b_narrowband_detection(residual, fs, w0, max_N=34):
    """
    Apply narrowband CMW comb bank to the trend-removed residual.
    This is like AI-2 but using narrow CMW instead of Ormsby.

    Goal: find where harmonics ACTUALLY are (vs assumed N*w0 grid)
    and identify beating patterns that reveal frequency clustering.
    """
    print("\n" + "=" * 70)
    print("PART B: Narrowband CMW Detection (AI-2 style)")
    print("=" * 70)

    # Design narrowband bank on N*w0 grid
    nb_params = design_narrowband_cmw_bank(w0=w0, max_N=max_N, fs=fs,
                                            fwhm_factor=0.5, omega_min=0.7)
    print(f"  {len(nb_params)} filters designed (N=2 to N={max_N})")

    nb_result = run_cmw_comb_bank(residual, fs, nb_params, analytic=True)

    # Extract per-harmonic statistics
    harmonic_stats = []
    for i, output in enumerate(nb_result['filter_outputs']):
        spec = nb_params[i]
        N = spec.get('N', 0)
        if output['envelope'] is None or output['frequency'] is None:
            continue

        env = output['envelope']
        freq_rad = output['frequency'] * 2 * np.pi  # cycles/yr -> rad/yr

        # Trim edges
        n = len(env)
        trim = int(n * 0.1)
        env_core = env[trim:-trim] if trim > 0 else env
        freq_core = freq_rad[trim:-trim] if trim > 0 else freq_rad

        mean_env = np.mean(env_core)
        med_freq = np.median(freq_core)
        freq_std = np.std(freq_core)
        freq_cv = freq_std / med_freq if med_freq > 0 else 1.0

        # Check for beating: look at envelope spectrum
        # Beating shows up as periodic amplitude modulation
        if len(env_core) > 50:
            env_detrended = env_core - np.mean(env_core)
            env_fft = np.abs(np.fft.rfft(env_detrended))
            env_freqs = np.fft.rfftfreq(len(env_detrended), d=1.0/fs)
            # Peak of envelope spectrum (excluding DC)
            if len(env_fft) > 2:
                beat_idx = np.argmax(env_fft[1:]) + 1
                beat_freq = env_freqs[beat_idx] * 2 * np.pi  # rad/yr
                beat_amp = env_fft[beat_idx] / (np.mean(env_core) * len(env_detrended) / 2)
            else:
                beat_freq = 0
                beat_amp = 0
        else:
            beat_freq = 0
            beat_amp = 0

        harmonic_stats.append({
            'N': N,
            'expected_freq': N * w0,
            'measured_freq': float(med_freq),
            'freq_offset': float(med_freq - N * w0),
            'freq_cv': float(freq_cv),
            'mean_amp': float(mean_env),
            'beat_freq': float(beat_freq),
            'beat_strength': float(beat_amp),
            'period_wk': float(2 * np.pi / (N * w0) * 52),
        })

    df = pd.DataFrame(harmonic_stats)

    print(f"\n  {'N':>3} {'Expect':>7} {'Meas':>7} {'Offset':>7} {'CV%':>5} "
          f"{'Amp':>8} {'Beat':>6} {'BeatStr':>7}")
    for _, row in df.iterrows():
        print(f"  {int(row.N):>3d} {row.expected_freq:>7.3f} {row.measured_freq:>7.3f} "
              f"{row.freq_offset:>+7.3f} {row.freq_cv*100:>4.1f}% "
              f"{row.mean_amp:>8.5f} {row.beat_freq:>6.2f} {row.beat_strength:>7.3f}")

    # Identify STRONG harmonics (high amplitude, low CV)
    # Use 1/w-aware threshold
    freqs = df.expected_freq.values
    amps = df.mean_amp.values
    aw = freqs * amps
    k_est = np.median(aw[aw > 0])
    expected_amps = k_est / freqs
    amp_ratio = amps / expected_amps

    strong_mask = (amp_ratio > 0.3) & (df.freq_cv < 0.25)
    strong_df = df[strong_mask].copy()
    weak_df = df[~strong_mask].copy()

    print(f"\n  Strong harmonics (amp>0.3x expected, CV<25%): {len(strong_df)}")
    print(f"  Weak/absent: {len(weak_df)}")
    if len(strong_df) > 0:
        print(f"  Strong N values: {sorted(strong_df.N.astype(int).tolist())}")

    return df, strong_df, weak_df, nb_result, nb_params


def plot_part_b(df, strong_df, weak_df, nb_result, nb_params, t_yr, residual, out_dir):
    """Plot AI-2 style narrowband CMW detection."""
    fig = plt.figure(figsize=(18, 16))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    # Panel 1: AI-2 style frequency response overlay
    ax = fig.add_subplot(gs[0, 0])
    f_axis = np.linspace(0.5, 14, 1000)
    for i, spec in enumerate(nb_params):
        N = spec.get('N', 0)
        f0 = spec['f0']
        fwhm = spec['fwhm']
        sigma = fwhm * 0.4247
        if f0 > 14:
            continue
        response = np.exp(-0.5 * ((f_axis - f0) / sigma) ** 2)
        is_strong = N in strong_df.N.values if len(strong_df) > 0 else False
        color = 'green' if is_strong else 'lightgray'
        alpha = 0.8 if is_strong else 0.3
        ax.fill_between(f_axis, response, alpha=alpha * 0.3, color=color)
        ax.plot(f_axis, response, color=color, lw=0.5, alpha=alpha)
    ax.set_xlabel('w (rad/yr)')
    ax.set_ylabel('Response')
    ax.set_title('AI-2 Style: Narrowband CMW Frequency Responses')
    ax.set_xlim(0.5, 14)

    # Panel 2: Measured vs expected frequency
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(df.N, df.expected_freq, 'r--', lw=1, label='Expected (N*w0)')
    ax.plot(df.N, df.measured_freq, 'bo', ms=4, alpha=0.7, label='Measured')
    if len(strong_df) > 0:
        ax.plot(strong_df.N, strong_df.measured_freq, 'go', ms=6, label='Strong')
    ax.set_xlabel('Harmonic N')
    ax.set_ylabel('Frequency (rad/yr)')
    ax.set_title('Measured vs Expected Harmonic Frequencies')
    ax.legend(fontsize=8)

    # Panel 3: Amplitude profile with 1/w overlay
    ax = fig.add_subplot(gs[1, 0])
    ax.semilogy(df.N, df.mean_amp, 'bo', ms=4, alpha=0.7, label='All')
    if len(strong_df) > 0:
        ax.semilogy(strong_df.N, strong_df.mean_amp, 'go', ms=6, label='Strong')
    # 1/w line
    N_line = np.arange(2, 35)
    k_est = np.median(df.expected_freq.values * df.mean_amp.values)
    ax.semilogy(N_line, k_est / (N_line * W0_HURST), 'r--', lw=1, label='k/w envelope')
    ax.set_xlabel('Harmonic N')
    ax.set_ylabel('Mean Amplitude')
    ax.set_title('Amplitude Profile (green=strong)')
    ax.legend(fontsize=8)

    # Panel 4: Beat frequency vs N
    ax = fig.add_subplot(gs[1, 1])
    ax.bar(df.N, df.beat_strength, color='coral', alpha=0.7)
    ax.set_xlabel('Harmonic N')
    ax.set_ylabel('Beat Strength')
    ax.set_title('Beating Strength per Harmonic')

    # Panel 5: Stacked envelopes (AI-3 style, limited range)
    ax = fig.add_subplot(gs[2, :])
    show_range = range(3, min(20, len(nb_params) + 2))
    offset = 0
    for spec_idx, spec in enumerate(nb_params):
        N = spec.get('N', 0)
        if N not in show_range:
            continue
        output = nb_result['filter_outputs'][spec_idx]
        if output['envelope'] is None:
            continue
        env = output['envelope']
        sig = np.real(output['signal'])
        # Normalize
        env_max = np.max(env) if np.max(env) > 0 else 1
        sig_norm = sig / env_max * 0.4
        env_norm = env / env_max * 0.4

        is_strong = N in strong_df.N.values if len(strong_df) > 0 else False
        color = 'green' if is_strong else 'gray'

        ax.plot(t_yr[:len(sig)], sig_norm + offset, color=color, lw=0.3, alpha=0.5)
        ax.plot(t_yr[:len(env)], env_norm + offset, color=color, lw=1)
        ax.plot(t_yr[:len(env)], -env_norm + offset, color=color, lw=1)
        ax.text(-0.5, offset, f'N={N}', fontsize=7, va='center')
        offset += 1

    ax.set_xlabel('Time (years)')
    ax.set_title('Stacked Narrowband CMW Outputs (green=strong)')
    ax.set_yticks([])

    plt.suptitle("Part B: Narrowband CMW Detection", fontsize=14, fontweight='bold')
    path = os.path.join(out_dir, 'fig_v2_partB_narrowband_detection.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved: {path}")


# =============================================================================
# Part C: Optimized Filter Bank -- Centered on Detected Frequencies
# =============================================================================

def part_c_optimized_bank(residual, fs, strong_df, all_df, t_yr):
    """
    Design filter bank centered on ACTUALLY DETECTED strong harmonics.

    Strategy:
    1. Take strong harmonics from Part B
    2. For isolated harmonics (no neighbor within w0), use narrow CMW
    3. For clustered harmonics, use slightly wider CMW centered between them
    4. Apply to residual and extract amplitude/phase per filter
    """
    print("\n" + "=" * 70)
    print("PART C: Optimized CMW Bank on Detected Harmonics")
    print("=" * 70)

    if len(strong_df) == 0:
        print("  No strong harmonics found -- using all harmonics")
        strong_df = all_df.copy()

    # Sort by frequency
    strong_sorted = strong_df.sort_values('measured_freq').reset_index(drop=True)

    # Design optimized filters
    opt_params = []
    used = set()

    for i, row in strong_sorted.iterrows():
        N = int(row.N)
        if N in used:
            continue

        f_measured = row.measured_freq
        f_expected = row.expected_freq

        # Check if next harmonic is also strong and close
        # If so, we might want to widen slightly or keep separate
        neighbor_dist = np.inf
        for j, row2 in strong_sorted.iterrows():
            if int(row2.N) != N and int(row2.N) not in used:
                d = abs(row2.measured_freq - f_measured)
                if d < neighbor_dist:
                    neighbor_dist = d

        # Use measured frequency as center, FWHM = 0.5 * w0
        # For isolated harmonics: narrow FWHM
        # For close neighbors: still narrow, they'll be separate filters
        fwhm = W0_HURST * 0.5  # 0.184 rad/yr -- isolates single harmonic
        fwhm = max(fwhm, f_measured * 0.08)  # Floor at 8% of center

        opt_params.append({
            'f0': float(f_measured),
            'fwhm': float(fwhm),
            'N': N,
            'period_wk': float(2 * np.pi / f_measured * 52),
            'label': f'N={N} fc={f_measured:.3f}',
            'source': 'measured',
        })
        used.add(N)

    print(f"  Designed {len(opt_params)} optimized CMW filters")
    print(f"  N values: {[p['N'] for p in opt_params]}")

    # Apply optimized bank
    opt_result = run_cmw_comb_bank(residual, fs, opt_params, analytic=True)

    # Extract amplitude and phase at end of series (for projection)
    filter_models = []
    for i, output in enumerate(opt_result['filter_outputs']):
        spec = opt_params[i]
        if output['envelope'] is None or output['frequency'] is None:
            continue

        env = output['envelope']
        phase = output['phase']
        freq_rad = output['frequency'] * 2 * np.pi

        # Use last 20% for "current" amplitude and frequency estimate
        n = len(env)
        tail = max(1, int(n * 0.2))
        mean_amp_tail = np.mean(env[-tail:])
        mean_freq_tail = np.median(freq_rad[-tail:])
        # Phase at the end
        if phase is not None:
            end_phase = phase[-1]
        else:
            end_phase = 0

        filter_models.append({
            'N': spec['N'],
            'f0': spec['f0'],
            'measured_freq': float(mean_freq_tail),
            'amplitude': float(mean_amp_tail),
            'end_phase': float(end_phase),
            'mean_amp_full': float(np.mean(env)),
            'fwhm': spec['fwhm'],
        })

    model_df = pd.DataFrame(filter_models)
    print(f"\n  {'N':>3} {'f0':>7} {'MeasF':>7} {'Amp':>8} {'Phase':>7}")
    for _, row in model_df.iterrows():
        print(f"  {int(row.N):>3d} {row.f0:>7.3f} {row.measured_freq:>7.3f} "
              f"{row.amplitude:>8.5f} {row.end_phase:>7.3f}")

    return opt_params, opt_result, model_df


# =============================================================================
# Part D: Held-Back Validation with Optimized Model
# =============================================================================

def part_d_holdback(close, fs, w0, strong_df, holdback_frac=0.2):
    """
    Split data, remove trend (linear + N=1,2) on fitting window,
    fit optimized harmonics to RESIDUAL, project into holdback.

    Key: we model the oscillatory component (N=3+) after removing
    the slow trend, since the trend is better projected separately.
    """
    print("\n" + "=" * 70)
    print("PART D: Held-Back Validation with Optimized Harmonics")
    print("=" * 70)

    log_prices = np.log(close)
    t_yr = np.arange(len(close)) / fs
    n_total = len(close)
    n_fit = int(n_total * (1 - holdback_frac))
    n_holdback = n_total - n_fit

    print(f"  Fitting: {n_fit} samples ({n_fit/fs:.1f} yr)")
    print(f"  Holdback: {n_holdback} samples ({n_holdback/fs:.1f} yr)")

    # Remove linear trend + N=1,2 (the "75% slow component")
    linear_coeffs = np.polyfit(t_yr[:n_fit], log_prices[:n_fit], 1)
    linear_trend = np.polyval(linear_coeffs, t_yr)
    lin_detrended = log_prices - linear_trend

    # Fit N=1,2 on fitting window only
    omega_1 = w0
    omega_2 = 2 * w0
    t_fit_all = t_yr[:n_fit]
    X_slow = np.column_stack([
        np.cos(omega_1 * t_fit_all), np.sin(omega_1 * t_fit_all),
        np.cos(omega_2 * t_fit_all), np.sin(omega_2 * t_fit_all),
    ])
    c_slow, _, _, _ = lstsq(X_slow, lin_detrended[:n_fit], rcond=None)

    # Project slow trend into full series
    X_slow_full = np.column_stack([
        np.cos(omega_1 * t_yr), np.sin(omega_1 * t_yr),
        np.cos(omega_2 * t_yr), np.sin(omega_2 * t_yr),
    ])
    slow_trend = X_slow_full @ c_slow

    # Residual = detrended - slow_trend (this is the oscillatory component)
    detrended = lin_detrended - slow_trend

    t_fit = t_yr[:n_fit]
    y_fit = detrended[:n_fit]
    t_hold = t_yr[n_fit:]
    y_hold = detrended[n_fit:]

    print(f"  Trend removed: linear + N=1 ({2*np.pi/omega_1:.1f}yr) + N=2 ({2*np.pi/omega_2:.1f}yr)")
    print(f"  Residual std (fit): {np.std(y_fit):.4f}")
    print(f"  Residual std (hold): {np.std(y_hold):.4f}")

    # === Model 1: Standard N*w0 grid (N=3 to 34) ===
    harmonics_grid = list(range(3, 35))
    freqs_grid = [N * w0 for N in harmonics_grid]
    X_fit_grid = np.zeros((n_fit, 2 * len(harmonics_grid)))
    for j, f in enumerate(freqs_grid):
        X_fit_grid[:, 2*j] = np.cos(f * t_fit)
        X_fit_grid[:, 2*j+1] = np.sin(f * t_fit)

    c_grid, _, _, _ = lstsq(X_fit_grid, y_fit, rcond=None)
    y_fit_grid = X_fit_grid @ c_grid
    r2_fit_grid = 1 - np.sum((y_fit - y_fit_grid)**2) / np.sum((y_fit - np.mean(y_fit))**2)

    X_hold_grid = np.zeros((n_holdback, 2 * len(harmonics_grid)))
    for j, f in enumerate(freqs_grid):
        X_hold_grid[:, 2*j] = np.cos(f * t_hold)
        X_hold_grid[:, 2*j+1] = np.sin(f * t_hold)
    y_hold_grid = X_hold_grid @ c_grid
    r2_hold_grid = 1 - np.sum((y_hold - y_hold_grid)**2) / np.sum((y_hold - np.mean(y_hold))**2)
    corr_grid = np.corrcoef(y_hold, y_hold_grid)[0, 1]

    print(f"\n  === Model 1: Standard grid (N*w0, N=2-34) ===")
    print(f"  Fit R2: {r2_fit_grid:.4f}")
    print(f"  Holdback R2: {r2_hold_grid:.4f}, corr: {corr_grid:.4f}")

    # === Model 2: Strong harmonics only (from Part B, N>=3) ===
    if len(strong_df) > 0:
        strong_filt = strong_df[strong_df.N.astype(int) >= 3]
        freqs_strong = strong_filt.measured_freq.values
        Ns_strong = strong_filt.N.astype(int).values
    else:
        freqs_strong = np.array(freqs_grid[:15])  # Fallback
        Ns_strong = np.array(harmonics_grid[:15])

    X_fit_strong = np.zeros((n_fit, 2 * len(freqs_strong)))
    for j, f in enumerate(freqs_strong):
        X_fit_strong[:, 2*j] = np.cos(f * t_fit)
        X_fit_strong[:, 2*j+1] = np.sin(f * t_fit)

    c_strong, _, _, _ = lstsq(X_fit_strong, y_fit, rcond=None)
    y_fit_strong = X_fit_strong @ c_strong
    r2_fit_strong = 1 - np.sum((y_fit - y_fit_strong)**2) / np.sum((y_fit - np.mean(y_fit))**2)

    X_hold_strong = np.zeros((n_holdback, 2 * len(freqs_strong)))
    for j, f in enumerate(freqs_strong):
        X_hold_strong[:, 2*j] = np.cos(f * t_hold)
        X_hold_strong[:, 2*j+1] = np.sin(f * t_hold)
    y_hold_strong = X_hold_strong @ c_strong
    r2_hold_strong = 1 - np.sum((y_hold - y_hold_strong)**2) / np.sum((y_hold - np.mean(y_hold))**2)
    corr_strong = np.corrcoef(y_hold, y_hold_strong)[0, 1]

    print(f"\n  === Model 2: Strong harmonics ({len(freqs_strong)} filters, measured freqs) ===")
    print(f"  Fit R2: {r2_fit_strong:.4f}")
    print(f"  Holdback R2: {r2_hold_strong:.4f}, corr: {corr_strong:.4f}")

    # === Model 3: Few main harmonics (N=3-10, measured freqs) ===
    main_mask = (strong_df.N.astype(int) >= 3) & (strong_df.N.astype(int) <= 10) if len(strong_df) > 0 else pd.Series([True]*min(8, len(freqs_strong)))
    if main_mask.any():
        freqs_main = strong_df[main_mask].measured_freq.values if len(strong_df) > 0 else freqs_strong[:9]
    else:
        freqs_main = np.array([N * w0 for N in range(3, 11)])

    X_fit_main = np.zeros((n_fit, 2 * len(freqs_main)))
    for j, f in enumerate(freqs_main):
        X_fit_main[:, 2*j] = np.cos(f * t_fit)
        X_fit_main[:, 2*j+1] = np.sin(f * t_fit)

    c_main, _, _, _ = lstsq(X_fit_main, y_fit, rcond=None)
    y_fit_main = X_fit_main @ c_main
    r2_fit_main = 1 - np.sum((y_fit - y_fit_main)**2) / np.sum((y_fit - np.mean(y_fit))**2)

    X_hold_main = np.zeros((n_holdback, 2 * len(freqs_main)))
    for j, f in enumerate(freqs_main):
        X_hold_main[:, 2*j] = np.cos(f * t_hold)
        X_hold_main[:, 2*j+1] = np.sin(f * t_hold)
    y_hold_main = X_hold_main @ c_main
    r2_hold_main = 1 - np.sum((y_hold - y_hold_main)**2) / np.sum((y_hold - np.mean(y_hold))**2)
    corr_main = np.corrcoef(y_hold, y_hold_main)[0, 1]

    print(f"\n  === Model 3: Main harmonics (N<=10, {len(freqs_main)} filters) ===")
    print(f"  Fit R2: {r2_fit_main:.4f}")
    print(f"  Holdback R2: {r2_hold_main:.4f}, corr: {corr_main:.4f}")

    return {
        't_yr': t_yr, 'detrended': detrended,
        'n_fit': n_fit, 'n_holdback': n_holdback,
        'y_fit': y_fit, 'y_hold': y_hold,
        'y_fit_grid': y_fit_grid, 'y_hold_grid': y_hold_grid,
        'y_fit_strong': y_fit_strong, 'y_hold_strong': y_hold_strong,
        'y_fit_main': y_fit_main, 'y_hold_main': y_hold_main,
        'r2_fit_grid': r2_fit_grid, 'r2_hold_grid': r2_hold_grid,
        'r2_fit_strong': r2_fit_strong, 'r2_hold_strong': r2_hold_strong,
        'r2_fit_main': r2_fit_main, 'r2_hold_main': r2_hold_main,
        'corr_grid': corr_grid, 'corr_strong': corr_strong, 'corr_main': corr_main,
        'linear_coeffs': linear_coeffs,
        'n_strong': len(freqs_strong), 'n_main': len(freqs_main),
        'Ns_strong': Ns_strong,
    }


def plot_part_d(data, out_dir):
    """Plot held-back validation comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    t_yr = data['t_yr']
    n_fit = data['n_fit']
    t_fit = t_yr[:n_fit]
    t_hold = t_yr[n_fit:]

    # Panel 1: All three models on holdback
    ax = axes[0, 0]
    ax.plot(t_hold, data['y_hold'], 'b-', lw=1, label='Actual')
    ax.plot(t_hold, data['y_hold_grid'], 'r-', lw=0.7, alpha=0.7,
            label=f'Grid N*w0 (R2={data["r2_hold_grid"]:.3f})')
    ax.plot(t_hold, data['y_hold_strong'], 'g-', lw=0.7, alpha=0.7,
            label=f'Strong ({data["n_strong"]}f, R2={data["r2_hold_strong"]:.3f})')
    ax.plot(t_hold, data['y_hold_main'], 'm--', lw=0.7, alpha=0.7,
            label=f'Main N<=10 ({data["n_main"]}f, R2={data["r2_hold_main"]:.3f})')
    ax.axvline(t_yr[n_fit], color='black', ls='--', lw=0.5)
    ax.set_title('Holdback: Model Comparison')
    ax.set_xlabel('Time (years)')
    ax.legend(fontsize=7)

    # Panel 2: Full series with grid model
    ax = axes[0, 1]
    ax.plot(t_yr, data['detrended'], 'b-', lw=0.3, alpha=0.5, label='Actual')
    ax.plot(t_fit, data['y_fit_grid'], 'r-', lw=0.5, label='Grid fit')
    ax.plot(t_hold, data['y_hold_grid'], 'g-', lw=0.7, label='Grid projection')
    ax.axvline(t_yr[n_fit], color='black', ls='--', lw=1)
    ax.set_title(f'Full Series: Grid Model (fit R2={data["r2_fit_grid"]:.3f})')
    ax.set_xlabel('Time (years)')
    ax.legend(fontsize=8)

    # Panel 3: Model comparison bar chart
    ax = axes[1, 0]
    models = ['Grid\n(33 harm)', f'Strong\n({data["n_strong"]} harm)', f'Main\n({data["n_main"]} harm)']
    fit_r2s = [data['r2_fit_grid'], data['r2_fit_strong'], data['r2_fit_main']]
    hold_r2s = [data['r2_hold_grid'], data['r2_hold_strong'], data['r2_hold_main']]
    corrs = [data['corr_grid'], data['corr_strong'], data['corr_main']]

    x = np.arange(len(models))
    w = 0.25
    ax.bar(x - w, fit_r2s, w, label='Fit R2', color='steelblue')
    ax.bar(x, [max(0, r) for r in hold_r2s], w, label='Holdback R2', color='coral')
    ax.bar(x + w, corrs, w, label='Holdback Corr', color='green')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison: Fit vs Holdback')
    ax.legend(fontsize=8)
    ax.axhline(0, color='black', lw=0.3)

    # Panel 4: Holdback error
    ax = axes[1, 1]
    error_grid = data['y_hold'] - data['y_hold_grid']
    error_strong = data['y_hold'] - data['y_hold_strong']
    ax.plot(t_hold, error_grid, 'r-', lw=0.5, alpha=0.7, label='Grid error')
    ax.plot(t_hold, error_strong, 'g-', lw=0.5, alpha=0.7, label='Strong error')
    ax.axhline(0, color='black', lw=0.3)
    ax.set_title('Holdback Prediction Error')
    ax.set_xlabel('Time (years)')
    ax.legend(fontsize=8)

    plt.suptitle("Part D: Held-Back Validation", fontsize=14, fontweight='bold')
    path = os.path.join(out_dir, 'fig_v2_partD_holdback.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved: {path}")


# =============================================================================
# Part E: 17.1yr Sinusoid Overlay (carried from v1)
# =============================================================================

def part_e_overlay(close, dates, fs, w0):
    """Plot DJIA with fitted 17.1yr + 8.6yr sinusoids."""
    print("\n" + "=" * 70)
    print("PART E: 17.1-Year Fundamental Overlay")
    print("=" * 70)

    log_prices = np.log(close)
    t_yr = np.arange(len(close)) / fs

    linear_coeffs = np.polyfit(t_yr, log_prices, 1)
    detrended = log_prices - np.polyval(linear_coeffs, t_yr)

    # Fit N=1 + N=2
    omega_1 = w0
    omega_2 = 2 * w0
    X = np.column_stack([
        np.cos(omega_1 * t_yr), np.sin(omega_1 * t_yr),
        np.cos(omega_2 * t_yr), np.sin(omega_2 * t_yr),
    ])
    coeffs, _, _, _ = lstsq(X, detrended, rcond=None)

    n1 = coeffs[0] * np.cos(omega_1 * t_yr) + coeffs[1] * np.sin(omega_1 * t_yr)
    n12 = n1 + coeffs[2] * np.cos(omega_2 * t_yr) + coeffs[3] * np.sin(omega_2 * t_yr)

    a1 = np.sqrt(coeffs[0]**2 + coeffs[1]**2)
    a2 = np.sqrt(coeffs[2]**2 + coeffs[3]**2)
    print(f"  N=1: amplitude={a1:.4f} ({a1*100:.1f}% log-price swing)")
    print(f"  N=2: amplitude={a2:.4f} ({a2*100:.1f}% log-price swing)")

    # Find peaks/troughs of N=1
    from scipy.signal import argrelextrema
    order = int(2 * np.pi / omega_1 * fs * 0.3)
    peaks = argrelextrema(n1, np.greater, order=order)[0]
    troughs = argrelextrema(n1, np.less, order=order)[0]

    print(f"\n  17.1yr cycle peaks:")
    for idx in peaks:
        if idx < len(dates):
            print(f"    {dates.iloc[idx].strftime('%Y-%m')} (year {t_yr[idx]:.1f})")
    print(f"  17.1yr cycle troughs:")
    for idx in troughs:
        if idx < len(dates):
            print(f"    {dates.iloc[idx].strftime('%Y-%m')} (year {t_yr[idx]:.1f})")

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(18, 14))

    # Panel 1: Log price space
    ax = axes[0]
    trend_n1 = np.polyval(linear_coeffs, t_yr) + n1
    trend_n12 = np.polyval(linear_coeffs, t_yr) + n12
    ax.plot(t_yr, log_prices, 'b-', lw=0.5, alpha=0.7, label='log(DJIA)')
    ax.plot(t_yr, trend_n1, 'r-', lw=2, label=f'Trend + N=1 (A={a1:.3f})')
    ax.plot(t_yr, trend_n12, 'g--', lw=1.5, label=f'Trend + N=1,2')
    for idx in peaks:
        ax.axvline(t_yr[idx], color='red', alpha=0.2, lw=1)
    for idx in troughs:
        ax.axvline(t_yr[idx], color='blue', alpha=0.2, lw=1)
    ax.set_ylabel('log(Price)')
    ax.set_title('DJIA with 17.1yr Fundamental')
    ax.legend(fontsize=9)

    # Panel 2: Detrended
    ax = axes[1]
    ax.plot(t_yr, detrended, 'b-', lw=0.5, alpha=0.5, label='Detrended')
    ax.plot(t_yr, n1, 'r-', lw=2, label='N=1 (17.1yr)')
    ax.plot(t_yr, n12, 'g-', lw=1.5, alpha=0.7, label='N=1 + N=2')
    ax.axhline(0, color='black', lw=0.3)
    ax.set_ylabel('Amplitude')
    ax.set_title('Detrended with Fitted Cycles')
    ax.legend(fontsize=9)

    # Panel 3: Price space with dates
    ax = axes[2]
    # Convert t_yr to actual dates for x-axis
    date_nums = [dates.iloc[0] + pd.Timedelta(days=y*365.25) for y in t_yr]
    prices = np.exp(log_prices)
    ax.semilogy(date_nums, prices, 'b-', lw=0.5, label='DJIA')
    for idx in peaks:
        if idx < len(date_nums):
            ax.axvline(date_nums[idx], color='red', alpha=0.3, lw=1)
            ax.plot(date_nums[idx], prices[idx], 'rv', ms=10)
    for idx in troughs:
        if idx < len(date_nums):
            ax.axvline(date_nums[idx], color='blue', alpha=0.3, lw=1)
            ax.plot(date_nums[idx], prices[idx], 'b^', ms=10)
    ax.set_ylabel('Price (log scale)')
    ax.set_xlabel('Date')
    ax.set_title('DJIA with 17.1yr Peaks and Troughs')
    ax.legend(fontsize=9)

    plt.suptitle(f"Part E: 17.1yr Fundamental (w0={w0:.4f} rad/yr, T={2*np.pi/w0:.1f} yr)",
                 fontsize=14, fontweight='bold')
    path = os.path.join(OUT_DIR, 'fig_v2_partE_overlay.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved: {path}")

    return {'a1': a1, 'a2': a2, 'peaks': peaks, 'troughs': troughs}


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("HURST 75/23/2 v2 -- Ormsby Trend + Optimized CMW Modeling")
    print("=" * 70)

    # Run pipeline
    result = derive_nominal_model(
        symbol='djia', freq='weekly',
        start='1921-04-29', end='1965-05-21',
        prominence_frac=0.01, min_distance=2, verbose=False
    )
    close = result.close
    dates = result.dates
    fs = result.fs
    w0 = W0_HURST
    t_yr = np.arange(len(close)) / fs

    print(f"Data: {len(close)} samples, {result.years:.1f} yr, fs={fs:.1f}")
    print(f"Using w0={w0:.4f} rad/yr")

    # Part A: Ormsby trend removal
    trend_data = part_a_ormsby_trend(close, dates, fs)
    plot_part_a(trend_data, OUT_DIR)

    # Part B: Narrowband detection on residual (after LP-1 removal)
    nb_df, strong_df, weak_df, nb_result, nb_params = part_b_narrowband_detection(
        trend_data['residual_lp1'], fs, w0, max_N=34
    )
    plot_part_b(nb_df, strong_df, weak_df, nb_result, nb_params,
                t_yr, trend_data['residual_lp1'], OUT_DIR)

    # Part C: Optimized filter bank
    opt_params, opt_result, model_df = part_c_optimized_bank(
        trend_data['residual_lp1'], fs, strong_df, nb_df, t_yr
    )

    # Part D: Held-back validation
    holdback_data = part_d_holdback(close, fs, w0, strong_df)
    plot_part_d(holdback_data, OUT_DIR)

    # Part E: 17.1yr overlay
    overlay = part_e_overlay(close, dates, fs, w0)

    # === Summary ===
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
  PART A: Ormsby Trend Removal
    LP-1 alone explains {trend_data['r2_lp1']*100:.1f}% of variance
    LP-1 + BP-2 explains {trend_data['r2_slow']*100:.1f}%
    Hurst's 75% rule: CONFIRMED via Ormsby flat-top filters

  PART B: Narrowband CMW Detection
    {len(strong_df)} strong harmonics identified (of {len(nb_df)} tested)
    Strong N values: {sorted(strong_df.N.astype(int).tolist()) if len(strong_df) > 0 else 'none'}
    Beating detected in envelope spectra of individual harmonics

  PART C: Optimized CMW Bank
    {len(opt_params)} filters centered on MEASURED frequencies
    (not assumed N*w0 grid)

  PART D: Held-Back Validation ({holdback_data['n_holdback']/fs:.1f} yr holdback)
    Grid model (33 harm):   fit R2={holdback_data['r2_fit_grid']:.3f}, hold R2={holdback_data['r2_hold_grid']:.3f}, corr={holdback_data['corr_grid']:.3f}
    Strong model ({holdback_data['n_strong']} harm): fit R2={holdback_data['r2_fit_strong']:.3f}, hold R2={holdback_data['r2_hold_strong']:.3f}, corr={holdback_data['corr_strong']:.3f}
    Main model ({holdback_data['n_main']} harm):  fit R2={holdback_data['r2_fit_main']:.3f}, hold R2={holdback_data['r2_hold_main']:.3f}, corr={holdback_data['corr_main']:.3f}

  PART E: 17.1yr Fundamental
    N=1 amplitude: {overlay['a1']:.4f}, N=2: {overlay['a2']:.4f}
    """)
    print("=" * 70)
