# -*- coding: utf-8 -*-
"""
Phase 2A-0: Daily Data Comb Filter Analysis

Reproduces Phase 2 analysis using daily DJIA data instead of weekly,
providing ~5x finer frequency resolution in period-based measurements.

Key insight: With weekly data, peak-to-peak period = N integer weeks,
giving frequency quantization of ~0.3 rad/yr -- nearly identical to the
nominal line spacing (0.37 rad/yr). Daily data reduces this to ~0.06 rad/yr,
revealing the smooth frequency traces Hurst drew in Figure AI-4.

Also generates overlay of AI-3 envelopes on AI-4 frequency traces to
confirm the FM-AM coupling observed in Hurst's original figures.

Data range: 1921-04-29 to 1965-05-21
Reference: J.M. Hurst, Appendix A, Figures AI-3 and AI-4
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from src.filters import (
    design_hurst_comb_bank,
    create_filter_kernels,
    apply_filter_bank,
    plot_filter_bank_response,
    plot_idealized_comb_response,
    print_filter_specs
)
from src.spectral.frequency_measurement import (
    measure_freq_at_peaks,
    measure_freq_at_troughs,
    measure_freq_at_zero_crossings
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_window_indices(dates_dt, date_start, date_end):
    """Return start_idx, end_idx for a date window."""
    mask = (dates_dt >= pd.to_datetime(date_start)) & \
           (dates_dt <= pd.to_datetime(date_end))
    if not mask.any():
        return 0, len(dates_dt)
    indices = np.where(mask)[0]
    return indices[0], indices[-1] + 1


def samples_to_weeks(n_samples, fs_daily):
    """Convert daily sample count to approximate weeks."""
    return n_samples / fs_daily * 52


# ============================================================================
# CONFIGURATION
# ============================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, '../../data/raw/^dji_d.csv')
output_dir = os.path.join(script_dir, '../../data/processed')
os.makedirs(output_dir, exist_ok=True)

# Hurst's analysis window
DATE_START = '1921-04-29'
DATE_END = '1965-05-21'

# Display windows (same as Phase 2)
BOOK_DATE_START = '1935-01-01'
BOOK_DATE_END = '1954-02-01'

# Comb filter bank parameters - same frequency specs as Phase 2
N_FILTERS = 23
W1_START = 7.2         # rad/year
W_STEP = 0.2           # rad/year
PASSBAND_WIDTH = 0.2   # rad/year
SKIRT_WIDTH = 0.3      # rad/year

# ============================================================================
# LOAD DAILY DATA AND DETERMINE SAMPLING RATE
# ============================================================================

print("=" * 80)
print("Phase 2A-0: Daily Data Comb Filter Analysis")
print("=" * 80)
print()

print("Loading DJIA daily data...")
df = pd.read_csv(csv_path)
df['Date'] = pd.to_datetime(df['Date'])
df_hurst = df[df.Date.between(DATE_START, DATE_END)].copy()
close_prices = df_hurst.Close.values
dates = df_hurst.Date.values
dates_dt = pd.to_datetime(dates)

n_points = len(close_prices)
total_years = (dates_dt[-1] - dates_dt[0]).days / 365.25
FS_DAILY = n_points / total_years  # effective trading days per year

print(f"  Data range: {DATE_START} to {DATE_END}")
print(f"  Data points: {n_points}")
print(f"  Total years: {total_years:.2f}")
print(f"  Effective sampling rate: {FS_DAILY:.1f} trading days/year")
print(f"  (vs weekly: 52 samples/year)")
print()

# Quantization comparison
print("Frequency quantization comparison (near 10 rad/yr):")
N_wk = 32  # ~32-week period
N_day = round(N_wk * FS_DAILY / 52)
w_wk = 2 * np.pi * 52 / N_wk
w_wk_next = 2 * np.pi * 52 / (N_wk + 1)
w_day = 2 * np.pi * FS_DAILY / N_day
w_day_next = 2 * np.pi * FS_DAILY / (N_day + 1)
print(f"  Weekly:  N={N_wk} wk -> {w_wk:.3f} rad/yr, step = {abs(w_wk - w_wk_next):.3f} rad/yr")
print(f"  Daily:   N={N_day} days -> {w_day:.3f} rad/yr, step = {abs(w_day - w_day_next):.3f} rad/yr")
print(f"  Improvement factor: {abs(w_wk - w_wk_next) / abs(w_day - w_day_next):.1f}x")
print()

# ============================================================================
# FILTER LENGTH SCALING
# ============================================================================

# Scale NW proportionally to maintain similar frequency resolution
# Weekly: NW=1999 at fs=52 -> 38.4 years, resolution ~0.026 rad/yr
# Daily: want similar time span -> NW = 38.4 * FS_DAILY
# But that's ~10500 taps. Use moderate scaling for practical speed.
# NW=5001 -> ~18 years at daily rate, resolution ~0.055 rad/yr
# Still much finer than passband (0.2 rad/yr)

NW_DAILY = 5001

print(f"Filter length: NW={NW_DAILY}")
print(f"  Time span: {NW_DAILY / FS_DAILY:.1f} years")
print(f"  Freq resolution: ~{FS_DAILY / NW_DAILY:.4f} rad/yr")
print(f"  (passband = {PASSBAND_WIDTH} rad/yr, skirt = {SKIRT_WIDTH} rad/yr)")
print()

# ============================================================================
# DESIGN AND CREATE COMB FILTER BANK (same frequencies, daily fs)
# ============================================================================

print("Designing comb filter bank for daily data...")
specs = design_hurst_comb_bank(
    n_filters=N_FILTERS,
    w1_start=W1_START,
    w_step=W_STEP,
    passband_width=PASSBAND_WIDTH,
    skirt_width=SKIRT_WIDTH,
    nw=NW_DAILY,
    fs=FS_DAILY
)

print(f"  Filter 1:  center={specs[0]['f_center']:.2f} rad/yr")
print(f"  Filter 23: center={specs[22]['f_center']:.2f} rad/yr")
print()

print("Creating complex analytic filter kernels...")
filters = create_filter_kernels(
    filter_specs=specs,
    fs=FS_DAILY,
    filter_type='modulate',
    analytic=True
)
print(f"  Created {len(filters)} filters, kernel length: {filters[0]['nw']}")
print()

# ============================================================================
# VERIFY FILTER RESPONSE
# ============================================================================

print("Generating filter bank frequency response...")
fig_response = plot_filter_bank_response(filters, fs=FS_DAILY)
fig_response_path = os.path.join(script_dir, 'phase2A0_filter_response.png')
fig_response.savefig(fig_response_path, dpi=150, bbox_inches='tight')
plt.close(fig_response)
print(f"  Saved: {fig_response_path}")
print()

# ============================================================================
# APPLY FILTER BANK TO DAILY DJIA DATA
# ============================================================================

print(f"Applying {N_FILTERS} comb filters to daily DJIA data (mode='reflect')...")
print(f"  Signal length: {n_points}, Filter length: {NW_DAILY}")
print(f"  This may take a moment...")
results = apply_filter_bank(
    signal=close_prices,
    filters=filters,
    fs=FS_DAILY,
    mode='reflect'
)
print(f"  Completed: {len(results['filter_outputs'])} filter outputs")
print()

# ============================================================================
# FIGURE AI-3 DAILY: COMB FILTER OUTPUTS WITH ENVELOPES
# ============================================================================

def plot_AI3_daily(results, dates_dt, date_start, date_end, n_display,
                   save_path, title_suffix, fs_daily):
    """Plot comb filter time-domain outputs with envelopes, x-axis in weeks."""
    s_idx, e_idx = get_window_indices(dates_dt, date_start, date_end)
    n_samples = e_idx - s_idx
    weeks = np.arange(n_samples) / fs_daily * 52  # convert to weeks

    fig, axes = plt.subplots(n_display, 1, figsize=(14, 2.0 * n_display),
                              sharex=True)

    for i in range(n_display):
        output = results['filter_outputs'][i]
        ax = axes[i]

        sig_seg = output['signal'][s_idx:e_idx].real
        ax.plot(weeks, sig_seg, 'b-', linewidth=0.3)

        if output['envelope'] is not None:
            env_seg = output['envelope'][s_idx:e_idx]
            ax.plot(weeks, env_seg, 'r-', linewidth=0.8)
            ax.plot(weeks, -env_seg, 'r-', linewidth=0.8)

        center_freq = output['spec']['f_center']
        period_wk = 2 * np.pi / center_freq * 52
        ax.set_ylabel(f"FC-{i+1}\n{center_freq:.1f} r/y\n({period_wk:.0f}wk)",
                       fontsize=7)
        ax.axhline(0, color='gray', linewidth=0.3)
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel('Weeks')
    fig.suptitle(f'COMB OUTPUT (Daily Data)\n{title_suffix}',
                  fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


print("Generating AI-3 daily (comb filter outputs)...")
plot_AI3_daily(results, dates_dt, BOOK_DATE_START, BOOK_DATE_END, 10,
               os.path.join(script_dir, 'phase2A0_AI3_daily.png'),
               f'({BOOK_DATE_START} to {BOOK_DATE_END})', FS_DAILY)

plot_AI3_daily(results, dates_dt, DATE_START, DATE_END, 10,
               os.path.join(script_dir, 'phase2A0_AI3_daily_full.png'),
               f'Full Period ({DATE_START} to {DATE_END})', FS_DAILY)
print()

# ============================================================================
# MEASURE FREQUENCY (DAILY RESOLUTION)
# ============================================================================

print("Measuring frequency at peaks, troughs, and zero crossings (daily)...")

freq_measurements = []
for i, output in enumerate(results['filter_outputs']):
    sig_real = output['signal'].real
    phase = output['phase']

    pk = measure_freq_at_peaks(sig_real, phase, fs=FS_DAILY)
    tr = measure_freq_at_troughs(sig_real, phase, fs=FS_DAILY)
    zc = measure_freq_at_zero_crossings(sig_real, fs=FS_DAILY)

    freq_measurements.append({
        'filter_index': i,
        'center_freq': output['spec']['f_center'],
        'peaks': pk,
        'troughs': tr,
        'zero_crossings': zc,
        'envelope': output['envelope'],
        'signal_real': sig_real
    })

    n_pk = len(pk['freqs_period'])
    n_tr = len(tr['freqs_period'])
    if i < 3 or i >= N_FILTERS - 2:
        print(f"  FC-{i+1:>2d}: {n_pk:>4d} peak pairs, {n_tr:>4d} trough pairs")
    elif i == 3:
        print(f"  ...")
print()

# ============================================================================
# FIGURE AI-4 DAILY: FREQUENCY VS TIME (Connected Lines)
# ============================================================================

def plot_AI4_daily(freq_measurements, dates_dt, date_start, date_end,
                   method, save_path, title_extra, fs_daily,
                   connect_lines=True):
    """
    Plot frequency vs time with daily resolution.
    Connected lines match Hurst's drawing style.
    """
    s_idx, e_idx = get_window_indices(dates_dt, date_start, date_end)
    weeks_per_sample = 52 / fs_daily

    fig, ax = plt.subplots(figsize=(14, 10))
    colors = plt.cm.tab20(np.linspace(0, 1, min(len(freq_measurements), 20)))

    for i, fm in enumerate(freq_measurements):
        color = colors[i % len(colors)]

        if method == 'peaks_period':
            times = fm['peaks']['times']
            freqs = fm['peaks']['freqs_period']
        elif method == 'peaks_phase':
            times = fm['peaks']['times']
            freqs = fm['peaks']['freqs_phase']
            if freqs is None:
                continue
        elif method == 'troughs_period':
            times = fm['troughs']['times']
            freqs = fm['troughs']['freqs_period']
        elif method == 'zero_crossings':
            times = fm['zero_crossings']['times']
            freqs = fm['zero_crossings']['freqs']
        else:
            continue

        if len(times) == 0:
            continue

        # Window to display range
        mask = (times >= s_idx) & (times < e_idx)
        t_disp = (times[mask] - s_idx) * weeks_per_sample
        f_disp = freqs[mask]

        # Filter outliers (> 1.5x from center frequency)
        center = fm['center_freq']
        valid = (f_disp > center * 0.7) & (f_disp < center * 1.3)
        t_disp = t_disp[valid]
        f_disp = f_disp[valid]

        if connect_lines:
            ax.plot(t_disp, f_disp, '-', color=color, linewidth=0.8,
                    alpha=0.8, label=f"FC-{i+1}")
        else:
            ax.plot(t_disp, f_disp, 'o', color=color, markersize=2,
                    alpha=0.6, label=f"FC-{i+1}")

    ax.set_xlabel('Weeks')
    ax.set_ylabel('Radians/Year')
    ax.set_ylim(7.0, 13.0)

    method_labels = {
        'peaks_period': 'Peak-to-Peak Period',
        'peaks_phase': 'Phase Derivative at Peaks',
        'troughs_period': 'Trough-to-Trough Period',
        'zero_crossings': 'Zero-Crossing Half-Period'
    }
    method_label = method_labels.get(method, method)
    ax.set_title(f'FREQUENCY VERSUS TIME - Daily Data ({method_label})\n'
                 f'Figure AI-4 {title_extra}')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7,
              markerscale=2)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig


print("Generating AI-4 daily (frequency vs time with connected lines)...")

# Peak-to-peak - connected lines (Hurst style)
plot_AI4_daily(freq_measurements, dates_dt, BOOK_DATE_START, BOOK_DATE_END,
               method='peaks_period',
               save_path=os.path.join(script_dir, 'phase2A0_AI4_peaks_daily.png'),
               title_extra=f'({BOOK_DATE_START} to {BOOK_DATE_END})',
               fs_daily=FS_DAILY, connect_lines=True)
print(f"  Saved: phase2A0_AI4_peaks_daily.png")

# Trough-to-trough
plot_AI4_daily(freq_measurements, dates_dt, BOOK_DATE_START, BOOK_DATE_END,
               method='troughs_period',
               save_path=os.path.join(script_dir, 'phase2A0_AI4_troughs_daily.png'),
               title_extra=f'({BOOK_DATE_START} to {BOOK_DATE_END})',
               fs_daily=FS_DAILY, connect_lines=True)
print(f"  Saved: phase2A0_AI4_troughs_daily.png")

# Zero crossings
plot_AI4_daily(freq_measurements, dates_dt, BOOK_DATE_START, BOOK_DATE_END,
               method='zero_crossings',
               save_path=os.path.join(script_dir, 'phase2A0_AI4_zeros_daily.png'),
               title_extra=f'({BOOK_DATE_START} to {BOOK_DATE_END})',
               fs_daily=FS_DAILY, connect_lines=True)
print(f"  Saved: phase2A0_AI4_zeros_daily.png")

# Phase derivative at peaks
plot_AI4_daily(freq_measurements, dates_dt, BOOK_DATE_START, BOOK_DATE_END,
               method='peaks_phase',
               save_path=os.path.join(script_dir, 'phase2A0_AI4_phase_daily.png'),
               title_extra=f'({BOOK_DATE_START} to {BOOK_DATE_END})',
               fs_daily=FS_DAILY, connect_lines=True)
print(f"  Saved: phase2A0_AI4_phase_daily.png")

# Full period - peaks
plot_AI4_daily(freq_measurements, dates_dt, DATE_START, DATE_END,
               method='peaks_period',
               save_path=os.path.join(script_dir, 'phase2A0_AI4_peaks_daily_full.png'),
               title_extra=f'Full Period ({DATE_START} to {DATE_END})',
               fs_daily=FS_DAILY, connect_lines=True)
print(f"  Saved: phase2A0_AI4_peaks_daily_full.png")
print()

# ============================================================================
# FM-AM COUPLING OVERLAY: AI-3 Envelope on AI-4 Frequency
# ============================================================================

def plot_fmam_overlay(freq_measurements, dates_dt, date_start, date_end,
                      filter_indices, save_path, fs_daily):
    """
    Overlay AI-3 envelope (amplitude) on AI-4 frequency (period-based)
    for selected filters, confirming FM-AM coupling.

    Left axis: frequency (rad/yr), Right axis: envelope amplitude (normalized)
    """
    s_idx, e_idx = get_window_indices(dates_dt, date_start, date_end)
    n_display = len(filter_indices)
    weeks_per_sample = 52 / fs_daily

    fig, axes = plt.subplots(n_display, 1, figsize=(16, 3.0 * n_display),
                              sharex=True)
    if n_display == 1:
        axes = [axes]

    for plot_i, fi in enumerate(filter_indices):
        fm = freq_measurements[fi]
        ax1 = axes[plot_i]
        center = fm['center_freq']
        period_wk = 2 * np.pi / center * 52

        # -- Envelope (right axis, gray fill) --
        ax2 = ax1.twinx()
        envelope = fm['envelope']
        if envelope is not None:
            env_seg = envelope[s_idx:e_idx]
            env_weeks = np.arange(len(env_seg)) * weeks_per_sample
            env_norm = env_seg / np.max(env_seg) if np.max(env_seg) > 0 else env_seg
            ax2.fill_between(env_weeks, 0, env_norm, alpha=0.15, color='gray')
            ax2.plot(env_weeks, env_norm, color='gray', linewidth=0.5, alpha=0.5)
            ax2.set_ylim(0, 1.5)
            ax2.set_ylabel('Envelope\n(normalized)', fontsize=7, color='gray')
            ax2.tick_params(axis='y', labelcolor='gray', labelsize=7)

        # -- Frequency at peaks (blue) --
        pk_times = fm['peaks']['times']
        pk_freqs = fm['peaks']['freqs_period']
        if len(pk_times) > 0:
            mask = (pk_times >= s_idx) & (pk_times < e_idx)
            t_pk = (pk_times[mask] - s_idx) * weeks_per_sample
            f_pk = pk_freqs[mask]
            valid = (f_pk > center * 0.7) & (f_pk < center * 1.3)
            ax1.plot(t_pk[valid], f_pk[valid], 'b.-', linewidth=1.0,
                     markersize=3, alpha=0.8, label='Peak-to-peak freq')

        # -- Frequency at troughs (red) --
        tr_times = fm['troughs']['times']
        tr_freqs = fm['troughs']['freqs_period']
        if len(tr_times) > 0:
            mask = (tr_times >= s_idx) & (tr_times < e_idx)
            t_tr = (tr_times[mask] - s_idx) * weeks_per_sample
            f_tr = tr_freqs[mask]
            valid = (f_tr > center * 0.7) & (f_tr < center * 1.3)
            ax1.plot(t_tr[valid], f_tr[valid], 'r.-', linewidth=1.0,
                     markersize=3, alpha=0.8, label='Trough-to-trough freq')

        ax1.axhline(center, color='k', linewidth=0.5, linestyle='--', alpha=0.5)
        ax1.set_ylabel(f'FC-{fi+1}\n{center:.1f} r/y\n({period_wk:.0f}wk)',
                        fontsize=8)
        ax1.set_ylim(center - 1.0, center + 1.0)
        ax1.grid(True, alpha=0.2)
        if plot_i == 0:
            ax1.legend(fontsize=7, loc='upper left')

    axes[-1].set_xlabel('Weeks')
    fig.suptitle('FM-AM COUPLING: Envelope (gray) vs Frequency (blue=peaks, red=troughs)\n'
                 f'Daily Data ({date_start} to {date_end})',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


print("Generating FM-AM coupling overlay...")

# Show first 6 filters (lowest frequencies, most visible coupling)
plot_fmam_overlay(
    freq_measurements, dates_dt, BOOK_DATE_START, BOOK_DATE_END,
    filter_indices=[0, 2, 5, 10, 15, 22],
    save_path=os.path.join(script_dir, 'phase2A0_fmam_overlay.png'),
    fs_daily=FS_DAILY
)

# Detailed view of first filter only (your observation from the screenshot)
plot_fmam_overlay(
    freq_measurements, dates_dt, BOOK_DATE_START, BOOK_DATE_END,
    filter_indices=[0, 1, 2],
    save_path=os.path.join(script_dir, 'phase2A0_fmam_detail_fc1_3.png'),
    fs_daily=FS_DAILY
)
print()

# ============================================================================
# WEEKLY vs DAILY COMPARISON
# ============================================================================

def plot_weekly_vs_daily_comparison(freq_measurements_daily, dates_dt_daily,
                                    date_start, date_end,
                                    save_path, fs_daily):
    """
    Side-by-side comparison showing quantization improvement.
    Left: simulated weekly quantization. Right: daily resolution.
    """
    s_idx, e_idx = get_window_indices(dates_dt_daily, date_start, date_end)
    weeks_per_sample = 52 / fs_daily

    # Pick 5 representative filters spanning the range
    filter_indices = [0, 5, 11, 17, 22]

    fig, axes = plt.subplots(len(filter_indices), 2,
                              figsize=(18, 2.5 * len(filter_indices)),
                              sharex='col')

    for row, fi in enumerate(filter_indices):
        fm = freq_measurements_daily[fi]
        center = fm['center_freq']

        for col, (method, label) in enumerate([
            ('peaks_period', 'Peak-to-Peak'),
            ('troughs_period', 'Trough-to-Trough')
        ]):
            ax = axes[row][col]

            if method == 'peaks_period':
                times = fm['peaks']['times']
                freqs = fm['peaks']['freqs_period']
            else:
                times = fm['troughs']['times']
                freqs = fm['troughs']['freqs_period']

            if len(times) == 0:
                continue

            mask = (times >= s_idx) & (times < e_idx)
            t_disp = (times[mask] - s_idx) * weeks_per_sample
            f_disp = freqs[mask]
            valid = (f_disp > center * 0.7) & (f_disp < center * 1.3)
            t_disp = t_disp[valid]
            f_disp = f_disp[valid]

            # Daily: connected line
            ax.plot(t_disp, f_disp, '-', color='C0', linewidth=0.8,
                    alpha=0.8, label='Daily')

            # Simulated weekly: quantize to nearest integer-week period
            if len(f_disp) > 0:
                # Convert freq to period in weeks, round to integer, convert back
                period_wk = 2 * np.pi * 52 / f_disp
                period_wk_int = np.round(period_wk)
                f_quantized = 2 * np.pi * 52 / period_wk_int
                ax.plot(t_disp, f_quantized, 'o', color='C3', markersize=1.5,
                        alpha=0.5, label='Weekly (quantized)')

            ax.axhline(center, color='k', linewidth=0.3, linestyle='--')
            ax.set_ylim(center - 1.0, center + 1.0)
            ax.grid(True, alpha=0.2)

            if row == 0:
                ax.set_title(f'{label}', fontsize=10)
            if col == 0:
                ax.set_ylabel(f'FC-{fi+1}\n{center:.1f} r/y', fontsize=8)
            if row == 0 and col == 0:
                ax.legend(fontsize=7)

    axes[-1][0].set_xlabel('Weeks')
    axes[-1][1].set_xlabel('Weeks')
    fig.suptitle('WEEKLY vs DAILY: Frequency Quantization Comparison\n'
                 f'({date_start} to {date_end})',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


print("Generating weekly vs daily comparison...")
plot_weekly_vs_daily_comparison(
    freq_measurements, dates_dt, BOOK_DATE_START, BOOK_DATE_END,
    save_path=os.path.join(script_dir, 'phase2A0_weekly_vs_daily.png'),
    fs_daily=FS_DAILY
)
print()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

s_book, e_book = get_window_indices(dates_dt, BOOK_DATE_START, BOOK_DATE_END)

print(f"{'Filter':>8s}  {'Center':>8s}  {'Pk Median':>10s}  {'Tr Median':>10s}  "
      f"{'Pk Std':>8s}  {'Tr Std':>8s}")
print(f"{'':>8s}  {'(r/yr)':>8s}  {'(r/yr)':>10s}  {'(r/yr)':>10s}  "
      f"{'(r/yr)':>8s}  {'(r/yr)':>8s}")
print("-" * 65)

for fm in freq_measurements:
    idx = fm['filter_index']
    center = fm['center_freq']

    def stats_in_window(times, freqs, s, e, center):
        if len(times) == 0:
            return np.nan, np.nan
        mask = (times >= s) & (times < e)
        vals = freqs[mask]
        vals = vals[(vals > center * 0.7) & (vals < center * 1.3)]
        if len(vals) == 0:
            return np.nan, np.nan
        return np.median(vals), np.std(vals)

    pk_med, pk_std = stats_in_window(fm['peaks']['times'],
                                      fm['peaks']['freqs_period'],
                                      s_book, e_book, center)
    tr_med, tr_std = stats_in_window(fm['troughs']['times'],
                                      fm['troughs']['freqs_period'],
                                      s_book, e_book, center)

    print(f"  FC-{idx+1:>2d}   {center:8.2f}  {pk_med:10.3f}  {tr_med:10.3f}  "
          f"{pk_std:8.3f}  {tr_std:8.3f}")

print()
print("Phase 2A-0 Complete")
print("=" * 80)
