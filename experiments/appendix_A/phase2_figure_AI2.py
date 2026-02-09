# -*- coding: utf-8 -*-
"""
Phase 2 Deliverable: Appendix A Figures AI-2, AI-3, AI-4 Reproduction
Overlapping Comb Filter Bank

This script reproduces Figures AI-2 through AI-4 from Hurst's Appendix A:
- AI-2: Idealized comb filter frequency response
- AI-3: Comb filter time-domain outputs with envelopes
- AI-4: Frequency versus time (measured at peaks/troughs/zero crossings)

Frequency measurement follows Hurst's method: measure period between
successive peaks, troughs, or zero crossings of the filtered signal,
rather than continuous instantaneous frequency from phase derivative.

Data range: 1921-04-29 to 1965-05-21
Key assumptions: Weekly spacing (52 pts/yr), all frequencies in rad/year
Reference: J.M. Hurst, The Profit Magic of Stock Transaction Timing,
           Appendix A, Figures AI-2 through AI-4
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from datetime import datetime

from src.filters import (
    design_hurst_comb_bank,
    create_filter_kernels,
    apply_filter_bank,
    plot_filter_bank_response,
    plot_idealized_comb_response,
    print_filter_specs
)

# ============================================================================
# FREQUENCY MEASUREMENT FUNCTIONS (Hurst's method)
# ============================================================================

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

    # Phase-based: average dφ/dt between successive peaks
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
    Frequency = 2π / (2 * half_period).

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
    freqs = np.pi / half_period_years  # 2π / (2 * half_period) = π / half_period
    times = (crossing_indices[:-1] + crossing_indices[1:]) / 2.0

    return {'times': times, 'freqs': freqs,
            'crossing_indices': crossing_indices}


# ============================================================================
# HELPER: get date window indices
# ============================================================================

def get_window_indices(dates_dt, date_start, date_end):
    """Return start_idx, end_idx for a date window."""
    mask = (dates_dt >= pd.to_datetime(date_start)) & \
           (dates_dt <= pd.to_datetime(date_end))
    if not mask.any():
        return 0, len(dates_dt)
    indices = np.where(mask)[0]
    return indices[0], indices[-1] + 1


# ============================================================================
# CONFIGURATION
# ============================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, '../../data/raw/^dji_w.csv')
output_dir = os.path.join(script_dir, '../../data/processed')
results_path = os.path.join(output_dir, 'phase2_results.txt')
os.makedirs(output_dir, exist_ok=True)

# Hurst's analysis window
DATE_START = '1921-04-29'
DATE_END = '1965-05-21'

# Data parameters
FS = 52  # Weekly sampling rate (samples per year)

# Comb filter bank parameters (from Hurst, Appendix A, p. 192)
N_FILTERS = 23
W1_START = 7.2         # rad/year - lower skirt edge of first filter
W_STEP = 0.2           # rad/year - step between successive filters
PASSBAND_WIDTH = 0.2   # rad/year - flat passband width (w3 - w2)
SKIRT_WIDTH = 0.3      # rad/year - transition band width (w2 - w1 = w4 - w3)
NW = 1999              # Filter length - increased from 1393 for sharper selectivity

# Display windows
# Book reference window (editorial error in Hurst's AI-3 header corrected)
BOOK_DATE_START = '1935-01-01'
BOOK_DATE_END = '1954-02-01'

# ============================================================================
# DESIGN COMB FILTER BANK
# ============================================================================

print("=" * 80)
print("Phase 2 Deliverables: Appendix A Figures AI-2, AI-3, AI-4")
print("Overlapping Comb Filter Bank")
print("=" * 80)
print()

print("Designing comb filter bank...")
specs = design_hurst_comb_bank(
    n_filters=N_FILTERS,
    w1_start=W1_START,
    w_step=W_STEP,
    passband_width=PASSBAND_WIDTH,
    skirt_width=SKIRT_WIDTH,
    nw=NW,
    fs=FS
)
print()

print_filter_specs(specs)

print("Verification:")
print(f"  Filter 1:  f1={specs[0]['f1']:.1f}, f2={specs[0]['f2']:.1f}, "
      f"f3={specs[0]['f3']:.1f}, f4={specs[0]['f4']:.1f} rad/yr  "
      f"(expected: 7.2, 7.5, 7.7, 8.0)")
print(f"  Filter 23: f1={specs[22]['f1']:.1f}, f2={specs[22]['f2']:.1f}, "
      f"f3={specs[22]['f3']:.1f}, f4={specs[22]['f4']:.1f} rad/yr  "
      f"(expected: 11.6, 11.9, 12.1, 12.4)")
print()

# ============================================================================
# CREATE FILTER KERNELS
# ============================================================================

print("Creating complex analytic filter kernels (method='modulate')...")
filters = create_filter_kernels(
    filter_specs=specs,
    fs=FS,
    filter_type='modulate',
    analytic=True
)
print(f"Created {len(filters)} filter kernels")
print(f"  Kernel length: {filters[0]['nw']} samples")
print(f"  Complex (analytic): {np.iscomplexobj(filters[0]['kernel'])}")
print()

# ============================================================================
# FIGURE AI-2: IDEALIZED COMB FILTER RESPONSE
# ============================================================================

print("Generating Figure AI-2 (idealized comb filter response)...")
fig_AI2 = plot_idealized_comb_response(
    filter_specs=specs,
    filters=filters,
    fs=FS
)
fig_AI2_path = os.path.join(script_dir, 'figure_AI2_reproduction.png')
fig_AI2.savefig(fig_AI2_path, dpi=150, bbox_inches='tight')
print(f"  Saved: {fig_AI2_path}")
plt.close(fig_AI2)

print("Generating actual FFT frequency response...")
fig_actual = plot_filter_bank_response(filters, fs=FS)
fig_AI2_actual_path = os.path.join(script_dir, 'figure_AI2_actual_response.png')
fig_actual.savefig(fig_AI2_actual_path, dpi=150, bbox_inches='tight')
print(f"  Saved: {fig_AI2_actual_path}")
plt.close(fig_actual)
print()

# ============================================================================
# LOAD DJIA DATA
# ============================================================================

print("Loading DJIA weekly data...")
df = pd.read_csv(csv_path)
df['Date'] = pd.to_datetime(df['Date'])
df_hurst = df[df.Date.between(DATE_START, DATE_END)]
close_prices = df_hurst.Close.values
dates = df_hurst.Date.values
dates_dt = pd.to_datetime(dates)

print(f"  Data range: {DATE_START} to {DATE_END}")
print(f"  Data points: {len(close_prices)}")
print(f"  Sampling: Weekly ({FS} points/year)")
print()

# ============================================================================
# APPLY FILTER BANK TO DJIA DATA
# ============================================================================

print("Applying 23 comb filters to DJIA data (mode='reflect')...")
results = apply_filter_bank(
    signal=close_prices,
    filters=filters,
    fs=FS,
    mode='reflect'
)
print(f"  Completed: {len(results['filter_outputs'])} filter outputs")
print()

# ============================================================================
# FIGURE AI-3: COMB FILTER OUTPUTS (TIME DOMAIN)
# Two versions: book reference window + full Hurst period
# ============================================================================

def plot_AI3(results, dates_dt, date_start, date_end, n_display, save_path, title_suffix):
    """Plot comb filter time-domain outputs with envelopes."""
    s_idx, e_idx = get_window_indices(dates_dt, date_start, date_end)
    weeks = np.arange(e_idx - s_idx)

    fig, axes = plt.subplots(n_display, 1, figsize=(14, 2.0 * n_display),
                              sharex=True)

    for i in range(n_display):
        output = results['filter_outputs'][i]
        ax = axes[i]

        sig_seg = output['signal'][s_idx:e_idx].real
        ax.plot(weeks, sig_seg, 'b-', linewidth=0.5)

        if output['envelope'] is not None:
            env_seg = output['envelope'][s_idx:e_idx]
            ax.plot(weeks, env_seg, 'r--', linewidth=0.7)
            ax.plot(weeks, -env_seg, 'r--', linewidth=0.7)

        center_freq = output['spec']['f_center']
        ax.set_ylabel(f"FC-{i+1}\n({center_freq:.1f} r/y)", fontsize=8)
        ax.axhline(0, color='gray', linewidth=0.3)
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel('Weeks')
    fig.suptitle(f'COMB OUTPUT EXAMPLE\nFigure AI-3 {title_suffix}',
                  fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


print("Generating Figure AI-3 (comb filter outputs)...")

# Book reference window
plot_AI3(results, dates_dt, BOOK_DATE_START, BOOK_DATE_END, 10,
         os.path.join(script_dir, 'figure_AI3_reproduction.png'),
         f'({BOOK_DATE_START} to {BOOK_DATE_END})')

# Full Hurst period
plot_AI3(results, dates_dt, DATE_START, DATE_END, 10,
         os.path.join(script_dir, 'figure_AI3_full_period.png'),
         f'Full Period ({DATE_START} to {DATE_END})')
print()

# ============================================================================
# MEASURE FREQUENCY AT PEAKS, TROUGHS, AND ZERO CROSSINGS
# ============================================================================

print("Measuring frequency at peaks, troughs, and zero crossings...")

freq_measurements = []
for i, output in enumerate(results['filter_outputs']):
    sig_real = output['signal'].real
    phase = output['phase']

    pk = measure_freq_at_peaks(sig_real, phase, fs=FS)
    tr = measure_freq_at_troughs(sig_real, phase, fs=FS)
    zc = measure_freq_at_zero_crossings(sig_real, fs=FS)

    freq_measurements.append({
        'filter_index': i,
        'center_freq': output['spec']['f_center'],
        'peaks': pk,
        'troughs': tr,
        'zero_crossings': zc
    })

    n_pk = len(pk['freqs_period'])
    n_tr = len(tr['freqs_period'])
    n_zc = len(zc['freqs'])
    if i < 5 or i >= N_FILTERS - 2:
        print(f"  FC-{i+1:>2d}: {n_pk:>3d} peak pairs, "
              f"{n_tr:>3d} trough pairs, {n_zc:>3d} zero-crossing pairs")
    elif i == 5:
        print(f"  ...")
print()

# ============================================================================
# FIGURE AI-4: FREQUENCY VERSUS TIME (Hurst's method)
# Three versions: peaks, zero crossings, and phase derivative
# ============================================================================

def plot_AI4(freq_measurements, dates_dt, date_start, date_end,
             method='peaks_period', save_path=None, title_extra=''):
    """
    Plot frequency vs time using Hurst's discrete measurement method.

    method: 'peaks_period', 'peaks_phase', 'troughs_period', 'zero_crossings'
    """
    s_idx, e_idx = get_window_indices(dates_dt, date_start, date_end)

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
        t_disp = times[mask] - s_idx  # offset to display weeks
        f_disp = freqs[mask]

        # Filter out extreme outliers (> 2x from center frequency)
        center = fm['center_freq']
        valid = (f_disp > center * 0.5) & (f_disp < center * 2.0)
        t_disp = t_disp[valid]
        f_disp = f_disp[valid]

        ax.plot(t_disp, f_disp, 'o', color=color, markersize=2.5,
                alpha=0.7, label=f"FC-{i+1}")

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
    ax.set_title(f'FREQUENCY VERSUS TIME ({method_label})\n'
                 f'Figure AI-4 {title_extra}')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7,
              markerscale=2)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig


print("Generating Figure AI-4 (frequency vs time - Hurst method)...")

# AI-4 book window: peaks (period-based)
plot_AI4(freq_measurements, dates_dt, BOOK_DATE_START, BOOK_DATE_END,
         method='peaks_period',
         save_path=os.path.join(script_dir, 'figure_AI4_peaks.png'),
         title_extra=f'({BOOK_DATE_START} to {BOOK_DATE_END})')
print(f"  Saved: figure_AI4_peaks.png")

# AI-4 book window: peaks (phase derivative)
plot_AI4(freq_measurements, dates_dt, BOOK_DATE_START, BOOK_DATE_END,
         method='peaks_phase',
         save_path=os.path.join(script_dir, 'figure_AI4_peaks_phase.png'),
         title_extra=f'({BOOK_DATE_START} to {BOOK_DATE_END})')
print(f"  Saved: figure_AI4_peaks_phase.png")

# AI-4 book window: zero crossings
plot_AI4(freq_measurements, dates_dt, BOOK_DATE_START, BOOK_DATE_END,
         method='zero_crossings',
         save_path=os.path.join(script_dir, 'figure_AI4_zeros.png'),
         title_extra=f'({BOOK_DATE_START} to {BOOK_DATE_END})')
print(f"  Saved: figure_AI4_zeros.png")

# AI-4 full period: peaks (period-based)
plot_AI4(freq_measurements, dates_dt, DATE_START, DATE_END,
         method='peaks_period',
         save_path=os.path.join(script_dir, 'figure_AI4_full_period.png'),
         title_extra=f'Full Period ({DATE_START} to {DATE_END})')
print(f"  Saved: figure_AI4_full_period.png")

# Keep original instantaneous frequency version for comparison
print("Generating Figure AI-4 (instantaneous frequency for comparison)...")
s_idx, e_idx = get_window_indices(dates_dt, BOOK_DATE_START, BOOK_DATE_END)
weeks_display = np.arange(e_idx - s_idx)

fig_AI4_inst, ax = plt.subplots(figsize=(14, 10))
for i, output in enumerate(results['filter_outputs']):
    if output['frequency'] is not None:
        inst_freq_radyr = output['frequency'][s_idx:e_idx] * 2 * np.pi
        if output['envelope'] is not None:
            env_seg = output['envelope'][s_idx:e_idx]
            max_env = np.max(env_seg) if np.max(env_seg) > 0 else 1.0
            reliable = env_seg > 0.1 * max_env
            inst_freq_radyr = np.where(reliable, inst_freq_radyr, np.nan)
        ax.plot(weeks_display, inst_freq_radyr, '.', markersize=1,
                alpha=0.5, label=f"FC-{i+1}")
ax.set_xlabel('Weeks')
ax.set_ylabel('Radians/Year')
ax.set_ylim(7.0, 13.0)
ax.set_title('FREQUENCY VERSUS TIME (Instantaneous - phase gradient)\n'
             f'({BOOK_DATE_START} to {BOOK_DATE_END})')
ax.grid(True, alpha=0.3)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, markerscale=5)
fig_AI4_inst.tight_layout()
fig_AI4_inst.savefig(os.path.join(script_dir, 'figure_AI4_instantaneous.png'),
                      dpi=150, bbox_inches='tight')
plt.close(fig_AI4_inst)
print(f"  Saved: figure_AI4_instantaneous.png")
print()

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("Saving results...")
with open(results_path, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("Phase 2 Deliverables: Comb Filter Bank Analysis\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 80 + "\n\n")

    f.write("DATA\n")
    f.write("-" * 40 + "\n")
    f.write(f"Source: DJIA Weekly (stooq.com)\n")
    f.write(f"Date range: {DATE_START} to {DATE_END}\n")
    f.write(f"Data points: {len(close_prices)}\n")
    f.write(f"Sampling rate: {FS} samples/year (weekly)\n\n")

    f.write("COMB FILTER BANK SPECIFICATION\n")
    f.write("-" * 40 + "\n")
    f.write(f"Number of filters: {N_FILTERS}\n")
    f.write(f"Passband width: {PASSBAND_WIDTH} rad/year\n")
    f.write(f"Skirt width: {SKIRT_WIDTH} rad/year\n")
    f.write(f"Step between filters: {W_STEP} rad/year\n")
    f.write(f"Total span per filter: {PASSBAND_WIDTH + 2*SKIRT_WIDTH} rad/year\n")
    f.write(f"Filter length (nw): {NW} samples\n")
    f.write(f"Filter method: modulate (complex analytic)\n\n")

    f.write("FILTER EDGE FREQUENCIES (rad/year)\n")
    f.write("-" * 40 + "\n")
    f.write(f"{'Filter':>8s}  {'w1':>8s}  {'w2':>8s}  {'w3':>8s}  {'w4':>8s}  "
            f"{'Center':>8s}  {'Period(wk)':>10s}\n")
    for spec in specs:
        period_wk = 2 * np.pi / spec['f_center'] * FS
        f.write(f"  FC-{spec['index']+1:>2d}   {spec['f1']:8.2f}  {spec['f2']:8.2f}  "
                f"{spec['f3']:8.2f}  {spec['f4']:8.2f}  {spec['f_center']:8.2f}  "
                f"{period_wk:10.1f}\n")

    f.write("\nENVELOPE AMPLITUDES (RMS over full data window)\n")
    f.write("-" * 40 + "\n")
    for i, output in enumerate(results['filter_outputs']):
        if output['envelope'] is not None:
            rms_env = np.sqrt(np.mean(output['envelope'] ** 2))
            max_env = np.max(output['envelope'])
            f.write(f"  FC-{i+1:>2d}: RMS={rms_env:.4f}, Max={max_env:.4f}\n")

    f.write("\nFREQUENCY MEASUREMENTS (median over book window)\n")
    f.write("-" * 40 + "\n")
    f.write(f"{'Filter':>8s}  {'Center':>8s}  {'Pk Period':>10s}  "
            f"{'Pk Phase':>10s}  {'Zero Xing':>10s}\n")
    f.write(f"{'':>8s}  {'(rad/yr)':>8s}  {'(rad/yr)':>10s}  "
            f"{'(rad/yr)':>10s}  {'(rad/yr)':>10s}\n")
    s_book, e_book = get_window_indices(dates_dt, BOOK_DATE_START, BOOK_DATE_END)
    for fm in freq_measurements:
        idx = fm['filter_index']
        center = fm['center_freq']

        # Median frequency in book window for each method
        def median_in_window(times, freqs, s, e):
            if len(times) == 0:
                return np.nan
            mask = (times >= s) & (times < e)
            vals = freqs[mask]
            vals = vals[(vals > center * 0.5) & (vals < center * 2.0)]
            return np.median(vals) if len(vals) > 0 else np.nan

        pk_per = median_in_window(fm['peaks']['times'],
                                   fm['peaks']['freqs_period'], s_book, e_book)
        pk_ph = np.nan
        if fm['peaks']['freqs_phase'] is not None:
            pk_ph = median_in_window(fm['peaks']['times'],
                                      fm['peaks']['freqs_phase'], s_book, e_book)
        zc = median_in_window(fm['zero_crossings']['times'],
                               fm['zero_crossings']['freqs'], s_book, e_book)

        f.write(f"  FC-{idx+1:>2d}   {center:8.2f}  {pk_per:10.3f}  "
                f"{pk_ph:10.3f}  {zc:10.3f}\n")

    f.write("\nFIGURES GENERATED\n")
    f.write("-" * 40 + "\n")
    f.write(f"  figure_AI2_reproduction.png      - Idealized comb filter response\n")
    f.write(f"  figure_AI2_actual_response.png   - Actual FFT frequency response\n")
    f.write(f"  figure_AI3_reproduction.png      - Filter outputs (book window)\n")
    f.write(f"  figure_AI3_full_period.png       - Filter outputs (full period)\n")
    f.write(f"  figure_AI4_peaks.png             - Freq vs time (peak-to-peak period)\n")
    f.write(f"  figure_AI4_peaks_phase.png       - Freq vs time (phase at peaks)\n")
    f.write(f"  figure_AI4_zeros.png             - Freq vs time (zero crossings)\n")
    f.write(f"  figure_AI4_full_period.png       - Freq vs time (full period)\n")
    f.write(f"  figure_AI4_instantaneous.png     - Freq vs time (continuous, comparison)\n")

print(f"  Saved: {results_path}")
print()

# ============================================================================
# SUMMARY
# ============================================================================

print("=" * 80)
print("Phase 2 Reproduction Complete")
print("=" * 80)
print()
print(f"Comb bank: {N_FILTERS} filters, nw={NW}")
print(f"Centers: {specs[0]['f_center']:.1f} - {specs[-1]['f_center']:.1f} rad/yr "
      f"({2*np.pi/specs[-1]['f_center']*FS:.1f} - "
      f"{2*np.pi/specs[0]['f_center']*FS:.1f} weeks)")
