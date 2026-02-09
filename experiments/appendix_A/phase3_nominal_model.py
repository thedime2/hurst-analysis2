# -*- coding: utf-8 -*-
"""
Phase 3 Deliverable: Appendix A Figures AI-5, AI-6 and Nominal Model Derivation

This script reproduces Figures AI-5 and AI-6 from Hurst's Appendix A and
derives the Nominal Model -- the period hierarchy central to Hurst's
spectral analysis framework.

- AI-5: Modulation Sidebands ("The 'Line' Frequency Phenomena")
- AI-6: LSE Frequency vs Time Analysis ("Smoothing Filtered Outputs")
- Nominal Model: Discrete line frequencies and period hierarchy

Three frequency ranges are analyzed:
  - 7.6-12.0 rad/yr: Existing 23-filter comb bank from Phase 2
  - 3.5-7.6 rad/yr:  New medium-frequency comb bank
  - 0-3.5 rad/yr:    Phase 1 Fourier spectrum peaks (comb filters impractical)

Data range: 1921-04-29 to 1965-05-21
Key assumptions: Weekly spacing (52 pts/yr), all frequencies in rad/year
Reference: J.M. Hurst, The Profit Magic of Stock Transaction Timing,
           Appendix A, Figures AI-5, AI-6
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

from src.filters import (
    design_hurst_comb_bank,
    create_filter_kernels,
    apply_filter_bank,
    print_filter_specs
)
from src.spectral import lanczos_spectrum, find_spectral_peaks
from src.spectral.frequency_measurement import (
    measure_freq_at_peaks,
    measure_freq_at_troughs,
    measure_freq_at_zero_crossings
)
from src.nominal_model.sideband_analysis import (
    group_filters_into_lines,
    compute_sideband_envelopes
)
from src.nominal_model.lse_smoothing import (
    smooth_frequency_trace,
    fit_frequency_line
)
from src.nominal_model.derivation import (
    identify_line_frequencies,
    compute_line_spacings,
    build_nominal_model,
    validate_against_fourier
)

# ============================================================================
# CONFIGURATION
# ============================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, '../../data/raw/^dji_w.csv')
output_dir = os.path.join(script_dir, '../../data/processed')
results_path = os.path.join(output_dir, 'phase3_results.txt')
nominal_csv_path = os.path.join(output_dir, 'nominal_model.csv')
os.makedirs(output_dir, exist_ok=True)

# Hurst's analysis window
DATE_START = '1921-04-29'
DATE_END = '1965-05-21'
FS = 52  # Weekly sampling rate

# Display window (for AI-5/AI-6 figures, matching Hurst's ~275 week window)
BOOK_DATE_START = '1935-01-01'
BOOK_DATE_END = '1940-04-01'  # ~275 weeks

# ---- High-frequency comb bank (Phase 2 replication) ----
HF_N_FILTERS = 23
HF_W1_START = 7.2
HF_W_STEP = 0.2
HF_PASSBAND = 0.2
HF_SKIRT = 0.3
HF_NW = 1999

# ---- Medium-frequency comb bank (new for Phase 3) ----
MF_N_FILTERS = 15
MF_W1_START = 3.0
MF_W_STEP = 0.3
MF_PASSBAND = 0.3
MF_SKIRT = 0.4
MF_NW = 2999

# ---- Low-frequency: use Fourier peaks directly ----
# Phase 1 peaks below 3.5 rad/yr (from phase1_results.txt)
LOW_FREQ_PEAKS = np.array([2.2768])  # Only one peak below 3.5 rad/yr
# Also include troughs as potential lines
LOW_FREQ_TROUGHS = np.array([1.7076, 2.8460])


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


def run_comb_bank(close_prices, specs, fs, label):
    """Design, create, and apply a comb filter bank."""
    print(f"  Creating filter kernels for {label}...")
    filters = create_filter_kernels(
        filter_specs=specs, fs=fs, filter_type='modulate', analytic=True
    )
    print(f"  Applying {len(filters)} filters (mode='reflect')...")
    results = apply_filter_bank(signal=close_prices, filters=filters,
                                 fs=fs, mode='reflect')
    return results


def measure_all_frequencies(results, fs):
    """Measure frequencies for all filter outputs."""
    measurements = []
    for i, output in enumerate(results['filter_outputs']):
        sig_real = output['signal'].real
        phase = output['phase']

        pk = measure_freq_at_peaks(sig_real, phase, fs=fs)
        tr = measure_freq_at_troughs(sig_real, phase, fs=fs)
        zc = measure_freq_at_zero_crossings(sig_real, fs=fs)

        measurements.append({
            'filter_index': i,
            'center_freq': output['spec']['f_center'],
            'peaks': pk,
            'troughs': tr,
            'zero_crossings': zc
        })
    return measurements


# ============================================================================
# MAIN EXECUTION
# ============================================================================

print("=" * 80)
print("Phase 3: Line Spectrum and Nominal Model Derivation")
print("Appendix A, Figures AI-5 and AI-6")
print("=" * 80)
print()

# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading DJIA weekly data...")
df = pd.read_csv(csv_path)
df['Date'] = pd.to_datetime(df['Date'])
df_hurst = df[df.Date.between(DATE_START, DATE_END)]
close_prices = df_hurst.Close.values
dates = df_hurst.Date.values
dates_dt = pd.to_datetime(dates)

print(f"  Date range: {DATE_START} to {DATE_END}")
print(f"  Data points: {len(close_prices)}")
print()

# ============================================================================
# STEP 1: HIGH-FREQUENCY COMB BANK (7.6-12.0 rad/yr) -- Phase 2 replication
# ============================================================================

print("Step 1: High-frequency comb bank (7.6-12.0 rad/yr)")
hf_specs = design_hurst_comb_bank(
    n_filters=HF_N_FILTERS, w1_start=HF_W1_START, w_step=HF_W_STEP,
    passband_width=HF_PASSBAND, skirt_width=HF_SKIRT, nw=HF_NW, fs=FS
)
hf_results = run_comb_bank(close_prices, hf_specs, FS, "HF bank")
hf_measurements = measure_all_frequencies(hf_results, FS)
print(f"  {HF_N_FILTERS} filters, centers {hf_specs[0]['f_center']:.1f}-"
      f"{hf_specs[-1]['f_center']:.1f} rad/yr")
print()

# ============================================================================
# STEP 2: MEDIUM-FREQUENCY COMB BANK (3.5-7.6 rad/yr) -- New for Phase 3
# ============================================================================

print("Step 2: Medium-frequency comb bank (~3.5-7.6 rad/yr)")
mf_specs = design_hurst_comb_bank(
    n_filters=MF_N_FILTERS, w1_start=MF_W1_START, w_step=MF_W_STEP,
    passband_width=MF_PASSBAND, skirt_width=MF_SKIRT, nw=MF_NW, fs=FS
)
mf_results = run_comb_bank(close_prices, mf_specs, FS, "MF bank")
mf_measurements = measure_all_frequencies(mf_results, FS)
print(f"  {MF_N_FILTERS} filters, centers {mf_specs[0]['f_center']:.1f}-"
      f"{mf_specs[-1]['f_center']:.1f} rad/yr")
print()

# ============================================================================
# STEP 3: FOURIER SPECTRUM FOR LOW FREQUENCIES (0-3.5 rad/yr)
# ============================================================================

print("Step 3: Fourier spectrum for low frequencies (0-3.5 rad/yr)")
w, wRad, cosprt, sinprt, amp, phRad, phGrad = lanczos_spectrum(
    close_prices, 1, FS
)
omega_yr = w * FS

# Find peaks in the low-frequency range
peak_idx, peak_freq, peak_amp = find_spectral_peaks(
    amp, omega_yr, min_distance=6, prominence=1.2, freq_range=(0.5, 3.5)
)
low_freq_lines = peak_freq
print(f"  Fourier peaks below 3.5 rad/yr: {low_freq_lines}")
print(f"  Corresponding periods: {[f'{2*np.pi/f:.2f} yr' for f in low_freq_lines]}")
print()

# ============================================================================
# STEP 4: FIGURE AI-5 -- MODULATION SIDEBANDS (HF bank only)
# ============================================================================

print("Step 4: Figure AI-5 -- Modulation Sidebands")

# Group the 23 HF filters into 6 line families
grouping = group_filters_into_lines(hf_measurements, n_lines=6)

print("  Line grouping results:")
for i, (freq, period, group) in enumerate(zip(
        grouping['line_frequencies'],
        grouping['line_periods_weeks'],
        grouping['groups'])):
    filter_labels = [f"FC-{g+1}" for g in group]
    print(f"    Line {i+1}: {freq:.1f} rad/yr ({period:.1f} wks) "
          f"<- {', '.join(filter_labels)}")

# Compute sideband envelopes over book window
s_book, e_book = get_window_indices(dates_dt, BOOK_DATE_START, BOOK_DATE_END)
envelopes = compute_sideband_envelopes(
    hf_measurements, grouping,
    time_range=(s_book, e_book), fs=FS
)

# Plot Figure AI-5
fig_AI5, axes = plt.subplots(6, 1, figsize=(12, 14), sharex=True)
fig_AI5.suptitle('MODULATION SIDEBANDS\nFigure AI-5', fontsize=14, fontweight='bold')

for i in range(5, -1, -1):  # Plot from highest to lowest frequency (top to bottom)
    ax_idx = 5 - i
    ax = axes[ax_idx]
    env = envelopes[i]

    if env is None:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
        continue

    t_grid = env['time_grid'] - s_book  # offset to weeks from window start
    upper = env['upper_envelope']
    lower = env['lower_envelope']
    mean_t = env['mean_trace']

    # Fill between upper and lower envelope with hatching
    valid = ~np.isnan(upper) & ~np.isnan(lower)
    if np.any(valid):
        ax.fill_between(t_grid[valid], lower[valid], upper[valid],
                        alpha=0.3, color='gray', hatch='///', edgecolor='black',
                        linewidth=0.5)
        ax.plot(t_grid[valid], mean_t[valid], 'k-', linewidth=1.0)

    # Horizontal reference line at nominal frequency
    line_freq = env['line_freq']
    line_period = env['line_period_weeks']
    ax.axhline(y=line_freq, color='black', linewidth=0.8, linestyle='--')

    # Labels
    ax.set_ylabel(f'{line_freq:.1f}\nrad/yr', fontsize=9)
    ax.yaxis.set_label_position('left')

    # Period label on right
    ax2 = ax.twinx()
    ax2.set_ylabel(f'{line_period:.1f}\nwks', fontsize=9)
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks([])

    ax.grid(True, alpha=0.2)

axes[-1].set_xlabel('Weeks')
fig_AI5.tight_layout()
fig_AI5_path = os.path.join(script_dir, 'figure_AI5_reproduction.png')
fig_AI5.savefig(fig_AI5_path, dpi=150, bbox_inches='tight')
plt.close(fig_AI5)
print(f"  Saved: {fig_AI5_path}")
print()

# ============================================================================
# STEP 5: LSE SMOOTHING OF ALL FREQUENCY TRACES
# ============================================================================

print("Step 5: LSE smoothing of frequency traces")

all_line_fits = []

# --- HF bank (7.6-12 rad/yr) ---
print("  Smoothing HF bank traces...")
hf_smoothed = []
for i, fm in enumerate(hf_measurements):
    times = fm['peaks']['times']
    freqs = fm['peaks']['freqs_period']
    center = fm['center_freq']

    smoothed = smooth_frequency_trace(times, freqs, center_freq=center)
    line_fit = fit_frequency_line(times, freqs, center_freq=center)
    hf_smoothed.append(smoothed)
    all_line_fits.append(line_fit)

# --- MF bank (3.5-7.6 rad/yr) ---
print("  Smoothing MF bank traces...")
mf_smoothed = []
for i, fm in enumerate(mf_measurements):
    times = fm['peaks']['times']
    freqs = fm['peaks']['freqs_period']
    center = fm['center_freq']

    smoothed = smooth_frequency_trace(times, freqs, center_freq=center)
    line_fit = fit_frequency_line(times, freqs, center_freq=center)
    mf_smoothed.append(smoothed)
    all_line_fits.append(line_fit)

# --- Low-frequency lines (from Fourier peaks, no smoothing needed) ---
print("  Adding low-frequency Fourier lines...")
for freq in low_freq_lines:
    # Synthetic line fit for Fourier peaks (stationary by definition)
    all_line_fits.append({
        'mean_freq': float(freq),
        'drift_rate': 0.0,
        'drift_rate_annual': 0.0,
        'intercept': float(freq),
        'r_squared': 1.0,
        'std_dev': 0.0,
        'period_weeks': float(2 * np.pi / freq * FS),
        'n_points': 100  # synthetic
    })

print(f"  Total line fits: {len(all_line_fits)}")
print()

# ============================================================================
# STEP 6: IDENTIFY LINE FREQUENCIES AND MERGE CLOSE LINES
# ============================================================================

print("Step 6: Identifying distinct line frequencies")

# Extract raw line frequencies
raw_lines = identify_line_frequencies(all_line_fits, min_points=3)
print(f"  Raw lines before merging: {raw_lines['n_lines']}")

# Merge lines that are very close together (within 0.15 rad/yr)
# This handles overlapping comb filter outputs that detect the same line
MERGE_THRESHOLD = 0.15  # rad/yr

merged_freqs = []
merged_stabilities = []
merged_drifts = []
used = set()

raw_f = raw_lines['frequencies']
raw_s = raw_lines['stabilities']
raw_d = raw_lines['drift_rates']

for i in range(len(raw_f)):
    if i in used:
        continue
    # Find all lines within threshold of this one
    cluster = [i]
    for j in range(i + 1, len(raw_f)):
        if j in used:
            continue
        if abs(raw_f[j] - raw_f[i]) < MERGE_THRESHOLD:
            cluster.append(j)
            used.add(j)
    used.add(i)

    # Merge: take mean frequency, min stability (most stable)
    merged_freqs.append(np.mean(raw_f[cluster]))
    merged_stabilities.append(np.min(raw_s[cluster]))
    merged_drifts.append(np.mean(raw_d[cluster]))

merged_freqs = np.array(sorted(merged_freqs))
merged_stabilities = np.array(merged_stabilities)
merged_drifts = np.array(merged_drifts)

print(f"  Merged lines (threshold={MERGE_THRESHOLD} rad/yr): {len(merged_freqs)}")
for i, f in enumerate(merged_freqs):
    period_wk = 2 * np.pi / f * FS
    period_yr = 2 * np.pi / f
    print(f"    Line {i+1:>2d}: {f:6.2f} rad/yr = {period_wk:6.1f} wks = {period_yr:5.2f} yr")
print()

# ============================================================================
# STEP 7: FIGURE AI-6 -- LSE FREQUENCY VS TIME ANALYSIS
# ============================================================================

print("Step 7: Figure AI-6 -- LSE Frequency vs Time Analysis")

s_book, e_book = get_window_indices(dates_dt, BOOK_DATE_START, BOOK_DATE_END)
n_weeks = e_book - s_book

fig_AI6, ax = plt.subplots(figsize=(14, 10))
ax.set_title('LSE, FREQUENCY VS TIME ANALYSIS\nFigure AI-6', fontsize=14,
             fontweight='bold')

# Plot HF bank smoothed traces
for i, smoothed in enumerate(hf_smoothed):
    if len(smoothed['times']) == 0:
        continue
    mask = (smoothed['times'] >= s_book) & (smoothed['times'] < e_book)
    t_disp = smoothed['times'][mask] - s_book
    f_disp = smoothed['freqs_smoothed'][mask]
    if len(t_disp) > 0:
        ax.plot(t_disp, f_disp, 'k-', linewidth=0.6, alpha=0.7)

# Plot MF bank smoothed traces
for i, smoothed in enumerate(mf_smoothed):
    if len(smoothed['times']) == 0:
        continue
    mask = (smoothed['times'] >= s_book) & (smoothed['times'] < e_book)
    t_disp = smoothed['times'][mask] - s_book
    f_disp = smoothed['freqs_smoothed'][mask]
    if len(t_disp) > 0:
        ax.plot(t_disp, f_disp, 'k-', linewidth=0.6, alpha=0.7)

# Plot low-frequency lines as horizontal bands
for freq in low_freq_lines:
    ax.axhline(y=freq, color='black', linewidth=0.8, alpha=0.6, linestyle='-')

# Mark merged line frequencies on right axis
for f in merged_freqs:
    ax.axhline(y=f, color='gray', linewidth=0.3, alpha=0.3, linestyle=':')

ax.set_xlabel('Weeks', fontsize=12)
ax.set_ylabel('Radians/Year', fontsize=12)
ax.set_xlim(0, n_weeks)
ax.set_ylim(0, 13)
ax.grid(True, alpha=0.3)

# Period labels on right y-axis
ax2 = ax.twinx()
ax2.set_ylim(0, 13)
period_ticks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
period_labels = [f'{2*np.pi/w*FS:.0f}wk' if w > 0 else '' for w in period_ticks]
ax2.set_yticks(period_ticks)
ax2.set_yticklabels(period_labels, fontsize=8)
ax2.set_ylabel('Period (weeks)', fontsize=10)

fig_AI6.tight_layout()
fig_AI6_path = os.path.join(script_dir, 'figure_AI6_reproduction.png')
fig_AI6.savefig(fig_AI6_path, dpi=150, bbox_inches='tight')
plt.close(fig_AI6)
print(f"  Saved: {fig_AI6_path}")
print()

# ============================================================================
# STEP 8: NOMINAL MODEL DERIVATION
# ============================================================================

print("Step 8: Nominal Model Derivation")

# Line spacings
spacing_analysis = compute_line_spacings(merged_freqs)
print(f"  Mean line spacing: {spacing_analysis['mean_spacing']:.4f} rad/yr")
print(f"  Median spacing:    {spacing_analysis['median_spacing']:.4f} rad/yr")
print(f"  Std dev spacing:   {spacing_analysis['std_spacing']:.4f} rad/yr")
print(f"  Min spacing:       {spacing_analysis['min_spacing']:.4f} rad/yr")
print(f"  Max spacing:       {spacing_analysis['max_spacing']:.4f} rad/yr")
print(f"  Regularity (CV):   {spacing_analysis['regularity']:.3f}")
print(f"  Hurst fine struct:  0.3676 rad/yr")
print()

# Build nominal model table
nominal_model = build_nominal_model(merged_freqs, fs=FS)
print("  NOMINAL MODEL:")
print(f"  {'Line':>4s}  {'Freq (rad/yr)':>13s}  {'Period (wks)':>12s}  "
      f"{'Period (yr)':>11s}  {'Spacing':>10s}")
print(f"  {'----':>4s}  {'-'*13}  {'-'*12}  {'-'*11}  {'-'*10}")
for entry in nominal_model:
    spacing_str = f"{entry['spacing_from_prev']:.4f}" if entry['spacing_from_prev'] is not None else "   ---"
    print(f"  {entry['line_number']:>4d}  {entry['frequency']:>13.4f}  "
          f"{entry['period_weeks']:>12.1f}  {entry['period_years']:>11.2f}  "
          f"{spacing_str:>10s}")
print()

# ============================================================================
# STEP 9: VALIDATION AGAINST FOURIER PEAKS
# ============================================================================

print("Step 9: Validation against Phase 1 Fourier peaks")

fourier_peaks = np.array([2.2768, 3.6999, 5.2652, 6.6882, 8.1112,
                           9.6765, 11.9534, 13.0918, 14.5148, 16.5070, 18.7839])

validation = validate_against_fourier(merged_freqs, fourier_peaks, tolerance=0.5)
print(f"  Nominal lines:  {validation['n_nominal']}")
print(f"  Fourier peaks:  {validation['n_fourier']}")
print(f"  Matched:        {validation['n_matched']}")
print(f"  Match fraction: {validation['match_fraction']:.1%}")

if validation['matches']:
    print("  Matches:")
    for m in validation['matches']:
        print(f"    Nominal {m['nominal_freq']:.2f} <-> Fourier {m['fourier_freq']:.2f} "
              f"(delta = {m['distance']:.3f} rad/yr)")

if len(validation['unmatched_nominal']) > 0:
    print(f"  Unmatched nominal: {[f'{f:.2f}' for f in validation['unmatched_nominal']]}")
if len(validation['unmatched_fourier']) > 0:
    print(f"  Unmatched Fourier: {[f'{f:.2f}' for f in validation['unmatched_fourier']]}")
print()

# ============================================================================
# STEP 10: SAVE RESULTS
# ============================================================================

print("Step 10: Saving results...")

# Save text results
with open(results_path, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("Phase 3 Deliverables: Line Spectrum and Nominal Model\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 80 + "\n\n")

    f.write("DATA\n")
    f.write("-" * 40 + "\n")
    f.write(f"Source: DJIA Weekly (stooq.com)\n")
    f.write(f"Date range: {DATE_START} to {DATE_END}\n")
    f.write(f"Data points: {len(close_prices)}\n")
    f.write(f"Sampling rate: {FS} samples/year (weekly)\n\n")

    f.write("COMB FILTER BANKS\n")
    f.write("-" * 40 + "\n")
    f.write(f"HF bank: {HF_N_FILTERS} filters, {hf_specs[0]['f_center']:.1f}-"
            f"{hf_specs[-1]['f_center']:.1f} rad/yr\n")
    f.write(f"MF bank: {MF_N_FILTERS} filters, {mf_specs[0]['f_center']:.1f}-"
            f"{mf_specs[-1]['f_center']:.1f} rad/yr\n")
    f.write(f"LF lines: {len(low_freq_lines)} Fourier peaks below 3.5 rad/yr\n\n")

    f.write("MODULATION SIDEBANDS (Figure AI-5)\n")
    f.write("-" * 40 + "\n")
    f.write(f"{'Line':>4s}  {'Freq (r/yr)':>11s}  {'Period (wk)':>11s}  "
            f"{'N filters':>9s}  {'Hurst ref':>10s}\n")
    hurst_ref = [7.8, 8.5, 9.4, 10.2, 11.0, 11.8]
    for i, (freq, period, group) in enumerate(zip(
            grouping['line_frequencies'],
            grouping['line_periods_weeks'],
            grouping['groups'])):
        ref = f"{hurst_ref[i]:.1f}" if i < len(hurst_ref) else "---"
        f.write(f"  {i+1:>2d}    {freq:>11.2f}  {period:>11.1f}  "
                f"{len(group):>9d}  {ref:>10s}\n")

    f.write(f"\nLINE SPACING ANALYSIS\n")
    f.write("-" * 40 + "\n")
    f.write(f"Mean spacing:      {spacing_analysis['mean_spacing']:.4f} rad/yr\n")
    f.write(f"Median spacing:    {spacing_analysis['median_spacing']:.4f} rad/yr\n")
    f.write(f"Std dev:           {spacing_analysis['std_spacing']:.4f} rad/yr\n")
    f.write(f"Min spacing:       {spacing_analysis['min_spacing']:.4f} rad/yr\n")
    f.write(f"Max spacing:       {spacing_analysis['max_spacing']:.4f} rad/yr\n")
    f.write(f"Regularity (CV):   {spacing_analysis['regularity']:.3f}\n")
    f.write(f"Hurst fine struct: 0.3676 rad/yr\n\n")

    f.write("NOMINAL MODEL\n")
    f.write("-" * 40 + "\n")
    f.write(f"{'Line':>4s}  {'Freq (rad/yr)':>13s}  {'Period (wks)':>12s}  "
            f"{'Period (yr)':>11s}  {'Period (mo)':>11s}  {'Spacing':>10s}\n")
    for entry in nominal_model:
        spacing_str = f"{entry['spacing_from_prev']:.4f}" if entry['spacing_from_prev'] is not None else "   ---"
        f.write(f"  {entry['line_number']:>2d}   {entry['frequency']:>13.4f}  "
                f"{entry['period_weeks']:>12.1f}  {entry['period_years']:>11.2f}  "
                f"{entry['period_months']:>11.1f}  {spacing_str:>10s}\n")

    f.write(f"\nVALIDATION AGAINST FOURIER PEAKS\n")
    f.write("-" * 40 + "\n")
    f.write(f"Nominal lines:  {validation['n_nominal']}\n")
    f.write(f"Fourier peaks:  {validation['n_fourier']}\n")
    f.write(f"Matched:        {validation['n_matched']}\n")
    f.write(f"Match fraction: {validation['match_fraction']:.1%}\n\n")

    f.write("FIGURES GENERATED\n")
    f.write("-" * 40 + "\n")
    f.write(f"  figure_AI5_reproduction.png  - Modulation sidebands\n")
    f.write(f"  figure_AI6_reproduction.png  - LSE frequency vs time analysis\n")

print(f"  Saved: {results_path}")

# Save nominal model as CSV
nominal_df = pd.DataFrame(nominal_model)
nominal_df.to_csv(nominal_csv_path, index=False)
print(f"  Saved: {nominal_csv_path}")
print()

# ============================================================================
# SUMMARY
# ============================================================================

print("=" * 80)
print("Phase 3 Complete")
print("=" * 80)
print()
print(f"Lines identified: {len(merged_freqs)}")
print(f"Frequency range:  {merged_freqs[0]:.2f} - {merged_freqs[-1]:.2f} rad/yr")
print(f"Period range:     {2*np.pi/merged_freqs[-1]*FS:.1f} - "
      f"{2*np.pi/merged_freqs[0]*FS:.1f} weeks")
print(f"Mean spacing:     {spacing_analysis['mean_spacing']:.3f} rad/yr")
print(f"Fourier match:    {validation['match_fraction']:.0%}")
