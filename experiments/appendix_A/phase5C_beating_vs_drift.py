# -*- coding: utf-8 -*-
"""
Phase 5C: Beating vs Drift Hypothesis Testing

Determines whether comb filter envelope wobble arises from:
  (a) Multi-line beating (interference between closely-spaced spectral lines)
  (b) Single-line frequency drift over time

Four tests:
  Test 1: Drift rate distribution (are ridges stationary?)
  Test 2: Envelope wobble spectrum (do envelopes show beat frequency peaks?)
  Test 3: FM-AM coupling (does frequency variation correlate with amplitude?)
  Test 4: Synthetic beating (does a two-tone model reproduce real envelopes?)

Reference: J.M. Hurst, Appendix A; prd/supplementary_parametric_methods.md
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.time_frequency import (
    compute_scalogram,
    detect_ridges,
    test_drift_rate_distribution,
    test_envelope_wobble_spectrum,
    test_fm_am_coupling,
    test_synthetic_beating,
)
from src.filters import (
    design_hurst_comb_bank,
    create_filter_kernels,
    apply_filter_bank,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, '../../data/raw/^dji_w.csv')
nominal_path = os.path.join(script_dir, '../../data/processed/nominal_model.csv')

DATE_START = '1921-04-29'
DATE_END = '1965-05-21'
FS = 52
TWOPI = 2 * np.pi

# Nominal line spacing from Phase 3
NOMINAL_SPACING = 0.3719  # rad/yr

print("=" * 70)
print("Phase 5C: Beating vs Drift Hypothesis Testing")
print("=" * 70)
print()

# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading weekly DJIA data...")
df = pd.read_csv(csv_path)
df['Date'] = pd.to_datetime(df['Date'])
df_hurst = df[df.Date.between(DATE_START, DATE_END)].copy()
close_prices = df_hurst.Close.values
n_samples = len(close_prices)

nominal = pd.read_csv(nominal_path)
nominal_freqs = nominal['frequency'].values
print(f"  {n_samples} samples, {len(nominal_freqs)} nominal lines")
print()

# ============================================================================
# COMPUTE SCALOGRAM + RIDGES
# ============================================================================

print("Computing scalogram and detecting ridges...")
scalo = compute_scalogram(
    close_prices, freq_range=(0.5, 80.0), n_scales=200, fs=FS,
    fwhm_mode='constant_q', q_factor=5.0, freq_spacing='log',
)
ridges = detect_ridges(
    scalo['matrix'], scalo['frequencies'],
    min_prominence=0.08, min_duration_samples=52,
)
print(f"  {len(ridges)} ridges detected")
print()

# ============================================================================
# COMPUTE COMB FILTER BANK
# ============================================================================

print("Computing 23-filter HF comb bank (7.6-12 rad/yr)...")
comb_specs = design_hurst_comb_bank(
    n_filters=23, w1_start=7.2, w_step=0.2,
    passband_width=0.2, skirt_width=0.3, nw=1999, fs=FS
)
comb_filters = create_filter_kernels(comb_specs, fs=FS,
                                      filter_type='modulate', analytic=True)
comb_results = apply_filter_bank(close_prices, comb_filters, fs=FS, mode='reflect')
comb_outputs = comb_results['filter_outputs']
print(f"  {len(comb_outputs)} filter outputs")
print()

# ============================================================================
# TEST 1: Drift Rate Distribution
# ============================================================================

print("=" * 50)
print("TEST 1: Drift Rate Distribution")
print("=" * 50)

drift_result = test_drift_rate_distribution(ridges)

print(f"  N ridges: {len(drift_result['drift_rates'])}")
print(f"  Mean drift: {drift_result['mean_drift']:.4f} rad/yr/yr")
print(f"  Std drift:  {drift_result['std_drift']:.4f} rad/yr/yr")
print(f"  Median drift: {drift_result['median_drift']:.4f} rad/yr/yr")
print(f"  t-statistic: {drift_result['t_statistic']:.3f}")
print(f"  p-value: {drift_result['p_value']:.4f}")
print(f"  --> Conclusion: {drift_result['conclusion']}")
print()

# ============================================================================
# TEST 2: Envelope Wobble Spectrum
# ============================================================================

print("=" * 50)
print("TEST 2: Envelope Wobble Spectrum")
print("=" * 50)

wobble_result = test_envelope_wobble_spectrum(
    comb_outputs, fs=FS, nominal_line_spacing=NOMINAL_SPACING
)

n_beat = sum(1 for p in wobble_result['per_filter'] if p['has_beat_peak'])
print(f"  Filters with beat peak near {NOMINAL_SPACING:.3f} rad/yr: "
      f"{n_beat}/{len(wobble_result['per_filter'])}")
print(f"  Fraction: {wobble_result['fraction_with_beat_peak']:.1%}")

# Show dominant envelope frequencies for a few filters
for p in wobble_result['per_filter'][:5]:
    fc = p['center_freq']
    fc_str = f"fc={fc:.2f}" if fc else f"#{p['filter_index']}"
    print(f"    {fc_str}: dominant envelope freq={p['peak_frequency']:.3f} rad/yr "
          f"(T={p['peak_period_years']:.1f} yr), beat_peak={'YES' if p['has_beat_peak'] else 'no'}")

print(f"  --> Conclusion: {wobble_result['conclusion']}")
print()

# ============================================================================
# TEST 3: FM-AM Coupling
# ============================================================================

print("=" * 50)
print("TEST 3: FM-AM Coupling")
print("=" * 50)

fmam_result = test_fm_am_coupling(comb_outputs, fs=FS)

print(f"  Mean |correlation|: {fmam_result['mean_abs_correlation']:.3f}")
print(f"  Significant (p<0.05): {fmam_result['fraction_significant']:.1%}")

# Show a few
for p in fmam_result['per_filter'][:5]:
    print(f"    Filter {p['filter_index']}: r={p['correlation']:.3f}, "
          f"p={p['p_value']:.4f}")

print(f"  --> Conclusion: {fmam_result['conclusion']}")
print()

# ============================================================================
# TEST 4: Synthetic Beating
# ============================================================================

print("=" * 50)
print("TEST 4: Synthetic Beating")
print("=" * 50)

# Pick two adjacent nominal lines in the comb bank range
comb_range_lines = nominal_freqs[(nominal_freqs >= 7.6) & (nominal_freqs <= 12.0)]
if len(comb_range_lines) >= 2:
    # Find the pair with spacing closest to nominal spacing
    spacings = np.diff(comb_range_lines)
    best_pair_idx = np.argmin(np.abs(spacings - NOMINAL_SPACING))
    f1 = comb_range_lines[best_pair_idx]
    f2 = comb_range_lines[best_pair_idx + 1]
else:
    # Fallback: use approximate values
    f1, f2 = 8.34, 8.85

print(f"  Test frequencies: f1={f1:.2f}, f2={f2:.2f} rad/yr")
print(f"  Spacing: {abs(f2 - f1):.3f} rad/yr "
      f"(nominal: {NOMINAL_SPACING:.3f})")

synth_result = test_synthetic_beating(
    f1, f2, duration_samples=n_samples, fs=FS,
)

print(f"  Expected beat freq: {synth_result['beat_freq_expected']:.3f} rad/yr")
print(f"  Expected beat period: {synth_result['beat_period_expected']:.1f} years")
print(f"  Measured beat freq: {synth_result['beat_freq_measured']:.3f} rad/yr")
print(f"  Measured beat period: {synth_result['beat_period_measured']:.1f} years")
print(f"  Period match (within 20%): {synth_result['period_match']}")
print()

# ============================================================================
# FIGURE 1: Drift Rate Distribution
# ============================================================================

print("Generating figures...")

fig1, ax1 = plt.subplots(figsize=(10, 6))
drifts = drift_result['drift_rates']
ax1.hist(drifts, bins=25, color='steelblue', edgecolor='white', alpha=0.8)
ax1.axvline(0, color='red', linewidth=2, linestyle='--',
            label='Zero drift (stationary)')
ax1.axvline(drift_result['mean_drift'], color='orange', linewidth=2,
            label=f'Mean={drift_result["mean_drift"]:.4f}')
ax1.set_xlabel('Drift Rate (rad/yr per year)', fontsize=12)
ax1.set_ylabel('Count', fontsize=12)
ax1.set_title(
    f'Test 1: Ridge Drift Rate Distribution\n'
    f'p={drift_result["p_value"]:.4f} --> {drift_result["conclusion"]}',
    fontsize=13, fontweight='bold'
)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.2)

plt.tight_layout()
out1 = os.path.join(script_dir, 'phase5C_drift_distribution.png')
fig1.savefig(out1, dpi=150, bbox_inches='tight')
print(f"  Saved: {out1}")

# ============================================================================
# FIGURE 2: Envelope Wobble Spectra Heatmap
# ============================================================================

fig2, ax2 = plt.subplots(figsize=(14, 8))

# Build heatmap: 23 filters x frequency
n_filt = len(wobble_result['per_filter'])
freq_limit = 5.0  # Show up to 5 rad/yr in envelope frequency
per0 = wobble_result['per_filter'][0]
mask_freq = per0['envelope_spectrum_freqs'] <= freq_limit
n_freq = np.sum(mask_freq)
env_freqs = per0['envelope_spectrum_freqs'][mask_freq]

heatmap = np.zeros((n_filt, n_freq))
center_freqs = []
for i, p in enumerate(wobble_result['per_filter']):
    mask_f = p['envelope_spectrum_freqs'] <= freq_limit
    heatmap[i, :] = p['envelope_spectrum_amps'][mask_f]
    center_freqs.append(p['center_freq'] if p['center_freq'] else i)

# Normalize each row
for i in range(n_filt):
    row_max = heatmap[i, :].max()
    if row_max > 0:
        heatmap[i, :] /= row_max

im = ax2.imshow(heatmap, aspect='auto', origin='lower',
                cmap='hot', extent=[env_freqs[0], env_freqs[-1],
                0, n_filt], interpolation='bilinear')
ax2.axvline(NOMINAL_SPACING, color='cyan', linewidth=2, linestyle='--',
            label=f'Nominal spacing: {NOMINAL_SPACING:.3f} rad/yr')
ax2.set_xlabel('Envelope Modulation Frequency (rad/yr)', fontsize=12)
ax2.set_ylabel('Comb Filter Index', fontsize=12)
ax2.set_title(
    f'Test 2: Envelope Wobble Spectra\n'
    f'{wobble_result["fraction_with_beat_peak"]:.0%} show beat peak '
    f'--> {wobble_result["conclusion"]}',
    fontsize=13, fontweight='bold'
)
ax2.legend(loc='upper right', fontsize=11)
plt.colorbar(im, ax=ax2, label='Normalized Amplitude')

plt.tight_layout()
out2 = os.path.join(script_dir, 'phase5C_envelope_spectra.png')
fig2.savefig(out2, dpi=150, bbox_inches='tight')
print(f"  Saved: {out2}")

# ============================================================================
# FIGURE 3: FM-AM Coupling
# ============================================================================

fig3, ax3 = plt.subplots(figsize=(10, 6))

corrs = [p['correlation'] for p in fmam_result['per_filter']]
pvals = [p['p_value'] for p in fmam_result['per_filter']]
colors = ['red' if p < 0.05 else 'gray' for p in pvals]

ax3.bar(range(len(corrs)), corrs, color=colors, edgecolor='white', alpha=0.8)
ax3.axhline(0, color='black', linewidth=0.5)
ax3.set_xlabel('Comb Filter Index', fontsize=12)
ax3.set_ylabel('Pearson Correlation (env vs |f-f_mean|)', fontsize=12)
ax3.set_title(
    f'Test 3: FM-AM Coupling\n'
    f'{fmam_result["fraction_significant"]:.0%} significant (red) '
    f'--> {fmam_result["conclusion"]}',
    fontsize=13, fontweight='bold'
)
ax3.grid(True, alpha=0.2, axis='y')

plt.tight_layout()
out3 = os.path.join(script_dir, 'phase5C_fm_am_coupling.png')
fig3.savefig(out3, dpi=150, bbox_inches='tight')
print(f"  Saved: {out3}")

# ============================================================================
# FIGURE 4: Synthetic Beating
# ============================================================================

fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(14, 8))

t_years = np.arange(n_samples) / FS

# Panel A: Synthetic signal + envelope
ax4a.plot(t_years, synth_result['synthetic_signal'], 'b-',
          linewidth=0.3, alpha=0.5, label='Two-tone signal')
ax4a.plot(t_years, synth_result['envelope'], 'r-',
          linewidth=1.5, label='Envelope')
ax4a.plot(t_years, -synth_result['envelope'], 'r-', linewidth=1.5)
ax4a.set_ylabel('Amplitude')
ax4a.set_title(
    f'Test 4: Synthetic Beating (f1={f1:.2f}, f2={f2:.2f} rad/yr)\n'
    f'Expected beat period: {synth_result["beat_period_expected"]:.1f} yr, '
    f'Measured: {synth_result["beat_period_measured"]:.1f} yr, '
    f'Match: {synth_result["period_match"]}',
    fontsize=13, fontweight='bold'
)
ax4a.legend(fontsize=10)
ax4a.grid(True, alpha=0.2)

# Panel B: Envelope spectrum
env_detrend = synth_result['envelope'] - np.mean(synth_result['envelope'])
nfft = max(len(env_detrend) * 4, 8192)
env_fft = np.abs(np.fft.rfft(env_detrend, n=nfft))
freqs_hz = np.fft.rfftfreq(nfft, d=1.0 / FS)
freqs_rad = freqs_hz * TWOPI

ax4b.plot(freqs_rad, env_fft, 'b-', linewidth=1)
ax4b.axvline(synth_result['beat_freq_expected'], color='red', linewidth=2,
             linestyle='--', label=f'Expected: {synth_result["beat_freq_expected"]:.3f} rad/yr')
ax4b.axvline(NOMINAL_SPACING, color='green', linewidth=1.5,
             linestyle=':', label=f'Nominal spacing: {NOMINAL_SPACING:.3f} rad/yr')
ax4b.set_xlim(0, 3.0)
ax4b.set_xlabel('Envelope Modulation Frequency (rad/yr)')
ax4b.set_ylabel('Amplitude')
ax4b.set_title('Envelope FFT of Synthetic Beating Signal')
ax4b.legend(fontsize=10)
ax4b.grid(True, alpha=0.2)

plt.tight_layout()
out4 = os.path.join(script_dir, 'phase5C_synthetic_beating.png')
fig4.savefig(out4, dpi=150, bbox_inches='tight')
print(f"  Saved: {out4}")

# ============================================================================
# FIGURE 5: Verdict Summary
# ============================================================================

fig5, axes = plt.subplots(2, 2, figsize=(14, 10))

# Mini versions of each test
# 1: Drift
ax = axes[0, 0]
ax.hist(drifts, bins=20, color='steelblue', edgecolor='white', alpha=0.8)
ax.axvline(0, color='red', linewidth=2, linestyle='--')
ax.set_title(f'T1: Drift Rate\np={drift_result["p_value"]:.3f} '
             f'({drift_result["conclusion"]})', fontweight='bold')
ax.set_xlabel('Drift (rad/yr/yr)')

# 2: Envelope
ax = axes[0, 1]
peak_freqs = [p['peak_frequency'] for p in wobble_result['per_filter']]
ax.hist(peak_freqs, bins=20, color='coral', edgecolor='white', alpha=0.8)
ax.axvline(NOMINAL_SPACING, color='cyan', linewidth=2, linestyle='--',
           label=f'{NOMINAL_SPACING:.2f}')
ax.set_title(f'T2: Envelope Peaks\n{wobble_result["fraction_with_beat_peak"]:.0%} '
             f'beat ({wobble_result["conclusion"]})', fontweight='bold')
ax.set_xlabel('Dominant env freq (rad/yr)')
ax.legend(fontsize=9)

# 3: FM-AM
ax = axes[1, 0]
ax.bar(range(len(corrs)), corrs, color=colors, alpha=0.8)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_title(f'T3: FM-AM Coupling\n{fmam_result["fraction_significant"]:.0%} '
             f'significant ({fmam_result["conclusion"]})', fontweight='bold')
ax.set_xlabel('Filter index')

# 4: Synthetic
ax = axes[1, 1]
ax.plot(t_years[:520], synth_result['envelope'][:520], 'r-', linewidth=1.5)
ax.set_title(f'T4: Synthetic Beat\nPeriod match: {synth_result["period_match"]}',
             fontweight='bold')
ax.set_xlabel('Time (years)')

fig5.suptitle('Phase 5C: Beating vs Drift Verdict Summary',
              fontsize=15, fontweight='bold')
plt.tight_layout()
out5 = os.path.join(script_dir, 'phase5C_verdict_summary.png')
fig5.savefig(out5, dpi=150, bbox_inches='tight')
print(f"  Saved: {out5}")

# ============================================================================
# SAVE LINE STABILITY DATA
# ============================================================================

stability_path = os.path.join(script_dir, '../../data/processed/line_stability.csv')
stability_data = []
for r in ridges:
    stability_data.append({
        'ridge_id': r['ridge_id'],
        'mean_freq': r['mean_freq'],
        'std_freq': r['std_freq'],
        'drift_rate': r['drift_rate'],
        'r_squared': r['r_squared'],
        'duration_years': r['duration_years'],
        'mean_period_weeks': r['mean_period_weeks'],
    })
stability_df = pd.DataFrame(stability_data)
stability_df.to_csv(stability_path, index=False)
print(f"\nSaved: {stability_path}")

# ============================================================================
# OVERALL VERDICT
# ============================================================================

print()
print("=" * 60)
print("OVERALL VERDICT")
print("=" * 60)
print()
print(f"  Test 1 (Drift): {drift_result['conclusion']} "
      f"(p={drift_result['p_value']:.4f})")
print(f"  Test 2 (Envelope): {wobble_result['conclusion']} "
      f"({wobble_result['fraction_with_beat_peak']:.0%} beat peaks)")
print(f"  Test 3 (FM-AM): {fmam_result['conclusion']} "
      f"({fmam_result['fraction_significant']:.0%} significant)")
print(f"  Test 4 (Synthetic): period match = {synth_result['period_match']}")
print()

# Tally
votes_beating = 0
votes_drift = 0
if drift_result['conclusion'] == 'stationary':
    votes_beating += 1
    print("  T1 -> Supports BEATING (lines are stationary)")
else:
    votes_drift += 1
    print("  T1 -> Supports DRIFT (lines show significant drift)")

if wobble_result['fraction_with_beat_peak'] > 0.3:
    votes_beating += 1
    print("  T2 -> Supports BEATING (envelope shows beat frequency)")
else:
    votes_drift += 1
    print("  T2 -> Supports DRIFT (no clear beat peaks)")

if fmam_result['fraction_significant'] > 0.3:
    votes_beating += 1
    print("  T3 -> Supports BEATING (FM-AM coupling detected)")
else:
    votes_drift += 1
    print("  T3 -> Against beating (no FM-AM coupling)")

if synth_result['period_match']:
    votes_beating += 1
    print("  T4 -> Supports BEATING (synthetic period matches)")
else:
    print("  T4 -> Inconclusive (period mismatch)")

print()
if votes_beating > votes_drift:
    print(f"  VERDICT: BEATING dominates ({votes_beating}/{votes_beating + votes_drift} tests)")
elif votes_drift > votes_beating:
    print(f"  VERDICT: DRIFT dominates ({votes_drift}/{votes_beating + votes_drift} tests)")
else:
    print(f"  VERDICT: MIXED -- both mechanisms likely contribute")

plt.show()
print()
print("Done.")
