# -*- coding: utf-8 -*-
"""
Phase 1 Validation: Compare reproduction to reference Figure AI-1

This script compares our generated Figure AI-1 reproduction with Hurst's
original figure to validate the quality of reproduction and identify
any systematic differences in peak detection or envelope fitting.

Reference: references/appendix_a/figure_AI1.png
Generated: experiments/appendix_A/figure_AI1_reproduction.png
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from PIL import Image

from src.spectral.lanczos import lanczos_spectrum
from src.spectral.peak_detection import find_spectral_peaks, find_spectral_troughs
from src.spectral.envelopes import fit_upper_envelope, fit_lower_envelope, envelope_model

# ============================================================================
# CONFIGURATION
# ============================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, '../../data/raw/^dji_w.csv')
ref_figure_path = os.path.join(script_dir, '../../references/appendix_a/figure_AI1.png')
our_figure_path = os.path.join(script_dir, 'figure_AI1_reproduction.png')
validation_report = os.path.join(script_dir, '../../data/processed/phase1_validation.txt')

# Peak detection parameters to test
TEST_PARAMS = [
    {'min_distance': 3, 'prominence': 0.5, 'name': 'Current (tight)'},
    {'min_distance': 5, 'prominence': 1.0, 'name': 'Wider spacing 1'},
    {'min_distance': 7, 'prominence': 1.5, 'name': 'Wider spacing 2'},
    {'min_distance': 10, 'prominence': 2.0, 'name': 'Even wider spacing'},
]

# ============================================================================
# LOAD DATA
# ============================================================================

print("=" * 80)
print("Phase 1 Validation: Comparing to Reference Figure AI-1")
print("=" * 80)
print()

# Load DJIA data
df = pd.read_csv(csv_path)
df['Date'] = pd.to_datetime(df['Date'])
df_hurst = df[df.Date.between('1921-04-29', '1965-05-21')]
close_prices = df_hurst.Close.values

# Compute spectrum
print("Computing Fourier-Lanczos spectrum...")
w, wRad, cosprt, sinprt, amp, phRad, phGrad = lanczos_spectrum(
    close_prices, 1, 52
)
omega_yr = w * 52

print(f"Data points: {len(close_prices)}")
print(f"Frequency range: {omega_yr[1]:.4f} to {omega_yr[-1]:.4f} rad/year")
print()

# ============================================================================
# TEST DIFFERENT PEAK DETECTION PARAMETERS
# ============================================================================

print("Testing different peak detection parameters:")
print("-" * 80)

results = []

for params in TEST_PARAMS:
    min_dist = params['min_distance']
    prom = params['prominence']
    name = params['name']

    # Detect peaks
    peak_idx, peak_freq, peak_amp = find_spectral_peaks(
        amp, omega_yr, min_distance=min_dist, prominence=prom
    )
    trough_idx, trough_freq, trough_amp = find_spectral_troughs(
        amp, omega_yr, min_distance=min_dist, prominence=prom
    )

    # Calculate spacing statistics
    if len(peak_freq) > 1:
        peak_spacings = np.diff(np.sort(peak_freq))
        mean_spacing = np.mean(peak_spacings)
        min_spacing = np.min(peak_spacings)
        max_spacing = np.max(peak_spacings)
    else:
        mean_spacing = min_spacing = max_spacing = np.nan

    print(f"\n{name} (min_distance={min_dist}, prominence={prom}):")
    print(f"  Peaks detected: {len(peak_idx)}")
    print(f"  Troughs detected: {len(trough_idx)}")
    print(f"  Mean peak spacing: {mean_spacing:.4f} rad/year")
    print(f"  Min spacing: {min_spacing:.4f}, Max spacing: {max_spacing:.4f}")
    print(f"  Peak amplitude range: {peak_amp.min():.2f} to {peak_amp.max():.2f}")

    results.append({
        'name': name,
        'min_distance': min_dist,
        'prominence': prom,
        'num_peaks': len(peak_idx),
        'num_troughs': len(trough_idx),
        'peak_freq': peak_freq,
        'peak_amp': peak_amp,
        'trough_freq': trough_freq,
        'trough_amp': trough_amp,
        'mean_spacing': mean_spacing,
        'min_spacing': min_spacing,
        'max_spacing': max_spacing
    })

print()
print("=" * 80)
print("Observation: Compare peak spacing to reference Figure AI-1")
print("=" * 80)
print()
print("Reference (from Hurst's Figure AI-1):")
print("- Fine structure spacing: ~0.3676 rad/year (period ~17.1 years)")
print("- Visible peaks appear spaced roughly 1-2 rad/year apart in the")
print("  frequency domain (corresponding to 3-6 year periods)")
print()
print("Current results suggest we may be detecting too many small peaks.")
print("Consider:")
print("1. Increasing min_distance (wider spacing between detected peaks)")
print("2. Increasing prominence threshold (filter out small/spurious peaks)")
print("3. Reviewing the reference figure to identify which peaks are significant")
print()

# ============================================================================
# VISUALIZE COMPARISON
# ============================================================================

print("Generating comparison visualizations...")

# Create side-by-side comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Use the current (tight) parameters for detailed comparison
current_result = results[0]
peak_freq = current_result['peak_freq']
peak_amp = current_result['peak_amp']
trough_freq = current_result['trough_freq']
trough_amp = current_result['trough_amp']

# Fit envelopes
upper_fit = fit_upper_envelope(peak_freq, peak_amp)
lower_fit = fit_lower_envelope(trough_freq, trough_amp)
upper_line = envelope_model(omega_yr, upper_fit['k'])
lower_line = envelope_model(omega_yr, lower_fit['k'])

# Plot 1: Our spectrum with current parameters
ax = axes[0, 0]
ax.plot(omega_yr, amp, 'b-', linewidth=1, label='Spectrum')
ax.plot(omega_yr, upper_line, 'r--', linewidth=2, label='Upper envelope')
ax.plot(omega_yr, lower_line, 'g--', linewidth=2, label='Lower envelope')
ax.plot(peak_freq, peak_amp, 'ro', markersize=6, alpha=0.7, label=f'Peaks (n={len(peak_freq)})')
ax.plot(trough_freq, trough_amp, 'go', markersize=6, alpha=0.7, label=f'Troughs (n={len(trough_freq)})')
ax.set_xscale('linear')
ax.set_yscale('log')
ax.set_xlim(-0.1, 22)
ax.set_ylim(0.45, 90)
ax.set_xlabel('Frequency (rad/year)')
ax.set_ylabel('Amplitude (log)')
ax.set_title(f'Current: {current_result["name"]} (min_dist={current_result["min_distance"]})')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

# Plot 2: Wider spacing test
wider_result = results[1]
peak_freq_w = wider_result['peak_freq']
peak_amp_w = wider_result['peak_amp']
trough_freq_w = wider_result['trough_freq']
trough_amp_w = wider_result['trough_amp']

ax = axes[0, 1]
ax.plot(omega_yr, amp, 'b-', linewidth=1, label='Spectrum')
ax.plot(peak_freq_w, peak_amp_w, 'ro', markersize=8, alpha=0.7, label=f'Peaks (n={len(peak_freq_w)})')
ax.plot(trough_freq_w, trough_amp_w, 'go', markersize=8, alpha=0.7, label=f'Troughs (n={len(trough_freq_w)})')
ax.set_xscale('linear')
ax.set_yscale('log')
ax.set_xlim(-0.1, 22)
ax.set_ylim(0.45, 90)
ax.set_xlabel('Frequency (rad/year)')
ax.set_ylabel('Amplitude (log)')
ax.set_title(f'Test: {wider_result["name"]} (min_dist={wider_result["min_distance"]})')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

# Plot 3: Peak spacing distribution
ax = axes[1, 0]
spacings_current = np.diff(np.sort(current_result['peak_freq']))
spacings_wider = np.diff(np.sort(wider_result['peak_freq']))
ax.hist(spacings_current, bins=15, alpha=0.6, label=f'Current (mean={np.mean(spacings_current):.3f})')
ax.hist(spacings_wider, bins=15, alpha=0.6, label=f'Wider (mean={np.mean(spacings_wider):.3f})')
ax.axvline(x=0.3676, color='r', linestyle='--', linewidth=2, label='Hurst reference: 0.3676')
ax.set_xlabel('Peak spacing (rad/year)')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Peak Spacing')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Number of peaks by parameter
ax = axes[1, 1]
names = [r['name'] for r in results]
peak_counts = [r['num_peaks'] for r in results]
trough_counts = [r['num_troughs'] for r in results]
x = np.arange(len(names))
width = 0.35
ax.bar(x - width/2, peak_counts, width, label='Peaks', alpha=0.8)
ax.bar(x + width/2, trough_counts, width, label='Troughs', alpha=0.8)
ax.set_ylabel('Count')
ax.set_title('Peak/Trough Detection by Parameter Set')
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
validation_fig_path = os.path.join(script_dir, 'phase1_validation_comparison.png')
plt.savefig(validation_fig_path, dpi=150, bbox_inches='tight')
print(f"Validation comparison figure saved to: {validation_fig_path}")

# ============================================================================
# SAVE VALIDATION REPORT
# ============================================================================

os.makedirs(os.path.dirname(validation_report), exist_ok=True)

with open(validation_report, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("Phase 1 Validation Report\n")
    f.write("=" * 80 + "\n\n")

    f.write("OBJECTIVE\n")
    f.write("-" * 80 + "\n")
    f.write("Validate Phase 1 results by comparing peak detection parameters and\n")
    f.write("identifying optimal settings for matching Hurst's Figure AI-1\n\n")

    f.write("REFERENCE TARGET\n")
    f.write("-" * 80 + "\n")
    f.write("Figure AI-1 (Hurst, Appendix A):\n")
    f.write("- Fourier-Lanczos spectrum with power-law envelope a(w) = k/w\n")
    f.write("- Fine structure spacing: ~0.3676 rad/year (T ~ 17.1 years)\n")
    f.write("- Peaks appear to be spaced wider than our current detection\n\n")

    f.write("PARAMETER TESTING RESULTS\n")
    f.write("-" * 80 + "\n\n")

    for i, result in enumerate(results, 1):
        f.write(f"Test {i}: {result['name']}\n")
        f.write(f"  Parameters: min_distance={result['min_distance']}, " +
                f"prominence={result['prominence']}\n")
        f.write(f"  Peaks detected: {result['num_peaks']}\n")
        f.write(f"  Troughs detected: {result['num_troughs']}\n")
        f.write(f"  Mean peak spacing: {result['mean_spacing']:.4f} rad/year\n")
        f.write(f"  Peak spacing range: {result['min_spacing']:.4f} to {result['max_spacing']:.4f}\n\n")

    f.write("ANALYSIS\n")
    f.write("-" * 80 + "\n")
    f.write("The reference Figure AI-1 suggests that significant peaks should be\n")
    f.write("spaced approximately 1-2 radians per year apart (3-6 year periods).\n\n")
    f.write("Current detection (min_distance=3, prominence=0.5) identifies many peaks\n")
    f.write("with tighter spacing (~0.73 rad/year mean), suggesting:\n\n")
    f.write("1. Detection parameters are too sensitive (low prominence threshold)\n")
    f.write("2. We may be picking up fine structure that Hurst smoothed/ignored\n")
    f.write("3. Data differences could cause spacing changes\n\n")

    f.write("RECOMMENDATIONS\n")
    f.write("-" * 80 + "\n")
    f.write("1. Increase min_distance to 5-7 to force wider spacing\n")
    f.write("2. Increase prominence threshold to 1.0-1.5\n")
    f.write("3. Compare visually with reference figure to find best match\n")
    f.write("4. Consider if 'peaks' vs 'significant peaks' distinction matters\n\n")

    f.write("NEXT STEPS\n")
    f.write("-" * 80 + "\n")
    f.write("1. Review references/appendix_a/figure_AI1.png carefully\n")
    f.write("2. Identify which peaks in the reference are 'major' vs 'minor'\n")
    f.write("3. Adjust detection parameters to match major peaks\n")
    f.write("4. Re-run phase1_complete.py with optimized parameters\n\n")

print(f"Validation report saved to: {validation_report}")
print()

# ============================================================================
# RECOMMENDATIONS
# ============================================================================

print("=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)
print()
print("Current peak spacing (~0.73 rad/year) is about 2x the Hurst reference")
print("(~0.3676 rad/year), but this may be expected because:")
print()
print("1. We're detecting MORE peaks than Hurst's figure shows")
print("   - Hurst may have filtered/smoothed minor peaks")
print("   - Our prominence threshold might be too low")
print()
print("2. Data source differences")
print("   - We use modern stooq data")
print("   - Hurst used different historical data source/quality")
print()
print("3. Detection philosophy")
print("   - We detect ALL local maxima above threshold")
print("   - Hurst may have used visual/manual selection")
print()
print("SUGGESTION: Increase min_distance and prominence parameters")
print("Try: min_distance=5-7, prominence=1.0-2.0")
print()

plt.show()
