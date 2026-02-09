# -*- coding: utf-8 -*-
"""
Phase 1 Complete Deliverables: Appendix A Figure AI-1 Reproduction

This script generates the complete Phase 1 deliverables as specified in the PRD:
- Fourier-Lanczos spectrum of DJIA (1921-1965)
- Peak and trough frequency lists
- Envelope fit parameters (a(w) = k/w)
- Figure AI-1 reproduction with fitted envelopes

Reference: J.M. Hurst, The Profit Magic of Stock Transaction Timing, Appendix A
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime

from src.spectral.lanczos import lanczos_spectrum
from src.spectral.peak_detection import (
    find_spectral_peaks,
    find_spectral_troughs,
    detect_fine_structure_spacing
)
from src.spectral.envelopes import (
    fit_upper_envelope,
    fit_lower_envelope,
    envelope_model,
    fit_dual_envelope
)

# ============================================================================
# CONFIGURATION
# ============================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, '../../data/raw/^dji_w.csv')
output_dir = os.path.join(script_dir, '../../data/processed')
figure_path = os.path.join(script_dir, 'figure_AI1_reproduction.png')
results_path = os.path.join(output_dir, 'phase1_results.txt')

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Hurst's analysis window
DATE_START = '1921-04-29'
DATE_END = '1965-05-21'

# Data parameters
DATASPACING = 1  # Weekly data, no gaps
DATAPOINTSPERYR = 52  # Weekly sampling

# Peak detection parameters
# Optimized based on validation against Figure AI-1
# Original: min_distance=3, prominence=0.5 detected 25 peaks (too tight)
# Optimized: wider spacing to match Hurst's figure (1-2 rad/year peak spacing)
PEAK_MIN_DISTANCE = 6
PEAK_PROMINENCE = 1.2

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================

print("=" * 80)
print("Phase 1 Deliverables: Appendix A Figure AI-1 Reproduction")
print("=" * 80)
print()

# Load DJIA weekly data
df = pd.read_csv(csv_path)
df['Date'] = pd.to_datetime(df['Date'])

# Filter to Hurst's time window
df_hurst = df[df.Date.between(DATE_START, DATE_END)]
close_prices = df_hurst.Close.values

print(f"Data Range: {DATE_START} to {DATE_END}")
print(f"Data Points: {len(close_prices)}")
print(f"Sampling: Weekly (52 points/year)")
print()

# ============================================================================
# COMPUTE LANCZOS SPECTRUM
# ============================================================================

print("Computing Fourier-Lanczos spectrum...")
w, wRad, cosprt, sinprt, amp, phRad, phGrad = lanczos_spectrum(
    close_prices, DATASPACING, DATAPOINTSPERYR
)

# Convert to radians/year
omega_yr = w * DATAPOINTSPERYR

# Compute frequency resolution
freq_resolution = 2 * np.pi / (len(close_prices) / DATAPOINTSPERYR)
print(f"Frequency Resolution: {freq_resolution:.4f} radians/year")
print(f"Record Length: {len(close_prices) / DATAPOINTSPERYR:.2f} years")
print()

# ============================================================================
# DETECT PEAKS AND TROUGHS
# ============================================================================

print("Detecting spectral peaks and troughs...")
peak_idx, peak_freq, peak_amp = find_spectral_peaks(
    amp, omega_yr,
    min_distance=PEAK_MIN_DISTANCE,
    prominence=PEAK_PROMINENCE
)

trough_idx, trough_freq, trough_amp = find_spectral_troughs(
    amp, omega_yr,
    min_distance=PEAK_MIN_DISTANCE,
    prominence=PEAK_PROMINENCE
)

print(f"Detected {len(peak_idx)} peaks")
print(f"Detected {len(trough_idx)} troughs")
print()

# ============================================================================
# ANALYZE FINE STRUCTURE SPACING
# ============================================================================

print("Analyzing fine frequency structure...")
spacings, mean_spacing, std_spacing = detect_fine_structure_spacing(
    peak_freq, max_spacing=1.0
)

print(f"Mean peak spacing: {mean_spacing:.4f} rad/year (std = {std_spacing:.4f})")
if not np.isnan(mean_spacing):
    mean_period = 2 * np.pi / mean_spacing
    print(f"Corresponding period: {mean_period:.2f} years")
print()
print(f"Note: Hurst identified fine structure spacing ~0.3676 rad/year (T ~ 17.1 years)")
print()

# ============================================================================
# FIT ENVELOPES
# ============================================================================

print("Fitting power-law envelopes...")

# Fit upper envelope (peaks)
upper_fit = fit_upper_envelope(peak_freq, peak_amp)
print(f"\nUpper Envelope: a(w) = k_upper / w")
print(f"  k_upper = {upper_fit['k']:.4f}")
print(f"  R² = {upper_fit['r_squared']:.4f}")
print(f"  RMSE = {upper_fit['rmse']:.4f}")

# Fit lower envelope (troughs)
lower_fit = fit_lower_envelope(trough_freq, trough_amp)
print(f"\nLower Envelope: a(w) = k_lower / w")
print(f"  k_lower = {lower_fit['k']:.4f}")
print(f"  R² = {lower_fit['r_squared']:.4f}")
print(f"  RMSE = {lower_fit['rmse']:.4f}")

# Envelope ratio
envelope_ratio = upper_fit['k'] / lower_fit['k']
print(f"\nEnvelope Ratio (k_upper / k_lower): {envelope_ratio:.4f}")
print()

# Compare to hardcoded values from original test script
print("Comparison to original hardcoded values:")
print(f"  Original k_upper ~ 0.1875 (hardcoded)")
print(f"  Fitted k_upper = {upper_fit['k']:.4f}")
print(f"  Difference: {abs(upper_fit['k'] - 0.1875):.4f}")
print()
print(f"  Original k_lower ~ 0.0575 (hardcoded)")
print(f"  Fitted k_lower = {lower_fit['k']:.4f}")
print(f"  Difference: {abs(lower_fit['k'] - 0.0575):.4f}")
print()

# ============================================================================
# GENERATE FIGURE AI-1 REPRODUCTION
# ============================================================================

print("Generating Figure AI-1 reproduction...")

# Generate envelope lines
upper_line = envelope_model(omega_yr, upper_fit['k'])
lower_line = envelope_model(omega_yr, lower_fit['k'])

# Create figure
fig, ax = plt.subplots(figsize=(12, 7))

# Plot spectrum
ax.plot(omega_yr, amp, 'b-', linewidth=1.5, label='Lanczos Spectrum', zorder=1)

# Plot envelope lines
ax.plot(omega_yr, upper_line, 'r-', linewidth=2,
        label=f'Upper envelope: k = {upper_fit["k"]:.4f}', zorder=2)
ax.plot(omega_yr, lower_line, 'g-', linewidth=2,
        label=f'Lower envelope: k = {lower_fit["k"]:.4f}', zorder=2)

# Plot detected peaks and troughs
ax.plot(peak_freq, peak_amp, 'ro', markersize=5, alpha=0.6,
        label=f'Peaks (n={len(peak_idx)})', zorder=3)
ax.plot(trough_freq, trough_amp, 'go', markersize=5, alpha=0.6,
        label=f'Troughs (n={len(trough_idx)})', zorder=3)

# Set scales and limits
ax.set_xscale('linear')
ax.set_yscale('log')
ax.set_xlim(-0.1, 22)
ax.set_ylim(0.45, 90)

# Format axes
ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.2))
ax.minorticks_on()
ax.grid(which='major', color='#666666', linestyle='-', alpha=0.6)
ax.grid(which='minor', color='#999999', linestyle=':', alpha=0.4)

# Labels and title
ax.set_xlabel("Angular Frequency w (radians per year)", fontsize=12)
ax.set_ylabel("Amplitude (log scale)", fontsize=12)
ax.set_title(
    "Appendix A Figure AI-1 Reproduction\n"
    f"Fourier-Lanczos Spectrum of DJIA ({DATE_START} to {DATE_END})\n"
    "with Fitted Power-Law Envelopes",
    fontsize=13, fontweight='bold'
)
ax.legend(loc='upper right', fontsize=10)

# Add annotation box with fit quality
textstr = f'Envelope Fits:\nR²_upper = {upper_fit["r_squared"]:.4f}\nR²_lower = {lower_fit["r_squared"]:.4f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='bottom', bbox=props)

plt.tight_layout()
plt.savefig(figure_path, dpi=150, bbox_inches='tight')
print(f"Figure saved to: {figure_path}")
print()

# ============================================================================
# SAVE DELIVERABLES TO FILE
# ============================================================================

print("Saving deliverables to file...")

with open(results_path, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("Phase 1 Deliverables: Appendix A Figure AI-1 Reproduction\n")
    f.write("=" * 80 + "\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("\n")

    f.write("DATA SUMMARY\n")
    f.write("-" * 80 + "\n")
    f.write(f"Data Range: {DATE_START} to {DATE_END}\n")
    f.write(f"Data Points: {len(close_prices)}\n")
    f.write(f"Sampling: Weekly ({DATAPOINTSPERYR} points/year)\n")
    f.write(f"Frequency Resolution: {freq_resolution:.4f} radians/year\n")
    f.write(f"Record Length: {len(close_prices) / DATAPOINTSPERYR:.2f} years\n")
    f.write("\n")

    f.write("PEAK FREQUENCIES (radians/year)\n")
    f.write("-" * 80 + "\n")
    for i, (freq, ampl) in enumerate(zip(peak_freq, peak_amp), 1):
        period_yr = 2 * np.pi / freq if freq > 0 else np.inf
        f.write(f"{i:3d}.  w = {freq:7.4f} rad/yr,  T = {period_yr:7.2f} yr,  amp = {ampl:7.2f}\n")
    f.write("\n")

    f.write("TROUGH FREQUENCIES (radians/year)\n")
    f.write("-" * 80 + "\n")
    for i, (freq, ampl) in enumerate(zip(trough_freq, trough_amp), 1):
        period_yr = 2 * np.pi / freq if freq > 0 else np.inf
        f.write(f"{i:3d}.  w = {freq:7.4f} rad/yr,  T = {period_yr:7.2f} yr,  amp = {ampl:7.2f}\n")
    f.write("\n")

    f.write("FINE STRUCTURE ANALYSIS\n")
    f.write("-" * 80 + "\n")
    f.write(f"Mean peak spacing: {mean_spacing:.4f} rad/year (std = {std_spacing:.4f})\n")
    if not np.isnan(mean_spacing):
        f.write(f"Corresponding period: {2*np.pi/mean_spacing:.2f} years\n")
    f.write(f"Note: Hurst identified spacing ~0.3676 rad/year (T ~ 17.1 years)\n")
    f.write("\n")

    f.write("ENVELOPE FIT PARAMETERS\n")
    f.write("-" * 80 + "\n")
    f.write("Upper Envelope: a(w) = k_upper / w\n")
    f.write(f"  k_upper = {upper_fit['k']:.6f}\n")
    f.write(f"  R² = {upper_fit['r_squared']:.6f}\n")
    f.write(f"  RMSE = {upper_fit['rmse']:.6f}\n")
    f.write("\n")
    f.write("Lower Envelope: a(w) = k_lower / w\n")
    f.write(f"  k_lower = {lower_fit['k']:.6f}\n")
    f.write(f"  R² = {lower_fit['r_squared']:.6f}\n")
    f.write(f"  RMSE = {lower_fit['rmse']:.6f}\n")
    f.write("\n")
    f.write(f"Envelope Ratio (k_upper / k_lower): {envelope_ratio:.6f}\n")
    f.write("\n")

    f.write("COMPARISON TO ORIGINAL HARDCODED VALUES\n")
    f.write("-" * 80 + "\n")
    f.write(f"Original k_upper ~ 0.1875 (hardcoded in test script)\n")
    f.write(f"Fitted k_upper = {upper_fit['k']:.6f}\n")
    f.write(f"Difference: {abs(upper_fit['k'] - 0.1875):.6f}\n")
    f.write("\n")
    f.write(f"Original k_lower ~ 0.0575 (hardcoded in test script)\n")
    f.write(f"Fitted k_lower = {lower_fit['k']:.6f}\n")
    f.write(f"Difference: {abs(lower_fit['k'] - 0.0575):.6f}\n")
    f.write("\n")

    f.write("=" * 80 + "\n")
    f.write("End of Phase 1 Deliverables\n")
    f.write("=" * 80 + "\n")

print(f"Results saved to: {results_path}")
print()

# ============================================================================
# SUMMARY
# ============================================================================

print("=" * 80)
print("PHASE 1 COMPLETE")
print("=" * 80)
print()
print("Deliverables generated:")
print(f"  1. Peak frequency list: {len(peak_idx)} peaks identified")
print(f"  2. Trough frequency list: {len(trough_idx)} troughs identified")
print(f"  3. Envelope parameters fitted: k_upper = {upper_fit['k']:.4f}, k_lower = {lower_fit['k']:.4f}")
print(f"  4. Figure AI-1 reproduction saved to: {figure_path}")
print(f"  5. Full results saved to: {results_path}")
print()
print("Next steps:")
print("  - Review fitted envelope parameters and compare to Hurst's Figure AI-1")
print("  - Proceed to Phase 2: Overlapping Comb Filter Analysis")
print()

plt.show()
